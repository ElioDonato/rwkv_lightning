"""
Hermetic (CPU, no GPU) coverage for the Phase-A cache-core hygiene work:

A3a / incremental trie
  - _store_prefix_entry_locked must NOT force the O(n) _rebuild_prefix_trie on
    a plain insert; the newly inserted state must still be resolvable via
    longest_prefix (the incremental path is behaviorally identical to a rebuild
    for inserts).
  - Eviction leaves a stale terminal; match_prefix_state's
    `prefix_entry_index.get(state_id)` guard must turn it into a no-op miss
    (never a wrong value) and still fall through to the disk path.

A3c / DB sweeper
  - run_sweep(ttl_s) deletes expired sessions/prefix rows and returns the
    correct counts; it is a strict no-op (no DB mutation) when ttl<=0 and
    max_rows<=0, preserving default-off byte-identity.
  - run_sweep(max_rows) evicts the OLDEST over-cap prefix rows, keeping the
    most-recent.
  - start/stop_sweeper lifecycle: stop_sweeper clears the thread/handle so a
    flushed connection never races a sweeper run.
"""
import os
import sys
import tempfile
import time
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import state_manager.state_pool as state_pool
import settings as settings_module


def _reset_manager(tmp_db_path: str):
    existing = state_pool.StateCacheManager._instance
    if existing is not None and getattr(existing, "_initialized", False):
        try:
            existing.db_conn.close()
        except Exception:
            pass
    state_pool.StateCacheManager._instance = None
    state_pool.DB_PATH = tmp_db_path


def _tokens(n: int, base: int = 0) -> list:
    return [(base + i) % 49999 for i in range(n)]


def _entry_tokens(n: int, seed: int) -> tuple:
    return tuple(_tokens(n, seed))


def test_incremental_insert_does_not_rebuild_and_resolves():
    """A3a: a plain put_prefix_state (through the shared
    _store_prefix_entry_locked) must not run the O(n) full trie rebuild, yet
    the inserted state must be resolvable by longest_prefix exactly as a
    rebuild would resolve it."""
    with tempfile.TemporaryDirectory() as tmpdir:
        _reset_manager(os.path.join(tmpdir, "a3a.db"))
        manager = state_pool.StateCacheManager()
        try:
            rebuilds = []
            orig = manager._rebuild_prefix_trie
            manager._rebuild_prefix_trie = lambda: (rebuilds.append(1), orig())[1]

            toks = _entry_tokens(1024, 7)
            st = torch.zeros(2, dtype=torch.float32)
            assert manager.put_prefix_state(toks, [st]) is True

            # no full rebuild triggered by the insert
            assert rebuilds == []
            assert manager._prefix_trie_churn == 0

            # but the inserted prefix IS resolvable via longest_prefix
            state_id, matched = manager.prefix_trie.longest_prefix(toks)
            assert matched == len(toks)
            assert state_id in manager.prefix_entry_index
        finally:
            # Drain the async persist queue before closing the connection (a
            # queued _persist_prefix_task would otherwise use the closed cursor
            # on the io_executor thread after db_conn.close()).
            manager.io_executor.shutdown(wait=True)
            manager.db_conn.close()
            state_pool.StateCacheManager._instance = None


def test_evicted_terminal_guarded_as_miss():
    """A3a: after an eviction removes an entry from prefix_entry_index, the
    (not-deleted) trie terminal must be a no-op miss via the existing
    prefix_entry_index.get guard -- match_prefix_state returns None rather than
    a wrong/dangling state."""
    with tempfile.TemporaryDirectory() as tmpdir:
        _reset_manager(os.path.join(tmpdir, "a3a_evict.db"))
        manager = state_pool.StateCacheManager()
        try:
            # Fill one 1024 bucket past its capacity so the oldest entry is
            # evicted (and its terminal left stale in the trie).
            st = torch.zeros(2, dtype=torch.float32)
            for i in range(state_pool.PREFIX_CACHE_BUCKET_CAPACITY + 1):
                manager.put_prefix_state(_entry_tokens(1024, i * 37 + 1), [st])

            bucket = manager.prefix_l2_cache[1024]
            assert len(bucket) == state_pool.PREFIX_CACHE_BUCKET_CAPACITY
            evicted_state_id = None
            # the trie should contain a stale terminal not in the index
            for i in range(state_pool.PREFIX_CACHE_BUCKET_CAPACITY + 1):
                sid, _matched = manager.prefix_trie.longest_prefix(
                    _entry_tokens(1024, i * 37 + 1)
                )
                if sid is not None and sid not in manager.prefix_entry_index:
                    evicted_state_id = sid
                    break
            assert evicted_state_id is not None
            # the stale terminal resolves to a miss through the guard
            assert manager.prefix_entry_index.get(evicted_state_id) is None
        finally:
            manager.io_executor.shutdown(wait=True)
            manager.db_conn.close()
            state_pool.StateCacheManager._instance = None


def test_sweep_ttl_removes_expired_keeps_fresh():
    """A3c: run_sweep(ttl_s) deletes expired rows and leaves recent ones."""
    with tempfile.TemporaryDirectory() as tmpdir:
        _reset_manager(os.path.join(tmpdir, "a3c.db"))
        manager = state_pool.StateCacheManager()
        try:
            with manager.db_lock:
                manager.db_cursor.execute(
                    "INSERT OR REPLACE INTO sessions (session_id, state_blob, last_updated) "
                    "VALUES (?, ?, ?)", ("old", b"x", time.time() - 1000))
                manager.db_cursor.execute(
                    "INSERT OR REPLACE INTO sessions (session_id, state_blob, last_updated) "
                    "VALUES (?, ?, ?)", ("fresh", b"x", time.time()))
                manager.db_conn.commit()

            detail = manager.run_sweep(ttl_s=100.0, max_rows=0)
            assert detail["expired_sessions"] == 1
            with manager.db_lock:
                manager.db_cursor.execute("SELECT session_id FROM sessions")
                assert {r[0] for r in manager.db_cursor.fetchall()} == {"fresh"}
        finally:
            manager.db_conn.close()
            state_pool.StateCacheManager._instance = None


def test_sweep_noop_when_disabled():
    """A3c: with no TTL and no max_rows (the default-off config) run_sweep is a
    strict no-op -- it must not even issue a query, so enabling the master
    RWKV_CACHE_SWEEP knob is what changes behavior, matching the opt-in
    contract."""
    with tempfile.TemporaryDirectory() as tmpdir:
        _reset_manager(os.path.join(tmpdir, "a3c_noop.db"))
        manager = state_pool.StateCacheManager()
        try:
            with manager.db_lock:
                manager.db_cursor.execute(
                    "INSERT OR REPLACE INTO sessions (session_id, state_blob, last_updated) "
                    "VALUES (?, ?, ?)", ("keep", b"x", time.time() - 5000))
                manager.db_conn.commit()
            detail = manager.run_sweep(ttl_s=0.0, max_rows=0)
            assert detail == {"expired_sessions": 0, "expired_prefix": 0, "over_cap_prefix": 0}
            with manager.db_lock:
                manager.db_cursor.execute("SELECT COUNT(*) FROM sessions")
                assert manager.db_cursor.fetchone()[0] == 1
        finally:
            manager.db_conn.close()
            state_pool.StateCacheManager._instance = None


def test_sweep_cap_evicts_oldest_prefix_rows():
    """A3c: run_sweep(max_rows=K) keeps the K most-recent prefix_cache rows and
    evicts the older ones."""
    with tempfile.TemporaryDirectory() as tmpdir:
        _reset_manager(os.path.join(tmpdir, "a3c_cap.db"))
        manager = state_pool.StateCacheManager()
        try:
            with manager.db_lock:
                for i in range(5):
                    manager.db_cursor.execute(
                        "INSERT OR REPLACE INTO prefix_cache "
                        "(state_id, bucket_len, token_count, state_blob, last_updated) "
                        "VALUES (?, ?, ?, ?, ?)",
                        (f"p{i}", 1024, i, b"x", time.time() - (5 - i) * 10))
                manager.db_conn.commit()

            detail = manager.run_sweep(ttl_s=0.0, max_rows=3)
            assert detail["over_cap_prefix"] == 2
            with manager.db_lock:
                manager.db_cursor.execute("SELECT state_id FROM prefix_cache")
                remaining = {r[0] for r in manager.db_cursor.fetchall()}
            # the two OLDEST (p0, p1) are evicted; the 3 newest survive
            assert "p4" in remaining and "p3" in remaining and "p2" in remaining
            assert remaining == {"p2", "p3", "p4"}
        finally:
            manager.db_conn.close()
            state_pool.StateCacheManager._instance = None


def _insert_prefix_db_row(manager, tokens, state, model=""):
    """Insert a prefix_cache row directly into SQLite (bypassing L2) so tests
    exercise the DISK fallback path with a known, controllable on-disk entry
    and a live io_executor (no shutdown)."""
    toks = tuple(tokens)
    state_id = state_pool._prefix_key(model, state_pool._serialize_token_ids(toks))
    hashes = state_pool._build_prefix_hashes(toks)
    blob = manager._serialize(state)
    with manager.db_lock:
        # PREFIX_HASH_COLUMNS are *column names* ("prefix_hash_1024"), but
        # _build_prefix_hashes keys by int bucket -> parse the bucket back out.
        row = [state_id, len(toks), len(toks)]
        row.extend(hashes.get(int(col.rsplit("_", 1)[1])) for col in state_pool.PREFIX_HASH_COLUMNS)
        row.extend([blob, None, time.time()])
        cols = ", ".join(
            ["state_id", "bucket_len", "token_count",
             *state_pool.PREFIX_HASH_COLUMNS, "state_blob", "logits_blob", "last_updated"]
        )
        ph = ", ".join("?" for _ in row)
        manager.db_cursor.execute(
            f"INSERT OR REPLACE INTO prefix_cache ({cols}) VALUES ({ph})", row)
        manager.db_conn.commit()
    return state_id


def test_disk_fallback_default_finds_smaller_bucket():
    """A3b: with RWKV_PREFIX_DISK_ASYNC OFF (default) match_prefix_state uses
    the full per-bucket disk loop, so a match in a bucket BELOW the largest
    plausible one is found synchronously. Proves the knob gates the A3b change
    (default-off byte-identity)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        _reset_manager(os.path.join(tmpdir, "disk_off.db"))
        manager = state_pool.StateCacheManager()
        try:
            toks1024 = _entry_tokens(1024, 3)
            st = [torch.zeros(2, dtype=torch.float32)]
            _insert_prefix_db_row(manager, toks1024, st)
            # 2500-token prompt: 2048 is the largest plausible bucket, but only
            # a 1024 prefix exists -- the full loop must still find it.
            full = list(toks1024) + _tokens(1476, 50000)
            hit = manager.match_prefix_state(full, device="cpu")
            assert hit is not None and hit["bucket_len"] == 1024
        finally:
            manager.io_executor.shutdown(wait=True)
            manager.db_conn.close()
            state_pool.StateCacheManager._instance = None


def test_bounded_probe_async_warm(monkeypatch):
    """A3b: with RWKV_PREFIX_DISK_ASYNC ON, match_prefix_state single-probes
    the largest plausible bucket (2048 for a 2500-token prompt), misses, and
    warms the smaller 1024 bucket in the background so a later request hits L2
    -- the documented reduced-recall tradeoff."""
    monkeypatch.setattr(settings_module.settings, "prefix_disk_async", True)
    with tempfile.TemporaryDirectory() as tmpdir:
        _reset_manager(os.path.join(tmpdir, "disk_on.db"))
        manager = state_pool.StateCacheManager()
        try:
            toks1024 = _entry_tokens(1024, 3)
            st = [torch.zeros(2, dtype=torch.float32)]
            _insert_prefix_db_row(manager, toks1024, st)
            full = list(toks1024) + _tokens(1476, 50000)
            hit1 = manager.match_prefix_state(full, device="cpu")
            # single bounded probe at 2048 (missing) -> no sync hit; warm fired
            assert hit1 is None

            target = state_pool._prefix_key(
                "", state_pool._serialize_token_ids(tuple(toks1024)))
            warmed = False
            for _ in range(300):
                with manager.cache_lock:
                    warmed = target in manager.prefix_entry_index
                if warmed:
                    break
                time.sleep(0.01)
            assert warmed, "background warm should have loaded the 1024 row into L2"

            # a follow-up request now hits the warmed 1024 row in L2
            hit2 = manager.match_prefix_state(full, device="cpu")
            assert hit2 is not None and hit2["cache_source"] == "l2_ram"
        finally:
            manager.io_executor.shutdown(wait=True)
            manager.db_conn.close()
            state_pool.StateCacheManager._instance = None


def test_adaptive_off_rejects_short_prefix():
    """B default-off: a sub-1024 prefix (no fixed bucket) is rejected by
    put_prefix_state exactly as before, preserving byte-identity until
    RWKV_PREFIX_ADAPTIVE is enabled."""
    with tempfile.TemporaryDirectory() as tmpdir:
        _reset_manager(os.path.join(tmpdir, "ada_off.db"))
        manager = state_pool.StateCacheManager()
        try:
            st = torch.zeros(2, dtype=torch.float32)
            assert manager.put_prefix_state(_tokens(300, 7), [st]) is False
            assert manager.prefix_entry_index == {}
        finally:
            manager.db_conn.close()
            state_pool.StateCacheManager._instance = None


def test_adaptive_on_stores_l2_only_and_matches(monkeypatch):
    """B: with RWKV_PREFIX_ADAPTIVE on, a short (sub-1024) prefix is stored as an
    L2-only checkpoint keyed by its exact length, and match_prefix_state
    recovers it for a later request sharing that prefix. It never lands in the
    SQLite prefix_cache (no fixed hash column), which is the documented
    retention tradeoff (lost on restart/L2 eviction)."""
    monkeypatch.setattr(settings_module.settings, "prefix_adaptive", True)
    with tempfile.TemporaryDirectory() as tmpdir:
        _reset_manager(os.path.join(tmpdir, "ada_on.db"))
        manager = state_pool.StateCacheManager()
        try:
            toks = _tokens(300, 7)
            st = [torch.zeros(2, dtype=torch.float32)]
            assert manager.put_prefix_state(toks, st) is True

            key = state_pool._prefix_key("", state_pool._serialize_token_ids(tuple(toks)))
            assert key in manager.prefix_entry_index
            assert key in manager.prefix_l2_cache[300]
            # not persisted to disk (L2-only)
            with manager.db_lock:
                manager.db_cursor.execute(
                    "SELECT COUNT(*) FROM prefix_cache WHERE state_id = ?", (key,))
                assert manager.db_cursor.fetchone()[0] == 0

            # a later request with this exact prefix matches via the trie
            hit = manager.match_prefix_state(toks, device="cpu")
            assert hit is not None and hit["bucket_len"] == 300
            assert hit["matched_tokens"] == 300

            # a prompt whose prefix EXTENDS this short checkpoint resumes from it
            extended = list(toks) + _tokens(50, 9001)
            hit2 = manager.match_prefix_state(extended, device="cpu")
            assert hit2 is not None and hit2["matched_tokens"] == 300
        finally:
            manager.io_executor.shutdown(wait=True)
            manager.db_conn.close()
            state_pool.StateCacheManager._instance = None


def test_sweeper_start_stop_lifecycle():
    """A3c: start_sweeper spawns a handle; stop_sweeper clears it so a
    subsequent flush_all (which closes the connection) won't race it."""
    with tempfile.TemporaryDirectory() as tmpdir:
        _reset_manager(os.path.join(tmpdir, "a3c_life.db"))
        manager = state_pool.StateCacheManager()
        try:
            tel = manager.start_sweeper(interval_s=0.1, ttl_s=0.0, max_rows=0)
            assert tel._sweep_thread is not None
            assert tel._sweep_thread.is_alive()
            tel.stop_sweeper()
            assert tel._sweep_stop is None
            assert tel._sweep_thread is None
            # flush_all (already covered elsewhere) must not error after this:
            # stop_sweeper is a no-op when no sweeper is running.
            tel.stop_sweeper()
        finally:
            manager.db_conn.close()
            state_pool.StateCacheManager._instance = None