"""
Coverage for state_manager/state_pool.py's in-memory L1 (VRAM)/L2 (RAM)
session-cache paths, complementing test_prefix_state_cache.py (which only
covers the *prefix* cache's L2/disk paths, not the plain per-session
put_state/get_state L1<->L2 machinery exercised here).

Reads state_pool.py's actual eviction policy directly rather than assuming
one:
  - put_state(session_id, state) always (re)inserts at the END of
    `l1_cache` (an OrderedDict), after first deleting any existing L1/L2
    entry for that session_id (state_pool.py:312-337).
  - If len(l1_cache) > L1_CAPACITY after insert, the OLDEST entry (i.e.
    `popitem(last=False)`) is evicted to `l2_cache` -- this is a real
    LRU policy, not pure FIFO, because get_state() calls
    `l1_cache.move_to_end(session_id)` on an L1 hit (state_pool.py:352),
    so a recently-*read* entry is not the oldest anymore even if it was
    inserted first.
  - If len(l2_cache) > L2_CAPACITY after that, the oldest L2 entry is
    popped and handed to the (real, non-test-doubled) io_executor to be
    persisted to sqlite -- covered by test_prefix_state_cache.py's disk
    path already, not re-tested here.
  - get_state() on an L2 hit *pops* the entry out of l2_cache, moves every
    tensor to 'cuda' (hardcoded, not parameterized by caller device --
    see state_pool.py:369), and re-inserts via put_state(), i.e. promotion
    back to L1 goes through the exact same insert-then-maybe-evict path
    as a fresh put, which can itself immediately evict a *different*
    entry from L1 into L2 if L1 was already full. This test asserts that
    cascading behavior explicitly rather than assuming a simpler model.

No GPU/model required for the L1<->L2 store/evict tests (they use plain
CPU tensors -- put_state doesn't force a device, and L1->L2 eviction's
`.to('cpu')` is a no-op on CPU tensors). The one L2->L1 *promotion* test
needs real CUDA because get_state()'s L2 hit path hardcodes
`.to('cuda')` regardless of the state's original device; it's skipped if
CUDA isn't available on the runner.
"""
import os
import sys
import tempfile
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import state_manager.state_pool as state_pool


# state_pool.py's methods read L1_CAPACITY/L2_CAPACITY as bare module-level
# globals, not self.*/a config object, so tests that need non-default
# capacities monkeypatch the module attribute directly (see _reset_manager
# below). Snapshot the real defaults once so every test can restore them
# afterward -- without this, a test that patches L1_CAPACITY=1 would leak
# that value into every test that runs later in the same pytest session
# (regardless of file), silently changing their eviction behavior. Confirmed
# this was a real, live bug: test_l2_overflow_persists_oldest_to_disk sets
# L1_CAPACITY=1 and never restored it, which broke an unrelated
# StateCacheManager test in a different file when both ran in one session.
_DEFAULT_L1_CAPACITY = state_pool.L1_CAPACITY
_DEFAULT_L2_CAPACITY = state_pool.L2_CAPACITY


def _reset_manager(tmp_db_path: str, l1_capacity=None, l2_capacity=None):
    """Mirrors test_prefix_state_cache.py's _reset_manager, plus optional
    monkeypatching of the module-level capacity constants. state_pool.py's
    methods reference L1_CAPACITY/L2_CAPACITY as bare global names (not
    self.* or a config object), so patching the module attribute before
    constructing a fresh singleton is sufficient -- no need to patch the
    instance."""
    try:
        existing = state_pool.StateCacheManager._instance
        if existing is not None and getattr(existing, "_initialized", False):
            try:
                existing.db_conn.close()
            except Exception:
                pass
    finally:
        state_pool.StateCacheManager._instance = None
        state_pool.DB_PATH = tmp_db_path
        state_pool.L1_CAPACITY = l1_capacity if l1_capacity is not None else _DEFAULT_L1_CAPACITY
        state_pool.L2_CAPACITY = l2_capacity if l2_capacity is not None else _DEFAULT_L2_CAPACITY


def _fake_state(marker: float, token_count: int = 3):
    """Synthetic state matching the [*, *, token_count_tensor, ...] shape
    the real code inspects (state[2].item() for logging) -- not a real
    RWKV state, just plain tensors distinguishable by `marker`."""
    return [
        torch.tensor([marker, marker + 0.5]),
        torch.tensor([marker * 2]),
        torch.tensor(token_count, dtype=torch.int32),
    ]


def test_l1_store_and_retrieve_roundtrip():
    """Basic L1 (VRAM-slot) put/get: state comes back with equal content,
    but as a distinct clone (mutating the caller's copy must not corrupt
    the cached original -- state_pool._clone_state() is what's supposed
    to guarantee this)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        _reset_manager(os.path.join(tmpdir, "l1.db"), l1_capacity=16, l2_capacity=64)
        manager = state_pool.StateCacheManager()
        try:
            state = _fake_state(1.0)
            manager.put_state("sess-a", state)

            assert "sess-a" in manager.l1_cache
            assert manager.has_state("sess-a") is True

            retrieved = manager.get_state("sess-a")
            assert retrieved is not None
            assert torch.equal(retrieved[0], state[0])
            assert torch.equal(retrieved[2], state[2])

            # must be a clone, not the same tensor object / storage
            assert retrieved[0] is not state[0]
            retrieved[0][0] = 999.0
            assert manager.l1_cache["sess-a"][0][0].item() != 999.0
        finally:
            manager.db_conn.close()
            state_pool.StateCacheManager._instance = None


def test_l1_overflow_evicts_lru_not_fifo_to_l2():
    """L1_CAPACITY=2. Insert A, B (L1 full). Read A back (L1 hit) so A
    becomes most-recently-used. Insert C -> must evict B (the actual LRU
    victim), not A (which would be the FIFO-by-insertion-order victim) --
    this is the detail that distinguishes state_pool's real
    move_to_end()-based LRU from naive FIFO."""
    with tempfile.TemporaryDirectory() as tmpdir:
        _reset_manager(os.path.join(tmpdir, "l1_evict.db"), l1_capacity=2, l2_capacity=64)
        manager = state_pool.StateCacheManager()
        try:
            manager.put_state("A", _fake_state(1.0))
            manager.put_state("B", _fake_state(2.0))
            assert list(manager.l1_cache.keys()) == ["A", "B"]

            # touch A -> becomes most-recently-used
            manager.get_state("A")
            assert list(manager.l1_cache.keys()) == ["B", "A"]

            manager.put_state("C", _fake_state(3.0))

            # B (oldest / least-recently-used) evicted to L2; A survives in L1
            assert "B" not in manager.l1_cache
            assert "B" in manager.l2_cache
            assert set(manager.l1_cache.keys()) == {"A", "C"}
            assert "A" in manager.l1_cache
        finally:
            manager.db_conn.close()
            state_pool.StateCacheManager._instance = None


def test_l1_full_overwrite_of_existing_session_does_not_double_evict():
    """put_state() on a session_id already present deletes the old L1/L2
    entry first, so overwriting an existing session must NOT trigger an
    extra eviction just because it happens to be re-inserted at the end."""
    with tempfile.TemporaryDirectory() as tmpdir:
        _reset_manager(os.path.join(tmpdir, "l1_overwrite.db"), l1_capacity=2, l2_capacity=64)
        manager = state_pool.StateCacheManager()
        try:
            manager.put_state("A", _fake_state(1.0))
            manager.put_state("B", _fake_state(2.0))
            assert len(manager.l1_cache) == 2

            manager.put_state("A", _fake_state(10.0))  # overwrite, still 2 unique sessions
            assert len(manager.l1_cache) == 2
            assert manager.l2_cache == {}
            assert torch.equal(manager.l1_cache["A"][0], torch.tensor([10.0, 10.5]))
        finally:
            manager.db_conn.close()
            state_pool.StateCacheManager._instance = None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="get_state()'s L2 hit path hardcodes .to('cuda')")
def test_l2_hit_promotes_back_to_l1_and_can_cascade_evict():
    """L1_CAPACITY=1, L2_CAPACITY=64. Put A, then B -> A evicted to L2
    (L1 can only hold 1). get_state(A) must:
      1. return A's content correctly (round-tripped through L2 storage)
      2. remove A from l2_cache (get_state pops L2 entries, doesn't just peek)
      3. re-insert A into l1_cache (promotion)
      4. because L1_CAPACITY=1 and B currently occupies the only L1 slot,
         re-inserting A must cascade-evict B back out to L2 -- i.e.
         promotion goes through the same put_state() capacity check, it
         isn't a special-cased "just add it, ignore capacity" path.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        _reset_manager(os.path.join(tmpdir, "l2_promote.db"), l1_capacity=1, l2_capacity=64)
        manager = state_pool.StateCacheManager()
        try:
            state_a = _fake_state(7.0)
            manager.put_state("A", state_a)
            manager.put_state("B", _fake_state(8.0))

            assert "A" not in manager.l1_cache
            assert "A" in manager.l2_cache
            assert list(manager.l1_cache.keys()) == ["B"]

            promoted = manager.get_state("A")
            assert promoted is not None
            assert torch.equal(promoted[0].cpu(), state_a[0])
            assert torch.equal(promoted[2].cpu(), state_a[2])

            # A promoted into L1, popped out of L2
            assert "A" in manager.l1_cache
            assert "A" not in manager.l2_cache

            # L1_CAPACITY=1 was exceeded by the promotion (B was already
            # there) -> B must have cascaded back down into L2
            assert list(manager.l1_cache.keys()) == ["A"]
            assert "B" in manager.l2_cache
        finally:
            manager.db_conn.close()
            state_pool.StateCacheManager._instance = None


def test_l2_overflow_persists_oldest_to_disk():
    """L2_CAPACITY=1. Force two separate L1->L2 evictions (L1_CAPACITY=1
    forces every second put to evict); the second L2 arrival must push
    the L2 that's already there over capacity, handing it off to
    io_executor for a real (synchronous-waited) disk persist -- exercised
    end-to-end against the actual sqlite table, not mocked."""
    with tempfile.TemporaryDirectory() as tmpdir:
        _reset_manager(os.path.join(tmpdir, "l2_overflow.db"), l1_capacity=1, l2_capacity=1)
        manager = state_pool.StateCacheManager()
        try:
            manager.put_state("A", _fake_state(1.0))  # L1=[A]
            manager.put_state("B", _fake_state(2.0))  # L1=[B], L2=[A]
            assert list(manager.l2_cache.keys()) == ["A"]

            manager.put_state("C", _fake_state(3.0))  # L1=[C], L2 gets B -> overflow(1) -> A persisted to disk
            assert list(manager.l1_cache.keys()) == ["C"]
            assert list(manager.l2_cache.keys()) == ["B"]

            # A's disk persist was fired via io_executor (async, 1 worker,
            # FIFO queue) -- wait for that queue to drain rather than
            # sleeping/polling arbitrarily.
            manager.io_executor.shutdown(wait=True)

            manager.db_cursor.execute("SELECT session_id FROM sessions")
            rows = {r[0] for r in manager.db_cursor.fetchall()}
            assert rows == {"A"}
            assert "A" not in manager.l1_cache
            assert "A" not in manager.l2_cache
        finally:
            manager.db_conn.close()
            state_pool.StateCacheManager._instance = None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
