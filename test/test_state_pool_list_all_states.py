"""Regression test for StateCacheManager.list_all_states()'s include_prefix_db
default-off fix.

Background: list_all_states() previously always ran
list_prefix_states_in_db() -- a full unindexed `ORDER BY last_updated` scan
over the prefix_cache table. On this deployment's real database (bloated to
tens of GB from years of INSERT OR REPLACE without VACUUM) that scan took
20+ seconds and, because it runs synchronously on the asyncio event loop
thread, stalled *every* other in-flight and incoming request for that whole
duration -- confirmed live via py-spy (the event loop's only thread was
stuck inside list_prefix_states_in_db, called from
/multi_state/'s allocate_next_dialogue_idx -> collect_session_indices).

None of list_all_states()'s three callers (collect_session_indices,
/state/delete's delete_prefix cleanup, /state/status) actually read
prefix_l2_counts/prefix_l2_cache/prefix_database_count from its result, so
the fix makes that scan opt-in via include_prefix_db=False by default.

CPU-only, no model/GPU needed:
    uv run pytest test/test_state_pool_list_all_states.py -v
"""
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

import state_manager.state_pool as state_pool

# Captured at module import (pytest imports all test modules during
# collection, before running any test body), so this is the real
# module-level default -- not whatever some other test file's L1_CAPACITY
# monkeypatch last left it as. state_pool.py reads L1_CAPACITY as a bare
# module global, so any test that mutates it without restoring would
# otherwise leak into every test that runs afterward in the same pytest
# session, regardless of which file it's in.
_DEFAULT_L1_CAPACITY = state_pool.L1_CAPACITY
_DEFAULT_L2_CAPACITY = state_pool.L2_CAPACITY


def _reset_manager(tmp_db_path: str):
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
        state_pool.L1_CAPACITY = _DEFAULT_L1_CAPACITY
        state_pool.L2_CAPACITY = _DEFAULT_L2_CAPACITY


def test_list_all_states_defaults_to_skipping_prefix_db_scan(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = f"{tmpdir}/test.db"
        _reset_manager(db_path)
        manager = state_pool.StateCacheManager()

        called = {"count": 0}
        original = manager.list_prefix_states_in_db

        def _tracking_wrapper():
            called["count"] += 1
            return original()

        monkeypatch.setattr(manager, "list_prefix_states_in_db", _tracking_wrapper)

        result = manager.list_all_states()

        assert called["count"] == 0, (
            "list_all_states() must not call list_prefix_states_in_db() by default -- "
            "that query has no index for its ORDER BY and can hang for tens of "
            "seconds on a large/bloated prefix_cache table, blocking the whole "
            "server since this runs on the event loop thread."
        )
        assert result["prefix_database_count"] is None
        assert result["l1_cache"] == []
        assert result["l2_cache"] == []
        assert result["database"] == []


def test_list_all_states_include_prefix_db_true_still_works(monkeypatch):
    """The opt-in path must still function correctly for any caller that
    genuinely needs prefix-DB visibility."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = f"{tmpdir}/test.db"
        _reset_manager(db_path)
        manager = state_pool.StateCacheManager()

        bucket = state_pool.PREFIX_CACHE_BUCKETS[0]
        state = [
            torch.tensor([1.0]),
            torch.tensor([2.0]),
            torch.tensor(bucket, dtype=torch.int32),
        ]
        tokens = list(range(bucket))
        assert manager.put_prefix_state(tokens, state, torch.tensor([0.1])) is True

        # put_prefix_state() persists to disk asynchronously via io_executor;
        # run the persist task synchronously here (same technique as
        # test_prefix_state_cache.py) instead of racing the background thread.
        entry = manager.prefix_entry_index[state_pool._serialize_token_ids(tuple(tokens))]
        manager._persist_prefix_task(entry)

        result = manager.list_all_states(include_prefix_db=True)
        assert result["prefix_database_count"] == 1


def test_list_all_states_still_reports_session_states():
    """The fix must not regress the actual session-listing behavior that
    collect_session_indices/state_status/state_delete depend on."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = f"{tmpdir}/test.db"
        _reset_manager(db_path)
        manager = state_pool.StateCacheManager()

        manager.put_state("sess_a", [torch.tensor([1.0])])
        manager.put_state("sess_b", [torch.tensor([2.0])])

        result = manager.list_all_states()
        assert set(result["l1_cache"]) == {"sess_a", "sess_b"}
        assert result["total_count"] == 2


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
