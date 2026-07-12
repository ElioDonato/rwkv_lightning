import os
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import state_manager.state_pool as state_pool


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


def test_prefix_cache_longest_match_and_disk_reload():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "prefix_cache_test.db")
        _reset_manager(db_path)
        manager = state_pool.StateCacheManager()

        bucket_small, bucket_large = state_pool.PREFIX_CACHE_BUCKETS[:2]

        state_small = [
            torch.tensor([1.0]),
            torch.tensor([2.0]),
            torch.tensor(bucket_small, dtype=torch.int32),
        ]
        logits_small = torch.tensor([0.1, 0.2, 0.3])
        tokens_small = list(range(bucket_small))

        state_large = [
            torch.tensor([10.0]),
            torch.tensor([20.0]),
            torch.tensor(bucket_large, dtype=torch.int32),
        ]
        logits_large = torch.tensor([0.5, 0.6, 0.7])
        tokens_large = list(range(bucket_large))

        assert manager.put_prefix_state(tokens_small, state_small, logits_small) is True
        assert manager.put_prefix_state(tokens_large, state_large, logits_large) is True

        prompt_tokens = list(range(bucket_large + 32))
        match = manager.match_prefix_state(prompt_tokens, device="cpu")
        assert match is not None
        assert match["matched_tokens"] == bucket_large
        assert match["cache_source"] == "l2_ram"
        assert torch.equal(match["state"][2], torch.tensor(bucket_large, dtype=torch.int32))

        entry = manager.prefix_entry_index[state_pool._serialize_token_ids(tokens_large)]
        manager._persist_prefix_task(entry)

        with manager.cache_lock:
            manager.prefix_l2_cache = {
                bucket: state_pool.OrderedDict()
                for bucket in state_pool.PREFIX_CACHE_BUCKETS
            }
            manager.prefix_entry_index.clear()
            manager._rebuild_prefix_trie()

        disk_match = manager.match_prefix_state(prompt_tokens, device="cpu")
        assert disk_match is not None
        assert disk_match["matched_tokens"] == bucket_large
        assert disk_match["cache_source"] == "disk"
        assert torch.equal(disk_match["logits"], logits_large)

        manager.db_conn.close()
        state_pool.StateCacheManager._instance = None
