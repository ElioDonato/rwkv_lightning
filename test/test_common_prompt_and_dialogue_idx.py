"""CPU-only unit tests for three pure/near-pure helpers in
API_servers/router/common.py that previously had zero direct test coverage
(only exercised indirectly via GPU integration tests or manual curl):

- normalize_state_prompts(): decides whether a /state/ or /multi_state/
  continuation turn gets a "\n\n" prefix prepended, based on whether an
  existing session state is being reused. Get this wrong and multi-turn
  conversations either run the new turn on from the model's last token
  (missing prefix) or double a blank line into the transcript (prefix
  applied to a turn that already starts with one).
- collect_session_indices() / allocate_next_dialogue_idx(): the counter
  logic behind /multi_state/'s dialogue_idx branching. Uses a fake
  state_manager (list_all_states() stub) and a fake app_state
  (SimpleNamespace with dialogue_idx_lock/dialogue_idx_counters, mirroring
  API_servers/fastapi_service.py's real initialization) so no real
  StateCacheManager or GPU is needed.

Run with: uv run pytest test/test_common_prompt_and_dialogue_idx.py -v
"""
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from API_servers.router.common import (
    allocate_next_dialogue_idx,
    collect_session_indices,
    normalize_state_prompts,
)


# -- normalize_state_prompts --------------------------------------------------

def test_normalize_no_reuse_returns_prompts_unchanged():
    """First turn of a brand new session (no prior state) -- prompts must
    pass through untouched, since there's no prior model output to
    separate this turn from."""
    prompts = ["User: hello\n\nAssistant:"]
    assert normalize_state_prompts(prompts, reuse_existing_state=False) == prompts


def test_normalize_reuse_prefixes_blank_line():
    """Continuation turn (existing state reused) that doesn't already start
    with a blank line gets one prepended, so the new turn reads as a fresh
    paragraph rather than running on from the model's last generated
    token."""
    prompts = ["User: what's my name?\n\nAssistant:"]
    result = normalize_state_prompts(prompts, reuse_existing_state=True)
    assert result == ["\n\nUser: what's my name?\n\nAssistant:"]


def test_normalize_reuse_does_not_double_prefix():
    """A prompt that already starts with '\\n\\n' (e.g. a client that
    pre-formats it, or a retried request) must not get a second blank
    line stacked on top."""
    prompts = ["\n\nUser: already prefixed\n\nAssistant:"]
    result = normalize_state_prompts(prompts, reuse_existing_state=True)
    assert result == ["\n\nUser: already prefixed\n\nAssistant:"]
    assert not result[0].startswith("\n\n\n")


def test_normalize_reuse_handles_empty_string():
    """An empty prompt string is falsy, so the `prompt and not ...`
    condition must short-circuit to leaving it as "" rather than prefixing
    an empty string (which would just be "\\n\\n" and probably not what
    any caller wants for a genuinely empty turn)."""
    result = normalize_state_prompts([""], reuse_existing_state=True)
    assert result == [""]


def test_normalize_reuse_multiple_prompts_independent():
    """Each prompt in the list is normalized independently -- a batch
    where one entry already has the prefix and another doesn't must not
    cross-contaminate."""
    prompts = ["User: a\n\nAssistant:", "\n\nUser: b\n\nAssistant:"]
    result = normalize_state_prompts(prompts, reuse_existing_state=True)
    assert result == ["\n\nUser: a\n\nAssistant:", "\n\nUser: b\n\nAssistant:"]


# -- collect_session_indices / allocate_next_dialogue_idx --------------------

class _FakeStateManager:
    """Stub matching the subset of StateCacheManager.list_all_states()'s
    return shape that collect_session_indices() actually reads."""

    def __init__(self, l1_cache=(), l2_cache=(), database=()):
        self._l1 = list(l1_cache)
        self._l2 = list(l2_cache)
        self._db = list(database)

    def list_all_states(self):
        return {"l1_cache": self._l1, "l2_cache": self._l2, "database": self._db}


def _fake_app_state():
    """Mirrors API_servers/fastapi_service.py's real app.state init:
    app.state.dialogue_idx_lock = Lock(); app.state.dialogue_idx_counters = {}"""
    return SimpleNamespace(dialogue_idx_lock=threading.Lock(), dialogue_idx_counters={})


def test_collect_session_indices_filters_by_prefix_and_parses_ints():
    manager = _FakeStateManager(
        l1_cache=["sess_a:0", "sess_a:2", "sess_b:0"],
        l2_cache=["sess_a:1"],
        database=["sess_a:5", "other_prefix"],
    )
    indices = collect_session_indices(manager, "sess_a")
    assert sorted(indices) == [0, 1, 2, 5]


def test_collect_session_indices_ignores_non_digit_suffix():
    """A key like 'sess_a:abc' (malformed/foreign data) must be skipped,
    not raise or silently coerce -- `tail.isdigit()` guards this."""
    manager = _FakeStateManager(l1_cache=["sess_a:abc", "sess_a:3"])
    assert collect_session_indices(manager, "sess_a") == [3]


def test_collect_session_indices_empty_when_no_match():
    manager = _FakeStateManager(l1_cache=["other_session:0"])
    assert collect_session_indices(manager, "sess_a") == []


def test_allocate_first_index_with_no_existing_sessions_is_one():
    """dialogue_idx=0 is reserved for the root/first turn (created
    directly, not via this allocator -- see state_routes.py); the first
    allocated continuation index for a brand new session must be 1."""
    manager = _FakeStateManager()
    app_state = _fake_app_state()
    assert allocate_next_dialogue_idx(app_state, manager, "new_session") == 1


def test_allocate_continues_from_max_existing_index_on_cold_cache():
    """If dialogue_idx_counters has no entry yet (e.g. after a server
    restart) but the state manager has prior sessions on disk/L1/L2, the
    allocator must resume from max(existing)+1, not restart at 1 and
    silently overwrite an old dialogue_idx."""
    manager = _FakeStateManager(database=["sess_a:0", "sess_a:3", "sess_a:7"])
    app_state = _fake_app_state()
    assert allocate_next_dialogue_idx(app_state, manager, "sess_a") == 8


def test_allocate_increments_monotonically_via_cached_counter():
    """Once a session's counter is warm in app_state.dialogue_idx_counters,
    subsequent calls must not re-query the state manager (each call
    increments the cached counter) and must return strictly increasing
    values."""
    manager = _FakeStateManager(database=["sess_a:0"])
    app_state = _fake_app_state()

    first = allocate_next_dialogue_idx(app_state, manager, "sess_a")
    second = allocate_next_dialogue_idx(app_state, manager, "sess_a")
    third = allocate_next_dialogue_idx(app_state, manager, "sess_a")

    assert (first, second, third) == (1, 2, 3)


def test_allocate_does_not_requery_state_manager_once_cached(monkeypatch):
    """After the first call warms the cache for a session, list_all_states()
    must not be called again for that session -- verifies the counter
    caching is actually load-bearing, not just returning coincidentally
    correct values while still hitting the (slow, DB-backed in production)
    state manager every time."""
    manager = _FakeStateManager(database=["sess_a:0"])
    app_state = _fake_app_state()

    call_count = {"n": 0}
    original = manager.list_all_states

    def _tracking(*args, **kwargs):
        call_count["n"] += 1
        return original(*args, **kwargs)

    allocate_next_dialogue_idx(app_state, manager, "sess_a")  # warms cache
    monkeypatch.setattr(manager, "list_all_states", _tracking)

    allocate_next_dialogue_idx(app_state, manager, "sess_a")
    allocate_next_dialogue_idx(app_state, manager, "sess_a")

    assert call_count["n"] == 0


def test_allocate_independent_sessions_do_not_share_counters():
    manager = _FakeStateManager()
    app_state = _fake_app_state()

    a1 = allocate_next_dialogue_idx(app_state, manager, "sess_a")
    b1 = allocate_next_dialogue_idx(app_state, manager, "sess_b")
    a2 = allocate_next_dialogue_idx(app_state, manager, "sess_a")

    assert (a1, b1, a2) == (1, 1, 2)


def test_allocate_thread_safety_no_duplicate_indices():
    """dialogue_idx_lock (a real threading.Lock, since allocate_next_dialogue_idx
    itself is synchronous code called from inside async route handlers) must
    actually serialize concurrent allocations -- N threads racing to allocate
    for the same session must get N distinct, gap-free indices, not
    duplicates from a read-modify-write race."""
    manager = _FakeStateManager()
    app_state = _fake_app_state()
    results = []
    results_lock = threading.Lock()

    def _allocate():
        idx = allocate_next_dialogue_idx(app_state, manager, "race_session")
        with results_lock:
            results.append(idx)

    threads = [threading.Thread(target=_allocate) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert sorted(results) == list(range(1, 21)), (
        f"expected 20 distinct sequential indices 1..20, got {sorted(results)} "
        "-- duplicates indicate the lock isn't actually serializing allocation"
    )


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
