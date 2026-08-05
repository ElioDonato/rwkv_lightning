"""Regression tests for the _session_locks memory-leak fix in
API_servers/router/responses_routes.py::_response_lock_context.

Background: the first version of /v1/responses locked on `previous_response_id
or response_id` unconditionally -- so a first-turn request (no
previous_response_id) locked on its own freshly-minted response_id UUID.
That UUID is unknown to any other request until this one returns it in its
HTTP response, so no concurrent request could ever contend for it; the lock
provided zero correctness benefit while permanently adding one entry to
API_servers.router.common._session_locks (a plain dict, never evicted --
documented there as "bounded by the number of distinct session_ids seen",
an assumption that held for /state/'s client-chosen, typically-reused
session_ids but not for a fresh UUID minted on every single first-turn
/v1/responses call). Every first-turn /v1/responses request, forever, would
leak one dict entry + one asyncio.Lock object for the life of the process.

Fix: only acquire session_lock() when there's a real previous_response_id
to serialize concurrent branches/retries against; use contextlib.nullcontext()
otherwise.

CPU-only, no model needed:
    uv run pytest test/test_responses_lock_leak.py -v
"""
import sys
from contextlib import nullcontext
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import API_servers.router.common as common
from API_servers.router.responses_routes import _response_lock_context


@pytest.fixture
def anyio_backend():
    return "asyncio"


def test_first_turn_uses_nullcontext_not_a_real_lock():
    """No previous_response_id -- there's nothing to serialize against, so
    this must be a no-op context manager, not session_lock()."""
    ctx = _response_lock_context(None)
    assert isinstance(ctx, type(nullcontext()))


def test_empty_string_previous_response_id_also_uses_nullcontext():
    """Falsy-but-not-None previous_response_id (e.g. a client sending "")
    must be treated the same as None -- there's still no real prior
    response to serialize against."""
    ctx = _response_lock_context("")
    assert isinstance(ctx, type(nullcontext()))


def test_continuation_turn_uses_a_real_session_lock():
    """A genuine previous_response_id must still get real serialization --
    this is the one case where two requests actually could race (a client
    retry, or two branches chained off the same prior turn)."""
    ctx = _response_lock_context("resp_abc123")
    assert not isinstance(ctx, type(nullcontext()))
    # session_lock() is an @asynccontextmanager-wrapped generator function;
    # calling it returns an _AsyncGeneratorContextManager, not a plain object.
    assert hasattr(ctx, "__aenter__") and hasattr(ctx, "__aexit__")


@pytest.mark.anyio
async def test_first_turn_lock_context_does_not_touch_session_locks_dict():
    """The actual leak this fix closes: entering/exiting the first-turn
    lock context must never add an entry to common._session_locks."""
    before = set(common._session_locks.keys())

    ctx = _response_lock_context(None)
    async with ctx:
        pass

    after = set(common._session_locks.keys())
    assert after == before, (
        f"first-turn /v1/responses request leaked into _session_locks: "
        f"new keys = {after - before}"
    )


@pytest.mark.anyio
async def test_many_first_turn_requests_leave_session_locks_dict_unchanged():
    """Stronger version of the above: simulate a burst of first-turn
    requests (the realistic failure mode -- a busy server handling many
    independent /v1/responses calls, each with a distinct fresh UUID) and
    confirm none of them ever touch _session_locks."""
    before_size = len(common._session_locks)

    for _ in range(50):
        async with _response_lock_context(None):
            pass

    assert len(common._session_locks) == before_size


@pytest.mark.anyio
async def test_continuation_requests_do_use_session_locks_as_designed():
    """Sanity check the fix didn't overcorrect: a real previous_response_id
    must still register (and later release) a lock in _session_locks --
    that's the actual concurrency guard for chained/retried requests."""
    key = "resp_test_continuation_lock_registration"
    assert key not in common._session_locks

    async with _response_lock_context(key):
        assert key in common._session_locks
        assert common._session_locks[key].locked()

    # Lock itself persists in the dict (that's the documented, accepted
    # tradeoff for real session/response ids -- see common.py's comment),
    # but must be released, not left held.
    assert not common._session_locks[key].locked()


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
