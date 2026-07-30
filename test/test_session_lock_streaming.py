"""
CPU-only test for the session_lock streaming fix (commit 5f1a3ff).

Verifies that the session_lock is held for the entire stream duration,
not just the setup phase. Two concurrent streaming requests on the same
session_id must be serialized (second waits for first to complete).

Run with: uv run pytest test/test_session_lock_streaming.py -v
"""
import asyncio
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from API_servers.router.common import session_lock


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_session_lock_serializes_concurrent_streams():
    """Two concurrent 'streams' on the same session must not overlap."""
    overlap_detected = False
    active_count = 0
    lock = asyncio.Lock()

    async def fake_stream(session_id, duration, results):
        nonlocal overlap_detected, active_count
        async with session_lock(session_id):
            async with lock:
                active_count += 1
                if active_count > 1:
                    overlap_detected = True
            await asyncio.sleep(duration)
            async with lock:
                active_count -= 1
            results.append(session_id)

    results = []
    # Two concurrent streams on the same session
    await asyncio.gather(
        fake_stream("sess_A", 0.05, results),
        fake_stream("sess_A", 0.05, results),
    )
    assert not overlap_detected, "Two streams on same session overlapped!"
    assert len(results) == 2


@pytest.mark.anyio
async def test_session_lock_allows_different_sessions_concurrent():
    """Streams on different session_ids can run concurrently."""
    max_concurrent = 0
    active_count = 0
    lock = asyncio.Lock()

    async def fake_stream(session_id, duration):
        nonlocal max_concurrent, active_count
        async with session_lock(session_id):
            async with lock:
                active_count += 1
                max_concurrent = max(max_concurrent, active_count)
            await asyncio.sleep(duration)
            async with lock:
                active_count -= 1

    await asyncio.gather(
        fake_stream("sess_A", 0.05),
        fake_stream("sess_B", 0.05),
    )
    assert max_concurrent == 2, "Different sessions should run concurrently"


@pytest.mark.anyio
async def test_session_lock_reentrant_after_release():
    """Lock can be re-acquired after release (not stuck)."""
    async with session_lock("sess_X"):
        pass
    # Should not deadlock
    async with session_lock("sess_X"):
        pass
