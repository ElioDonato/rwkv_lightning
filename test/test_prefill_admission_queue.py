"""
Coverage for infer/inference.py's prefill-admission queue, specifically the
incremental-release mechanism (acquire_prefill_permit's `outstanding_bsz`
box, release_prefill_capacity, and release_prefill_permit's `permit`-aware
double-release guard).

Pure asyncio bookkeeping, no model/GPU needed -- uses a minimal fake `model`
object exposing just the attributes InferenceEngine's admission logic reads
(max_prefill_bsz_limit, max_prefill_bsz, refresh_max_prefill_bsz). Uses
anyio's pytest plugin (already a dependency via FastAPI/Starlette) for the
async test functions rather than adding a new pytest-asyncio dependency.

Run with: source env.sh && uv run pytest test/test_prefill_admission_queue.py -v
(The import chain pulls in torch/CUDA; CUDA_HOME must be set even though
no GPU computation actually runs.)
"""
import asyncio
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

pytest.importorskip("torch", reason="torch not installed (source env.sh first)")

try:
    from infer.inference import InferenceEngine
    from infer.cancellation import PrefillBszLimitExceeded
except OSError as exc:
    pytest.skip(f"CUDA environment not configured: {exc}", allow_module_level=True)


@pytest.fixture
def anyio_backend():
    return "asyncio"


class _FakeModel:
    """Exposes just what InferenceEngine's admission logic reads."""

    def __init__(self, max_prefill_bsz):
        self.max_prefill_bsz = max_prefill_bsz
        self.max_prefill_bsz_limit = max_prefill_bsz

    def refresh_max_prefill_bsz(self):
        return self.max_prefill_bsz


def _make_engine(max_prefill_bsz=8):
    return InferenceEngine(
        model=_FakeModel(max_prefill_bsz), tokenizer=None, args=None, rocm_flag=False
    )


@pytest.mark.anyio
async def test_basic_acquire_release_roundtrip():
    """Baseline: a single acquire/release pair returns the shared budget to
    exactly where it started -- the pre-existing behavior this feature must
    not regress."""
    engine = _make_engine(max_prefill_bsz=8)
    permit = await engine.acquire_prefill_permit(request_bsz=8, request_label="test")
    assert engine._prefill_reserved_bsz == 8
    await engine.release_prefill_permit(
        request_bsz=8, request_label="test", ticket=permit["ticket"], permit=permit
    )
    assert engine._prefill_reserved_bsz == 0


@pytest.mark.anyio
async def test_release_prefill_permit_without_permit_arg_is_unchanged():
    """Callers that never pass `permit=` (every route before this feature)
    must get byte-identical behavior: a plain request_bsz release."""
    engine = _make_engine(max_prefill_bsz=8)
    permit = await engine.acquire_prefill_permit(request_bsz=5, request_label="test")
    assert engine._prefill_reserved_bsz == 5
    await engine.release_prefill_permit(request_bsz=5, request_label="test", ticket=permit["ticket"])
    assert engine._prefill_reserved_bsz == 0


@pytest.mark.anyio
async def test_incremental_release_frees_capacity_immediately():
    """The core new behavior: release_prefill_capacity() partially frees a
    permit's reservation before the request itself ends, and a second
    request that was blocked on the full budget can now be admitted."""
    engine = _make_engine(max_prefill_bsz=10)
    big_permit = await engine.acquire_prefill_permit(request_bsz=10, request_label="big")
    assert engine._prefill_reserved_bsz == 10

    # A second request needing capacity must be blocked right now (would
    # hang if awaited directly) -- verify via wait_for with a short timeout
    # rather than actually blocking the test.
    small_task = asyncio.create_task(
        engine.acquire_prefill_permit(request_bsz=3, request_label="small")
    )
    await asyncio.sleep(0.05)
    assert not small_task.done(), "second request should still be queued (no capacity yet)"

    # Simulate 4 rows finishing and getting compacted out of the big batch.
    await engine.release_prefill_capacity(4, big_permit, request_label="big")
    assert engine._prefill_reserved_bsz == 6
    assert big_permit["outstanding_bsz"][0] == 6

    # The queued small request should now be admitted without needing the
    # big request to finish at all.
    small_permit = await asyncio.wait_for(small_task, timeout=1.0)
    assert engine._prefill_reserved_bsz == 9  # 6 (big, remaining) + 3 (small)

    # Finish cleanup: releasing the big permit's *remaining* outstanding
    # amount (6, not the original 10) must not double-release the 4 already
    # freed via release_prefill_capacity.
    await engine.release_prefill_permit(
        request_bsz=10, request_label="big", ticket=big_permit["ticket"], permit=big_permit
    )
    assert engine._prefill_reserved_bsz == 3  # only small's reservation remains
    await engine.release_prefill_permit(
        request_bsz=3, request_label="small", ticket=small_permit["ticket"], permit=small_permit
    )
    assert engine._prefill_reserved_bsz == 0


@pytest.mark.anyio
async def test_release_prefill_permit_after_full_incremental_release_is_noop():
    """If every row is released incrementally before the request ends, the
    final release_prefill_permit() call must be a safe no-op, not an
    over-release that drives the shared counter negative (which would
    silently over-admit future requests beyond real GPU capacity)."""
    engine = _make_engine(max_prefill_bsz=10)
    permit = await engine.acquire_prefill_permit(request_bsz=5, request_label="test")
    assert engine._prefill_reserved_bsz == 5

    await engine.release_prefill_capacity(5, permit, request_label="test")
    assert engine._prefill_reserved_bsz == 0
    assert permit["outstanding_bsz"][0] == 0

    # Final release at request end: must not go negative or double-release.
    await engine.release_prefill_permit(
        request_bsz=5, request_label="test", ticket=permit["ticket"], permit=permit
    )
    assert engine._prefill_reserved_bsz == 0


@pytest.mark.anyio
async def test_release_prefill_capacity_clamps_to_outstanding():
    """A caller accidentally releasing more than a permit ever reserved
    (e.g. a bug double-counting compacted rows) must clamp to what's
    actually outstanding, not drive the shared counter negative."""
    engine = _make_engine(max_prefill_bsz=10)
    permit = await engine.acquire_prefill_permit(request_bsz=3, request_label="test")
    assert engine._prefill_reserved_bsz == 3

    # Ask to release 100 when only 3 is actually reserved for this permit.
    await engine.release_prefill_capacity(100, permit, request_label="test")
    assert engine._prefill_reserved_bsz == 0
    assert permit["outstanding_bsz"][0] == 0

    # A further release call (e.g. the request's own final cleanup) is a
    # harmless no-op, not a crash or a negative counter.
    await engine.release_prefill_capacity(5, permit, request_label="test")
    assert engine._prefill_reserved_bsz == 0
    await engine.release_prefill_permit(
        request_bsz=3, request_label="test", ticket=permit["ticket"], permit=permit
    )
    assert engine._prefill_reserved_bsz == 0


@pytest.mark.anyio
async def test_release_prefill_capacity_zero_amount_is_noop():
    engine = _make_engine(max_prefill_bsz=10)
    permit = await engine.acquire_prefill_permit(request_bsz=5, request_label="test")
    await engine.release_prefill_capacity(0, permit, request_label="test")
    assert engine._prefill_reserved_bsz == 5
    assert permit["outstanding_bsz"][0] == 5


@pytest.mark.anyio
async def test_multiple_incremental_releases_accumulate_correctly():
    """Rows finishing across several separate compaction events (the real
    big_batch_stream pattern: one release call per compaction, not one big
    release at the end) must each correctly reduce both the permit's own
    outstanding count and the shared reserved total."""
    engine = _make_engine(max_prefill_bsz=20)
    permit = await engine.acquire_prefill_permit(request_bsz=10, request_label="test")
    assert engine._prefill_reserved_bsz == 10

    await engine.release_prefill_capacity(2, permit, request_label="test")
    assert permit["outstanding_bsz"][0] == 8
    assert engine._prefill_reserved_bsz == 8

    await engine.release_prefill_capacity(3, permit, request_label="test")
    assert permit["outstanding_bsz"][0] == 5
    assert engine._prefill_reserved_bsz == 5

    await engine.release_prefill_capacity(5, permit, request_label="test")
    assert permit["outstanding_bsz"][0] == 0
    assert engine._prefill_reserved_bsz == 0

    await engine.release_prefill_permit(
        request_bsz=10, request_label="test", ticket=permit["ticket"], permit=permit
    )
    assert engine._prefill_reserved_bsz == 0


@pytest.mark.anyio
async def test_reject_over_limit_request_unaffected_by_incremental_release():
    """PrefillBszLimitExceeded's rejection path (a request bigger than the
    server can ever admit) is untouched by this feature -- sanity check
    that acquire_prefill_permit's existing behavior survives the new
    outstanding_bsz plumbing added to its success-path return value."""
    engine = _make_engine(max_prefill_bsz=4)
    with pytest.raises(PrefillBszLimitExceeded):
        await engine.acquire_prefill_permit(request_bsz=100, request_label="test")
    assert engine._prefill_reserved_bsz == 0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
