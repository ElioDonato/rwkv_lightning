"""
Hermetic coverage for server-side CONTINUOUS BATCHING of the embed endpoints
(infer/embed_aggregator.EmbedAggregator, RWKV_EMBED_AGGREGATE opt-in).

Zero network, zero real GPU, zero real model. The unit under test is the
scheduler's admission/splitting bookkeeping, so ``embed_texts`` -- the only GPU
unit the aggregator ever touches -- is replaced by a deterministic CPU stand-in
that returns a per-text vector purely as a function of the text. Because the
aggregator slices the concatenated result back by each request's own text
count, matching the per-request vectors against per-text embeddings proves the
split accounting (text counts preserved, ordering preserved) without touching
any CUDA kernel. The pattern mirrors test_embedding_batch.py / test_async_forward.py
(fake model + monkeypatched unit of work); anyio's async runner (a dependency
via FastAPI/Starlette) drives the coroutine tests.

Requirements covered: (a) per-request split accounting, (b) default-off routes
straight to inline embed_texts (no queue), (c) max_prefill_bsz cap -- extra
requests wait for a later window and still complete, (d) single-text smoke,
plus a scheduler-shutdown test proving queued futures are resolved (never
orphaned).
"""
import asyncio
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

pytest.importorskip("torch", reason="torch not installed (source env.sh first)")

try:
    import infer.embed_aggregator as agg_mod
except OSError as exc:
    pytest.skip(f"CUDA environment not configured: {exc}", allow_module_level=True)

from infer.embed_aggregator import EmbedAggregator


@pytest.fixture
def anyio_backend():
    return "asyncio"


class _FakeModel:
    """Minimal stand-in exposing the max_prefill_bsz attribute the aggregator's
    default cap resolution reads (0 == no cap)."""

    def __init__(self, max_prefill_bsz=0):
        self.max_prefill_bsz = max_prefill_bsz


# Deterministic per-text "embedding" so split accounting is checkable exactly.
def _vec(text):
    """A vector that is injectively a function of the text (length + charsum),
    so distinct texts are distinguishable and the per-request slice is
    verifiable independently of the aggregator."""
    return [float(len(text)), float(sum(ord(c) for c in text))]


class _Collector:
    """Replaces the module's embed_texts. Records every call's full text list
    (the concatenated batch) and returns the exact per-text vectors a real
    embed_texts would for that list."""

    def __init__(self):
        self.calls = []  # list of text-lists, one entry per embed_texts call

    def embed(self, model, tokenizer, texts, normalize=True):
        self.calls.append(list(texts))
        return [_vec(t) for t in texts]


@pytest.fixture
def fake(monkeypatch):
    collector = _Collector()
    monkeypatch.setattr(agg_mod, "embed_texts", collector.embed)
    return collector


def _expected(reqs):
    """reqs: list of request text-lists -> list of expected per-request vectors."""
    return [[_vec(t) for t in texts] for texts in reqs]


# ---------------------------------------------------------------------------
# (b) Default-off: routes straight to the existing per-request embed, no queue.
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_default_off_runs_inline_no_queue(fake):
    agg = EmbedAggregator(_FakeModel(), None, enabled=False)
    assert agg.enabled is False
    assert agg._task is None

    r1 = await agg.submit(["alpha", "beta"])
    r2 = await agg.submit(["gamma"])

    # Two submits -> two independent inline embed_texts calls (no aggregation),
    # and exactly what a direct embed_texts would have returned.
    assert r1 == _expected([["alpha", "beta"]])[0] == [_vec("alpha"), _vec("beta")]
    assert r2 == _expected([["gamma"]])[0] == [_vec("gamma")]
    assert fake.calls == [["alpha", "beta"], ["gamma"]]
    assert agg.pending_count == 0, "default-off must never enqueue"
    assert agg._task is None, "default-off must never spawn a scheduler task"

    # start() must be a no-op while disabled.
    agg.start()
    assert agg._task is None


# ---------------------------------------------------------------------------
# (a) Aggregation: per-request split accounting + order preservation.
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_aggregation_splits_results_by_text_count(fake):
    agg = EmbedAggregator(_FakeModel(), None, enabled=True, window_ms=50, max_bsz=100)
    agg.start()

    reqs = [["a", "bb"], ["ccc"], ["dd", "e", "f"]]  # 2 + 1 + 3 = 6 texts
    futures = [agg.submit(req) for req in reqs]
    results = await asyncio.gather(*futures)

    assert results == _expected(reqs)
    # One shared embed_texts over the FIFO-concatenated texts of ALL requests.
    assert fake.calls == [["a", "bb", "ccc", "dd", "e", "f"]]
    await agg.stop()


@pytest.mark.anyio
async def test_aggregation_preserves_request_order_even_with_interleave(fake):
    agg = EmbedAggregator(_FakeModel(), None, enabled=True, window_ms=50, max_bsz=100)
    agg.start()

    # Requests with identical structure but distinguishable text -> order of the
    # returned lists must match submission order across many concurrent requests.
    reqs = [[f"t-{i}", f"u-{i}"] for i in range(8)]
    futures = [agg.submit(req) for req in reqs]
    results = await asyncio.gather(*futures)

    assert results == _expected(reqs)
    assert fake.calls == [[t for req in reqs for t in req]]
    await agg.stop()


# ---------------------------------------------------------------------------
# (c) max_prefill_bsz cap: extra requests wait for a later window.
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_cap_admits_extra_requests_in_later_window(fake):
    # Cap = 2 texts. Three 1-text requests: the first window holds 2, the 3rd
    # waits and is embedded in a second window.
    agg = EmbedAggregator(_FakeModel(), None, enabled=True, window_ms=10, max_bsz=2)
    agg.start()

    futures = [agg.submit([f"w{t}"]) for t in range(3)]
    results = await asyncio.gather(*futures)

    assert results == [[_vec("w0")], [_vec("w1")], [_vec("w2")]]
    # Two shared batches, cap-respecting: first window admitted 2 texts, the
    # deferred request got a second window. Never one batch of 3.
    assert fake.calls == [["w0", "w1"], ["w2"]]
    await agg.stop()


@pytest.mark.anyio
async def test_cap_holds_multitext_request_across_windows(fake):
    # Cap = 3 texts. Concurrent requests: a 2-text + a 1-text + a 2-text. The
    # first window admits 2+1 = 3 (cap full); the 2-text request cannot be split,
    # so it waits and is embedded whole in a second window.
    agg = EmbedAggregator(_FakeModel(), None, enabled=True, window_ms=10, max_bsz=3)
    agg.start()

    futures = [agg.submit(["a", "b"]), agg.submit(["c"]), agg.submit(["d", "e"])]
    results = await asyncio.gather(*futures)

    assert results == [
        [_vec("a"), _vec("b")], [_vec("c")], [_vec("d"), _vec("e")]
    ]
    # window1: a,b,c (3 texts, cap full); window2: d,e. The 2-text request never split.
    assert fake.calls == [["a", "b", "c"], ["d", "e"]]
    await agg.stop()


@pytest.mark.anyio
async def test_cap_resolves_from_model_when_no_override(fake):
    # Default cap resolution reads model.max_prefill_bsz (2 here).
    agg = EmbedAggregator(_FakeModel(max_prefill_bsz=2), None,
                          enabled=True, window_ms=10)
    agg.start()

    futures = [agg.submit([f"m{t}"]) for t in range(3)]
    results = await asyncio.gather(*futures)

    assert results == [[_vec(f"m{t}")] for t in range(3)]
    assert fake.calls == [["m0", "m1"], ["m2"]]
    await agg.stop()


# ---------------------------------------------------------------------------
# (d) Single-text correctness smoke through the aggregated path.
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_single_text_smoke_aggregated(fake):
    agg = EmbedAggregator(_FakeModel(), None, enabled=True, window_ms=10, max_bsz=8)
    agg.start()

    result = await agg.submit(["hello"])
    assert result == [_vec("hello")]
    assert fake.calls == [["hello"]]
    await agg.stop()


# ---------------------------------------------------------------------------
# Failure handling: scheduler shutdown resolves queued futures (never orphaned).
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_fail_pending_resolves_queued_and_inflight_futures(fake):
    """The scheduler-shutdown contract: a cancelled scheduler must resolve every
    not-yet-run request -- both still-queued (_pending) jobs and jobs already
    admitted into an in-flight window batch (_inflight_batch) -- with an
    exception, so no awaiting caller's future is ever orphaned. Tested
    deterministically by injecting jobs and running the exact cleanup the
    _run_scheduler CancelledError handler uses."""
    agg = EmbedAggregator(_FakeModel(), None, enabled=True, window_ms=0, max_bsz=100)
    loop = asyncio.get_running_loop()

    queued = [agg_mod._EmbedJob(i, [f"q{i}"], loop) for i in range(3)]
    inflight = [agg_mod._EmbedJob(10 + i, [f"f{i}"], loop) for i in range(2)]
    agg._pending.extend(queued)
    agg._inflight_batch = inflight

    agg._fail_pending(RuntimeError("embed aggregator shut down before embedding"))

    for job in queued + inflight:
        assert isinstance(job.future.exception(), RuntimeError)
    assert agg.pending_count == 0, "no pending jobs may remain after shutdown"
    assert agg._inflight_batch is None, "inflight batch must be released"

    # stop() on a running scheduler must not hang and must be idempotent.
    agg = EmbedAggregator(_FakeModel(), None, enabled=True, window_ms=0, max_bsz=100)
    agg.start()
    await agg.stop()
    assert agg._task is None
    await agg.stop()  # idempotent
    assert agg._task is None


# ---------------------------------------------------------------------------
# Cap-admission regression: the HEAD job is never rejected by the cap.
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_single_job_larger_than_cap_completes(fake):
    """Finding-1 regression: cap=2 but one request carries 3 texts -- it alone
    exceeds the cap, so it must be admitted as its OWN batch (embed_texts'
    sub-batching keeps it bounded) rather than wedging the queue with a
    never-admitted HEAD (the scheduler would spin empty batches forever). A tiny
    job queued behind it must ALSO complete (proves no wedge / no starvation)."""
    agg = EmbedAggregator(_FakeModel(), None, enabled=True, window_ms=0, max_bsz=2)
    agg.start()

    big_future = agg.submit(["a", "b", "c"])  # 3 texts > cap 2
    small_future = agg.submit(["d"])          # behind the oversized HEAD

    results = await asyncio.gather(big_future, small_future)

    assert results[0] == [_vec("a"), _vec("b"), _vec("c")]
    assert results[1] == [_vec("d")]
    # The oversized HEAD got its own batch; the 1-text job a later window. The
    # queue drained fully -- no wedge.
    assert fake.calls == [["a", "b", "c"], ["d"]]
    assert agg.pending_count == 0
    await agg.stop()


# ---------------------------------------------------------------------------
# Scheduler survives a failed embed_texts window (finding-2 class).
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_embed_texts_raise_fails_jobs_but_scheduler_survives(fake, monkeypatch):
    """When embed_texts RAISES inside a window, that window's awaiting
    submit(s) must fail with per-request exceptions (-> 500s) AND the scheduler
    must survive to service a subsequent window -- one bad window cannot sink
    the aggregator or orphan later requests."""
    calls = {"n": 0}

    def flaky(model, tokenizer, texts, normalize=True):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("boom: first window fails")
        return [_vec(t) for t in texts]

    monkeypatch.setattr(agg_mod, "embed_texts", flaky)
    agg = EmbedAggregator(_FakeModel(), None, enabled=True, window_ms=0, max_bsz=4)
    agg.start()

    caught = agg.submit(["w0a", "w0b"])  # first window -> embed_texts raises
    with pytest.raises(RuntimeError, match="boom"):
        await caught
    assert agg.pending_count == 0
    assert calls["n"] == 1

    # The scheduler kept looping: a later request is serviced normally.
    ok = await agg.submit(["ok1"])
    assert ok == [_vec("ok1")]
    assert calls["n"] == 2
    await agg.stop()


# ---------------------------------------------------------------------------
# No-cap (max_prefill_bsz<=0) mode: aggregated batch bounded by hard ceiling.
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_no_cap_mode_bounds_aggregated_batch_by_hard_ceiling(fake):
    """Finding-3 regression: with a model carrying no cap (max_prefill_bsz=0)
    and no explicit override, the collected burst must be chunked into
    embed_texts calls each bounded by the hard ceiling -- never one huge
    un-isolated shared batch against a co-resident :8081 chat."""
    ceiling = agg_mod._HARD_CEILING_DEFAULT
    assert ceiling > 0
    agg = EmbedAggregator(_FakeModel(), None, enabled=True, window_ms=0)
    agg.start()

    n_jobs = ceiling + 4  # more single-text jobs than one window's ceiling
    futures = [agg.submit([f"x{i}"]) for i in range(n_jobs)]
    results = await asyncio.gather(*futures)

    assert all(len(r) == 1 for r in results)
    assert sum(len(r) for r in results) == n_jobs
    # Every shared embed_texts call is <= the hard ceiling; the extra jobs span
    # more than one call, so the window never collapsed into a single giant
    # batch.
    assert fake.calls
    assert all(len(c) <= ceiling for c in fake.calls)
    assert sum(len(c) for c in fake.calls) == n_jobs
    assert len(fake.calls) > 1
    await agg.stop()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))