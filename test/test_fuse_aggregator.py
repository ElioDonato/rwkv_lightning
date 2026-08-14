"""
Hermetic coverage for the server-side decode COMBINE-QUEUE
(infer/fuse_aggregator.ChatFuseAggregator, RWKV_FUSE_CHAT_BATCH opt-in, item 3).

Zero network, zero real GPU, zero real model. The schedulers' admission /
split-back / homogeneity / cap / cancellation bookkeeping is the unit under
test; the engine (single_infer_stream / big_batch_stream / prefill permits) is
a deterministic CPU stand-in. Pattern mirrors test_embed_aggregator.py /
test_async_forward.py; anyio's asyncio runner drives the coroutine tests.

Covers: (a) default-off is an inline single_infer_stream proxy (byte-identical,
no queue/scheduler); (b) HEAD-FIRE: a lone request runs solo through
single_infer_stream, never a window-delayed big_batch; (c) fusion: homogeneous
concurrent requests -> ONE big_batch_stream over the merged prompts, split back
per request exactly; (d) homogeneity gate: differing temperature/max_tokens/
stop_tokens never fuse; (e) VRAM cap (RWKV_FUSE_CHAT_MAX_BSZ) splits a burst
into multiple <= cap batches; (f) per-row cancellation never fails the batch;
(g) batch-level [DONE] is not forwarded (per-row _END ends each stream).
"""
import asyncio
import json
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

pytest.importorskip("torch", reason="torch not installed (source env.sh first)")

try:
    import infer.fuse_aggregator as fuse_mod
except OSError as exc:
    pytest.skip(f"CUDA environment not configured: {exc}", allow_module_level=True)

from infer.fuse_aggregator import ChatFuseAggregator


@pytest.fixture
def anyio_backend():
    return "asyncio"


class _FakeEngine:
    """Deterministic engine stand-in: records single_infer_stream / big_batch_stream
    calls and prefill-permit admissions, stream returns controllable per-prompt
    SSE chunks in the real methods' format."""

    def __init__(self):
        self.solo_calls = []   # (prompt, max_length, temperature, stop_tokens)
        self.solo_kw = []      # per-solo sampler controls (top_k/top_p/alpha_*/prefix_cache_manager)
        self.batch_calls = []  # (prompts_list, max_length, temperature, stop_tokens)
        self.permits = []      # admitted request_bsz values
        self.model = types.SimpleNamespace(max_prefill_bsz_limit=64)

    def _solo_chunk(self, prompt):
        data = json.dumps({"object": "chat.completion.chunk",
                           "choices": [{"index": 0, "delta": {"content": f"[{prompt}:solo]"}}]},
                          ensure_ascii=False)
        return f"data: {data}\n\n"

    def _finish_chunk(self):
        data = json.dumps({"object": "chat.completion.chunk",
                           "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
                          ensure_ascii=False)
        return f"data: {data}\n\n"

    async def acquire_prefill_permit(self, request_bsz, request_label="", cancel_token=None):
        self.permits.append(request_bsz)
        return {"ticket": len(self.permits) - 1, "request_bsz": request_bsz,
                "outstanding_bsz": [request_bsz]}

    async def release_prefill_permit(self, request_bsz, request_label="", ticket=None, permit=None):
        pass

    async def single_infer_stream(self, prompt, max_length, temperature, stop_tokens,
                                  chunk_size=1, **kw):
        self.solo_calls.append((prompt, max_length, temperature, tuple(stop_tokens)))
        self.solo_kw.append(dict(kw))
        yield self._solo_chunk(prompt)
        yield self._finish_chunk()
        yield "data: [DONE]\n\n"

    async def big_batch_stream(self, prompts, max_length, temperature, stop_tokens,
                               chunk_size=1, permit_box=None, **kw):
        self.batch_calls.append((list(prompts), max_length, temperature, tuple(stop_tokens)))
        for i, p in enumerate(prompts):
            data = json.dumps({"object": "chat.completion.chunk",
                               "choices": [{"index": i, "delta": {"content": f"[{p}:fused]"}}]},
                              ensure_ascii=False)
            yield f"data: {data}\n\n"
        for i, p in enumerate(prompts):
            data = json.dumps({"object": "chat.completion.chunk",
                               "choices": [{"index": i, "delta": {}, "finish_reason": "stop"}]},
                              ensure_ascii=False)
            yield f"data: {data}\n\n"
        yield "data: [DONE]\n\n"


async def _drain(stream):
    """Collect every SSE string the fuse stream yields."""
    out = []
    async for item in stream:
        out.append(item)
    return out


async def _collect_text(stream):
    """Accumulate content + finish_reason the way the non-stream route does."""
    parts = []
    finish = None
    async for item in stream:
        if not item.startswith("data: "):
            continue
        payload = item[6:].strip()
        if payload == "[DONE]":
            continue
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            continue
        choices = data.get("choices") or []
        if not choices:
            continue
        if choices[0].get("finish_reason") is not None:
            finish = choices[0]["finish_reason"]
        content = choices[0].get("delta", {}).get("content")
        if content:
            parts.append(content)
    return "".join(parts), finish


# ---------------------------------------------------------------------------
# (a) Default-off: inline single_infer_stream proxy, no queue / scheduler.
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_default_off_is_inline_solo_no_scheduler():
    engine = _FakeEngine()
    agg = ChatFuseAggregator(engine, enabled=False)
    assert agg.enabled is False
    assert agg._task is None

    stream = await agg.submit("p1", 32, 1.0, ["\nUser:"], 2)
    chunks = await _drain(stream)

    # Inline solo: went through single_infer_stream, one permit for bsz=1.
    assert engine.solo_calls == [("p1", 32, 1.0, ("\nUser:",))]
    assert engine.batch_calls == []
    assert [c for c in chunks if "[p1:solo]" in c], "solo content must be streamed"
    assert chunks[-1] == "data: [DONE]\n\n"

    agg.start()  # no-op while disabled
    assert agg._task is None
    assert agg.pending_count == 0


# ---------------------------------------------------------------------------
# (b) HEAD-FIRE: a lone request runs solo, never window-gathered into big_batch.
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_head_fire_lone_request_runs_solo():
    engine = _FakeEngine()
    agg = ChatFuseAggregator(engine, enabled=True, window_ms=1000, max_bsz=8)
    agg.start()

    # Single job with a large window: must fire IMMEDIATELY via single_infer_stream
    # (no 1000ms gather delay, no big_batch), because the queue was empty.
    stream = await agg.submit("lone", 32, 1.0, ["\nUser:"], 2)
    chunks = await asyncio.wait_for(_drain(stream), timeout=0.5)

    assert engine.solo_calls, "lone request must run solo (head-fire)"
    assert engine.batch_calls == [], "a lone request must never use big_batch"
    assert any("[lone:solo]" in c for c in chunks)
    await agg.stop()


# ---------------------------------------------------------------------------
# (c) Fusion: homogeneous concurrent requests -> ONE big_batch decode, split back.
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_fuses_homogeneous_concurrent_into_one_shared_batch():
    engine = _FakeEngine()
    agg = ChatFuseAggregator(engine, enabled=True, window_ms=0, max_bsz=8)
    agg.start()

    # No await-yield between the two submits (submit() is synchronous in the
    # enabled path), so the scheduler drains both together -> one fused batch.
    s1 = await agg.submit("alpha", 32, 1.0, ["\nUser:"], 2)
    s2 = await agg.submit("beta", 32, 1.0, ["\nUser:"], 2)

    out1 = await _drain(s1)
    out2 = await _drain(s2)

    # ONE shared multi-row big_batch_stream over the merged prompts.
    assert engine.batch_calls == [(["alpha", "beta"], 32, 1.0, ("\nUser:",))]
    assert engine.solo_calls == [], "concurrent homogeneous rows must fuse, not solo"
    assert any("[alpha:fused]" in c for c in out1)
    assert any("[beta:fused]" in c for c in out2)
    # Per-row split keeps per-index text and finish; the batch-level [DONE] is
    # NOT forwarded -- each row stream ends after its own finish chunk.
    assert "data: [DONE]" not in out1
    assert any("finish_reason" in c for c in out1)
    assert any("finish_reason" in c for c in out2)
    await agg.stop()


@pytest.mark.anyio
async def test_fused_permits_count_each_row():
    engine = _FakeEngine()
    agg = ChatFuseAggregator(engine, enabled=True, window_ms=0, max_bsz=8)
    agg.start()

    s1 = await agg.submit("a", 32, 1.0, ["\nUser:"], 2)
    s2 = await agg.submit("b", 32, 1.0, ["\nUser:"], 2)
    s3 = await agg.submit("c", 32, 1.0, ["\nUser:"], 2)
    await _drain(s1)
    await _drain(s2)
    await _drain(s3)

    # The fused batch admitted request_bsz = 3 (one permit covering all rows).
    assert engine.permits == [3]
    await agg.stop()


# ---------------------------------------------------------------------------
# (d) Homogeneity gate: differing temperature / max_tokens / stop never fuse.
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_homogeneity_gate_temperature_mismatch_never_fuses():
    engine = _FakeEngine()
    agg = ChatFuseAggregator(engine, enabled=True, window_ms=0, max_bsz=8)
    agg.start()

    s1 = await agg.submit("t10", 32, 1.0, ["\nUser:"], 2)
    s2 = await agg.submit("t20", 32, 0.8, ["\nUser:"], 2)  # temperature differs

    await _drain(s1)
    await _drain(s2)

    assert engine.solo_calls, "non-matching temperature must run solo"
    assert engine.batch_calls == [], "requests that disagree must never share a batch"
    assert len(engine.solo_calls) == 2
    await agg.stop()


@pytest.mark.anyio
async def test_homogeneity_gate_max_tokens_mismatch_never_fuses():
    engine = _FakeEngine()
    agg = ChatFuseAggregator(engine, enabled=True, window_ms=0, max_bsz=8)
    agg.start()

    s1 = await agg.submit("a", 32, 1.0, ["\nUser:"], 2)
    s2 = await agg.submit("b", 64, 1.0, ["\nUser:"], 2)  # max_tokens differs

    await _drain(s1)
    await _drain(s2)

    assert engine.solo_calls and engine.batch_calls == []
    assert len(engine.solo_calls) == 2
    await agg.stop()


@pytest.mark.anyio
async def test_homogeneity_gate_stop_tokens_mismatch_never_fuses():
    engine = _FakeEngine()
    agg = ChatFuseAggregator(engine, enabled=True, window_ms=0, max_bsz=8)
    agg.start()

    s1 = await agg.submit("a", 32, 1.0, ["\nUser:"], 2)
    s2 = await agg.submit("b", 32, 1.0, ["\n\n"], 2)  # stop differs

    await _drain(s1)
    await _drain(s2)

    assert engine.solo_calls and engine.batch_calls == []
    assert len(engine.solo_calls) == 2
    await agg.stop()


# ---------------------------------------------------------------------------
# (d2) Sampler-divergence gate (CR1/CR2 item 1): a job carrying a NON-default
# top_k/top_p/alpha or use_prefix_cache=True is marked not-fusable and NEVER
# shares a fused batch; it runs solo (the request's sampler controls are then
# ignored per the documented contract, but it no longer silently fuses with
# rows whose top_* differ).
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_homogeneity_gate_top_p_mismatch_never_fuses():
    engine = _FakeEngine()
    agg = ChatFuseAggregator(engine, enabled=True, window_ms=0, max_bsz=8)
    agg.start()

    s1 = await agg.submit("a", 32, 1.0, ["\nUser:"], 2)  # default top_p=0.6
    s2 = await agg.submit("b", 32, 1.0, ["\nUser:"], 2, top_p=0.9)  # non-default

    await _drain(s1)
    await _drain(s2)

    # The non-default top_p job must NOT fuse into a shared batch; both run solo.
    assert engine.batch_calls == [], "a top_p-mismatch pair must never share a batch"
    assert len(engine.solo_calls) == 2
    await agg.stop()


@pytest.mark.anyio
async def test_homogeneity_gate_top_k_non_default_never_fuses():
    engine = _FakeEngine()
    agg = ChatFuseAggregator(engine, enabled=True, window_ms=0, max_bsz=8)
    agg.start()

    s1 = await agg.submit("a", 32, 1.0, ["\nUser:"], 2, top_k=1)  # non-default top_k
    s2 = await agg.submit("b", 32, 1.0, ["\nUser:"], 2)

    await _drain(s1)
    await _drain(s2)

    assert engine.batch_calls == [], "non-default top_k must never share a batch"
    assert len(engine.solo_calls) == 2
    await agg.stop()


@pytest.mark.anyio
async def test_homogeneity_gate_alpha_non_default_never_fuses():
    engine = _FakeEngine()
    agg = ChatFuseAggregator(engine, enabled=True, window_ms=0, max_bsz=8)
    agg.start()

    # Non-default repetition-penalty control (alpha_presence).
    s1 = await agg.submit("a", 32, 1.0, ["\nUser:"], 2, alpha_presence=1.5)
    s2 = await agg.submit("b", 32, 1.0, ["\nUser:"], 2)

    await _drain(s1)
    await _drain(s2)

    assert engine.batch_calls == [], "non-default alpha must never share a batch"
    assert len(engine.solo_calls) == 2
    await agg.stop()


@pytest.mark.anyio
async def test_use_prefix_cache_job_excluded_from_fusion():
    engine = _FakeEngine()
    agg = ChatFuseAggregator(engine, enabled=True, window_ms=0, max_bsz=8)
    agg.start()

    s1 = await agg.submit("a", 32, 1.0, ["\nUser:"], 2)
    # use_prefix_cache=True: the fused path has no prefix-cache wiring, so the
    # job is excluded from fusion entirely (runs solo).
    s2 = await agg.submit("b", 32, 1.0, ["\nUser:"], 2, use_prefix_cache=True)

    await _drain(s1)
    await _drain(s2)

    assert engine.batch_calls == [], \
        "use_prefix_cache=True must exclude a job from fusion entirely"
    assert len(engine.solo_calls) == 2
    await agg.stop()


# ---------------------------------------------------------------------------
# (e) VRAM cap: a burst larger than the hard cap splits into <= cap batches.
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_cap_splits_burst_into_bounded_batches():
    engine = _FakeEngine()
    agg = ChatFuseAggregator(engine, enabled=True, window_ms=0, max_bsz=3)
    agg.start()

    streams = [await agg.submit(f"w{i}", 32, 1.0, ["\nUser:"], 2) for i in range(6)]
    for s in streams:
        await _drain(s)

    # 6 homogeneous jobs, cap 3 -> two fused batches of 3 each, never a batch of 6.
    assert engine.batch_calls == [
        (["w0", "w1", "w2"], 32, 1.0, ("\nUser:",)),
        (["w3", "w4", "w5"], 32, 1.0, ("\nUser:",)),
    ]
    assert all(len(calls[0]) <= 3 for calls in engine.batch_calls)
    assert engine.solo_calls == []
    await agg.stop()


# ---------------------------------------------------------------------------
# (f) Cancellation: dropping one row never fails the shared batch.
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_cancel_one_row_never_fails_batch():
    engine = _FakeEngine()
    agg = ChatFuseAggregator(engine, enabled=True, window_ms=0, max_bsz=8)
    agg.start()

    s1 = await agg.submit("live", 32, 1.0, ["\nUser:"], 2)
    s2 = await agg.submit("dead", 32, 1.0, ["\nUser:"], 2)
    s2.cancel()  # client disconnect on row 2

    out1 = await _drain(s1)  # the live row still completes
    out_dead = await _drain(s2)  # dropped row yields nothing / ends immediately

    # The batch still ran as ONE fused decode (never failed).
    assert engine.batch_calls == [(["live", "dead"], 32, 1.0, ("\nUser:",))]
    # live row got its content + finish; dead row stream ended without content.
    assert any("[live:fused]" in c for c in out1)
    assert any("finish_reason" in c for c in out1)
    assert not any("[dead:fused]" in c for c in out_dead)
    await agg.stop()


# ---------------------------------------------------------------------------
# (g) Route helper: collect_fuse_nonstream accumulates solo content+finish.
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_collect_fuse_nonstream_reproduces_text_and_finish():
    from API_servers.router.openai_routes import collect_fuse_nonstream

    engine = _FakeEngine()
    agg = ChatFuseAggregator(engine, enabled=False)
    stream = await agg.submit("alpha", 32, 1.0, ["\nUser:"], 16)

    text, finish = await collect_fuse_nonstream(stream)
    assert text == "[alpha:solo]"
    assert finish == "stop"


# ---------------------------------------------------------------------------
# (h) Shutdown ends queued/in-flight jobs (no orphaned fuse stream).
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_stop_is_idempotent_and_submitted_never_orphans():
    engine = _FakeEngine()
    agg = ChatFuseAggregator(engine, enabled=True, window_ms=0, max_bsz=8)
    agg.start()

    stream = await agg.submit("queued", 32, 1.0, ["\nUser:"], 2)
    await agg.stop()
    await agg.stop()  # idempotent

    # A submitted stream must never hang on shutdown: it either completes (if the
    # job already ran to finish before stop) or is abandoned -> ends with the
    # shutdown exception raised to the reader. Both within the deadline.
    try:
        await asyncio.wait_for(_drain(stream), timeout=0.5)
    except (RuntimeError, asyncio.CancelledError):
        pass
    assert agg._task is None


@pytest.mark.anyio
async def test_abandon_resolves_queued_and_inflight_jobs():
    """The scheduler shutdown contract: every not-yet-run request -- both
    still-queued (_pending) jobs and jobs admitted into an in-flight batch
    (_inflight_batch) -- is ended (queue sentinel) and its future raised, so no
    awaiting fuse stream is orphaned. Exercised deterministically by injecting
    jobs and calling the exact cleanup the shutdown/crash handler uses."""
    agg = ChatFuseAggregator(_FakeEngine(), enabled=True, window_ms=0, max_bsz=8)
    loop = asyncio.get_running_loop()

    queued = [
        fuse_mod._ChatJob(i, f"q{i}", 32, 1.0, ("\nUser:",), 2,
                          20, 0.6, 1.0, 0.1, 0.996, False, None, loop) for i in range(2)
    ]
    inflight = [
        fuse_mod._ChatJob(10 + i, f"f{i}", 32, 1.0, ("\nUser:",), 2,
                          20, 0.6, 1.0, 0.1, 0.996, False, None, loop) for i in range(2)
    ]
    agg._pending.extend(queued)
    agg._inflight_batch = inflight

    agg._abandon_jobs(RuntimeError("chat fuse aggregator shut down"))

    for job in queued + inflight:
        assert job.ended, "every job queue must be ended"
        assert job.future.exception() is not None, "every job future must be raised"
        assert job.queue.empty() is False  # the _END sentinel is present
        # Draining yields exactly the sentinel (no data), deterministically.
        out = []
        while not job.queue.empty():
            item = job.queue.get_nowait()
            if item is not fuse_mod._END:
                out.append(item)
        assert out == [], "abandoned jobs must carry no stray data"

    assert agg.pending_count == 0
    assert agg._inflight_batch is None


# ---------------------------------------------------------------------------
# (i) Pending-deque overload bound (CR1 item 3): a burst beyond the pending cap
# is REJECTED (served inline solo through single_infer_stream) rather than
# buffered unboundedly -- the caller falls back to the solo path.
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_pending_cap_rejects_burst_instead_of_buffering():
    engine = _FakeEngine()
    # pending_cap=2: the first two concurrent submits fill the deque; the third
    # arrives at capacity and is rejected (inline solo), never queued forever.
    agg = ChatFuseAggregator(engine, enabled=True, window_ms=0, max_bsz=8,
                             pending_cap=2)
    agg.start()

    # All three submits are synchronous (no await between them), so the
    # scheduler -- awaiting _wake.wait() -- has not run yet and _pending holds
    # exactly the two queued jobs when the third arrives.
    s1 = await agg.submit("a", 32, 1.0, ["\nUser:"], 2)
    s2 = await agg.submit("b", 32, 1.0, ["\nUser:"], 2)
    s3 = await agg.submit("c", 32, 1.0, ["\nUser:"], 2)

    # Third request rejected: inline solo proxy, not a queued fuse job.
    assert isinstance(s3, fuse_mod._InlineFuseStream), \
        "at pending-capacity a submit must be rejected to the inline solo path"
    assert agg.pending_count == 2, "pending deque must never exceed the cap"

    out1 = await _drain(s1)
    out2 = await _drain(s2)
    out3 = await _drain(s3)

    # The two queued homogeneous jobs fused into ONE shared batch; the rejected
    # request ran solo via single_infer_stream. Nothing was queued forever.
    assert engine.batch_calls == [(["a", "b"], 32, 1.0, ("\nUser:",))]
    assert any("[c:solo]" in c for c in out3), "rejected request served solo"
    assert agg.pending_count == 0
    await agg.stop()


# ---------------------------------------------------------------------------
# (j) HIGH-2: an UNFUSED (solo) request is byte-identical to fuse=OFF -- the
# solo path forwards the request's OWN sampler controls (top_k/top_p/alpha_*)
# into single_infer_stream, so enabling the flag never changes a request that
# isn't actually fused.
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_non_fusable_top_k_non_default_solo_passes_exact_top_k():
    """(a) A non-fusable job (top_k non-default, always excluded from fusion)
    served solo must pass its EXACT top_k to the underlying single_infer_stream
    -- not fall back to a hardcoded temp-only sampler."""
    engine = _FakeEngine()
    agg = ChatFuseAggregator(engine, enabled=True, window_ms=0, max_bsz=8)
    agg.start()

    # top_k=1 is non-default -> not fusable; a lone request head-fires solo.
    stream = await agg.submit("k", 32, 1.0, ["\nUser:"], 2, top_k=1)
    await _drain(stream)

    assert engine.solo_calls, "non-fusable job must run solo"
    assert engine.batch_calls == [], "non-fusable job must never fuse"
    # route default passes top_k=20 -> the CHAT route value; the solo path must
    # forward the REQUEST's top_k (1), not the sampler's standalone default (50).
    assert engine.solo_kw and engine.solo_kw[0]["top_k"] == 1, \
        f"solo must forward the request's exact top_k=1, got {engine.solo_kw}"
    await agg.stop()


@pytest.mark.anyio
async def test_default_solo_passes_route_top_k_20_not_50():
    """(b) A DEFAULT request under fuse=ON served solo must pass top_k=20 (the
    /openai/v1/chat/completions route default) -- not single_infer_stream's own
    standalone default of 50. This is the 'fuse=ON-solo == fuse=OFF-solo for the
    same request' byte-identity guarantee."""
    engine = _FakeEngine()
    agg = ChatFuseAggregator(engine, enabled=True, window_ms=0, max_bsz=8)
    agg.start()

    # Default top_k (submit default = _FUSE_TOP_K_DEFAULT = 20); lone -> solo.
    stream = await agg.submit("d", 32, 1.0, ["\nUser:"], 2)
    await _drain(stream)

    assert engine.solo_calls, "default lone request must head-fire solo"
    assert engine.solo_kw and engine.solo_kw[0]["top_k"] == 20, \
        f"default solo must forward top_k=20 (route default), got {engine.solo_kw}"
    # top_p default is 0.6 in the route AND in single_infer_stream, so it must
    # be forwarded verbatim either way -- a regression that drops it to the
    # route/standalone default would be invisible, so assert it is PRESENT.
    assert engine.solo_kw[0].get("top_p") == 0.6
    await agg.stop()


# ---------------------------------------------------------------------------
# (k) HIGH-1: the pending-cap REJECT path must respect VRAM admission -- it
# acquires + releases a bsz=1 prefill permit around its inline single_infer
# (mirroring _run_solo), so a before-merge path never bypasses the shared
# max-prefill-bsz budget exactly when a burst has overloaded the queue.
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_cap_reject_acquires_prefill_permit():
    engine = _FakeEngine()
    agg = ChatFuseAggregator(engine, enabled=True, window_ms=0, max_bsz=8,
                             pending_cap=1)
    agg.start()

    s1 = await agg.submit("a", 32, 1.0, ["\nUser:"], 2)
    s2 = await agg.submit("b", 32, 1.0, ["\nUser:"], 2)  # queue already at cap=1

    # The second submit is rejected to the inline solo path.
    assert isinstance(s2, fuse_mod._InlineFuseStream)

    await _drain(s1)
    out2 = await _drain(s2)
    assert any("[b:solo]" in c for c in out2)

    # The reject's inline solo acquired its own bsz=1 prefill permit (in
    # addition to the fused/queued job's). Without HIGH-1 this reject path was
    # the ONE before-merge route that iterated single_infer_stream with NO
    # permit -- exactly the sustained-burst OOM hole this test closes.
    assert 1 in engine.permits, \
        f"reject path must acquire a bsz=1 prefill permit, permits={engine.permits}"
    await agg.stop()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))