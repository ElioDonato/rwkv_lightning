"""Hermetic coverage for the DYNAMIC BATCH DECODER (Phase-4 sub-step 2,
infer/dynamic_batch.DynamicBatchDecoder, RWKV_DYNAMIC_BATCH opt-in).

Zero network, zero real GPU, zero real model. The decoder's admission / refill /
per-row sampling / compaction / capacity-release / SSE bookkeeping is the unit
under test; the engine is the REAL InferenceEngine driven by a deterministic CPU
_FakeModel + _FakeTokenizer, with the CUDA sampling ops monkeypatched to CPU
stand-ins (mirroring test_per_row_sampling.py / test_async_forward.py). The real
engine's stop-state, prefill and compaction methods run for real, so this
exercises the actual integration points, not another copy of the logic.

Covers: (a) a lone enabled job head-fires (no gather-window latency) and
completes, emitting chunks + [DONE]; (b) two concurrent jobs with DIFFERENT
sampler params share one batch, each row sampled with its OWN params, and both
complete with per-row-consistent output; (c) a finished row is dropped from the
batch and its capacity released; (d) disabled mode returns the inline default
path and never spawns a scheduler.
"""
import asyncio
import json
import sys
from pathlib import Path

import pytest

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

pytest.importorskip("torch", reason="torch not installed (source env.sh first)")

try:
    from infer import inference_deps
    from infer.async_forward import GpuAsyncExecutor
    from infer.inference import InferenceEngine
    from infer.dynamic_batch import DynamicBatchDecoder
    from infer.fuse_aggregator import _InlineFuseStream
except OSError as exc:
    pytest.skip(f"CUDA environment not configured: {exc}", allow_module_level=True)

from settings import settings


VOCAB = 64  # > any token id the fake decode produces, so argmax is well-defined


class _FakeModel:
    """CPU model stand-in with the shape of the model interface the decoder
    reads. Prefill gives each row its own one-hot logits (row r starts at token
    r+1); the decode step maps token t -> one-hot(t+1), so each row walks a
    deterministic, distinguishable counter trajectory (never token 0, which the
    engine's tokenizer treats as EOS)."""

    def __init__(self):
        self.vocab = VOCAB
        self.prefill_chunk_size = 256
        self.max_prefill_bsz_limit = 64
        self.max_prefill_bsz = 64
        self.forward_calls = 0
        self.same_length_calls = 0

    def generate_zero_state(self, bsz):
        # Shapes match what engine._compact_active_rows / _admit's tensor-grow
        # expect: state[0][.., X, B], state[1][.., X], state[2][B].
        return [
            torch.zeros(1, 2, bsz),
            torch.zeros(1, bsz, 1, 1, 1),
            torch.zeros(bsz, dtype=torch.long),
        ]

    def forward_batch(self, tokens, state, full_output=False):
        self.forward_calls += 1
        t = (tokens.reshape(-1) + 1) % self.vocab
        return torch.nn.functional.one_hot(t.to(torch.long), self.vocab).float()

    def forward_batch_same_length(self, batch_tokens, batch_state, full_output=False):
        self.same_length_calls += 1
        n = len(batch_tokens)
        return torch.nn.functional.one_hot(
            (torch.arange(1, n + 1) % self.vocab).to(torch.long), self.vocab
        ).float()

    def forward(self, tokens, state):
        # solo single_infer prefill gives token 1 as the first decode token.
        return torch.nn.functional.one_hot(
            torch.tensor([1 % self.vocab]), self.vocab
        ).float()


class _FakeTokenizer:
    def encode(self, text):
        return [1, 2, 3]

    def decode(self, tokens, utf8_errors="ignore"):
        return "".join(chr(97 + (int(t) % 26)) for t in tokens)


def _make_engine():
    engine = InferenceEngine(
        model=_FakeModel(), tokenizer=_FakeTokenizer(), args=None, rocm_flag=False
    )
    engine._gpu_executor = GpuAsyncExecutor(enabled=False)
    return engine


def _patch_sampler(monkeypatch, record_box=None):
    """Monkeypatch the CUDA sampling ops to deterministic CPU stand-ins:
    setup_rand -> a dummy per-row block, the whole-batch op -> argmax (1-D [B]),
    and sample_batch_per_row -> argmax assembling [B,1] (recording each step's
    per-row scalar lists into record_box for assertions)."""
    fake_sample = type("_FakeSample", (), {})()

    def _setup_rand(seed, batch_size):
        return torch.zeros(batch_size, dtype=torch.long)

    def _op(logits, penalties, states, ap, af, ad, temp, top_k, top_p):
        return logits.argmax(dim=-1)

    fake_sample.setup_rand = _setup_rand
    fake_sample.batch_sampling_repetition_temperature_topk_topp = _op

    if record_box is None:
        record_box = {"records": []}

    def _sample_batch_per_row(logits, penalties, rand_states, temperature, top_k,
                              top_p, alpha_presence, alpha_frequency, alpha_decay):
        record_box["records"].append(
            {
                "temperature": [float(x) for x in temperature],
                "top_k": [int(x) for x in top_k],
                "top_p": [float(x) for x in top_p],
                "alpha_presence": [float(x) for x in alpha_presence],
                "alpha_frequency": [float(x) for x in alpha_frequency],
                "alpha_decay": [float(x) for x in alpha_decay],
            }
        )
        return logits.argmax(dim=-1).reshape(logits.size(0), 1)

    monkeypatch.setattr(inference_deps, "get_sample", lambda: fake_sample)
    monkeypatch.setattr(
        inference_deps, "get_sample_batch_per_row", lambda: _sample_batch_per_row
    )
    return record_box


async def _drain(stream):
    out = []
    async for item in stream:
        out.append(item)
    return out


def _collect_text(chunks):
    """Accumulate content + finish_reason the way the route's non-stream helper
    does (index-0 dynamic/solo SSE chunks)."""
    parts = []
    finish = None
    for c in chunks:
        if not c.startswith("data: "):
            continue
        payload = c[6:].strip()
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
# (a) A lone enabled job head-fires (no gather-window latency) and completes.
# ---------------------------------------------------------------------------

def test_lone_job_head_fires_and_completes(monkeypatch):
    engine = _make_engine()
    _patch_sampler(monkeypatch)

    async def run():
        dec = DynamicBatchDecoder(engine, enabled=True, window_ms=1000, max_bsz=8)
        dec.start()
        stream = await dec.submit(
            engine, "hello", max_tokens=4, temperature=0.5, stop_tokens=[],
            chunk_size=1, top_k=5, top_p=0.8,
        )
        # window is 1000ms but the drain deadline is 500ms: if the lone job were
        # gather-window delayed it would time out -> the head-fire guard is what
        # makes this complete in time.
        chunks = await asyncio.wait_for(_drain(stream), timeout=0.5)
        await dec.stop()
        return chunks

    chunks = asyncio.run(run())
    assert chunks[-1] == "data: [DONE]\n\n", "lone job must end with [DONE]"
    text, finish = _collect_text(chunks)
    assert finish == "length", "lone job exhausts its small max_tokens"
    assert text, "lone job must stream some content"
    # Went through the dynamic batch (batched prefill used `forward_batch_same_length`),
    # not the inline solo single_infer path.
    assert engine.model.same_length_calls > 0, "dynamic batched prefill must have run"


# ---------------------------------------------------------------------------
# (b) Concurrent jobs with DIFFERENT sampler params share one batch, each row
# sampled with its OWN params, per-row-consistent output.
# ---------------------------------------------------------------------------

def test_two_jobs_different_sampler_params_per_row(monkeypatch):
    engine = _make_engine()
    _patch_sampler(monkeypatch)

    async def run():
        dec = DynamicBatchDecoder(engine, enabled=True, window_ms=0, max_bsz=8)
        dec.start()
        sA = await dec.submit(
            engine, "A", max_tokens=3, temperature=0.5, stop_tokens=[],
            chunk_size=1, top_k=5, top_p=0.8,
        )
        sB = await dec.submit(
            engine, "B", max_tokens=3, temperature=1.0, stop_tokens=[],
            chunk_size=1, top_k=20, top_p=0.6,
        )
        outA = await _drain(sA)
        outB = await _drain(sB)
        await dec.stop()
        return outA, outB

    outA, outB = asyncio.run(run())
    assert outA[-1] == "data: [DONE]\n\n" and outB[-1] == "data: [DONE]\n\n"
    textA, finishA = _collect_text(outA)
    textB, finishB = _collect_text(outB)
    assert finishA == "length" and finishB == "length"

    # Per-row consistency: row 0 walks tokens 1,2,3 -> "bcd"; row 1 walks
    # 2,3,4 -> "cde". (Deterministic under the fake model.)
    assert textA == "bcd", f"row A content wrong: {textA!r}"
    assert textB == "cde", f"row B content wrong: {textB!r}"


def test_different_sampler_params_route_to_own_rows(monkeypatch):
    engine = _make_engine()
    box = _patch_sampler(monkeypatch)

    async def run():
        dec = DynamicBatchDecoder(engine, enabled=True, window_ms=0, max_bsz=8)
        dec.start()
        sA = await dec.submit(
            engine, "A", max_tokens=3, temperature=0.5, stop_tokens=[],
            chunk_size=1, top_k=5, top_p=0.8, alpha_presence=1.5,
        )
        sB = await dec.submit(
            engine, "B", max_tokens=3, temperature=1.0, stop_tokens=[],
            chunk_size=1, top_k=20, top_p=0.6, alpha_presence=0.0,
        )
        await asyncio.gather(_drain(sA), _drain(sB))
        await dec.stop()

    asyncio.run(run())

    # The two jobs were admitted into the SAME batch (row0=A, row1=B); the first
    # decode step routed each row's OWN sampler scalars to the per-row sampler.
    assert box["records"], "per-row sampler must have run"
    first = box["records"][0]
    assert first["top_k"] == [5, 20], f"per-row top_k wrong: {first['top_k']}"
    assert first["temperature"] == [0.5, 1.0]
    assert first["top_p"] == [0.8, 0.6]
    assert first["alpha_presence"] == [1.5, 0.0]
    # And the values are stable across every decode step of the shared batch.
    for rec in box["records"]:
        assert rec["top_k"] == [5, 20], "per-row param must not leak across steps"


# ---------------------------------------------------------------------------
# (c) A finished row is dropped from the batch and its capacity released.
# ---------------------------------------------------------------------------

def test_finished_row_dropped_and_capacity_released(monkeypatch):
    engine = _make_engine()
    _patch_sampler(monkeypatch)

    async def run():
        dec = DynamicBatchDecoder(engine, enabled=True, window_ms=0, max_bsz=8)
        dec.start()
        # Short row (max_tokens=2) and long row (max_tokens=6), admitted together.
        sShort = await dec.submit(
            engine, "S", max_tokens=2, temperature=0.5, stop_tokens=[],
            chunk_size=1, top_k=5,
        )
        sLong = await dec.submit(
            engine, "L", max_tokens=6, temperature=0.5, stop_tokens=[],
            chunk_size=1, top_k=5,
        )
        outShort = await _drain(sShort)
        # When the short row's stream ends, only the long row still holds one
        # slot of the shared prefill-admission budget (admission acquired bsz=2,
        # short released its 1 via per-row release_prefill_capacity).
        reserved_after_short = engine._prefill_reserved_bsz
        outLong = await _drain(sLong)
        reserved_after_long = engine._prefill_reserved_bsz
        await dec.stop()
        return outShort, outLong, reserved_after_short, reserved_after_long

    outShort, outLong, r_short, r_long = asyncio.run(run())

    assert outShort[-1] == "data: [DONE]\n\n" and outLong[-1] == "data: [DONE]\n\n"
    textShort, _ = _collect_text(outShort)
    textLong, _ = _collect_text(outLong)
    # Short row walked tokens 1,2 -> "bc"; long row token 2,3,4,5,6,7 -> "cdefgh".
    assert textShort == "bc", f"short content wrong: {textShort!r}"
    assert textLong == "cdefgh", f"long content wrong: {textLong!r}"

    # Capacity: 2 acquired on admission; short row freed its 1 (reserved now 1),
    # long row freed its 1 too (reserved back to 0) -- incremental release works.
    assert r_short == 1, f"short row must release its capacity: reserved={r_short}"
    assert r_long == 0, f"long row must fully release capacity: reserved={r_long}"


# ---------------------------------------------------------------------------
# (d) Disabled mode returns the inline default path and spawns no scheduler.
# ---------------------------------------------------------------------------

def test_disabled_returns_inline_default_no_scheduler():
    engine = _make_engine()
    dec = DynamicBatchDecoder(engine, enabled=False)
    assert dec.enabled is False
    assert dec._task is None
    dec.start()  # no-op while disabled
    assert dec._task is None, "no scheduler task may be spawned while disabled"

    loop = asyncio.new_event_loop()
    try:
        stream = loop.run_until_complete(
            dec.submit(
                engine, "prompt", max_tokens=4, temperature=1.0, stop_tokens=[],
                chunk_size=1,
            )
        )
    finally:
        loop.close()

    # Disabled submit returns the inline solo proxy (the default path), not a
    # queued dynamic job -- and it never enqueues into / spawns the shared batch.
    assert isinstance(stream, _InlineFuseStream), (
        "disabled submit must return the inline single_infer_stream proxy"
    )
    assert dec._task is None, "disabled submit must not spawn the scheduler"
    assert dec.pending_count == 0, "disabled submit must not enqueue a job"


def test_env_default_off_no_override(monkeypatch):
    """With RWKV_DYNAMIC_BATCH unset (settings default False) and no explicit
    override, the decoder is disabled -- so with no env set the server is
    byte-identical to today."""
    monkeypatch.setattr(settings, "dynamic_batch", False)
    monkeypatch.setattr(settings, "dynamic_batch_window_ms", 8)
    monkeypatch.setattr(settings, "dynamic_batch_max_bsz", 8)
    engine = _make_engine()
    dec = DynamicBatchDecoder(engine)
    assert dec.enabled is False
    assert dec._task is None