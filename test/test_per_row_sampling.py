"""Hermetic coverage for Phase 4 sub-step 1: per-row sampling in the shared
multi-row decode.

Covers the new ``infer.rwkv_batch.sampler.sample_batch_per_row`` helper (the 
per-row application of the existing ``batch_sampling_repetition_temperature_
topk_topp`` op) and its wiring into ``big_batch_stream``:

(a) the helper samples EACH row with that row's OWN temperature / top_k / top_p
    / alpha_presence / alpha_frequency / alpha_decay, passing the row's sliced
    logits / penalties / rand-state and returning a [B, 1] long tensor;
(b) with identical per-row scalars the per-row loop is numerically identical to
    a single whole-batch op call (the equivalence requirement #3) -- proven
    with the REAL CUDA sampling op when a GPU is present (skipped otherwise,
    so the file also runs headless);
(c) ``big_batch_stream`` in per-row mode (caller passes per_row_*) drives its
    ``_gpu_decode_step`` through the helper with each row's own controls and
    feeds the sampled [B,1] tensor into forward_batch, while the DEFAULT path
    (no per_row_*) keeps the existing temperature-only gumbel sampler untouched
    (byte-identical default; covered separately by test_async_forward.py on CPU).

Monkeypatch / FakeModel idiom mirrors test_async_forward.py. The custom CUDA
extensions are imported via the standard infer.rwkv_batch.sampler module, so
this file requires ``source env.sh`` on the box running it.
"""
import asyncio
import sys
import threading
from pathlib import Path

import pytest

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

pytest.importorskip("torch", reason="torch not installed (source env.sh first)")

try:
    import infer.rwkv_batch.sampler as sampler_mod
    from infer.async_forward import GpuAsyncExecutor
    from infer.inference import InferenceEngine
except OSError as exc:
    pytest.skip(f"CUDA environment not configured: {exc}", allow_module_level=True)


# ---------------------------------------------------------------------------
# (a) The helper samples each row with its own scalars.
# ---------------------------------------------------------------------------

def test_sample_batch_per_row_uses_each_rows_own_params(monkeypatch):
    """The helper must call the op ONCE PER ROW, passing that row's sliced
    logits/penalties/rand-state and that row's OWN scalars, and assemble the
    per-row results into a [B, 1] long tensor."""
    calls = []
    next_ctx = [11]

    def _record(logits, penalties, rand_states, ap, af, ad, temp, top_k, top_p):
        calls.append({
            "n": int(logits.shape[0]),
            "pen_shape": tuple(penalties.shape),
            "pen_val": float(penalties[0, 0]),
            "temp": float(temp), "top_k": int(top_k), "top_p": float(top_p),
            "ap": float(ap), "af": float(af), "ad": float(ad),
        })
        tok = next_ctx[0]
        next_ctx[0] += 10
        return torch.tensor([tok], dtype=torch.long)

    monkeypatch.setattr(
        sampler_mod.sample, "batch_sampling_repetition_temperature_topk_topp", _record
    )

    logits = torch.randn(3, 8)
    penalties = torch.zeros(3, 8)
    rand_states = torch.arange(3, dtype=torch.long)
    temps = [1.0, 0.5, 2.0]
    top_k = [50, 20, 0]
    top_p = [0.6, 0.8, 1.0]
    presence = [0.0, 1.0, 0.5]
    freq = [0.1, 0.2, 0.3]
    decay = [0.996, 0.99, 1.0]

    out = sampler_mod.sample_batch_per_row(
        logits, penalties, rand_states,
        temps, top_k, top_p, presence, freq, decay,
    )

    # One call per row, each on that row's sliced (1, V) logits/penalties and
    # a [1]-shaped rand-state slice.
    assert len(calls) == 3
    assert all(c["n"] == 1 for c in calls)
    assert all(c["pen_shape"] == (1, 8) for c in calls)
    assert all(c["pen_val"] == 0.0 for c in calls)

    # Each call received its own row's scalars, in row order.
    assert [c["temp"] for c in calls] == [1.0, 0.5, 2.0]
    assert [c["top_k"] for c in calls] == [50, 20, 0]
    assert [c["top_p"] for c in calls] == [0.6, 0.8, 1.0]
    assert [c["ap"] for c in calls] == [0.0, 1.0, 0.5]
    assert [c["af"] for c in calls] == [0.1, 0.2, 0.3]
    assert [c["ad"] for c in calls] == [0.996, 0.99, 1.0]

    # Results assembled in row order as a [B, 1] long tensor.
    assert out.shape == (3, 1)
    assert out.dtype == torch.long
    assert out[:, 0].tolist() == [11, 21, 31]


def test_sample_batch_per_row_empty_batch_returns_empty():
    out = sampler_mod.sample_batch_per_row(
        torch.zeros(0, 8),
        torch.zeros(0, 8),
        torch.zeros(0, dtype=torch.long),
        [], [], [], [], [], [],
    )
    assert out.shape == (0, 1)
    assert out.dtype == torch.long


# ---------------------------------------------------------------------------
# (b) Row-wise application == whole-batch application when scalars are equal
# (requirement #3) -- proven with the REAL op on GPU, skipped headless.
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="real GPU required")
def test_rowwise_equals_batchwide_real_cuda():
    """With all rows sharing the SAME sampler scalars and identical fresh
    per-row penalties + rand-states (same seed), the per-row loop must return
    exactly the tokens a single whole-batch op call returns. This is the
    'don't change the math, only the distribution-of-application' guarantee:
    the underlying op is already per-row on penalties/states."""
    torch.manual_seed(0)
    B, V = 4, 512  # V must be a multiple of 4 for the sampling kernel
    logits = torch.randn(B, V, device="cuda")
    op = sampler_mod.sample

    # Whole-batch application.
    penalties_batch = torch.zeros(B, V, device="cuda")
    states_batch = op.setup_rand(1234, B)
    batch_tokens = op.batch_sampling_repetition_temperature_topk_topp(
        logits, penalties_batch, states_batch,
        1.0, 0.1, 0.996,   # presence / frequency / decay
        0.9, 30, 0.7,      # temperature / top_k / top_p
    )

    # Per-row application: fresh identical penalties + states, same seed, and
    # every row given the SAME scalar config.
    penalties_row = torch.zeros(B, V, device="cuda")
    states_row = op.setup_rand(1234, B)
    row_tokens = sampler_mod.sample_batch_per_row(
        logits, penalties_row, states_row,
        [0.9] * B, [30] * B, [0.7] * B,
        [1.0] * B, [0.1] * B, [0.996] * B,
    )

    assert row_tokens.shape == (B, 1)
    assert row_tokens[:, 0].tolist() == batch_tokens.tolist()


# ---------------------------------------------------------------------------
# (c) big_batch_stream wiring: per-row mode drives sample_batch_per_row with
# each row's own controls end-to-end; default mode is untouched elsewhere.
# ---------------------------------------------------------------------------

class _FakeModel:
    """CPU model stand-in with the shape of the model interface the decode loop
    reads, recording whether forward_batch received a [B, 1] GPU tensor."""

    def __init__(self, vocab=6):
        self.vocab = vocab
        self.prefill_chunk_size = 2
        self.decode_tensor_inputs = []

    def generate_zero_state(self, bsz):
        return [
            torch.zeros(1, 2, bsz),
            torch.zeros(1, bsz, 1, 1, 1),
            torch.zeros(bsz, dtype=torch.int32),
        ]

    def forward_batch(self, tokens, state, full_output=False):
        self.decode_tensor_inputs.append(isinstance(tokens, torch.Tensor))
        bsz = tokens.shape[0] if isinstance(tokens, torch.Tensor) else len(tokens)
        return torch.zeros(bsz, self.vocab)

    def forward_batch_same_length(self, batch_tokens, batch_state, full_output=False):
        bsz = (
            batch_tokens.shape[0]
            if isinstance(batch_tokens, torch.Tensor)
            else len(batch_tokens)
        )
        return torch.zeros(bsz, self.vocab)


class _FakeTokenizer:
    def encode(self, text):
        return [1, 2, 3]

    def decode(self, tokens, utf8_errors="strict"):
        return ""


def test_big_batch_per_row_mode_honors_each_rows_config(monkeypatch):
    """big_batch_stream in per-row mode must route each decode step through the
    helper with each row's own controls, feed the sampled [B, 1] tensor into
    forward_batch, and stream a complete SSE response. (Default /big_batch --
    no per_row_* -- continues to use gumbel temperature-only and is covered by
    test_async_forward.py.)"""
    seen = []

    def _fake_setup_rand(seed, batch_size):
        return torch.zeros(batch_size, dtype=torch.long)

    def _fake_sample(logits, penalties, rand_states, ap, af, ad, temp, top_k, top_p):
        seen.append(
            (float(temp), int(top_k), float(top_p), float(ap), float(af), float(ad))
        )
        # token 5 is non-zero -> never triggers the token-0 stop path.
        return torch.full((logits.shape[0],), 5, dtype=torch.long)

    monkeypatch.setattr(sampler_mod.sample, "setup_rand", _fake_setup_rand)
    monkeypatch.setattr(
        sampler_mod.sample,
        "batch_sampling_repetition_temperature_topk_topp",
        _fake_sample,
    )

    engine = InferenceEngine(
        model=_FakeModel(), tokenizer=_FakeTokenizer(), args=None, rocm_flag=False
    )
    engine._gpu_executor = GpuAsyncExecutor(enabled=False)

    async def _run():
        chunks = []
        async for c in engine.big_batch_stream(
            prompts=["hello world", "hi there"],
            max_length=4,
            per_row_temperatures=[1.0, 0.5],
            per_row_top_k=[10, 3],
            per_row_top_p=[1.0, 0.8],
            per_row_alpha_presence=[0.0, 1.0],
            per_row_alpha_frequency=[0.1, 0.2],
            per_row_alpha_decay=[1.0, 0.99],
        ):
            chunks.append(c)
        return chunks

    chunks = asyncio.run(_run())

    assert chunks and chunks[-1] == "data: [DONE]\n\n"
    # forward_batch received the sampled tensor fast path.
    assert engine.model.decode_tensor_inputs, "forward_batch must have been called"
    assert all(engine.model.decode_tensor_inputs), "sampled [B,1] tensor fed to forward"

    # Each decode step sampled once per row with that row's OWN scalars.
    assert seen, "per-row sampler must have run per decode step"
    # seen is interleaved row0,row1 across steps; first step:
    assert seen[0] == (1.0, 10, 1.0, 0.0, 0.1, 1.0), f"row0 config wrong: {seen[0]}"
    assert seen[1] == (0.5, 3, 0.8, 1.0, 0.2, 0.99), f"row1 config wrong: {seen[1]}"
    # Every row got its identical config on every step (4 steps x 2 rows).
    assert [seen[2 * k] for k in range(len(seen) // 2)] == [
        (1.0, 10, 1.0, 0.0, 0.1, 1.0)
    ] * (len(seen) // 2)