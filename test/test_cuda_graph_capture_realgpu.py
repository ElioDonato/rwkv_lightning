"""
Real-GPU CUDA-graph capture smoke for the decode forward (REQUIRES a real
checkpoint + GPU; skipped otherwise).

Phase 3b: the seq/batch WKV kernel previously launched on the implicit LEGACY
CUDA stream (no stream arg), unlike `forward_one`/`spmv` which pass the current
stream. Under eager execution current-stream == default, so that was invisible.
But inside a CUDA-graph capture the legacy-stream kernel runs on a DIFFERENT
stream than every other op, so it was never captured -- replay produced stale /
wrong results, which is exactly why decode-step graph capture was blocked (see
the old dead `_init_cuda_graph_state` note). The fix passed the current stream
to the seq kernel. This test proves `forward_batch` (which drives the seq/
``*_seq_batch`` kernels) is now capturable: it captures one step, replays it on
a fresh initial state with different input tokens, and asserts each replay
matches a fresh eager forward within fp16 tolerance -- AND that replays are
deterministic (mutual consistency). With the legacy-stream divergence, replay
diverges from eager; with the fix it reproduces it.

Mirrors test_single_decode_realgpu.py skip idiom: no CUDA or no checkpoint ->
skip, so GPU-less CI stays green. Run with the repo env sourced:

    source env.sh && .venv/bin/python -m pytest test/test_cuda_graph_capture_realgpu.py -v
"""
import os
import sys
from pathlib import Path

import pytest

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from infer.rwkv_batch import rwkv7  # noqa: F401  (builds/loads the CUDA ext)
except OSError as exc:
    pytest.skip(f"CUDA environment not configured: {exc}", allow_module_level=True)

_PYTEST_GPU = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a real CUDA GPU"
)

_DEFAULT_MODEL = None
_p = ROOT / "models" / "rwkv7-g1i-2.9b-20260805-ctx16384.pth"
if _p.exists():
    _DEFAULT_MODEL = str(_p)


def _resolve_model_path():
    explicit = os.environ.get("RWKV_EMBED_TEST_MODEL")
    if explicit:
        return explicit
    if _DEFAULT_MODEL is None:
        pytest.skip(
            "no real checkpoint found (set RWKV_EMBED_TEST_MODEL to run this "
            "real-GPU CUDA-graph smoke)"
        )
    return _DEFAULT_MODEL


_PROMPT = "the quick brown fox jumps over the lazy dog and keeps running"


@_PYTEST_GPU
def test_real_gpu_forward_batch_is_cuda_graph_capturable():
    from model_load.model_loader import load_model_and_tokenizer

    model_path = _resolve_model_path()
    model, tokenizer, _args, _rocm = load_model_and_tokenizer(model_path)
    device = "cuda"

    tokens = tokenizer.encode(_PROMPT)
    assert len(tokens) > 1

    # Bsz=1 batch-layout state seeded from the bsz=0 prefill (same reshape the
    # decode loops use: forward_one -> forward_batch).
    state0 = model.generate_zero_state(0)
    out0 = model.forward(tokens, state0)
    vocab = out0.size(-1)
    base = [
        state0[0].unsqueeze(2).contiguous(),  # [Layer,2,1,C]
        state0[1].unsqueeze(1).contiguous(),  # [Layer,1,H,N,N]
        state0[2].reshape(1),                 # [1]
    ]

    def fresh_state():
        return [t.clone() for t in base]

    # fixed token tensors whose ADDRESSES the graph will read on replay; we
    # mutate the VALUE between replays to prove the graph re-reads inputs.
    tok_a = (tokens[-1] + 7) % vocab
    tok_b = (tokens[-1] + 11) % vocab
    ta = torch.tensor([tok_a], dtype=torch.long, device=device)
    tb = torch.tensor([tok_b], dtype=torch.long, device=device)

    def eager(t, st):
        with torch.no_grad():
            return model.forward_batch(t, st)

    # Reference (fresh state each call) for comparison.
    ref_a = eager(ta, fresh_state())
    ref_b = eager(tb, fresh_state())

    # Graph-static tensors: the SAME objects every replay (addresses must not
    # change), reset in-place to the base state before each replay.
    g_state = fresh_state()
    g_out = torch.empty_like(ref_a)

    # Warm up the op path before capture (lazy init / allocator steady state).
    eager(ta, fresh_state())
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.no_grad():
        with torch.cuda.graph(g):
            o = model.forward_batch(ta, g_state)
            g_out.copy_(o)

    def replay_with(token_val, st_reset):
        for gs, rs in zip(g_state, st_reset):
            gs.copy_(rs)
        # Feed the requested token through the graph's CAPTURED input buffer so
        # replay actually computes with it (replay re-reads `ta`'s address).
        ta.fill_(token_val)
        g.replay()  # execution happens on replay, and is host-synced by replay
        return g_out.float().detach().cpu()

    def close_enough(actual, reference, what):
        a = actual.float()
        b = reference.float().detach().cpu()
        assert a.shape == b.shape, f"{what} shape {tuple(a.shape)} vs {tuple(b.shape)}"
        peak = max(a.abs().max().item(), b.abs().max().item(), 1.0)
        rel = (a - b).abs().max().item() / peak
        assert rel < 0.02, (
            f"{what} peak-relative diff {rel:.3e} >= 2%: captured replay diverges "
            "from eager (seq kernel not captured? stream divergence?)"
        )

    # First replay on token A, fresh state => must reproduce eager A (executes
    # the captured graph and reads host-synced output).
    act_a = replay_with(tok_a, fresh_state())
    close_enough(act_a, ref_a, "replay(A)")
    # Replay on token B => must reproduce eager B (proves inputs are re-read,
    # not baked at capture).
    act_b = replay_with(tok_b, fresh_state())
    close_enough(act_b, ref_b, "replay(B)")
    # Replays on identical (input,state) must be mutually deterministic.
    again_a1 = replay_with(tok_a, fresh_state())
    again_a2 = replay_with(tok_a, fresh_state())
    assert torch.equal(again_a1, again_a2), (
        "replay not deterministic -- captured work has data-dependent/stream "
        "divergence"
    )
    del g, g_state, g_out, ta, tb, model
    torch.cuda.empty_cache()