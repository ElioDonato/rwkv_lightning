"""
Regression coverage for `BatchInferenceMixin._batch_prefill`'s prefix-cache
branch (the `prefix_cache_manager is not None` path in
infer/batch_inference.py).

The bug this guards against: that branch stacks the per-prompt states
returned by `_prefill_prompt_with_prefix_cache`, which are single-sequence
bsz=0 states (from `generate_zero_state(0)`) -- i.e. `state[0]` is
`[Layer, 2, C]` and `state[1]` is `[Layer, H, N, N]`, with NO batch
dimension. The stacking originally concatenated along the trailing feature
axis without inserting a batch dim, producing a malformed `[Layer, 2, B*C]`
state that `forward_seq_batch_tensor` rejected with
"RuntimeError: Tensors must have same number of dimensions", which surfaced
as a 500 on `/v1/chat/completions` and `/v2/chat/completions` whenever
`use_prefix_cache` was true (the default).

This test drives the real `_batch_prefill` with `_prefill_prompt_with_prefix_cache`
monkeypatched to emit controlled bsz=0 states, then asserts the stacked
result has the batched `generate_zero_state(B)` layout and that each batch
row maps back to the right prompt's state.

No model / GPU computation needed, but the import chain pulls in torch and
JIT-compiled CUDA kernels, so CUDA_HOME must be set (source env.sh first),
exactly like test_prefill_admission_queue.py.

Run with: source env.sh && uv run pytest test/test_batch_prefill_prefix_cache_stacking.py -v
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

pytest.importorskip("torch", reason="torch not installed (source env.sh first)")
import torch

try:
    from infer.inference import InferenceEngine
except OSError as exc:
    pytest.skip(f"CUDA environment not configured: {exc}", allow_module_level=True)


def _make_engine():
    # _batch_prefill's cache path only touches _prefill_prompt_with_prefix_cache
    # (monkeypatched below) and _raise_if_cancelled, so model/tokenizer/args are
    # never dereferenced and can stay None.
    return InferenceEngine(model=None, tokenizer=None, args=None, rocm_flag=False)


def _install_fake_prefill(monkeypatch, engine, state_factory, vocab):
    def fake_prefill(prompt, prefix_cache_manager=None, cancel_token=None):
        s = state_factory(prompt)
        o = torch.zeros(vocab)
        return ([], s, o, 0, None)

    monkeypatch.setattr(engine, "_prefill_prompt_with_prefix_cache", fake_prefill)


def test_batch_prefill_prefix_cache_path_produces_batched_state_shapes(monkeypatch):
    L, C, H, N, VOCAB, B = 2, 8, 2, 4, 100, 3
    engine = _make_engine()

    def state_factory(prompt):
        # Exact bsz=0 layout of generate_zero_state(0): no batch dimension.
        return [
            torch.zeros(L, 2, C),
            torch.zeros(L, H, N, N),
            torch.tensor(7, dtype=torch.int32),
        ]

    _install_fake_prefill(monkeypatch, engine, state_factory, VOCAB)

    state, out = engine._batch_prefill(
        ["a", "b", "c"], prefix_cache_manager=object(), cancel_token=None
    )

    # A regression to concatenating-without-a-batch-dim would yield
    # state[0].shape == (L, 2, B*C) instead of (L, 2, B, C), failing these.
    assert state[0].shape == (L, 2, B, C), (
        f"state[0] must be [Layer,2,B,C]; got {tuple(state[0].shape)}"
    )
    assert state[1].shape == (L, B, H, N, N), (
        f"state[1] must be [Layer,B,H,N,N]; got {tuple(state[1].shape)}"
    )
    assert state[2].shape == (B,), f"state[2] must be [B]; got {tuple(state[2].shape)}"
    assert out.shape == (B, VOCAB), f"out must be [B,vocab]; got {tuple(out.shape)}"


def test_batch_prefill_prefix_cache_stacking_preserves_per_row_content(monkeypatch):
    """Each batch row must correspond to the same-index prompt's state --
    catches an off-by-one or swapped stacking, not just a shape match."""
    L, C, H, N, VOCAB = 2, 4, 1, 2, 10
    engine = _make_engine()

    def state_factory(prompt):
        val = float(prompt)  # prompts below are "1", "2", "3"
        return [
            torch.full((L, 2, C), val),
            torch.full((L, H, N, N), val),
            torch.tensor(int(val), dtype=torch.int32),
        ]

    _install_fake_prefill(monkeypatch, engine, state_factory, VOCAB)

    state, _ = engine._batch_prefill(
        ["1", "2", "3"], prefix_cache_manager=object(), cancel_token=None
    )

    for i in range(3):
        expected = float(i + 1)
        assert torch.all(state[0][:, :, i, :] == expected), (
            f"batch row {i} of state[0] does not map to prompt {i}"
        )
        assert torch.all(state[1][:, i, :, :, :] == expected), (
            f"batch row {i} of state[1] does not map to prompt {i}"
        )
        assert state[2][i].item() == i + 1, (
            f"state[2][{i}] should be token count {i + 1}, got {state[2][i].item()}"
        )


def test_batch_prefill_prefix_cache_single_prompt_shape(monkeypatch):
    """bsz=1 is the degenerate case that still needs a real batch dim --
    the original crash reproduced with a single prompt, so guard it directly."""
    L, C, H, N, VOCAB = 2, 6, 2, 3, 50
    engine = _make_engine()

    def state_factory(prompt):
        return [
            torch.zeros(L, 2, C),
            torch.zeros(L, H, N, N),
            torch.tensor(0, dtype=torch.int32),
        ]

    _install_fake_prefill(monkeypatch, engine, state_factory, VOCAB)

    state, out = engine._batch_prefill(
        ["only"], prefix_cache_manager=object(), cancel_token=None
    )

    assert state[0].shape == (L, 2, 1, C)
    assert state[1].shape == (L, 1, H, N, N)
    assert state[2].shape == (1,)
    assert out.shape == (1, VOCAB)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
