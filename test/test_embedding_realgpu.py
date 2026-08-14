"""
Real-GPU equivalence test for Mod B (REQUIRES a real GPU + a real checkpoint).

The commit-file coverage in test_embedding_batch.py runs the CUDA kernels
through deterministic CPU stand-ins, so it proves embed_texts' *control flow*
(single vs batched equivalence) but NOT that the real ``RWKV_x070_*_seq`` /
``RWKV_x070_*_seq_batch`` kernels actually agree on-device. This test closes
that CR1 gap ("missing real-GPU equivalence test"): it loads the actual model
and asserts that embedding a set of real sentences one at a time
(``embed_texts([X])``) yields per-row vectors equal -- within a tight fp16
tolerance -- to embedding them together via the batched path
(``embed_texts([X,Y,Z])``).

Because it needs a real model AND a CUDA device it is skipped (never failed)
when either is absent, so a GPU-less CI still passes:

  * ``@pytest.mark.skipif(not torch.cuda.is_available(), ...)`` mirrors the
    real-GPU gating in test_state_pool_l1_l2_cache.py -- on a CPU-only host
    the test is skipped at collection, before any model load.
  * The model path comes from ``RWKV_EMBED_TEST_MODEL`` if set, else defaults
    to the 2.9b g1i checkpoint in models/; if the resolved path does not exist
    (steamlined CI clone without weights), the test is ``pytest.skip``ped.

Run with the repo env sourced on a GPU box:

    source env.sh && uv run pytest test/test_embedding_realgpu.py -v \
        RWKV_EMBED_TEST_MODEL=models/rwkv7-g1i-2.9b-20260805-ctx16384.pth

Tolerance: fp16 kernels are nondeterministic at the 1e-3 level across kernel
invocations (observed in T2 live on the same GPU), so asserting exact equality
would false-fail on fp16 jitter, not on a real regression. The point is to
catch a kernel/math regression in the shared-kernel batched path; ``1e-2``
max-abs-diff is an order of magnitude above the jitter floor while still far
below any real mismatch a broken kernel would produce.
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
    import infer.embedding as emb
except OSError as exc:
    pytest.skip(f"CUDA environment not configured: {exc}", allow_module_level=True)

# mirror test_state_pool_l1_l2_cache.py's real-GPU gating: skip (not fail) on a
# host without CUDA so a GPU-less CI still passes this module.
_PYTEST_GPU = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a real CUDA GPU"
)

# Default to the small 2.9b g1i checkpoint the repo hosts; allow an override so
# a box with different weights can still run it without editing the file.
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
            "real-GPU equivalence test)"
        )
    return _DEFAULT_MODEL


# Three short real sentences of heterogeneous length, so the batched call runs
# several chunked forward passes with a shrinking active set (not a trivially
# equal-length single-chunk row).
_SENTENCES = [
    "the quick brown fox jumps over the lazy dog",
    "rwkv is a recurrent neural network language model",
    "sentence embeddings must be stable across batched and single inference",
]

# A long German paragraph (~12x a repeated sentence, comfortably >= 300 tokens,
# so well over one prefill_chunk_size (256)-token chunk). Included in the
# batched call it forces at least one row to span TWO+ chunks while the short
# companions finish early and the active set shrinks -- the multi-chunk
# batched-vs-single path that all-short sentences can never exercise. The
# batched row must still match the single path's row within tolerance.
_LONG_TEXT = (
    "Dies ist ein langer deutscher Satz, der mehrfach wiederholt wird, um "
    "einen Textblock zu erzeugen, der deutlich länger als ein einzelnes "
    "Prefill-Chunk ist und daher die gestaffelte Batch-Prefill mit einem "
    "schrumpfenden aktiven Satz tatsächlich beansprucht. "
) * 12

# The full input to the batched call: the short sentences plus the long text.
_TEXTS = _SENTENCES + [_LONG_TEXT]


@_PYTEST_GPU
def test_real_gpu_batched_matches_per_text():
    """Load the actual model on the GPU and assert per-row embeddings are equal
    (< 1e-2 max abs diff) whether embedded one at a time (``embed_texts([X])``)
    or together via the batched path (``embed_texts([X,Y,Z])``)."""
    from model_load.model_loader import load_model_and_tokenizer

    model_path = _resolve_model_path()
    model, tokenizer, _args, _rocm = load_model_and_tokenizer(model_path)
    # device default is already "cuda" in embed_texts; pass it explicitly for
    # clarity so the single and batched calls provably run on the same device.
    device = "cuda"

    # Batched over the short sentences AND the long (>256-token) text, so the
    # long row spans at least two chunked passes with a shrinking active set.
    batched = emb.embed_texts(model, tokenizer, _TEXTS, device=device)

    for i, sent in enumerate(_TEXTS):
        single = emb.embed_texts(model, tokenizer, [sent], device=device)[0]
        a = torch.as_tensor(batched[i], dtype=torch.float32)
        b = torch.as_tensor(single, dtype=torch.float32)
        assert a.shape == b.shape
        diff = (a - b).abs().max().item()
        assert diff < 1e-2, (
            f"row {i} batched-vs-single max abs diff = {diff:.3e} >= 1e-2; "
            "batched and single paths disagree on the REAL kernels"
        )

    # Explicitly double-check the two multi-chunk-relevant rows: the long text
    # (required to span 2+ chunks) and the shortest companion (must match even
    # though it finishes in the first chunk while others continue).
    long_i = len(_TEXTS) - 1
    short_i = 0
    for row_i in (short_i, long_i):
        a = torch.as_tensor(batched[row_i], dtype=torch.float32)
        b = torch.as_tensor(
            emb.embed_texts(model, tokenizer, [_TEXTS[row_i]], device=device)[0],
            dtype=torch.float32,
        )
        diff = (a - b).abs().max().item()
        assert diff < 1e-2, (
            f"{'long' if row_i == long_i else 'short'} companion row "
            f"{row_i} batched-vs-single max abs diff = {diff:.3e} >= 1e-2"
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))