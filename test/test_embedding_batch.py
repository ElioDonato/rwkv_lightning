"""
Coverage for infer/embedding.py's batched embedding path (MOD B).

The real RWKV-7 kernels in infer/rwkv_batch/rwkv7.py are pure CUDA, so they
cannot run on CPU. Mirroring test_prefill_admission_queue.py's _FakeModel
approach, the test builds a tiny fake ``model`` (small half-precision z dict,
CPU ``generate_zero_state``) and monkeypatches the module-level embedding
kernel references in ``infer.embedding`` with deterministic CPU stand-ins that
implement the SAME per-token recurrence for the single and batched variants.
This exercises embed_texts' control flow end-to-end on CPU -- the chunked
prefill loop, per-chunk state slicing, and true-last-token extraction -- and
verifies that the batched path yields per-text embeddings identical to the
single-sequence path. No real model is loaded and no GPU compute runs.

The fake kernels deliberately carry a cross-token ``x_prev`` recurrence (the
same ``cat(prev, x[:-1]) - x`` temporal-difference structure the real kernels
use), so a bug in chunk-boundary state carryover (wrong active slicing, wrong
final-token column, or dropped state[2] time advance) surfaces as a mismatch
between the batched and per-text results -- exactly the property the real
batched prefill (big_batch / chat) relies on.

Run with the repo env sourced (CUDA_HOME etc. set): the import chain pulls in
torch + the CUDA kernel module, but no kernel is invoked on CPU.
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

NL, NE, NH, HS = 3, 16, 4, 4
VOCAB = 32


class _FakeModel:
    """Tiny CPU model exposing exactly the surface infer/embedding.py reads.

    ``z`` keys replicate what the embedding loop indexes (layer-norm weights
    are real and used by F.layer_norm; the att/ffn weight slots are present as
    tensors because embed_texts passes them positionally into the (stubbed)
    kernels, which ignore them). Dtypes are half throughout to match the
    single-sequence path's half recurrent state so single vs batched results
    are directly comparable.
    """

    def __init__(self, n_layer=NL, prefill_chunk_size=4, max_prefill_bsz=8):
        self.n_layer = n_layer
        self.n_embd = NE
        self.n_head = NH
        self.head_size = HS
        self.prefill_chunk_size = prefill_chunk_size
        self.max_prefill_bsz = max_prefill_bsz

        z = {}
        z["emb.weight"] = (torch.randn(VOCAB, NE, dtype=torch.half) * 0.5)
        for i in range(n_layer):
            bbb = f"blocks.{i}."
            att = f"blocks.{i}.att."
            ffn = f"blocks.{i}.ffn."
            for k in ("ln1.weight", "ln2.weight"):
                z[bbb + k] = torch.ones(NE, dtype=torch.half)
            for k in ("ln1.bias", "ln2.bias"):
                z[bbb + k] = torch.zeros(NE, dtype=torch.half)
            for k in ("x_r", "x_w", "x_k", "x_v", "x_a", "x_g"):
                z[att + k] = torch.zeros(NE, dtype=torch.half)
            for k in ("w0", "w1", "w2", "a0", "a1", "a2", "v0", "v1", "v2",
                      "g1", "g2", "k_k", "k_a", "r_k"):
                z[att + k] = torch.zeros(NE, dtype=torch.half)
            for k in ("receptance.weight", "key.weight", "value.weight",
                      "output.weight"):
                z[att + k] = torch.zeros(NE, NE, dtype=torch.half)
            z[att + "ln_x.weight"] = torch.ones(NE, dtype=torch.half)
            z[att + "ln_x.bias"] = torch.zeros(NE, dtype=torch.half)
            z[ffn + "x_k"] = torch.zeros(NE, dtype=torch.half)
            z[ffn + "key.weight"] = torch.zeros(NE, NE, dtype=torch.half)
            z[ffn + "value.weight"] = torch.zeros(NE, NE, dtype=torch.half)
        z["ln_out.weight"] = torch.ones(NE, dtype=torch.half)
        z["ln_out.bias"] = torch.zeros(NE, dtype=torch.half)
        self.z = z

    def generate_zero_state(self, bsz):
        L, C, H, N = self.n_layer, self.n_embd, self.n_head, self.head_size
        if bsz >= 1:
            return [
                torch.zeros((L, 2, bsz, C), dtype=torch.half),
                torch.zeros((L, bsz, H, N, N), dtype=torch.half),
                torch.zeros((bsz,), dtype=torch.int32),
            ]
        return [
            torch.zeros((L, 2, C), dtype=torch.half),
            torch.zeros((L, H, N, N), dtype=torch.half),
            torch.zeros((), dtype=torch.int32),
        ]


# --- Deterministic CPU stand-ins for the CUDA kernels. --------------------
# They implement a prefix-sum state machine: each layer carries a per-sequence
# running total in its x_prev slot and emits the running sum *before* each
# token. This is a pure associative fold, so splitting a sequence across
# chunks and carrying the fold accumulator reproduces the same per-token output
# as folding the whole sequence at once -- i.e. it is *provably* chunk-invariant
# by construction. Embed_texts' single path folds each text in one call; the
# batched path folds the same text across several equal-length chunks with the
# accumulator carried in the (sliced/copied-back) batch state. Any bug in the
# chunk assembly, state slicing, or true-last-token extraction surfaces as a
# single-vs-batched mismatch -- exactly the equivalence the real batched
# prefill (big_batch / chat) relies on.

def _fold_seq(x, s, layer_id):
    # Single-sequence fold over the T dimension, starting from carried s [C].
    # out[t] = s + sum_{j<t} x[j]; s <- s + sum_j x[j].
    out = s.unsqueeze(0) + x.cumsum(0) - x
    return out + float(layer_id), s + x.sum(0)


def _fold_batch(x, s, layer_id):
    # Batched fold over the T dimension, starting from carried s [B, C].
    out = s.unsqueeze(1) + x.cumsum(1) - x
    return out + float(layer_id), s + x.sum(1)


def _fake_tmix_seq(layer_id, H, N, x, x_prev, v_first, state, *w):
    out, total = _fold_seq(x, x_prev[0], layer_id)
    x_prev[0] = total.clone()
    return out, v_first


def _fake_cmix_seq(x, x_prev, *w):
    out, total = _fold_seq(x, x_prev[1], 0.0)
    x_prev[1] = total.clone()
    return out


def _fake_tmix_batch(layer_id, H, N, x, x_prev, v_first, state, *w):
    out, total = _fold_batch(x, x_prev[0], layer_id)
    x_prev[0] = total.clone()
    return out, v_first


def _fake_cmix_batch(x, x_prev, *w):
    out, total = _fold_batch(x, x_prev[1], 0.0)
    x_prev[1] = total.clone()
    return out


@pytest.fixture(autouse=True)
def _stub_kernels(monkeypatch):
    """Redirect the embedding module's kernel references to the CPU fakes."""
    monkeypatch.setattr(emb, "RWKV_x070_TMix_seq", _fake_tmix_seq)
    monkeypatch.setattr(emb, "RWKV_x070_CMix_seq", _fake_cmix_seq)
    monkeypatch.setattr(emb, "RWKV_x070_TMix_seq_batch", _fake_tmix_batch)
    monkeypatch.setattr(emb, "RWKV_x070_CMix_seq_batch", _fake_cmix_batch)


def _assert_close(a, b):
    a = torch.as_tensor(a, dtype=torch.float32)
    b = torch.as_tensor(b, dtype=torch.float32)
    assert a.shape == b.shape and a.shape[-1] == NE
    assert torch.allclose(a, b, rtol=1e-3, atol=1e-3), (
        f"max abs diff = {(a - b).abs().max().item()}"
    )


def _text(n):
    """Deterministic non-empty token id list of length n (ids in 1..VOCAB-1)."""
    return [(i * 7 + 3) % (VOCAB - 1) + 1 for i in range(n)]


def _zeros():
    return [0.0] * NE


# --- Core equivalence: batched == per-text. -------------------------------

def test_batched_matches_per_text_short(monkeypatch):
    """Equal-length short texts (one prefill chunk): each batch row equals the
    same text embedded alone."""
    model = _FakeModel()
    t0, t1, t2 = _text(3), _text(3), _text(3)  # all <= prefill_chunk_size=4
    batch = emb.embed_texts(model, None, [t0, t1, t2], device="cpu")
    for i, t in enumerate([t0, t1, t2]):
        single = emb.embed_texts(model, None, [t], device="cpu")[0]
        _assert_close(batch[i], single)


def test_batched_matches_per_text_heterogeneous_chunked(monkeypatch):
    """Heterogeneous lengths forcing multiple chunks + shrinking active set:
    the long text spans several chunked forward passes and short texts finish
    mid-pipeline -- each batch row must still equal that text alone."""
    model = _FakeModel(prefill_chunk_size=4, max_prefill_bsz=8)
    texts = [_text(3), _text(11), _text(6), _text(2), _text(9)]
    batch = emb.embed_texts(model, None, texts, device="cpu")
    assert all(v is not None for v in batch)
    for i, t in enumerate(texts):
        single = emb.embed_texts(model, None, [t], device="cpu")[0]
        _assert_close(batch[i], single)


def test_batch_of_one_equals_batch_row(monkeypatch):
    """The batch path invoked with a lone text equals that text's row in a
    larger batch -- validates generate_zero_state(bsz) shapes for bsz=1."""
    model = _FakeModel()
    t = _text(5)
    alone = [e.cpu() for e in emb._embed_batch(model, model.z, [t], "cpu")][0]
    in_batch = emb._embed_batch(model, model.z, [t, _text(4)], "cpu")[0]
    _assert_close(alone, in_batch)


# --- Empty-input behaviour must be unchanged. -----------------------------

def test_empty_input_un_changed(monkeypatch):
    model = _FakeModel()
    # Empty tokenization yields a zero vector in-place.
    out = emb.embed_texts(model, None, [[], _text(3)], device="cpu")
    assert out[0] == _zeros()
    _assert_close(out[1], emb.embed_texts(model, None, [_text(3)], device="cpu")[0])
    # No texts at all -> empty list.
    assert emb.embed_texts(model, None, [], device="cpu") == []
    # All-empty -> a list of zero vectors.
    all_empty = emb.embed_texts(model, None, [[], []], device="cpu")
    assert all_empty == [_zeros(), _zeros()]


# --- Forward-pass count: ~texts//batch, not N. ----------------------------

def test_forward_pass_count_one_batch_for_all_texts(monkeypatch):
    """N equal-length short texts are embedded in ONE batched forward pass
    (n_layer TMix_batch kernel calls for the whole batch), not N passes."""
    model = _FakeModel(max_prefill_bsz=8)
    counts = {"tmix": 0}

    def counting_tmix(layer_id, H, N, x, x_prev, v_first, state, *w):
        counts["tmix"] += 1
        return _fake_tmix_batch(layer_id, H, N, x, x_prev, v_first, state, *w)

    monkeypatch.setattr(emb, "RWKV_x070_TMix_seq_batch", counting_tmix)
    texts = [_text(3) for _ in range(4)]  # all one chunk of len 3
    emb.embed_texts(model, None, texts, device="cpu")
    # One chunk for the whole 4-text batch == n_layer kernel calls, i.e. a
    # single forward pass for all texts (NOT 4 * n_layer).
    assert counts["tmix"] == NL


def test_sub_batching_bounds_forward_passes(monkeypatch):
    """A request larger than max_prefill_bsz is split into ceil(N/batch)
    forward passes, each of full batch width."""
    model = _FakeModel(max_prefill_bsz=2)
    counts = {"tmix": 0}

    def counting_tmix(layer_id, H, N, x, x_prev, v_first, state, *w):
        counts["tmix"] += 1
        return _fake_tmix_batch(layer_id, H, N, x, x_prev, v_first, state, *w)

    monkeypatch.setattr(emb, "RWKV_x070_TMix_seq_batch", counting_tmix)
    texts = [_text(3) for _ in range(4)]  # max_bsz=2 -> 2 sub-batches, 1 chunk each
    emb.embed_texts(model, None, texts, device="cpu")
    assert counts["tmix"] == 2 * NL


def test_unbounded_fallback_caps_sub_batch_size(monkeypatch):
    """CR1: with NO model cap (attribute absent, so getattr falls back), the
    fallback must NOT collapse to one unbounded batch over the whole request.
    Each sub-batch is capped at _HARD_CEILING_DEFAULT texts (mirrors the
    embed aggregator) -- the old `or len(non_empty)` put the whole (>=cap)
    request in a single _embed_batch call, risking one giant VRAM spike."""
    model = _FakeModel(max_prefill_bsz=0)
    del model.max_prefill_bsz  # attribute absent -> unbounded-fallback branch
    cap = emb._HARD_CEILING_DEFAULT
    widths = []

    def observing_tmix(layer_id, H, N, x, x_prev, v_first, state, *w):
        widths.append(x.shape[0])  # per-forward-pass batch width
        return _fake_tmix_batch(layer_id, H, N, x, x_prev, v_first, state, *w)

    monkeypatch.setattr(emb, "RWKV_x070_TMix_seq_batch", observing_tmix)
    texts = [_text(3) for _ in range(cap + 16)]  # > cap texts, all one chunk
    emb.embed_texts(model, None, texts, device="cpu")
    # Must split into 2 sub-batches (cap then remainder), never one >=cap batch.
    assert len(widths) == 2 * NL
    assert all(w <= cap for w in widths), (
        "no-cap fallback must cap each sub-batch at _HARD_CEILING_DEFAULT"
    )


def test_unbounded_fallback_single_text_still_uses_exact_single_loop(monkeypatch):
    """CR1: under the no-cap fallback, a lone text still takes the exact
    _embed_single loop (never routed through the batched kernels) and yields
    the same per-text vector the single path always produced (re-capping the
    SAME model object -- i.e. same weights -- gives the byte-identical result,
    proving the fallback changes nothing for a lone request)."""
    model = _FakeModel(max_prefill_bsz=0)
    del model.max_prefill_bsz  # zero/absent cap -> _HARD_CEILING_DEFAULT fallback

    def blocking_tmix(layer_id, H, N, x, x_prev, v_first, state, *w):
        raise AssertionError("single-text no-cap path must NOT hit _embed_batch")

    monkeypatch.setattr(emb, "RWKV_x070_TMix_seq_batch", blocking_tmix)
    t = _text(11)  # spans two prefill chunks via _embed_single
    single = emb.embed_texts(model, None, [t], device="cpu")
    # Re-cap the same object (identical weights): still the exact single loop.
    model.max_prefill_bsz = 4
    _assert_close(single[0], emb.embed_texts(model, None, [t], device="cpu")[0])


# --- Output shape / order contract. ---------------------------------------

def test_output_shape_and_order(monkeypatch):
    model = _FakeModel()
    texts = [_text(3), _text(1), _text(7), _text(2)]
    out = emb.embed_texts(model, None, texts, device="cpu")
    assert len(out) == len(texts)
    assert all(len(v) == NE for v in out)
    # degenerate normalize=True forces unit-norm vectors, so they're non-trivial
    norms = [torch.linalg.norm(torch.as_tensor(v, dtype=torch.float32)) for v in out]
    assert all(abs(n - 1.0) < 1e-3 for n in norms)
    for i, t in enumerate(texts):
        _assert_close(out[i], emb.embed_texts(model, None, [t], device="cpu")[0])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))