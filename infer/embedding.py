"""
Thread-safe RWKV-7 embedding extraction.

Reuses the exact-weights, exact-kernels forward loop of rwkv7.py but STOPS
BEFORE the output head. The embedding returned for each input text is the
final-token hidden state after the last block, run through the `ln_out` layer
norm (the same vector the head would otherwise consume to predict the next
token) -- the standard RWKV sentence-embedding representation.

Design notes
------------
* Pure add-on: it imports the module-level scripted kernels
  (``RWKV_x070_TMix_seq`` / ``RWKV_x070_CMix_seq`` and their batched
  ``*_seq_batch`` counterparts) and reads ``model.z`` weights directly. It
  does NOT modify or monkey-patch ``RWKV_x070``, so the existing chat /
  big_batch / FIM / responses paths are untouched.
* Stateless: each call allocates a fresh zero recurrent state (O(1) in
  sequence length, like all RWKV forward passes) and frees it on exit. A
  single ``torch.cuda.empty_cache()`` is issued once per batched request
  (not per text) -- per-text cache thrash was killing GPU occupancy.
* Hidden dimension = ``model.n_embd`` (2560 for the 2.9b g1i checkpoint).

Batching
--------
Multiple texts are embedded in as few batched forward passes as possible by
running them through the exact per-sequence batched prefill kernels big_batch
and the chat prefill already use (``RWKV_x070_TMix_seq_batch`` /
``RWKV_x070_CMix_seq_batch`` over a ``generate_zero_state(bsz)`` state), in
equal-length chunks that mirror ``_forward_batch_prompts_chunked``. Every
column of every chunk is a real token (all active rows advance exactly
``step`` each pass), so there is zero padding waste; the chunk step is bounded
by ``prefill_chunk_size`` (256) and per-call batch size by
``max_prefill_bsz`` so a co-resident :8081 chat on the same 24GB GPU is not
starved of VRAM. A sequence's embedding is captured the moment the chunk that
consumes its *remaining* tokens runs -- because every active row in a chunk
has equal length, that is always its true last token, never a padded slot.

Numerical equivalence: this is a throughput-only change. A single non-empty
text keeps the exact pre-MOD single-sequence loop (``_embed_single``), so the
lone-text path is byte-identical to the previous release. The multi-text
batched path uses the same kernels/dimensions as big_batch's prefill, so each
row is numerically identical to embedding that text alone: byte-identical for
texts that fit in one prefill chunk, and matching the batched-vs-single
equivalence the production chunked prefill already relies on for longer texts.
"""

import logging

logger = logging.getLogger("infer.embedding")

import torch
from torch.nn import functional as F

from infer.rwkv_batch.rwkv7 import (
    RWKV_x070_CMix_seq,
    RWKV_x070_CMix_seq_batch,
    RWKV_x070_TMix_seq,
    RWKV_x070_TMix_seq_batch,
)

# Hard ceiling (texts) applied to a sub-batch when the model carries no
# max_prefill_bsz cap (attribute absent or <= 0). Mirrors
# infer.embed_aggregator's _HARD_CEILING_DEFAULT so the two never diverge; it is
# duplicated (not imported) because embed_aggregator already imports
# embed_texts from this module, so importing back would be a circular import.
# Without a cap, "no cap / 0 cap" must never mean "whole request in one batch"
# (a lone oversized request could otherwise spike VRAM on a GPU co-resident
# with live :8081 chat) -- cap each sub-batch here instead.
_HARD_CEILING_DEFAULT = 64


def _embed_single(model, z, tokens, device):
    """Return the pre-``ln_out`` final-token hidden state ``[n_embd]`` for one
    non-empty token list.

    This is the exact pre-MOD single-sequence loop, kept verbatim so a lone
    text request is byte-identical to the previous release (only the
    multi-text batched path is new).
    """
    n_embd = model.n_embd
    n_layer = model.n_layer
    n_head = model.n_head
    head_size = model.head_size

    # Zero recurrent state, single sequence (match generate_zero_state(0)).
    state = [
        torch.zeros(
            (n_layer, 2, n_embd), device=device, dtype=torch.half, requires_grad=False
        ),
        torch.zeros(
            (n_layer, n_embd // head_size, head_size, head_size),
            device=device,
            dtype=torch.half,
            requires_grad=False,
        ),
        torch.zeros((), dtype=torch.int32, device=device),
    ]

    tok = torch.as_tensor(tokens, device=device, dtype=torch.long)
    x = z["emb.weight"][tok]  # [T, C]
    v_first = torch.empty_like(x)

    for i in range(n_layer):
        bbb = f"blocks.{i}."
        att = f"blocks.{i}.att."
        ffn = f"blocks.{i}.ffn."

        xx = F.layer_norm(
            x, (n_embd,), weight=z[bbb + "ln1.weight"], bias=z[bbb + "ln1.bias"]
        )
        xx, v_first = RWKV_x070_TMix_seq(
            i, n_head, head_size, xx, state[0][i], v_first, state[1][i],
            z[att + "x_r"], z[att + "x_w"], z[att + "x_k"], z[att + "x_v"],
            z[att + "x_a"], z[att + "x_g"], z[att + "w0"], z[att + "w1"],
            z[att + "w2"], z[att + "a0"], z[att + "a1"], z[att + "a2"],
            z[att + "v0"], z[att + "v1"], z[att + "v2"], z[att + "g1"],
            z[att + "g2"], z[att + "k_k"], z[att + "k_a"], z[att + "r_k"],
            z[att + "receptance.weight"], z[att + "key.weight"],
            z[att + "value.weight"], z[att + "output.weight"],
            z[att + "ln_x.weight"], z[att + "ln_x.bias"], state[2],
        )
        x = x + xx

        xx = F.layer_norm(
            x, (n_embd,), weight=z[bbb + "ln2.weight"], bias=z[bbb + "ln2.bias"]
        )
        xx = RWKV_x070_CMix_seq(
            xx, state[0][i], z[ffn + "x_k"], z[ffn + "key.weight"],
            z[ffn + "value.weight"],
        )
        x = x + xx

    h = x[-1]
    del x, v_first, state, tok
    return h


def _embed_batch(model, z, token_lists, device):
    """Return a list of pre-``ln_out`` final-token hidden states ``[n_embd]``,
    one per ``token_lists`` entry (all non-empty), in input order.

    Processes the whole subset in one batched chunked prefill: an
    equal-length forward for every still-active row at the current chunk step,
    mirroring ``_forward_batch_prompts_chunked``. A row hands its embedding in
    the chunk that consumes its remaining tokens (its true last token).
    """
    n_embd = model.n_embd
    n_layer = model.n_layer
    n_head = model.n_head
    head_size = model.head_size

    bsz = len(token_lists)
    lengths = [len(t) for t in token_lists]
    pos = [0] * bsz
    out = [None] * bsz
    state = model.generate_zero_state(bsz)
    chunk_size = int(getattr(model, "prefill_chunk_size", 256))

    while True:
        active = [i for i in range(bsz) if pos[i] < lengths[i]]
        if not active:
            break
        step = min(chunk_size, min(lengths[i] - pos[i] for i in active))

        # Slice the batch state down to the still-active rows (copy, like
        # forward_batch's `batch_state`); the kernels update it in place and
        # it is copied back below, exactly as _forward_batch_prompts_chunked
        # does for the shared state across hetero-length chunks.
        if len(active) == bsz:
            batch_state = state
        else:
            batch_state = [
                state[0][:, :, active],
                state[1][:, active],
                state[2][active],
            ]

        chunk = [token_lists[i][pos[i]:pos[i] + step] for i in active]
        x = z["emb.weight"][torch.as_tensor(chunk, device=device, dtype=torch.long)]
        v_first = torch.empty_like(x)

        for i in range(n_layer):
            bbb = f"blocks.{i}."
            att = f"blocks.{i}.att."
            ffn = f"blocks.{i}.ffn."

            xx = F.layer_norm(
                x, (n_embd,), weight=z[bbb + "ln1.weight"], bias=z[bbb + "ln1.bias"]
            )
            xx, v_first = RWKV_x070_TMix_seq_batch(
                i, n_head, head_size, xx, batch_state[0][i], v_first,
                batch_state[1][i], z[att + "x_r"], z[att + "x_w"], z[att + "x_k"],
                z[att + "x_v"], z[att + "x_a"], z[att + "x_g"], z[att + "w0"],
                z[att + "w1"], z[att + "w2"], z[att + "a0"], z[att + "a1"],
                z[att + "a2"], z[att + "v0"], z[att + "v1"], z[att + "v2"],
                z[att + "g1"], z[att + "g2"], z[att + "k_k"], z[att + "k_a"],
                z[att + "r_k"], z[att + "receptance.weight"], z[att + "key.weight"],
                z[att + "value.weight"], z[att + "output.weight"],
                z[att + "ln_x.weight"], z[att + "ln_x.bias"], batch_state[2],
            )
            x = x + xx

            xx = F.layer_norm(
                x, (n_embd,), weight=z[bbb + "ln2.weight"], bias=z[bbb + "ln2.bias"]
            )
            xx = RWKV_x070_CMix_seq_batch(
                xx, batch_state[0][i], z[ffn + "x_k"], z[ffn + "key.weight"],
                z[ffn + "value.weight"],
            )
            x = x + xx

        # Pre-ln_out final-token hidden state per active row. Every active row
        # consumed exactly `step` tokens, so rows finishing here end on their
        # true last token (column step-1), not on padded slots.
        h = x[:, -1, :]
        for k, i in enumerate(active):
            pos[i] += step
            if pos[i] >= lengths[i]:
                out[i] = h[k]

        # Mirror forward_seq_batch's post-loop `state[2] += len(idxs[0])`: the
        # per-token RWKV time decay advances by exactly one chunk step per row.
        batch_state[2][:] += step

        if len(active) != bsz:
            for k, i in enumerate(active):
                state[0][:, :, i] = batch_state[0][:, :, k]
                state[1][:, i] = batch_state[1][:, k]
                state[2][i] = batch_state[2][k]

        del x, v_first

    del state
    return out


@torch.no_grad()
def embed_texts(model, tokenizer, texts, normalize=True, device="cuda"):
    """Return a list of ``[n_embd]`` float vectors, one per input text.

    ``prompt``-style instructions are intentionally NOT prepended: this raw
    RWKV checkpoint is not instruction-tuned for retrieval, so the embedding is
    taken straight from the (layer-normed) final-token hidden state. Callers
    that previously sent an E5-style instruction can keep sending it; it is
    simply ignored for the projection itself.
    """
    z = model.z
    n_embd = model.n_embd

    results = [None] * len(texts)
    non_empty = []
    for i, text in enumerate(texts):
        tokens = tokenizer.encode(text) if isinstance(text, str) else text
        if not tokens:
            logger.warning("embed_texts: empty tokenization for %r -> zero vector", text)
            results[i] = [0.0] * n_embd
            continue
        non_empty.append((i, tokens))

    if non_empty:
        # Multi-text is batched in sub-batches of at most `max_bsz`, so a single
        # huge request can't blow VRAM on a GPU shared with live :8081 chat. The
        # cap is the model's throttled max-prefill-batch estimate, which already
        # accounts for currently free VRAM; a lone text (or the trailing
        # remainder sub-batch of size 1) instead takes the exact pre-MOD
        # single-sequence loop for a byte-identical result. Divide the request
        # into sub-batches first so every unit runs through the same finalize.
        max_bsz = max(
            1, int(getattr(model, "max_prefill_bsz", 0) or _HARD_CEILING_DEFAULT)
        )
        sub_batches = [non_empty[i:i + max_bsz] for i in range(0, len(non_empty), max_bsz)]

        for sub in sub_batches:
            if len(sub) == 1:
                i, tokens = sub[0]
                h = _embed_single(model, z, tokens, device)
                h = F.layer_norm(
                    h, (n_embd,), weight=z["ln_out.weight"], bias=z["ln_out.bias"]
                )
                if normalize:
                    h = F.normalize(h, dim=0)
                results[i] = h.cpu().float().tolist()
                continue

            hs = _embed_batch(model, z, [t for _, t in sub], device)
            for (i, _), h in zip(sub, hs):
                h = F.layer_norm(
                    h, (n_embd,), weight=z["ln_out.weight"], bias=z["ln_out.bias"]
                )
                if normalize:
                    h = F.normalize(h, dim=0)
                results[i] = h.cpu().float().tolist()
            del hs

        # One empty_cache for the whole request (not per text).
        torch.cuda.empty_cache()

    return results


def embedding_dim(model):
    return model.n_embd