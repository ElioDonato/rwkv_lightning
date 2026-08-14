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
  (``RWKV_x070_TMix_seq`` / ``RWKV_x070_CMix_seq``) and reads ``model.z``
  weights directly. It does NOT modify or monkey-patch ``RWKV_x070``, so the
  existing chat / big_batch / FIM / responses paths are untouched.
* Stateless: each call allocates a fresh zero recurrent state (O(1) in
  sequence length, like all RWKV forward passes) and frees it on exit.
* ``model.z['emb.weight']`` is already layer-norm-normalized at load time
  (done in ``RWKV_x070.__init__``), matching what the chat path sees.
* Hidden dimension = ``model.n_embd`` (2560 for the 2.9b g1i checkpoint).
"""

import logging

logger = logging.getLogger("infer.embedding")

import torch
from torch.nn import functional as F

from infer.rwkv_batch.rwkv7 import (
    RWKV_x070_CMix_seq,
    RWKV_x070_TMix_seq,
)


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
    n_layer = model.n_layer
    n_head = model.n_head
    head_size = model.head_size

    results = []
    for text in texts:
        tokens = tokenizer.encode(text) if isinstance(text, str) else text
        if not tokens:
            logger.warning("embed_texts: empty tokenization for %r -> zero vector", text)
            results.append([0.0] * n_embd)
            continue

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

        # Final-token hidden state, then final layer norm (pre-head vector).
        h = x[-1]
        h = F.layer_norm(
            h, (n_embd,), weight=z["ln_out.weight"], bias=z["ln_out.bias"]
        )
        if normalize:
            h = F.normalize(h, dim=0)

        results.append(h.cpu().float().tolist())

        del x, v_first, state, tok
        torch.cuda.empty_cache()

    return results


def embedding_dim(model):
    return model.n_embd