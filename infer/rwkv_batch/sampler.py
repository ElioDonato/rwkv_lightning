import torch, os
from torch.utils.cpp_extension import load
current_path = os.path.dirname(os.path.abspath(__file__))

ROCm_flag = torch.version.hip is not None
if ROCm_flag:
    sample = load(
        name="sample",
        sources = [f"{current_path}/hip/sampling_op.hip",f"{current_path}/hip/sampling.hip"],
        extra_cuda_cflags=['-fopenmp', '-ffast-math', '-O3', '-munsafe-fp-atomics'],
        verbose=True,
    )
else:
    sample = load(
        name="sample",
        sources = [f"{current_path}/cuda/sampling.cpp",f"{current_path}/cuda/sampling.cu"],
        extra_cuda_cflags=["-O3", "-res-usage", "--extra-device-vectorization", "-Xptxas -O3"],
        verbose=True,
    )

def sample_batch_per_row(
    logits,
    penalties,
    sample_rand_states,
    temperature,
    top_k,
    top_p,
    alpha_presence,
    alpha_frequency,
    alpha_decay,
):
    """Sample ONE token per row of a multi-row batched decode, each row with its
    OWN sampler controls.

    This is the per-row building block behind the shared multi-row decode
    (big_batch_stream) honoring each row's own temperature / top_k / top_p /
    repetition-penalty scalars instead of one batch-wide gumbel-temperature
    sample. It applies the existing ``batch_sampling_repetition_temperature_
    topk_topp`` op ONCE PER ROW, slicing that row's logits / penalties /
    rand-state and passing that row's own scalars -- because the underlying op
    is already per-row on ``penalties`` and ``states`` (only temperature /
    top_k / top_p were batch-wide scalars), row-by-row application is
    numerically equal to a single whole-batch call when every row shares the
    same scalars.

    Parameters
    ----------
    logits : torch.Tensor[B, vocab], float32, CUDA
        Last-step (post-prefill / post-forward) logits for the B active rows.
    penalties : torch.Tensor[B, vocab], float32, CUDA
        Per-row repetition-penalty state. MUTATED IN PLACE, row by row, exactly
        like _V1BatchSampler's handling -- it is the caller's persisted
        per-row penalty state, carried across decode steps and compacted
        alongside the batch (so a row's penalty state follows that row).
    sample_rand_states : torch.Tensor, CUDA
        Per-row RNG states, as produced by ``sample.setup_rand(seed, B)`` (a
        flat byte tensor of B contiguous row-blocks). Row ``b`` is the ``b``-th
        contiguous block; it may also already be shaped ``[B, block]``.
    temperature / top_k / top_p / alpha_presence / alpha_frequency / alpha_decay :
        Sequence[B] (list, tuple, or 1-D tensor) of PER-ROW sampler scalars,
        slot-indexed parallel to the logits rows. Row ``b`` is sampled with
        ``temperature[b]``, ``top_k[b]``, etc.

    Returns
    -------
    torch.Tensor[B, 1] long -- one sampled token id per row, in row order
    (ready to feed straight into ``forward_batch``, the same fast path the
    batch-wide gumbel sampler fed).

    MUST be called from a CUDA-guarded / decode-step context (e.g. inside the
    _gpu_decode_step offload closure routed through ``_offload_gpu`` / under
    ``cuda_guard``), never from pure event-loop bookkeeping: each per-row call
    launches a CUDA sampling kernel and the returned tensor must reach
    ``forward_batch`` on the same thread. Runs under ``torch.no_grad()``
    (NOT ``inference_mode()`` -- the latter is thread-local and unsafe across
    the _offload_gpu async/thread boundary) because the sample op must not
    build an autograd graph.
    """
    B = logits.size(0)
    device = logits.device
    out = torch.empty((B, 1), dtype=torch.long, device=device)
    if B == 0:
        return out
    # Resolve the op once per call so a caller-side monkeypatch of
    # sample.batch_sampling_repetition_temperature_topk_topp (the idle used by
    # the hermetic tests) takes effect for the whole loop.
    op = sample.batch_sampling_repetition_temperature_topk_topp
    # rand_states is a flat byte tensor of B * rowsize bytes (one contiguous
    # RNG struct per row, as setup_rand produces); view as [B, rowsize] so
    # row `b`'s slice is that row's whole contiguous block -- the same byte
    # range the whole-batch op reads for row `b` -- never a scalar 1-byte slice.
    row_states = sample_rand_states.reshape(B, -1)
    with torch.no_grad():
        for b in range(B):
            # The CUDA op requires temperature in [0.001, 1000]; a client's
            # temperature=0 (pure greedy) would otherwise raise and crash the
            # decode scheduler. Clamp so 0 -> near-greedy instead.
            temp = min(1000.0, max(0.001, float(temperature[b])))
            row = op(
                logits[b : b + 1],
                penalties[b : b + 1],
                row_states[b],
                float(alpha_presence[b]),
                float(alpha_frequency[b]),
                float(alpha_decay[b]),
                temp,
                int(top_k[b]),
                float(top_p[b]),
            )
            out[b, 0] = row[0]
    return out


if __name__ == "__main__":
    batch_size = 128
    vocab_size = 131072
    temperature = 1.0
    top_p = 0.5
    top_k = -1
    presence_penalty = 1.0
    repetition_penalty = 0.1
    penalty_decay = 0.996
    states = sample.setup_rand(0, batch_size)
    logits = torch.rand(batch_size, vocab_size).to(0)
    penalties = torch.zeros(batch_size, vocab_size).to(0)
    print(logits)
    print(logits.shape)
    # samples = sample.batch_sampling_temperature_topk_topp(logits, states, temperature, top_k, top_p)
    samples = sample.batch_sampling_repetition_temperature_topk_topp(logits, penalties, states,
                                                                     presence_penalty, repetition_penalty, penalty_decay,
                                                                     temperature, top_k, top_p)
    print(samples)
