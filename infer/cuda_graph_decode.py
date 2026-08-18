"""Replay ``model.forward_batch`` as a CUDA graph, per batch-size, opt-in default-off.

Phase 5 of IMPL_PLAN_GPU_THROUGHPUT.md: cut per-token host launch overhead on the
decode step. ``forward_batch`` (which drives the seq / ``*_seq_batch`` WKV kernels)
was made capturable back in 3b (the seq kernel now launches on the CURRENT CUDA
stream); this module wraps it so a decode loop can replay a captured graph instead
of re-launching the whole kernel set every token.

Contract / invariants
---------------------
* **DEFAULT OFF**: with ``RWKV_CUDA_GRAPH`` unset (``settings.cuda_graph_decode``
  False), ``CudaGraphForward.forward`` just calls ``model.forward_batch`` --
  byte-identical to the pre-feature path.
* **Per-size graph pool**: one graph per batch size in ``sizes`` (e.g. B in
  {1,2,4,8}); at a decode step with B live rows, replay the SMALLEST captured size
  >= B (grid is B*H, so padded rows still cost compute, hence the pool instead of a
  single max-B graph). If no captured size covers B we fall back to eager
  ``forward_batch``.
* **Static buffers + replay** (mirror test_cuda_graph_capture_realgpu.py): each
  captured size has a static ``tokens_in`` [B,1] long buffer, static ``state``
  buffers (list like the batch state -- ``[state0[L,2,B,C], state1[L,B,H,N,N],
  state2[B]]``, contiguous) sized to B, and a static ``out`` [B,vocab] fp32 buffer
  filled inside capture via ``g_out.copy_(o)``. Capture runs ``forward_batch``
  under ``torch.no_grad()``. Sampling and ``.tolist()`` stay OUTSIDE the graph
  (host sync inside a captured region is illegal).
* **State in/out contract**: ``forward_batch`` mutates ``state`` IN PLACE, so after
  a replay the static state buffers already hold the post-forward (next-step)
  state. Contract: on entry the caller copies its live-state DATA into the used
  size's static buffers, and on exit the caller's state is copied back from them --
  a couple of cheap memcpys per step vs the forward.
* **Safety**: everything under ``torch.no_grad()`` (never ``inference_mode``).
  Capture is lazy on first use, after a warm-up ``forward_batch``. If capture or a
  replay fails (no GPU, model not on cuda, allocator not warm), a flag is set and
  we permanently fall back to eager ``forward_batch`` with a warning -- never raise
  in the hot path.
"""
import logging

import torch

from settings import settings

logger = logging.getLogger("infer.cuda_graph_decode")

# Extra warm-up forward passes per batch size before capture: lets CUDNN/
# allocator reach steady state so the captured algorithm matches a steady-state
# eager forward (avoids cold-start algo wobble on the first few replays).
_WARMUP_PASSES = 4


def _cg_enabled(enabled=None) -> bool:
    """Resolve the opt-in, honoring an explicit override (used by tests)."""
    if enabled is not None:
        return bool(enabled)
    return bool(settings.cuda_graph_decode)


def _cg_max_bsz(max_bsz=None) -> int:
    """Max pooled batch size. Explicit override wins over settings; always int>=1."""
    if max_bsz is not None:
        m = int(max_bsz)
    else:
        m = max(
            int(settings.fuse_chat_max_bsz), int(settings.dynamic_batch_max_bsz)
        )
    return max(1, m)


class _SizePackage:
    """Static buffers + captured graph for ONE pooled batch size."""

    __slots__ = ("graph", "tokens_in", "state", "out")

    def __init__(self, graph, tokens_in, state, out):
        self.graph = graph          # torch.cuda.CUDAGraph
        self.tokens_in = tokens_in  # [size,1] long, cuda, static address
        self.state = state          # [state0[L,2,size,C], state1[L,size,H,N,N], state2[size]]
        self.out = out              # [size,vocab] fp32, cuda, static address


class CudaGraphForward:
    """Replay ``model.forward_batch`` as a CUDA graph, per batch-size, opt-in
    default-off."""

    # Batch dimension of state[0] / state[1] / state[2] (matches _admit's
    # torch.cat and _compact_active_rows' index_select dims).
    _BATCH_DIMS = (2, 1, 0)

    def __init__(self, model, max_bsz=8, sizes=(1, 2, 4, 8), enabled=None):
        self._model = model
        self._enabled = _cg_enabled(enabled)
        self._max_bsz = _cg_max_bsz(max_bsz)
        # Pool sizes are a power-of-two-ish ladder clamped to [1, max_bsz].
        self._sizes = sorted({int(s) for s in sizes if 1 <= s <= self._max_bsz})
        if not self._sizes:
            self._sizes = [1]
        self._pool = {}
        self._captured = False
        self._failed = False
        self._vocab = None

    # -- introspection -----------------------------------------------------

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def captured(self) -> bool:
        return self._captured

    @property
    def sizes(self):
        return list(self._sizes)

    # -- pure pool bookkeeping (no cuda, unit-testable) ---------------------

    def _match(self, bsz: int):
        """Smallest captured size >= ``bsz``, or None to eager-fallback."""
        for s in self._sizes:
            if s >= bsz:
                return s
        return None

    @classmethod
    def _grow(cls, base, size):
        """Grow a batch-1 state template to batch `size` (replicating the base
        row along each state element's batch dim, then making it contiguous)."""
        return [
            torch.cat([base[0]] * size, dim=2).contiguous(),
            torch.cat([base[1]] * size, dim=1).contiguous(),
            torch.cat([base[2]] * size, dim=0).contiguous(),
        ]

    @classmethod
    def _to_base(cls, state):
        """Reduce a live batch-state list to a batch-1 template (row 0 slices)."""
        return [
            state[0][:, :, :1].contiguous(),
            state[1][:, :1].contiguous(),
            state[2][:1].contiguous(),
        ]

    @classmethod
    def _slice(cls, t, i, bsz):
        """Slice ``t`` (state element ``i``) down to its first ``bsz`` batch rows,
        returning a view (shared storage, so writes land in the static buffer)."""
        d = cls._BATCH_DIMS[i]
        slc = [slice(None)] * t.dim()
        slc[d] = slice(0, bsz)
        return t[tuple(slc)]

    # -- lifecycle ----------------------------------------------------------

    def capture(self, base_state_templates=None) -> None:
        """Warm + capture one graph per pooled batch size.

        ``base_state_templates`` is a batch-state list (a live one is reduced to its
        batch-1 first row internally); it only needs to ASK for the right shapes, its
        DATA is a warm-up seed. Safe to call repeatedly; on any failure we set the
        permanent-fallback flag (never raise from the hot path)."""
        if not self._enabled or self._failed:
            return
        if self._captured:
            return
        device = "cuda"
        if not torch.cuda.is_available():
            self._fail("CUDA is not available")
            return
        if base_state_templates is None:
            return
        try:
            base = self._to_base(base_state_templates)
            # Warm minimal to discover vocab + warm allocator/cudnn.
            tok1 = torch.zeros([1, 1], device=device, dtype=torch.long)
            with torch.no_grad():
                w_state = [t.clone() for t in base]
                w_out = self._model.forward_batch(tok1, w_state)
            self._vocab = int(w_out.size(-1))

            for size in self._sizes:
                tok = torch.zeros([size, 1], device=device, dtype=torch.long)
                st = self._grow(base, size)
                # Warm up THIS size (lazy init / allocator steady state) before
                # capturing, exactly like the passing 3b real-GPU test. Multiple
                # in-place passes on the SAME static buffer warm CUDNN/allocator so
                # the captured algo matches a steady-state eager forward (avoids
                # cold-start algo wobble on the first few replays).
                with torch.no_grad():
                    for _ in range(_WARMUP_PASSES):
                        self._model.forward_batch(tok, st)
                torch.cuda.synchronize()
                g = torch.cuda.CUDAGraph()
                # fp32 out, matching the eager decode loop's `.float()`.
                g_out = torch.zeros(
                    [size, self._vocab], device=device, dtype=torch.float32
                )
                with torch.no_grad():
                    with torch.cuda.graph(g):
                        o = self._model.forward_batch(tok, st)
                        g_out.copy_(o)
                self._pool[size] = _SizePackage(g, tok, st, g_out)
            self._captured = True
            logger.info(
                "[CudaGraphForward] captured graph decode for batch sizes %s",
                sorted(self._pool),
            )
        except Exception as exc:
            self._fail(f"capture failed: {exc}")

    def _fail(self, reason: str):
        self._failed = True
        self._pool = {}
        logger.warning(
            "[CudaGraphForward] permanently falling back to eager forward_batch: %s",
            reason,
        )

    def shutdown(self):
        """Free the captured graphs / static buffers. Capture becomes lazy again
        on the next forward (safe, order-preserving for the caller)."""
        self._pool = {}
        self._captured = False
        self._failed = False
        self._vocab = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # -- hot path -----------------------------------------------------------

    def forward(self, tokens, state):
        """Replay the captured graph for size >= B (else eager ``forward_batch``).

        Mutates ``state`` in place exactly as ``model.forward_batch`` does (the
        static buffers own the per-size mutation; the caller's state is copied in
        on entry and back out on exit). Returns the outputs for rows [0:B] as fp32.
        Never raises: any failure permanently selects eager fallback.
        """
        if not self._enabled or self._failed:
            return self._model.forward_batch(tokens, state)
        # Graph replay is only meaningful for CUDA decode steps. On a CPU/host
        # path (e.g. hermetic tests with a CPU fake model) fall straight back to
        # the raw forward_batch -- byte-identical, and avoids capturing a fake.
        if not (tokens.is_cuda and torch.cuda.is_available()):
            return self._model.forward_batch(tokens, state)
        if not self._captured:
            # Lazy capture on first use (runs warm-up + capture; failure sets flag).
            self.capture(state)
            if not self._captured:
                return self._model.forward_batch(tokens, state)
        bsz = int(tokens.shape[0])
        size = self._match(bsz)
        if size is None:
            return self._model.forward_batch(tokens, state)
        pkg = self._pool[size]
        try:
            # (a) copy the caller's live-state DATA into the static buffer.
            for i in range(len(pkg.state)):
                self._slice(pkg.state[i], i, bsz).copy_(state[i])
            # (b) feed the sampled tokens through the graph's captured input buffer
            #     (values re-read on replay; the ADDRESS is fixed by capture).
            pkg.tokens_in[:bsz].copy_(tokens)
            # (c) replay; forward_batch mutated pkg.state in place during capture,
            #     so the static buffers now hold the NEXT-step state.
            pkg.graph.replay()
            # (d) copy the mutated state back to the caller's live tensors.
            for i in range(len(pkg.state)):
                state[i].copy_(self._slice(pkg.state[i], i, bsz))
            # (e) only the first `bsz` rows are meaningful (padded rows, if any,
            #     are garbage-but-unused).
            return pkg.out[:bsz]
        except Exception as exc:
            # Never raise in the hot path; permanently fall back with a warning.
            self._fail(f"replay failed: {exc}")
            return self._model.forward_batch(tokens, state)