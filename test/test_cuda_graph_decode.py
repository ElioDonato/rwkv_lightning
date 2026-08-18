"""Hermetic CPU coverage + real-GPU replay-vs-eager check for the opt-in CUDA-graph
decode forward (Phase 5, infer/cuda_graph_decode.CudaGraphForward, RWKV_CUDA_GRAPH).

CPU part (no cuda): pool/ladder math, disabled-no-use byte-identical fallthrough,
and the no-GPU permanent-eager-fallback. GPU part (skip if no CUDA / no checkpoint):
capture sizes {1,2}, run several decode steps through the graph forward and assert
each step matches an eager forward_batch on the SAME inputs within fp16 tolerance
and with identical argmax, plus the caller-state round-trip.

Mirrors test_cuda_graph_capture_realgpu.py's skip idiom (the same seq-stream fix
that makes forward_batch capturable is the precondition here).
"""
import os
import sys
from pathlib import Path

import pytest

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

pytest.importorskip("torch", reason="torch not installed (source env.sh first)")

from settings import settings

try:
    from infer.cuda_graph_decode import CudaGraphForward
except OSError as exc:
    pytest.skip(f"CUDA environment not configured: {exc}", allow_module_level=True)


VOCAB = 64


class _FakeModel:
    """CPU model stand-in; forward_batch maps token t -> one-hot(t+1) so a greedy
    decode walks a deterministic counter trajectory (same idiom as the dynamic
    batch hermetic tests)."""

    def __init__(self):
        self.vocab = VOCAB
        self.forward_calls = 0

    def generate_zero_state(self, bsz):
        return [
            torch.zeros(1, 2, bsz),
            torch.zeros(1, bsz, 1, 1, 1),
            torch.zeros(bsz, dtype=torch.long),
        ]

    def forward_batch(self, tokens, state, full_output=False):
        self.forward_calls += 1
        t = (tokens.reshape(-1) + 1) % self.vocab
        return torch.nn.functional.one_hot(t.to(torch.long), self.vocab).float()


# ---------------------------------------------------------------------------
# CPU: pool / ladder bookkeeping + default-off + no-GPU fallback
# ---------------------------------------------------------------------------

def test_pool_ladder_selection():
    fwd = CudaGraphForward(_FakeModel(), enabled=True, max_bsz=8, sizes=(1, 2, 4, 8))
    assert fwd.sizes == [1, 2, 4, 8]
    assert fwd._match(1) == 1
    assert fwd._match(2) == 2
    assert fwd._match(3) == 4
    assert fwd._match(4) == 4
    assert fwd._match(5) == 8
    assert fwd._match(8) == 8
    # B above the largest pooled size -> None (eager fallback).
    assert fwd._match(9) is None


def test_pool_sizes_capped_by_max_bsz():
    fwd = CudaGraphForward(_FakeModel(), enabled=True, max_bsz=4, sizes=(1, 2, 4, 8))
    assert fwd.sizes == [1, 2, 4]
    assert fwd._match(5) is None
    fwd2 = CudaGraphForward(_FakeModel(), enabled=True, max_bsz=0, sizes=(4, 8))
    assert fwd2.sizes == [1], "empty ladder degrades to a lone size-1 graph"


def test_disabled_forward_is_byte_identical():
    model = _FakeModel()
    fwd = CudaGraphForward(model, enabled=False, max_bsz=8)
    assert fwd.enabled is False
    toks = torch.tensor([[3]], dtype=torch.long)
    state = model.generate_zero_state(1)
    got = fwd.forward(toks, state)
    model.forward_calls = 0
    ref = model.forward_batch(toks, state)
    assert model.forward_calls == 1, "disabled forward must route to forward_batch"
    assert got.shape == ref.shape
    assert torch.equal(got, ref), "disabled forward must be byte-identical"


def test_disabled_from_env_default_off(monkeypatch):
    monkeypatch.setattr(settings, "cuda_graph_decode", False)
    fwd = CudaGraphForward(_FakeModel())  # enabled=None -> settings
    assert fwd.enabled is False


def test_non_cuda_forward_falls_back_to_eager():
    """Non-CUDA tensors (or no GPU) on an ENABLED graph path must silently route
    every forward to the raw forward_batch and never capture -- so CPU/hermetic
    usage stays byte-identical to eager. Deterministic on any host."""
    model = _FakeModel()
    fwd = CudaGraphForward(model, enabled=True, max_bsz=8)
    toks = torch.tensor([[5]], dtype=torch.long)  # CPU tensor
    state = model.generate_zero_state(1)
    got = fwd.forward(toks, state)
    model.forward_calls = 0
    ref = model.forward_batch(toks, state)
    assert model.forward_calls == 1, "non-CUDA path must call forward_batch"
    assert torch.equal(got, ref)
    assert fwd._captured is False, "no capture may happen on a non-CUDA path"


def test_failed_flag_permanently_falls_back_to_eager(monkeypatch):
    """A capture that died (self._failed set) permanently skips re-capture and
    routes every forward to the raw forward_batch -- never raises."""
    model = _FakeModel()
    fwd = CudaGraphForward(model, enabled=True, max_bsz=8)
    fwd._failed = True  # simulate a prior capture failure
    capture_calls = []
    monkeypatch.setattr(
        CudaGraphForward, "capture", lambda self, b: capture_calls.append(True)
    )
    toks = torch.tensor([[5]], dtype=torch.long)
    state = model.generate_zero_state(1)
    got = fwd.forward(toks, state)
    assert capture_calls == [], "a failed graph must not re-attempt capture"
    model.forward_calls = 0
    ref = model.forward_batch(toks, state)
    assert model.forward_calls == 1
    assert torch.equal(got, ref)


# ---------------------------------------------------------------------------
# Real GPU: replay-vs-eager equivalence across decode steps + state round-trip
# ---------------------------------------------------------------------------

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
            "real-GPU CUDA-graph decode smoke)"
        )
    return _DEFAULT_MODEL


_PROMPT = "the quick brown fox jumps over the lazy dog and keeps running"


def _close_enough(actual, reference, what, tol=0.02):
    a = actual.float()
    b = reference.float()
    peak = max(a.abs().max().item(), b.abs().max().item(), 1.0)
    rel = (a - b).abs().max().item() / peak
    assert rel < tol, (
        f"{what} peak-relative diff {rel:.3e} >= {tol}: graph replay diverges "
        "from eager"
    )


@_PYTEST_GPU
def test_real_gpu_cuda_graph_decode_matches_eager():
    from model_load.model_loader import load_model_and_tokenizer

    model_path = _resolve_model_path()
    model, tokenizer, _args, _rocm = load_model_and_tokenizer(model_path)
    device = "cuda"

    tokens = tokenizer.encode(_PROMPT)
    assert len(tokens) > 1

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

    def eager_forward(tok_val, st):
        with torch.no_grad():
            return model.forward_batch(
                torch.tensor([[tok_val]], dtype=torch.long, device=device), st
            ).float()

    # Warm the allocator/CUDNN to steady state (as the dynamic decoder does with
    # its prefill before the first decode step) so the captured graph's kernels
    # match a steady-state eager forward.
    st_warm = fresh_state()
    for tk in (int(tokens[-1]), int(tokens[-1]) + 3, int(tokens[-1]) + 7):
        eager_forward(tk % vocab, st_warm)
    torch.cuda.synchronize()

    cgf = CudaGraphForward(model, enabled=True, max_bsz=8, sizes=(1, 2))
    cgf.capture(base)
    assert cgf.captured, "capture must have succeeded on a real GPU"
    assert sorted(cgf.sizes) == [1, 2]

    # FAIR interleaved comparison: at each decode step run eager AND the graph on
    # the same input token from independent states, then step both by their (equal,
    # verified) argmax. This measures real graph-vs-eager equivalence, unlike a
    # cold-then-warm serial comparison which only measures allocator settling.
    st_e = fresh_state()
    st_g = fresh_state()
    cur = int(tokens[-1])
    for step in range(3):
        oe = eager_forward(cur, st_e)
        og = cgf.forward(
            torch.tensor([[cur]], dtype=torch.long, device=device), st_g
        )
        _close_enough(og, oe, f"decode step {step}")
        a = int(oe.argmax(dim=-1).item())
        b = int(og.argmax(dim=-1).item())
        assert a == b, f"step {step} argmax mismatch"
        cur = a

    # The caller-state round-trip must leave the graph path's state equal (within
    # fp16 tolerance) to the eager path's after the same steps.
    for i, (ge, ga) in enumerate(zip(st_e, st_g)):
        peak = max(ge.abs().max().item(), ga.abs().max().item(), 1.0)
        rel = (ge - ga.float()).abs().max().item() / peak
        assert rel < 0.02, f"state[{i}] diverged after steps (rel {rel:.3e})"

    # Replaying the graph on identical (input, state) must be deterministic.
    rep1 = cgf.forward(
        torch.tensor([[cur]], dtype=torch.long, device=device), fresh_state()
    )
    rep2 = cgf.forward(
        torch.tensor([[cur]], dtype=torch.long, device=device), fresh_state()
    )
    assert torch.equal(rep1, rep2), "graph replay is not deterministic"

    del cgf, model, st_e, st_g
    torch.cuda.empty_cache()


@_PYTEST_GPU
def test_real_gpu_padded_size_batch_matches_eager():
    """bsz=2 routed through the size-2 graph (no waste) must equal a bsz=2 eager
    forward on the same inputs."""
    from model_load.model_loader import load_model_and_tokenizer

    model_path = _resolve_model_path()
    model, tokenizer, _args, _rocm = load_model_and_tokenizer(model_path)
    device = "cuda"

    tokens = tokenizer.encode(_PROMPT)
    state0 = model.generate_zero_state(0)
    out0 = model.forward(tokens, state0)
    base = [
        state0[0].unsqueeze(2).contiguous(),  # [L,2,1,C]
        state0[1].unsqueeze(1).contiguous(),  # [L,1,H,N,N]
        state0[2].reshape(1),
    ]

    def to_bsz2(st1):
        return [
            torch.cat([s, s.clone()], dim=(2, 1, 0)[di])
            for di, s in enumerate(st1)
        ]

    def eager2(st2):
        with torch.no_grad():
            return model.forward_batch(
                torch.tensor([[5], [6]], dtype=torch.long, device=device), st2
            ).float()

    # Warm the allocator/CUDNN with a couple of bsz-2 eager passes before capture.
    eager2(to_bsz2(base))
    eager2(to_bsz2(base))
    torch.cuda.synchronize()

    cgf = CudaGraphForward(model, enabled=True, max_bsz=8, sizes=(1, 2))
    cgf.capture(base)
    st_e2 = to_bsz2(base)
    eager_out = eager2(st_e2)
    st_g2 = to_bsz2(base)
    graph_out = cgf.forward(
        torch.tensor([[5], [6]], dtype=torch.long, device=device), st_g2
    )
    assert graph_out.shape == eager_out.shape, (
        tuple(graph_out.shape), tuple(eager_out.shape)
    )
    _close_enough(graph_out, eager_out, "bsz=2 graph vs eager")
    del cgf, model
    torch.cuda.empty_cache()