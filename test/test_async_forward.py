"""
Coverage for Mod A: the opt-in, default-off mechanism that moves the heavy
synchronous GPU forward (prefill + each forward_batch decode step) off the
single uvicorn event-loop thread onto a dedicated single GPU-worker thread.

Pure asyncio/threading bookkeeping with CPU/mock tensors only -- no CUDA, no
real model. Uses a FakeModel exposing just the shape of the model interface the
async decode loops read (generate_zero_state / forward_batch /
forward_batch_same_length / prefill_chunk_size) plus a stub tokenizer and the
real sampler_gumbel_batch (which is CUDA-free and works identically on CPU
tensors). Pattern mirrors test_prefill_admission_queue.py's _FakeModel /
_make_engine approach; anyio's pytest plugin (already a dependency via
FastAPI/Starlette) provides the async test runner.

Run with: source env.sh && uv run pytest test/test_async_forward.py -v
"""
import asyncio
import os
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
    from infer.async_forward import GpuAsyncExecutor
    from infer.inference import InferenceEngine
except OSError as exc:
    pytest.skip(f"CUDA environment not configured: {exc}", allow_module_level=True)


@pytest.fixture
def anyio_backend():
    return "asyncio"


class _FakeModel:
    """Minimal stand-in for the real RWKV model: CPU tensors only, records which
    thread each GPU-shaped op ran on so tests can prove the offload actually
    hands the work to the single worker thread."""

    def __init__(self, vocab=6, prefill_chunk_size=2):
        self.vocab = vocab
        self.prefill_chunk_size = prefill_chunk_size
        self.forward_threads = []  # thread ids that forward_batch ran on
        self.prefill_threads = []  # thread ids that prefill ran on

    def generate_zero_state(self, bsz):
        return [
            torch.zeros(1, 2, bsz),
            torch.zeros(1, bsz, 1, 1, 1),
            torch.zeros(bsz, dtype=torch.int32),
        ]

    def forward(self, tokens, state):
        # single-sequence path (used by _forward_tokens_chunked)
        self.prefill_threads.append(threading.get_ident())
        if isinstance(tokens, list):
            n = len(tokens)
        else:
            n = 1
        return torch.zeros(n, self.vocab)

    def forward_batch(self, tokens, state, full_output=False):
        # decode-step path; `tokens` is a [B, 1] CPU LongTensor
        self.forward_threads.append(threading.get_ident())
        bsz = tokens.shape[0] if isinstance(tokens, torch.Tensor) else len(tokens)
        return torch.zeros(bsz, self.vocab)

    def forward_batch_same_length(self, batch_tokens, batch_state, full_output=False):
        # prefill path; `batch_tokens` is a list of equal-length token lists
        self.prefill_threads.append(threading.get_ident())
        bsz = (
            batch_tokens.shape[0]
            if isinstance(batch_tokens, torch.Tensor)
            else len(batch_tokens)
        )
        return torch.zeros(bsz, self.vocab)


class _FakeTokenizer:
    """Stub tokenizer: deterministic encode, empty decode (so the decode loop
    never emits text content and terminates on token 0 or max_length)."""

    def encode(self, text):
        return [1, 2, 3]

    def decode(self, tokens, utf8_errors="strict"):
        return ""


def _make_engine(enabled=False):
    """Build an InferenceEngine wired to a FakeModel, with the opt-in GPU-worker
    executor explicitly pinned to the requested mode regardless of env."""
    engine = InferenceEngine(
        model=_FakeModel(), tokenizer=_FakeTokenizer(), args=None, rocm_flag=False
    )
    engine._gpu_executor = GpuAsyncExecutor(enabled=enabled)
    return engine


def _collect(engine, **kw):
    """Collect every SSE chunk big_batch_stream yields for a tiny 2-prompt batch."""
    chunks = []

    async def _run():
        async for c in engine.big_batch_stream(
            prompts=["hello world", "hi there"], **kw
        ):
            chunks.append(c)

    asyncio.run(_run())
    return chunks


# ---------------------------------------------------------------------------
# Executor-level tests
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_executor_default_off_runs_inline_on_caller():
    """Default (RWKV_ASYNC_FORWARD unset/0): offload() is a pure pass-through
    that runs the callable on the calling (event-loop) thread and never spawns
    a worker thread -- byte-identical to the pre-feature serving path."""
    executor = GpuAsyncExecutor(enabled=False)
    assert executor.enabled is False
    assert executor._executor is None, "disabled executor must not create a thread"

    caller = threading.get_ident()
    ran_on = []
    result = await executor.offload(lambda: ran_on.append(threading.get_ident()) or 42)

    assert result == 42
    assert ran_on == [caller], "disabled offload must run on the calling thread"


@pytest.mark.anyio
async def test_executor_enabled_runs_on_worker_thread():
    """When enabled, offload() hands the callable to a distinct worker thread,
    so the event loop is free to do other work while it awaits the result."""
    executor = GpuAsyncExecutor(enabled=True)
    assert executor.enabled is True
    caller = threading.get_ident()
    ran_on = []
    result = await executor.offload(
        lambda: ran_on.append(threading.get_ident()) or "from-worker"
    )
    assert result == "from-worker"
    assert len(ran_on) == 1
    assert ran_on[0] != caller, "enabled offload must run off the event-loop thread"
    executor.shutdown()


@pytest.mark.anyio
async def test_executor_serializes_to_exactly_one_worker():
    """CRITICAL safety invariant: torch CUDA work must stay serialized to one
    backing thread. Even N overlapping offloads must all land on the same
    single worker thread (ThreadPoolExecutor(max_workers=1)), never on two."""
    executor = GpuAsyncExecutor(enabled=True)
    assert executor.max_workers == 1

    caller = threading.get_ident()
    worker_ids = []

    async def one(i):
        def _work():
            worker_ids.append(threading.get_ident())
            return i
        return await executor.offload(_work)

    results = await asyncio.gather(*[one(i) for i in range(8)])
    assert results == list(range(8))
    assert len(worker_ids) == 8
    assert len(set(worker_ids)) == 1, "all offloads must run on exactly one thread"
    assert worker_ids[0] != caller
    executor.shutdown()


# ---------------------------------------------------------------------------
# Engine-level tests
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_engine_offload_gpu_default_off_is_byte_identical():
    """engine._offload_gpu with the executor disabled returns exactly what a
    direct call would, executed on the event-loop thread -- nothing about the
    default serving path changes."""
    engine = _make_engine(enabled=False)
    assert engine._gpu_executor.enabled is False
    caller = threading.get_ident()
    ran_on = []
    result = await engine._offload_gpu(
        lambda: ran_on.append(threading.get_ident()) or 21 * 2
    )
    assert result == 42
    assert ran_on == [caller]


@pytest.mark.anyio
async def test_engine_offload_gpu_enabled_runs_on_worker():
    engine = _make_engine(enabled=True)
    caller = threading.get_ident()
    ran_on = []
    result = await engine._offload_gpu(
        lambda: ran_on.append(threading.get_ident()) or "off-epoch"
    )
    assert result == "off-epoch"
    assert ran_on and ran_on[0] != caller
    engine.shutdown()


# ---------------------------------------------------------------------------
# Integration tests: big_batch_stream (the documented occupancy bottleneck)
# ---------------------------------------------------------------------------


def test_big_batch_stream_default_off_runs_forward_on_event_loop():
    """Default-off must keep every GPU forward on the event-loop thread (the
    pre-feature behavior) and still produce a complete, well-formed stream."""
    engine = _make_engine(enabled=False)
    caller = threading.get_ident()
    chunks = _collect(engine, max_length=4)

    assert chunks and chunks[-1] == "data: [DONE]\n\n"
    assert engine.model.forward_threads, "forward_batch should have been called"
    assert engine.model.prefill_threads, "prefill should have been called"
    # With the opt-in off, all GPU-shaped work runs inline on the event-loop thread.
    assert all(t == caller for t in engine.model.forward_threads)
    assert all(t == caller for t in engine.model.prefill_threads)


def test_big_batch_stream_opt_in_runs_forward_on_single_worker():
    """Opt-in (enabled): big_batch_stream's heavy forward is handed to the
    single GPU-worker thread, off the event loop, and still streams correct
    output to [DONE]."""
    engine = _make_engine(enabled=True)
    caller = threading.get_ident()
    chunks = _collect(engine, max_length=4)

    assert chunks and chunks[-1] == "data: [DONE]\n\n"
    assert engine.model.forward_threads, "forward_batch should have been called"
    assert engine.model.prefill_threads, "prefill should have been called (offloaded)"
    # All GPU forwards ran on the worker thread(s), never on the event loop.
    assert all(t != caller for t in engine.model.forward_threads)
    assert all(t != caller for t in engine.model.prefill_threads)
    # And they all landed on exactly one worker thread (single-thread CUDA).
    all_gpu_threads = set(engine.model.forward_threads + engine.model.prefill_threads)
    assert len(all_gpu_threads) == 1, "GPU work must stay on exactly one worker thread"
    engine.shutdown()


def test_big_batch_both_modes_produce_valid_sse_streams():
    """Both modes must produce a well-formed, complete SSE stream ending in
    [DONE]. Exact finish/stop chunk counts differ run-to-run (sampler_gumbel_batch
    consumes per-thread torch RNG, and the opt-in runs the sampler on the worker
    thread while default-off runs it on the event loop -- so the two are not
    bit-comparable) -- what matters is that neither regime truncates or corrupts
    the stream a downstream SSE consumer depends on."""
    for enabled in (False, True):
        chunks = _collect(_make_engine(enabled), max_length=4)
        assert chunks, "stream must not be empty"
        assert all(c.startswith("data: ") and c.endswith("\n\n") for c in chunks)
        assert chunks[-1] == "data: [DONE]\n\n", "stream must terminate with [DONE]"


# ---------------------------------------------------------------------------
# CR1 regressions: (1) no CUDA op may run on the event-loop thread while the
# worker is active under the opt-in (cleanup/_cleanup_cuda_memory + sampler
# init/setup_rand were left on the loop by the original diff); (2) shutdown()
# must disable the executor so nothing ever falls through to the loop's DEFAULT
# multi-thread executor; (4) offloaded batch_inference decode closures must
# re-enter torch.no_grad() in BOTH modes (no mode-dependent asymmetry).
# ---------------------------------------------------------------------------


# -- Finding 2: shutdown() must disable, and post-shutdown offload stays safe --

@pytest.mark.anyio
async def test_shutdown_sets_enabled_false_and_post_shutdown_offload_runs_inline():
    """shutdown() must flip _enabled off (exposed via `enabled`) AND release the
    executor, so that any post-shutdown offload() NEVER reaches the loop's default
    multi-thread ThreadPoolExecutor (which would re-open multi-threaded CUDA).
    The post-shutdown offload must run inline on the caller, proving no new
    executor/thead is used."""
    executor = GpuAsyncExecutor(enabled=True)
    assert executor.enabled is True

    first = []
    await executor.offload(lambda: first.append(threading.get_ident()) or "worker")

    executor.shutdown()
    assert executor.enabled is False, "shutdown must flip enabled off"
    assert executor._executor is None, "shutdown must release the worker"

    caller = threading.get_ident()
    after = []
    result = await executor.offload(
        lambda: after.append(threading.get_ident()) or 42
    )
    assert result == 42
    # Runs inline on the caller -> cannot have gone to any executor.
    assert after == [caller], (
        "post-shutdown offload must run inline, never on the loop's DEFAULT executor"
    )


def test_engine_shutdown_flips_executor_disabled_and_post_shutdown_runs_inline():
    """engine.shutdown() (the hook app.py calls) must leave _gpu_executor
    disabled and running inline, so a shutdown engine never submits to a
    default multi-thread executor either."""
    engine = _make_engine(enabled=True)
    assert engine._gpu_executor.enabled is True

    caller = threading.get_ident()
    ran_on = []

    async def _main():
        # exercise one real worker offload before shutdown
        await engine._offload_gpu(lambda: ran_on.append(threading.get_ident()))
        engine.shutdown()
        assert engine._gpu_executor.enabled is False, "engine.shutdown must disable"
        result = await engine._offload_gpu(
            lambda: ran_on.append(threading.get_ident()) or 7
        )
        assert result == 7

    asyncio.run(_main())
    # Only the pre-shutdown offload left the event loop; the post-shutdown one
    # must have run inline on the caller thread.
    assert ran_on[0] != caller
    assert ran_on[-1] == caller


# -- Finding 1: cleanup (_cleanup_cuda_memory) moves to the worker when enabled

def _record_cleanup_thread():
    """Return a (list, fn) pair: fn shadows engine._cleanup_cuda_memory to
    record the thread id it runs on and mirrors the real static's CPU-safe
    behavior (gc.collect(); on CPU cuda.is_available() is False so it returns
    before any synchronize/empty_cache)."""
    import gc

    threads = []

    def _record():
        threads.append(threading.get_ident())
        gc.collect()

    return threads, _record


def test_big_batch_cleanup_offloaded_to_worker_when_enabled():
    """Finding 1: big_batch_stream's finally cleanup (_cleanup_cuda_memory, a
    CUDA-touching op) must run on the single worker thread under the opt-in --
    never on the event-loop thread worrying about a worker forward in flight."""
    threads, record = _record_cleanup_thread()
    engine = _make_engine(enabled=True)
    engine._cleanup_cuda_memory = record
    caller = threading.get_ident()

    _collect(engine, max_length=4)

    assert threads, "finally cleanup must have run"
    assert threads[0] != caller, "opt-in: cleanup must run on the worker thread"
    engine.shutdown()


def test_big_batch_cleanup_stays_on_loop_when_default_off():
    """Findings 1: with the opt-in off, cleanup stays on the (calling) event-loop
    thread exactly as before -- only the opt-in path ever moves it."""
    threads, record = _record_cleanup_thread()
    engine = _make_engine(enabled=False)
    engine._cleanup_cuda_memory = record
    caller = threading.get_ident()

    _collect(engine, max_length=4)

    assert threads, "finally cleanup must have run"
    assert threads[0] == caller, "default-off: cleanup must stay on the loop thread"


# -- Findings 1 + 4: batch_infer_stream sampler-init thread + no_grad re-entry

def test_batch_infer_stream_sampler_init_and_decode_no_grad():
    """Finding 1 + 4, both modes:
    - sampler construction (setup_rand / cuda-alloc, the CUDA ops the seam must
      funnel) must run on the worker thread when enabled, and on the calling
      (event-loop) thread when default-off;
    - the offloaded _gpu_decode_step closure must re-enter torch.no_grad() in
      BOTH modes (no mode-dependent asymmetry), asserted by observing
      torch.is_grad_enabled() inside sampler.sample.
    Uses a fake _V1BatchSampler (CPU, no CUDA alloc) so this is hermetic; the
    real sampler's setup_rand genuinely needs a CUDA device and is excised here.
    """
    from types import SimpleNamespace

    import infer.batch_inference as batch_mod

    init_threads = []
    grad_flags = []

    class _FakeV1Sampler:
        """CPU-only stand-in: __init__ records its thread (setup_rand/cuda-alloc
        would be the CUDA ops here); sample() is a greedy argmax that records
        whether no_grad was re-entered by the enclosing decode closure."""

        def __init__(self, batch_size, vocab_size, device, temperature,
                     top_k, top_p, alpha_presence, alpha_frequency, alpha_decay):
            init_threads.append(threading.get_ident())

        def sample(self, logits):
            grad_flags.append(torch.is_grad_enabled())
            return logits.argmax(dim=-1, keepdim=True)

        def compact(self, idx_t):
            pass

    for enabled in (False, True):
        init_threads.clear()
        grad_flags.clear()
        engine = _make_engine(enabled=enabled)
        engine.model.args = SimpleNamespace(vocab_size=engine.model.vocab)
        batch_mod._V1BatchSampler = _FakeV1Sampler

        async def _run():
            async for _ in engine.batch_infer_stream(
                prompts=["hello", "world"], max_length=4
            ):
                pass

        caller = threading.get_ident()
        asyncio.run(_run())

        assert init_threads, "sampler init must have run"
        if enabled:
            assert init_threads[0] != caller, (
                "opt-in: sampler init (setup_rand/cuda-alloc) must run on the worker"
            )
        else:
            assert init_threads[0] == caller, (
                "default-off: sampler init must stay on the calling thread"
            )

        assert grad_flags, "decode path must have sampled"
        assert all(not g for g in grad_flags), (
            "offloaded decode closure must re-enter no_grad in both modes"
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))