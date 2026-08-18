import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor

from settings import settings

logger = logging.getLogger("infer.async_forward")

# Default OFF: the heavy synchronous GPU forward stays on the uvicorn event-loop
# thread, byte-identical to the pre-feature serving path. Only when the knob is
# enabled (settings.async_forward, backed by the RWKV_ASYNC_FORWARD env var -- the
# same name as before, now sourced from the central settings module) are the
# blocking torch calls handed to the single GPU-worker thread -- an opt-in,
# default-off mechanism to be validated against live GPU occupancy before it is
# promoted to default.


def _async_forward_enabled() -> bool:
    return settings.async_forward


def _invoke(fn, args, kwargs):
    # Module-level helper so the worker thread executes the plain callable
    # without reconstructing it from a lambda -- keeps the submit path trivial
    # and avoids any accidental closure/lambda capture at submit time.
    return fn(*args, **kwargs)


class GpuAsyncExecutor:
    """Single-worker executor for the heavy synchronous GPU forward.

    The bottleneck this solves: big_batch_stream and the async chat path call
    model.forward_batch()/prefill synchronously inside the single uvicorn
    event-loop thread. One heavy batch blocks the whole loop, and concurrent
    heavy requests serialize/wedge (2026-08-13 incident) while the GPU sits
    ~76% idle because the server only ever sees one request's forward at a
    time. Offloading those blocking torch calls to a worker lets the event loop
    keep servicing other requests' bookkeeping and admission queue while the
    GPU is busy -- measuring whether the GPU actually stays more occupied is
    the point of the opt-in experiment.

    CRITICAL SAFETY: torch CUDA work MUST stay serialized to exactly ONE
    backing thread. Concurrent CUDA calls from multiple worker threads would
    race on the CUDA driver/context and can corrupt or deadlock the model.
    ThreadPoolExecutor(max_workers=1) provides that one-thread guarantee: every
    GPU call funneled through offload() executes on the same dedicated worker.

    SCOPE OF THE GUARANTEE: this one-thread serialization covers ONLY the code
    paths routed through offload(). Those are the async streaming decode loops
    -- big_batch_stream, _batch_decode_streaming (behind batch_infer_stream /
    batch_infer_stream_v2), and the single/state streaming endpoints -- which
    hand every GPU-touching unit (prefill, each sampling+forward+decode step,
    sampler init/setup_rand, cleanup/_cleanup_cuda_memory) to offload(), so
    under the opt-in NO CUDA op runs on the event-loop thread while the worker
    has a forward in flight. It does NOT claim to be a process-wide "single CUDA
    thread": the blocking non-streaming cores (_batch_decode_blocking behind
    batch_generate/batch_generate_v2, plus batch_generate_state) run on
    Starlette's anyio threadpool on their own, are NOT routed through offload(),
    and are therefore outside this guarantee (they are not on the event-loop
    thread either, which is the hazard this executor exists to prevent). See
    infer/inference._offload_gpu for the same boundary note.

    Default-off: while disabled (settings.async_forward False, i.e. the
    RWKV_ASYNC_FORWARD env var unset/0) offload() runs the
    callable directly on the calling (event-loop) thread and never spins up a
    worker thread, so the default serving path is byte-identical to before this
    feature. The executor object is created eagerly (cheap; it spawns no thread
    while disabled) so the engine can call offload() uniformly regardless of mode.
    """

    def __init__(self, enabled=None):
        # enabled=None resolves from settings.async_forward at construction time
        # (env var RWKV_ASYNC_FORWARD) so tests can pin the mode deterministically
        # regardless of env.
        self._enabled = _async_forward_enabled() if enabled is None else bool(enabled)
        self._executor = (
            ThreadPoolExecutor(
                max_workers=1, thread_name_prefix=settings.async_forward_thread_name
            )
            if self._enabled
            else None
        )

    @property
    def enabled(self):
        return self._enabled

    @property
    def max_workers(self) -> int:
        # The one-thread serialization guarantee backing this executor; exposed
        # so tests can assert the invariant directly rather than relying on the
        # private _executor attribute.
        return 1

    async def offload(self, fn, *args, **kwargs):
        """Run `fn` such that it executes on the single GPU-worker thread.

        Disabled (default): runs `fn(*args, **kwargs)` inline on the calling
        event-loop thread -- identical to the pre-feature code path. Enabled:
        submits it to the single-worker ThreadPoolExecutor and awaits the
        result, yielding the event loop while the worker blocks on CUDA.

        After shutdown() (see below) `_enabled` is False and `_executor` is
        None, so this runs inline again -- it must NEVER fall through to
        `loop.run_in_executor(None, ...)` (the loop's DEFAULT multi-thread
        executor), which would reintroduce multi-threaded CUDA precisely the
        corruption class the single-worker invariant prevents. The
        `self._executor is not None` guard is the belt-and-braces for that.
        """
        if self._enabled and self._executor is not None:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(self._executor, _invoke, fn, args, kwargs)
        return fn(*args, **kwargs)

    def shutdown(self, wait=False):
        if self._executor is not None:
            self._executor.shutdown(wait=wait)
        # Release the worker and invalidate the one-thread guarantee: after
        # shutdown there is no single backing thread left, so any further use
        # must NOT hit the loop's default multi-thread executor. Flipping
        # _enabled off (and exposing that via the `enabled` property) makes any
        # post-shutdown offload() fall back to running inline rather than
        # silently spawning/using a pooled multi-thread executor.
        self._executor = None
        self._enabled = False