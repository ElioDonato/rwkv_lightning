import asyncio
import logging
import os
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger("infer.async_forward")

# Default OFF: the heavy synchronous GPU forward stays on the uvicorn event-loop
# thread, byte-identical to the pre-feature serving path. Only when
# RWKV_ASYNC_FORWARD is set to a nonzero value are the blocking torch calls
# handed to the single GPU-worker thread -- an opt-in, default-off mechanism to
# be validated against live GPU occupancy before it is promoted to default.
_ASYNC_FORWARD_ENV = "RWKV_ASYNC_FORWARD"
_ASYNC_FORWARD_DEFAULT = "0"


def _async_forward_enabled() -> bool:
    raw = os.environ.get(_ASYNC_FORWARD_ENV, _ASYNC_FORWARD_DEFAULT)
    try:
        return int(raw) != 0
    except (TypeError, ValueError):
        logger.warning(
            f"[AsyncForward] invalid {_ASYNC_FORWARD_ENV}={raw!r}, treating as off"
        )
        return False


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

    Default-off: while disabled (RWKV_ASYNC_FORWARD unset/0) offload() runs the
    callable directly on the calling (event-loop) thread and never spins up a
    worker thread, so the default serving path is byte-identical to before this
    feature. The executor object is created eagerly (cheap; it spawns no thread
    while disabled) so the engine can call offload() uniformly regardless of mode.
    """

    def __init__(self, enabled=None):
        # enabled=None resolves from RWKV_ASYNC_FORWARD at construction time so
        # tests can pin the mode deterministically regardless of env.
        self._enabled = _async_forward_enabled() if enabled is None else bool(enabled)
        self._executor = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="gpu-forward")
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
        """
        if self._enabled:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(self._executor, _invoke, fn, args, kwargs)
        return fn(*args, **kwargs)

    def shutdown(self, wait=False):
        if self._executor is not None:
            self._executor.shutdown(wait=wait)
            self._executor = None