"""
Server-side DECODE COMBINE-QUEUE for /openai/v1/chat/completions (the :8081
occupancy lever).

Today every /openai/v1/chat/completions request runs a bsz=1 decode, so N
concurrent requests re-serialize on the event-loop thread at bsz=1 and the GPU
sits ~9-19% occupied (Mod A RWKV_ASYNC_FORWARD moved the forward to a worker
but each request is still bsz=1). This module is the occupancy lever: an
opt-in, default-OFF scheduler that merges concurrent, HOMOGENEOUS chat requests
into ONE shared multi-row decode through the existing ``engine.big_batch_stream``
(which already does row-compaction + incremental release_prefill_capacity),
then splits the per-index SSE streams back to each request.

Design
------
* ``ChatFuseAggregator`` owns a single scheduler coroutine + a FIFO queue of
  pending chat requests. Each request enqueues an ``_ChatJob`` and gets back a
  per-request *fuse stream* (an async-iterable of SSE chunk strings).
* **Head-fire starvation guard** (avoids the embed size-1 regression): a request
  arriving at an EMPTY queue is drained by itself and fires IMMEDIATELY through
  the ORIGINAL ``single_infer_stream`` path (window=0 -> byte-identical latency
  AND output/latency to today for an isolated request -- no added gather latency,
  same repetition-penalty sampler). Only when >=2 jobs are pending at a drain
  point does the scheduler open the gather window (``RWKV_FUSE_CHAT_WINDOW_MS``)
  to collect more homogeneous siblings into ONE shared ``big_batch_stream``
  decode.
* **Homogeneity gate**: ``big_batch_stream`` takes ONE temperature / max_tokens
  / stop_tokens for the batch, so only requests that AGREE on all three
  (require-equal, the safer choice) fuse at all; non-matching requests are
  deferred and run in their own batch (solo via ``single_infer_stream``). We
  deliberately require **equal** max_tokens rather than "within a cap range":
  big_batch_stream has a single shared max_length and no per-row max_tokens, so
  fusing rows with different max_tokens would over-generate the shorter rows
  (violating per-row finish_reason exactness) unless /big_batch were modified.
* **VRAM cap**: the fused batch acquires ONE prefill-admission permit for its
  full row count (``engine.acquire_prefill_permit(request_bsz=len(batch))``), so
  the shared admission budget caps it, AND a hard cap on fused rows
  (``RWKV_FUSE_CHAT_MAX_BSZ``, default 8) bounds a huge concurrent burst so it
  can never OOM a co-resident process. big_batch_stream's per-row compaction
  incremental-release (permit_box) hands freed row capacity back to the shared
  pool as rows finish.
* **Cancellation**: one row's disconnect marks that row's job cancelled -- the
  scheduler stops delivering chunks to it and never fails the batch. big_batch
  has no forced per-row drop (we must NOT modify /big_batch), so a mid-generation
  disconnect lets that row finish naturally and release its capacity via
  compaction; only its client-facing delivery is dropped. The batch (and the
  other rows) continue unaffected.
* **Exactness**: RWKV recurrent state is per-row independent (the same argument
  big_batch_stream relies on), so a fused multi-row decode yields each row the
  same tokens it would get from big_batch_stream run alone. SSE split-back keeps
  per-row stop / finish_reason semantics.

Sampler note (IMPORTANT for reviewers): solo requests ride ``single_infer_stream``
which uses the repetition-penalty / top_k / top_p sampler
(``batch_sampling_repetition_temperature_topk_topp``) -- byte-identical to today.
FUSED batches ride ``big_batch_stream`` which uses ``sampler_gumbel_batch``
(temp-only gumbel, no penalties / top_k / top_p). So a request's output is NOT
identical depending on whether it was solo or fused when the feature is ON: fused
output matches the established /big_batch/completions semantics, not the solo
chat semantics. This is inherent to re-using big_batch_stream as mandated; it is
flagged as a residual risk for CR1/CR2 (the deletion/extraction workload shares a
template + stop_tokens, so its rows are homogeneous and the gumbel-with-temp
sampling is acceptable).

Default-OFF
-----------
Matching Mod A (``RWKV_ASYNC_FORWARD``) and Mod B (``RWKV_EMBED_AGGREGATE``)
precedent. When off (``RWKV_FUSE_CHAT_BATCH`` unset/0, the default) the
edge route never calls into the aggregator (it uses the original
single_infer/single_infer_stream path), and ``submit()`` itself is a pure inline
``single_infer_stream`` proxy -- no queue, no scheduler, byte-identical latency
and output. The aggregator object may be created eagerly (it spawns no task
while disabled).

Not thread-safe by design: an asyncio component driven from the single
event-loop thread, exactly where the chat endpoint already runs.
"""

import asyncio
import json
import logging
import os
from collections import deque

logger = logging.getLogger("infer.fuse_aggregator")

_FUSE_ENV = "RWKV_FUSE_CHAT_BATCH"
_FUSE_DEFAULT = "0"
_WINDOW_MS_ENV = "RWKV_FUSE_CHAT_WINDOW_MS"
# Conservative: the window only ever delays requests that ALREADY have a
# sibling pending (the head-fire guard fires a lone request immediately), and it
# must only cover the few ms it takes a concurrent burst's requests to arrive.
_WINDOW_MS_DEFAULT = 5
_MAX_BSZ_ENV = "RWKV_FUSE_CHAT_MAX_BSZ"
# Hard ceiling on fused rows per shared big_batch_stream call. 8 concurrent
# chats is already a large occupancy gain over bsz=1 while bounding the VRAM
# spike of one fused batch against a co-resident :8083.
_MAX_BSZ_DEFAULT = 8

# Job-queue terminal sentinel (asyncio.Queue can't hold a bare None that
# collides with real data, so use a unique object).
_END = object()


def _fuse_enabled(enabled=None) -> bool:
    """Resolve the opt-in, honoring an explicit override (used by tests)."""
    if enabled is not None:
        return bool(enabled)
    raw = os.environ.get(_FUSE_ENV, _FUSE_DEFAULT)
    try:
        return bool(int(raw))
    except (TypeError, ValueError):
        logger.warning(f"[ChatFuse] invalid {_FUSE_ENV}={raw!r}, treating as off")
        return False


def _window_seconds(window_ms=None) -> float:
    """Gather window in seconds (>= 0). Explicit override wins over env."""
    if window_ms is not None:
        return max(0.0, int(window_ms)) / 1000.0
    raw = os.environ.get(_WINDOW_MS_ENV, str(_WINDOW_MS_DEFAULT))
    try:
        return max(0.0, int(raw) / 1000.0)
    except (TypeError, ValueError):
        logger.warning(f"[ChatFuse] invalid {_WINDOW_MS_ENV}={raw!r}, "
                       f"using {_WINDOW_MS_DEFAULT}ms")
        return _WINDOW_MS_DEFAULT / 1000.0


def _max_bsz_value(max_bsz=None) -> int:
    """Hard cap on fused rows. Explicit override wins over env; always int>=1."""
    if max_bsz is not None:
        cap = int(max_bsz)
    else:
        raw = os.environ.get(_MAX_BSZ_ENV, str(_MAX_BSZ_DEFAULT))
        try:
            cap = int(raw)
        except (TypeError, ValueError):
            logger.warning(f"[ChatFuse] invalid {_MAX_BSZ_ENV}={raw!r}, "
                           f"using {_MAX_BSZ_DEFAULT}")
            cap = _MAX_BSZ_DEFAULT
    return max(1, cap)


class _ChatJob:
    """One admitted chat request in the combine queue. ``queue`` receives SSE
    chunk strings (single_infer_stream-format: index-0 choices, ``data: ...``
    lines) plus the ``_END`` sentinel. ``cancelled`` marks a client-disconnected
    row for the scheduler to stop delivering chunks to; ``ended`` guards the
    exactly-once ``_END`` push."""

    __slots__ = ("request_id", "prompt", "max_tokens", "temperature",
                 "stop_tokens", "chunk_size", "queue", "cancelled", "ended",
                 "future")

    def __init__(self, request_id, prompt, max_tokens, temperature,
                 stop_tokens, chunk_size, loop):
        self.request_id = request_id
        self.prompt = prompt
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.stop_tokens = tuple(stop_tokens)
        self.chunk_size = chunk_size
        self.queue = asyncio.Queue()
        self.cancelled = False
        self.ended = False
        self.future = loop.create_future()


class _InlineFuseStream:
    """Default-off (and single-job) stream: a thin async-iterable over the
    engine's original ``single_infer_stream`` generator. Byte-identical to the
    pre-feature chat stream."""

    def __init__(self, engine, prompt, max_tokens, temperature, stop_tokens, chunk_size):
        self._engine = engine
        self._params = dict(
            prompt=prompt,
            max_length=max_tokens,
            temperature=temperature,
            stop_tokens=stop_tokens,
            chunk_size=chunk_size,
        )
        self._gen = None

    def __aiter__(self):
        return self._agen()

    async def _agen(self):
        self._gen = self._engine.single_infer_stream(**self._params)
        try:
            async for chunk in self._gen:
                yield chunk
        finally:
            try:
                await self._gen.aclose()
            except Exception:
                pass
            self._gen = None

    def cancel(self):
        # Solo path: closing/abandoning the iterator ends generation (the route's
        # body generator finalization calls aclose on _gen -> GPU freed). Mark
        # internally so a cancelled solo stream stops being drained if the route
        # is still iterating.
        pass

    async def aclose(self):
        # Mirrors an async generator's aclose() so the OpenAI route's relay can
        # close both the inline solo stream and a fused job stream uniformly.
        self.cancel()
        gen = self._gen
        if gen is not None:
            try:
                await gen.aclose()
            except Exception:
                pass
            self._gen = None


class _QueuedFuseStream:
    """Stream for a job in the combine queue: drains the job's SSE chunk queue
    until the ``_END`` sentinel, surfacing any batch exception via the job's
    future. ``cancel()`` marks the job cancelled (scheduler drops delivery)."""

    def __init__(self, job):
        self._job = job

    def __aiter__(self):
        return self._agen()

    async def _agen(self):
        job = self._job
        try:
            while True:
                item = await job.queue.get()
                if item is _END:
                    break
                yield item
        finally:
            if job.future.done() and job.future.exception() is not None:
                raise job.future.exception()

    def cancel(self):
        job = self._job
        job.cancelled = True
        if not job.ended:
            job.ended = True
            job.queue.put_nowait(_END)

    async def aclose(self):
        self.cancel()


class ChatFuseAggregator:
    """Bounded admission/combine worker for the chat endpoint.

    Enabled (opt-in via RWKV_FUSE_CHAT_BATCH): concurrent ``submit()`` calls are
    drained into shared multi-row ``big_batch_stream`` decodes (solo requests
    head-fire through ``single_infer_stream``). Disabled (default): ``submit()``
    is a pure inline ``single_infer_stream`` proxy.

    Owns the prefill-admission permit for each batch it runs (request_bsz = row
    count), so fused rows count against the shared max_prefill_bsz budget exactly
    as if run separately.
    """

    def __init__(self, engine, *, enabled=None, window_ms=None, max_bsz=None):
        self._engine = engine
        # enabled=None resolves from env at construction so tests pin the mode
        # deterministically regardless of environment.
        self._enabled = _fuse_enabled(enabled)
        self._window = _window_seconds(window_ms)
        # max_bsz=hard fused-row cap; None resolves from env/default.
        self._max_bsz = max_bsz
        self._pending = deque()  # of _ChatJob, FIFO admission order
        self._wake = asyncio.Event()
        self._task = None
        self._inflight_batch = None
        self._next_request_id = 0
        self._dead_warned = False

    # -- properties ---------------------------------------------------------

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def window(self) -> float:
        return self._window

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    # -- lifecycle ----------------------------------------------------------

    def start(self):
        if not self._enabled or self._task is not None:
            return
        self._task = asyncio.create_task(self._run_scheduler())

    async def stop(self):
        """Stop the scheduler, ending any still-queued/in-flight jobs so no
        awaiting fuse stream is orphaned on shutdown."""
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
            self._task = None
        # Abandon any jobs that were still queued/in-flight when the task ended.
        # This must NOT rely solely on the task's own CancelledError handler: if
        # the task was cancelled before it ever ran its loop (e.g. stop() right
        # after submit()), its body never executes and nothing would end the jobs.
        self._abandon_jobs(
            RuntimeError("chat fuse aggregator shut down before generating")
        )

    # -- request-facing API -------------------------------------------------

    async def submit(self, prompt, max_tokens, temperature, stop_tokens, chunk_size):
        """Enqueue a chat request and return a fuse stream (async-iterable of SSE
        chunk strings; ``cancel()`` marks the request's row dropped).

        Disabled (default): returns an inline ``single_infer_stream`` proxy --
        byte-identical output and latency to the pre-feature path. Enabled:
        enqueues the request; the scheduler runs it solo (head-fire) or fused.
        """
        if not self._enabled:
            return _InlineFuseStream(
                self._engine, prompt, max_tokens, temperature, stop_tokens, chunk_size
            )
        if self._task is None:
            self.start()
            if self._task is None:
                # start() was a no-op (e.g. feature became disabled); fall back
                # to the inline solo stream so this request still completes.
                return _InlineFuseStream(
                    self._engine, prompt, max_tokens, temperature, stop_tokens, chunk_size
                )
            if not self._dead_warned:
                self._dead_warned = True
                logger.warning(
                    "[ChatFuse] scheduler task was absent while enabled; re-started "
                    "it before serving request -- batching restored. If this recurs, "
                    "a prior scheduler crash was the cause."
                )
        loop = asyncio.get_running_loop()
        job = _ChatJob(
            self._next_request_id, prompt, max_tokens, temperature, stop_tokens,
            chunk_size, loop,
        )
        self._next_request_id += 1
        self._pending.append(job)
        self._wake.set()
        return _QueuedFuseStream(job)

    # -- scheduler internals ------------------------------------------------

    def _cap(self):
        """Hard cap on fused rows per batch, additionally bounded by the model's
        prefill-admission limit so acquire_prefill_permit(request_bsz=cap) never
        rejects the whole batch. Always int>=1."""
        cap = _max_bsz_value(self._max_bsz)
        limit = int(getattr(self._engine.model, "max_prefill_bsz_limit", 0) or 0)
        if limit > 0 and cap > limit:
            cap = limit
        return max(1, cap)

    def _max_permit_bsz(self):
        """Upper bound a fused batch could ever need to admit (cap) filtered by
        the model's absolute prefill limit -- used to split an oversized burst."""
        return self._cap()

    @staticmethod
    def _homogeneous(a, b):
        return (
            a.max_tokens == b.max_tokens
            and a.temperature == b.temperature
            and tuple(a.stop_tokens) == tuple(b.stop_tokens)
        )

    def _fill_batch(self, batch, cap):
        """Admit the HEAD pending job always (never starve it), then admit
        homogeneous jobs from the FRONT of the FIFO queue up to ``cap``. Stops at
        the first non-matching job (deferred to its own batch) so FIFO order is
        preserved and no job is ever skipped. Returns rows admitted."""
        if not batch:
            if not self._pending:
                return 0
            batch.append(self._pending.popleft())  # HEAD always admitted first
        head = batch[0]
        while self._pending and len(batch) < cap:
            if self._homogeneous(head, self._pending[0]):
                batch.append(self._pending.popleft())
            else:
                break
        return len(batch)

    async def _run_scheduler(self):
        loop = asyncio.get_running_loop()
        try:
            while True:
                while not self._pending:
                    self._wake.clear()
                    await self._wake.wait()

                cap = self._cap()
                self._inflight_batch = []
                n = self._fill_batch(self._inflight_batch, cap)
                if n == 0 and self._pending:
                    logger.error(
                        "[ChatFuse] 0-admission window with %d pending; refusing to "
                        "spin empty batches", len(self._pending),
                    )
                    raise RuntimeError("chat fuse aggregator admitted no jobs in a busy window")

                if n == 1:
                    # HEAD-FIRE: exactly one job pending and no sibling -> run it
                    # solo through the ORIGINAL single_infer_stream path with no
                    # gather window (byte-identical latency/output to today). This
                    # is the critical guard against the embed size-1 regression
                    # (a lone request must not be delayed by the gather window).
                    # self._inflight_batch is left pointing at the running solo
                    # job so a mid-run cancel/abandon ends it, not just queued jobs.
                    job = self._inflight_batch[0]
                    try:
                        await self._run_solo(job)
                    finally:
                        self._inflight_batch = None
                    continue

                # >=2 jobs pending, a sibling is already waiting -> open the
                # gather window so a burst of concurrent arrivals joins this same
                # fused batch. Fire EARLY on cap-full; otherwise drain until the
                # window elapses.
                deadline = loop.time() + self._window
                self._wake.clear()
                while len(self._inflight_batch) < cap:
                    remaining = deadline - loop.time()
                    if remaining <= 0:
                        break
                    try:
                        await asyncio.wait_for(self._wake.wait(), timeout=remaining)
                    except asyncio.TimeoutError:
                        continue
                    self._wake.clear()
                    self._fill_batch(self._inflight_batch, cap)

                batch = self._inflight_batch
                try:
                    await self._run_fused(batch)
                finally:
                    self._inflight_batch = None
        except asyncio.CancelledError:
            self._abandon_jobs(
                RuntimeError("chat fuse aggregator shut down before generating")
            )
            raise
        except Exception as exc:
            logger.error(
                "[ChatFuse] scheduler failed, failing pending request(s): %s", exc
            )
            self._abandon_jobs(RuntimeError(f"chat fuse aggregator scheduler failed: {exc}"))
            self._task = None

    def _abandon_jobs(self, exc):
        """On shutdown/crash: end every still-queued and in-flight job so no fuse
        stream hangs; surface the exception via each job's future."""
        for job in list(self._pending) + (self._inflight_batch or []):
            if not job.future.done():
                job.future.set_exception(exc)
            if not job.ended:
                job.ended = True
                job.queue.put_nowait(_END)
        self._pending.clear()
        self._inflight_batch = None
        self._wake.set()

    async def _run_solo(self, job):
        """Run one job through the ORIGINAL single_infer_stream path (solo)."""
        permit = None
        try:
            permit = await self._engine.acquire_prefill_permit(
                request_bsz=1, request_label="/openai/v1/chat/completions(fuse-solo)"
            )
            stream = self._engine.single_infer_stream(
                prompt=job.prompt,
                max_length=job.max_tokens,
                temperature=job.temperature,
                stop_tokens=job.stop_tokens,
                chunk_size=job.chunk_size,
            )
            try:
                async for chunk in stream:
                    if job.cancelled:
                        break
                    job.queue.put_nowait(chunk)
            finally:
                await stream.aclose()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if not job.future.done():
                job.future.set_exception(exc)
            logger.error("[ChatFuse] solo job %s failed: %s", job.request_id, exc)
        finally:
            if permit is not None:
                await self._engine.release_prefill_permit(
                    request_bsz=1,
                    request_label="/openai/v1/chat/completions(fuse-solo)",
                    permit=permit,
                )
            self._end_job(job)

    async def _run_fused(self, batch):
        """Run a homogeneous batch through ONE shared big_batch_stream decode and
        split the per-index SSE chunks back to each job's queue."""
        engine = self._engine
        head = batch[0]
        permit_box = [None]
        permit = None
        try:
            permit = await engine.acquire_prefill_permit(
                request_bsz=len(batch),
                request_label="/openai/v1/chat/completions(fuse)",
            )
            permit_box[0] = permit
            stream = engine.big_batch_stream(
                prompts=[job.prompt for job in batch],
                max_length=head.max_tokens,
                temperature=head.temperature,
                stop_tokens=head.stop_tokens,
                chunk_size=head.chunk_size,
                permit_box=permit_box,
            )
            async for chunk in stream:
                if not isinstance(chunk, str) or not chunk.startswith("data: "):
                    continue
                payload = chunk[6:].strip()
                if not payload or payload == "[DONE]":
                    # batch-level [DONE] is not forwarded; each finished row gets
                    # its own per-row _END below.
                    continue
                try:
                    data = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                for choice in data.get("choices") or []:
                    idx = choice.get("index")
                    if idx is None or not (0 <= idx < len(batch)):
                        continue
                    job = batch[idx]
                    if job.cancelled:
                        continue
                    delta = choice.get("delta") or {}
                    finish = choice.get("finish_reason")
                    content = delta.get("content")
                    if content:
                        job.queue.put_nowait(self._client_chunk({"index": 0, "delta": {"content": content}}))
                    if finish is not None:
                        # Emit finish as its OWN chunk (delta {} + finish_reason),
                        # matching the solo single_infer_stream format so the
                        # client-facing relay treats both paths uniformly.
                        job.queue.put_nowait(
                            self._client_chunk({"index": 0, "delta": {}, "finish_reason": finish})
                        )
                        self._end_job(job)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            for job in batch:
                if not job.future.done():
                    job.future.set_exception(exc)
            logger.error("[ChatFuse] fused batch of %d failed: %s", len(batch), exc)
        finally:
            for job in batch:
                self._end_job(job)
            if permit is not None:
                await engine.release_prefill_permit(
                    request_bsz=len(batch),
                    request_label="/openai/v1/chat/completions(fuse)",
                    permit=permit,
                )

    @staticmethod
    def _client_chunk(choice):
        data = json.dumps(
            {"object": "chat.completion.chunk", "choices": [choice]},
            ensure_ascii=False,
        )
        return f"data: {data}\n\n"

    def _end_job(self, job):
        if not job.ended:
            job.ended = True
            job.queue.put_nowait(_END)