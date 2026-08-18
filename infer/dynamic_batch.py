"""Server-side DYNAMIC BATCH DECODER for /openai/v1/chat/completions (Phase-4
sub-step 2: the "general + both" occupancy win).

Where the fuse combine-queue (infer/fuse_aggregator.py) merges only HOMOGENEOUS
requests into one ``big_batch_stream`` decode (and its fused rows drop
top_k/top_p/alpha/prefix-caching), this module merges ANY mix of concurrent
chat requests into ONE long-running multi-row decode with PER-ROW sampling:
each row keeps its own temperature / top_k / top_p / alpha_presence /
alpha_frequency / alpha_decay and its own per-row RNG + penalty state, and may
JOIN the batch mid-stream (as new requests arrive) and LEAVE as it finishes.
Rows are compacted out of the shared batch as they complete and newly-arrived
rows are reprefilled into the freed slots, so the GPU decode batch stays full
under a burst of heterogeneous requests.

It is opt-in and default-OFF (``RWKV_DYNAMIC_BATCH``, default off). With no env
set the server is byte-identical to today: the edge route never calls into the
decoder, and ``submit()`` is a pure inline ``single_infer_stream`` proxy (no
queue, no decoder task) -- exactly mirroring the fuse default-off contract so
the two opt-ins can coexist without changing today's output.

Design
------
* ``DynamicBatchDecoder`` owns ONE persistent scheduler coroutine
  (``_run_scheduler``) that keeps a single ``_RunningBatch`` decode alive and
  full. It holds a FIFO ``deque`` of pending jobs; each ``submit`` enqueues a
  ``_DynamicJob`` and returns a per-request *dynamic stream* (an async-iterable
  of SSE chunk strings the route forwards verbatim).
* **Admission / refill**: when the batch has room (bounded by
  ``settings.dynamic_batch_max_bsz``, default 8, additionally capped by the
  model's ``max_prefill_bsz_limit``), pending jobs are popped, batch-prefilled
  (``engine._forward_batch_prompts_chunked``) to get each new row's batch-layout
  state + initial logits, given fp32 per-row ``penalties`` + per-row-seeded RNG
  (``sample.setup_rand(job.seed, 1)`` per row, cat'd into the flat byte tensor),
  and the shared ``state``/``out``/``penalties``/``rand_states`` tensors are
  grown to the new B (``torch.cat`` along the batch dims). A batch-layout permit
  is acquired via ``engine.acquire_prefill_permit(request_bsz=n)`` per admitted
  group and released per finished row via ``release_prefill_capacity``.
* **Head-fire guard** (mirrors fuse): a LONE request arriving at an idle decoder
  fires immediately with NO gather window; the gather window (default 8ms) only
  opens when >=2 jobs are pending, and between decode steps newly-pending jobs
  are re-admitted immediately (no window) so freed slots refill fast.
* **Decode step** (ONE ``engine._offload_gpu`` unit, ``no_grad()``, never
  ``inference_mode``): ``sample_batch_per_row(out, penalties, rand_states,
  per_row_...)`` then ``out = model.forward_batch(new_tokens, state)``, with
  ``.tolist()`` inside the same closure. All per-token bookkeeping (stop-state,
  SSE chunks, remaining, chunk buffers) runs on the event loop between offload
  units, matching big_batch_stream's merged-step pattern.
* **Compaction**: when a row finishes (stop / max_tokens / cancellation) it is
  dropped from the shared batch -- ``engine._compact_active_rows`` shrinks
  ``state``/``out``/``penalties``/``rand_states`` to the surviving rows, the
  per-row scalar lists and the ``_RunningBatch`` bookkeeping arrays are
  re-filtered to the surviving slots, and that row's capacity is released back
  to the shared prefill budget immediately.
* **Exactness**: RWKV recurrent state is per-row independent, so a shared
  multi-row decode yields each row the SAME tokens a solo decode with the same
  RNG seed + that row's own sampler controls would produce. Each row's RNG is
  per-request-seeded (drawn at job creation).
* **Cancellation**: ``cancel()``/``aclose()`` on a dynamic stream marks the job
  cancelled; the decoder force-drops that row at the next decode step (frees the
  slot + releases its capacity) since the dynamic decoder owns its own batch --
  it does not leave a cancelled row occupying a GPU slot to the end (unlike the
  fuse path). The rest of the batch is unaffected.

Default-OFF / mirror fuse
-------------------------
When off, ``submit()`` returns an ``_InlineFuseStream`` (imported from
infer/fuse_aggregator.py) -- an inline ``single_infer_stream`` proxy that
forwards the request's own sampler controls and acquires/releases its own bsz=1
prefill permit, byte-identical to the pre-feature chat path. The decoder spawns
no task while disabled. Not thread-safe by design (asyncio-only, driven from the
single event-loop thread)."""

import asyncio
import json
import logging
import random
from collections import deque

from settings import settings

from infer import inference_deps
from infer.fuse_aggregator import _InlineFuseStream

logger = logging.getLogger("infer.dynamic_batch")

# Request label for prefill-admission permit bookkeeping.
_DYNAMIC_LABEL = "/openai/v1/chat/completions(dynamic)"

# Job-queue terminal sentinel (asyncio.Queue can't hold a bare None that could
# collide with real data, so use a unique object).
_END = object()


def _dynamic_enabled(enabled=None) -> bool:
    """Resolve the opt-in, honoring an explicit override (used by tests)."""
    if enabled is not None:
        return bool(enabled)
    return bool(settings.dynamic_batch)


def _window_seconds(window_ms=None) -> float:
    """Gather window in seconds (>= 0). Explicit override wins over settings."""
    if window_ms is not None:
        return max(0.0, int(window_ms)) / 1000.0
    return max(0.0, int(settings.dynamic_batch_window_ms)) / 1000.0


def _max_bsz_value(max_bsz=None) -> int:
    """Hard cap on fused rows per decode batch. Explicit override wins over
    settings; always int>=1."""
    if max_bsz is not None:
        cap = int(max_bsz)
    else:
        cap = int(settings.dynamic_batch_max_bsz)
    return max(1, cap)


class _DynamicJob:
    """One admitted chat request in the dynamic batch. ``queue`` receives SSE
    chunk strings (index-0 choices, ``data: ...`` lines) plus the ``_END``
    sentinel; ``future`` carries a shutdown/crash exception if the scheduler
    dies before generating. ``cancelled`` marks a client-disconnected row for
    the scheduler to drop and free its slot; ``finished`` guards the exactly-once
    end-of-row processing (capacity release + stream end)."""

    __slots__ = ("request_id", "prompt", "max_tokens", "temperature", "top_k",
                 "top_p", "alpha_presence", "alpha_frequency", "alpha_decay",
                 "stop_tokens", "chunk_size", "seed", "queue", "cancelled",
                 "ended", "finished", "permit", "cancel_token", "future")

    def __init__(self, request_id, prompt, max_tokens, temperature, stop_tokens,
                 chunk_size, top_k, top_p, alpha_presence, alpha_frequency,
                 alpha_decay, cancel_token, loop):
        self.request_id = request_id
        self.prompt = prompt
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.alpha_presence = alpha_presence
        self.alpha_frequency = alpha_frequency
        self.alpha_decay = alpha_decay
        # stop_tokens may be None (a request body can pass `stop_tokens: null`);
        # convert safely to an empty tuple so engine._create_stop_state never sees
        # ``tuple(None)``.
        self.stop_tokens = tuple(stop_tokens) if stop_tokens else ()
        self.chunk_size = chunk_size
        # Per-request RNG seed: keeps each row's decode the same as a solo decode
        # with the same seed + that row's own sampler controls (exactness).
        self.seed = random.randint(0, 2**63 - 1)
        self.queue = asyncio.Queue()
        self.cancelled = False
        self.ended = False
        self.finished = False
        self.permit = None       # prefill-admission permit this row's group holds
        self.cancel_token = cancel_token
        self.future = loop.create_future()


class _RunningBatch:
    """The persistent shared decode batch. The GPU tensors (``state``/``out``/
    ``penalties``/``rand_states``) and every slot-indexed bookkeeping list are
    kept at the same length (the number of active rows B); rows are appended on
    admission and removed (compacted) as they finish, so the tensors and lists
    always agree in slot order. ``empty()`` means no active rows."""

    __slots__ = ("jobs", "state", "out", "penalties", "rand_states",
                 "row_temperature", "row_top_k", "row_top_p",
                 "row_alpha_presence", "row_alpha_frequency", "row_alpha_decay",
                 "stop_states", "remaining", "text_buffers", "chunk_token_counts",
                 "permits")

    def __init__(self):
        self.clear()

    def clear(self):
        self.jobs = []
        self.state = None
        self.out = None
        self.penalties = None
        self.rand_states = None
        self.row_temperature = []
        self.row_top_k = []
        self.row_top_p = []
        self.row_alpha_presence = []
        self.row_alpha_frequency = []
        self.row_alpha_decay = []
        self.stop_states = []
        self.remaining = []
        self.text_buffers = []
        self.chunk_token_counts = []
        self.permits = []

    def empty(self):
        return not self.jobs


class _QueuedDynamicStream:
    """Stream for a job in the dynamic batch: drains the job's SSE chunk queue
    until the ``_END`` sentinel, appends the trailing ``data: [DONE]``, and
    surfaces any batch exception via the job's future. ``cancel()`` marks the job
    cancelled (the decoder force-drops the row and frees the slot)."""

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
            # Per-request stream ends with the same [DONE] the solo fused/
            # streaming chat path emits, so both paths format identically.
            yield "data: [DONE]\n\n"

    def cancel(self):
        job = self._job
        job.cancelled = True
        if not job.ended:
            job.ended = True
            job.queue.put_nowait(_END)

    async def aclose(self):
        self.cancel()


class DynamicBatchDecoder:
    """Scheduler that merges concurrent, HETEROGENEOUS chat requests into one
    shared multi-row decode with per-row sampling.

    Enabled (opt-in via ``RWKV_DYNAMIC_BATCH``): ``submit()`` enqueues jobs that
    a single persistent scheduler coroutine decodes in one shared batch; rows
    join/leave as requests arrive/finish. Disabled (default): ``submit()`` is a
    pure inline ``single_infer_stream`` proxy (byte-identical to today).

    Owns the prefill-admission permit for each admitted group (request_bsz = row
    count), releasing capacity per finished row so queued requests get admitted
    into freed budget immediately.
    """

    def __init__(self, engine, *, enabled=None, window_ms=None, max_bsz=None):
        self._engine = engine
        # enabled=None resolves from env at construction so tests pin the mode
        # deterministically regardless of environment.
        self._enabled = _dynamic_enabled(enabled)
        self._window = _window_seconds(window_ms)
        self._max_bsz = max_bsz
        self._pending = deque()  # FIFO of _DynamicJob, admission order
        self._running = _RunningBatch()
        self._wake = asyncio.Event()
        self._task = None
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
        """Stop the scheduler, ending any still-pending/in-flight jobs so no
        awaiting dynamic stream is orphaned on shutdown, and release any
        still-held prefill-admission permit capacity."""
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
            self._task = None
        # Abandon any jobs the task didn't get to (e.g. stopped right after
        # submit, before the loop body ran). Idempotent with the task's handler.
        self._abandon(
            RuntimeError("dynamic batch decoder shut down before generating")
        )
        await self._reset_batch(self._running)

    # -- request-facing API -------------------------------------------------

    def _build_inline_stream(self, prompt, max_tokens, temperature, stop_tokens,
                             chunk_size, top_k, top_p, alpha_presence,
                             alpha_frequency, alpha_decay):
        """Construct the inline solo ``_InlineFuseStream`` proxy for the
        default-off path: an async-iterable over ``single_infer_stream`` that
        forwards the request's OWN sampler controls and acquires/releases its
        own bsz=1 prefill-admission permit -- byte-identical to the pre-feature
        solo route."""
        return _InlineFuseStream(
            self._engine, prompt, max_tokens, temperature, stop_tokens, chunk_size,
            top_k=top_k, top_p=top_p, alpha_presence=alpha_presence,
            alpha_frequency=alpha_frequency, alpha_decay=alpha_decay,
        )

    async def submit(self, engine=None, prompt=None, max_tokens=512,
                     temperature=1.0, stop_tokens=("\nUser:",), chunk_size=32,
                     top_k=None, top_p=None, alpha_presence=None,
                     alpha_frequency=None, alpha_decay=None, cancel_token=None):
        """Enqueue a chat request and return its dynamic stream (async-iterable
        of SSE chunk strings; ``cancel()`` marks the row force-dropped).

        Disabled (default): returns an inline ``single_infer_stream`` proxy --
        byte-identical output and latency to the pre-feature path (the request's
        own sampler controls are forwarded). Enabled: the job is queued and the
        scheduler decodes it in the shared batch (a lone job head-fires into a
        bsz=1 decode with no gather latency; concurrent jobs share one batch,
        each row sampled with its OWN sampler controls). ANY request is
        accepted -- no homogeneity check; per-row sampling preserves each
        request's own sampler settings.
        """
        engine = engine or self._engine
        top_k = int(top_k) if top_k is not None else settings.fuse_sampler_top_k
        top_p = float(top_p) if top_p is not None else settings.fuse_sampler_top_p
        if alpha_presence is None:
            alpha_presence = settings.fuse_sampler_alpha_presence
        if alpha_frequency is None:
            alpha_frequency = settings.fuse_sampler_alpha_frequency
        if alpha_decay is None:
            alpha_decay = settings.fuse_sampler_alpha_decay

        if not self._enabled:
            return self._build_inline_stream(
                prompt, max_tokens, temperature, stop_tokens, chunk_size,
                top_k, top_p, alpha_presence, alpha_frequency, alpha_decay,
            )
        if self._task is None:
            self.start()
            if self._task is None:
                # start() was a no-op (feature became disabled); fall back to
                # the inline solo stream so this request still completes.
                return self._build_inline_stream(
                    prompt, max_tokens, temperature, stop_tokens, chunk_size,
                    top_k, top_p, alpha_presence, alpha_frequency, alpha_decay,
                )
            if not self._dead_warned:
                self._dead_warned = True
                logger.warning(
                    "[DynamicBatch] scheduler task was absent while enabled; "
                    "re-started it before serving request. If this recurs, a "
                    "prior scheduler crash was the cause."
                )
        loop = asyncio.get_running_loop()
        job = _DynamicJob(
            self._next_request_id, prompt, max_tokens, float(temperature),
            stop_tokens, chunk_size, top_k, top_p, alpha_presence,
            alpha_frequency, alpha_decay, cancel_token, loop,
        )
        self._next_request_id += 1
        self._pending.append(job)
        self._wake.set()
        return _QueuedDynamicStream(job)

    # -- scheduler internals ------------------------------------------------

    def _cap(self):
        """Hard cap on rows per shared batch, additionally bounded by the
        model's prefill-admission limit so acquire_prefill_permit(request_bsz=
        cap) never rejects."""
        cap = _max_bsz_value(self._max_bsz)
        limit = int(getattr(self._engine.model, "max_prefill_bsz_limit", 0) or 0)
        if limit > 0 and cap > limit:
            cap = limit
        return max(1, cap)

    def _take_pending(self, running):
        """Pop up to (cap - active rows) pending jobs for admission. Non-blocking.
        Returns [] when the batch is full or nothing is pending."""
        room = self._cap() - len(running.jobs)
        if room <= 0 or not self._pending:
            return []
        take = min(room, len(self._pending))
        return [self._pending.popleft() for _ in range(take)]

    async def _gather_first(self, running, loop):
        """Wait for the first pending job, then admit up to cap. Head-fire guard:
        a LONE job is admitted immediately with NO gather window; the window only
        opens when >=2 jobs are pending, mirroring fuse so an isolated request
        never pays gather latency."""
        while not self._pending:
            self._wake.clear()
            await self._wake.wait()
        if len(self._pending) >= 2:
            deadline = loop.time() + self._window
            self._wake.clear()
            while len(self._pending) >= 2:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    break
                try:
                    await asyncio.wait_for(self._wake.wait(), timeout=remaining)
                except asyncio.TimeoutError:
                    continue
                self._wake.clear()
        return self._take_pending(running)

    async def _admit(self, running, jobs):
        """Admit a group of ``jobs`` into the shared ``running`` batch: acquire a
        batch-layout prefill-admission permit, batch-prefill the new prompts into
        a new row block, allocate per-row penalties + per-request-seeded RNG, and
        grow the shared state/out/penalties/rand_states + slot lists to the new
        B. All GPU work in a single _offload_gpu unit."""
        engine = self._engine
        n = len(jobs)
        permit = await engine.acquire_prefill_permit(
            request_bsz=n, request_label=_DYNAMIC_LABEL
        )
        for j in jobs:
            j.permit = permit
        running.permits.append(permit)

        def _gpu_admit():
            with inference_deps.get_torch().no_grad():
                encoded = [engine.tokenizer.encode(j.prompt) for j in jobs]
                new_state = engine.model.generate_zero_state(n)
                new_out = engine._forward_batch_prompts_chunked(encoded, new_state)
                new_out = new_out.float()
                vocab = new_out.size(-1)
                device = new_out.device
                torch = inference_deps.get_torch()
                new_penalties = torch.zeros(
                    n, vocab, device=device, dtype=new_out.dtype
                )
                # Per-row-seeded RNG: one block per row, cat'd into the flat byte
                # tensor (matching setup_rand's flat layout sample_batch_per_row
                # reshapes to [B, rowsize]).
                new_rand = torch.cat(
                    [inference_deps.get_sample().setup_rand(j.seed, 1) for j in jobs],
                    dim=0,
                )
                if running.out is None:
                    return new_state, new_out, new_penalties, new_rand
                state = [
                    torch.cat([running.state[0], new_state[0]], dim=2),
                    torch.cat([running.state[1], new_state[1]], dim=1),
                    torch.cat([running.state[2], new_state[2]], dim=0),
                ]
                out = torch.cat([running.out, new_out], dim=0)
                penalties = torch.cat([running.penalties, new_penalties], dim=0)
                rand_states = torch.cat([running.rand_states, new_rand], dim=0)
                return state, out, penalties, rand_states

        new_state, new_out, new_penalties, new_rand = await engine._offload_gpu(
            _gpu_admit
        )
        running.state = new_state
        running.out = new_out
        running.penalties = new_penalties
        running.rand_states = new_rand
        running.jobs.extend(jobs)
        running.row_temperature.extend(float(j.temperature) for j in jobs)
        running.row_top_k.extend(int(j.top_k) for j in jobs)
        running.row_top_p.extend(float(j.top_p) for j in jobs)
        running.row_alpha_presence.extend(float(j.alpha_presence) for j in jobs)
        running.row_alpha_frequency.extend(float(j.alpha_frequency) for j in jobs)
        running.row_alpha_decay.extend(float(j.alpha_decay) for j in jobs)
        for j in jobs:
            running.stop_states.append(engine._create_stop_state(j.stop_tokens))
            running.remaining.append(int(j.max_tokens))
            running.text_buffers.append("")
            running.chunk_token_counts.append(0)

    async def _run_single_decode_step(self, running):
        """One shared decode step: sample a token per row (each with its own
        sampler controls + RNG + penalty state), feed the sampled tokens straight
        back through forward_batch, and pull tokens to CPU -- all in ONE
        _offload_gpu unit. Then, on the event loop between offload units, run the
        per-row stop-state/SSE bookkeeping and compact any rows that finished."""
        engine = self._engine
        prev_out = running.out

        def _gpu_decode_step():
            with inference_deps.get_torch().no_grad():
                new_tokens_t = inference_deps.get_sample_batch_per_row()(
                    running.out,
                    running.penalties,
                    running.rand_states,
                    running.row_temperature,
                    running.row_top_k,
                    running.row_top_p,
                    running.row_alpha_presence,
                    running.row_alpha_frequency,
                    running.row_alpha_decay,
                )
                new_out = engine.model.forward_batch(
                    new_tokens_t, running.state
                ).float()
                new_tokens = new_tokens_t.tolist()
                return new_out, new_tokens

        running.out, new_tokens = await engine._offload_gpu(_gpu_decode_step)
        del prev_out

        finished_any = False
        for slot, job in enumerate(running.jobs):
            if job.finished:
                continue
            if job.cancelled or (
                job.cancel_token is not None and job.cancel_token.is_cancelled()
            ):
                # Client disconnected: force-drop the row (free the slot + release
                # its capacity). The rest of the batch is unaffected.
                await self._drop_row(running, job)
                finished_any = True
                continue

            tok = new_tokens[slot][0]
            content, should_stop = engine._ingest_token_with_stop(
                running.stop_states[slot], tok
            )
            if content:
                running.text_buffers[slot] += content
            running.remaining[slot] -= 1
            maxed = running.remaining[slot] <= 0
            if should_stop or maxed:
                reason = "stop" if should_stop else "length"
                await self._finish_row(running, slot, job, reason)
                finished_any = True
                continue

            running.chunk_token_counts[slot] += 1
            if (
                running.chunk_token_counts[slot] >= job.chunk_size
                and running.text_buffers[slot]
            ):
                job.queue.put_nowait(self._content_chunk(running.text_buffers[slot]))
                running.text_buffers[slot] = ""
                running.chunk_token_counts[slot] = 0

        await asyncio.sleep(0)
        if finished_any:
            await self._compact(running)

    async def _finish_row(self, running, slot, job, reason):
        """End a row cleanly: flush its stop-state buffer, push the row's content
        + finish chunks to its stream, mark it finished, release its capacity,
        and push the stream terminal. Capacity is released BEFORE the stream
        terminal so a reader sees the row ended only after its budget is back."""
        engine = self._engine
        flushed = engine._flush_stop_state(running.stop_states[slot], final=True)
        if flushed:
            running.text_buffers[slot] += flushed
        if running.text_buffers[slot]:
            job.queue.put_nowait(self._content_chunk(running.text_buffers[slot]))
            running.text_buffers[slot] = ""
        job.queue.put_nowait(self._finish_chunk(reason))
        job.finished = True
        if job.permit is not None:
            try:
                await engine.release_prefill_capacity(
                    1, job.permit, request_label=_DYNAMIC_LABEL
                )
            except Exception:
                pass
        if not job.ended:
            job.ended = True
            job.queue.put_nowait(_END)

    async def _drop_row(self, running, job):
        """Force-drop a cancelled (disconnected) row: marked finished (so it is
        compacted out) and its capacity released. No content/finish chunks are
        delivered -- the client is gone."""
        job.finished = True
        if job.permit is not None:
            try:
                await self._engine.release_prefill_capacity(
                    1, job.permit, request_label=_DYNAMIC_LABEL
                )
            except Exception:
                pass
        # ensure the stream is ended
        if not job.ended:
            job.ended = True
            job.queue.put_nowait(_END)

    async def _compact(self, running):
        """Shrink the shared batch to the rows still generating: compact the GPU
        state/out/penalties/rand_states via engine._compact_active_rows and
        re-filter every slot-indexed bookkeeping list to the surviving slots."""
        engine = self._engine
        still_active = [
            slot for slot, job in enumerate(running.jobs) if not job.finished
        ]
        if len(still_active) == len(running.jobs):
            return
        if not still_active:
            running.jobs = []
            return
        active_indices = list(range(len(running.jobs)))

        def _gpu_compact():
            with inference_deps.get_torch().no_grad():
                return engine._compact_active_rows(
                    running.state,
                    running.out,
                    active_indices,
                    still_active,
                    penalties=running.penalties,
                    rand_states=running.rand_states,
                )

        new_state, new_out, _new_active, new_penalties, new_rand = (
            await engine._offload_gpu(_gpu_compact)
        )
        running.state = new_state
        running.out = new_out
        running.penalties = new_penalties
        running.rand_states = new_rand
        running.jobs = [running.jobs[s] for s in still_active]
        running.row_temperature = [running.row_temperature[s] for s in still_active]
        running.row_top_k = [running.row_top_k[s] for s in still_active]
        running.row_top_p = [running.row_top_p[s] for s in still_active]
        running.row_alpha_presence = [
            running.row_alpha_presence[s] for s in still_active
        ]
        running.row_alpha_frequency = [
            running.row_alpha_frequency[s] for s in still_active
        ]
        running.row_alpha_decay = [running.row_alpha_decay[s] for s in still_active]
        running.stop_states = [running.stop_states[s] for s in still_active]
        running.remaining = [running.remaining[s] for s in still_active]
        running.text_buffers = [running.text_buffers[s] for s in still_active]
        running.chunk_token_counts = [
            running.chunk_token_counts[s] for s in still_active
        ]

    async def _run_scheduler(self):
        loop = asyncio.get_running_loop()
        running = self._running
        try:
            while True:
                # -- wait for / gather the first admission when the batch is empty --
                if running.empty():
                    await self._reset_batch(running)
                    jobs = await self._gather_first(running, loop)
                    if not jobs:
                        continue
                    await self._admit(running, jobs)
                    if running.empty():
                        continue

                # -- decode loop, refilling between steps as rows finish/arrive --
                while not running.empty():
                    await self._run_single_decode_step(running)
                    # Re-admit newly-pending jobs into freed slots (bounded by cap).
                    ad = self._take_pending(running)
                    if ad:
                        await self._admit(running, ad)
        except asyncio.CancelledError:
            self._abandon(
                RuntimeError("dynamic batch decoder shut down before generating")
            )
            await self._reset_batch(running)
            raise
        except Exception as exc:
            logger.error("[DynamicBatch] scheduler failed: %s", exc)
            self._abandon(
                RuntimeError(f"dynamic batch decoder scheduler failed: {exc}")
            )
            await self._reset_batch(running)
            self._task = None

    def _abandon(self, exc):
        """On shutdown/crash: end every still-queued and in-flight job so no
        dynamic stream hangs, surfacing the exception via each job's future.
        Idempotent (jobs already ended/abandoned are skipped)."""
        running = self._running
        for job in list(self._pending) + list(running.jobs):
            if not job.future.done():
                job.future.set_exception(exc)
            if not job.ended:
                job.ended = True
                job.queue.put_nowait(_END)
            job.finished = True
        self._pending.clear()
        self._wake.set()

    async def _reset_batch(self, running):
        """Release any still-held prefill-admission permit capacity and clear the
        running batch back to an empty state. Per-row capacity is normally
        already released incrementally as rows finish; this gives back whatever
        remains (e.g. on shutdown mid-batch) and is a safe no-op otherwise."""
        for permit in running.permits:
            try:
                await self._engine.release_prefill_permit(
                    request_bsz=0, permit=permit, request_label=_DYNAMIC_LABEL
                )
            except Exception:
                pass
        running.permits.clear()
        running.clear()

    # -- SSE chunk builders --------------------------------------------------

    @staticmethod
    def _content_chunk(content):
        data = json.dumps(
            {"object": "chat.completion.chunk",
             "choices": [{"index": 0, "delta": {"content": content}}]},
            ensure_ascii=False,
        )
        return f"data: {data}\n\n"

    @staticmethod
    def _finish_chunk(reason):
        data = json.dumps(
            {"object": "chat.completion.chunk",
             "choices": [{"index": 0, "delta": {}, "finish_reason": reason}]},
            ensure_ascii=False,
        )
        return f"data: {data}\n\n"