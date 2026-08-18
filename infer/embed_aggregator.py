"""
Server-side CONTINUOUS BATCHING for the /embedding and /v1/embeddings endpoints.

Today each concurrent /embedding request calls ``embed_texts`` synchronously
inline on the event-loop thread, one request at a time, so the GPU sits
under-occupied while N requests serialize. This module is the throughput lever:
an opt-in, default-OFF admission/aggregation worker that collects N concurrent
requests into ONE shared ``embed_texts`` call over the concatenated texts,
splitting the per-text vectors back per-request.

All tunable knobs for this feature live in the central ``settings`` module
(``settings.embed_aggregate``, ``settings.embed_aggregate_window_ms``,
``settings.embed_hard_ceiling``, backed by the ``RWKV_EMBED_AGGREGATE``,
``RWKV_EMBED_AGGREGATE_WINDOW_MS``, and ``RWKV_EMBED_HARD_CEILING``
environment variables respectively) rather than as module-local constants here;
explicit constructor overrides (used by tests) always win over those.

Design
------
* ``EmbedAggregator`` owns a single scheduler coroutine and a FIFO queue of
  pending embedding requests. Each request enqueues an ``_EmbedJob`` carrying
  its texts and an asyncio future, then awaits the future.
* The scheduler drains the queue in small timing windows
  (``RWKV_EMBED_AGGREGATE_WINDOW_MS`` via ``settings.embed_aggregate_window_ms``):
  it fires EARLY the instant the batching cap is reached, otherwise when the
  window elapses. Requests that don't fit under the cap (a request whose text
  count would exceed the remaining batch budget) stay queued and are admitted
  on the next window -- bounded, never one giant unbounded batch.
* The batching cap is taken from the model's ``max_prefill_bsz`` (the same
  admission budget big_batch's prefill admission uses). When
  ``max_prefill_bsz > 0`` the concatenated batch is capped at exactly that so
  the batch never exceeds the shared VRAM estimate. ``max_prefill_bsz=0`` means
  no model-level cap, so ``embedding.embed_texts`` falls back to its own hard
  ceiling (``settings.embed_hard_ceiling``, backed by ``RWKV_EMBED_HARD_CEILING``,
  read from the central settings module); here it triggers the same hard ceiling
  on the aggregated batch instead of unbounded accumulation, so a whole window
  burst is never packed into one un-isolated VRAM spike against a co-resident
  chat server on the same GPU. The cap never rejects a pending HEAD job: one
  that alone exceeds the cap is still admitted and run. In capped mode
  ``embed_texts``' own sub-batching keeps an oversized HEAD within its
  model-level budget; in no-cap mode the aggregator sub-chunks the lone job's
  texts itself (see ``_embed_texts_bounded``) -- a redundant-but-consistent
  double bound, since embedding.py already self-caps at the same ceiling -- so
  no single ``embed_texts`` call exceeds the hard ceiling either way.

After a scheduler crash (a non-cancel failure in the scheduler logic), the
scheduler nulls ``self._task`` so a later ``submit()`` re-starts it via the
idempotent ``start()`` and warns once -- the feature self-heals instead of
silently degrading to inline serial embedding with no re-start.

Exactness
---------
Aggregation is exactly equivalent to running each request's texts separately
for one reason: the unit of work is ``infer.embedding.embed_texts`` acting on
the CONCATENATED text list, and Mod B already proved that batched
``embed_texts`` is numerically identical to embedding each text alone (both use
the exact ``RWKV_x070_TMix_seq_batch`` / ``RWKV_x070_CMix_seq_batch`` prefill
kernels over equal-length, zero-padding-free chunks; per-row state is computed
independently). Concatenating K requests' texts into one list and calling
``embed_texts`` once therefore yields, per text, the identical vector that
separate per-request calls would produce -- and the default-off inline path
(see below) is byte-identical to today because it never touches the queue.

Per-request failure isolation
-----------------------------
``embed_texts`` is atomic over its whole input: a single forward on the
concatenated batch either succeeds whole or raises whole, so per-request
*within-a-window* isolation at the text level is not achievable without
defeating the shared-batch point. Isolation is provided at the edges instead:
a raised window fails that batch's jobs individually (each awaiting route gets
its own exception -> the existing 500 handler), the scheduler SURVIVES and
keeps draining later windows, and on scheduler shutdown any queued-not-yet-run
jobs are resolved (exceptions) rather than left orphaned. The exactness
invariant (see ``infer/embedding.py``) is preserved because we never reimplement
the per-text math.

Default-OFF
-----------
Matching Mod A (``RWKV_ASYNC_FORWARD``) and Mod B precedent. When off
(``RWKV_EMBED_AGGREGATE`` unset/0, the default), ``submit()`` calls
``embed_texts`` inline on the calling thread exactly as the pre-feature route
did -- no queue, no aggregation, byte-identical result and latency. The
aggregator object may be created eagerly (it spawns no task while disabled).
"""

import asyncio
import logging
from collections import deque

from settings import settings
from infer.memo import MISS, BoundedLRU

logger = logging.getLogger("infer.embed_aggregator")

from infer.embedding import embed_texts


def _aggregate_enabled(enabled=None) -> bool:
    """Resolve the opt-in, honoring an explicit override (used by tests)."""
    if enabled is not None:
        return bool(enabled)
    return bool(settings.embed_aggregate)


def _window_seconds(window_ms=None) -> float:
    """Collect window in seconds (>= 0). Explicit override wins over settings."""
    if window_ms is not None:
        return max(0.0, int(window_ms)) / 1000.0
    return max(0.0, float(settings.embed_aggregate_window_ms)) / 1000.0


class _EmbedJob:
    """One admitted embedding request: its texts, its completion future, and a
    stable request id for logging. The scheduler slices the concatenated result
    back to this request by ``n_texts``."""

    __slots__ = ("request_id", "texts", "n_texts", "future")

    def __init__(self, request_id, texts, loop):
        self.request_id = request_id
        self.texts = texts
        self.n_texts = len(texts)
        self.future = loop.create_future()


class EmbedAggregator:
    """Bounded admission/aggregation worker for the embed endpoints.

    Enabled (opt-in via RWKV_EMBED_AGGREGATE): concurrent ``submit()`` calls
    enqueue and are drained in shared batches by one scheduler coroutine.
    Disabled (default): ``submit()`` is a pure inline ``embed_texts`` call.

    Not thread-safe by design -- it is an asyncio component driven from the
    single event-loop thread, exactly where the embed endpoint already runs.
    """

    def __init__(self, model, tokenizer, *, enabled=None, window_ms=None, max_bsz=None,
                 cache_namespace_=None):
        self._model = model
        self._tokenizer = tokenizer
        # enabled=None resolves from settings at construction so tests pin the
        # mode deterministically regardless of environment.
        self._enabled = _aggregate_enabled(enabled)
        self._window = _window_seconds(window_ms)
        # max_bsz=None resolves from the model's max_prefill_bsz (0 == no cap);
        # an explicit value overrides for deterministic tests.
        self._max_bsz = max_bsz
        # A1: checkpoint fingerprint (from `cache_namespace(slot)`, NEVER
        # None-for-default) gating the embedding output LRU. When the cache is
        # enabled (RWKV_EMBED_CACHE) but no fingerprint is bound, the cache is a
        # no-op -- so a stale vector can never be served after a checkpoint swap.
        self._cache_ns = cache_namespace_
        self._cache = BoundedLRU(settings.embed_cache_capacity) if settings.embed_cache else None
        self._pending = deque()  # of _EmbedJob, FIFO admission order
        self._wake = asyncio.Event()
        self._task = None
        # Jobs captured into this window's batch but not yet embedded. Tracked
        # separately from ``_pending`` so a shutdown that cancels the scheduler
        # MID-window (while it awaits more arrivals) can resolve them cleanly
        # too -- they were popped off ``_pending`` the moment they were admitted
        # to the in-flight batch, so failing only ``_pending`` would orphan them.
        self._inflight_batch = None
        self._next_request_id = 0
        # One-time "scheduler was dead, re-started" warning (Finding 3): after a
        # crash self._fail_pending shuts the scheduler off (self._task=None) and
        # the submit() re-start path warns at most once, not on every request.
        self._dead_warned = False

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
        """Spawn the scheduler task. No-op when disabled or already started."""
        if not self._enabled or self._task is not None:
            return
        self._task = asyncio.create_task(self._run_scheduler())

    async def stop(self):
        """Stop the scheduler, resolving (failing) any still-queued jobs so no
        awaiting caller's future is left orphaned on shutdown."""
        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except (asyncio.CancelledError, Exception):
            pass
        self._task = None

    # -- request-facing API -------------------------------------------------

    async def submit(self, texts):
        """Embed ``texts`` (a non-empty list of strings) -> list of [n_embd].

        Disabled (default): runs ``embed_texts`` inline and returns -- byte-
        identical to the pre-feature route. Enabled: enqueues the request and
        returns once the scheduler has embedded it in a shared batch.
        """
        if not self._enabled:
            # Default-off path: never hold a future that nothing will resolve.
            return self._embed_with_cache(texts)
        if self._task is None:
            # Scheduler task missing while aggregation is on: either it was never
            # started (route init should have done so) or it DIED from a
            # non-cancel crash (the except handler nulls self._task). Rather than
            # silently degrade to inline serial embedding (correct results, but
            # the feature silently off with no re-start), re-invoke the
            # idempotent start() so this request is served by a live scheduler.
            self.start()
            if self._task is None:
                # start() was still a no-op (e.g. aggregation became disabled);
                # fall back to the inline embed so this request still completes.
                return self._embed_with_cache(texts)
            if not self._dead_warned:
                self._dead_warned = True
                logger.warning(
                    "[EmbedAggregate] scheduler task was absent while enabled; "
                    "re-started it before serving request -- batching restored. "
                    "If this recurs, a prior scheduler crash was the cause."
                )
        loop = asyncio.get_running_loop()
        job = _EmbedJob(self._next_request_id, texts, loop)
        self._next_request_id += 1
        self._pending.append(job)
        self._wake.set()
        return await job.future

    # -- scheduler internals ------------------------------------------------

    def _raw_bsz_cap(self) -> int:
        """The single source of the batching-cap value as an int: the explicit
        ``max_bsz`` override when set, else the model's live
        ``max_prefill_bsz`` (read from ``getattr(..., 0) or 0``). May be <= 0 to
        mean "no cap" -- both callers decide how to collapse that themselves.
        This is the one place the raw value is resolved, so ``_cap()`` (which
        maps <= 0 to the hard ceiling) and ``_no_cap_mode()`` (which only asks
        whether it is absent) can never drift on which source lives where."""
        if self._max_bsz is not None:
            return int(self._max_bsz)
        return int(getattr(self._model, "max_prefill_bsz", 0) or 0)

    def _cap(self):
        """Batching cap: the model's max_prefill_bsz when > 0, else a sane hard
        ceiling (``settings.embed_hard_ceiling``). Always returns int >= 1, so
        the aggregated batch is never VRAM-unbounded even in the model's "no cap"
        (max_prefill_bsz<=0) mode -- a whole window burst never collapses into
        ONE un-isolated embed_texts call. An explicit ``max_bsz`` override
        (tests) wins unconditionally, and an override <= 0 means "no cap" too,
        so it maps to the hard ceiling just like the model's 0."""
        cap = self._raw_bsz_cap()
        # ANY cap <= 0 -- the model's "no cap" OR an explicit override <= 0 --
        # collapses to the hard ceiling. _cap() must ALWAYS yield a positive
        # int: a None/<=0 cap would hand _fill_batch / the window drain an
        # unbounded aggregate window (never breaks, never bounded).
        if cap <= 0:
            cap = settings.embed_hard_ceiling
        return cap

    def _no_cap_mode(self) -> bool:
        """True when the batching-cap source is genuinely absent: the model's
        ``max_prefill_bsz <= 0`` and (if overridden) the override itself is
        ``<= 0``. Here ``embed_texts`` has no model-level ``max_prefill_bsz``
        to sub-batch at (it only self-caps each sub-batch at
        ``settings.embed_hard_ceiling``), so ``_run_batch`` sub-chunks a
        lone job's own texts itself to keep every shared-batch call within the
        aggregation ceiling; capped mode already has embed_texts' own
        ``max_prefill_bsz`` sub-batching to lean on."""
        return self._raw_bsz_cap() <= 0

    def _fill_batch(self, batch, cap):
        """Pop pending jobs into ``batch`` (FIFO) up to the total-text cap.
        Returns the number of texts admitted. The FIRST (HEAD) pending job is
        ALWAYS admitted, even when it alone would exceed the remaining budget --
        ``embed_texts`` sub-batches oversized input itself, so an oversized HEAD
        runs as its own batch rather than wedging the queue forever; jobs BEHIND
        an admitted one are deferred to a later window while they don't fit. cap
        is None to mean unbounded-off (unreachable in practice: ``_cap()``
        always yields a ceil now). Never admits a zero-text job (routes validate
        non-empty input before reaching here)."""
        n = sum(j.n_texts for j in batch)
        while self._pending:
            job = self._pending[0]
            # n > 0 means at least one job is already in this batch, so the
            # cap is enforced only off the HEAD: the HEAD itself (n == 0) is
            # always admitted regardless of exceeding the residual budget.
            if cap is not None and n > 0 and n + job.n_texts > cap:
                break
            self._pending.popleft()
            batch.append(job)
            n += job.n_texts
            if cap is not None and n >= cap:
                break
        return n

    async def _run_scheduler(self):
        loop = asyncio.get_running_loop()
        try:
            while True:
                # Wait until at least one request is pending.
                while not self._pending:
                    self._wake.clear()
                    await self._wake.wait()

                cap = self._cap()
                self._inflight_batch = []
                n = self._fill_batch(self._inflight_batch, cap)
                # n >= 1: the HEAD job is always admitted (its n_texts >= 1),
                # so with pending work a window must never admit zero texts. If
                # it somehow does, fail loudly rather than spin an empty
                # _run_batch([]) silently forever (the scheduler wedges: queue
                # never drains, every callers' future hangs).
                if n == 0 and self._pending:
                    logger.error(
                        "[EmbedAggregate] 0-admission window with %d pending; "
                        "refusing to spin empty batches",
                        len(self._pending),
                    )
                    raise RuntimeError("embed aggregator admitted no jobs in a busy window")

                # ALWAYS hold the window open for `self._window` (or until the
                # cap fills), so a burst of requests arriving just after the
                # first one joins this same shared batch. Firing EARLY happens
                # only on cap-full; otherwise we drain for the full window so
                # the aggregation actually gathers concurrent arrivals (firing
                # the instant the queue empties would defeat batching whenever a
                # single request's turn came before its siblings had enqueued).
                # An isolated request therefore pays `self._window` of latency;
                # that is the documented aggregation tradeoff (bounded, never an
                # unbounded batch -- requests left over win the next window).
                deadline = loop.time() + self._window
                self._wake.clear()
                while cap is None or n < cap:
                    remaining = deadline - loop.time()
                    if remaining <= 0:
                        break
                    try:
                        await asyncio.wait_for(self._wake.wait(), timeout=remaining)
                    except asyncio.TimeoutError:
                        continue  # re-evaluate; the deadline may have just passed
                    self._wake.clear()
                    n = self._fill_batch(self._inflight_batch, cap)

                batch = self._inflight_batch
                # Synchronous (no await) so the current batch always completes
                # and resolves its futures: a shutdown cancel is only delivered
                # here between windows, never mid-batch.
                self._inflight_batch = None
                self._run_batch(batch, cap)
        except asyncio.CancelledError:
            # Scheduled shutdown: resolve every job that has not yet been run --
            # both the still-queued ``_pending`` jobs and any jobs already
            # admitted into an in-flight window batch (a cancel can land mid-
            # window, while the scheduler awaits more arrivals) -- so no awaiting
            # caller's future ever hangs. Jobs replication is impossible: a job
            # is either in ``_pending`` or in ``_inflight_batch``, never both.
            self._fail_pending(
                RuntimeError("embed aggregator shut down before embedding")
            )
            raise
        except Exception as exc:
            # A NON-cancel failure in ``_cap()``/``_fill_batch``/window logic
            # must not orphan awaiting futures: otherwise every ``submit()``
            # waiting on a ``_pending``/``_inflight_batch`` future would hang
            # forever and ``self._task`` would still point at this now-dead task
            # (so ``start()`` / ``stop()`` would be wedged too). Log, fail all
            # pending/in-flight jobs with a per-request exception (-> 500s), free
            # ``self._task`` for a clean re-entry, then re-raise.
            logger.error(
                "[EmbedAggregate] scheduler failed, failing pending request(s): %s",
                exc,
            )
            self._fail_pending(
                RuntimeError(f"embed aggregator scheduler failed: {exc}")
            )
            self._task = None
            # Deliberately do NOT re-raise: with ``self._task`` already nulled
            # there is nothing left to retrieve, so re-raising would just leave
            # an "exception was never retrieved" warning in the event loop for
            # nobody to read. The failure was already surfaced to every waiting
            # caller via their futures (-> per-request 500s), and the next
            # ``submit()`` re-starts the scheduler via ``start()`` instead of
            # silently degrading to inline serial embedding.

    def _run_batch(self, batch, cap):
        """Embed the admitted jobs' texts (sub-chunked per ``_embed_texts_bounded``
        so no single ``embed_texts`` call exceeds the cap), then split the
        per-text vectors back per-request by each job's own text count and
        resolve its future. Synchronous (no await): a shutdown cancel lands
        between batches, never mid-batch.

        ``embed_texts`` is atomic over its input, so a failure fails the whole
        concatenated batch; each job in it gets its OWN copy of the exception
        (-> per-request 500s), and the scheduler loop keeps running for future
        windows -- one bad window cannot sink the aggregator.
        """
        if not batch:
            return
        texts = []
        for job in batch:
            texts.extend(job.texts)
        try:
            # Stays on the event-loop thread, exactly as the pre-feature route
            # ran embed_texts (Mod A's offload seam does not cover the embed
            # path; this preserves the existing single-thread-CUDA posture).
            vectors = self._embed_texts_bounded(texts, cap)
        except BaseException as exc:
            logger.error("[EmbedAggregate] window of %d request(s) failed: %s",
                         len(batch), exc)
            for job in batch:
                if not job.future.done():
                    job.future.set_exception(exc)
            return
        # vectors is in concatenated FIFO order; hand each job its own slice.
        pos = 0
        for job in batch:
            if not job.future.done():
                job.future.set_result(vectors[pos:pos + job.n_texts])
            pos += job.n_texts

    def _embed_texts_bounded(self, texts, cap):
        """Embed ``texts`` (the concatenated window) into per-text vectors,
        feeding each ``embed_texts`` call at most ``cap`` texts, and return all
        vectors in input order.

        In no-cap mode (``_no_cap_mode``) ``embed_texts`` runs on the hard
        ceiling (embedding.py self-caps each sub-batch at ``settings.embed_hard_ceiling``),
        so this module's sub-chunking of the aggregated window -- feeding each
        call at most ``cap`` texts -- is a redundant-but-consistent double bound
        that keeps the shared-batch VRAM spike within the aggregation ceiling
        even if embedding.py's ceiling were ever loosened. In capped mode
        ``embed_texts`` already sub-batches internally at ``max_prefill_bsz``,
        so the window is fed whole (ONE call), preserving the existing
        shared-batch composition exactly.

        Exactness / split-accounting are unaffected: every sub-slice runs the
        exact same per-text math, so concatenating the per-slice results is
        byte-identical to one call over the whole list, and ``_run_batch``
        splits that concatenated result back per request by text count as if a
        single call had been made."""
        if not self._no_cap_mode():
            return self._embed_with_cache(texts)
        out = []
        for i in range(0, len(texts), cap):
            out.extend(self._embed_with_cache(texts[i:i + cap]))
        return out

    def _embed_with_cache(self, texts):
        """Embed ``texts`` into per-text vectors, honoring the opt-in embedding
        output LRU (A1, RWKV_EMBED_CACHE):

        * key = ``(cache_namespace_, text, normalize=True)`` -- the checkpoint
          fingerprint (``cache_namespace(slot)``, never None-for-default) so a
          runtime model reload / checkpoint swap starts the cache cold and can
          never return a stale vector.
        * Look up every text; embed ONLY the misses (each distinct miss text
          once -- within-batch dedupe); splice the cached vectors back in input
          order; store the fresh results.
        * Disabled (default), or no fingerprint bound: exactly ``embed_texts``.

        Embedding is a deterministic pure function of (model, text), so a cached
        vector is byte-identical to recomputing it; splicing preserves the exact
        per-text ordering ``embed_texts`` would return."""
        if self._cache is None or self._cache_ns is None:
            return embed_texts(self._model, self._tokenizer, texts, normalize=True)

        keys = [(self._cache_ns, t, True) for t in texts]
        cached = [None] * len(texts)
        miss_idx = []
        for i, key in enumerate(keys):
            value = self._cache.get(key)
            if value is MISS:
                miss_idx.append(i)
            else:
                # Return a FRESH list on a hit (like CachedTokenizer.encode),
                # so a downstream caller mutating the returned vector can never
                # corrupt the shared cache entry.
                cached[i] = list(value)
        if miss_idx:
            # embed each distinct miss text exactly once, then fan back out
            # (dict.fromkeys dedupes while preserving first-occurrence order).
            uniq = {t: i for i, t in enumerate(dict.fromkeys(texts[i] for i in miss_idx))}
            fresh = embed_texts(self._model, self._tokenizer, list(uniq), normalize=True)
            for i in miss_idx:
                t = texts[i]
                # A FRESH copy per row: the returned vector is never the object we
                # store, so a downstream caller mutating its result can neither
                # corrupt the shared cache entry nor alias another duplicate row.
                out = list(fresh[uniq[t]])
                cached[i] = out
                self._cache.put((self._cache_ns, t, True), list(out))
        return cached

    def _fail_pending(self, exc):
        # Fail both the still-queued jobs and any admitted-but-not-yet-embedded
        # in-flight batch, so shutdown/cancel never orphans a caller's future.
        for job in list(self._pending) + (self._inflight_batch or []):
            if not job.future.done():
                job.future.set_exception(exc)
        self._pending.clear()
        self._inflight_batch = None
        self._wake.set()