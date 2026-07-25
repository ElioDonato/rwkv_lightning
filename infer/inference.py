import asyncio
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

from infer import inference_deps
from infer.batch_inference import BatchInferenceMixin
from infer.big_batch import BigBatchMixin
from infer.cancellation import InferenceCancelled, PrefillBszLimitExceeded
from infer.inference_utils import InferenceUtilsMixin


class InferenceEngine(
    InferenceUtilsMixin,
    BatchInferenceMixin,
    BigBatchMixin,
):
    def __init__(self, model, tokenizer, args, rocm_flag):
        self.model = model
        self.tokenizer = tokenizer
        self.args = args
        self.rocm_flag = rocm_flag
        self.model_lock = Lock()
        self.executor = ThreadPoolExecutor(
            max_workers=128, thread_name_prefix="model_inference"
        )
        self._prefill_queue = deque()
        self._prefill_reserved_bsz = 0
        self._prefill_next_ticket = 0
        self._prefill_condition = None

    def _get_prefill_condition(self):
        if self._prefill_condition is None:
            self._prefill_condition = asyncio.Condition()
        return self._prefill_condition

    async def acquire_prefill_permit(
        self, request_bsz: int, request_label: str = "", cancel_token=None
    ):
        request_bsz = max(1, int(request_bsz))
        max_prefill_bsz_limit = int(
            getattr(
                self.model,
                "max_prefill_bsz_limit",
                getattr(self.model, "max_prefill_bsz", request_bsz),
            )
        )
        if request_bsz > max_prefill_bsz_limit:
            print(
                f"[PrefillQueue] rejected path={request_label} "
                f"request_bsz={request_bsz} max_prefill_bsz_limit={max_prefill_bsz_limit}"
            )
            raise PrefillBszLimitExceeded(request_bsz, max_prefill_bsz_limit)

        condition = self._get_prefill_condition()

        async with condition:
            ticket = self._prefill_next_ticket
            self._prefill_next_ticket += 1
            self._prefill_queue.append(ticket)
            queued_logged = False

            try:
                while True:
                    if cancel_token is not None and cancel_token.is_cancelled():
                        raise InferenceCancelled("request disconnected while queued")

                    is_turn = self._prefill_queue and self._prefill_queue[0] == ticket
                    if is_turn and hasattr(self.model, "refresh_max_prefill_bsz"):
                        current_limit = self.model.refresh_max_prefill_bsz()
                    else:
                        current_limit = getattr(self.model, "max_prefill_bsz", request_bsz)
                    current_limit = min(int(current_limit), max_prefill_bsz_limit)
                    available_bsz = max(0, int(current_limit) - self._prefill_reserved_bsz)

                    if is_turn and request_bsz <= available_bsz:
                        self._prefill_reserved_bsz += request_bsz
                        self._prefill_queue.popleft()
                        condition.notify_all()
                        print(
                            f"[PrefillQueue] admitted ticket={ticket} path={request_label} "
                            f"request_bsz={request_bsz} reserved_bsz={self._prefill_reserved_bsz} "
                            f"max_prefill_bsz={current_limit}"
                        )
                        # `outstanding_bsz` is a single-element mutable box (not a bare
                        # int) so release_prefill_permit and release_prefill_capacity
                        # (below) can both read/update "how much of this permit's
                        # reservation is still held" without the caller needing to pass
                        # the current value back in -- a long-lived streaming request
                        # (e.g. /big_batch/completions) can release_prefill_capacity()
                        # incrementally as individual rows finish and get compacted out
                        # of its GPU batch, and the final release_prefill_permit() at
                        # request end only needs to give back whatever's left in the box,
                        # not the original full amount (which would double-release and
                        # corrupt the shared accounting).
                        return {
                            "ticket": ticket,
                            "request_bsz": request_bsz,
                            "max_prefill_bsz": int(current_limit),
                            "outstanding_bsz": [request_bsz],
                        }

                    if not queued_logged:
                        ahead = sum(1 for queued_ticket in self._prefill_queue if queued_ticket < ticket)
                        print(
                            f"[PrefillQueue] queued ticket={ticket} path={request_label} "
                            f"request_bsz={request_bsz} requests_ahead={ahead} "
                            f"reserved_bsz={self._prefill_reserved_bsz} max_prefill_bsz={current_limit}"
                        )
                        queued_logged = True

                    try:
                        await asyncio.wait_for(condition.wait(), timeout=0.1)
                    except asyncio.TimeoutError:
                        pass
            except BaseException:
                if ticket in self._prefill_queue:
                    self._prefill_queue.remove(ticket)
                    condition.notify_all()
                    print(
                        f"[PrefillQueue] removed ticket={ticket} path={request_label} "
                        f"request_bsz={request_bsz}"
                    )
                raise

    async def release_prefill_capacity(
        self, amount: int, permit: dict, request_label: str = ""
    ):
        """Give back part of an already-admitted permit's reservation before the
        request itself finishes, so other queued requests can be admitted into the
        freed capacity immediately instead of waiting for this request's full
        original bsz to be released all at once at the end.

        Intended for long-running batched streams (e.g. /big_batch/completions)
        that compact finished rows out of their GPU batch mid-request: each time a
        row finishes, the caller can release_prefill_capacity(1, permit) for that
        row instead of holding its slot in the shared admission budget until every
        other row in the same request also finishes.

        `amount` is clamped to whatever `permit["outstanding_bsz"]` still holds, so
        calling this more than the request's original bsz (or after the final
        release_prefill_permit already zeroed it out) is a safe no-op rather than
        corrupting the shared counter.
        """
        amount = max(0, int(amount))
        if amount == 0:
            return
        condition = self._get_prefill_condition()
        async with condition:
            outstanding = permit.get("outstanding_bsz")
            if not outstanding:
                return
            actual = min(amount, outstanding[0])
            if actual <= 0:
                return
            outstanding[0] -= actual
            self._prefill_reserved_bsz = max(0, self._prefill_reserved_bsz - actual)
            print(
                f"[PrefillQueue] partial-released ticket={permit.get('ticket')} "
                f"path={request_label} amount={actual} "
                f"outstanding_bsz={outstanding[0]} reserved_bsz={self._prefill_reserved_bsz}"
            )
            condition.notify_all()

    async def release_prefill_permit(
        self,
        request_bsz: int,
        request_label: str = "",
        ticket: int | None = None,
        permit: dict | None = None,
    ):
        # If the caller already made incremental release_prefill_capacity() calls
        # against this permit (e.g. compaction released some rows mid-stream), only
        # release whatever's left outstanding -- releasing the original full
        # request_bsz again here would double-release the already-freed portion and
        # let _prefill_reserved_bsz drift below the true outstanding total, silently
        # over-admitting future requests. Callers that never partially released
        # (the common case, and the only case before this method gained the
        # `permit` param) still pass a plain request_bsz and get the exact prior
        # behavior.
        if permit is not None and permit.get("outstanding_bsz"):
            request_bsz = permit["outstanding_bsz"][0]
            permit["outstanding_bsz"][0] = 0
        request_bsz = max(0, int(request_bsz))
        if request_bsz == 0:
            return
        condition = self._get_prefill_condition()

        async with condition:
            self._prefill_reserved_bsz = max(0, self._prefill_reserved_bsz - request_bsz)
            current_limit = (
                self.model.refresh_max_prefill_bsz()
                if hasattr(self.model, "refresh_max_prefill_bsz")
                else request_bsz
            )
            max_prefill_bsz_limit = int(
                getattr(
                    self.model,
                    "max_prefill_bsz_limit",
                    getattr(self.model, "max_prefill_bsz", request_bsz),
                )
            )
            current_limit = min(int(current_limit), max_prefill_bsz_limit)
            print(
                f"[PrefillQueue] released ticket={ticket} path={request_label} "
                f"request_bsz={request_bsz} reserved_bsz={self._prefill_reserved_bsz} "
                f"max_prefill_bsz={current_limit}"
            )
            condition.notify_all()

    def shutdown(self):
        self.executor.shutdown(wait=False)
