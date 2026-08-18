import asyncio
import json
import random

from infer import inference_deps

# Single-sourced default for the big_batch gumbel-temp sampler. The fused
# big_batch path only uses `temperature` (it ignores top_k/top_p/alpha); the
# per-route top_k/top_p/alpha defaults for the non-gumbel V1/V2 samplers live
# in batch_inference.py.
_DEFAULT_TEMPERATURE = 1.0

# Per-row repetition-penalty sampler "no-op" defaults (only engaged when the
# caller opts into per-row control via per_row_*). With these values the
# batch_sampling_repetition_temperature_topk_topp op reduces to softmax(·/temp)
# inverse-CDF sampling -- the per-row generalization of the temperature-only
# default. penalties never accumulate because alpha_presence==alpha_frequency
# ==0, and top_k=0/top_p=1.0 leave the full distribution in place.
_DEFAULT_TOP_K = 0
_DEFAULT_TOP_P = 1.0
_DEFAULT_ALPHA_PRESENCE = 0.0
_DEFAULT_ALPHA_FREQUENCY = 0.0
_DEFAULT_ALPHA_DECAY = 1.0


def _resolve_per_row(list_or_none, scalar, n):
    """Return a length-``n`` list of per-row sampler scalars: the caller's
    per-row list when one is provided, else ``scalar`` replicated for every
    row (slot-indexed, re-filtered on compaction in big_batch_stream)."""
    if list_or_none is None:
        return [float(scalar)] * n
    return [float(x) for x in list_or_none]


class BigBatchMixin:
    async def big_batch_stream(
        self,
        prompts,
        max_length=512,
        temperature=_DEFAULT_TEMPERATURE,
        top_k=_DEFAULT_TOP_K,
        top_p=_DEFAULT_TOP_P,
        alpha_presence=_DEFAULT_ALPHA_PRESENCE,
        alpha_frequency=_DEFAULT_ALPHA_FREQUENCY,
        alpha_decay=_DEFAULT_ALPHA_DECAY,
        stop_tokens=("\nUser:",),
        chunk_size=32,
        cancel_token=None,
        permit_box=None,
        per_row_temperatures=None,
        per_row_top_k=None,
        per_row_top_p=None,
        per_row_alpha_presence=None,
        per_row_alpha_frequency=None,
        per_row_alpha_decay=None,
    ):
        # permit_box, if given, is a single-element mutable list populated by the
        # caller (see API_servers.router.common.prefill_sse_response's on_permit
        # callback) with this request's admission permit once it's been granted
        # -- the permit doesn't exist yet when this generator object is first
        # created (admission happens lazily on first iteration), so it can't be
        # passed in directly as a plain argument. When present, each row that
        # finishes and gets compacted out of the GPU batch below immediately
        # releases its one slot of reserved prefill-admission capacity back to
        # the shared pool via engine.release_prefill_capacity(), instead of
        # this whole request holding its *original* full-batch reservation
        # until every row finishes. This lets other queued requests get
        # admitted into the freed capacity as soon as it's actually free,
        # rather than only once this request's slowest row also completes.
        batch_size = len(prompts)
        state = None
        encoded_prompts = None
        out = None
        penalties = None
        rand_states = None
        finished = None
        new_tokens_tensor = None
        new_tokens = None

        # Per-row sampling (Phase 4): when the caller opts in by passing any
        # per_row_* list, each fused row is sampled with ITS OWN
        # temperature/top_k/top_p/alpha_presence/alpha_frequency/alpha_decay and
        # per-row penalty state via sample_batch_per_row (the per-row application
        # of batch_sampling_repetition_temperature_topk_topp). Note the row
        # lists and penalty/rand-state tensors are ALL slot-indexed and shrink
        # together with `state`/`out` in _compact_active_rows as rows finish.
        # When NO per_row_* is given (the default -- nothing opts into per-row
        # control), the decode samples the whole batch with the existing
        # temperature-only sampler_gumbel_batch exactly as before: byte-identical
        # default behavior, and no penalty/RNG state is even allocated.
        per_row_mode = any(
            p is not None
            for p in (
                per_row_temperatures,
                per_row_top_k,
                per_row_top_p,
                per_row_alpha_presence,
                per_row_alpha_frequency,
                per_row_alpha_decay,
            )
        )
        row_temperatures = _resolve_per_row(per_row_temperatures, temperature, batch_size)
        row_top_k = _resolve_per_row(per_row_top_k, top_k, batch_size)
        row_top_p = _resolve_per_row(per_row_top_p, top_p, batch_size)
        row_alpha_presence = _resolve_per_row(
            per_row_alpha_presence, alpha_presence, batch_size
        )
        row_alpha_frequency = _resolve_per_row(
            per_row_alpha_frequency, alpha_frequency, batch_size
        )
        row_alpha_decay = _resolve_per_row(per_row_alpha_decay, alpha_decay, batch_size)

        try:
            # NOTE: must be no_grad(), not inference_mode(). inference_mode()'s
            # guard is thread-local (not coroutine-local), so under concurrent
            # asyncio requests sharing this event-loop thread, one request's
            # `with` block can exit while another is still inside its own,
            # desyncing the ambient mode mid-flight for the still-running one.
            # Tensors created while the guard *was* active stay permanently
            # tagged as inference tensors even after the desync, so a later
            # in-place op on them (e.g. sampler_gumbel_batch's logits.mul_())
            # raises "Inplace update to inference tensor outside InferenceMode"
            # -- this fired repeatedly in production before this fix.
            # no_grad() only affects autograd-graph
            # construction, checked live at op-dispatch time, so it has no
            # such hazard -- the tradeoff is that no_grad() (unlike
            # inference_mode()) won't loudly reject an in-place mutation of a
            # decode-loop tensor that somehow got captured by reference and
            # outlived this scope; not a concern today since every tensor
            # here is discarded at the end of each streamed response, but
            # worth knowing if this loop is ever refactored to persist state
            # across requests.
            with inference_deps.get_torch().no_grad():
                state = await self._offload_gpu(
                    self.model.generate_zero_state, batch_size
                )
                encoded_prompts = [self.tokenizer.encode(p) for p in prompts]

                # Prefill is the first heavy GPU forward of the request; run it
                # on the GPU-worker thread under the opt-in. The no_grad() re-entry
                # mirrors this method's ambient no_grad scope (no_grad is safe
                # across the async/thread boundary; inference_mode would NOT be).
                # In per-row mode we ALSO build that mode's per-row sampler state
                # (RNG + zero penalty rows, fp32 logits) here, under the same
                # offload unit, so no CUDA setup ever runs on the event-loop
                # thread under the opt-in.
                def _gpu_prefill():
                    with inference_deps.get_torch().no_grad():
                        out = self._forward_batch_prompts_chunked(
                            encoded_prompts, state, cancel_token=cancel_token
                        )
                        if per_row_mode:
                            torch = inference_deps.get_torch()
                            out = out.float()
                            rand_states = inference_deps.get_sample().setup_rand(
                                random.randint(0, 2**63 - 1), batch_size
                            )
                            penalties = torch.zeros(
                                batch_size, out.size(-1), device=out.device,
                                dtype=out.dtype,
                            )
                            return out, penalties, rand_states
                        return out

                if per_row_mode:
                    out, penalties, rand_states = await self._offload_gpu(_gpu_prefill)
                else:
                    out = await self._offload_gpu(_gpu_prefill)

                finished = [False] * batch_size
                stop_states = [self._create_stop_state(stop_tokens) for _ in range(batch_size)]
                chunk_token_counts = [0] * batch_size
                text_buffers = [""] * batch_size

                # active_indices[slot] = original prompt index currently occupying
                # GPU batch row `slot`. Starts as an identity mapping (slot i ==
                # original index i); as rows finish, _compact_active_rows()
                # below removes them from `state`/`out`/this mapping so the
                # decode loop's GPU tensors shrink to only the sequences still
                # generating, instead of carrying already-finished rows as
                # dead weight through every remaining forward_batch() call for
                # however many steps the slowest sequence in the batch still
                # needs. All the batch_size-sized bookkeeping below (finished,
                # stop_states, chunk_token_counts, text_buffers) stays indexed
                # by the *original* prompt index throughout -- only the GPU
                # tensors and the active_indices mapping shrink.
                active_indices = list(range(batch_size))

                while not all(finished) and max_length > 0:
                    self._raise_if_cancelled(cancel_token)

                    # One full GPU decode step -- sample a token from the current
                    # logits, feed it straight back through the model, and pull the
                    # tokens to CPU -- executed as a single unit on the GPU-worker
                    # thread under the opt-in. Keeping sampling + forward_batch +
                    # .tolist() together in ONE offload unit is what preserves the
                    # single-thread-CUDA guarantee under concurrent requests: no
                    # event-loop thread ever issues its own CUDA call (sampling is
                    # a GPU op too) while the worker has a forward in flight.
                    # no_grad() re-entry mirrors the ambient no_grad scope above.
                    prev_out = out

                    def _gpu_decode_step():
                        with inference_deps.get_torch().no_grad():
                            if per_row_mode:
                                new_tokens_tensor = inference_deps.get_sample_batch_per_row()(
                                    out,
                                    penalties,
                                    rand_states,
                                    row_temperatures,
                                    row_top_k,
                                    row_top_p,
                                    row_alpha_presence,
                                    row_alpha_frequency,
                                    row_alpha_decay,
                                )
                            else:
                                new_tokens_tensor = inference_deps.get_sampler_gumbel_batch()(
                                    logits=out, temp=temperature
                                )
                            new_out = self.model.forward_batch(new_tokens_tensor, state)
                            if per_row_mode:
                                # The repetition-penalty op requires fp32 logits
                                # (the model feeds fp16), so keep the decode
                                # tensor float32, matching the V1 path.
                                new_out = new_out.float()
                            new_tokens = new_tokens_tensor.tolist()
                            return new_out, new_tokens

                    out, new_tokens = await self._offload_gpu(_gpu_decode_step)
                    del prev_out

                    max_length -= 1

                    contents_to_send = [""] * batch_size
                    finish_reasons = [None] * batch_size
                    newly_finished = False

                    for slot, i in enumerate(active_indices):
                        # active_indices only ever contains not-yet-finished
                        # original indices (finished ones are removed by
                        # compaction below), so this should never be True --
                        # kept as a defensive skip rather than an assert so a
                        # future bug here degrades to "one extra step of
                        # wasted compute for that row" instead of corrupting
                        # output, matching this codebase's general preference
                        # for graceful degradation in the hot decode loop.
                        if finished[i]:
                            continue

                        tok = new_tokens[slot][0]

                        content, should_stop = self._ingest_token_with_stop(
                            stop_states[i], tok
                        )
                        if content:
                            text_buffers[i] += content

                        if should_stop:
                            finished[i] = True
                            newly_finished = True
                            flushed = self._flush_stop_state(
                                stop_states[i], final=True
                            )
                            if flushed:
                                text_buffers[i] += flushed
                            if text_buffers[i]:
                                contents_to_send[i] += text_buffers[i]
                                text_buffers[i] = ""
                            finish_reasons[i] = "stop"
                            continue

                        chunk_token_counts[i] += 1
                        if chunk_token_counts[i] >= chunk_size and text_buffers[i]:
                            contents_to_send[i] += text_buffers[i]
                            text_buffers[i] = ""
                            chunk_token_counts[i] = 0

                    if any(contents_to_send) or any(finish_reasons):
                        choices = []
                        for i in range(batch_size):
                            if not contents_to_send[i] and finish_reasons[i] is None:
                                continue
                            choice = {"index": i, "delta": {"content": contents_to_send[i]}}
                            if finish_reasons[i] is not None:
                                # Explicit per-item completion signal: this index
                                # will receive no further delta chunks. Lets a
                                # smart client stop waiting on this specific item
                                # without needing to wait for [DONE] (which only
                                # fires once the whole batch call completes).
                                choice["finish_reason"] = finish_reasons[i]
                            choices.append(choice)
                        if choices:
                            chunk = {
                                "object": "chat.completion.chunk",
                                "choices": choices,
                            }
                            yield (
                                f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                            )

                    new_tokens = None
                    await asyncio.sleep(0)

                    # Only compact when something actually finished this step
                    # (compaction is a no-op otherwise, so this check avoids
                    # gratuitous work on every step). Compacting immediately
                    # rather than on a periodic interval keeps state/out at
                    # their true active size for every subsequent
                    # forward_batch() call -- benchmarked at 3-11% of one
                    # forward step's cost across bsz 8-128, so paying it
                    # every time a row finishes is still a net win, not a
                    # tuning tradeoff.
                    if newly_finished and len(active_indices) > 0 and not all(finished):
                        still_active_slots = [
                            slot for slot, i in enumerate(active_indices) if not finished[i]
                        ]
                        if len(still_active_slots) < len(active_indices):
                            newly_freed = len(active_indices) - len(still_active_slots)

                            def _gpu_compact():
                                with inference_deps.get_torch().no_grad():
                                    return self._compact_active_rows(
                                        state, out, active_indices, still_active_slots,
                                        penalties=penalties, rand_states=rand_states,
                                    )

                            packed = await self._offload_gpu(_gpu_compact)
                            state, out, active_indices, penalties, rand_states = packed
                            # The per-row scalar lists are slot-indexed like the
                            # GPU tensors; re-filter them to the survical slots
                            # so the next _gpu_decode_step samples the right
                            # rows (slot order must match the shrunk `out`).
                            if per_row_mode:
                                row_temperatures = [row_temperatures[s] for s in still_active_slots]
                                row_top_k = [row_top_k[s] for s in still_active_slots]
                                row_top_p = [row_top_p[s] for s in still_active_slots]
                                row_alpha_presence = [row_alpha_presence[s] for s in still_active_slots]
                                row_alpha_frequency = [row_alpha_frequency[s] for s in still_active_slots]
                                row_alpha_decay = [row_alpha_decay[s] for s in still_active_slots]
                            if permit_box and permit_box[0] is not None:
                                # Give the freed row(s) back to the shared
                                # prefill-admission budget right now, not at
                                # this whole request's eventual end -- see
                                # this method's docstring-style comment above
                                # for why. permit_box[0] is only set once
                                # acquire_prefill_permit has actually
                                # succeeded (see prefill_sse_response's
                                # on_permit callback), so this is always a
                                # real, already-admitted permit by the time
                                # any row can finish.
                                await self.release_prefill_capacity(
                                    newly_freed, permit_box[0], request_label="/big_batch/completions"
                                )

                remaining_contents = [""] * batch_size
                for i in range(batch_size):
                    flushed = self._flush_stop_state(
                        stop_states[i], final=True
                    )
                    if flushed:
                        text_buffers[i] += flushed
                    remaining_contents[i] = text_buffers[i]

                # Items that never hit should_stop (finished[i] still False)
                # exited the while loop because max_length ran out, not
                # because of a stop string -- give them an explicit
                # finish_reason="length" chunk too, matching the single-item
                # /openai/v1/chat/completions convention (v1_routes.py) so
                # every index in a /big_batch/completions response gets a
                # completion signal one way or another, not just the ones
                # that stopped early.
                choices = []
                for i in range(batch_size):
                    reason = None if finished[i] else "length"
                    if not remaining_contents[i] and reason is None:
                        continue
                    choice = {"index": i, "delta": {"content": remaining_contents[i]}}
                    if reason is not None:
                        choice["finish_reason"] = reason
                    choices.append(choice)
                if choices:
                    chunk = {
                        "object": "chat.completion.chunk",
                        "choices": choices,
                    }
                    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

        finally:
            if new_tokens_tensor is not None:
                del new_tokens_tensor
            if out is not None:
                del out
            if state is not None:
                del state
            if penalties is not None:
                del penalties
            if rand_states is not None:
                del rand_states
            if encoded_prompts is not None:
                del encoded_prompts
            if finished is not None:
                del finished
            if new_tokens is not None:
                del new_tokens
            # Per-request cleanup removed (item 1): rely on CUDA's allocator
            # cache reuse plus one low-frequency background task
            # (InferenceEngine.run_periodic_gpu_cleanup), instead of the old
            # finally _cleanup_cuda_memory (a synchronize + empty_cache on this
            # decode hot path).

        yield "data: [DONE]\n\n"

    @staticmethod
    def _compact_active_rows(state, out, active_indices, still_active_slots,
                             penalties=None, rand_states=None):
        """Shrink `state`/`out` down to only `still_active_slots` (positions
        within the *current* active batch, not original prompt indices), and
        return the correspondingly shrunk `active_indices` mapping.

        In per-row sampling mode this also shrinks the slot-indexed per-row
        sampler state -- `penalties` [B, vocab] and `rand_states` (the flat
        byte tensor of B contiguous row-blocks from sample.setup_rand) -- to
        the same surviving rows, so a finished row drops its penalty/RNG state
        along with its GPU tensors and a row that persists keeps its own state
        (the row-wise independence that makes per-row sampling exact). When in
        the default (non-per-row) mode, `penalties`/`rand_states` are None and
        the two returned values are None, matching how the caller stores them.

        Reuses the same active-index tensor-slicing pattern already proven
        correct in RWKV_x070.forward_batch's chunked-prefill path
        (infer/rwkv_batch/rwkv7.py, the `batch_state = [state[0][:,:,active],
        state[1][:,active], state[2][active]]` branch) -- this is the decode-
        time equivalent of that same technique, applied once per finish event
        instead of once per prefill chunk.
        """
        torch = inference_deps.get_torch()
        idx = torch.tensor(still_active_slots, device=state[2].device, dtype=torch.long)
        new_state = [
            state[0].index_select(2, idx).contiguous(),
            state[1].index_select(1, idx).contiguous(),
            state[2].index_select(0, idx).contiguous(),
        ]
        new_out = out.index_select(0, idx).contiguous()
        new_active_indices = [active_indices[slot] for slot in still_active_slots]
        new_penalties = None
        new_rand_states = None
        if penalties is not None:
            new_penalties = penalties.index_select(0, idx).contiguous()
        if rand_states is not None:
            # rand_states is a flat byte tensor of `active_rows * rowsize`
            # elements; each row's state is one contiguous `rowsize` block, so
            # view as [rows, -1], gather the surviving blocks, flatten back.
            new_rand_states = (
                rand_states.view(len(active_indices), -1)[idx].contiguous().view(-1)
            )
        return new_state, new_out, new_active_indices, new_penalties, new_rand_states
