import asyncio
import json

from infer import inference_deps


class BigBatchMixin:
    async def big_batch_stream(
        self,
        prompts,
        max_length=512,
        temperature=1.0,
        stop_tokens=("\nUser:",),
        chunk_size=32,
        cancel_token=None,
    ):
        batch_size = len(prompts)
        state = None
        encoded_prompts = None
        out = None
        finished = None
        generated_tokens = None
        token_buffers = None
        new_tokens_tensor = None
        new_tokens = None

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
                state = self.model.generate_zero_state(batch_size)
                encoded_prompts = [self.tokenizer.encode(p) for p in prompts]
                out = self._forward_batch_prompts_chunked(
                    encoded_prompts, state, cancel_token=cancel_token
                )

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

                step_count = 0
                cleanup_interval = 100

                while not all(finished) and max_length > 0:
                    self._raise_if_cancelled(cancel_token)
                    new_tokens_tensor = inference_deps.get_sampler_gumbel_batch()(
                        logits=out, temp=temperature
                    )
                    new_tokens = new_tokens_tensor.tolist()
                    del new_tokens_tensor
                    new_tokens_tensor = None

                    prev_out = out
                    out = self.model.forward_batch(new_tokens, state)
                    del prev_out

                    max_length -= 1
                    step_count += 1

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

                        tok = (
                            new_tokens[slot][0]
                            if isinstance(new_tokens[slot], list)
                            else new_tokens[slot]
                        )

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
                            state, out, active_indices = self._compact_active_rows(
                                state, out, active_indices, still_active_slots
                            )

                    if step_count % cleanup_interval == 0:
                        self._cleanup_cuda_memory()

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
            if encoded_prompts is not None:
                del encoded_prompts
            if finished is not None:
                del finished
            if generated_tokens is not None:
                del generated_tokens
            if token_buffers is not None:
                del token_buffers
            if new_tokens is not None:
                del new_tokens
            self._cleanup_cuda_memory()

        yield "data: [DONE]\n\n"

    @staticmethod
    def _compact_active_rows(state, out, active_indices, still_active_slots):
        """Shrink `state`/`out` down to only `still_active_slots` (positions
        within the *current* active batch, not original prompt indices), and
        return the correspondingly shrunk `active_indices` mapping.

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
        return new_state, new_out, new_active_indices
