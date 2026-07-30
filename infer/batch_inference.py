import asyncio
import gc
import json
import random

from infer import inference_deps
from infer.inference_utils import sample_logits_batch_cuda


class BatchInferenceMixin:
### batch generation for V1 endpoint ###

    def _normalize_v2_sampling_params(self, temperature, top_k, top_p, vocab_size):
        sample_temperature = float(temperature)
        sample_top_p = float(top_p)
        if sample_temperature <= 0:
            sample_temperature = 1.0
            sample_top_p = 0.0
        else:
            sample_temperature = max(0.2, sample_temperature)
        return (
            sample_temperature,
            sample_top_p,
            min(max(1, int(top_k)), int(vocab_size)),
        )

    def _sample_v2_tokens(
        self,
        logits,
        occurrence_count,
        occurrence_presence,
        batch_rows,
        temperature,
        top_k,
        top_p,
        alpha_presence,
        alpha_frequency,
        alpha_decay,
        active_mask=None,
    ):
        if alpha_frequency:
            logits.sub_(occurrence_count, alpha=float(alpha_frequency))
        if alpha_presence:
            logits.sub_(occurrence_presence)

        sample_temperature, sample_top_p, sample_top_k = self._normalize_v2_sampling_params(
            temperature, top_k, top_p, logits.size(-1)
        )
        sampled_tensor = sample_logits_batch_cuda(
            logits,
            sample_temperature,
            sample_top_p,
            sample_top_k,
        )

        if alpha_decay != 1:
            occurrence_count.mul_(float(alpha_decay))
        if active_mask is None:
            active_rows = batch_rows
            active_tokens = sampled_tensor
        else:
            active_rows = batch_rows[active_mask]
            active_tokens = sampled_tensor[active_mask]

        occurrence_count[active_rows, active_tokens] += 1
        if alpha_presence:
            occurrence_presence[active_rows, active_tokens] = float(alpha_presence)

        return sampled_tensor

    def batch_generate_v2(
        self,
        prompts,
        max_length=512,
        temperature=1.0,
        top_k=500,
        top_p=0.5,
        alpha_presence=1.0,
        alpha_frequency=0.1,
        alpha_decay=0.99,
        stop_tokens=("\nUser:",),
        cancel_token=None,
    ):
        state = None
        try:
            batch_size = len(prompts)
            state = self.model.generate_zero_state(batch_size)
            encoded_prompts = [self.tokenizer.encode(p) for p in prompts]
            out = self._forward_batch_prompts_chunked(
                encoded_prompts, state, cancel_token=cancel_token
            ).float()

            finished = [False] * batch_size
            stop_states = [self._create_stop_state(stop_tokens) for _ in range(batch_size)]
            generated_text = [""] * batch_size
            torch = inference_deps.get_torch()
            occurrence_count = torch.zeros(
                batch_size, out.size(-1), device=out.device, dtype=out.dtype
            )
            occurrence_presence = torch.zeros_like(occurrence_count)
            batch_rows = torch.arange(batch_size, device=out.device)

            for _ in range(max_length):
                self._raise_if_cancelled(cancel_token)
                active_mask = torch.tensor(
                    [not flag for flag in finished], device=out.device, dtype=torch.bool
                )
                new_tokens_tensor = self._sample_v2_tokens(
                    out,
                    occurrence_count,
                    occurrence_presence,
                    batch_rows,
                    temperature,
                    top_k,
                    top_p,
                    alpha_presence,
                    alpha_frequency,
                    alpha_decay,
                    active_mask=active_mask,
                )
                # Single host sync for stop-string detection (needs Python ints). The
                # sampled tensor is fed straight back into the model as-is, avoiding the
                # extra host->device torch.tensor(idxs) rebuild the old List[List[int]]
                # path incurred inside forward_batch.
                new_tokens = new_tokens_tensor.tolist()
                out = self.model.forward_batch(new_tokens_tensor, state).float()

                for i in range(batch_size):
                    tok = new_tokens[i]
                    if finished[i]:
                        continue

                    content, should_stop = self._ingest_token_with_stop(stop_states[i], tok)
                    if content:
                        generated_text[i] += content

                    if should_stop:
                        finished[i] = True

                if all(finished):
                    break

            decoded = []
            reasons = []
            for i in range(batch_size):
                generated_text[i] += self._flush_stop_state(stop_states[i], final=True)
                decoded.append(generated_text[i])
                reasons.append("stop" if finished[i] else "length")
            return decoded, reasons
        finally:
            if state is not None:
                del state
            gc.collect()
            inference_deps.get_torch().cuda.empty_cache()

    async def batch_infer_stream_v2(
        self,
        prompts,
        max_length=512,
        temperature=1.0,
        top_k=500,
        top_p=0.5,
        alpha_presence=1.0,
        alpha_frequency=0.1,
        alpha_decay=0.99,
        stop_tokens=("\nUser:",),
        chunk_size=32,
        cancel_token=None,
    ):
        state = None

        try:
            batch_size = len(prompts)
            state = self.model.generate_zero_state(batch_size)
            encoded_prompts = [self.tokenizer.encode(p) for p in prompts]
            out = self._forward_batch_prompts_chunked(
                encoded_prompts, state, cancel_token=cancel_token
            ).float()

            # Per-row tracking: finish_reasons[i] is None while active, "stop" or
            # "length" once finished. active_indices maps current compacted batch
            # position -> original prompt index (for correct SSE "index" fields).
            finish_reasons = [None] * batch_size
            active_indices = list(range(batch_size))
            stop_states = [self._create_stop_state(stop_tokens) for _ in range(batch_size)]
            chunk_token_counts = [0] * batch_size
            text_buffers = [""] * batch_size
            torch = inference_deps.get_torch()
            occurrence_count = torch.zeros(
                batch_size, out.size(-1), device=out.device, dtype=out.dtype
            )
            occurrence_presence = torch.zeros_like(occurrence_count)
            batch_rows = torch.arange(batch_size, device=out.device)

            while active_indices and max_length > 0:
                self._raise_if_cancelled(cancel_token)
                n_active = len(active_indices)
                # All rows in the compacted batch are active by construction.
                new_tokens_tensor = self._sample_v2_tokens(
                    out,
                    occurrence_count,
                    occurrence_presence,
                    batch_rows,
                    temperature,
                    top_k,
                    top_p,
                    alpha_presence,
                    alpha_frequency,
                    alpha_decay,
                    active_mask=None,
                )
                # Single host sync for stop-string detection (needs Python ints). The
                # sampled tensor is fed straight back into the model as-is, avoiding the
                # extra host->device torch.tensor(idxs) rebuild the old List[List[int]]
                # path incurred inside forward_batch.
                new_tokens = new_tokens_tensor.tolist()
                out = self.model.forward_batch(new_tokens_tensor, state).float()
                max_length -= 1

                contents_to_send = {}  # orig_index -> content string
                terminal_chunks = []   # (orig_index, finish_reason)
                newly_finished_positions = []  # positions in compacted batch that finished

                for pos in range(n_active):
                    orig_i = active_indices[pos]
                    tok = new_tokens[pos]
                    content, should_stop = self._ingest_token_with_stop(stop_states[orig_i], tok)
                    if content:
                        text_buffers[orig_i] += content

                    if should_stop:
                        finish_reasons[orig_i] = "stop"
                        flushed = self._flush_stop_state(stop_states[orig_i], final=True)
                        if flushed:
                            text_buffers[orig_i] += flushed
                        if text_buffers[orig_i]:
                            contents_to_send[orig_i] = text_buffers[orig_i]
                            text_buffers[orig_i] = ""
                        terminal_chunks.append((orig_i, "stop"))
                        newly_finished_positions.append(pos)
                        continue

                    chunk_token_counts[orig_i] += 1
                    if chunk_token_counts[orig_i] >= chunk_size and text_buffers[orig_i]:
                        contents_to_send[orig_i] = text_buffers[orig_i]
                        text_buffers[orig_i] = ""
                        chunk_token_counts[orig_i] = 0

                # Emit content deltas and terminal finish_reason chunks
                if contents_to_send or terminal_chunks:
                    choices = []
                    for orig_i, text in contents_to_send.items():
                        choices.append({"index": orig_i, "delta": {"content": text}})
                    for orig_i, reason in terminal_chunks:
                        choices.append({"index": orig_i, "delta": {}, "finish_reason": reason})
                    if choices:
                        chunk = {"object": "chat.completion.chunk", "choices": choices}
                        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

                # Decode-time row compaction: remove finished rows from the GPU batch
                # so remaining rows get faster per-step compute.
                if newly_finished_positions:
                    still_active = [p for p in range(n_active) if p not in set(newly_finished_positions)]
                    if still_active:
                        idx_t = torch.tensor(still_active, device=out.device, dtype=torch.long)
                        out = out[idx_t]
                        occurrence_count = occurrence_count[idx_t]
                        occurrence_presence = occurrence_presence[idx_t]
                        batch_rows = torch.arange(len(still_active), device=out.device)
                        # Compact state: state[0]=[Layer][2][B][C], state[1]=[Layer][B][H][N][N]
                        state[0] = state[0][:, :, idx_t]
                        state[1] = state[1][:, idx_t]
                        active_indices = [active_indices[p] for p in still_active]
                    else:
                        active_indices = []

                await asyncio.sleep(0)

            # Flush remaining text for rows that hit max_length without a stop token
            remaining_choices = []
            for orig_i in range(batch_size):
                if finish_reasons[orig_i] is None:
                    finish_reasons[orig_i] = "length"
                    flushed = self._flush_stop_state(stop_states[orig_i], final=True)
                    if flushed:
                        text_buffers[orig_i] += flushed
                if text_buffers[orig_i]:
                    remaining_choices.append({"index": orig_i, "delta": {"content": text_buffers[orig_i]}})
                    text_buffers[orig_i] = ""
                # Emit terminal chunk for length-finished rows (stop rows already emitted)
                if finish_reasons[orig_i] == "length":
                    remaining_choices.append({"index": orig_i, "delta": {}, "finish_reason": "length"})

            if remaining_choices:
                chunk = {"object": "chat.completion.chunk", "choices": remaining_choices}
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
        finally:
            if state is not None:
                del state
            inference_deps.get_torch().cuda.empty_cache()
            gc.collect()

        yield "data: [DONE]\n\n"

    def batch_generate(
        self,
        prompts,
        max_length=512,
        temperature=1.0,
        top_k=50,
        top_p=0.6,
        alpha_presence=1.0,
        alpha_frequency=0.1,
        alpha_decay=0.996,
        stop_tokens=("\nUser:",),
        cancel_token=None,
    ):
        state = None
        try:
            batch_size = len(prompts)
            state = self.model.generate_zero_state(batch_size)
            encoded_prompts = [self.tokenizer.encode(p) for p in prompts]
            out = self._forward_batch_prompts_chunked(
                encoded_prompts, state, cancel_token=cancel_token
            ).float()

            finished = [False] * batch_size
            stop_states = [self._create_stop_state(stop_tokens) for _ in range(batch_size)]
            generated_text = [""] * batch_size
            sample_rand_states = inference_deps.get_sample().setup_rand(
                random.randint(0, 2**63 - 1), batch_size
            )
            penalties = inference_deps.get_torch().zeros(
                batch_size, out.size(-1), device=out.device
            )

            for _ in range(max_length):
                self._raise_if_cancelled(cancel_token)
                new_tokens_tensor = inference_deps.get_sample().batch_sampling_repetition_temperature_topk_topp(
                    out,
                    penalties,
                    sample_rand_states,
                    alpha_presence,
                    alpha_frequency,
                    alpha_decay,
                    temperature,
                    top_k,
                    top_p,
                )
                # Feed the sampled tokens straight back into the model as a GPU tensor (no
                # host round-trip); the single .tolist() sync below is only for stop-string
                # detection, which needs plain Python ints.
                out = self.model.forward_batch(new_tokens_tensor, state).float()
                new_tokens = new_tokens_tensor.tolist()

                for i in range(batch_size):
                    tok = new_tokens[i]
                    if finished[i]:
                        continue

                    content, should_stop = self._ingest_token_with_stop(stop_states[i], tok)
                    if content:
                        generated_text[i] += content

                    if should_stop:
                        finished[i] = True
                        continue

                if all(finished):
                    break

            decoded = []
            reasons = []
            for i in range(batch_size):
                generated_text[i] += self._flush_stop_state(stop_states[i], final=True)
                decoded.append(generated_text[i])
                reasons.append("stop" if finished[i] else "length")
            return decoded, reasons
        finally:
            if state is not None:
                del state
            gc.collect()
            inference_deps.get_torch().cuda.empty_cache()

    async def batch_infer_stream(
        self,
        prompts,
        max_length=512,
        temperature=1.0,
        top_k=50,
        top_p=0.6,
        alpha_presence=1.0,
        alpha_frequency=0.1,
        alpha_decay=0.996,
        stop_tokens=("\nUser:",),
        chunk_size=32,
        cancel_token=None,
    ):
        state = None

        try:
            batch_size = len(prompts)
            state = self.model.generate_zero_state(batch_size)
            encoded_prompts = [self.tokenizer.encode(p) for p in prompts]
            out = self._forward_batch_prompts_chunked(
                encoded_prompts, state, cancel_token=cancel_token
            ).float()

            finished = [False] * batch_size
            finish_reasons = [None] * batch_size
            stop_states = [self._create_stop_state(stop_tokens) for _ in range(batch_size)]
            chunk_token_counts = [0] * batch_size
            text_buffers = [""] * batch_size
            sample_rand_states = inference_deps.get_sample().setup_rand(
                random.randint(0, 2**63 - 1), batch_size
            )
            penalties = inference_deps.get_torch().zeros(
                batch_size, out.size(-1), device=out.device
            )

            while not all(finished) and max_length > 0:
                self._raise_if_cancelled(cancel_token)
                new_tokens_tensor = inference_deps.get_sample().batch_sampling_repetition_temperature_topk_topp(
                    out,
                    penalties,
                    sample_rand_states,
                    alpha_presence,
                    alpha_frequency,
                    alpha_decay,
                    temperature,
                    top_k,
                    top_p,
                )
                # Feed the sampled tokens straight back into the model as a GPU tensor (no
                # host round-trip); the single .tolist() sync below is only for stop-string
                # detection, which needs plain Python ints.
                out = self.model.forward_batch(new_tokens_tensor, state).float()
                new_tokens = new_tokens_tensor.tolist()
                max_length -= 1

                contents_to_send = [""] * batch_size
                terminal_chunks = []  # (index, finish_reason)

                for i in range(batch_size):
                    if finished[i]:
                        continue

                    tok = new_tokens[i]

                    content, should_stop = self._ingest_token_with_stop(stop_states[i], tok)
                    if content:
                        text_buffers[i] += content

                    if should_stop:
                        finished[i] = True
                        finish_reasons[i] = "stop"
                        flushed = self._flush_stop_state(stop_states[i], final=True)
                        if flushed:
                            text_buffers[i] += flushed
                        if text_buffers[i]:
                            contents_to_send[i] += text_buffers[i]
                            text_buffers[i] = ""
                        terminal_chunks.append((i, "stop"))
                        continue

                    chunk_token_counts[i] += 1
                    if chunk_token_counts[i] >= chunk_size and text_buffers[i]:
                        contents_to_send[i] += text_buffers[i]
                        text_buffers[i] = ""
                        chunk_token_counts[i] = 0

                if any(contents_to_send) or terminal_chunks:
                    choices = [
                        {"index": i, "delta": {"content": contents_to_send[i]}}
                        for i in range(batch_size)
                        if contents_to_send[i]
                    ]
                    for idx, reason in terminal_chunks:
                        choices.append({"index": idx, "delta": {}, "finish_reason": reason})
                    if choices:
                        chunk = {"object": "chat.completion.chunk", "choices": choices}
                        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

                await asyncio.sleep(0)

            remaining_choices = []
            for i in range(batch_size):
                if finish_reasons[i] is None:
                    finish_reasons[i] = "length"
                flushed = self._flush_stop_state(stop_states[i], final=True)
                if flushed:
                    text_buffers[i] += flushed
                if text_buffers[i]:
                    remaining_choices.append({"index": i, "delta": {"content": text_buffers[i]}})
                if finish_reasons[i] == "length":
                    remaining_choices.append({"index": i, "delta": {}, "finish_reason": "length"})

            if remaining_choices:
                chunk = {"object": "chat.completion.chunk", "choices": remaining_choices}
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
        finally:
            if state is not None:
                del state
            inference_deps.get_torch().cuda.empty_cache()
            gc.collect()

        yield "data: [DONE]\n\n"

### generation for state reuse endpoint ### 

    def batch_generate_state(
        self,
        prompts,
        state,
        max_length=512,
        temperature=1.0,
        top_k=50,
        top_p=0.6,
        alpha_presence=1.0,
        alpha_frequency=0.1,
        alpha_decay=0.996,
        stop_tokens=("\nUser:",),
        cancel_token=None,
    ):
        try:
            encoded_prompts = [self.tokenizer.encode(p) for p in prompts]

            tokens = encoded_prompts[0]
            out = self._forward_tokens_chunked(tokens, state, cancel_token=cancel_token)
            sample_rand_states = inference_deps.get_sample().setup_rand(random.randint(0, 2**63 - 1), 1)
            penalties = inference_deps.get_torch().zeros(1, out.size(-1), device=out.device)

            stop_state = self._create_stop_state(stop_tokens)
            generated_text = ""
            for _ in range(max_length):
                self._raise_if_cancelled(cancel_token)
                if out.dim() == 1:
                    out = out.unsqueeze(0)

                new_tokens = inference_deps.get_sample().batch_sampling_repetition_temperature_topk_topp(
                    out,
                    penalties,
                    sample_rand_states,
                    alpha_presence,
                    alpha_frequency,
                    alpha_decay,
                    temperature,
                    top_k,
                    top_p,
                ).tolist()

                tok = new_tokens[0]

                content, should_stop = self._ingest_token_with_stop(stop_state, tok)
                if content:
                    generated_text += content

                if should_stop:
                    break

                out = self._forward_tokens_chunked([tok], state, cancel_token=cancel_token)
            generated_text += self._flush_stop_state(stop_state, final=True)
            return [generated_text]
        finally:
            gc.collect()
            inference_deps.get_torch().cuda.empty_cache()

    async def batch_infer_stream_state(
        self,
        prompts,
        state,
        max_length=512,
        temperature=1.0,
        top_k=50,
        top_p=0.6,
        alpha_presence=1.0,
        alpha_frequency=0.1,
        alpha_decay=0.996,
        stop_tokens=("\nUser:",),
        chunk_size=32,
        session_id=None,
        state_manager=None,
        cancel_token=None,
    ):
        encoded_prompts = [self.tokenizer.encode(p) for p in prompts]
        chunk_size = max(1, int(chunk_size))
        should_store_state = True

        try:
            tokens = encoded_prompts[0]
            out = self._forward_tokens_chunked(tokens, state, cancel_token=cancel_token)
            sample_rand_states = inference_deps.get_sample().setup_rand(random.randint(0, 2**63 - 1), 1)
            penalties = inference_deps.get_torch().zeros(1, out.size(-1), device=out.device)

            stop_state = self._create_stop_state(stop_tokens)
            buffered_tokens = 0
            text_buffer = ""

            while max_length > 0:
                self._raise_if_cancelled(cancel_token)
                max_length -= 1
                if out.dim() == 1:
                    out = out.unsqueeze(0)

                new_tokens = inference_deps.get_sample().batch_sampling_repetition_temperature_topk_topp(
                    out,
                    penalties,
                    sample_rand_states,
                    alpha_presence,
                    alpha_frequency,
                    alpha_decay,
                    temperature,
                    top_k,
                    top_p,
                ).tolist()

                tok = new_tokens[0]

                content, should_stop = self._ingest_token_with_stop(stop_state, tok)
                if content:
                    text_buffer += content

                if should_stop:
                    flushed = self._flush_stop_state(stop_state, final=True)
                    if flushed:
                        text_buffer += flushed
                    if text_buffer:
                        chunk = {
                            "object": "chat.completion.chunk",
                            "choices": [{"index": 0, "delta": {"content": text_buffer}}],
                        }
                        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                        text_buffer = ""
                    break

                buffered_tokens += 1
                if buffered_tokens >= chunk_size and text_buffer:
                    chunk = {
                        "object": "chat.completion.chunk",
                        "choices": [{"index": 0, "delta": {"content": text_buffer}}],
                    }
                    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                    text_buffer = ""
                    buffered_tokens = 0

                out = self._forward_tokens_chunked([tok], state, cancel_token=cancel_token)

                await asyncio.sleep(0)

            flushed = self._flush_stop_state(stop_state, final=True)
            if flushed:
                text_buffer += flushed
            if text_buffer:
                chunk = {
                    "object": "chat.completion.chunk",
                    "choices": [{"index": 0, "delta": {"content": text_buffer}}],
                }
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

        finally:
            if cancel_token is not None and cancel_token.is_cancelled():
                should_store_state = False

            if state_manager and session_id and should_store_state:
                state_manager.put_state(session_id, state)
                print("[RESPONSE] /state/chat/completions state[2]: ", state[2], "\n")

            del state
            inference_deps.get_torch().cuda.empty_cache()
            gc.collect()

        yield "data: [DONE]\n\n"

### generation for OpenAI compatible bsz=1 endpoint ###

    async def singe_infer(
        self,
        prompt,
        max_length=512,
        temperature=1.0,
        top_k=50,
        top_p=0.6,
        alpha_presence=1.0,
        alpha_frequency=0.1,
        alpha_decay=0.996,
        stop_tokens=("\nUser:",),
        prefix_cache_manager=None,
        cancel_token=None,
    ):
        stop_state = self._create_stop_state(stop_tokens)
        generated_text = ""
        finish_reason = "length"
        state = None

        try:
            _, state, out, _, _ = self._prefill_prompt_with_prefix_cache(
                prompt,
                prefix_cache_manager=prefix_cache_manager,
                cancel_token=cancel_token,
            )
            sample_rand_states = inference_deps.get_sample().setup_rand(random.randint(0, 2**63 - 1), 1)
            penalties = inference_deps.get_torch().zeros(1, out.size(-1), device=out.device)

            while max_length > 0:
                self._raise_if_cancelled(cancel_token)
                max_length -= 1
                logits_reshaped = out.unsqueeze(0) if out.dim() == 1 else out
                new_tokens = inference_deps.get_sample().batch_sampling_repetition_temperature_topk_topp(
                    logits_reshaped,
                    penalties,
                    sample_rand_states,
                    alpha_presence,
                    alpha_frequency,
                    alpha_decay,
                    temperature,
                    top_k,
                    top_p,
                ).tolist()

                tok = new_tokens[0]
                content, should_stop = self._ingest_token_with_stop(stop_state, tok)
                if content:
                    generated_text += content

                if should_stop:
                    finish_reason = "stop"
                    break

                out = self._forward_tokens_chunked([tok], state, cancel_token=cancel_token)
                await asyncio.sleep(0)

            generated_text += self._flush_stop_state(stop_state, final=True)
            return generated_text, finish_reason
        finally:
            if state is not None:
                del state
            inference_deps.get_torch().cuda.empty_cache()
            gc.collect()

    async def singe_infer_stream(
        self,
        prompt,
        max_length=512,
        temperature=1.0,
        top_k=50,
        top_p=0.6,
        alpha_presence=1.0,
        alpha_frequency=0.1,
        alpha_decay=0.996,
        stop_tokens=("\nUser:",),
        chunk_size=32,
        prefix_cache_manager=None,
        cancel_token=None,
    ):
        finish_reason = "length"
        state = None

        try:
            _, state, out, _, _ = self._prefill_prompt_with_prefix_cache(
                prompt,
                prefix_cache_manager=prefix_cache_manager,
                cancel_token=cancel_token,
            )
            stop_state = self._create_stop_state(stop_tokens)
            buffered_tokens = 0
            text_buffer = ""
            sample_rand_states = inference_deps.get_sample().setup_rand(random.randint(0, 2**63 - 1), 1)
            penalties = inference_deps.get_torch().zeros(1, out.size(-1), device=out.device)

            while max_length > 0:
                self._raise_if_cancelled(cancel_token)
                max_length -= 1
                logits_reshaped = out.unsqueeze(0) if out.dim() == 1 else out
                new_tokens = inference_deps.get_sample().batch_sampling_repetition_temperature_topk_topp(
                    logits_reshaped,
                    penalties,
                    sample_rand_states,
                    alpha_presence,
                    alpha_frequency,
                    alpha_decay,
                    temperature,
                    top_k,
                    top_p,
                ).tolist()

                tok = new_tokens[0]
                content, should_stop = self._ingest_token_with_stop(stop_state, tok)
                if content:
                    text_buffer += content

                if should_stop:
                    finish_reason = "stop"
                    flushed = self._flush_stop_state(stop_state, final=True)
                    if flushed:
                        text_buffer += flushed
                    if text_buffer:
                        chunk = {
                            "object": "chat.completion.chunk",
                            "choices": [{"index": 0, "delta": {"content": text_buffer}}],
                        }
                        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                        text_buffer = ""
                    break

                buffered_tokens += 1
                if buffered_tokens >= chunk_size and text_buffer:
                    chunk = {
                        "object": "chat.completion.chunk",
                        "choices": [{"index": 0, "delta": {"content": text_buffer}}],
                    }
                    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                    text_buffer = ""
                    buffered_tokens = 0

                out = self._forward_tokens_chunked([tok], state, cancel_token=cancel_token)
                await asyncio.sleep(0)

            flushed = self._flush_stop_state(stop_state, final=True)
            if flushed:
                text_buffer += flushed
            if text_buffer:
                chunk = {
                    "object": "chat.completion.chunk",
                    "choices": [{"index": 0, "delta": {"content": text_buffer}}],
                }
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

            chunk = {
                "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
            }
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
        finally:
            if state is not None:
                del state
            inference_deps.get_torch().cuda.empty_cache()
            gc.collect()

        yield "data: [DONE]\n\n"
