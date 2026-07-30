import logging

logger = logging.getLogger("infer.batch")

import asyncio
import gc
import json
import random

from infer import inference_deps
from infer.inference_utils import sample_logits_batch_cuda


# ---------------------------------------------------------------------------
# Sampler abstractions: encapsulate init / sample / compact for V1 and V2.
# Both produce a [B] GPU LongTensor of sampled token ids per decode step.
# compact(idx_t) shrinks internal state when rows are removed from the batch.
# ---------------------------------------------------------------------------

class _V1BatchSampler:
    """V1 sampler: penalties[B,vocab] + rand_states[B,...]."""

    __slots__ = ("penalties", "rand_states", "temperature", "top_k", "top_p",
                 "alpha_presence", "alpha_frequency", "alpha_decay")

    def __init__(self, batch_size, vocab_size, device,
                 temperature, top_k, top_p, alpha_presence, alpha_frequency, alpha_decay):
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.alpha_presence = alpha_presence
        self.alpha_frequency = alpha_frequency
        self.alpha_decay = alpha_decay
        self.rand_states = inference_deps.get_sample().setup_rand(
            random.randint(0, 2**63 - 1), batch_size
        )
        self.penalties = inference_deps.get_torch().zeros(batch_size, vocab_size, device=device)

    def sample(self, logits):
        return inference_deps.get_sample().batch_sampling_repetition_temperature_topk_topp(
            logits, self.penalties, self.rand_states,
            self.alpha_presence, self.alpha_frequency, self.alpha_decay,
            self.temperature, self.top_k, self.top_p,
        )

    def compact(self, idx_t):
        self.penalties = self.penalties[idx_t]
        self.rand_states = self.rand_states[idx_t]


class _V2BatchSampler:
    """V2 sampler: occurrence_count[B,vocab] + occurrence_presence[B,vocab] + batch_rows[B]."""

    __slots__ = ("occurrence_count", "occurrence_presence", "batch_rows",
                 "temperature", "top_k", "top_p",
                 "alpha_presence", "alpha_frequency", "alpha_decay")

    def __init__(self, batch_size, vocab_size, device, dtype,
                 temperature, top_k, top_p, alpha_presence, alpha_frequency, alpha_decay):
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.alpha_presence = alpha_presence
        self.alpha_frequency = alpha_frequency
        self.alpha_decay = alpha_decay
        torch = inference_deps.get_torch()
        self.occurrence_count = torch.zeros(batch_size, vocab_size, device=device, dtype=dtype)
        self.occurrence_presence = torch.zeros_like(self.occurrence_count)
        self.batch_rows = torch.arange(batch_size, device=device)

    def sample(self, logits):
        if self.alpha_frequency:
            logits.sub_(self.occurrence_count, alpha=float(self.alpha_frequency))
        if self.alpha_presence:
            logits.sub_(self.occurrence_presence)

        sample_temperature = float(self.temperature)
        sample_top_p = float(self.top_p)
        if sample_temperature <= 0:
            sample_temperature = 1.0
            sample_top_p = 0.0
        else:
            sample_temperature = max(0.2, sample_temperature)
        sample_top_k = min(max(1, int(self.top_k)), int(logits.size(-1)))

        sampled = sample_logits_batch_cuda(logits, sample_temperature, sample_top_p, sample_top_k)

        if self.alpha_decay != 1:
            self.occurrence_count.mul_(float(self.alpha_decay))
        self.occurrence_count[self.batch_rows, sampled] += 1
        if self.alpha_presence:
            self.occurrence_presence[self.batch_rows, sampled] = float(self.alpha_presence)

        return sampled

    def compact(self, idx_t):
        self.occurrence_count = self.occurrence_count[idx_t]
        self.occurrence_presence = self.occurrence_presence[idx_t]
        self.batch_rows = inference_deps.get_torch().arange(len(idx_t), device=idx_t.device)


# ---------------------------------------------------------------------------
# BatchInferenceMixin: public API + shared decode cores.
# ---------------------------------------------------------------------------

class BatchInferenceMixin:

    # -- Shared blocking decode core (with compaction) ----------------------

    def _batch_prefill(self, prompts, prefix_cache_manager=None, cancel_token=None):
        """Prefill a batch of prompts, optionally using prefix cache.
        Returns (state, out) ready for the decode loop."""
        batch_size = len(prompts)
        torch = inference_deps.get_torch()

        if prefix_cache_manager is None:
            # Fast path: no cache, batched prefill from zero state
            state = self.model.generate_zero_state(batch_size)
            encoded_prompts = [self.tokenizer.encode(p) for p in prompts]
            out = self._forward_batch_prompts_chunked(
                encoded_prompts, state, cancel_token=cancel_token
            ).float()
            return state, out

        # Cache path: prefill each prompt individually (reuses proven bsz=1 logic)
        states = []
        logits = []
        for prompt in prompts:
            self._raise_if_cancelled(cancel_token)
            _, s, o, _, _ = self._prefill_prompt_with_prefix_cache(
                prompt,
                prefix_cache_manager=prefix_cache_manager,
                cancel_token=cancel_token,
            )
            states.append(s)
            logits.append(o)

        # Stack individual bsz=1 states into a batch state
        state = [
            torch.cat([s[0] for s in states], dim=2),  # [Layer, 2, B, C]
            torch.cat([s[1] for s in states], dim=1),  # [Layer, B, H, N, N]
            torch.cat([s[2].unsqueeze(0) for s in states], dim=0),  # [B]
        ]
        out = torch.stack(logits, dim=0).float()  # [B, vocab]

        # Free individual state references
        for s in states:
            del s

        return state, out

    def _batch_decode_blocking(self, prompts, sampler, max_length, stop_tokens, cancel_token,
                               prefix_cache_manager=None):
        """Shared blocking decode loop for V1/V2. Returns (decoded_texts, finish_reasons)."""
        state = None
        try:
            batch_size = len(prompts)
            state, out = self._batch_prefill(prompts, prefix_cache_manager, cancel_token)

            torch = inference_deps.get_torch()
            finish_reasons = [None] * batch_size
            active_indices = list(range(batch_size))
            stop_states = [self._create_stop_state(stop_tokens) for _ in range(batch_size)]
            generated_text = [""] * batch_size

            while active_indices and max_length > 0:
                self._raise_if_cancelled(cancel_token)
                n_active = len(active_indices)
                new_tokens_tensor = sampler.sample(out)
                out = self.model.forward_batch(new_tokens_tensor, state).float()
                new_tokens = new_tokens_tensor.tolist()
                max_length -= 1

                newly_finished_positions = []
                for pos in range(n_active):
                    orig_i = active_indices[pos]
                    content, should_stop = self._ingest_token_with_stop(stop_states[orig_i], new_tokens[pos])
                    if content:
                        generated_text[orig_i] += content
                    if should_stop:
                        finish_reasons[orig_i] = "stop"
                        newly_finished_positions.append(pos)

                if newly_finished_positions:
                    still_active = [p for p in range(n_active) if p not in set(newly_finished_positions)]
                    if still_active:
                        idx_t = torch.tensor(still_active, device=out.device, dtype=torch.long)
                        out = out[idx_t]
                        sampler.compact(idx_t)
                        state[0] = state[0][:, :, idx_t]
                        state[1] = state[1][:, idx_t]
                        state[2] = state[2][idx_t]
                        active_indices = [active_indices[p] for p in still_active]
                    else:
                        active_indices = []

            decoded = []
            reasons = []
            for i in range(batch_size):
                generated_text[i] += self._flush_stop_state(stop_states[i], final=True)
                decoded.append(generated_text[i])
                reasons.append(finish_reasons[i] if finish_reasons[i] is not None else "length")
            return decoded, reasons
        finally:
            if state is not None:
                del state
            gc.collect()
            inference_deps.get_torch().cuda.empty_cache()

    # -- Shared streaming decode core (with compaction + SSE) ---------------

    async def _batch_decode_streaming(self, prompts, sampler, max_length, stop_tokens, chunk_size, cancel_token,
                                      prefix_cache_manager=None):
        """Shared streaming decode loop for V1/V2. Yields SSE chunks."""
        state = None
        try:
            batch_size = len(prompts)
            state, out = self._batch_prefill(prompts, prefix_cache_manager, cancel_token)

            torch = inference_deps.get_torch()
            finish_reasons = [None] * batch_size
            active_indices = list(range(batch_size))
            stop_states = [self._create_stop_state(stop_tokens) for _ in range(batch_size)]
            chunk_token_counts = [0] * batch_size
            text_buffers = [""] * batch_size

            while active_indices and max_length > 0:
                self._raise_if_cancelled(cancel_token)
                n_active = len(active_indices)
                new_tokens_tensor = sampler.sample(out)
                out = self.model.forward_batch(new_tokens_tensor, state).float()
                new_tokens = new_tokens_tensor.tolist()
                max_length -= 1

                contents_to_send = {}
                terminal_chunks = []
                newly_finished_positions = []

                for pos in range(n_active):
                    orig_i = active_indices[pos]
                    content, should_stop = self._ingest_token_with_stop(stop_states[orig_i], new_tokens[pos])
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

                if contents_to_send or terminal_chunks:
                    choices = []
                    for orig_i, text in contents_to_send.items():
                        choices.append({"index": orig_i, "delta": {"content": text}})
                    for idx, reason in terminal_chunks:
                        choices.append({"index": idx, "delta": {}, "finish_reason": reason})
                    if choices:
                        chunk = {"object": "chat.completion.chunk", "choices": choices}
                        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

                if newly_finished_positions:
                    still_active = [p for p in range(n_active) if p not in set(newly_finished_positions)]
                    if still_active:
                        idx_t = torch.tensor(still_active, device=out.device, dtype=torch.long)
                        out = out[idx_t]
                        sampler.compact(idx_t)
                        state[0] = state[0][:, :, idx_t]
                        state[1] = state[1][:, idx_t]
                        state[2] = state[2][idx_t]
                        active_indices = [active_indices[p] for p in still_active]
                    else:
                        active_indices = []

                await asyncio.sleep(0)

            # Flush remaining text for rows that hit max_length
            remaining_choices = []
            for orig_i in range(batch_size):
                if finish_reasons[orig_i] is None:
                    finish_reasons[orig_i] = "length"
                flushed = self._flush_stop_state(stop_states[orig_i], final=True)
                if flushed:
                    text_buffers[orig_i] += flushed
                if text_buffers[orig_i]:
                    remaining_choices.append({"index": orig_i, "delta": {"content": text_buffers[orig_i]}})
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

    # -- V2 public API ------------------------------------------------------

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
        prefix_cache_manager=None,
    ):
        batch_size = len(prompts)
        # Peek at vocab size from model for sampler init
        vocab_size = self.model.args.vocab_size
        torch = inference_deps.get_torch()
        sampler = _V2BatchSampler(
            batch_size, vocab_size, device="cuda", dtype=torch.float32,
            temperature=temperature, top_k=top_k, top_p=top_p,
            alpha_presence=alpha_presence, alpha_frequency=alpha_frequency, alpha_decay=alpha_decay,
        )
        return self._batch_decode_blocking(prompts, sampler, max_length, stop_tokens, cancel_token,
                                           prefix_cache_manager=prefix_cache_manager)

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
        prefix_cache_manager=None,
    ):
        batch_size = len(prompts)
        vocab_size = self.model.args.vocab_size
        torch = inference_deps.get_torch()
        sampler = _V2BatchSampler(
            batch_size, vocab_size, device="cuda", dtype=torch.float32,
            temperature=temperature, top_k=top_k, top_p=top_p,
            alpha_presence=alpha_presence, alpha_frequency=alpha_frequency, alpha_decay=alpha_decay,
        )
        async for chunk in self._batch_decode_streaming(prompts, sampler, max_length, stop_tokens, chunk_size, cancel_token,
                                                        prefix_cache_manager=prefix_cache_manager):
            yield chunk

    # -- V1 public API ------------------------------------------------------

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
        prefix_cache_manager=None,
    ):
        batch_size = len(prompts)
        vocab_size = self.model.args.vocab_size
        sampler = _V1BatchSampler(
            batch_size, vocab_size, device="cuda",
            temperature=temperature, top_k=top_k, top_p=top_p,
            alpha_presence=alpha_presence, alpha_frequency=alpha_frequency, alpha_decay=alpha_decay,
        )
        return self._batch_decode_blocking(prompts, sampler, max_length, stop_tokens, cancel_token,
                                           prefix_cache_manager=prefix_cache_manager)

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
        prefix_cache_manager=None,
    ):
        batch_size = len(prompts)
        vocab_size = self.model.args.vocab_size
        sampler = _V1BatchSampler(
            batch_size, vocab_size, device="cuda",
            temperature=temperature, top_k=top_k, top_p=top_p,
            alpha_presence=alpha_presence, alpha_frequency=alpha_frequency, alpha_decay=alpha_decay,
        )
        async for chunk in self._batch_decode_streaming(prompts, sampler, max_length, stop_tokens, chunk_size, cancel_token,
                                                        prefix_cache_manager=prefix_cache_manager):
            yield chunk

    # -- State reuse endpoint (bsz=1, unchanged) ----------------------------

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
                logger.info(f"[RESPONSE] /state/chat/completions state[2]: {state[2]}")

            del state
            inference_deps.get_torch().cuda.empty_cache()
            gc.collect()

        yield "data: [DONE]\n\n"

    # -- OpenAI compatible bsz=1 endpoint (unchanged) -----------------------

    async def single_infer(
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

    async def single_infer_stream(
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
