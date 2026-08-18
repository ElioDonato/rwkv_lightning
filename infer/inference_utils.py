import gc
from collections import deque

import torch

from infer import inference_deps
from infer.cancellation import InferenceCancelled

# NOTE (CUDA-graph decode support, removed as dead code in Phase 1C): the
# static-buffer + torch.cuda.CUDAGraph skeleton (previously _init_cuda_graph_state)
# was deleted; if it is ever revived, a batch-layout forward under graph capture
# diverges from eager because the batched seq WKV kernel
# (kernel_forward_w0_fp16_dither_seq) is launched WITHOUT an explicit CUDA
# stream, unlike cuda_forward_one / cuda_spmv_forward which pass
# at::cuda::getCurrentCUDAStream(). See IMPL_PLAN_GPU_THROUGHPUT.md Phase 5.


@torch.jit.script
def sample_logits_batch_cuda(logits, temperature: float, top_p: float, k: int):
    if top_p <= 0.0 or k == 1:
        return torch.argmax(logits, dim=-1)
    vals, ids = torch.topk(logits.float(), k=k, dim=-1, sorted=True)
    if temperature == 1.0:
        probs = torch.softmax(vals, dim=-1)
    else:
        probs = torch.softmax(vals / temperature, dim=-1)
    cdf = torch.cumsum(probs, dim=-1)
    if top_p < 1.0:
        keep = torch.argmax((cdf >= top_p).to(torch.int32), dim=-1)
        mass = cdf.gather(1, keep.view(-1, 1)).view(-1)
    else:
        mass = cdf[:, -1]
    r = torch.rand((logits.size(0), 1), device=logits.device) * mass.view(-1, 1)
    out = torch.searchsorted(cdf, r).view(-1, 1)
    return ids.gather(1, out).view(-1)


class InferenceUtilsMixin:
    @staticmethod
    def _raise_if_cancelled(cancel_token=None):
        if cancel_token is not None and cancel_token.is_cancelled():
            raise InferenceCancelled("request disconnected")

    @staticmethod
    def _normalize_stop_strings(stop_tokens):
        if not stop_tokens:
            return ()
        return tuple(token for token in stop_tokens if isinstance(token, str) and token)

    @staticmethod
    def _match_stop_suffix(decoded_text, stop_strings):
        for stop_string in stop_strings:
            if decoded_text.endswith(stop_string):
                return stop_string
        return None

    def _create_stop_state(self, stop_tokens):
        return {
            "stop_strings": self._normalize_stop_strings(stop_tokens),
            "pending_tokens": deque(),
            "window_size": 6,
        }

    def _ingest_token_with_stop(self, stop_state, token):
        if token == 0:
            return "", True

        pending_tokens = stop_state["pending_tokens"]
        pending_tokens.append(token)

        decoded_window = self.tokenizer.decode(
            list(pending_tokens), utf8_errors="ignore"
        )
        matched_stop = self._match_stop_suffix(
            decoded_window, stop_state["stop_strings"]
        )
        if matched_stop is not None:
            pending_tokens.clear()
            return decoded_window[: -len(matched_stop)], True

        if len(pending_tokens) > stop_state["window_size"]:
            pending_tokens.popleft()
            trailing_text = self.tokenizer.decode(
                list(pending_tokens), utf8_errors="ignore"
            )
            if not trailing_text:
                return decoded_window, False
            if decoded_window.endswith(trailing_text):
                return decoded_window[: -len(trailing_text)], False

            return self.tokenizer.decode([token], utf8_errors="ignore"), False

        return "", False

    def _flush_stop_state(self, stop_state, final=False):
        pending_tokens = stop_state["pending_tokens"]
        if not pending_tokens:
            return ""

        if not final and len(pending_tokens) <= stop_state["window_size"]:
            return ""

        decoded = self.tokenizer.decode(list(pending_tokens), utf8_errors="ignore")
        pending_tokens.clear()
        return decoded

    @staticmethod
    def _cleanup_cuda_memory():
        gc.collect()
        if not inference_deps.get_torch().cuda.is_available():
            return
        try:
            inference_deps.get_torch().cuda.synchronize()
        except Exception:
            pass
        inference_deps.get_torch().cuda.empty_cache()

    def _forward_tokens_chunked(self, tokens, state, cancel_token=None):
        if not isinstance(tokens, list):
            self._raise_if_cancelled(cancel_token)
            return self.model.forward(tokens, state).float()

        if not tokens:
            raise ValueError("Empty prompt")

        chunk_size = max(1, int(getattr(self.model, "prefill_chunk_size", len(tokens))))
        out = None
        for start in range(0, len(tokens), chunk_size):
            self._raise_if_cancelled(cancel_token)
            chunk = tokens[start : start + chunk_size]
            out = self.model.forward(chunk, state).float()

        self._raise_if_cancelled(cancel_token)
        return out

    def _forward_batch_prompts_chunked(self, tokens, state, cancel_token=None, full_output=False):
        assert type(tokens) is list
        lengths = [len(x) for x in tokens]
        bsz = len(tokens)
        pos = [0] * bsz
        out = None if not full_output else [None] * bsz

        while True:
            self._raise_if_cancelled(cancel_token)
            active = [i for i in range(bsz) if pos[i] < lengths[i]]
            if not active:
                break

            step = min(
                getattr(self.model, "prefill_chunk_size", 256),
                min(lengths[i] - pos[i] for i in active),
            )
            batch_tokens = [tokens[i][pos[i] : pos[i] + step] for i in active]
            if len(active) == bsz:
                batch_state = state
            else:
                batch_state = [state[0][:, :, active], state[1][:, active], state[2][active]]

            new_out = self.model.forward_batch_same_length(batch_tokens, batch_state, full_output)

            if not full_output and out is None:
                out = inference_deps.get_torch().empty(
                    (bsz, new_out.size(-1)),
                    dtype=new_out.dtype,
                    device=new_out.device,
                )

            for k, i in enumerate(active):
                if not full_output:
                    out[i] = new_out[k]
                else:
                    if out[i] is None:
                        out[i] = new_out[k]
                    else:
                        out[i] = inference_deps.get_torch().cat([out[i], new_out[k]], dim=0)

                if len(active) != bsz:
                    state[0][:, :, i] = batch_state[0][:, :, k]
                    state[1][:, i] = batch_state[1][:, k]
                    state[2][i] = batch_state[2][k]
                pos[i] += step

        self._raise_if_cancelled(cancel_token)
        return out

    def _prefill_prompt_with_prefix_cache(self, prompt, prefix_cache_manager=None, cancel_token=None):
        encoded_prompt = self.tokenizer.encode(prompt)
        if not encoded_prompt:
            raise ValueError("Empty prompt")

        state = None
        out = None
        matched_tokens = 0
        cache_source = None

        if prefix_cache_manager is not None:
            cache_match = prefix_cache_manager.match_prefix_state(encoded_prompt, device="cuda")
            if cache_match is not None:
                state = cache_match["state"]
                out = cache_match["logits"]
                matched_tokens = int(cache_match["matched_tokens"])
                cache_source = cache_match["cache_source"]

        if state is None:
            state = self.model.generate_zero_state(0)

        if prefix_cache_manager is not None:
            buckets = getattr(prefix_cache_manager, "prefix_l2_cache", {})
            # Snapshot the (now dynamically-growing) adaptive bucket-key set under
            # the pool lock so a concurrent add/evict of an adaptive-length bucket
            # can't raise "dictionary changed size during iteration" on another
            # request thread.
            try:
                _lock = prefix_cache_manager.cache_lock
            except AttributeError:
                _lock = None
            if _lock is not None:
                with _lock:
                    bucket_keys = list(buckets.keys())
            else:
                bucket_keys = list(buckets.keys())
            bucket_checkpoints = [
                bucket for bucket in bucket_keys
                if matched_tokens < bucket <= len(encoded_prompt)
            ]
            bucket_checkpoints.sort()
        else:
            bucket_checkpoints = []

        cursor = matched_tokens
        for checkpoint in bucket_checkpoints:
            self._raise_if_cancelled(cancel_token)
            segment = encoded_prompt[cursor:checkpoint]
            if segment:
                out = self._forward_tokens_chunked(segment, state, cancel_token=cancel_token)
                prefix_cache_manager.put_prefix_state(encoded_prompt[:checkpoint], state, out)
                cursor = checkpoint

        remaining_tokens = encoded_prompt[cursor:]
        if remaining_tokens:
            out = self._forward_tokens_chunked(
                remaining_tokens, state, cancel_token=cancel_token
            )
        elif out is None:
            # Older cache rows may exist without logits. Fall back to recomputing once.
            del state
            state = self.model.generate_zero_state(0)
            out = self._forward_tokens_chunked(
                encoded_prompt, state, cancel_token=cancel_token
            )
            matched_tokens = 0
            cache_source = None

        # B (RWKV_PREFIX_ADAPTIVE): the fixed buckets start at 1024, so a short
        # prompt (<1024) never gets a checkpoint and is fully re-prefilled on
        # every repeat/multi-turn continuation. When adaptive is on and we
        # actually advanced the prefix, store a checkpoint at the FULL computed
        # length (L2-only) so an identical or prefix-extension prompt next time
        # matches via the trie and resumes from here instead of re-prefilling.
        if (
            prefix_cache_manager is not None
            and len(encoded_prompt) > matched_tokens
        ):
            try:
                from settings import settings as _settings
                adaptive = bool(
                    getattr(_settings, "prefix_adaptive", False)
                    or getattr(_settings, "turn_state_reuse", False)
                )
            except Exception:
                adaptive = False
            if adaptive:
                prefix_cache_manager.put_prefix_state(encoded_prompt, state, out)

        return encoded_prompt, state, out, matched_tokens, cache_source
