"""
OpenAI Responses API compatible endpoint (/v1/responses).

The Responses API is stateful -- it supports `previous_response_id` to chain
multi-turn conversations. This maps naturally to RWKV's recurrent state: we
store the RWKV state keyed by response ID (reusing the same StateCacheManager
that backs /state/ and /multi_state/, with its existing GPU/RAM/disk L1/L2/L3
eviction), and resume from it on the next turn instead of re-processing the
full conversation history.

Decode itself is not reimplemented here: this module builds a prompt/state
and delegates to InferenceEngine.batch_generate_state / batch_infer_stream_state
(bsz=1, state = a stored previous turn or a fresh zero state) -- the same
tested decode loops used by /state/chat/completions. This matters for
correctness, not just code reuse: the stored state must encode everything the
model has said so far, including this turn's own generated reply, so the
*same* decode call that produces the reply must also be the one whose final
state gets persisted (a separate "replay the prompt to reconstruct state"
step would silently drop the just-generated continuation from memory).

Request format:
    POST /v1/responses
    {
        "model": "rwkv7",
        "input": "Hello!" | [{"role": "user", "content": "Hello!"}],
        "instructions": "You are a helpful assistant.",  (optional)
        "previous_response_id": "resp_...",  (optional, for multi-turn)
        "max_output_tokens": 1024,  (optional)
        "temperature": 0.8,  (optional)
        "top_p": 0.6,  (optional)
        "stream": false  (optional)
    }

Response format (non-streaming):
    {
        "id": "resp_...",
        "object": "response",
        "created_at": 1234567890,
        "model": "rwkv7",
        "status": "completed",
        "output": [
            {
                "id": "msg_...",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "...", "annotations": []}]
            }
        ],
        "usage": {"input_tokens": N, "output_tokens": M, "total_tokens": N+M}
    }
"""
import json
import logging
import os
import time
import uuid
from contextlib import nullcontext

from fastapi import APIRouter, Request

from infer.cancellation import CancellationToken, InferenceCancelled, PrefillBszLimitExceeded
from state_manager.state_pool import get_state_manager, remove_session_from_any_level

from API_servers.router.common import (
    check_openai_auth,
    client_closed_response,
    json_response,
    parse_request_model,
    prefill_bsz_limit_response,
    prefill_sse_response,
    reserve_prefill_capacity,
    run_sync_with_disconnect_watch,
    session_lock,
)
from API_servers.router.openai_routes import sanitize_text_block
from API_servers.router.schemas import ResponsesRequest

logger = logging.getLogger("api.responses")

router = APIRouter()

# Responses live in the same StateCacheManager (L1 GPU / L2 RAM / L3 disk,
# with LRU eviction) that backs /state/ and /multi_state/, just under their
# own "resp_..." keys -- so a busy server doesn't need a second, unbounded,
# in-process eviction policy to reason about.
_RESPONSE_STOP_TOKENS = ("\nUser:",)


def _build_prompt_from_input(input_data, instructions: str | None, is_continuation: bool) -> str:
    """Convert Responses API input (string or message array) to a prompt string.

    When `is_continuation` is True (resuming a previous_response_id's stored
    state), only the new turn's text is returned and prefixed with a blank
    line -- the existing state already encodes everything said so far, the
    same way normalize_state_prompts() treats new turns for /state/.
    """
    parts = []

    if not is_continuation and instructions:
        parts.append(f"System: {sanitize_text_block(instructions)}")

    if isinstance(input_data, str):
        text = sanitize_text_block(input_data)
        if text:
            parts.append(f"User: {text}")
    elif isinstance(input_data, list):
        for item in input_data:
            if isinstance(item, str):
                text = sanitize_text_block(item)
                if text:
                    parts.append(f"User: {text}")
            elif isinstance(item, dict):
                role = str(item.get("role", "user")).capitalize()
                content = sanitize_text_block(item.get("content", ""))
                if content:
                    parts.append(f"{role}: {content}")

    prompt = "\n\n".join(parts)
    if is_continuation and prompt:
        prompt = f"\n\n{prompt}"
    return f"{prompt}\n\nAssistant:"


def _build_message_output(msg_id: str, text: str) -> dict:
    return {
        "id": msg_id,
        "type": "message",
        "status": "completed",
        "role": "assistant",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }


def _response_lock_context(previous_response_id: str | None):
    """Pick the concurrency guard for a /v1/responses request.

    session_lock(previous_response_id) is only meaningful -- and only
    acquired -- when there's an existing previous_response_id two requests
    could actually race on (a client retry, or two branches chained off the
    same prior turn). A first-turn request's response_id is a UUID minted
    fresh inside the request itself, unknown to any other request until
    this one returns it in its HTTP response, so no concurrent request
    could ever contend for it; locking on it anyway would leak one
    permanent entry into common.py's process-lifetime _session_locks dict
    per request for zero correctness benefit. See the call site in
    create_response() for the full incident writeup.
    """
    if previous_response_id:
        return session_lock(previous_response_id)
    return nullcontext()


def _build_response_payload(response_id, created_at, model_name, msg_id, text, usage, status="completed"):
    return {
        "id": response_id,
        "object": "response",
        "created_at": created_at,
        "model": model_name,
        "status": status,
        "output": [_build_message_output(msg_id, text)],
        "usage": usage,
    }


@router.post("/v1/responses")
async def create_response(request: Request):
    engine = request.app.state.engine
    password = request.app.state.password

    try:
        body = await request.json()
    except Exception:
        return json_response(400, {"error": "Invalid JSON in request body"})

    auth_error = check_openai_auth(request, body, password)
    if auth_error is not None:
        return auth_error

    req, parse_error = parse_request_model(ResponsesRequest, body)
    if parse_error is not None:
        return parse_error

    previous_response_id = req.previous_response_id
    state_manager = get_state_manager()

    # Look up the previous turn's state (if any) up front, so a 404 for an
    # unknown/expired previous_response_id is reported before any GPU work
    # or prefill-admission reservation happens. get_state() always returns a
    # fresh clone (see StateCacheManager._clone_state), so mutating it below
    # during decode never corrupts the original entry -- that keeps
    # previous_response_id resumable for other branches even after this turn.
    prev_state = None
    if previous_response_id:
        prev_state = state_manager.get_state(previous_response_id)
        if prev_state is None:
            return json_response(404, {
                "error": f"Response '{previous_response_id}' not found or expired"
            })

    is_continuation = prev_state is not None
    prompt = _build_prompt_from_input(req.input, req.instructions, is_continuation=is_continuation)
    prompt_tokens = len(engine.tokenizer.encode(prompt))

    response_id = f"resp_{uuid.uuid4().hex[:24]}"
    msg_id = f"msg_{uuid.uuid4().hex[:24]}"
    created_at = int(time.time())
    model_name = os.path.basename(engine.args.MODEL_NAME)

    # See _response_lock_context()'s docstring for why this is *not*
    # session_lock(response_id) unconditionally.
    lock_ctx = _response_lock_context(previous_response_id)

    try:
        if req.stream:
            cancel_token = CancellationToken()

            async def response_stream():
                # NOTE: prefill_sse_response() (below) already acquires the
                # shared prefill-admission permit for this cancel_token before
                # driving this generator -- reserving it again here would
                # double-count this request against the server's admission
                # budget, so this generator only needs the session lock.
                async with lock_ctx:
                    state = prev_state if is_continuation else engine.model.generate_zero_state(0)
                    inner_stream = engine.batch_infer_stream_state(
                        prompts=[prompt],
                        state=state,
                        max_length=req.max_output_tokens,
                        temperature=req.temperature,
                        top_k=req.top_k,
                        top_p=req.top_p,
                        alpha_presence=req.alpha_presence,
                        alpha_frequency=req.alpha_frequency,
                        alpha_decay=req.alpha_decay,
                        stop_tokens=_RESPONSE_STOP_TOKENS,
                        chunk_size=1,
                        session_id=None,
                        state_manager=None,
                        cancel_token=cancel_token,
                    )

                    generated_text = ""
                    try:
                        async for item in inner_stream:
                            if not item.startswith("data: "):
                                continue
                            payload = item[len("data: "):].strip()
                            if payload == "[DONE]":
                                break
                            try:
                                chunk = json.loads(payload)
                            except ValueError:
                                continue
                            choices = chunk.get("choices") or []
                            if not choices:
                                continue
                            content = (choices[0].get("delta") or {}).get("content")
                            if content:
                                generated_text += content
                                out_chunk = {
                                    "type": "response.output_text.delta",
                                    "delta": content,
                                    "item_id": msg_id,
                                    "output_index": 0,
                                    "content_index": 0,
                                }
                                yield f"data: {json.dumps(out_chunk, ensure_ascii=False)}\n\n"
                    finally:
                        await inner_stream.aclose()

                    if req.store and not cancel_token.is_cancelled():
                        state_manager.put_state(response_id, state)

                    output_tokens = len(engine.tokenizer.encode(generated_text)) if generated_text else 0
                    usage = {
                        "input_tokens": prompt_tokens,
                        "output_tokens": output_tokens,
                        "total_tokens": prompt_tokens + output_tokens,
                    }
                    done_event = {
                        "type": "response.completed",
                        "response": _build_response_payload(
                            response_id, created_at, model_name, msg_id, generated_text, usage,
                        ),
                    }
                    yield f"data: {json.dumps(done_event, ensure_ascii=False)}\n\n"
                    yield "data: [DONE]\n\n"

            return prefill_sse_response(request, response_stream(), cancel_token, 1)

        # Non-streaming path.
        async with lock_ctx:
            cancel_token = CancellationToken()
            async with reserve_prefill_capacity(request, 1, cancel_token=cancel_token):
                state = prev_state if is_continuation else engine.model.generate_zero_state(0)
                texts, _finish_reasons = await run_sync_with_disconnect_watch(
                    request,
                    engine.batch_generate_state,
                    cancel_token=cancel_token,
                    prompts=[prompt],
                    state=state,
                    max_length=req.max_output_tokens,
                    temperature=req.temperature,
                    top_k=req.top_k,
                    top_p=req.top_p,
                    alpha_presence=req.alpha_presence,
                    alpha_frequency=req.alpha_frequency,
                    alpha_decay=req.alpha_decay,
                    stop_tokens=_RESPONSE_STOP_TOKENS,
                )
                generated_text = texts[0]
                if cancel_token.is_cancelled():
                    raise InferenceCancelled("request disconnected")

            if req.store:
                state_manager.put_state(response_id, state)

        output_tokens = len(engine.tokenizer.encode(generated_text)) if generated_text else 0
        usage = {
            "input_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "total_tokens": prompt_tokens + output_tokens,
        }
        return _build_response_payload(response_id, created_at, model_name, msg_id, generated_text, usage)

    except InferenceCancelled:
        return client_closed_response()
    except PrefillBszLimitExceeded as exc:
        return prefill_bsz_limit_response(exc)
    except Exception as exc:
        logger.error(f"[ERROR] /v1/responses: {exc}")
        return json_response(500, {"error": str(exc)})


@router.delete("/v1/responses/{response_id}")
async def delete_response(response_id: str, request: Request):
    """Explicitly discard a stored response state (frees its GPU/RAM/disk slot
    early instead of waiting for LRU eviction)."""
    password = request.app.state.password
    auth_error = check_openai_auth(request, {}, password)
    if auth_error is not None:
        return auth_error

    deleted = remove_session_from_any_level(response_id)
    return {"id": response_id, "object": "response", "deleted": bool(deleted)}
