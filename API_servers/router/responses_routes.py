"""
OpenAI Responses API compatible endpoint (/v1/responses).

The Responses API is stateful — it supports `previous_response_id` to chain
multi-turn conversations. This maps naturally to RWKV's recurrent state:
we store the RWKV state keyed by response ID, and resume from it on the
next turn instead of re-processing the full conversation history.

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
import logging
import os
import time
import uuid

from fastapi import APIRouter, Request

from infer.cancellation import CancellationToken, InferenceCancelled, PrefillBszLimitExceeded
from state_manager.state_pool import get_state_manager

from API_servers.router.common import (
    check_password,
    client_closed_response,
    json_response,
    prefill_bsz_limit_response,
    prefill_sse_response,
    reserve_prefill_capacity,
    run_sync_with_disconnect_watch,
)

logger = logging.getLogger("api.responses")

router = APIRouter()

# In-memory store for response states (response_id -> RWKV state).
# Bounded: evicts oldest entries when exceeding capacity.
_response_states: dict[str, dict] = {}
_RESPONSE_STATE_CAPACITY = 64


def _store_response_state(response_id: str, state, token_count: int):
    """Store RWKV state for a response ID, evicting oldest if over capacity."""
    if len(_response_states) >= _RESPONSE_STATE_CAPACITY:
        oldest_key = next(iter(_response_states))
        old = _response_states.pop(oldest_key)
        del old["state"]
    _response_states[response_id] = {
        "state": state,
        "token_count": token_count,
        "created_at": time.time(),
    }


def _get_response_state(response_id: str):
    """Retrieve stored RWKV state for a response ID."""
    entry = _response_states.get(response_id)
    if entry is None:
        return None, 0
    return entry["state"], entry["token_count"]


def _build_prompt_from_input(input_data, instructions: str | None) -> str:
    """Convert Responses API input (string or message array) to a prompt string."""
    parts = []

    if instructions:
        parts.append(f"System: {instructions}")

    if isinstance(input_data, str):
        parts.append(f"User: {input_data}")
    elif isinstance(input_data, list):
        for item in input_data:
            if isinstance(item, str):
                parts.append(f"User: {item}")
            elif isinstance(item, dict):
                role = item.get("role", "user").capitalize()
                content = item.get("content", "")
                if isinstance(content, list):
                    # Multimodal content array
                    text_parts = [
                        c.get("text", "") for c in content
                        if isinstance(c, dict) and c.get("type") == "text"
                    ]
                    content = " ".join(text_parts)
                if content:
                    parts.append(f"{role}: {content}")

    return "\n\n".join(parts) + "\n\nAssistant:"


def _check_auth(request: Request, body: dict, password) -> object | None:
    """Check Bearer token auth (OpenAI-style) or body password."""
    if password is None:
        return None
    auth_header = request.headers.get("authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header[7:]
        if token == password:
            return None
        return json_response(401, {"error": "Invalid API key"})
    # Fall back to body password
    if body.get("password") == password:
        return None
    return json_response(401, {"error": "Authentication required"})


@router.post("/v1/responses")
async def create_response(request: Request):
    engine = request.app.state.engine
    password = request.app.state.password

    try:
        body = await request.json()
    except Exception:
        return json_response(400, {"error": "Invalid JSON in request body"})

    auth_error = _check_auth(request, body, password)
    if auth_error is not None:
        return auth_error

    # Parse request fields
    input_data = body.get("input", "")
    instructions = body.get("instructions")
    previous_response_id = body.get("previous_response_id")
    max_output_tokens = body.get("max_output_tokens", body.get("max_tokens", 1024))
    temperature = body.get("temperature", 1.0)
    top_p = body.get("top_p", 0.6)
    stream = body.get("stream", False)
    store = body.get("store", True)

    if not input_data:
        return json_response(400, {"error": "input is required"})

    # Build prompt
    prompt = _build_prompt_from_input(input_data, instructions)

    # Generate response ID
    response_id = f"resp_{uuid.uuid4().hex[:24]}"
    msg_id = f"msg_{uuid.uuid4().hex[:24]}"
    created_at = int(time.time())
    model_name = os.path.basename(engine.args.MODEL_NAME)

    # Check for previous response state (multi-turn)
    prev_state = None
    prev_token_count = 0
    if previous_response_id:
        prev_state, prev_token_count = _get_response_state(previous_response_id)
        if prev_state is None:
            return json_response(404, {
                "error": f"Response '{previous_response_id}' not found or expired"
            })

    state_manager = get_state_manager()

    try:
        if stream:
            cancel_token = CancellationToken()

            async def response_stream():
                state = prev_state
                if state is None:
                    state = engine.model.generate_zero_state(0)

                # Prefill
                encoded = engine.tokenizer.encode(prompt)
                if prev_state is not None and prev_token_count > 0:
                    # Only process new tokens (after previous state)
                    out = engine._forward_tokens_chunked(encoded, state, cancel_token=cancel_token)
                else:
                    out = engine._forward_tokens_chunked(encoded, state, cancel_token=cancel_token)

                # Decode loop
                stop_state = engine._create_stop_state(["\nUser:"])
                generated_text = ""
                output_tokens = 0
                sample_module = __import__("infer.inference_deps", fromlist=["get_sample"])
                import random
                sample_rand_states = sample_module.get_sample().setup_rand(
                    random.randint(0, 2**63 - 1), 1
                )
                import infer.inference_deps as inference_deps
                penalties = inference_deps.get_torch().zeros(1, out.size(-1), device=out.device)

                if out.dim() == 1:
                    out = out.unsqueeze(0)

                for _ in range(max_output_tokens):
                    if cancel_token.is_cancelled():
                        break
                    new_tokens = sample_module.get_sample().batch_sampling_repetition_temperature_topk_topp(
                        out, penalties, sample_rand_states,
                        1.0, 0.1, 0.996,
                        temperature, 50, top_p,
                    ).tolist()
                    tok = new_tokens[0]

                    content, should_stop = engine._ingest_token_with_stop(stop_state, tok)
                    if content:
                        generated_text += content
                        # Emit streaming chunk
                        chunk = {
                            "type": "response.output_text.delta",
                            "delta": content,
                            "item_id": msg_id,
                            "output_index": 0,
                            "content_index": 0,
                        }
                        yield f"data: {__import__('json').dumps(chunk)}\n\n"

                    output_tokens += 1
                    if should_stop:
                        break

                    out = engine._forward_tokens_chunked([tok], state, cancel_token=cancel_token)
                    if out.dim() == 1:
                        out = out.unsqueeze(0)

                generated_text += engine._flush_stop_state(stop_state, final=True)

                # Store state for multi-turn
                if store:
                    _store_response_state(response_id, state, prev_token_count + len(encoded) + output_tokens)

                # Emit completion event
                done_event = {
                    "type": "response.completed",
                    "response": {
                        "id": response_id,
                        "object": "response",
                        "created_at": created_at,
                        "model": model_name,
                        "output": [{
                            "id": msg_id,
                            "type": "message",
                            "status": "completed",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": generated_text, "annotations": []}],
                        }],
                        "usage": {
                            "input_tokens": len(encoded),
                            "output_tokens": output_tokens,
                            "total_tokens": len(encoded) + output_tokens,
                        },
                    },
                }
                yield f"data: {__import__('json').dumps(done_event)}\n\n"
                yield "data: [DONE]\n\n"

            return prefill_sse_response(request, response_stream(), cancel_token, 1)

        # Non-streaming path
        cancel_token = CancellationToken()
        async with reserve_prefill_capacity(request, 1, cancel_token=cancel_token):
            state = prev_state
            if state is None:
                state = engine.model.generate_zero_state(0)

            encoded = engine.tokenizer.encode(prompt)
            out = engine._forward_tokens_chunked(encoded, state, cancel_token=cancel_token)

            stop_state = engine._create_stop_state(["\nUser:"])
            generated_text = ""
            output_tokens = 0
            import random
            import infer.inference_deps as inference_deps
            sample_module = inference_deps.get_sample()
            sample_rand_states = sample_module.setup_rand(random.randint(0, 2**63 - 1), 1)
            penalties = inference_deps.get_torch().zeros(1, out.size(-1), device=out.device)

            if out.dim() == 1:
                out = out.unsqueeze(0)

            for _ in range(max_output_tokens):
                if cancel_token.is_cancelled():
                    raise InferenceCancelled("request disconnected")
                new_tokens = sample_module.batch_sampling_repetition_temperature_topk_topp(
                    out, penalties, sample_rand_states,
                    1.0, 0.1, 0.996,
                    temperature, 50, top_p,
                ).tolist()
                tok = new_tokens[0]

                content, should_stop = engine._ingest_token_with_stop(stop_state, tok)
                if content:
                    generated_text += content

                output_tokens += 1
                if should_stop:
                    break

                out = engine._forward_tokens_chunked([tok], state, cancel_token=cancel_token)
                if out.dim() == 1:
                    out = out.unsqueeze(0)

            generated_text += engine._flush_stop_state(stop_state, final=True)

        # Store state for multi-turn
        if store:
            _store_response_state(response_id, state, prev_token_count + len(encoded) + output_tokens)

        finish_reason = "stop" if output_tokens < max_output_tokens else "max_output_tokens"

        return {
            "id": response_id,
            "object": "response",
            "created_at": created_at,
            "model": model_name,
            "status": "completed",
            "output": [{
                "id": msg_id,
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": generated_text, "annotations": []}],
            }],
            "usage": {
                "input_tokens": len(encoded),
                "output_tokens": output_tokens,
                "total_tokens": len(encoded) + output_tokens,
            },
        }

    except InferenceCancelled:
        return client_closed_response()
    except PrefillBszLimitExceeded as exc:
        return prefill_bsz_limit_response(exc)
    except Exception as exc:
        logger.error(f"[ERROR] /v1/responses: {exc}")
        return json_response(500, {"error": str(exc)})
