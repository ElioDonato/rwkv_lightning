import logging

logger = logging.getLogger("api.openai")

import asyncio
import json
import os
import time
from contextlib import suppress
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pydantic import ValidationError

from infer.cancellation import CancellationToken, InferenceCancelled, PrefillBszLimitExceeded
from state_manager.state_pool import get_state_manager

from API_servers.router.common import (
    SSE_HEADERS,
    check_openai_auth as common_check_openai_auth,
    cleanup_disconnect_watcher,
    client_closed_response,
    emit_finish_reason_chunk,
    extract_bearer_token as common_extract_bearer_token,
    extract_sse_payload,
    json_response,
    prefill_bsz_limit_response,
    prefill_sse_response,
    reserve_prefill_capacity,
    watch_disconnect,
)
from API_servers.router.schemas import ChatRequest

from settings import settings


router = APIRouter()


def normalize_message_content(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(item.get("text", ""))
            elif isinstance(item, str):
                text_parts.append(item)
        return "".join(text_parts)
    if content is None:
        return ""
    return str(content)


def sanitize_text_block(content) -> str:
    normalized = normalize_message_content(content)
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    lines = []
    for line in normalized.split("\n"):
        lines.append(line.rstrip())
    # Strip leading/trailing blank lines but preserve interior blanks and indentation
    text = "\n".join(lines)
    return text.strip("\n")


def collect_openai_prompt_parts(body: dict) -> tuple[str, list[str]]:
    messages = body.get("messages") or []
    contents = body.get("contents") or []

    system_parts = []
    transcript_parts = []

    system_field = sanitize_text_block(body.get("system"))
    if system_field:
        system_parts.append(system_field)

    for message in messages:
        if not isinstance(message, dict):
            continue

        role = str(message.get("role", "user")).lower()
        content = sanitize_text_block(message.get("content", ""))

        if role in {"system", "developer"}:
            if content:
                system_parts.append(content)
            continue

        if role == "user":
            if content:
                transcript_parts.append(f"User: {content}")
            continue

        if role == "assistant":
            if content:
                transcript_parts.append(f"Assistant: {content}")
            continue

    if contents:
        content_prompt = sanitize_text_block(contents[0])
        if content_prompt:
            transcript_parts.append(f"User: {content_prompt}")

    system_text = "\n".join(part for part in system_parts if part).strip()
    transcript_parts = [part for part in transcript_parts if part]
    return system_text, transcript_parts


def extract_openai_prompt(body: dict) -> str:
    system_text, transcript_parts = collect_openai_prompt_parts(body)
    prompt_parts = []
    if system_text:
        prompt_parts.append(system_text)
    prompt_parts.extend(transcript_parts)
    return "\n".join(part for part in prompt_parts if part).strip()


def format_openai_prompt(body: dict, enable_think: bool) -> str:
    system_text, transcript_parts = collect_openai_prompt_parts(body)
    prompt_parts = []
    if system_text:
        prompt_parts.append(f"System: {system_text}")
    prompt_parts.extend(transcript_parts)

    prompt_text = "\n\n".join(part for part in prompt_parts if part).strip()
    if not prompt_text:
        raise ValueError("OpenAI chat completions require system or user text")

    if enable_think:
        return f"{prompt_text}\n\nAssistant: <think"
    return f"{prompt_text}\n\nAssistant: <think>\n</think>\n"


def build_openai_usage(tokenizer, prompt_text: str, completion_text: str) -> dict:
    prompt_tokens = len(tokenizer.encode(prompt_text))
    completion_tokens = len(tokenizer.encode(completion_text)) if completion_text else 0
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def build_internal_chat_request(body: dict, prompt: str) -> dict:
    stream = body.get("stream", False)
    chunk_size = body.get("chunk_size")
    if chunk_size is None:
        chunk_size = 1 if stream else 16

    return {
        "model": body.get("model", "rwkv7"),
        "contents": [prompt],
        "messages": body.get("messages", []),
        "system": body.get("system"),
        "max_tokens": body.get("max_tokens", settings.chat_max_tokens_default),
        "stop_tokens": body.get("stop_tokens", ["\nUser:"]),
        "temperature": body.get("temperature", 1.0),
        "top_k": body.get("top_k", settings.fuse_sampler_top_k),
        "top_p": body.get("top_p", settings.fuse_sampler_top_p),
        "stream": stream,
        "pad_zero": body.get("pad_zero", False),
        "alpha_presence": body.get("alpha_presence", settings.fuse_sampler_alpha_presence),
        "alpha_frequency": body.get("alpha_frequency", settings.fuse_sampler_alpha_frequency),
        "alpha_decay": body.get("alpha_decay", settings.fuse_sampler_alpha_decay),
        "enable_think": body.get("enable_think", False),
        "chunk_size": chunk_size,
        "password": body.get("password"),
        "session_id": body.get("session_id"),
        "use_prefix_cache": body.get("use_prefix_cache", True),
    }


def build_openai_message_response(
    result_text: str, finish_reason: str, body: dict
) -> tuple[dict[str, Any], str]:
    return {"role": "assistant", "content": result_text}, finish_reason


# extract_bearer_token / check_openai_auth now live in API_servers.router.common
# (shared with /v1/responses and any other Bearer-token-auth'd endpoint). Kept
# as re-exports here since existing code/tests may still import them from
# this module.
extract_bearer_token = common_extract_bearer_token
check_openai_auth = common_check_openai_auth


async def _relay_openai_stream(stream, response_id, created, model_name, cancel_token):
    """Relay an opaque SSE chunk stream (single_infer_stream or a fuse-aggregator
    stream -- both index-0, ``data: ...`` lines, ending with [DONE] for the
    solo path / silent end-for-fused) into OpenAI chat.completion.chunk packets,
    re-stamping id/created/model and emitting the finish chunk + [DONE].
    Closing the stream (aclose) is owned here so both solo and fused paths
    release the row uniformly."""
    emitted_finish_reason = False
    start_chunk = {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model_name,
        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
    }
    yield f"data: {json.dumps(start_chunk, ensure_ascii=False)}\n\n"

    try:
        async for item in stream:
            payload = extract_sse_payload(item)
            if payload is None:
                continue

            if payload == "[DONE]":
                if not emitted_finish_reason and not cancel_token.is_cancelled():
                    yield emit_finish_reason_chunk(response_id, created, model_name, "stop")
                break

            try:
                chunk_payload = json.loads(payload)
            except json.JSONDecodeError:
                continue

            choices = chunk_payload.get("choices") or []
            if not choices:
                continue

            finish_reason = choices[0].get("finish_reason")
            if finish_reason is not None:
                if not cancel_token.is_cancelled():
                    yield emit_finish_reason_chunk(
                        response_id,
                        created,
                        model_name,
                        finish_reason,
                    )
                    emitted_finish_reason = True
                continue

            content = choices[0].get("delta", {}).get("content")
            if not content or cancel_token.is_cancelled():
                continue

            chunk = {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_name,
                "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}],
            }
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
    finally:
        await stream.aclose()

    if not cancel_token.is_cancelled():
        yield "data: [DONE]\n\n"


async def stream_openai_chunks(
    engine,
    req,
    prompt_formatted: str,
    response_id: str,
    created: int,
    model_name: str,
    cancel_token: CancellationToken,
    prefix_cache_manager=None,
):
    stream = engine.single_infer_stream(
        prompt=prompt_formatted,
        max_length=req.max_tokens,
        temperature=req.temperature,
        top_k=req.top_k,
        top_p=req.top_p,
        alpha_presence=req.alpha_presence,
        alpha_frequency=req.alpha_frequency,
        alpha_decay=req.alpha_decay,
        stop_tokens=req.stop_tokens,
        chunk_size=req.chunk_size,
        prefix_cache_manager=prefix_cache_manager,
        cancel_token=cancel_token,
    )
    async for item in _relay_openai_stream(stream, response_id, created, model_name, cancel_token):
        yield item


async def collect_fuse_nonstream(fuse_stream) -> tuple[str, str]:
    """Accumulate a fuse stream into (text, finish_reason) for non-stream chat.
    The fuse stream carries content deltas and a finish chunk in the same
    format single_infer_stream emits, so accumulation reproduces the exact text
    and finish reason a non-stream single_infer call would have produced."""
    text_parts = []
    finish_reason = "length"
    async for item in fuse_stream:
        payload = extract_sse_payload(item)
        if payload is None or payload == "[DONE]":
            continue
        try:
            chunk_payload = json.loads(payload)
        except json.JSONDecodeError:
            continue
        choices = chunk_payload.get("choices") or []
        if not choices:
            continue
        fr = choices[0].get("finish_reason")
        if fr is not None:
            finish_reason = fr
        content = choices[0].get("delta", {}).get("content")
        if content:
            text_parts.append(content)
    return "".join(text_parts), finish_reason


def fuse_sse_response(request: Request, stream_gen, cancel_token: CancellationToken):
    """StreamingResponse for the fuse path. Unlike prefill_sse_response it does
    NOT own a prefill-admission permit -- the ChatFuseAggregator owns the fused
    batch's permit. It only watches for client disconnect (cancelling the token
    the relay checks) and closes the fuse stream via the relay generator's
    finally."""

    async def body():
        watcher = asyncio.create_task(watch_disconnect(request, cancel_token))
        try:
            async for chunk in stream_gen:
                if cancel_token.is_cancelled():
                    break
                yield chunk
        finally:
            await cleanup_disconnect_watcher(watcher)
            if cancel_token.is_cancelled():
                # On disconnect the relay's finally closes the fuse stream (drops
                # just this row); if the relay was left mid-iteration above, force
                # its aclose so the row is released promptly.
                with suppress(asyncio.CancelledError, Exception):
                    await stream_gen.aclose()

    return StreamingResponse(
        body(),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )


@router.get("/openai/v1/models")
async def openai_list_models(request: Request):
    engine = request.app.state.engine
    password = request.app.state.password
    auth_error = check_openai_auth(request, {}, password)
    if auth_error is not None:
        return auth_error

    model_name = os.path.basename(f"{engine.args.MODEL_NAME}")
    return {
        "object": "list",
        "data": [{"id": model_name, "object": "model", "owned_by": "rwkv_lightning"}],
    }


@router.post("/openai/v1/chat/completions")
async def openai_chat_completions(request: Request):
    engine = request.app.state.engine
    password = request.app.state.password
    try:
        body = await request.json()
        auth_error = check_openai_auth(request, body, password)
        if auth_error is not None:
            return auth_error

        prompt = extract_openai_prompt(body)
        if not prompt and not (body.get("messages") or []):
            return json_response(400, {"error": "Empty prompt"})

        req = ChatRequest(**build_internal_chat_request(body, prompt))
        prompt_formatted = format_openai_prompt(body, req.enable_think)

        response_id = f"chatcmpl-{os.urandom(12).hex()}"
        created = int(time.time())
        model_name = os.path.basename(f"{engine.args.MODEL_NAME}")
        prefix_cache_manager = get_state_manager() if req.use_prefix_cache else None

        # Opt-in decode combine-queue (CHAT_FUSE): when enabled, route the
        # request through the ChatFuseAggregator (solo head-fire or fused
        # multi-row big_batch_stream decode) instead of the per-request
        # single_infer path. The aggregator owns the batch's prefill-admission
        # permit, so these branches do NOT use reserve_prefill_capacity /
        # prefill_sse_response. When disabled, the original path is used
        # unchanged (byte-identical).
        fuse = getattr(request.app.state, "chat_fuse_aggregator", None)
        # Opt-in DYNAMIC decode batching (RWKV_DYNAMIC_BATCH, Phase 4 sub-step
        # 2): when enabled it takes precedence and routes EVERY chat request
        # through the DynamicBatchDecoder -- a shared multi-row decode with
        # per-row sampling that accepts ANY request (no homogeneity check, each
        # row keeps its own sampler controls). When disabled this branch is
        # skipped, so behavior is byte-identical to today (the original
        # single_infer path, or the fuse path below when that flag is on).
        dyn = getattr(request.app.state, "chat_dynamic_decoder", None)
        if dyn is not None and dyn.enabled:
            cancel_token = CancellationToken()
            dyn_stream = await dyn.submit(
                engine,
                prompt_formatted,
                req.max_tokens,
                req.temperature,
                req.stop_tokens,
                req.chunk_size,
                top_k=req.top_k,
                top_p=req.top_p,
                alpha_presence=req.alpha_presence,
                alpha_frequency=req.alpha_frequency,
                alpha_decay=req.alpha_decay,
                cancel_token=cancel_token,
            )
            if req.stream:
                stream = _relay_openai_stream(
                    dyn_stream, response_id, created, model_name, cancel_token
                )
                return fuse_sse_response(request, stream, cancel_token)

            collect_task = asyncio.create_task(collect_fuse_nonstream(dyn_stream))
            watcher = asyncio.create_task(watch_disconnect(request, cancel_token))
            pending = set()
            try:
                done, pending = await asyncio.wait(
                    {collect_task, watcher}, return_when=asyncio.FIRST_COMPLETED
                )
                if collect_task not in done:
                    # watcher finished -> client disconnected
                    collect_task.cancel()
                    with suppress(asyncio.CancelledError, Exception):
                        await collect_task
                    await dyn_stream.aclose()
                    raise InferenceCancelled("request disconnected")
                result_text, finish_reason = await collect_task
            finally:
                for t in pending:
                    t.cancel()
                with suppress(asyncio.CancelledError, Exception):
                    for t in pending:
                        await t

            message, response_finish_reason = build_openai_message_response(
                result_text, finish_reason, body
            )
            return {
                "id": response_id,
                "object": "chat.completion",
                "created": created,
                "model": model_name,
                "choices": [
                    {
                        "index": 0,
                        "message": message,
                        "finish_reason": response_finish_reason,
                    }
                ],
                "usage": build_openai_usage(engine.tokenizer, prompt_formatted, result_text),
            }
        elif fuse is not None and fuse.enabled:
            # Fusion-eligibility for prefix caching (CR1 HIGH-2): fuse=OFF (and
            # req.use_prefix_cache above) defaults use_prefix_cache=True, which
            # would make EVERY default chat request non-fusable (the fused path
            # has no prefix-cache wiring) and the combine-queue would never
            # engage. Under the fuse flag the DEFAULT workload (body omits
            # use_prefix_cache) is opted INTO fusion -- no prefix caching, so a
            # homogeneous default request is fusable -- while an EXPLICIT
            # use_prefix_cache=True keeps its exact per-request prefix cache and
            # is served by the aggregator's faithful SOLO path (matching
            # fuse=OFF). Prefix caching per se is served by the solo path, not
            # the fused path.
            fuse_prefix_cache = body.get("use_prefix_cache", settings.fuse_prefix_cache_default)
            fuse_prefix_cache_manager = get_state_manager() if fuse_prefix_cache else None
            cancel_token = CancellationToken()
            fuse_stream = await fuse.submit(
                prompt_formatted,
                req.max_tokens,
                req.temperature,
                req.stop_tokens,
                req.chunk_size,
                top_k=req.top_k,
                top_p=req.top_p,
                alpha_presence=req.alpha_presence,
                alpha_frequency=req.alpha_frequency,
                alpha_decay=req.alpha_decay,
                use_prefix_cache=fuse_prefix_cache,
                prefix_cache_manager=fuse_prefix_cache_manager,
            )
            if req.stream:
                stream = _relay_openai_stream(
                    fuse_stream, response_id, created, model_name, cancel_token
                )
                return fuse_sse_response(request, stream, cancel_token)

            collect_task = asyncio.create_task(collect_fuse_nonstream(fuse_stream))
            watcher = asyncio.create_task(watch_disconnect(request, cancel_token))
            pending = set()
            try:
                done, pending = await asyncio.wait(
                    {collect_task, watcher}, return_when=asyncio.FIRST_COMPLETED
                )
                if collect_task not in done:
                    # watcher finished -> client disconnected
                    collect_task.cancel()
                    with suppress(asyncio.CancelledError, Exception):
                        await collect_task
                    await fuse_stream.aclose()
                    raise InferenceCancelled("request disconnected")
                result_text, finish_reason = await collect_task
            finally:
                for t in pending:
                    t.cancel()
                with suppress(asyncio.CancelledError, Exception):
                    for t in pending:
                        await t

            message, response_finish_reason = build_openai_message_response(
                result_text, finish_reason, body
            )
            return {
                "id": response_id,
                "object": "chat.completion",
                "created": created,
                "model": model_name,
                "choices": [
                    {
                        "index": 0,
                        "message": message,
                        "finish_reason": response_finish_reason,
                    }
                ],
                "usage": build_openai_usage(engine.tokenizer, prompt_formatted, result_text),
            }

        if req.stream:
            cancel_token = CancellationToken()
            stream = stream_openai_chunks(
                engine,
                req,
                prompt_formatted,
                response_id,
                created,
                model_name,
                cancel_token,
                prefix_cache_manager,
            )
            return prefill_sse_response(request, stream, cancel_token, 1)

        cancel_token = CancellationToken()
        async with reserve_prefill_capacity(request, 1, cancel_token=cancel_token):
            result_text, finish_reason = await engine.single_infer(
                prompt=prompt_formatted,
                max_length=req.max_tokens,
                temperature=req.temperature,
                top_k=req.top_k,
                top_p=req.top_p,
                alpha_presence=req.alpha_presence,
                alpha_frequency=req.alpha_frequency,
                alpha_decay=req.alpha_decay,
                stop_tokens=req.stop_tokens,
                prefix_cache_manager=prefix_cache_manager,
                cancel_token=cancel_token,
            )
            if cancel_token.is_cancelled():
                raise InferenceCancelled("request disconnected")

        message, response_finish_reason = build_openai_message_response(
            result_text, finish_reason, body
        )
        return {
            "id": response_id,
            "object": "chat.completion",
            "created": created,
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": response_finish_reason,
                }
            ],
            "usage": build_openai_usage(engine.tokenizer, prompt_formatted, result_text),
        }
    except InferenceCancelled:
        return client_closed_response()
    except PrefillBszLimitExceeded as exc:
        return prefill_bsz_limit_response(exc)
    except ValidationError as exc:
        return json_response(400, {"error": f"invalid request: {exc}"})
    except json.JSONDecodeError as exc:
        return json_response(400, {"error": f"Invalid JSON: {str(exc)}"})
    except Exception as exc:
        logger.error(f"[ERROR] /openai/v1/chat/completions: {exc}")
        return json_response(500, {"error": str(exc)})
