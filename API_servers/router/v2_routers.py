from fastapi import APIRouter, Request

from infer.cancellation import CancellationToken, InferenceCancelled, PrefillBszLimitExceeded
from state_manager.state_pool import get_state_manager

from API_servers.router.common import (
    check_password,
    client_closed_response,
    model_namespace,
    parse_request_model,
    prefill_bsz_limit_response,
    prefill_sse_response,
    reserve_prefill_capacity,
    resolve_slot,
    run_sync_with_disconnect_watch,
)
from API_servers.router.schemas import ChatRequest


router = APIRouter()


class V2ChatRequest(ChatRequest):
    top_k: int = 500
    top_p: float = 0.5
    alpha_presence: float = 1.0
    alpha_frequency: float = 0.1
    alpha_decay: float = 0.99


@router.post("/v2/chat/completions")
async def chat_completions_v2(request: Request):
    password = request.app.state.password
    body = await request.json()
    req, parse_error = parse_request_model(V2ChatRequest, body)
    if parse_error is not None:
        return parse_error

    auth_error = check_password(req.password, password)
    if auth_error is not None:
        return auth_error

    slot = await resolve_slot(request, body.get("model"))
    engine = slot.engine

    prefix_cache_manager = get_state_manager(model_namespace(slot)) if req.use_prefix_cache else None

    if req.stream:
        cancel_token = CancellationToken()
        stream = engine.batch_infer_stream_v2(
            prompts=req.contents,
            max_length=req.max_tokens,
            temperature=req.temperature,
            top_k=req.top_k,
            top_p=req.top_p,
            alpha_presence=req.alpha_presence,
            alpha_frequency=req.alpha_frequency,
            alpha_decay=req.alpha_decay,
            stop_tokens=req.stop_tokens,
            chunk_size=req.chunk_size,
            cancel_token=cancel_token,
            prefix_cache_manager=prefix_cache_manager,
        )
        return prefill_sse_response(request, stream, cancel_token, len(req.contents), engine=engine)

    try:
        cancel_token = CancellationToken()
        async with reserve_prefill_capacity(request, len(req.contents), cancel_token=cancel_token, engine=engine):
            results, finish_reasons = await run_sync_with_disconnect_watch(
                request,
                engine.batch_generate_v2,
                cancel_token=cancel_token,
                prompts=req.contents,
                max_length=req.max_tokens,
                temperature=req.temperature,
                top_k=req.top_k,
                top_p=req.top_p,
                alpha_presence=req.alpha_presence,
                alpha_frequency=req.alpha_frequency,
                alpha_decay=req.alpha_decay,
                stop_tokens=req.stop_tokens,
                prefix_cache_manager=prefix_cache_manager,
            )
    except InferenceCancelled:
        return client_closed_response()
    except PrefillBszLimitExceeded as exc:
        return prefill_bsz_limit_response(exc)

    choices = []
    for i, text in enumerate(results):
        choices.append(
            {
                "index": i,
                "message": {"role": "assistant", "content": text},
                "finish_reason": finish_reasons[i],
            }
        )

    return {
        "id": "rwkv7-batch-v2",
        "object": "chat.completion",
        "model": req.model,
        "choices": choices,
    }
