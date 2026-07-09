from fastapi import APIRouter, Request

from infer.cancellation import CancellationToken, InferenceCancelled, PrefillBszLimitExceeded

from API_servers.router.common import (
    check_password,
    client_closed_response,
    prefill_bsz_limit_response,
    prefill_sse_response,
    reserve_prefill_capacity,
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
    engine = request.app.state.engine
    password = request.app.state.password
    body = await request.json()
    req = V2ChatRequest(**body)

    auth_error = check_password(req.password, password)
    if auth_error is not None:
        return auth_error

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
        )
        return prefill_sse_response(request, stream, cancel_token, len(req.contents))

    try:
        async with reserve_prefill_capacity(request, len(req.contents)):
            results = await run_sync_with_disconnect_watch(
                request,
                engine.batch_generate_v2,
                prompts=req.contents,
                max_length=req.max_tokens,
                temperature=req.temperature,
                top_k=req.top_k,
                top_p=req.top_p,
                alpha_presence=req.alpha_presence,
                alpha_frequency=req.alpha_frequency,
                alpha_decay=req.alpha_decay,
                stop_tokens=req.stop_tokens,
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
                "finish_reason": "stop",
            }
        )

    return {
        "id": "rwkv7-batch-v2",
        "object": "chat.completion",
        "model": req.model,
        "choices": choices,
    }
