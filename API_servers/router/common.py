import asyncio
import json
import traceback
from contextlib import asynccontextmanager
from contextlib import suppress

from fastapi import Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.background import BackgroundTask
from starlette.concurrency import run_in_threadpool

from infer.cancellation import CancellationToken, InferenceCancelled, PrefillBszLimitExceeded


SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "close",
    "X-Accel-Buffering": "no",
}


def json_response(status_code: int, payload: dict):
    return JSONResponse(status_code=status_code, content=payload)


def client_closed_response():
    return Response(status_code=499)


def prefill_bsz_limit_payload(exc: PrefillBszLimitExceeded):
    return {
        "error": f"bsz overflow, Max bsz={exc.max_bsz}",
        "request_bsz": exc.request_bsz,
        "max_bsz": exc.max_bsz,
    }


def prefill_bsz_limit_response(exc: PrefillBszLimitExceeded):
    return json_response(400, prefill_bsz_limit_payload(exc))


def check_password(body_password, password):
    if password and body_password != password:
        return json_response(401, {"error": "Unauthorized: invalid or missing password"})
    return None


def normalize_state_prompts(prompts: list[str], reuse_existing_state: bool) -> list[str]:
    if not reuse_existing_state:
        return prompts

    normalized_prompts = []
    for prompt in prompts:
        if prompt and not prompt.startswith("\n\n"):
            normalized_prompts.append(f"\n\n{prompt}")
        else:
            normalized_prompts.append(prompt)
    return normalized_prompts


def collect_session_indices(state_manager, session_index: str) -> list[int]:
    prefix = f"{session_index}:"
    all_states = state_manager.list_all_states()
    indices = []
    for key in all_states["l1_cache"] + all_states["l2_cache"] + all_states["database"]:
        if key.startswith(prefix):
            tail = key[len(prefix) :]
            if tail.isdigit():
                indices.append(int(tail))
    return indices


def allocate_next_dialogue_idx(app_state, state_manager, session_index: str) -> int:
    with app_state.dialogue_idx_lock:
        if session_index in app_state.dialogue_idx_counters:
            next_idx = app_state.dialogue_idx_counters[session_index]
            app_state.dialogue_idx_counters[session_index] = next_idx + 1
            return next_idx

        indices = collect_session_indices(state_manager, session_index)
        max_idx = max(indices) if indices else 0
        next_idx = max_idx + 1
        app_state.dialogue_idx_counters[session_index] = next_idx + 1
        return next_idx


async def watch_disconnect(request: Request, cancel_token: CancellationToken):
    try:
        while not cancel_token.is_cancelled():
            if await request.is_disconnected():
                cancel_token.cancel()
                return
            await asyncio.sleep(0.05)
    except asyncio.CancelledError:
        raise


async def cleanup_disconnect_watcher(task):
    task.cancel()
    with suppress(asyncio.CancelledError):
        await task


@asynccontextmanager
async def reserve_prefill_capacity(
    request: Request, request_bsz: int, cancel_token: CancellationToken | None = None
):
    engine = request.app.state.engine
    queue_cancel_token = cancel_token or CancellationToken()
    watcher = asyncio.create_task(watch_disconnect(request, queue_cancel_token))
    permit = None
    try:
        permit = await engine.acquire_prefill_permit(
            request_bsz=request_bsz,
            request_label=str(request.url.path),
            cancel_token=queue_cancel_token,
        )
        if queue_cancel_token.is_cancelled():
            raise InferenceCancelled("request disconnected while queued")
        yield permit
    finally:
        await cleanup_disconnect_watcher(watcher)
        if permit is not None:
            await engine.release_prefill_permit(
                request_bsz=request_bsz,
                request_label=str(request.url.path),
                ticket=permit["ticket"],
                permit=permit,
            )


async def run_sync_with_disconnect_watch(request: Request, func, **kwargs):
    cancel_token = CancellationToken()
    watcher = asyncio.create_task(watch_disconnect(request, cancel_token))
    try:
        result = await run_in_threadpool(func, cancel_token=cancel_token, **kwargs)
        if cancel_token.is_cancelled():
            raise InferenceCancelled("request disconnected")
        return result
    finally:
        await cleanup_disconnect_watcher(watcher)


async def _cleanup_prefill_stream_response(
    request: Request,
    stream,
    request_bsz: int,
    stream_state: dict,
):
    current_task = asyncio.current_task()
    cleanup_task = stream_state.get("cleanup_task")
    if stream_state.get("cleanup_done"):
        return
    if cleanup_task is not None and cleanup_task is not current_task:
        with suppress(asyncio.CancelledError, Exception):
            await cleanup_task
        return

    stream_state["cleanup_task"] = current_task
    try:
        watcher = stream_state.get("watcher")
        if watcher is not None:
            await cleanup_disconnect_watcher(watcher)
            stream_state["watcher"] = None

        try:
            await stream.aclose()
        except asyncio.CancelledError:
            pass
        except Exception:
            pass
    finally:
        permit = stream_state.get("permit")
        if permit is not None:
            await request.app.state.engine.release_prefill_permit(
                request_bsz=request_bsz,
                request_label=str(request.url.path),
                ticket=permit["ticket"],
                permit=permit,
            )
            stream_state["permit"] = None
        stream_state["cleanup_done"] = True


def _schedule_prefill_stream_cleanup(
    request: Request,
    stream,
    request_bsz: int,
    stream_state: dict,
):
    if stream_state.get("cleanup_done") or stream_state.get("cleanup_task") is not None:
        return

    stream_state["cleanup_task"] = asyncio.create_task(
        _cleanup_prefill_stream_response(
            request,
            stream,
            request_bsz,
            stream_state,
        )
    )


def prefill_sse_response(
    request: Request,
    stream,
    cancel_token: CancellationToken,
    request_bsz: int,
    on_permit=None,
):
    engine = request.app.state.engine
    stream_state = {
        "permit": None,
        "watcher": None,
        "cleanup_task": None,
        "cleanup_done": False,
    }

    async def body():
        try:
            stream_state["watcher"] = asyncio.create_task(
                watch_disconnect(request, cancel_token)
            )
            stream_state["permit"] = await engine.acquire_prefill_permit(
                request_bsz=request_bsz,
                request_label=str(request.url.path),
                cancel_token=cancel_token,
            )
            if on_permit is not None:
                # Lets a caller that pre-created the streaming generator (before
                # admission was possible -- the generator itself doesn't start
                # executing until iterated below) hand the now-acquired permit
                # into that generator's own closure, e.g. so
                # big_batch_stream's decode-time row compaction can call
                # engine.release_prefill_capacity() against this exact permit
                # as individual rows finish, instead of only releasing the
                # full reservation at the very end of the whole request.
                on_permit(stream_state["permit"])
            if cancel_token.is_cancelled():
                raise InferenceCancelled("request disconnected while queued")

            async for chunk in stream:
                if cancel_token.is_cancelled():
                    break
                yield chunk
                if isinstance(chunk, str) and chunk.strip() == "data: [DONE]":
                    break
        except PrefillBszLimitExceeded as exc:
            cancel_token.cancel()
            yield f"data: {json.dumps(prefill_bsz_limit_payload(exc), ensure_ascii=False)}\n\n"
        except InferenceCancelled:
            cancel_token.cancel()
        except asyncio.CancelledError:
            cancel_token.cancel()
            raise
        except Exception:
            # Any other failure (e.g. a bug in the decode loop) would
            # otherwise propagate past this generator uncaught, which
            # Starlette/uvicorn logs loudly but delivers to the client as
            # an abruptly-closed connection: no "data: [DONE]" marker and
            # no error payload, so a naive SSE consumer that just appends
            # `delta.content` until EOF can't distinguish a truncated
            # response from a complete one. Log with the full traceback
            # (matching how this was previously surfaced, so nothing about
            # visibility in the server log regresses) and send an explicit
            # error chunk before terminating, so the stream always ends
            # with a client-visible signal one way or another.
            cancel_token.cancel()
            print(f"[SSE] unhandled exception in stream body:\n{traceback.format_exc()}")
            yield f"data: {json.dumps({'error': {'message': 'internal error during generation', 'type': 'stream_error'}}, ensure_ascii=False)}\n\n"
        finally:
            _schedule_prefill_stream_cleanup(
                request,
                stream,
                request_bsz,
                stream_state,
            )

    return StreamingResponse(
        body(),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
        background=BackgroundTask(
            _cleanup_prefill_stream_response,
            request,
            stream,
            request_bsz,
            stream_state,
        ),
    )


def extract_sse_payload(item: str) -> str | None:
    if not isinstance(item, str) or not item.startswith("data: "):
        return None
    return item[6:].strip()


def emit_finish_reason_chunk(response_id: str, created: int, model_name: str, finish_reason: str):
    chunk = {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model_name,
        "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
    }
    return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
