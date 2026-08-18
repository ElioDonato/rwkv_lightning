import logging

logger = logging.getLogger("api.common")

import asyncio
import hmac
import json
import os
import traceback
from collections import defaultdict
from contextlib import asynccontextmanager
from contextlib import suppress

from fastapi import Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import ValidationError
from starlette.background import BackgroundTask
from starlette.concurrency import run_in_threadpool

from infer.cancellation import CancellationToken, InferenceCancelled, PrefillBszLimitExceeded


SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "close",
    "X-Accel-Buffering": "no",
}

# Per-session locks to prevent concurrent state corruption when two requests
# target the same session_id simultaneously (read-modify-write race on the
# state pool). Keyed by session_id string; locks are created on demand and
# never removed (bounded by the number of distinct session_ids seen).
_session_locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)


@asynccontextmanager
async def session_lock(session_id: str):
    """Serialize access to a single session's state across concurrent requests."""
    lock = _session_locks[session_id]
    await lock.acquire()
    try:
        yield
    finally:
        lock.release()


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
    if password and not hmac.compare_digest(str(body_password or ""), password):
        return json_response(401, {"error": "Unauthorized: invalid or missing password"})
    return None


def extract_bearer_token(request: Request):
    auth_header = request.headers.get("authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        return None
    return auth_header.split(" ", 1)[1].strip()


def check_openai_auth(request: Request, body: dict, password):
    """Timing-safe Bearer-token or body-password auth, OpenAI-client compatible.
    Shared by /openai/v1/* and /v1/responses (both accept either an
    `Authorization: Bearer <password>` header or a `password` body field)."""
    if not password:
        return None
    bearer_token = extract_bearer_token(request) or ""
    body_password = str(body.get("password") or "")
    if hmac.compare_digest(bearer_token, password) or hmac.compare_digest(body_password, password):
        return None
    return json_response(401, {"error": "Unauthorized: invalid or missing password"})


def parse_request_model(model_cls, body: dict):
    """Build a pydantic request model from a parsed JSON body, converting
    ValidationError (e.g. our max_tokens ceiling, or any other field
    constraint) into a clean 400 response instead of letting it propagate
    as an uncaught exception (which FastAPI turns into an opaque 500 with no
    indication of what was wrong with the request).

    Returns (model_instance, None) on success, or (None, error_response) on
    failure -- callers should check the second element and return it as-is.
    """
    if not isinstance(body, dict):
        return None, json_response(
            400, {"error": "invalid request: expected a JSON object"}
        )
    try:
        return model_cls(**body), None
    except ValidationError as exc:
        return None, json_response(400, {"error": f"invalid request: {exc}"})


class _DefaultSlotShim:
    """EngineSlot stand-in for an app that never installs a ModelManager
    (defensive fallback for isolated tests / harness apps).

    Routes read only ``.id`` / ``.engine`` / ``.embed`` / ``.fuse`` /
    ``.dynamic`` and call ``.ensure_wired()``. Delegating those to the
    historical ``app.state`` fields keeps the old single-engine behavior
    byte-identical when no manager is present."""

    def __init__(self, app_state):
        self.id = "default"
        self.engine = getattr(app_state, "engine", None)
        self.embed = getattr(app_state, "embed_aggregator", None)
        self.fuse = getattr(app_state, "chat_fuse_aggregator", None)
        self.dynamic = getattr(app_state, "chat_dynamic_decoder", None)
        self.ensure_wired = lambda: None


async def resolve_slot(request, model_field=None, role=None):
    """Resolve the :class:`EngineSlot` that should serve a request.

    ``request.app.state.model_manager`` (when present) picks a slot by the
    request's ``model`` field; omitted / empty / unknown model ids map to the
    default slot (or the role-specific default, e.g. ``role="embed"`` -> the
    designated embed model) and never raise. If no manager is installed, returns
    a shim over the historical ``app.state`` engine + aggregators. Calls
    ``slot.ensure_wired()`` on the running loop (idempotent) so the per-model
    decode aggregators exist before the route drives decode."""
    manager = getattr(request.app.state, "model_manager", None)
    if manager is None:
        return _DefaultSlotShim(request.app.state)
    if model_field and model_field in manager.ids():
        slot = await manager.get(model_field)
    elif role:
        slot = await manager.get(manager.endpoint_default(role))
    else:
        slot = await manager.get()
    slot.ensure_wired()
    return slot


def model_namespace(slot):
    """Stable per-model namespace for the state/prefix cache isolation: None for
    the default model (identity -> single-model byte-identical), else
    'id:size:mtime_ns' so a fine-tuned/state-tuned checkpoint swap auto-
    invalidates the cache. Handles the _DefaultSlotShim (no is_default/path)."""
    if getattr(slot, "is_default", True):
        return None
    path = getattr(slot, "path", None)
    if path:
        try:
            st = os.stat(path)
            return f"{slot.id}:{st.st_size}:{st.st_mtime_ns}"
        except OSError:
            pass
    return slot.id


def _default_engine(request, engine=None):
    """Resolve the engine for prefill/permit admission. An explicit ``engine``
    (the per-request slot's engine) wins; otherwise the default-slot engine.
    Returns None when the default model is not resident (unloaded) -- the
    admission helpers treat None as "skip accounting" rather than crash, so
    serving a NON-default model after the default is unloaded still works."""
    if engine is not None:
        return engine
    manager = getattr(request.app.state, "model_manager", None)
    if manager is None:
        return request.app.state.engine
    return manager.get_slot(manager.default_id).engine  # may be None


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


def allocate_next_dialogue_idx(app_state, state_manager, session_index: str, model=None) -> int:
    # Counter space is scoped by model so a second model can't reuse/collide with
    # the default model's dialogue indices (default/model-less maps to (None, si)).
    counter_key = (model, session_index)
    with app_state.dialogue_idx_lock:
        if counter_key in app_state.dialogue_idx_counters:
            next_idx = app_state.dialogue_idx_counters[counter_key]
            app_state.dialogue_idx_counters[counter_key] = next_idx + 1
            return next_idx

        indices = collect_session_indices(state_manager, session_index)
        max_idx = max(indices) if indices else 0
        next_idx = max_idx + 1
        app_state.dialogue_idx_counters[counter_key] = next_idx + 1
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


# Defaults for how long cleanup_disconnect_watcher retries cancel() before
# giving up on a stuck disconnect-watcher (see that function's docstring
# below for why this retry loop exists at all). Override via env vars if a
# deployment needs different behavior under heavier event-loop load; an
# abandoned watcher is harmless either way, it exits on its own once the
# client actually disconnects.
DISCONNECT_WATCHER_CLEANUP_TIMEOUT_S = float(
    os.environ.get("RWKV_DISCONNECT_WATCHER_CLEANUP_TIMEOUT_S", "1.0")
)
DISCONNECT_WATCHER_CLEANUP_POLL_INTERVAL_S = float(
    os.environ.get("RWKV_DISCONNECT_WATCHER_CLEANUP_POLL_INTERVAL_S", "0.05")
)


async def cleanup_disconnect_watcher(
    task,
    timeout: float = DISCONNECT_WATCHER_CLEANUP_TIMEOUT_S,
    poll_interval: float = DISCONNECT_WATCHER_CLEANUP_POLL_INTERVAL_S,
):
    # watch_disconnect polls request.is_disconnected(), which wraps its receive() call in an
    # already-cancelled anyio.CancelScope (see Starlette's Request.is_disconnected). If task.cancel()
    # lands while the watcher is inside that scope, anyio can treat the injected CancelledError as
    # satisfying its own (already-resolved) cancellation and swallow it, so the task keeps looping
    # instead of stopping -- an unbounded `await task` after cancel() then hangs forever (confirmed
    # live via asyncio task-stack introspection: the watcher was still asleep in its polling loop
    # after cancel() had already been called). Poll with repeated cancel() attempts instead of
    # awaiting the task directly, and give up after `timeout` rather than block the response --
    # an abandoned watcher is harmless, it will exit on its own once the client actually disconnects.
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not task.done():
        task.cancel()
        if loop.time() >= deadline:
            return
        await asyncio.sleep(poll_interval)
    with suppress(asyncio.CancelledError, Exception):
        task.result()


@asynccontextmanager
async def reserve_prefill_capacity(
    request: Request, request_bsz: int, cancel_token: CancellationToken | None = None,
    engine=None,
):
    engine = _default_engine(request, engine)
    queue_cancel_token = cancel_token or CancellationToken()
    watcher = asyncio.create_task(watch_disconnect(request, queue_cancel_token))
    permit = None
    try:
        if engine is not None:
            permit = await engine.acquire_prefill_permit(
                request_bsz=request_bsz,
                request_label=str(request.url.path),
                cancel_token=queue_cancel_token,
            )
            if queue_cancel_token.is_cancelled():
                raise InferenceCancelled("request disconnected while queued")
        yield permit  # None when no engine (unloaded default) -> no admission accounting
    finally:
        await cleanup_disconnect_watcher(watcher)
        if permit is not None and engine is not None:
            await engine.release_prefill_permit(
                request_bsz=request_bsz,
                request_label=str(request.url.path),
                ticket=permit["ticket"],
                permit=permit,
            )


async def run_sync_with_disconnect_watch(
    request: Request, func, cancel_token: CancellationToken | None = None, **kwargs
):
    # If the caller already has a watcher running against this request (e.g. via
    # reserve_prefill_capacity(..., cancel_token=cancel_token)), reuse its token instead of
    # starting a second concurrent watcher: two tasks polling request.is_disconnected() at once
    # can race inside Starlette's self-cancelling CancelScope and leave one watcher task stuck
    # forever, which then hangs whoever awaits it in cleanup_disconnect_watcher.
    owns_watcher = cancel_token is None
    if cancel_token is None:
        cancel_token = CancellationToken()
    watcher = asyncio.create_task(watch_disconnect(request, cancel_token)) if owns_watcher else None
    try:
        result = await run_in_threadpool(func, cancel_token=cancel_token, **kwargs)
        if cancel_token.is_cancelled():
            raise InferenceCancelled("request disconnected")
        return result
    finally:
        if watcher is not None:
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
            release_engine = stream_state.get("engine")  # the admitting engine
            if release_engine is not None:
                await release_engine.release_prefill_permit(
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
    engine=None,
):
    engine = _default_engine(request, engine)
    stream_state = {
        "permit": None,
        "watcher": None,
        "cleanup_task": None,
        "cleanup_done": False,
        "engine": engine,  # the admitting engine; used by the background release
    }

    async def body():
        try:
            stream_state["watcher"] = asyncio.create_task(
                watch_disconnect(request, cancel_token)
            )
            if engine is not None:
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
            logger.error(f"[SSE] unhandled exception in stream body:\n{traceback.format_exc()}")
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
