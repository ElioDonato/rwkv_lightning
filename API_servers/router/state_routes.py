import logging

logger = logging.getLogger("api.state")

import json
from datetime import datetime

from fastapi import APIRouter, Request

from infer.cancellation import CancellationToken, InferenceCancelled, PrefillBszLimitExceeded
from state_manager.state_pool import get_state_manager, remove_session_from_any_level

from API_servers.router.common import (
    allocate_next_dialogue_idx,
    check_password,
    client_closed_response,
    json_response,
    normalize_state_prompts,
    parse_request_model,
    prefill_bsz_limit_response,
    prefill_sse_response,
    reserve_prefill_capacity,
    resolve_slot,
    run_sync_with_disconnect_watch,
    session_lock,
)
from API_servers.router.schemas import ChatRequest


router = APIRouter()


@router.post("/state/chat/completions")
async def state_chat_completions(request: Request):
    password = request.app.state.password
    body = await request.json()
    req, parse_error = parse_request_model(ChatRequest, body)
    if parse_error is not None:
        return parse_error
    session_id = req.session_id
    if not session_id:
        return json_response(400, {"error": "Missing session_id parameter"})

    if len(req.contents) > 1:
        return json_response(400, {"error": "Request must be a single prompt (contents list of length 1)"})

    auth_error = check_password(req.password, password)
    if auth_error is not None:
        return auth_error

    slot = await resolve_slot(request)
    engine = slot.engine

    prompts = req.contents
    state_manager = get_state_manager()

    # Setup phase: read/create state under lock
    async with session_lock(session_id):
        state = state_manager.get_state(session_id)
        had_existing_state = state is not None

        if state is None:
            state = engine.model.generate_zero_state(0)
            state_manager.put_state(session_id, state)
            logger.info(f"[INIT] Created new state for session: {session_id}")
        else:
            logger.info(f"[REUSE] Reusing existing state for session: {session_id}")

        prompts = normalize_state_prompts(prompts, reuse_existing_state=had_existing_state)

    if req.stream:
        cancel_token = CancellationToken()
        inner_stream = engine.batch_infer_stream_state(
            prompts=prompts,
            state=state,
            max_length=req.max_tokens,
            temperature=req.temperature,
            top_k=req.top_k,
            top_p=req.top_p,
            alpha_presence=req.alpha_presence,
            alpha_frequency=req.alpha_frequency,
            alpha_decay=req.alpha_decay,
            stop_tokens=req.stop_tokens,
            chunk_size=req.chunk_size,
            session_id=session_id,
            state_manager=state_manager,
            cancel_token=cancel_token,
        )

        async def locked_stream():
            """Hold session_lock for the entire stream duration to prevent
            concurrent requests from mutating the same session state."""
            async with session_lock(session_id):
                async for chunk in inner_stream:
                    yield chunk

        return prefill_sse_response(request, locked_stream(), cancel_token, 1, engine=engine)

    # Non-streaming: hold lock for the entire inference
    async with session_lock(session_id):
        try:
            cancel_token = CancellationToken()
            async with reserve_prefill_capacity(request, 1, cancel_token=cancel_token, engine=engine):
                results, finish_reasons = await run_sync_with_disconnect_watch(
                    request,
                    engine.batch_generate_state,
                    cancel_token=cancel_token,
                    prompts=prompts,
                    state=state,
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
            del state
            return client_closed_response()
        except PrefillBszLimitExceeded as exc:
            del state
            return prefill_bsz_limit_response(exc)

        state_manager.put_state(session_id, state)

    choices = []
    for i, text in enumerate(results):
        choices.append(
            {
                "index": i,
                "message": {"role": "assistant", "content": text},
                # Per-item finish_reason ("stop" vs "length"): previously
                # hardcoded to "stop" regardless of whether the generation
                # actually hit a stop string or was truncated at max_tokens.
                "finish_reason": finish_reasons[i],
            }
        )

    response = {
        "id": "rwkv7-batch",
        "object": "chat.completion",
        "model": req.model,
        "choices": choices,
    }
    logger.info(f"[RESPONSE] /state/chat/completions state[2]: {state[2]}")
    del state
    return response


@router.post("/multi_state/chat/completions")
async def multi_state_chat_completions(request: Request):
    app_state = request.app.state
    password = app_state.password
    body = await request.json()
    req, parse_error = parse_request_model(ChatRequest, body)
    if parse_error is not None:
        return parse_error

    auth_error = check_password(req.password, password)
    if auth_error is not None:
        return auth_error

    slot = await resolve_slot(request)
    engine = slot.engine

    if "dialogue_idx" not in body:
        return json_response(400, {"error": "Missing dialogue_idx parameter"})

    session_index = req.session_id
    if not session_index:
        return json_response(400, {"error": "Missing session_id parameter"})

    prompts = req.contents
    if len(prompts) != 1:
        return json_response(400, {"error": "Request must be a single prompt (contents list of length 1)"})

    try:
        dialogue_idx = int(req.dialogue_idx or 0)
    except (ValueError, TypeError):
        return json_response(400, {"error": "dialogue_idx must be an integer"})
    state_key = f"{session_index}:{dialogue_idx}"
    state_manager = get_state_manager()

    # Setup phase: read/create state under lock
    async with session_lock(session_index):
        state = state_manager.get_state(state_key)
        had_existing_state = state is not None

        if state is None:
            if dialogue_idx != 0:
                return json_response(404, {"error": f"State not found for dialogue_idx={dialogue_idx}"})
            state = engine.model.generate_zero_state(0)
            logger.info(f"[INIT] Created new root state for session: {session_index}")
        else:
            logger.info(f"[REUSE] Reusing state for session: {state_key}")

        prompts = normalize_state_prompts(prompts, reuse_existing_state=had_existing_state)

    if req.stream:
        cancel_token = CancellationToken()

        async def stream_with_dialogue_idx():
            """Hold session_lock for the entire stream duration."""
            async with session_lock(session_index):
                inner_stream = engine.batch_infer_stream_state(
                    prompts=prompts,
                    state=state,
                    max_length=req.max_tokens,
                    temperature=req.temperature,
                    top_k=req.top_k,
                    top_p=req.top_p,
                    alpha_presence=req.alpha_presence,
                    alpha_frequency=req.alpha_frequency,
                    alpha_decay=req.alpha_decay,
                    stop_tokens=req.stop_tokens,
                    chunk_size=req.chunk_size,
                    session_id=None,
                    state_manager=None,
                    cancel_token=cancel_token,
                )
                stored = False
                try:
                    async for chunk in inner_stream:
                        if chunk == "data: [DONE]\n\n" and not stored and not cancel_token.is_cancelled():
                            new_dialogue_idx = allocate_next_dialogue_idx(
                                app_state, state_manager, session_index
                            )
                            new_session_id = f"{session_index}:{new_dialogue_idx}"
                            state_manager.put_state(new_session_id, state)
                            stored = True
                            meta = {
                                "object": "multi_state.dialogue_idx",
                                "session_id": new_session_id,
                                "dialogue_idx": new_dialogue_idx,
                            }
                            yield f"data: {json.dumps(meta, ensure_ascii=False)}\n\n"
                        yield chunk
                finally:
                    await inner_stream.aclose()
                    if not stored and not cancel_token.is_cancelled():
                        new_dialogue_idx = allocate_next_dialogue_idx(
                            app_state, state_manager, session_index
                        )
                        new_session_id = f"{session_index}:{new_dialogue_idx}"
                        state_manager.put_state(new_session_id, state)
                        logger.info(f"[RESPONSE] /multi_state/chat/completions state[2]: {state[2]}")

        return prefill_sse_response(request, stream_with_dialogue_idx(), cancel_token, 1, engine=engine)

    # Non-streaming: hold lock for the entire inference
    async with session_lock(session_index):
        try:
            cancel_token = CancellationToken()
            async with reserve_prefill_capacity(request, 1, cancel_token=cancel_token, engine=engine):
                results, finish_reasons = await run_sync_with_disconnect_watch(
                    request,
                    engine.batch_generate_state,
                    cancel_token=cancel_token,
                    prompts=prompts,
                    state=state,
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
            del state
            return client_closed_response()
        except PrefillBszLimitExceeded as exc:
            del state
            return prefill_bsz_limit_response(exc)

        new_dialogue_idx = allocate_next_dialogue_idx(app_state, state_manager, session_index)
        new_session_id = f"{session_index}:{new_dialogue_idx}"
        state_manager.put_state(new_session_id, state)

    choices = []
    for i, text in enumerate(results):
        choices.append(
            {
                "index": i,
                "message": {"role": "assistant", "content": text},
                "finish_reason": finish_reasons[i],
            }
        )

    response = {
        "id": "rwkv7-multi-state",
        "object": "chat.completion",
        "model": req.model,
        "choices": choices,
        "dialogue_idx": new_dialogue_idx,
    }
    logger.info(f"[RESPONSE] /multi_state/chat/completions state[2]: {state[2]}")
    del state
    return response


@router.post("/state/status")
async def state_status(request: Request):
    password = request.app.state.password
    try:
        body = await request.json() if (await request.body()) else {}
        auth_error = check_password(body.get("password"), password)
        if auth_error is not None:
            return auth_error

        manager = get_state_manager()
        all_states = manager.list_all_states()

        detailed_states = []
        for session_id in all_states["l1_cache"]:
            detailed_states.append(
                {
                    "session_id": session_id,
                    "cache_level": "L1 (VRAM)",
                    "last_updated": "In Memory",
                    "timestamp": datetime.now().timestamp(),
                }
            )

        for session_id in all_states["l2_cache"]:
            detailed_states.append(
                {
                    "session_id": session_id,
                    "cache_level": "L2 (RAM)",
                    "last_updated": "In Memory",
                    "timestamp": datetime.now().timestamp(),
                }
            )

        for session_id in all_states["database"]:
            # Must hold db_lock: manager.db_cursor is a single shared SQLite
            # cursor (opened with check_same_thread=False for cross-coroutine
            # use), and every other call site that touches it in
            # state_manager/state_pool.py wraps the access in this same lock.
            # Without it, a concurrent write (e.g. /state/save) executing on
            # the same cursor object can interleave with this read.
            with manager.db_lock:
                manager.db_cursor.execute(
                    "SELECT last_updated FROM sessions WHERE session_id = ?",
                    (session_id,),
                )
                row = manager.db_cursor.fetchone()
            if row:
                timestamp = row[0]
                readable_time = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
                detailed_states.append(
                    {
                        "session_id": session_id,
                        "cache_level": "Database (Disk)",
                        "last_updated": readable_time,
                        "timestamp": timestamp,
                    }
                )

        response_data = {
            "status": "success",
            "total_sessions": all_states["total_count"],
            "l1_cache_count": len(all_states["l1_cache"]),
            "l2_cache_count": len(all_states["l2_cache"]),
            "database_count": len(all_states["database"]),
            "sessions": detailed_states,
        }

        logger.info(f"[StatePool] Status requested. Total sessions: {all_states['total_count']}, "
            f"L1: {len(all_states['l1_cache'])}, L2: {len(all_states['l2_cache'])}, "
            f"DB: {len(all_states['database'])}")

        return response_data
    except Exception as exc:
        logger.error(f"[ERROR] /state/status: {exc}")
        return json_response(500, {"error": str(exc)})


@router.post("/state/delete")
async def state_delete(request: Request):
    password = request.app.state.password
    try:
        body = await request.json()
        session_id = body.get("session_id")
        delete_prefix = body.get("delete_prefix", False)

        if not session_id:
            return json_response(400, {"error": "Missing session_id parameter"})

        auth_error = check_password(body.get("password"), password)
        if auth_error is not None:
            return auth_error

        success = remove_session_from_any_level(session_id)
        if delete_prefix:
            manager = get_state_manager()
            prefix = f"{session_id}:"
            all_states = manager.list_all_states()
            ids = set()
            ids.update(all_states["l1_cache"])
            ids.update(all_states["l2_cache"])
            ids.update(all_states["database"])
            for sid in ids:
                if isinstance(sid, str) and sid.startswith(prefix):
                    remove_session_from_any_level(sid)

        if success or delete_prefix:
            response_data = {
                "status": "success",
                "message": f"Session {session_id} deleted successfully",
            }
            status_code = 200
        else:
            response_data = {
                "status": "not_found",
                "message": f"Session {session_id} not found in database",
            }
            status_code = 404

        logger.info(f"[StatePool] Delete session {session_id}: {'Success' if success else 'Not Found'}")

        return json_response(status_code, response_data)
    except Exception as exc:
        logger.error(f"[ERROR] /state/delete: {exc}")
        return json_response(500, {"error": str(exc)})
