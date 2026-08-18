"""Runtime model-management API (multi-model feature).

Lets an authenticated client inspect and control which models are resident in
the serving process, complementing the per-request dispatch:

    GET  /admin/models               list declared models + residency
    POST /admin/models/load          {"id": "<model>"}  -> load into VRAM
    POST /admin/models/unload        {"id": "<model>"}  -> unload, free VRAM
    POST /admin/models/unload_all                         -> unload everything

All endpoints require the API password (same auth as the serving routes).
"""
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from API_servers.router.common import check_openai_auth
from model_load.model_manager import ModelCapacityError

router = APIRouter(prefix="/admin/models", tags=["admin"])


def _manager_or_404(request: Request):
    manager = getattr(request.app.state, "model_manager", None)
    if manager is None:
        return None, JSONResponse(
            status_code=501, content={"error": "model_manager not configured"}
        )
    return manager, None


def _model_id(body: dict, request: Request):
    mid = (body or {}).get("id") or (body or {}).get("model")
    if mid:
        return str(mid)
    return None


async def _auth_ok(request: Request, body: dict):
    password = getattr(request.app.state, "password", None)
    return check_openai_auth(request, body, password)


@router.get("")
async def list_models(request: Request):
    err = await _auth_ok(request, {})
    if err is not None:
        return err
    manager, err = _manager_or_404(request)
    if err is not None:
        return err
    out = []
    for m in manager.known_models():
        slot = manager.get_slot(m["id"])
        m["resident_bytes"] = slot.vram_bytes
        m["last_used"] = round(slot.last_used, 3) if slot.last_used else None
        out.append(m)
    return {"models": out}


@router.post("/load")
async def load_model(request: Request):
    body = await request.json()
    err = await _auth_ok(request, body)
    if err is not None:
        return err
    manager, err = _manager_or_404(request)
    if err is not None:
        return err
    mid = _model_id(body, request)
    if not mid:
        return JSONResponse(status_code=400,
                            content={"error": "missing 'id'"})
    if mid not in manager.ids():
        return JSONResponse(
            status_code=404,
            content={"error": f"unknown model {mid!r}; known: {manager.ids()}"},
        )
    try:
        slot = await manager.load(mid)
    except ModelCapacityError as exc:
        # Hard concurrency/VRAM cap reached -> informative 409 (not a 500).
        return JSONResponse(status_code=409, content={"error": str(exc)})
    except Exception as exc:  # e.g. VRAM allocation failure
        return JSONResponse(status_code=500, content={"error": str(exc)})
    return {
        "id": slot.id,
        "resident": slot.resident,
        "resident_bytes": slot.vram_bytes,
    }


@router.post("/unload")
async def unload_model(request: Request):
    body = await request.json()
    err = await _auth_ok(request, body)
    if err is not None:
        return err
    manager, err = _manager_or_404(request)
    if err is not None:
        return err
    mid = _model_id(body, request)
    if not mid:
        return JSONResponse(status_code=400,
                            content={"error": "missing 'id'"})
    if mid not in manager.ids():
        return JSONResponse(
            status_code=404,
            content={"error": f"unknown model {mid!r}; known: {manager.ids()}"},
        )
    await manager.unload(mid)
    return {"id": mid, "resident": False}


@router.post("/unload_all")
async def unload_all_models(request: Request):
    err = await _auth_ok(request, {})
    if err is not None:
        return err
    manager, err = _manager_or_404(request)
    if err is not None:
        return err
    await manager.unload_all()
    return {"resident": manager.resident_ids()}