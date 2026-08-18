import asyncio
import json
import logging
from contextlib import asynccontextmanager
from contextlib import suppress
from threading import Lock

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute

from API_servers.router import (
    admin_router,
    embedding_router,
    openai_router,
    responses_router,
    state_router,
    v1_router,
    v2_router,
)
from infer import inference_deps
from settings import settings


def create_app(model_manager, password=None):
    # Default engine + its per-model aggregators are wired exactly as the old
    # single-model create_app did, so with one model nothing changes. Phase B
    # extends per-model wiring to every loaded engine via app.state.model_manager.
    default = model_manager.get_slot(model_manager.default_id)
    engine = default.engine

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger = logging.getLogger("uvicorn.error")
        logger.info("Registered FastAPI routes:")
        for route in app.routes:
            if not isinstance(route, APIRoute):
                continue
            methods = ",".join(sorted(route.methods - {"HEAD", "OPTIONS"}))
            logger.info("  %-20s %s", methods, route.path)

        # Per-model decode machinery for the default engine, started on the running
        # loop (builds + starts the embed/fuse/dynamic-batch aggregators on this
        # slot; idempotent). app.state.* keep the historical names so routes that
        # still read them fall back to the default; the per-request dispatch
        # resolves the per-model slot via app.state.model_manager.
        default.ensure_wired()
        app.state.embed_aggregator = default.embed
        app.state.chat_fuse_aggregator = default.fuse
        app.state.chat_dynamic_decoder = default.dynamic

        # Single low-frequency background GPU cache cleanup (item 1): replaces
        # the per-request torch.cuda.empty_cache() calls removed from the :8081
        # decode hot path. Runs InferenceEngine.run_periodic_gpu_cleanup at a
        # fixed cadence (RWKV_GPU_CLEANUP_INTERVAL_S, default 60s), routed
        # through the offload seam. Only spawned when CUDA is present (on a
        # CPU box _cleanup_cuda_memory is an early no-op, so a perpetual
        # sleeping task would be pure noise).
        cleanup_task = None
        if inference_deps.get_torch().cuda.is_available():
            cleanup_task = asyncio.create_task(engine.run_periodic_gpu_cleanup())

        # A3c: background state-DB sweep (RWKV_CACHE_SWEEP=1). Bounds the size
        # of rwkv_sessions.db / prefix_cache (TTL + max-row cap + VACUUM) on
        # the background io_executor thread, never the event loop. Off by
        # default; the sweeper is drained in the finally below before the
        # connection is closed by manager shutdown.
        if settings.cache_sweep:
            from state_manager.state_pool import get_state_manager
            _sweeper = get_state_manager().start_sweeper(
                settings.cache_sweep_interval_s,
                settings.prefix_cache_ttl_s,
                settings.prefix_cache_max_rows,
            )
        else:
            _sweeper = None

        try:
            yield
        finally:
            if _sweeper is not None:
                try:
                    _sweeper.stop_sweeper()
                except Exception:  # pragma: no cover - best-effort teardown
                    pass
            if cleanup_task is not None:
                cleanup_task.cancel()
                with suppress(asyncio.CancelledError, Exception):
                    await cleanup_task
            # Per-model decode schedulers wired on the default slot (embed, fuse,
            # dynamic batch) -- gracefully drain them on shutdown.
            for _agg in (default.dynamic, default.fuse, default.embed):
                if _agg is not None:
                    try:
                        await _agg.stop()
                    except Exception:  # pragma: no cover - best-effort teardown
                        pass

    app = FastAPI(lifespan=lifespan)
    # RWKV_CORS_ORIGINS: comma-separated allowlist, e.g. "http://localhost:3000,https://app.example.com"
    # Default "*" preserves prior behavior for trusted-LAN deployments. Read from the
    # central settings module (single source of truth for the knob).
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.state.engine = engine
    app.state.model_manager = model_manager
    app.state.password = password
    app.state.dialogue_idx_lock = Lock()
    app.state.dialogue_idx_counters = {}

    @app.exception_handler(json.JSONDecodeError)
    async def _json_decode_handler(request: Request, exc: json.JSONDecodeError):
        return JSONResponse(
            status_code=400,
            content={"error": "Invalid JSON in request body"},
        )

    app.include_router(v1_router)
    app.include_router(v2_router)
    app.include_router(state_router)
    app.include_router(openai_router)
    app.include_router(responses_router)
    app.include_router(embedding_router)
    app.include_router(admin_router)

    return app
