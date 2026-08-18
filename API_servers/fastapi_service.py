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
    embedding_router,
    openai_router,
    responses_router,
    state_router,
    v1_router,
    v2_router,
)
from infer import inference_deps
from infer.embed_aggregator import EmbedAggregator
from infer.fuse_aggregator import ChatFuseAggregator
from settings import settings


def create_app(engine, password=None):
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger = logging.getLogger("uvicorn.error")
        logger.info("Registered FastAPI routes:")
        for route in app.routes:
            if not isinstance(route, APIRoute):
                continue
            methods = ",".join(sorted(route.methods - {"HEAD", "OPTIONS"}))
            logger.info("  %-20s %s", methods, route.path)

        # Opt-in embed aggregation worker (RWKV_EMBED_AGGREGATE, default-off).
        # Attached to app.state so the embedding routes route concurrent
        # /embedding requests through it (shared GPU batches under the opt-in;
        # byte-identical inline embed_texts when off). Started here, and
        # stopped (resolving any still-queued requests) when the app shuts down.
        aggregator = EmbedAggregator(engine.model, engine.tokenizer)
        app.state.embed_aggregator = aggregator
        aggregator.start()

        # Opt-in decode combine-queue for /openai/v1/chat/completions
        # (RWKV_FUSE_CHAT_BATCH, default-off -- item 3). Attached to app.state so
        # openai_routes can route concurrent chat requests through it (fused
        # multi-row big_batch_stream decodes under the opt-in; the route uses the
        # original single_infer path when disabled). start() is a no-op while
        # disabled; stopped (any queued rows ended) on shutdown.
        fuse = ChatFuseAggregator(engine)
        app.state.chat_fuse_aggregator = fuse
        fuse.start()

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
        try:
            yield
        finally:
            if cleanup_task is not None:
                cleanup_task.cancel()
                with suppress(asyncio.CancelledError, Exception):
                    await cleanup_task
            await fuse.stop()
            await aggregator.stop()

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

    return app
