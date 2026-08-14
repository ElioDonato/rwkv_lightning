import json
import logging
import os
from contextlib import asynccontextmanager
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
from infer.embed_aggregator import EmbedAggregator


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
        try:
            yield
        finally:
            await aggregator.stop()

    app = FastAPI(lifespan=lifespan)
    # RWKV_CORS_ORIGINS: comma-separated allowlist, e.g. "http://localhost:3000,https://app.example.com"
    # Default "*" preserves prior behavior for trusted-LAN deployments.
    cors_origins_env = os.environ.get("RWKV_CORS_ORIGINS", "*").strip()
    cors_origins = [o.strip() for o in cors_origins_env.split(",") if o.strip()] or ["*"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
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
