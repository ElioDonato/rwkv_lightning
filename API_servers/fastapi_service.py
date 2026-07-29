import logging
import os
from contextlib import asynccontextmanager
from threading import Lock

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.routing import APIRoute

from API_servers.router import openai_router, state_router, v1_router, v2_router


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

        yield

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

    app.include_router(v1_router)
    app.include_router(v2_router)
    app.include_router(state_router)
    app.include_router(openai_router)

    return app
