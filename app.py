import logging

logger = logging.getLogger("app")

import argparse
import asyncio
import atexit
import json
import os
import signal
import sys

import uvicorn

from API_servers.fastapi_service import create_app
from model_load.model_loader import INFERENCE_ENGINES
from model_load.model_manager import ModelManager
from settings import settings
from state_manager.state_pool import shutdown_state_manager


def _load_models_config(path):
    """Return (configs_list, meta) where meta may carry optional ``default_model``
    / ``embed_model`` ids from the JSON's top-level object."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    meta = {}
    if isinstance(data, dict):
        meta = {
            k: data.get(k)
            for k in ("default_model", "embed_model")
        }
        data = data.get("models", [])
    if not isinstance(data, list) or not data:
        raise SystemExit(f"--models-config {path}: must be a non-empty list of "
                         "model objects {id?, path, engine?}")
    return data, meta


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        default=settings.model_path,
        help="RWKV model path (default: RWKV_MODEL_PATH env). Ignored when "
             "--models-config is set.",
    )
    parser.add_argument(
        "--models-config",
        type=str,
        default=os.getenv("RWKV_MODELS_CONFIG"),
        help="path to a JSON catalog of models (see models.json); enables "
             "multi-model serving with per-request model dispatch",
    )
    parser.add_argument(
        "--default-model",
        type=str,
        default=os.getenv("RWKV_DEFAULT_MODEL"),
        help="model id to use for requests that omit 'model' "
             "(default: the first declared model)",
    )
    parser.add_argument(
        "--embed-model",
        type=str,
        default=os.getenv("RWKV_EMBED_MODEL"),
        help="model id the embedding endpoints use by default "
             "(default: the default model)",
    )
    parser.add_argument(
        "--inference-engine",
        "--backend",
        dest="inference_engine",
        choices=INFERENCE_ENGINES,
        default=settings.inference_engine,
        help="model backend: fp16, GemLite packed quantization, or CUTLASS W8A16",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=settings.port,
        help="port to serve on (default: RWKV_PORT env, or 8000)",
    )
    parser.add_argument(
        "--password",
        type=str,
        default=settings.api_password,
        help="API password for authentication (default: RWKV_API_PASSWORD env)",
    )
    args = parser.parse_args()
    if args.models_config:
        configs, meta = _load_models_config(args.models_config)
    else:
        if not args.model_path:
            parser.error(
                "--model-path is required (or pass --models-config / set the "
                "RWKV_MODEL_PATH env var)"
            )
        configs = [{"path": args.model_path, "engine": args.inference_engine}]
        meta = {}
    return args, configs, meta


def main():
    args_cli, configs, meta = parse_args()
    default_id = args_cli.default_model or meta.get("default_model")
    embed_id = args_cli.embed_model or meta.get("embed_model")
    manager = ModelManager(
        configs,
        default_id=default_id,
        embed_id=embed_id,
        max_resident_bytes=int(os.getenv("RWKV_MAX_RESIDENT_BYTES", "0") or 0),
    )
    # Load the default model at startup (same behavior as the old single-model
    # boot). Additional models from the catalog are loaded lazily on request.
    asyncio.run(manager.load())

    app = create_app(manager, password=args_cli.password)

    def cleanup_handler(signum, frame):
        logger.info("\nShutting down server...")
        sys.exit(0)

    def cleanup_at_exit():
        logger.info("Persisting all states to database...")
        shutdown_state_manager()
        asyncio.run(manager.shutdown())
        logger.info("All states persisted to database.")

    signal.signal(signal.SIGINT, cleanup_handler)
    signal.signal(signal.SIGTERM, cleanup_handler)
    atexit.register(cleanup_at_exit)

    uvicorn.run(app, host=settings.host, port=args_cli.port)


if __name__ == "__main__":
    main()
