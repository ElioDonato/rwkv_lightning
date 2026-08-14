import logging

logger = logging.getLogger("api.embedding")

import json
import time

from fastapi import APIRouter, Request

from API_servers.router.common import check_openai_auth, json_response
from infer.embedding import embed_texts, embedding_dim

router = APIRouter()


def _coerce_inputs(body):
    """Accept input as a bare string or a list of strings (OpenAI shape)."""
    raw = body.get("input", "")
    if isinstance(raw, str):
        if not raw.strip():
            raise ValueError("empty input")
        return [raw]
    if isinstance(raw, list):
        if not raw or not all(isinstance(x, str) and x.strip() for x in raw):
            raise ValueError("input must be a non-empty list of non-empty strings")
        return raw
    raise ValueError("input must be a string or a list of strings")


@router.post("/v1/embeddings")
async def openai_embeddings(request: Request):
    return await _embed_hander(request, openai_shape=True)


@router.post("/embedding")
async def project_embedding(request: Request):
    return await _embed_hander(request, openai_shape=False)


async def _embed_hander(request: Request, openai_shape: bool):
    engine = request.app.state.engine
    password = request.app.state.password

    try:
        body = await request.json()
        auth_error = check_openai_auth(request, body, password)
        if auth_error is not None:
            return auth_error
        texts = _coerce_inputs(body)
    except json.JSONDecodeError as exc:
        return json_response(400, {"error": f"Invalid JSON: {str(exc)}"})
    except ValueError as exc:
        return json_response(400, {"error": str(exc)})

    model_name = engine.args.MODEL_NAME.split("/")[-1]

    try:
        t0 = time.perf_counter()
        # When the embed aggregation worker is wired into this app's lifespan
        # (infer/embed_aggregator.EmbedAggregator), route concurrent requests
        # through its queue so their texts are embedded in a shared GPU batch
        # under the opt-in; the default-off path inside submit() runs embed_texts
        # inline exactly as before, so serving behavior is byte-identical unless
        # RWKV_EMBED_AGGREGATE is explicitly enabled. Falling back to the direct
        # call keeps this route functional on an app that never created the
        # aggregator (e.g. an isolated test harness).
        aggregator = getattr(request.app.state, "embed_aggregator", None)
        if aggregator is not None:
            vectors = await aggregator.submit(texts)
        else:
            vectors = embed_texts(engine.model, engine.tokenizer, texts, normalize=True)
        elapsed = time.perf_counter() - t0
    except Exception as exc:
        logger.error("[ERROR] embedding endpoint failed: %s", exc)
        return json_response(500, {"error": str(exc)})

    dim = embedding_dim(engine.model)

    if openai_shape:
        data = [
            {"object": "embedding", "index": i, "embedding": vec}
            for i, vec in enumerate(vectors)
        ]
        return {
            "object": "list",
            "data": data,
            "model": model_name,
            "usage": {
                "prompt_tokens": sum(len(engine.tokenizer.encode(t)) for t in texts),
                "total_tokens": sum(len(engine.tokenizer.encode(t)) for t in texts),
            },
        }

    # Project-client-compatible shape: a TOP-LEVEL LIST. The client
    # (backend/document_storage/embedding.py) does `data = resp.json()` and
    # accepts data[0] being a dict with an "embedding" key, or a plain list of
    # floats. Include "dimensions" as an informational field.
    return [
        {"embedding": vec, "dimensions": dim, "model": model_name}
        for vec in vectors
    ]