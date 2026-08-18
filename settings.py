"""Central, env-driven settings for every configurable server knob.

This module is the single source of truth for what each tuning value is called
(the environment-variable name), what it defaults to, and how it is parsed.
All feature modules read from the module-level ``settings`` singleton instead of
defining their own module-local defaults, so nothing project- or
machine-specific (ports, host, model path, auth, batch/window/ceiling sizes,
sampler/chat defaults, worker-thread name) lives in the code. This lets the
server run on any machine by setting environment variables rather than editing
source.

Naming
------
Every knob keeps the project's ``RWKV_`` prefix, for continuity with existing
deployments and env files. Values are read once at import time into the
``settings`` singleton. Feature constructors accept explicit overrides (always
winning, used by tests) and otherwise fall back to ``settings``; tests that need
full control can build ``Settings(environ={...})`` directly.

Safety invariants that are deliberately NOT configurable:
* the CUDA worker count (``GpuAsyncExecutor``) is pinned at 1 -- concurrent
  CUDA calls from multiple threads corrupt the driver; this is a correctness
  invariant, not a tuning knob;
* the GPU-cleanup cadence floor (0.5s) -- a guard against a misconfigured
  near-zero interval busy-looping the event loop.
"""

import logging
import os
from typing import Dict, List, Optional

logger = logging.getLogger("settings")


def _warn(name: str, raw: str, default, type_desc: str):
    logger.warning(
        "[settings] invalid %s=%r (expected %s), using default %r",
        name,
        raw,
        type_desc,
        default,
    )


class Settings:
    """Parse the full set of env-driven knobs from ``environ`` (defaults to
    ``os.environ``). Attribute names are the code-level names; each maps to a
    documented environment variable."""

    def __init__(self, environ: Optional[Dict] = None):
        env = os.environ if environ is None else environ
        _get = env.get

        # -- server / CLI -----------------------------------------------------
        # app.py: --model-path, --port, --host, --password, --inference-engine
        self.model_path = self._get_str(
            env, "RWKV_MODEL_PATH", None, "path to a directory of RWKV weights"
        )
        self.api_password = self._get_str(
            env, "RWKV_API_PASSWORD", None, "API password (unset = auth disabled)"
        )
        self.port = self._get_int(env, "RWKV_PORT", 8000, "server bind port")
        self.host = self._get_str(env, "RWKV_HOST", "0.0.0.0", "server bind host")
        self.inference_engine = self._get_str(
            env, "RWKV_INFERENCE_ENGINE", "fp16", "model backend (fp16/GemLite/CUTLASS)"
        )
        # fastapi_service.py CORS allowlist (comma-separated).
        self.cors_origins = [
            o.strip()
            for o in self._get_str(env, "RWKV_CORS_ORIGINS", "*", "comma-sep origins").split(",")
            if o.strip()
        ] or ["*"]

        # -- embedding endpoint (:8083 head-less server) ----------------------
        self.embed_aggregate = self._get_bool(
            env, "RWKV_EMBED_AGGREGATE", False, "opt-in embed aggregation batching"
        )
        self.embed_aggregate_window_ms = self._get_int(
            env, "RWKV_EMBED_AGGREGATE_WINDOW_MS", 10,
            "embed aggregation gather window (ms)",
        )
        # Hard ceiling on per-sub-batch text count when the model has no
        # max_prefill_bsz cap (embedding.embed_texts + embed_aggregator).
        self.embed_hard_ceiling = self._get_int(
            env, "RWKV_EMBED_HARD_CEILING", 64,
            "embed sub-batch hard ceiling (texts)",
        )
        # Fallback chunk size (token steps per batched chunk) when the model
        # exposes no prefill_chunk_size attribute.
        self.prefill_chunk_size = self._get_int(
            env, "RWKV_PREFILL_CHUNK_SIZE", 256,
            "prefill chunk fallback (token steps/batch)",
        )

        # -- Mod A async GPU-worker offload ------------------------------------
        self.async_forward = self._get_bool(
            env, "RWKV_ASYNC_FORWARD", False, "opt-in GPU-worker offload"
        )
        self.async_forward_thread_name = self._get_str(
            env, "RWKV_ASYNC_FORWARD_THREAD_NAME", "gpu-forward",
            "GPU-worker thread name prefix",
        )
        self.gpu_cleanup_interval_s = self._get_float(
            env, "RWKV_GPU_CLEANUP_INTERVAL_S", 60.0,
            "periodic GPU cache-cleanup cadence (s)",
        )

        # -- fuse chat batch (combine-queue for /openai chat) ------------------
        self.fuse_chat_batch = self._get_bool(
            env, "RWKV_FUSE_CHAT_BATCH", False, "opt-in decode combine-queue"
        )
        self.fuse_chat_window_ms = self._get_int(
            env, "RWKV_FUSE_CHAT_WINDOW_MS", 5,
            "fuse gather window (ms)",
        )
        self.fuse_chat_max_bsz = self._get_int(
            env, "RWKV_FUSE_CHAT_MAX_BSZ", 8,
            "fused-row hard ceiling per big_batch_stream call",
        )
        self.fuse_chat_pending_cap = self._get_int(
            env, "RWKV_FUSE_CHAT_PENDING_CAP", 64,
            "fuse pending-queue overload bound",
        )
        # Sampler-control values that define "fusable" (homogeneous) requests:
        # the fused big_batch path ignores these (gumbel temp-only), so a job
        # carrying a non-default on any of them never fuses. Matching the chat
        # route's applied defaults -- single source, no silent divergence.
        self.fuse_sampler_top_k = self._get_int(
            env, "RWKV_FUSE_SAMPLER_TOP_K", 20, "fusable top_k gate"
        )
        self.fuse_sampler_top_p = self._get_float(
            env, "RWKV_FUSE_SAMPLER_TOP_P", 0.6, "fusable top_p gate"
        )
        self.fuse_sampler_alpha_presence = self._get_float(
            env, "RWKV_FUSE_SAMPLER_ALPHA_PRESENCE", 1.0, "fusable alpha_presence gate"
        )
        self.fuse_sampler_alpha_frequency = self._get_float(
            env, "RWKV_FUSE_SAMPLER_ALPHA_FREQUENCY", 0.1, "fusable alpha_frequency gate"
        )
        self.fuse_sampler_alpha_decay = self._get_float(
            env, "RWKV_FUSE_SAMPLER_ALPHA_DECAY", 0.996, "fusable alpha_decay gate"
        )
        # Default max_tokens applied by the chat route when the body omits it
        # (relevant to the fuse homogeneity gate, which requires equality).
        self.chat_max_tokens_default = self._get_int(
            env, "RWKV_CHAT_MAX_TOKENS_DEFAULT", 4096,
            "chat max_tokens default when omitted",
        )
        # Whether the fuse route opts default (body-omitted) chat traffic into
        # fusion; an explicit use_prefix_cache=True always stays solo-faithful.
        self.fuse_prefix_cache_default = self._get_bool(
            env, "RWKV_FUSE_PREFIX_CACHE_DEFAULT", False,
            "fuse: treat omitted use_prefix_cache as False",
        )

        # -- dynamic decode batching (per-row-sampled shared decode) -----------
        # Phase-4 sub-step 2: merge ANY concurrent chat requests into ONE shared
        # multi-row decode with per-row sampling (RWKV_DYNAMIC_BATCH, default
        # off). Rows join/leave as requests arrive/finish; each row keeps its own
        # sampler controls + RNG. A burst is capped so it can never OOM a
        # co-resident process. Default-off: no decoder activity, route unchanged.
        self.dynamic_batch = self._get_bool(
            env, "RWKV_DYNAMIC_BATCH", False, "opt-in dynamic multi-row decode batching"
        )
        self.dynamic_batch_window_ms = self._get_int(
            env, "RWKV_DYNAMIC_BATCH_WINDOW_MS", 8,
            "dynamic-batch gather window (ms)",
        )
        self.dynamic_batch_max_bsz = self._get_int(
            env, "RWKV_DYNAMIC_BATCH_MAX_BSZ", 8,
            "dynamic-batch fused-row hard ceiling per decode batch",
        )

        # -- CUDA-graph decode forward (opt-in, Phase 5) -----------------------
        # Replay model.forward_batch as a per-size CUDA graph to cut host launch
        # overhead on the decode step. Default-off: the graph module is inert and
        # the dynamic decoder calls the raw forward_batch, byte-identical to the
        # pre-Phase-5 path.
        self.cuda_graph_decode = self._get_bool(
            env, "RWKV_CUDA_GRAPH", False,
            "opt-in CUDA-graph replay of the decode forward",
        )

        # -- extensive caching (all opt-in, default-OFF) ----------------------
        # Every new cache below is keyed by a checkpoint fingerprint via
        # ``cache_namespace(slot)`` (see API_servers/router/common.py) so a
        # runtime model reload / checkpoint swap can never serve a stale entry.
        # Single-model behavior is byte-identical unless a knob is enabled.

        # A3c: background TTL / max-row sweep + VACUUM of the state DB, so
        # rwkv_sessions.db / prefix_cache stop growing unbounded. Master gate
        # MUST be on before the TTL/interval knobs have any effect (a nonzero
        # TTL alone would otherwise auto-start a DB-mutating sweeper with no
        # explicit opt-in). Sweeps run on the background io_executor thread and
        # never on the asyncio event loop.
        self.cache_sweep = self._get_bool(
            env, "RWKV_CACHE_SWEEP", False,
            "opt-in background DB sweep (TTL/cap/VACUUM)",
        )
        self.prefix_cache_ttl_s = self._get_float(
            env, "RWKV_PREFIX_CACHE_TTL_S", 86400.0,
            "prefix_cache/sessions TTL for the sweep (s; 0 = never expire)",
        )
        self.prefix_cache_max_rows = self._get_int(
            env, "RWKV_PREFIX_CACHE_MAX_ROWS", 0,
            "max prefix_cache rows before sweep evicts oldest (0 = unlimited)",
        )
        self.cache_sweep_interval_s = self._get_float(
            env, "RWKV_CACHE_SWEEP_INTERVAL_S", 3600.0,
            "DB sweep cadence (s)",
        )

        # A3b: off-gated replacement for the hot-path synchronous DB prefix
        # lookups. ON -> a single bounded disk probe (largest plausible bucket
        # <= prompt length) plus a background warm of the other buckets, instead
        # of up to 8 synchronous SELECTs on the request thread. OFF (default)
        # keeps the historical per-request disk fallback byte-identical.
        self.prefix_disk_async = self._get_bool(
            env, "RWKV_PREFIX_DISK_ASYNC", False,
            "opt-in single-probe + background-warm prefix disk fallback",
        )

        # A1: embedding output LRU (deterministic pure function of (model,text)).
        self.embed_cache = self._get_bool(
            env, "RWKV_EMBED_CACHE", False,
            "opt-in embedding output LRU",
        )
        self.embed_cache_capacity = self._get_int(
            env, "RWKV_EMBED_CACHE_CAPACITY", 4096,
            "embedding output LRU entry cap",
        )

        # A6: tokenize-encode memoization (pure function of (model,text)).
        self.encode_cache = self._get_bool(
            env, "RWKV_ENCODE_CACHE", False,
            "opt-in tokenizer encode memoization",
        )
        self.encode_cache_capacity = self._get_int(
            env, "RWKV_ENCODE_CACHE_CAPACITY", 8192,
            "tokenizer encode memo entry cap",
        )

        # B: adaptive short-prompt prefix reuse (relax the fixed bucket gate).
        self.prefix_adaptive = self._get_bool(
            env, "RWKV_PREFIX_ADAPTIVE", False,
            "opt-in adaptive short-prompt prefix checkpointing",
        )

        # C: turn-level state reuse on raw /v1 + /v2 (content turn-hash).
        self.turn_state_reuse = self._get_bool(
            env, "RWKV_TURN_STATE_REUSE", False,
            "opt-in content-addressed multi-turn state reuse (raw /v1+/v2 only)",
        )

    # -- typed environment accessors ------------------------------------------
    @staticmethod
    def _get_str(env, name: str, default: Optional[str], what: str) -> Optional[str]:
        raw = env.get(name)
        if raw is None:
            return default
        raw = raw.strip()
        if raw == "":
            return default
        return raw

    @staticmethod
    def _get_bool(env, name: str, default: bool, what: str) -> bool:
        raw = env.get(name)
        if raw is None:
            return default
        try:
            return int(raw) != 0
        except (TypeError, ValueError):
            _warn(name, raw, default, "0/1 or nonzero for True")
            return default

    @staticmethod
    def _get_int(env, name: str, default: int, what: str) -> int:
        raw = env.get(name)
        if raw is None:
            return default
        try:
            return int(raw)
        except (TypeError, ValueError):
            _warn(name, raw, default, "integer")
            return default

    @staticmethod
    def _get_float(env, name: str, default: float, what: str) -> float:
        raw = env.get(name)
        if raw is None:
            return default
        try:
            return float(raw)
        except (TypeError, ValueError):
            _warn(name, raw, default, "number")
            return default


settings = Settings()