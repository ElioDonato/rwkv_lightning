"""Multi-model registry for the serving process.

A single server can know about several RWKV models and dispatch requests to the
one named by the ``model`` field, loading each into GPU memory lazily on first
use and keeping a set resident concurrently (VRAM permitting). Models are
declared via a JSON config file (see ``models.json``):

    [
      {"id": "small", "path": "models/rwkv7-g1i-2.9b-....pth", "engine": "fp16"},
      {"id": "big",   "path": "models/rwkv7-g1i-7.2b-....pth"}
    ]

``id`` defaults to the file basename when omitted. The first model (or
``--default-model``) is the default used for requests that omit ``model`` --
for the long-standing single-model deployment the default behaves exactly like
the old one engine, so nothing changes when only one model is declared.

Runtime: ``await manager.get(model_id)`` returns an :class:`EngineSlot`
(loading on demand via a worker thread so the event loop never blocks on model
load), ``load``/``unload``/``unload_all`` drive the management API, and an
optional VRAM budget evicts the least-recently-used model before a new one is
loaded. All CUDA work follows the existing process-wide CUDA serialization
(``cuda_guard``/``_offload_gpu``) used by inference.
"""

import asyncio
import logging
from pathlib import Path

logger = logging.getLogger("model.manager")


class ModelCapacityError(Exception):
    """Raised when a model load cannot satisfy the configured concurrency caps
    (max_resident_models / max_resident_bytes) even after evicting the
    least-recently-used non-default models."""


def _slug(path: str) -> str:
    """Default model id = file/leaf basename without extension."""
    return Path(path).stem


def _estimate_vram_bytes(cfg) -> int:
    """Heuristic resident footprint = on-disk weight bytes x load factor, so the
    VRAM budget can refuse/evict before loading. A config may override with
    ``vram_bytes``; the 1.2x factor covers activations/state for decode."""
    if cfg.get("vram_bytes"):
        return int(cfg["vram_bytes"])
    try:
        size = Path(cfg["path"]).stat().st_size
    except OSError:
        return 0
    return int(size * 1.2)


class EngineSlot:
    """One declared model: config + lazily-loaded engine + per-model decode
    aggregators (wired in Phase B). ``engine``/``tokenizer`` become non-None
    once resident."""

    def __init__(self, cfg: dict, default_id: bool):
        self.cfg = cfg
        self.id = cfg["id"]
        self.is_default = default_id
        self.path = cfg["path"]
        self.inference_engine = cfg.get("engine", "fp16")
        if self.inference_engine not in ("fp16", "gemlite", "cutlass"):
            raise ValueError(
                f"model {self.id!r}: unsupported engine {self.inference_engine!r} "
                "(expected fp16/gemlite/cutlass)"
            )
        self.vram_bytes = _estimate_vram_bytes(cfg)

        # Filled in by ModelManager.load().
        self.model = None
        self.tokenizer = None
        self.args = None
        self.rocm_flag = False
        self.engine = None  # infer.InferenceEngine, once loaded

        # Per-model decode machinery (Phase B); None until wired.
        self.dynamic = None
        self.fuse = None
        self.embed = None
        self.wired = False

        self.resident = False
        self.last_used = 0.0

    @property
    def model_name(self) -> str:
        return self.args.MODEL_NAME if self.args is not None else self.path

    def ensure_wired(self):
        """Build + start this model's decode aggregators (embed, fuse, dynamic
        batch). MUST be called on the running event loop (the schedulers spawn
        asyncio tasks via ``.start()``, which binds to the current loop) --
        routes call it after ``manager.get()`` resolves the slot. Idempotent;
        no-op before the engine is resident."""
        if self.wired or self.engine is None:
            return
        from infer.embed_aggregator import EmbedAggregator
        from infer.fuse_aggregator import ChatFuseAggregator
        from infer.dynamic_batch import DynamicBatchDecoder
        from API_servers.router.common import cache_namespace

        # A1: bind the checkpoint fingerprint (cache_namespace(slot), never
        # None-for-default) into the embed aggregator so the embedding output
        # LRU can never serve a stale vector after a runtime checkpoint reload.
        self.embed = EmbedAggregator(
            self.engine.model, self.engine.tokenizer,
            cache_namespace_=cache_namespace(self),
        )
        self.embed.start()
        self.fuse = ChatFuseAggregator(self.engine)
        self.fuse.start()
        self.dynamic = DynamicBatchDecoder(self.engine)
        self.dynamic.start()
        self.wired = True

    def __repr__(self):
        return (f"<EngineSlot {self.id} path={self.path} "
                f"engine={self.inference_engine} resident={self.resident}>")


class ModelManager:
    def __init__(self, models_config, default_id=None, max_resident_bytes=0,
                 embed_id=None, max_resident_models=0):
        """``models_config``: list of model dicts. Concurrency caps (0 = unlimited):
        ``max_resident_bytes``: max resident VRAM footprint for the concurrently
        loaded models (the models' weight+decode footprint, not the CUDA
        allocator cache); ``max_resident_models``: max simultaneously resident
        model count. ``embed_id``: optional model id that the embedding endpoints
        use by default (when a request has no explicit ``model``); falls back to
        the default model when None."""
        self._by_id = {}
        for cfg in models_config:
            cfg = dict(cfg)
            cfg.setdefault("id", _slug(cfg.get("path", "")))
            if not cfg.get("path"):
                raise ValueError(f"model {cfg!r} missing 'path'")
            if cfg["id"] in self._by_id:
                raise ValueError(f"duplicate model id {cfg['id']!r}")
            self._by_id[cfg["id"]] = EngineSlot(cfg, default_id=False)
        if not self._by_id:
            raise ValueError("ModelManager requires at least one model")
        ids = list(self._by_id)
        if default_id:
            if default_id not in self._by_id:
                raise ValueError(f"unknown default model id {default_id!r}")
            self._default_id = default_id
        else:
            self._default_id = ids[0]  # first declared model is the default
        if embed_id is not None and embed_id not in self._by_id:
            raise ValueError(f"unknown embed model id {embed_id!r}")
        self._embed_id = embed_id
        for m in self._by_id.values():
            m.is_default = m.id == self._default_id

        self._max_resident_bytes = int(max_resident_bytes or 0)
        self._max_resident_models = int(max_resident_models or 0)
        self._lock = asyncio.Lock()  # serializes load/unload bookkeeping

    # -- catalog -------------------------------------------------------------

    @property
    def default_id(self):
        return self._default_id

    @property
    def embed_id(self):
        return self._embed_id

    def endpoint_default(self, role=None):
        """Model id used when a request of ``role`` has no explicit ``model``:
        'embed' -> the designated embed model (else default); any other/none ->
        the default model."""
        if role == "embed" and self._embed_id:
            return self._embed_id
        return self._default_id

    def ids(self):
        return list(self._by_id)

    def get_slot(self, model_id):
        if model_id not in self._by_id:
            raise KeyError(
                f"unknown model {model_id!r}; known: {sorted(self._by_id)}"
            )
        return self._by_id[model_id]

    def known_models(self):
        return [
            {
                "id": s.id,
                "path": s.path,
                "engine": s.inference_engine,
                "default": s.is_default,
                "resident": s.resident,
            }
            for s in self._by_id.values()
        ]

    def resident_ids(self):
        return [s.id for s in self._by_id.values() if s.resident]

    @property
    def resident_bytes(self):
        return sum(s.vram_bytes for s in self._by_id.values() if s.resident)

    # -- lifecycle -----------------------------------------------------------

    async def get(self, model_id=None):
        """Resolve the requested model (default when omitted/None/empty) and
        ensure its engine is resident (lazy load). Returns the EngineSlot."""
        if not model_id:
            model_id = self._default_id
        slot = self.get_slot(model_id)
        if not slot.resident:
            await self.load(model_id)
        else:
            slot.last_used = asyncio.get_event_loop().time()
        return slot

    async def load(self, model_id=None):
        """Load a model into VRAM (worker thread so the loop stays responsive),
        evicting LRU models first if the VRAM budget requires it."""
        if not model_id:
            model_id = self._default_id
        slot = self.get_slot(model_id)

        async with self._lock:
            if slot.resident:
                slot.last_used = asyncio.get_event_loop().time()
                return slot
            # The default model always loads (it backs un-issued 'model' requests
            # and the server boots on it); concurrency caps apply to ADDITIONAL
            # models. Without a cap the old unlimited behavior is unchanged.
            if not slot.is_default and (
                self._max_resident_bytes or self._max_resident_models
            ):
                if not await self._make_room(slot):
                    raise ModelCapacityError(
                        f"cannot load model {slot.id!r}: would exceed the "
                        f"configured concurrency caps (max_resident_bytes="
                        f"{self._max_resident_bytes}, max_resident_models="
                        f"{self._max_resident_models}, resident="
                        f"{len(self.resident_ids())})"
                    )
            await asyncio.to_thread(self._load_blocking, slot)
            slot.last_used = asyncio.get_event_loop().time()
            slot.resident = True
            self._update_co_resident_reservations()
            logger.info("[ModelManager] loaded %s engine=%s vram~%dMiB "
                        "resident=%s", slot.id, slot.inference_engine,
                        slot.vram_bytes // (1024*1024), self.resident_ids())
            return slot

    async def unload(self, model_id):
        slot = self.get_slot(model_id)
        async with self._lock:
            await self._unload_slot(slot)
        logger.info("[ModelManager] unloaded %s resident=%s",
                    slot.id, self.resident_ids())

    async def unload_all(self):
        async with self._lock:
            for slot in self._by_id.values():
                await self._unload_slot(slot)

    async def shutdown(self):
        await self.unload_all()

    # -- internals -----------------------------------------------------------

    def _update_co_resident_reservations(self):
        """After any load/unload, pin each resident engine's
        ``model.co_resident_reserved_bytes`` to the WEIGHT footprint of the other
        co-resident models. `refresh_max_prefill_bsz` subtracts this so each
        engine budgets headroom for co-residents and combined admission can't
        over-subscribe the GPU (prevents cross-model OOM)."""
        resident = [s for s in self._by_id.values()
                    if s.resident and s.engine is not None]
        for slot in resident:
            others = sum(s2.vram_bytes for s2 in resident if s2 is not slot)
            try:
                slot.engine.model.co_resident_reserved_bytes = int(others)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("[ModelManager] set co-resident reservation for "
                               "%s failed: %s", slot.id, exc)

    async def _make_room(self, slot):
        """Evict the least-recently-used non-default resident models until the
        new ``slot`` fits under BOTH configured caps (max_resident_bytes and
        max_resident_models). Returns True if it fits (or there are no caps),
        False if it can't (a hard cap would be exceeded even after evicting
        everything non-default)."""
        if not self._max_resident_bytes and not self._max_resident_models:
            return True
        evictable = sorted(
            (s for s in self._by_id.values()
             if s.resident and s.cfg["id"] != self._default_id),
            key=lambda s: s.last_used,
        )

        def fits():
            if self._max_resident_bytes and \
                    self.resident_bytes + slot.vram_bytes > self._max_resident_bytes:
                return False
            if self._max_resident_models and \
                    len(self.resident_ids()) + 1 > self._max_resident_models:
                return False
            return True

        for candidate in evictable:
            if fits():
                break
            await self._unload_slot(candidate)
        if not fits():
            logger.warning(
                "[ModelManager] cannot make room under caps for %s "
                "(bytes %dMiB/%d, models %d/%d, resident=%s); refusing load",
                slot.id,
                (self.resident_bytes + slot.vram_bytes) // (1024*1024)
                if self._max_resident_bytes else 0,
                self._max_resident_bytes // (1024*1024)
                if self._max_resident_bytes else 0,
                len(self.resident_ids()) + 1,
                self._max_resident_models,
                self.resident_ids(),
            )
            return False
        return True

    async def _unload_slot(self, slot):
        if not slot.resident:
            return
        # Stop the per-model decode machinery (dynamic, fuse AND the embed
        # aggregator -- the embed task was previously leaked, keeping the
        # unloaded model/tokenizer alive and its VRAM unreclaimable).
        for agg in (slot.dynamic, slot.fuse, slot.embed):
            if agg is not None:
                try:
                    await agg.stop()
                except Exception as exc:  # pragma: no cover - defensive
                    logger.warning("[ModelManager] stopping aggregator for %s "
                                   "failed: %s", slot.id, exc)
            else:
                break  # aggregators are created together; stop all at once
        slot.embed = None
        # Drop the engine and its tensors; a later torch.cuda.empty_cache at the
        # cleanup seam returns blocks to the allocator pool.
        slot.engine = None
        slot.model = None
        slot.tokenizer = None
        slot.args = None
        slot.rocm_flag = False
        slot.dynamic = None
        slot.fuse = None
        slot.wired = False
        slot.resident = False
        # Remaining co-resident engines must now reserve headroom for one model
        # fewer -- update them before returning.
        self._update_co_resident_reservations()
        # Return the model's freed blocks to the driver so "unload" actually
        # frees VRAM (nvidia-smi drops). This is a blocking sync -- fine for a
        # rare management/eviction op, and serialized so it never races an
        # in-flight decode on the worker thread.
        import torch
        if torch.cuda.is_available():
            from infer.async_forward import cuda_guard
            with cuda_guard():
                torch.cuda.empty_cache()

    def _load_blocking(self, slot):
        """Synchronous load: construct the engine. Runs on a worker thread."""
        from model_load.model_loader import load_model_and_tokenizer
        from infer.inference import InferenceEngine
        from settings import settings

        model, tokenizer, args, rocm = load_model_and_tokenizer(
            slot.path, slot.inference_engine
        )
        # A6: opt-in tokenizer encode memoization (RWKV_ENCODE_CACHE). When off
        # (default) tokenizer stays the original object => byte-identical, no
        # identity swap; when on, encode() is memoized by checkpoint namespace.
        if settings.encode_cache:
            from API_servers.router.common import cache_namespace
            from infer.encode_cache import CachedTokenizer
            tokenizer = CachedTokenizer(
                tokenizer, cache_namespace(slot), settings.encode_cache_capacity
            )
        slot.model, slot.tokenizer, slot.args, slot.rocm_flag = (
            model, tokenizer, args, rocm
        )
        slot.engine = InferenceEngine(
            model=model, tokenizer=tokenizer, args=args, rocm_flag=rocm
        )