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

        self.resident = False
        self.last_used = 0.0

    @property
    def model_name(self) -> str:
        return self.args.MODEL_NAME if self.args is not None else self.path

    def __repr__(self):
        return (f"<EngineSlot {self.id} path={self.path} "
                f"engine={self.inference_engine} resident={self.resident}>")


class ModelManager:
    def __init__(self, models_config, default_id=None, max_resident_bytes=0):
        """``models_config``: list of model dicts. ``max_resident_bytes``: hard
        VRAM budget (0 = no enforced budget)."""
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
        for m in self._by_id.values():
            m.is_default = m.id == self._default_id

        self._max_resident_bytes = int(max_resident_bytes or 0)
        self._lock = asyncio.Lock()  # serializes load/unload bookkeeping

    # -- catalog -------------------------------------------------------------

    @property
    def default_id(self):
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
            # Make room under the budget before loading the new model.
            if self._max_resident_bytes:
                needed = slot.vram_bytes
                await self._evict_for(needed)
            await asyncio.to_thread(self._load_blocking, slot)
            slot.last_used = asyncio.get_event_loop().time()
            slot.resident = True
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

    async def _evict_for(self, needed_bytes):
        """Unload the LRU resident models (not the one being loaded) until the
        current resident bytes + needed fit the budget. No-op if insufficient
        headroom is impossible (budget smaller than the model itself)."""
        if self.resident_bytes + needed_bytes <= self._max_resident_bytes:
            return
        to_evict = sorted(
            (s for s in self._by_id.values()
             if s.resident and s.cfg["id"] != self._default_id),
            key=lambda s: s.last_used,
        )
        for slot in to_evict:
            if self.resident_bytes + needed_bytes <= self._max_resident_bytes:
                break
            await self._unload_slot(slot)
        if self.resident_bytes + needed_bytes > self._max_resident_bytes:
            logger.warning(
                "[ModelManager] cannot fit %dMiB within budget %dMiB "
                "(only %dMiB resident to evict); model may fail to allocate",
                needed_bytes // (1024*1024),
                self._max_resident_bytes // (1024*1024),
                (self.resident_bytes - needed_bytes) // (1024*1024),
            )

    async def _unload_slot(self, slot):
        if not slot.resident:
            return
        # Stop the per-model decode machinery if it was wired (Phase B).
        for agg in (slot.dynamic, slot.fuse):
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
        slot.resident = False

    def _load_blocking(self, slot):
        """Synchronous load: construct the engine. Runs on a worker thread."""
        from model_load.model_loader import load_model_and_tokenizer
        from infer.inference import InferenceEngine

        model, tokenizer, args, rocm = load_model_and_tokenizer(
            slot.path, slot.inference_engine
        )
        slot.model, slot.tokenizer, slot.args, slot.rocm_flag = (
            model, tokenizer, args, rocm
        )
        slot.engine = InferenceEngine(
            model=model, tokenizer=tokenizer, args=args, rocm_flag=rocm
        )