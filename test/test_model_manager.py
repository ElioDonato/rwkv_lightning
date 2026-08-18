"""Hermetic (CPU, no GPU) coverage for the multi-model ModelManager.

Exercises catalog registration, default resolution, lazy load/unload lifecycle,
unknown-id behavior, and LRU VRAM eviction -- with ``_load_blocking`` swapped
for a fake that just marks the slot resident (no torch/model load involved), so
it runs in any plain venv without a GPU or checkpoint.
"""
import asyncio

import pytest

from model_load.model_manager import ModelCapacityError, ModelManager


def _fake_load(marker_engine):
    def _load(self, slot):
        class _FakeEngine:
            def __init__(self):
                self.marker = marker_engine

        slot.engine = _FakeEngine()
    return _load


def _configs():
    return [
        {"id": "a", "path": "/nonexistent/models/a.pth"},
        {"id": "b", "path": "/nonexistent/models/b.pth"},
    ]


def _run(coro):
    return asyncio.run(coro)


def test_catalog_default_and_ids():
    m = ModelManager(_configs(), default_id="b")
    assert m.default_id == "b"
    assert m.ids() == ["a", "b"]
    known = m.known_models()
    assert len(known) == 2
    by_id = {k["id"]: k for k in known}
    assert by_id["a"]["resident"] is False
    assert by_id["b"]["default"] is True
    assert by_id["a"]["default"] is False


def test_default_falls_back_to_first():
    m = ModelManager(_configs())
    assert m.default_id == "a"


def test_embed_model_endpoint_default():
    # Embed model designated -> embed endpoints resolve to it.
    m = ModelManager(_configs(), default_id="a", embed_id="b")
    assert m.embed_id == "b"
    assert m.endpoint_default("embed") == "b"
    # Other roles (chat/None) still use the default model.
    assert m.endpoint_default(None) == "a"
    assert m.endpoint_default("chat") == "a"


def test_embed_model_unset_falls_back_to_default():
    m = ModelManager(_configs(), default_id="a")
    assert m.embed_id is None
    assert m.endpoint_default("embed") == "a"  # no embed model -> default


def test_unknown_embed_id_rejected():
    with pytest.raises(ValueError):
        ModelManager(_configs(), embed_id="nope")


def test_duplicate_and_unknown_ids():
    with pytest.raises(ValueError):
        ModelManager([{"id": "a", "path": "x"}, {"id": "a", "path": "y"}])
    m = ModelManager(_configs())
    with pytest.raises(KeyError):
        m.get_slot("nope")


def test_lazy_load_and_unload(monkeypatch):
    monkeypatch.setattr(ModelManager, "_load_blocking", _fake_load("A"))
    m = ModelManager(_configs(), default_id="a")
    assert m.resident_ids() == []

    slot = _run(m.get("b"))
    assert slot.id == "b"
    assert slot.resident is True
    assert slot.engine is not None and slot.engine.marker == "A"
    assert m.resident_ids() == ["b"]
    # get(default) loads the default.
    _run(m.get("a"))
    assert m.resident_ids() == ["a", "b"]
    # get() with no arg resolves the default.
    assert _run(m.get()).id == "a"
    # unknown model on get
    with pytest.raises(KeyError):
        _run(m.get("nope"))


def test_unload_frees_slot(monkeypatch):
    monkeypatch.setattr(ModelManager, "_load_blocking", _fake_load("A"))
    m = ModelManager(_configs())
    _run(m.load("b"))
    assert m.resident_ids() == ["b"]
    _run(m.unload("b"))
    assert m.resident_ids() == []
    assert m.get_slot("b") is m.get_slot("b")  # slot object stable


def test_lru_eviction_under_budget(monkeypatch):
    monkeypatch.setattr(ModelManager, "_load_blocking", _fake_load("A"))
    # Equal 1000-byte vram each; budget fits default a + exactly one of b/c.
    # Loading c must evict the non-default LRU (b), never the default (a).
    cfg = [
        {"id": "a", "path": "/x/a.pth", "vram_bytes": 1000},
        {"id": "b", "path": "/x/b.pth", "vram_bytes": 1000},
        {"id": "c", "path": "/x/c.pth", "vram_bytes": 1000},
    ]
    m = ModelManager(cfg, default_id="a", max_resident_bytes=2500)
    _run(m.load("a"))   # default resident
    _run(m.load("b"))   # a+b = 2000 <= 2500 -> fits
    assert m.resident_ids() == ["a", "b"]
    _run(m.load("c"))   # a+b+c = 3000 > 2500 -> evict non-default LRU = b
    assert m.resident_ids() == ["a", "c"], m.resident_ids()
    assert m.resident_bytes == 2000


def test_shutdown_unloads_all(monkeypatch):
    monkeypatch.setattr(ModelManager, "_load_blocking", _fake_load("A"))
    m = ModelManager(_configs())
    _run(m.load())
    _run(m.load("b"))
    assert m.resident_ids() == ["a", "b"]
    _run(m.shutdown())
    assert m.resident_ids() == []


def test_max_resident_models_evicts_lru(monkeypatch):
    monkeypatch.setattr(ModelManager, "_load_blocking", _fake_load("A"))
    cfg = [
        {"id": "a", "path": "/x/a.pth"},
        {"id": "b", "path": "/x/b.pth"},
        {"id": "c", "path": "/x/c.pth"},
    ]
    m = ModelManager(cfg, default_id="a", max_resident_models=2)
    _run(m.load("a"))
    _run(m.load("b"))   # a+b = 2 of 2
    assert m.resident_ids() == ["a", "b"]
    _run(m.load("c"))   # would be 3 > 2 -> evict non-default LRU (b)
    assert m.resident_ids() == ["a", "c"], m.resident_ids()


def test_max_resident_models_refuses_when_no_room(monkeypatch):
    monkeypatch.setattr(ModelManager, "_load_blocking", _fake_load("A"))
    cfg = [
        {"id": "a", "path": "/x/a.pth"},
        {"id": "b", "path": "/x/b.pth"},
    ]
    # Cap of 1; the default (a) is always loadable/pinned, so b cannot fit.
    m = ModelManager(cfg, default_id="a", max_resident_models=1)
    _run(m.load("a"))
    with pytest.raises(ModelCapacityError):
        _run(m.load("b"))
    assert m.resident_ids() == ["a"]