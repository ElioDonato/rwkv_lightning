"""Hermetic (CPU, no GPU) coverage for the per-request model dispatch helper
``API_servers.router.common.resolve_slot``.

Builds a FAKE ModelManager holding two slots (``small`` / ``big``) with fake
engines, and asserts that resolve_slot picks the right slot for a request's
``model`` field -- ``small`` for ``model=small``, ``big`` for ``model=big``,
and the default slot for omitted / empty / unknown model ids (including the
legacy ``"rwkv7"`` string). Also covers the shim path when no ModelManager is
installed. Runs entirely in-process; no torch, no GPU.
"""
import asyncio
from types import SimpleNamespace

import pytest

from API_servers.router.common import resolve_slot


class _FakeSlot:
    def __init__(self, name):
        self.id = name
        self.marker = name
        self.engine = SimpleNamespace(marker=name)
        self.embed = self.fuse = self.dynamic = None
        self.wired = False

    def ensure_wired(self):
        self.wired = True


class _FakeManager:
    def __init__(self, default_id="small"):
        self.default_id = default_id
        self._slots = {"small": _FakeSlot("small"), "big": _FakeSlot("big")}

    def ids(self):
        return list(self._slots)

    async def get(self, model_id=None):
        if not model_id:
            model_id = self.default_id
        return self._slots[model_id]


def _make_request(manager):
    app = SimpleNamespace(state=SimpleNamespace(model_manager=manager))
    return SimpleNamespace(app=app)


def _run(coro):
    return asyncio.run(coro)


def test_routes_explicit_model_field_to_slot():
    m = _FakeManager(default_id="small")
    r = _make_request(m)
    assert _run(resolve_slot(r, "small")) is m._slots["small"]
    assert _run(resolve_slot(r, "big")) is m._slots["big"]
    assert _run(resolve_slot(r, "big")).engine.marker == "big"


def test_omitted_empty_and_unknown_model_map_to_default():
    m = _FakeManager(default_id="small")
    r = _make_request(m)
    assert _run(resolve_slot(r)).id == "small"
    assert _run(resolve_slot(r, None)).id == "small"
    assert _run(resolve_slot(r, "")).id == "small"
    # Unknown model ids (incl. the default-schema "rwkv7") must NEVER raise.
    assert _run(resolve_slot(r, "unknown")).id == "small"
    assert _run(resolve_slot(r, "rwkv7")).id == "small"


def test_non_default_default_model():
    m = _FakeManager(default_id="big")
    r = _make_request(m)
    assert _run(resolve_slot(r)).id == "big"
    assert _run(resolve_slot(r, "unknown")).id == "big"


def test_resolve_slot_wires_the_slot():
    m = _FakeManager(default_id="small")
    r = _make_request(m)
    s = _run(resolve_slot(r, "big"))
    assert s.wired is True


def test_resolve_slot_shim_without_manager():
    # No model_manager: falls back to historical app.state engine + aggregators.
    app = SimpleNamespace(
        state=SimpleNamespace(
            engine="ENGINE",
            embed_aggregator="EMBED",
            chat_fuse_aggregator="FUSE",
            chat_dynamic_decoder="DYN",
        )
    )
    request = SimpleNamespace(app=app)
    slot = _run(resolve_slot(request))
    assert slot.id == "default"
    assert slot.engine == "ENGINE"
    assert slot.embed == "EMBED"
    assert slot.fuse == "FUSE"
    assert slot.dynamic == "DYN"
    # ensure_wired is a no-op on the shim.
    slot.ensure_wired()