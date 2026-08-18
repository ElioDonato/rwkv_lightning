"""Auth-gated runtime model-management API coverage (no GPU: fake manager)."""
import pytest

from fastapi import FastAPI
from fastapi.testclient import TestClient
from types import SimpleNamespace

from API_servers.router.admin_routes import router


class _FakeManager:
    def __init__(self):
        self._ids = ["small", "big"]
        self._resident = set()

    def ids(self):
        return list(self._ids)

    def known_models(self):
        return [
            {"id": m, "path": f"/m/{m}.pth", "engine": "fp16",
             "default": m == "small", "resident": m in self._resident}
            for m in self._ids
        ]

    def get_slot(self, mid):
        return SimpleNamespace(id=mid, vram_bytes=1000, last_used=0.0,
                               resident=(mid in self._resident))

    async def load(self, mid=None):
        self._resident.add(mid or "small")
        return self.get_slot(mid)

    async def unload(self, mid):
        self._resident.discard(mid)

    async def unload_all(self):
        self._resident.clear()

    def resident_ids(self):
        return sorted(self._resident)


@pytest.fixture
def client():
    mgr = _FakeManager()
    app = FastAPI()
    app.state.model_manager = mgr
    app.state.password = "sekret"
    app.include_router(router)
    with TestClient(app) as c:
        c.app_state = mgr
        yield c


_AUTH = {"Authorization": "Bearer sekret"}


def test_list_requires_auth(client):
    assert client.get("/admin/models").status_code == 401


def test_list_with_auth(client):
    r = client.get("/admin/models", headers=_AUTH)
    assert r.status_code == 200
    assert [m["id"] for m in r.json()["models"]] == ["small", "big"]


def test_load_unknown_404(client):
    r = client.post("/admin/models/load", headers=_AUTH, json={"id": "nope"})
    assert r.status_code == 404


def test_load_and_unload(client):
    assert client.post("/admin/models/load", headers=_AUTH,
                       json={"id": "big"}).status_code == 200
    assert "big" in client.app_state._resident
    assert client.post("/admin/models/unload", headers=_AUTH,
                       json={"id": "big"}).status_code == 200
    assert "big" not in client.app_state._resident


def test_load_missing_id_400(client):
    assert client.post("/admin/models/load", headers=_AUTH, json={}).status_code == 400


def test_admin_requires_auth_401(client):
    assert client.post("/admin/models/unload_all", json={}).status_code == 401


def test_unload_all(client):
    client.app_state.load()
    client.post("/admin/models/unload_all", headers=_AUTH, json={})
    assert client.app_state.resident_ids() == []