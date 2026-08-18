"""Hermetic tests for multi-model isolation of the state + prefix caches.

The StateCacheManager is a process-wide singleton; with multiple models we must
scope every session/prefix key by a model namespace so model A's recurrent state
is NEVER served to model B (and the default/single-model path stays identity).
These run on CPU with tiny tensors; they touch the same SQLite DB as production
but only read/delete the namespaced keys they create.
"""
import torch
import pytest

from state_manager import state_pool
from state_manager.state_pool import StateCacheManager, get_state_manager


@pytest.fixture(autouse=True)
def _fresh_state_pool(tmp_path):
    """Make these tests hermetic: point the state DB at a fresh temp file and
    reset the process-wide StateCacheManager singleton, so they don't collide
    with (or break on) whatever other state-cache tests did to the shared
    DB_PATH / singleton. Restored afterwards."""
    _orig_path = state_pool.DB_PATH
    _orig_instance = StateCacheManager._instance
    state_pool.DB_PATH = str(tmp_path / "iso.db")
    StateCacheManager._instance = None
    yield
    StateCacheManager._instance = _orig_instance
    state_pool.DB_PATH = _orig_path


def _state(n):
    return [torch.tensor([n], dtype=torch.float32),
            torch.tensor([n], dtype=torch.float32),
            torch.tensor([n], dtype=torch.int32)]


def _cleanup(*keys_with_model):
    for model, key in keys_with_model:
        try:
            get_state_manager(model).delete_state_from_any_level(key)
        except Exception:
            pass


def test_session_state_isolated_between_models():
    m = get_state_manager()
    # put under model X
    m.put_state("s1", _state(1), model="X")
    # default (identity) and model Y must NOT see it
    assert get_state_manager().get_state("s1") is None
    assert get_state_manager(model="Y").get_state("s1") is None
    # model X sees it
    got = get_state_manager(model="X").get_state("s1")
    assert got is not None and got[2].item() == 1
    _cleanup(("X", "s1"))


def test_default_and_named_share_nothing(tmp_path):
    # default (no model) uses the identity key
    get_state_manager().put_state("sd", _state(7))
    assert get_state_manager().get_state("sd") is not None
    # an explicitly-named model uses "model:sd" -> isolated
    assert get_state_manager(model="M").get_state("sd") is None
    _cleanup((None, "sd"), ("M", "sd"))


def test_prefix_cache_isolated_between_models():
    tokens = list(range(1024))  # PREFIX_CACHE_BUCKETS includes 1024
    sX = _state(11)
    sY = _state(22)
    get_state_manager().put_prefix_state(tokens, sX, model="X")
    get_state_manager().put_prefix_state(tokens, sY, model="Y")
    prompt = list(range(1024 + 40))

    mX = get_state_manager(model="X").match_prefix_state(prompt, device="cpu")
    mY = get_state_manager(model="Y").match_prefix_state(prompt, device="cpu")
    assert mX is not None and mX["state"][2].item() == 11
    assert mY is not None and mY["state"][2].item() == 22  # Y's OWN state, not X's

    # default (identity) must not match the model-scoped entries
    assert get_state_manager().match_prefix_state(prompt, device="cpu") is None
    # cleanup the DB rows we created (best-effort)
    try:
        import sqlite3
        conn = sqlite3.connect(state_pool.DB_PATH)
        conn.execute("DELETE FROM prefix_cache WHERE state_id LIKE 'X:%' OR state_id LIKE 'Y:%'")
        conn.commit()
        conn.close()
    except Exception:
        pass


def test_scoped_proxy_is_stateless_view():
    base = get_state_manager()
    sx = get_state_manager(model="X")
    sy = get_state_manager(model="Y")
    # the proxy is a fresh lightweight view, but resolves the same singleton
    assert sx._manager is base
    assert sy._manager is base
    assert sx is not sy