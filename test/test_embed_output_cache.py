"""
Hermetic coverage for the A1 embedding output LRU + within-batch dedupe in
infer/embed_aggregator.py (RWKV_EMBED_CACHE, RWKV_EMBED_CACHE_CAPACITY).

Unit under test is the cache bookkeeping (hit/miss splice, checkpoint-fingerprint
invalidation, within-batch dedupe, LRU eviction), so ``embed_texts`` is replaced
by the same deterministic CPU stand-in as test_embed_aggregator.py. The cache is
off by default; these tests opt in by monkeypatching ``settings.embed_cache``
BEFORE constructing the aggregator (that is when the LRU is allocated).

The disabled (inline) path is exercised here because it routes every submit()
through the same ``_embed_with_cache`` choke point the aggregated path uses;
aggregation batching itself is already covered in test_embed_aggregator.py.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

pytest.importorskip("torch", reason="torch not installed (source env.sh first)")

# Avoid pulling the CUDA-bound infer/embedding module in at cold import (its JIT
# entry needs CUDA_HOME). These tests never call the real embed_texts -- the
# collector stand-in replaces it right after import -- so stub the module to make
# this unit test hermetic and CPU-runnable rather than gated on CUDA_HOME.
import types
_stub_embedding = types.ModuleType("infer.embedding")
_stub_embedding.embed_texts = lambda *a, **k: (_ for _ in ()).throw(
    NotImplementedError("stubbed; tests replace embed_texts")
)
sys.modules.setdefault("infer.embedding", _stub_embedding)

try:
    import infer.embed_aggregator as agg_mod
except OSError as exc:
    pytest.skip(f"CUDA environment not configured: {exc}", allow_module_level=True)

from infer.embed_aggregator import EmbedAggregator
from settings import settings


class _FakeModel:
    def __init__(self, max_prefill_bsz=0):
        self.max_prefill_bsz = max_prefill_bsz


def _vec(text):
    return [float(len(text)), float(sum(ord(c) for c in text))]


class _Collector:
    def __init__(self):
        self.calls = []

    def embed(self, model, tokenizer, texts, normalize=True):
        self.calls.append(list(texts))
        return [_vec(t) for t in texts]


@pytest.fixture
def fake_collector(monkeypatch):
    collector = _Collector()
    monkeypatch.setattr(agg_mod, "embed_texts", collector.embed)
    return collector


@pytest.fixture
def cache_on(monkeypatch):
    monkeypatch.setattr(settings, "embed_cache", True)
    monkeypatch.setattr(settings, "embed_cache_capacity", 64)
    return None


def _agg(cache_on, enabled=False, ns="fp:1:2"):
    return EmbedAggregator(_FakeModel(), None, enabled=enabled, cache_namespace_=ns)


def test_cache_off_by_default_calls_embed_texts_every_time(fake_collector):
    """A1 default-off: with settings.embed_cache False (the default), the
    aggregator allocates no LRU and every submit hits embed_texts -- byte
    identical to the pre-feature behavior."""
    agg = _agg(cache_on=None, enabled=False, ns="fp:1:2")
    assert agg._cache is None
    assert agg._embed_with_cache(["z"]) == [_vec("z")]
    assert agg._embed_with_cache(["z"]) == [_vec("z")]
    assert fake_collector.calls == [["z"], ["z"]]


def test_cache_hit_splices_miss_subset_in_order(fake_collector, cache_on):
    """A1: a request whose text is already cached only embeds the uncached
    subset, then splices the cached vectors back in the exact input order."""
    agg = _agg(cache_on, enabled=False, ns="ns1")
    assert agg._cache is not None

    first = agg._embed_with_cache(["a", "b"])
    assert first == [_vec("a"), _vec("b")]
    assert fake_collector.calls == [["a", "b"]]

    # 'b' is cached; only 'c' is a miss -> embed_texts sees just ['c']
    second = agg._embed_with_cache(["b", "c"])
    assert second == [_vec("b"), _vec("c")]
    assert fake_collector.calls == [["a", "b"], ["c"]]


def test_within_batch_dedupe_embeds_distinct_texts_once(fake_collector, cache_on):
    """A-extra: duplicate texts in one request are embedded exactly once and
    fanned back out to preserve row order."""
    agg = _agg(cache_on, enabled=False, ns="ns2")
    result = agg._embed_with_cache(["x", "x", "y"])
    assert result == [_vec("x"), _vec("x"), _vec("y")]
    # only the distinct texts reached embed_texts
    assert fake_collector.calls == [["x", "y"]]


def test_fingerprint_change_invalidates_cache(fake_collector, cache_on):
    """A1/N1: a different checkpoint fingerprint (a runtime model reload /
    checkpoint swap yields a new cache_namespace) must start the cache cold --
    the same text must NOT be served from the previous fingerprint's entry."""
    ns1 = _agg(cache_on, enabled=False, ns="modelA:size:123")
    ns1._embed_with_cache(["z"])  # embeds + caches under ns1
    assert fake_collector.calls == [["z"]]

    ns2 = _agg(cache_on, enabled=False, ns="modelA:size:456")  # swapped checkpoint
    ns2._embed_with_cache(["z"])
    assert fake_collector.calls == [["z"], ["z"]], (
        "a new checkpoint fingerprint must not reuse the old cache entry"
    )


def test_lru_eviction_bounds_memory(fake_collector, monkeypatch, cache_on):
    """A1: with a small capacity the LRU evicts the least-recently-used entry,
    so a far-sooner-embedded text is recomputed after eviction."""
    monkeypatch.setattr(settings, "embed_cache_capacity", 2)
    agg = _agg(cache_on, enabled=False, ns="ns3")
    agg._embed_with_cache(["a"])
    agg._embed_with_cache(["b"])
    agg._embed_with_cache(["c"])  # evicts 'a' (LRU)
    assert fake_collector.calls == [["a"], ["b"], ["c"]]
    # 'a' was evicted -> re-embed; 'c' still cached -> no embed call from it
    agg._embed_with_cache(["a", "c"])
    assert fake_collector.calls == [["a"], ["b"], ["c"], ["a"]]


def test_no_fingerprint_disables_cache_safely(fake_collector, cache_on):
    """A1 guard: even with the cache knob on, an aggregator bound with no
    fingerprint (e.g. a test harness that never installs a ModelManager) must
    NOT cache -- it falls back to plain embed_texts (avoids the None-namespace
    stale-vector hazard)."""
    agg = EmbedAggregator(_FakeModel(), None, enabled=False, cache_namespace_=None)
    assert agg._cache_ns is None  # no fingerprint -> _embed_with_cache must not cache
    assert agg._embed_with_cache(["q"]) == [_vec("q")]
    assert agg._embed_with_cache(["q"]) == [_vec("q")]
    assert fake_collector.calls == [["q"], ["q"]]


def test_miss_path_returns_fresh_copies_no_cache_corruption(fake_collector, cache_on):
    """SF-3 regression: on a cache MISS, mutating the returned vector must NOT
    corrupt the cache entry, and duplicate rows in one request must NOT alias
    each other (each row is a fresh list)."""
    agg = _agg(cache_on, enabled=False, ns="nsX")
    out = agg._embed_with_cache(["dup", "dup", "other"])
    assert fake_collector.calls == [["dup", "other"]]  # dedup embedded once
    assert out == [_vec("dup"), _vec("dup"), _vec("other")]
    # the two 'dup' rows are DIFFERENT list objects (no aliasing)
    assert out[0] is not out[1]
    # mutating one returned row must not corrupt the cached entry
    out[0].append(999.0)
    hit = agg._embed_with_cache(["dup"])
    assert hit[0] == _vec("dup"), "cache entry must not reflect the mutation"