"""
Hermetic coverage for the A6 tokenize-encode memoization
(infer/encode_cache.CachedTokenizer, RWKV_ENCODE_CACHE).

The wrapped "tokenizer" is a tiny deterministic stub: encode(text) returns a
list of ints as a pure function of the text. Tests assert: miss calls encode
once; a hit returns a fresh identical list without re-encoding; namespace
isolation (a checkpoint swap never reuses a stale tokenization); LRU eviction;
and that non-bare calls (extra kwargs) are routed through uncached.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from infer.encode_cache import CachedTokenizer


class _StubTokenizer:
    def __init__(self):
        self.encode_calls = []

    def encode(self, text, *args, **kwargs):
        self.encode_calls.append(text)
        if args or kwargs:
            return [9, 9]  # sentinel for a non-bare call
        return [len(text), sum(ord(c) for c in text)]


def test_miss_and_hit_roundtrip():
    tok = _StubTokenizer()
    cached = CachedTokenizer(tok, "ns", 16)
    first = cached.encode("hello world")
    assert first == [11, 1116]
    second = cached.encode("hello world")
    # identical tokens, but a FRESH list (mutating it can't corrupt the cache)
    assert second == first
    assert second is not first
    second.append(-1)
    # cache still holds the untampered tuple
    assert cached.encode("hello world") == [11, 1116]
    assert tok.encode_calls == ["hello world"]  # encode ran exactly once


def test_distinct_text_still_decodes():
    tok = _StubTokenizer()
    cached = CachedTokenizer(tok, "ns", 16)
    assert cached.encode("a") == [1, 97]
    assert cached.encode("b") == [1, 98]
    assert tok.encode_calls == ["a", "b"]


def test_namespace_isolation_no_stale_tokenization():
    a = _StubTokenizer()
    ca = CachedTokenizer(a, "modelA:size:100", 16)
    b = _StubTokenizer()
    cb = CachedTokenizer(b, "modelA:size:200", 16)  # swapped checkpoint
    assert ca.encode("z") == [1, 122]
    cb.encode("z")
    # a new checkpoint namespace must NOT serve 'b' from 'a's cache
    assert b.encode_calls == ["z"]


def test_non_bare_calls_bypass_memo():
    tok = _StubTokenizer()
    cached = CachedTokenizer(tok, "ns", 16)
    assert cached.encode("x", add_special_tokens=True) == [9, 9]
    assert cached.encode("x", add_special_tokens=True) == [9, 9]
    # both went straight through (kwargs not memoized)
    assert tok.encode_calls == ["x", "x"]


def test_lru_eviction_recomputes():
    tok = _StubTokenizer()
    cached = CachedTokenizer(tok, "ns", 2)
    cached.encode("a")
    cached.encode("b")
    cached.encode("c")  # evicts 'a'
    assert tok.encode_calls == ["a", "b", "c"]
    cached.encode("a")  # recomputed after eviction
    assert tok.encode_calls == ["a", "b", "c", "a"]


def test_delegation_of_other_attributes():
    class _Rich:
        eos_token_id = 3
        def decode(self, ids):
            return "d"

    cached = CachedTokenizer(_Rich(), "ns", 4)
    assert cached.eos_token_id == 3
    assert cached.decode([1]) == "d"