"""Opt-in tokenizer encode memoization (A6, RWKV_ENCODE_CACHE).

RWKV's tokenizer is a stateless pure function: ``encode(text)`` yields the same
token-id list for the same text every time. Many call sites encode the SAME
prompt text repeatedly -- the chat prefill encodes the formatted prompt once and
``build_openai_usage`` re-encodes it for usage counters; embed usage re-encodes
the batch. ``CachedTokenizer`` memoizes ``encode`` by ``(namespace, text)`` so
repeated encodes of the same text are cheap tuple lookups.

Correctness contract
--------------------
* A cache hit returns a FRESH ``list`` of the same tokens a fresh ``encode``
  would produce -- it never short-circuits the existing per-request encode call
  (each call site still calls ``encode``; the memo only removes the *redundant*
  work of re-deriving identical tokens), so usage counting stays intact.
* The key includes a checkpoint ``namespace`` (``cache_namespace(slot)``), never
  None-for-default when the feature is on, so a runtime checkpoint swap /
  vocab change can never return stale tokenization (N1).
* Only bare ``encode(text)`` calls are memoized. Calls that carry extra kwargs /
  positional args (e.g. any future ``add_special_tokens=`` variant) are routed
  straight through uncached, so no memo can collide across differing encode
  signatures.
* Default OFF: ``CachedTokenizer`` is only ever INSTALLED when RWKV_ENCODE_CACHE
  is set; otherwise the plain tokenizer is used unchanged (byte-identical, and
  no object-identity swap).
"""

from infer.memo import MISS, BoundedLRU


class CachedTokenizer:
    def __init__(self, tokenizer, namespace: str, capacity: int):
        self._tok = tokenizer
        self._ns = namespace if namespace else "_unknown"
        self._cache = BoundedLRU(capacity)

    def encode(self, text, *args, **kwargs):
        if args or kwargs:
            # Not a bare text->ids call; don't risk colliding across signatures.
            return self._tok.encode(text, *args, **kwargs)
        key = (self._ns, text)
        cached = self._cache.get(key)
        if cached is not MISS:
            return list(cached)  # fresh list, matches tokenizer.encode's return
        ids = self._tok.encode(text)
        self._cache.put(key, tuple(ids))
        return ids

    def __getattr__(self, name):
        # Delegate everything else (decode, eos_token_id, vocab, ...) to the
        # wrapped tokenizer.
        return getattr(self._tok, name)