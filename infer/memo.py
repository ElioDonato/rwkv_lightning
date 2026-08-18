"""Small bounded LRU shared by the opt-in caching features.

A ``BoundedLRU`` maps an exact cache key (a tuple) to a non-None object, evicting
the least-recently-used entry once it exceeds ``capacity``. It is intentionally
not thread-safe: each instance is confined to one call site on one thread (the
embedding LRU runs on the event-loop thread; the tokenize-encode memo on
whichever thread tokenizes), so no locking is needed here.

Values must never be ``None`` -- ``get()`` returns the module-level ``MISS``
sentinel on a miss so a cached ``None`` can't be confused with one.
"""

from collections import OrderedDict
import threading

MISS = object()


class BoundedLRU:
    def __init__(self, capacity: int):
        # Guard against a pathological capacity (0 / negative / huge) so the LRU
        # never becomes a degenerate unbounded dict or a no-op that refuses to
        # store anything (mouse-proof but useless).
        cap = int(capacity or 0)
        self._capacity = cap if cap > 0 else 1
        self._d: OrderedDict = OrderedDict()
        # Serializes get/put so two threads (e.g. the encode memo on :8081, where
        # prefill encodes on the threadpool AND dynamic-batch encodes on the
        # event loop) can't interleave a concurrent put's eviction between a
        # get's move_to_end and its __getitem__ (a KeyError out of a request).
        self._lock = threading.Lock()

    def get(self, key):
        with self._lock:
            if key not in self._d:
                return MISS
            self._d.move_to_end(key)
            return self._d[key]

    def put(self, key, value):
        if value is None:
            raise ValueError("BoundedLRU values must be non-None objects")
        with self._lock:
            self._d[key] = value
            self._d.move_to_end(key)
            while len(self._d) > self._capacity:
                self._d.popitem(last=False)

    def __contains__(self, key):
        with self._lock:
            return key in self._d

    def __len__(self):
        with self._lock:
            return len(self._d)

    def clear(self):
        with self._lock:
            self._d.clear()