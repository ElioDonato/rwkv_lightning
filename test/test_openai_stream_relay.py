"""
Hermetic coverage for the OpenAI chat streaming relay ``_relay_openai_stream``
(API_servers/router/openai_routes.py) -- the default-path SSE relay shared by
the feature-off ``stream_openai_chunks`` path AND the opt-in fuse-aggregator
stream (item 4, CR1 F + CR2 M2).

Zero network / GPU / model. The relay is the FINAL client-facing adapter that
re-stamps id/created/model onto the engine's SSE chunks, emits the finish
chunk (+ [DONE]) exactly once, and -- critically -- owns ``await stream.aclose()``
in its ``finally`` so the underlying engine stream (solo single_infer_stream or
a fused job stream) is always released, even on a mid-stream cancel (client
disconnect). This file pins three contracts:

  (a) FINISH DE-DUP: a fused-style source that ALREADY carries a per-row finish
      chunk yields that finish exactly ONCE (the trailing [DONE] must not emit a
      duplicate finish); a solo-style source without a finish chunk gets the
      finish synthesized once at [DONE]. In BOTH cases the relay emits exactly
      one finish chunk and one [DONE].
  (b) aclose-ALWAYS: on normal exhaustion AND on a mid-stream close of the relay
      (the disconnect path: the body generator finalizes -> relay finally ->
      ``await stream.aclose()``) the underlying engine stream's aclose is
      awaited -- no orphaned GPU row.
  (c) Cancel-parallel: once CancellationToken is cancelled the relay emits no
      finish chunk and no [DONE] (bytes stop immediately), while still releasing
      the underlying stream in the finally.
"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

pytest.importorskip("torch", reason="torch not installed (source env.sh first)")

try:
    from API_servers.router.openai_routes import _relay_openai_stream
except OSError as exc:
    pytest.skip(f"CUDA environment not configured: {exc}", allow_module_level=True)

from infer.cancellation import CancellationToken


# -- helpers ----------------------------------------------------------------

def _content_chunk(text: str) -> str:
    """A content SSE chunk in the engine's stream format (index-0 choices)."""
    data = json.dumps(
        {"object": "chat.completion.chunk",
         "choices": [{"index": 0, "delta": {"content": text}}]},
        ensure_ascii=False,
    )
    return f"data: {data}\n\n"


def _finish_chunk(finish_reason: str = "stop") -> str:
    """A per-row finish SSE chunk the fuse path emits (delta {} + finish_reason)."""
    data = json.dumps(
        {"object": "chat.completion.chunk",
         "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}]},
        ensure_ascii=False,
    )
    return f"data: {data}\n\n"


class _FakeStream:
    """Deterministic engine-side SSE stream: records whether its aclose() was
    awaited (no-orphan assertion) and which items were yielded."""

    def __init__(self, items):
        self._items = list(items)
        self.aclosed = False
        self.yielded = []

    def __aiter__(self):
        return self._agen()

    async def _agen(self):
        for item in self._items:
            self.yielded.append(item)
            yield item

    async def aclose(self):
        self.aclosed = True


async def _drain_relay(agen):
    """Collect every SSE string a relay generator yields."""
    out = []
    async for item in agen:
        out.append(item)
    return out


def _choices(sses):
    """Normalize relayed chunks into [(kind, payload)] for assertion."""
    kinds = []
    for s in sses:
        payload = s[len("data: "):].strip()
        if payload == "[DONE]":
            kinds.append(("done", None))
            continue
        data = json.loads(payload)
        c0 = (data.get("choices") or [{}])[0]
        if c0.get("finish_reason") is not None:
            kinds.append(("finish", c0["finish_reason"]))
        else:
            delta = c0.get("delta") or {}
            if delta.get("role") == "assistant":
                kinds.append(("start", None))
            elif delta.get("content"):
                kinds.append(("content", delta["content"]))
    return kinds


# ---------------------------------------------------------------------------
# (a) FINISH DE-DUP: exactly one finish + one [DONE], whether the source already
# carried a per-row finish (fused) or not (solo).
# ---------------------------------------------------------------------------

def test_relay_finish_dedup_fused_source_emits_one_finish_then_done():
    """Fused-style source: content + a per-row finish chunk, no source [DONE].
    The relay must emit the finish exactly ONCE and then its own [DONE] -- the
    trailing [DONE] must NOT synthesize a duplicate finish."""
    stream = _FakeStream([
        _content_chunk("Hi "), _content_chunk("there"),
        _finish_chunk("stop"),           # fused path's per-row finish
    ])

    gen = _relay_openai_stream(
        stream, "resp-1", 1234, "model-x", CancellationToken()
    )
    sses = _run(_drain_relay(gen))
    kinds = _choices(sses)

    assert kinds == [("start", None), ("content", "Hi "), ("content", "there"),
                     ("finish", "stop"), ("done", None)], kinds
    # The finish de-dup: the source finish was forwarded, NOT re-emitted at [DONE].
    assert kinds.count(("finish", "stop")) == 1
    assert stream.aclosed, "relay must aclose the underlying stream on exhaustion"


def test_relay_solo_source_emits_finish_on_done():
    """Solo-style source (single_infer_stream): content chunks then a source
    `data: [DONE]` with NO per-row finish. The relay synthesizes the finish "stop"
    exactly once at [DONE], then emits [DONE]."""
    stream = _FakeStream([
        _content_chunk("solo "),
        "data: [DONE]\n\n",              # solo path's terminal [DONE]
    ])

    gen = _relay_openai_stream(
        stream, "resp-2", 5678, "model-x", CancellationToken()
    )
    sses = _run(_drain_relay(gen))
    kinds = _choices(sses)

    assert kinds == [("start", None), ("content", "solo "),
                     ("finish", "stop"), ("done", None)], kinds
    assert kinds.count(("finish", "stop")) == 1
    assert stream.aclosed


# ---------------------------------------------------------------------------
# (b) aclose-ALWAYS / mid-stream cancel: no orphaned GPU row.
# ---------------------------------------------------------------------------

def test_relay_cancel_while_iterating_emits_nothing_after_cancel_and_aclos():
    """Cancel while the relay continues to iterate a finite stream: after the
    token is cancelled the relay emits no further finish/[DONE], but still
    aclose()s the underlying stream in its finally."""
    stream = _FakeStream([_content_chunk("a "), _content_chunk("b ")])
    token = CancellationToken()
    token.cancel()                                   # cancel before any iteration

    gen = _relay_openai_stream(stream, "resp-4", 11, "model-x", token)
    sses = _run(_drain_relay(gen))

    # Cancelled: only the start chunk is present (content suppressed), and NO
    # finish / [DONE] may be emitted.
    kinds = _choices(sses)
    assert kinds == [("start", None)], kinds
    assert not any(k[0] == "finish" for k in kinds)
    assert not any(k[0] == "done" for k in kinds)
    assert stream.aclosed, "relay must aclose the stream even when cancelled"


# -- tiny asyncio runner (no pytest-anyio dependency) --------------------------

def _run(coro):
    import asyncio
    return asyncio.run(coro)


def test_relay_cancel_midstream_closes_underlying_stream_no_orphan():
    """The disconnect path: the relay is closed mid-stream (the route body
    generator finalizes on client disconnect). The relay's ``finally`` must
    ``await stream.aclose()`` so the underlying engine row is released -- no
    orphan. (One content item was already delivered before the close.)"""
    stream = _FakeStream([
        _content_chunk("partial "),
        _content_chunk("more "),
        _content_chunk("never "),
    ])
    token = CancellationToken()

    async def _drive():
        gen = _relay_openai_stream(stream, "resp-3", 9, "model-x", token)
        it = gen.__aiter__()
        first = await anext(it)           # start chunk
        second = await anext(it)          # first content
        assert "partial" in second
        token.cancel()                    # client disconnects
        await it.aclose()                 # route body finalizes relay
        return stream.aclosed, list(stream.yielded)

    aclosed, yielded = _run(_drive())

    assert aclosed, "relay finally must aclose the engine stream mid-stream"
    # No orphan: the row is released even though the stream was interrupted.
    assert yielded == [_content_chunk("partial ")]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))