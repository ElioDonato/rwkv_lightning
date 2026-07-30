"""
Coverage for infer/batch_inference.py's V2 batch decode loop
(batch_generate_v2 / batch_infer_stream_v2), which previously had zero
test coverage -- distinct from infer/big_batch.py, which
test/verify_batch_compaction.py already covers.

Not a standard pytest test: requires a real loaded model and GPU, same as
test_local_state_and_batch.py and verify_batch_compaction.py. Run from the
repo root with a model path:

    uv run python test/verify_batch_v2_decode.py models/<model-dir>

Follows verify_batch_compaction.py's convention for this kind of test:
use distinctive, easily-misattributed-if-wrong per-prompt expected content
(a unique number each prompt must repeat) so any row/index mixup in the
occurrence-penalty tensors, active_mask, or per-index stop-state list is
categorically visible rather than just "different but plausible" text.

batch_inference.py's V2 loop differs from big_batch.py's in exactly the
things this test targets:
  - per-row `occurrence_count`/`occurrence_presence` tensors indexed via
    `batch_rows[active_mask]` (_sample_v2_tokens) -- a batch_rows/token
    misalignment would show up as one prompt's repetition penalty being
    applied to a different prompt's tokens.
  - a Python list `finished[i]` / `stop_states[i]` per-index bookkeeping
    parallel to the tensor ops, not slot-compaction like big_batch.py --
    so what's tested here is "do per-index reads/writes into flat
    Python lists stay aligned with per-row tensor slices", a different
    failure mode than big_batch.py's shrinking-batch row remapping.
  - batch_generate_v2 (blocking, returns a list[str]) AND
    batch_infer_stream_v2 (async generator, SSE-chunked like big_batch's
    stream) are both covered, since they duplicate the sampling loop
    independently (batch_inference.py:71-280) rather than one wrapping
    the other -- a bug in one would not necessarily show up in the other.
"""
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_engine(model_path):
    from model_load.model_loader import load_model_and_tokenizer
    from infer.inference import InferenceEngine

    model, tokenizer, args, rocm_flag = load_model_and_tokenizer(model_path)
    return InferenceEngine(model=model, tokenizer=tokenizer, args=args, rocm_flag=rocm_flag)


NUMBERS = [111, 222, 333, 444, 555, 666, 777, 888]
REPEAT_COUNTS = [1, 1, 3, 3, 6, 6, 10, 10]


def _make_prompts():
    return [
        f"Repeat the number {n} exactly {c} times, separated by spaces, then say DONE. Output nothing else."
        for n, c in zip(NUMBERS, REPEAT_COUNTS)
    ]


def _check_attribution(contents, label):
    ok = True
    for i, n in enumerate(NUMBERS):
        text = contents[i]
        has_own = str(n) in text
        leaked = [str(other) for other in NUMBERS if other != n and str(other) in text]
        if not has_own or leaked:
            ok = False
            print(f"  [{label}] MISATTRIBUTION at index {i}: expected={n} has_own={has_own} leaked={leaked}")
            print(f"    [{label}] text={text!r}")
    if ok:
        print(f"  [{label}] PASS: all {len(NUMBERS)} indices correctly attributed, zero cross-contamination")
    return ok


def check_batch_generate_v2(engine):
    """batch_generate_v2 is synchronous/blocking and returns (list[str], list[str])
    -- (decoded_texts, finish_reasons). Exercises the finished[]/stop_states[]/
    occurrence tensor bookkeeping in the non-streaming code path."""
    prompts = _make_prompts()
    decoded, reasons = engine.batch_generate_v2(
        prompts,
        max_length=80,
        temperature=0.01,
        top_k=500,
        top_p=0.5,
        stop_tokens=("DONE",),
    )
    assert len(decoded) == len(prompts), f"expected {len(prompts)} outputs, got {len(decoded)}"
    assert len(reasons) == len(prompts), f"expected {len(prompts)} reasons, got {len(reasons)}"
    for i, (text, reason) in enumerate(zip(decoded, reasons)):
        print(f"  [batch_generate_v2][{i}] finish_reason={reason} {text!r}")
    return _check_attribution({i: t for i, t in enumerate(decoded)}, "batch_generate_v2")


async def _run_stream_v2(engine, prompts, **kwargs):
    contents = {i: "" for i in range(len(prompts))}
    async for chunk in engine.batch_infer_stream_v2(prompts, **kwargs):
        if chunk.startswith("data: ") and not chunk.strip().endswith("[DONE]"):
            payload = json.loads(chunk[6:].strip())
            for choice in payload.get("choices", []):
                idx = choice["index"]
                contents[idx] += choice["delta"].get("content", "")
    return contents


async def check_batch_infer_stream_v2(engine):
    """batch_infer_stream_v2 is the async-generator SSE-chunked sibling
    of batch_generate_v2, with its own independent copy of the sampling
    loop plus chunk-buffering logic (text_buffers[i], chunk_token_counts[i])
    -- exercised here with a small chunk_size so multiple partial chunks
    per prompt actually get emitted and reassembled, not just one shot."""
    prompts = _make_prompts()
    contents = await _run_stream_v2(
        engine, prompts,
        max_length=80, temperature=0.01, top_k=500, top_p=0.5,
        stop_tokens=("DONE",), chunk_size=2,
    )
    for i in range(len(prompts)):
        print(f"  [batch_infer_stream_v2][{i}] {contents[i]!r}")
    return _check_attribution(contents, "batch_infer_stream_v2")


async def check_stream_v2_varying_lengths_no_misattribution(engine):
    """Same idea but forces prompts to finish at staggered times (varying
    stop-token arrival), so `active_mask`/`finished[]` differ per row for
    much of the loop -- the scenario most likely to expose an
    index-alignment bug between the per-row tensors and the flat Python
    per-index lists, since a uniform-length batch could mask that."""
    prompts = _make_prompts()  # REPEAT_COUNTS already staggers natural finish length
    contents = await _run_stream_v2(
        engine, prompts,
        max_length=120, temperature=0.01, top_k=500, top_p=0.5,
        stop_tokens=("DONE",), chunk_size=3,
    )
    for i in range(len(prompts)):
        print(f"  [staggered][{i}] {contents[i]!r}")
    return _check_attribution(contents, "staggered-lengths")


async def main():
    if len(sys.argv) < 2:
        print("usage: python test/verify_batch_v2_decode.py <model_path>")
        sys.exit(1)
    model_path = sys.argv[1]

    print(f"[INFO] Loading {model_path} ...")
    engine = _load_engine(model_path)

    print("\n=== Check 1: batch_generate_v2 (blocking) per-prompt attribution ===")
    ok1 = check_batch_generate_v2(engine)

    print("\n=== Check 2: batch_infer_stream_v2 (async SSE) per-prompt attribution ===")
    ok2 = await check_batch_infer_stream_v2(engine)

    print("\n=== Check 3: batch_infer_stream_v2 with staggered finish lengths ===")
    ok3 = await check_stream_v2_varying_lengths_no_misattribution(engine)

    if ok1 and ok2 and ok3:
        print("\nALL CHECKS PASSED")
    else:
        print("\nSOME CHECKS FAILED -- see output above")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
