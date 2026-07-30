"""
Coverage for infer/batch_inference.py's V1 streaming decode loop
(batch_infer_stream) with decode-time row compaction (commit ca72bb0).

Not a standard pytest test: requires a real loaded model and GPU. Run from
the repo root with a model path:

    uv run python test/verify_batch_v1_stream_compaction.py models/<model-dir>

Mirrors verify_batch_v2_decode.py's misattribution-testing convention:
8 prompts each required to repeat a distinctive unique number, so any
per-row indexing bug in the penalties tensor, sample_rand_states, or
active_indices remapping after compaction would show up as one prompt's
output containing a *different* prompt's number.

V1 streaming differs from V2 streaming in the sampler: V1 uses
batch_sampling_repetition_temperature_topk_topp with penalties[B,vocab]
and sample_rand_states[B,...] tensors that get compacted via index_select
when rows finish. This test proves that compaction preserves correct
per-row attribution under staggered finish conditions.
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


async def _run_stream_v1(engine, prompts, **kwargs):
    """Collect all SSE content from batch_infer_stream into per-index strings."""
    contents = {i: "" for i in range(len(prompts))}
    finish_reasons = {}
    async for chunk in engine.batch_infer_stream(prompts, **kwargs):
        if chunk.startswith("data: ") and not chunk.strip().endswith("[DONE]"):
            payload = json.loads(chunk[6:].strip())
            for choice in payload.get("choices", []):
                idx = choice["index"]
                contents[idx] += choice["delta"].get("content", "")
                if "finish_reason" in choice:
                    finish_reasons[idx] = choice["finish_reason"]
    return contents, finish_reasons


async def check_batch_infer_stream_v1(engine):
    """batch_infer_stream with default chunk_size: exercises the compaction
    path (rows finish at different times due to different repeat counts)."""
    prompts = _make_prompts()
    contents, reasons = await _run_stream_v1(
        engine, prompts,
        max_length=80, temperature=0.01, top_k=50, top_p=0.6,
        stop_tokens=("DONE",), chunk_size=2,
    )
    for i in range(len(prompts)):
        print(f"  [batch_infer_stream_v1][{i}] finish_reason={reasons.get(i, '?')} {contents[i]!r}")
    # Verify finish_reasons are present
    for i in range(len(prompts)):
        assert i in reasons, f"index {i} missing finish_reason"
    return _check_attribution(contents, "batch_infer_stream_v1")


async def check_stream_v1_staggered(engine):
    """Force staggered finishes: short prompts finish early, long ones late.
    This maximizes compaction events mid-decode."""
    prompts = _make_prompts()  # repeat counts 1,1,3,3,6,6,10,10 → staggered
    contents, reasons = await _run_stream_v1(
        engine, prompts,
        max_length=120, temperature=0.01, top_k=50, top_p=0.6,
        stop_tokens=("DONE",), chunk_size=1,  # chunk_size=1 for max SSE granularity
    )
    for i in range(len(prompts)):
        print(f"  [staggered-v1][{i}] finish_reason={reasons.get(i, '?')} {contents[i]!r}")
    return _check_attribution(contents, "staggered-v1")


def check_batch_generate_v1(engine):
    """batch_generate (non-streaming V1) also returns (decoded, reasons) now."""
    prompts = _make_prompts()
    decoded, reasons = engine.batch_generate(
        prompts,
        max_length=80,
        temperature=0.01,
        top_k=50,
        top_p=0.6,
        stop_tokens=("DONE",),
    )
    assert len(decoded) == len(prompts), f"expected {len(prompts)} outputs, got {len(decoded)}"
    assert len(reasons) == len(prompts), f"expected {len(prompts)} reasons, got {len(reasons)}"
    for i, (text, reason) in enumerate(zip(decoded, reasons)):
        print(f"  [batch_generate_v1][{i}] finish_reason={reason} {text!r}")
    return _check_attribution({i: t for i, t in enumerate(decoded)}, "batch_generate_v1")


async def main():
    if len(sys.argv) < 2:
        print("Usage: uv run python test/verify_batch_v1_stream_compaction.py models/<model-dir>")
        sys.exit(1)

    model_path = sys.argv[1]
    print(f"[INFO] Loading {model_path} ...")
    engine = _load_engine(model_path)
    print("[INFO] Model loaded successfully.\n")

    print("=== Check 1: batch_generate (V1 non-streaming) per-prompt attribution ===")
    ok1 = check_batch_generate_v1(engine)

    print("\n=== Check 2: batch_infer_stream (V1 streaming + compaction) ===")
    ok2 = await check_batch_infer_stream_v1(engine)

    print("\n=== Check 3: batch_infer_stream with staggered finish lengths ===")
    ok3 = await check_stream_v1_staggered(engine)

    if ok1 and ok2 and ok3:
        print("\nALL CHECKS PASSED")
    else:
        print("\nSOME CHECKS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
