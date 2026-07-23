"""
Verification for the /big_batch/completions decode-time slot-compaction
feature (infer/big_batch.py's _compact_active_rows / active_indices
mechanism, added alongside the per-item finish_reason SSE signal).

Not a standard pytest test: requires a real loaded model and GPU, like
test_local_state_and_batch.py. Run from the repo root with a model path:

    uv run python test/verify_batch_compaction.py models/<model-dir>

Covers exactly the two things an independent controller review flagged as
essential and non-obvious for this kind of change (hot decode loop, tensor
index bookkeeping across a shrinking batch):

1. No output misattribution across multiple composed compaction waves,
   using distinctive per-prompt expected content so a swapped/off-by-one
   row lookup would be immediately visible (not just "different but still
   plausible" -- categorically wrong).
2. That any observed ON-vs-OFF (compaction enabled vs disabled) content
   divergence at temperature>0 is a PRE-EXISTING property of this
   codebase's batch-size-dependent fp16 kernel numerics (confirmed by
   testing with compaction code entirely absent, at multiple batch sizes,
   comparing the same prompt across bsz 1/2/4/8), not something this
   feature introduced -- and that greedy decoding (temperature=0) is
   unaffected, since it doesn't have sampling noise to amplify tiny
   floating-point differences into a different token choice.

See reports/00-implementation-notes.md and reports/01-controller-review.md
(in the original feature/batch-early-return review worktree, not
necessarily present in this checkout) for the full investigation this
distilled from.
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


async def _run(engine, prompts, max_length, temperature, stop_tokens, chunk_size, seed=None):
    import torch

    if seed is not None:
        torch.manual_seed(seed)
    contents = {i: "" for i in range(len(prompts))}
    finish_reasons = {}
    async for chunk in engine.big_batch_stream(
        prompts=prompts, max_length=max_length, temperature=temperature,
        stop_tokens=stop_tokens, chunk_size=chunk_size,
    ):
        if chunk.startswith("data: ") and not chunk.strip().endswith("[DONE]"):
            payload = json.loads(chunk[6:].strip())
            for choice in payload.get("choices", []):
                idx = choice["index"]
                contents[idx] += choice["delta"].get("content", "")
                if choice.get("finish_reason"):
                    finish_reasons[idx] = choice["finish_reason"]
    return contents, finish_reasons


async def check_no_misattribution(engine):
    """8 prompts, each requiring a distinctive unique number in its output,
    staggered finish lengths forcing >=3 separate compaction waves. A
    swapped/off-by-one row lookup would show prompt i's output containing
    a DIFFERENT prompt's expected number -- categorically distinguishable
    from ordinary sampling variation."""
    numbers = [111, 222, 333, 444, 555, 666, 777, 888]
    repeat_counts = [1, 1, 3, 3, 6, 6, 10, 10]
    prompts = [
        f"Repeat the number {n} exactly {c} times, separated by spaces, then say DONE. Output nothing else."
        for n, c in zip(numbers, repeat_counts)
    ]
    contents, finish_reasons = await _run(
        engine, prompts, max_length=80, temperature=0.01,
        stop_tokens=("DONE",), chunk_size=2, seed=42,
    )

    ok = True
    for i, n in enumerate(numbers):
        text = contents[i]
        has_own = str(n) in text
        leaked = [str(other) for other in numbers if other != n and str(other) in text]
        if not has_own or leaked:
            ok = False
            print(f"  MISATTRIBUTION at index {i}: expected={n} has_own={has_own} leaked={leaked}")
            print(f"    text={text!r}")
    if ok:
        print("  PASS: all 8 indices correctly attributed, zero cross-contamination")
    return ok


async def check_batchsize_numerics_isolation(model, tokenizer, args, rocm_flag):
    """Confirms batch-size-dependent output variation exists WITHOUT any
    compaction ever actually firing -- i.e. it's a pre-existing property
    of the batched fp16 kernels, not something the compaction feature
    introduced. Uses the real, current big_batch_stream (compaction code
    included) but with filler prompts chosen so EVERY prompt in the batch
    runs the full max_length without ever hitting a stop condition -- the
    batch never shrinks, so _compact_active_rows is never actually
    invoked regardless of which version of the file is present. This
    isolates pure batch-size numerics without depending on any temporary
    backup file that may not exist in future checkouts."""
    target = "Tell me a short story about a robot learning to paint, at least 6 sentences."
    # Fillers deliberately open-ended (no natural stopping point within
    # max_length, and a stop_tokens value that won't realistically appear)
    # so they run the full max_length just like the target prompt --
    # batch size stays constant for the whole call, compaction never fires.
    fillers = [
        "List as many synonyms as you can think of for the word happy, one per line.",
        "Describe the water cycle in as much detail as possible.",
        "Explain how a car engine works, step by step, in detail.",
        "Describe the plot of a novel about time travel in detail.",
        "List the planets of the solar system and describe each one.",
        "Explain the rules of chess in detail.",
        "Describe how photosynthesis works in plants, in detail.",
    ]

    from infer.inference import InferenceEngine
    engine = InferenceEngine(model=model, tokenizer=tokenizer, args=args, rocm_flag=rocm_flag)

    results_temp0 = {}
    results_temp1 = {}
    for bsz in (1, 2, 4, 8):
        prompts = [target] + fillers[: bsz - 1]
        r0, fr0 = await _run(engine, prompts, max_length=60, temperature=0.0,
                              stop_tokens=("\x00no-such-stop\x00",), chunk_size=4, seed=999)
        r1, fr1 = await _run(engine, prompts, max_length=60, temperature=1.0,
                              stop_tokens=("\x00no-such-stop\x00",), chunk_size=4, seed=999)
        results_temp0[bsz] = r0[0]
        results_temp1[bsz] = r1[0]
        # Sanity: every prompt should exhaust max_length (finish_reason
        # "length" for all), confirming the batch genuinely never shrank
        # and compaction genuinely never fired during this measurement.
        assert all(v == "length" for v in fr0.values()), (
            f"bsz={bsz} temp=0: expected all finish_reason='length' (no compaction), got {fr0}"
        )
        assert all(v == "length" for v in fr1.values()), (
            f"bsz={bsz} temp=1: expected all finish_reason='length' (no compaction), got {fr1}"
        )

    temp0_identical = all(results_temp0[b] == results_temp0[1] for b in (2, 4, 8))
    temp1_any_diverge = any(results_temp1[b] != results_temp1[1] for b in (2, 4, 8))

    print(f"  temperature=0 identical across bsz 1/2/4/8: {temp0_identical}")
    print(f"  temperature=1 diverges for at least one bsz: {temp1_any_diverge}")
    print("  (both True is the expected, already-understood baseline behavior;")
    print("   if temp0_identical becomes False, that IS a new correctness bug")
    print("   worth investigating -- greedy decoding should never depend on bsz)")
    return temp0_identical


async def main():
    if len(sys.argv) < 2:
        print("usage: python test/verify_batch_compaction.py <model_path>")
        sys.exit(1)
    model_path = sys.argv[1]

    print("=== Check 1: no output misattribution across compaction waves ===")
    engine = _load_engine(model_path)
    ok1 = await check_no_misattribution(engine)

    print("\n=== Check 2: batch-size numerics isolation (compaction-independent) ===")
    ok2 = await check_batchsize_numerics_isolation(
        engine.model, engine.tokenizer, engine.args, engine.rocm_flag
    )

    if ok1 and ok2:
        print("\nALL CHECKS PASSED")
    else:
        print("\nSOME CHECKS FAILED -- see output above")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
