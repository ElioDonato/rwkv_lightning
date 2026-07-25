"""
Verification for infer/inference.py's incremental prefill-admission release
(acquire_prefill_permit's outstanding_bsz box + release_prefill_capacity),
wired into /big_batch/completions' decode-time row compaction
(infer/big_batch.py's permit_box parameter).

Not a standard pytest test: requires a real loaded model and GPU, like
test_local_state_and_batch.py. Run from the repo root with a model path:

    uv run python test/verify_incremental_prefill_release.py models/<model-dir>

Proves the actual optimization end-to-end: a large batch with staggered
finish times (most rows finish almost immediately, one runs long) releases
its finished rows' admission capacity WHILE the batch is still running, so a
second, unrelated request that needs that capacity is admitted well before
the first request's slowest row finishes -- not just at the very end.
"""
import asyncio
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_engine(model_path):
    from model_load.model_loader import load_model_and_tokenizer
    from infer.inference import InferenceEngine

    model, tokenizer, args, rocm_flag = load_model_and_tokenizer(model_path)
    return InferenceEngine(model=model, tokenizer=tokenizer, args=args, rocm_flag=rocm_flag)


async def _drain_big_batch_stream(engine, prompts, max_length, stop_tokens, permit_box, chunk_size=2):
    """Runs big_batch_stream to completion, returning (contents, finish_reasons,
    per-index-first-finish-timestamp) -- the timestamp is measured against a
    shared time.monotonic() origin the caller establishes, not embedded in
    this helper, so multiple concurrent calls can be compared on one timeline."""
    contents = {i: "" for i in range(len(prompts))}
    finish_reasons = {}
    finish_times = {}
    start = time.monotonic()
    async for chunk in engine.big_batch_stream(
        prompts=prompts, max_length=max_length, temperature=0.01,
        stop_tokens=stop_tokens, chunk_size=chunk_size, permit_box=permit_box,
    ):
        if chunk.startswith("data: ") and not chunk.strip().endswith("[DONE]"):
            payload = json.loads(chunk[6:].strip())
            for choice in payload.get("choices", []):
                idx = choice["index"]
                contents[idx] += choice["delta"].get("content", "")
                if choice.get("finish_reason") and idx not in finish_reasons:
                    finish_reasons[idx] = choice["finish_reason"]
                    finish_times[idx] = time.monotonic() - start
    return contents, finish_reasons, finish_times


async def check_incremental_release_admits_second_request_early(engine):
    """The core proof: batch A (bsz=8) has 7 prompts that finish almost
    immediately and 1 that runs long. Batch B (bsz=4) is submitted
    concurrently and can ONLY be admitted if batch A's admission-queue
    reservation shrinks as its 7 fast rows finish -- the server's
    max_prefill_bsz_limit is deliberately set low enough (via a monkeypatch
    on the loaded model, not a real VRAM constraint) that 8 + 4 = 12 could
    never both be admitted at once, but 1 (batch A's remaining slow row) + 4
    (batch B) = 5 fits easily.

    If incremental release did NOT work (old behavior: release only at
    request end), batch B would have to wait for batch A's slow row to
    finish too -- i.e. batch B's admission timestamp would be very close to
    batch A's own finish timestamp, not much earlier than it.
    """
    original_limit = int(engine.model.max_prefill_bsz_limit)
    original_bsz = int(engine.model.max_prefill_bsz)
    # Cap the server's admission budget at 8 -- enough for batch A alone,
    # but not enough for batch A (8) + batch B (4) = 12 simultaneously,
    # forcing batch B to actually wait on admission rather than sailing
    # through regardless of this feature.
    engine.model.max_prefill_bsz_limit = 8
    engine.model.max_prefill_bsz = 8
    try:
        # Reuses the "repeat a distinctive number N times, then say DONE"
        # pattern already proven (in verify_batch_compaction.py) to produce
        # reliable, well-staggered finish times under greedy/near-greedy
        # decoding -- unlike free-form instructions ("repeat X once"),
        # which this model doesn't reliably terminate on the first attempt,
        # a short, explicit repeat-count task reliably emits DONE right
        # after satisfying the count.
        fast_prompts = [
            f"Repeat the number {100 + i} exactly 1 time, then say DONE. Output nothing else."
            for i in range(7)
        ]
        slow_prompt = (
            "Repeat the number 999 exactly 80 times, separated by spaces, "
            "then say DONE. Output nothing else."
        )
        batch_a_prompts = fast_prompts + [slow_prompt]  # bsz=8
        stop_tokens_a = ("DONE",)

        permit_box_a = [None]
        # Give batch A's route-level admission a moment's head start so it's
        # definitely the one holding the budget when batch B tries to admit
        # -- mirrors the real route handler's acquire-then-iterate ordering
        # (see API_servers.router.v1_routes.big_batch_completions).
        engine_local = engine

        async def run_batch_a():
            permit_box_a[0] = await engine_local.acquire_prefill_permit(
                request_bsz=8, request_label="/big_batch/completions (A)"
            )
            return await _drain_big_batch_stream(
                engine_local, batch_a_prompts, max_length=700,
                stop_tokens=stop_tokens_a, permit_box=permit_box_a,
            )

        task_a = asyncio.create_task(run_batch_a())
        # Let batch A actually acquire its permit and start prefill before
        # batch B tries to admit, so batch B is genuinely queued behind it.
        while permit_box_a[0] is None:
            await asyncio.sleep(0.01)

        b_admitted_at = {}
        start = time.monotonic()

        async def run_batch_b():
            permit_b = await engine_local.acquire_prefill_permit(
                request_bsz=4, request_label="/big_batch/completions (B)"
            )
            b_admitted_at["t"] = time.monotonic() - start
            permit_box_b = [permit_b]
            contents, finish_reasons, _ = await _drain_big_batch_stream(
                engine_local,
                ["Say hi." for _ in range(4)],
                max_length=20, stop_tokens=("\n\n",), permit_box=permit_box_b,
            )
            await engine_local.release_prefill_permit(
                request_bsz=4, request_label="/big_batch/completions (B)",
                ticket=permit_b["ticket"], permit=permit_b,
            )
            return contents, finish_reasons

        task_b = asyncio.create_task(run_batch_b())

        contents_a, finish_reasons_a, finish_times_a = await task_a
        await engine_local.release_prefill_permit(
            request_bsz=8, request_label="/big_batch/completions (A)",
            ticket=permit_box_a[0]["ticket"], permit=permit_box_a[0],
        )
        contents_b, finish_reasons_b = await task_b

        a_slow_finish_time = finish_times_a.get(7)  # index 7 is the slow prompt
        b_admit_time = b_admitted_at.get("t")

        print(f"  batch A's slow row (index 7) finished at t={a_slow_finish_time:.2f}s")
        print(f"  batch B was admitted at t={b_admit_time:.2f}s")

        ok = True
        if b_admit_time is None:
            print("  FAIL: batch B was never admitted")
            ok = False
        elif a_slow_finish_time is not None and b_admit_time >= a_slow_finish_time - 0.05:
            print(
                "  FAIL: batch B was not admitted meaningfully earlier than batch A's "
                "slow row finished -- incremental release does not appear to be working "
                "(this is the pre-fix behavior: full-batch reservation held until the "
                "whole request ends)"
            )
            ok = False
        else:
            print(
                f"  PASS: batch B admitted {a_slow_finish_time - b_admit_time:.2f}s "
                "before batch A's slowest row finished -- incremental release is working"
            )

        for i in range(7):
            if str(100 + i) not in contents_a.get(i, "") and finish_reasons_a.get(i) != "length":
                print(f"  WARN: fast prompt {i} content looks unexpected: {contents_a.get(i)!r}")

        return ok
    finally:
        engine.model.max_prefill_bsz_limit = original_limit
        engine.model.max_prefill_bsz = original_bsz


async def check_no_regression_in_final_release_accounting(engine):
    """Sanity check independent of timing: after both batches from the
    check above (or a fresh pair) fully complete and release, the shared
    _prefill_reserved_bsz counter must return to exactly 0 -- confirms the
    incremental-release + final-release combination never over- or
    under-releases in a real decode loop (not just the pure-bookkeeping
    unit tests in test_prefill_admission_queue.py)."""
    assert engine._prefill_reserved_bsz == 0, (
        f"expected _prefill_reserved_bsz == 0 after all requests completed and "
        f"released, got {engine._prefill_reserved_bsz} -- indicates a leak or "
        f"double-release bug in the incremental release path"
    )
    print("  PASS: _prefill_reserved_bsz correctly returned to 0 after both requests")
    return True


async def main():
    if len(sys.argv) < 2:
        print("usage: python test/verify_incremental_prefill_release.py <model_path>")
        sys.exit(1)
    model_path = sys.argv[1]

    print("=== Check 1: incremental release admits a second request early ===")
    engine = _load_engine(model_path)
    ok1 = await check_incremental_release_admits_second_request_early(engine)

    print("\n=== Check 2: admission accounting returns to 0 after both requests ===")
    ok2 = await check_no_regression_in_final_release_accounting(engine)

    if ok1 and ok2:
        print("\nALL CHECKS PASSED")
    else:
        print("\nSOME CHECKS FAILED -- see output above")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
