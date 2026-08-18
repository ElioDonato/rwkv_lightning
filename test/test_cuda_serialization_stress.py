"""
Deterministic stress / property tests for the PROCESS-WIDE CUDA serialization
primitive added in Phase 2: `infer.async_forward.cuda_guard` (a module-level
`threading.RLock`) and `infer.async_forward.GpuAsyncExecutor`.

What this validates, with NO GPU and NO torch/CUDA toolchain (runs in a plain
venv; only stdlib threading/asyncio + the fake CPU callable + pytest):
  1. Mutual exclusion -- a shared active-depth counter tracked across many
     concurrent plain threads AND the async worker never observes depth > 1.
  2. Both modes -- GpuAsyncExecutor(enabled=True) (the Mod A / worker path) and
     enabled=False (inline on the calling thread) each maintain the invariant
     and return correct results (the fake unit returns its input).
  3. Cross-mode exclusion -- while the worker is mid-a-guarded-unit on its own
     thread, a plain thread acquiring cuda_guard() on another thread cannot
     enter until the worker's unit releases the guard.
  4. No deadlock -- every simulated unit completes under hard asyncio/threadjoin
     timeouts; a stuck run raises instead of hanging.
  5. Cross-thread non-reentrancy -- an *RLock* still enforces strict mutual
     exclusion across threads (reentrancy is only within the SAME thread).

Thread-counting idiom: a set of "currently inside the fake CUDA unit" thread
ids, mutated with GIL-atomic set.add/discard at the guarded boundaries and read
inline after the add. Because the ONLY serialization between the participants is
`cuda_guard()` itself (no separate measurement lock), observing len==2 would be
direct evidence of a guard failure.

Run with:  uv run pytest test/test_cuda_serialization_stress.py -v
(fresh venv is fine -- no torch/CUDA required.)
"""
import asyncio
import sys
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from infer.async_forward import cuda_guard, GpuAsyncExecutor
except Exception as exc:  # pragma: no cover - import env issues
    pytest.skip(f"infer.async_forward not importable: {exc}", allow_module_level=True)

_NUM_THREADS = 8
_ITERS = 50
_UNIT_SLEEP_S = 0.0002  # hold the lock a moment so contention can actually collide


class _SharedDepth:
    """Tracks the set of threads currently inside the fake CUDA unit.

    add/discard are GIL-atomic; the read of len() happens immediately after add
    while the unit still holds the guard, so a correct guard always sees len==1
    and a broken one can surface len>=2.  No separate measurement lock is used --
    the guard under test is the ONLY thing serializing the participants.
    """

    def __init__(self):
        self.inside = set()
        self.max_depth = 0


def _make_pseudo_unit(depth: _SharedDepth, sleep_s: float = _UNIT_SLEEP_S):
    """A fake GPU-guarded unit: guards its body with cuda_guard(), tracks depth,
    simulates a CUDA op by sleeping, and returns its input unchanged (so callers
    can assert the worker/threads return correct results)."""

    def pseudo_gpu(x):
        tid = threading.get_ident()
        with cuda_guard():
            depth.inside.add(tid)
            d = len(depth.inside)
            if d > depth.max_depth:
                depth.max_depth = d
            time.sleep(sleep_s)
            depth.inside.discard(tid)
        return x

    return pseudo_gpu


def _run_stress(enabled: bool, num_threads=_NUM_THREADS, iters=_ITERS,
                deadline_s: float = 10.0):
    """Run `num_threads` plain threads + one GpuAsyncExecutor async worker, all
    driving the same pseudo GPU unit concurrently, kicked off from a barrier to
    maximize contention.  Asserts the mutual-exclusion invariant and result
    correctness; raises if any participant fails to finish within the deadline.

    `enabled` selects the executor mode: True -> work lands on the single Mod-A
    worker thread; False -> offload runs inline on the event-loop thread.
    Returns (max_observed_depth, wall_seconds)."""
    depth = _SharedDepth()
    pseudo = _make_pseudo_unit(depth)
    executor = GpuAsyncExecutor(enabled=enabled)
    barrier = threading.Barrier(num_threads + 1)

    thread_results = []
    async_results = []
    thread_errors = []

    def _thread_work(tid):
        try:
            barrier.wait(timeout=10)
            local = [pseudo(tid * 100_000 + i) for i in range(iters)]
            thread_results.extend(local)
        except Exception as exc:  # pragma: no cover
            thread_errors.append(exc)

    async def _async_work():
        await asyncio.to_thread(barrier.wait, timeout=10)
        for i in range(iters):
            async_results.append(await executor.offload(pseudo, 900_000_000 + i))
        if enabled:
            executor.shutdown(wait=True)

    threads = [threading.Thread(target=_thread_work, args=(t,), daemon=True)
               for t in range(num_threads)]
    started = time.monotonic()

    for t in threads:
        t.start()
    asyncio.run(asyncio.wait_for(_async_work(), timeout=deadline_s))
    for t in threads:
        t.join(timeout=deadline_s)

    wall = time.monotonic() - started
    if enabled and executor.enabled:  # pragma: no cover - defensive
        executor.shutdown(wait=True)

    alive = [t for t in threads if t.is_alive()]
    assert not alive, f"{len(alive)} thread(s) did not finish within {deadline_s:.1f}s (deadlock?)"
    assert not thread_errors, f"worker thread(s) raised: {thread_errors}"
    assert depth.max_depth == 1, f"concurrent CUDA depth {depth.max_depth} != 1 (guard broken)"
    assert len(async_results) == iters
    assert all(r == (900_000_000 + i) for i, r in enumerate(async_results)), \
        "enabled worker must return each fake unit's input"
    assert len(thread_results) == num_threads * iters
    for tid, i in ((t, i) for t in range(num_threads) for i in range(iters)):
        assert (tid * 100_000 + i) in thread_results, f"missing thread result {tid},{i}"
    return depth.max_depth, wall


# ---------------------------------------------------------------------------
# 1. Mutual exclusion
# ---------------------------------------------------------------------------


def test_mutual_exclusion_worker_and_threads_never_exceed_depth_one():
    """Many plain threads + the async worker all stomp on cuda_guard() from a
    barrier; the observed in-CUDA depth must never exceed 1."""
    depth_max, _ = _run_stress(enabled=True)
    assert depth_max == 1


# ---------------------------------------------------------------------------
# 2. Both modes maintain the invariant + return correct results
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("enabled", [True, False], ids=["worker-offload", "inline"])
def test_both_modes_keep_invariant_and_return_results(enabled):
    """Enabled (worker thread) AND disabled (inline on caller) executor each hold
    the mutual-exclusion invariant across all threads and return correct results."""
    depth_max, _ = _run_stress(enabled=enabled, iters=30)
    assert depth_max == 1


# ---------------------------------------------------------------------------
# 3. Cross-mode exclusion: a thread is locked out while the worker holds it
# ---------------------------------------------------------------------------


def test_thread_is_blocked_while_worker_holds_a_guarded_unit():
    """While the Mod-A worker is mid-unit on its own thread, a plain thread
    acquiring cuda_guard() must NOT enter until the worker releases."""
    worker_entered = threading.Event()
    worker_released = threading.Event()
    observ = {"inside": set(), "max": 0, "plain_entry_at": None, "release_at": None}

    def worker_unit():
        with cuda_guard():
            observ["inside"].add("worker")
            observ["max"] = max(observ["max"], len(observ["inside"]))
            worker_entered.set()
            time.sleep(0.4)  # hold the guard long enough for the thread to try
            observ["inside"].discard("worker")
        worker_released.set()
        observ["release_at"] = time.monotonic()
        return "worker-done"

    def plain_thread_fn():
        worker_entered.wait(timeout=5)  # guarantee the worker is inside first
        with cuda_guard():
            # only we may be inside now (worker must have exited first)
            observ["max"] = max(observ["max"], len(observ["inside"]))
            observ["plain_entry_at"] = time.monotonic()
            time.sleep(0.01)
        return True

    executor = GpuAsyncExecutor(enabled=True)
    runs = {"plain": False}

    async def _main():
        task = asyncio.ensure_future(executor.offload(worker_unit))
        t2 = threading.Thread(target=lambda: runs.__setitem__("plain", plain_thread_fn()))
        t2.start()
        worker_result = await asyncio.wait_for(task, timeout=5)
        t2.join(timeout=5)
        assert not t2.is_alive(), "plain thread stuck while worker holds the guard"
        executor.shutdown(wait=True)
        return worker_result

    result = asyncio.run(_main())
    assert result == "worker-done"
    assert runs["plain"] is True
    assert observ["max"] == 1, "worker + thread were inside CUDA at once"
    assert observ["plain_entry_at"] is not None, "plain thread never progressed"
    assert observ["release_at"] is not None
    assert observ["plain_entry_at"] >= observ["release_at"], \
        "plain thread entered while the worker still held the guard"


# ---------------------------------------------------------------------------
# 4. No deadlock: full stress must complete under hard deadlines
# ---------------------------------------------------------------------------


def test_no_deadlock_full_stress_completes_within_budget():
    """The full joint stress (8 threads + async worker, enabled mode) must finish
    well under the 5s budget -- a stuck run hits the asyncio deadline + thread
    join timeouts and raises instead of hanging."""
    start = time.monotonic()
    _run_stress(enabled=True, num_threads=_NUM_THREADS, iters=_ITERS, deadline_s=5.0)
    wall = time.monotonic() - start
    assert wall < 5.0, f"stress took {wall:.2f}s; expected < 5s"


# ---------------------------------------------------------------------------
# 5. Cross-thread (non-)reentrancy: RLock does NOT reenter across threads
# ---------------------------------------------------------------------------


def test_guard_is_not_reentrant_across_threads():
    """An RLock reenters only within the SAME thread; a second thread acquiring
    the guard while the holder is inside must block and not enter until release."""
    holder_inside = threading.Event()
    release_holder = threading.Event()
    observ = {"inside": set(), "holder_tid": None}

    def holder():
        with cuda_guard():
            observ["holder_tid"] = threading.get_ident()
            observ["inside"].add("holder")
            holder_inside.set()
            release_holder.wait(timeout=5)  # hold the guard, do not release
            observ["inside"].discard("holder")

    def second():
        holder_inside.wait(timeout=5)
        with cuda_guard():
            # If the guard were reentrant across threads, "holder" would still be
            # in the set here.  Correct behavior: holder has exited, so only we
            # are inside and the set holds no "holder" marker.
            assert "holder" not in observ["inside"], \
                "holder was still inside when a second thread entered (cross-thread reentrancy!)"

    t1 = threading.Thread(target=holder)
    t2 = threading.Thread(target=second)
    t1.start()
    t2.start()

    # While t1 holds the guard, t2 must NOT have been able to enter its body:
    # the observed inside-set must be exactly {holder}.
    holder_inside.wait(timeout=5)
    time.sleep(0.2)  # give t2 a window to (erroneously) enter if the guard is broken
    assert observ["inside"] == {"holder"}, \
        f"second thread entered while holder held the guard: {observ['inside']}"

    release_holder.set()
    t1.join(timeout=5)
    t2.join(timeout=5)
    assert not t1.is_alive() and not t2.is_alive(), "threads stuck on the guard (deadlock)"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))