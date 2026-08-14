"""
Measurement scaffold for Mod A: compare sync (default-off, on the event-loop
thread) vs async-forward (opt-in, on the single GPU-worker thread) wall time and
-- when run on the live GPU -- GPU utilization, to validate whether offloading
the heavy forward actually keeps the GPU more occupied.

By default this runs a CPU/mock FakeModel so it is safe to execute anywhere
(no model on the GPU, no contention with the :8081/:8083 live services). To
measure the real occupancy ceiling later you must explicitly pass --model-path
AND the RWKV_ASYNC_FORWARD/RWKV_* env from env.sh; that path loads a real model
onto the GPU and is NOT intended to run while the live services are up.

Usage:
    python test/benchmark_async_forward.py                 # fake model (safe)
    python test/benchmark_async_forward.py --model-path /path/to/model.pth \
        --cuda --wall-steps 200 --gpu-samples 5            # real GPU (careful)
"""
import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _nvidia_smi_util() -> float | None:
    """Poll nvidia-smi once for the <gpu-util> percentage of GPU 0 (or None if
    unavailable -- not a CUDA machine or nvidia-smi missing)."""
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return None
    try:
        out = subprocess.run(
            [exe, "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        values = [float(v) for v in out.stdout.split(",") if v.strip()]
        return values[0] if values else None
    except (subprocess.SubprocessError, ValueError):
        return None


def _sample_gpu_while(seconds: float, sample_interval: float = 0.2) -> list[float]:
    """Sample GPU util in a background thread for `seconds`, so the harness can
    report peak/mean occupancy during an async-forward benchmark run."""
    import threading

    done = threading.Event()
    samples: list[float] = []

    def _poll():
        while not done.is_set():
            u = _nvidia_smi_util()
            if u is not None:
                samples.append(u)
            time.sleep(sample_interval)

    t = threading.Thread(target=_poll, daemon=True)
    t.start()
    time.sleep(seconds)
    done.set()
    t.join(timeout=seconds + 1)
    return samples


def _summary(samples: list[float]) -> str:
    if not samples:
        return "n/a (no GPU / no nvidia-smi)"
    return f"util={{mean {sum(samples)/len(samples):.1f}%, peak {max(samples):.0f}%, n={len(samples)}}}"


def _make_fake_model(bsz: int, vocab: int) -> object:
    """CPU stand-in that mimics the GPU-shaped forward the real model would run
    (same call signature big_batch_stream / forward_batch decode step use), so
    the harness measures the offload mechanics, not real matmul speed."""
    class _Fake:
        prefill_chunk_size = 256

        def generate_zero_state(self, b):
            return [
                torch.zeros(1, 2, b),
                torch.zeros(1, b, 1, 1, 1),
                torch.zeros(b, dtype=torch.int32),
            ]

        def forward_batch(self, tokens, state, full_output=False):
            if isinstance(tokens, torch.Tensor):
                b = tokens.shape[0]
            else:
                b = len(tokens)
            # A touch of real CPU work so the benchmark is not entirely no-op.
            return torch.full((b, self.vocab), 0.5)

    model = _Fake()
    model.vocab = vocab
    return model


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-path", default=None, help="real .pth to load (GPU; careful)")
    ap.add_argument("--bsz", type=int, default=32, help="batch per forward")
    ap.add_argument("--vocab", type=int, default=65536, help="vocab size (fake model)")
    ap.add_argument("--wall-steps", type=int, default=100, help="forward calls per mode")
    ap.add_argument("--gpu-samples", type=float, default=3.0,
                    help="seconds to sample nvidia-smi util during async run (0=skip)")
    args = ap.parse_args()

    use_real = args.model_path is not None
    if use_real:
        # Heavy: loads the real model onto the GPU. Deliberately not auto-run.
        from infer.inference import InferenceEngine
        from model_load.model_loader import load_model_and_tokenizer
        print("[bench] loading real model (GPU) -- do NOT run while :8081/:8083 are up")
        model, tokenizer, margs, rocm = load_model_and_tokenizer(
            args.model_path, inference_engine="fp16"
        )
        engine = InferenceEngine(model=model, tokenizer=tokenizer, args=margs, rocm_flag=rocm)
        def _fwd():
            state = model.generate_zero_state(args.bsz)
            toks = torch.randint(0, model.args.vocab_size, (args.bsz, 1), device="cuda")
            return model.forward_batch(toks, state)
    else:
        # Safe CPU default: use the engine only for its executor seam.
        from infer.async_forward import GpuAsyncExecutor
        from infer.inference import InferenceEngine
        fake = _make_fake_model(args.bsz, args.vocab)
        engine = InferenceEngine(model=fake, tokenizer=None, args=None, rocm_flag=False)
        state = fake.generate_zero_state(args.bsz)
        toks = torch.randint(0, args.vocab, (args.bsz, 1))
        def _fwd():
            return fake.forward_batch(toks, state)

    # Warm-up (JIT / allocator warmup must not pollute the timed runs).
    _fwd()

    # -- Sync baseline: executor disabled, forward inline on the event loop. --
    engine._gpu_executor = GpuAsyncExecutor(enabled=False)
    sync_t0 = time.perf_counter()
    for _ in range(args.wall_steps):
        _fwd()
    sync_elapsed = time.perf_counter() - sync_t0

    # -- Async variant: forward handed to the single GPU worker thread. --
    engine._gpu_executor = GpuAsyncExecutor(enabled=True)
    import asyncio
    util_samples: list[float] = []
    async def _async_bench():
        nonlocal util_samples
        if use_real and args.gpu_samples > 0.0:
            util_samples = _sample_gpu_while(args.gpu_samples)
        t0 = time.perf_counter()
        for _ in range(args.wall_steps):
            await engine._gpu_executor.offload(_fwd)
        return time.perf_counter() - t0
    async_elapsed = asyncio.run(_async_bench())
    engine._gpu_executor.shutdown()

    speedup = sync_elapsed / async_elapsed if async_elapsed > 0 else float("nan")
    print("=" * 72)
    print(f"mode                wall(s)   speedup     GPU")
    print(f"sync  (event loop)  {sync_elapsed:8.3f}   {'1.00x':>7}   (baseline)")
    print(f"async (worker x1)   {async_elapsed:8.3f}   {speedup:5.2f}x   {_summary(util_samples)}")
    print("=" * 72)
    print("Note: with a CPU fake model this measures the offload mechanics only;")
    print("      real GPU occupancy gains need --model-path on the idle GPU.")


if __name__ == "__main__":
    main()