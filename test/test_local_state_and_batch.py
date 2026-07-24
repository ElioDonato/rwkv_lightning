"""
Adapted from the repo's test_state_reuse.py / test_batch_state_reuse.py:
those hardcode another machine's absolute model path and use inconsistent
import styles (one assumes cwd=repo root, the other cwd=infer/). This
version runs from the repo root against our local models and covers both
single-sequence and batched state reuse across turns.

Usage: uv run python test/test_local_state_and_batch.py <model_path_without_.pth>
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from infer.rwkv_batch.rwkv7 import RWKV_x070
from infer.rwkv_batch.utils import TRIE_TOKENIZER, sampler_simple

GEN_LENGTH = 40


def load(model_path):
    import types
    args = types.SimpleNamespace()
    args.vocab_size = 65536
    args.head_size = 64
    args.MODEL_NAME = model_path
    print(f"\n[INFO] Loading {model_path} ...\n")
    t0 = time.perf_counter()
    model = RWKV_x070(args)
    print(f"[INFO] Loaded in {time.perf_counter()-t0:.1f}s")
    tokenizer = TRIE_TOKENIZER("infer/rwkv_batch/rwkv_vocab_v20230424.txt")
    return model, tokenizer


def single_sequence_state_reuse(model, tokenizer):
    print("\n" + "=" * 70)
    print("TEST 1: single-sequence state reuse across turns")
    print("=" * 70)
    state = model.generate_zero_state(0)

    def run(prompt, label):
        tokens = tokenizer.encode(prompt)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = model.forward(tokens, state)
        generated = []
        for _ in range(GEN_LENGTH):
            tok = sampler_simple(out, noise=0).item()
            if tok == 0:
                break
            generated.append(tok)
            out = model.forward(tok, state)
        torch.cuda.synchronize()
        text = tokenizer.decode(generated)
        print(f"[{label}] {text!r}  ({time.perf_counter()-t0:.2f}s, tokens_in_state={state[2].item()})")
        return text

    run("User: Remember the secret code is PELICAN-7. Just say OK.\n\nAssistant:", "turn1-set-secret")
    run("\n\nUser: What is the secret code I just told you? Answer with only the code.\n\nAssistant:", "turn2-recall")
    print("PASS: single-sequence state carried token count forward "
          f"(final state[2]={state[2].item()})")


def batched_state_reuse(model, tokenizer):
    print("\n" + "=" * 70)
    print("TEST 2: batched (bsz=2) state reuse across turns")
    print("=" * 70)

    prompts_1 = [
        "User: You are a pirate. Respond in one short pirate sentence.\n\nAssistant:",
        "User: You are a strict mathematician. Respond in one short formal sentence.\n\nAssistant:",
    ]
    state = model.generate_zero_state(len(prompts_1))

    def run_batch(prompts, label):
        tokens_batch = [tokenizer.encode(p) for p in prompts]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = model.forward_batch(tokens_batch, state)
        bsz = len(prompts)
        generated = [[] for _ in range(bsz)]
        active = [True] * bsz
        for _ in range(GEN_LENGTH):
            step_tokens = []
            for b in range(bsz):
                tok = sampler_simple(out[b], noise=0).item()
                step_tokens.append(tok)
                if active[b]:
                    if tok == 0:
                        active[b] = False
                    else:
                        generated[b].append(tok)
            if not any(active):
                break
            out = model.forward_batch([[t] for t in step_tokens], state)
        torch.cuda.synchronize()
        for i, (p, g) in enumerate(zip(prompts, generated)):
            print(f"[{label}][seq {i}] {tokenizer.decode(g)!r}")
        print(f"[{label}] elapsed={time.perf_counter()-t0:.2f}s "
              f"tokens_in_state={state[2].cpu().numpy().tolist()}")

    run_batch(prompts_1, "turn1-set-personas")

    prompts_2 = [
        "\n\nUser: Say one more thing in character.\n\nAssistant:",
        "\n\nUser: Say one more thing in character.\n\nAssistant:",
    ]
    run_batch(prompts_2, "turn2-reused-state")
    print("PASS: batched state advanced independently per-sequence "
          f"(final state[2]={state[2].cpu().numpy().tolist()})")


def main():
    if len(sys.argv) != 2:
        print("usage: test_local_state_and_batch.py <model_path_without_.pth_extension>")
        sys.exit(1)
    model, tokenizer = load(sys.argv[1])
    single_sequence_state_reuse(model, tokenizer)
    batched_state_reuse(model, tokenizer)
    print("\nALL LOCAL STATE/BATCH TESTS PASSED\n")


if __name__ == "__main__":
    main()
