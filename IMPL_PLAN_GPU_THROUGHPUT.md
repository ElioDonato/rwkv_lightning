# GPU Utilization & Cleanup — Safe Implementation Plan (2026-08-18, rev 2)

Goal: make the server *general* (good for one client AND many concurrent clients) by raising
GPU utilization / max throughput, and cut the over-engineering and duplication — **safely**,
step by step, with a hard verification gate between each step.

> **Rev 2 changes (from two independent agent reviews):** (1) Phase 0 must COMMIT the current
> dirty tree first, or rollback is meaningless (settings.py is untracked). (2) Phase 1B is NOT
> a pure refactor — `max_prefill_bsz_limit` (frozen) and `max_prefill_bsz` (live) are not
> interchangeable, and `_embed_texts_bounded` is a documented invariant, so I narrowed it.
> (3) `sampler_simple` must NOT be deleted as planned (breaks `benchmark.py` + a test) —
> excluded. (4) Phase 2 must also serialize the embed path (real CUDA 2-thread gap). (5) Phase 4
> premise corrected: `big_batch` does NOT have per-row sampling (batch-wide gumbel-temp), so
> "any-mix" batching needs a NEW per-row sampler. (6) Phase 5 given concrete capture mechanics
> + per-size graph pool (else solo latency regresses). (7) numeric, non-gameable measurement
> gates. (8) several Phase-3 fixes moved in from Phase 5 (the 1-line CUDA stream fix + graph
> smoke) to retire the biggest risk before Phase 4's spend.

## Constraints & ground rules
- **No push to the remote fork.** Everything stays local; we commit locally as checkpoints.
  User explicitly said do not merge yet.
- **Default-off is sacred.** Every opt-in (`RWKV_ASYNC_FORWARD`, `RWKV_EMBED_AGGREGATE`,
  `RWKV_FUSE_CHAT_BATCH`) stays default-OFF; with no env set the server stays behavior-identical.
- **no_grad, never inference_mode.** Every closure that crosses the async/thread boundary (and
  anything a CUDA graph later captures) MUST keep `torch.no_grad()` — `inference_mode` is
  thread-local, not coroutine-local, and would desync interleaved coroutines. This is enforced
  in Phases 2/3/4/5.
- **Evidence over assumption.** The 9–19% occupancy and kernel claims are code-inference.
  Phases 3/4/5 are gated on a *measured* improvement (numeric recipe below). No win → stop.
- **One commit per phase**, created by the coordinator only (agents edit, don't commit).
  No push.
- **Downtime OK** (user away for hours): real-GPU smoke runs are allowed between phases.
  Real-GPU smoke IS possible on this box (RTX 3090 Ti, `env.sh` provides CUDA_HOME;
  `TORCH_CUDA_ARCH_LIST=8.6`). A CUDA `.cu` change requires a **rebuild** — Phase 5 rollback
  means recompiling, not just `git checkout`.

## Verification gate (end of EVERY phase)
1. `python3 -m py_compile` every touched `.py`.
2. Targeted tests for touched modules. **These also exercise the opt-in paths** (test files
   construct aggregators with `enabled=True`) — keep that dependency explicit so opt-in logic is
   validated, not just the default path.
3. Full suite: `.venv/bin/python -m pytest test/ -q -p no:cacheprovider --continue-on-collection-errors`.
   Any NEW failure vs the Phase 0 baseline → fix or roll back (below) before moving on.
4. Runtime/perf phases: the numeric recipe (below) on a real GPU.
5. `git commit` a checkpoint (coordinator); **no push**.

## Rollback rule
- Phase 0 establishes a clean baseline commit `p0-baseline` (commits the entire current dirty
  tree incl. untracked `settings.py`, with `state_tune/data` datasets gitignored so they are NOT
  checkpoint state). After that, every phase starts from a clean commit.
- To roll back phase N: `git checkout <phase-(N-1)-commit> -- <paths-phase-N-touched>` (NEVER
  blanket reset --hard, so unrelated in-flight edits survive), then re-run the gate. For a CUDA
  kernel change, rollback = revert the `.cu` + rebuild.

## Numeric measurement recipe (used by Phases 3/4/5)
- Generator: slim `httpx`/`aiohttp` loop hitting `/openai/v1/chat/completions` `stream=true`,
  fixed 96-token prompt, `max_tokens=256`, at N ∈ {1,2,4,8,16} concurrent clients.
- Metrics: `p50 TTFT` + `p50/p95 per-token ms` (solo sensitivity, N=1); **aggregate tokens/s
  across clients at each N** (throughput scaling); `GPU-active%` via `nvidia-smi
  --query-gpu=utilization.gpu -lms 200` (corroboration only, never the gate).
- **The gate:** (i) `tokens/s@N=8 ≥ 1.6 × tokens/s@N=1`, AND (ii) solo `p50 per-token` & `TTFT`
  @N=1 within `+15%` of baseline, AND (iii) `GPU-active%@N=8 ≥ 1.5 × baseline`. Pair (i)+(ii) is
  the "general + both" test; neither alone counts.
- A/B: toggle only the target opt-in's env var between runs of the same process config;
  discard first response (warmup), interleave A/B/A/B ×3, take medians; record power/temp to
  catch thermal drift. Take one `py-spy` capture under N=8 to validate the Phase-3 CPU-hotspot
  claims live.

---

## Phase 0 — Baseline & rollback foundation
- Record full-suite baseline; confirm GPU smoke path works (`env.sh` → CUDA_HOME, kernel builds
  for arch 8.6); report remote state (see below).
- **Commit the whole current working tree as `p0-baseline`** (incl. untracked `settings.py`);
  add `state_tune/data/**` to `.gitignore` first. This alone makes every later rollback
  well-defined.
- Remote fact: 18 commits ahead of `fork/main` (ElioDonato), 77 ahead of `origin/main`
  (upstream). No drift.
- **Go/no-go:** baseline recorded + `p0-baseline` committed. No code change beyond the commit.

## Phase 1 — Safe de-duplication & cleanup (behavior-preserving; PARALLEL tracks)
Disjoint files per track; working-tree edits only; full-gate then ONE coordinator commit `p1-cleanup`.
**No behavior change when env unset.**

- **1A · fuse_aggregator** — factor the three (near-)identical `_InlineFuseStream(...)`
  construction blocks into one helper; delete dead `_max_permit_bsz` (0 callers). All three
  blocks pass identical params, so dedup is byte-identical. Files: `infer/fuse_aggregator.py`.
  Test: `test/test_fuse_aggregator.py`.
- **1B · embed internal dedup (narrowed)** — fold ONLY the *internal* duplicate reads in
  `embed_aggregator.py`: `_cap()` and `_no_cap_mode()` both resolve the **live**
  `model.max_prefill_bsz`; collapse to one private helper. **Do NOT** consolidate the frozen
  `max_prefill_bsz_limit` vs live `max_prefill_bsz` (fuse vs embed use different sources — keep
  both, document the asymmetry). **Do NOT** remove `_embed_texts_bounded` (documented
  defense-in-depth invariant). Files: `infer/embed_aggregator.py` only.
  Test: `test/test_embed_aggregator.py`, `test/test_embedding_batch.py`.
- **1C · dead symbols + default single-sourcing** — remove `_init_cuda_graph_state`
  (mirror its static-buffer+CUDAGraph skeleton into a Phase-5 stub comment/ticket, do NOT just
  keep the 1-liner), `_cleanup_cuda_state`, `_torch_top_k_top_p`, unused
  `generated_tokens`/`token_buffers` in `big_batch_stream`. Single-source the sampler defaults
  as **per-route constants preserving current values** (V1 top_k=50/top_p=0.6/alpha_decay=0.996,
  V2 top_k=500/top_p=0.5/alpha_decay=0.99, fuse gate top_k=20/top_p=0.6) — top_p AND alpha_decay
  too, not just top_k, so no later change silently breaks V2-vs-V1 divergence. **Excluded:**
  `sampler_simple` (breaks `benchmark.py`+`test_local_state_and_batch.py`), `inference_deps`
  indirection (too wide/low-value now). Files: `infer/inference_utils.py`, `infer/big_batch.py`,
  `infer/batch_inference.py`, route/schema defaults (values unchanged).
  Tests: affected modules.
- **Gate:** full-suite baseline; targeted tests (opt-in paths); commit `p1-cleanup`.

## Phase 2 — CUDA serialization unification (make "single CUDA thread" process-wide)
- Today the blocking cores (`_batch_decode_blocking`, `batch_generate_state`) and the **embed
  path** each can issue CUDA outside the Mod-A seam → under `RWKV_ASYNC_FORWARD=1`, two threads
  may touch CUDA simultaneously. Goal: every CUDA-touching unit (streaming decodes, blocking
  cores, embed path, sampler-init, cleanup) serializes through ONE primitive. This is what makes
  Mod A safe to enable.
- **Files:** `infer/inference.py`, `infer/batch_inference.py`, `API_servers/router/common.py`,
  **`infer/embedding.py`, `infer/embed_aggregator.py`** (cover the embed gap too).
- **Rule:** keep `no_grad()` in every closure crossing the thread boundary (no `inference_mode`).
- **Gate:** tests + threaded stress (streaming + blocking + `/embedding` concurrently under
  `RWKV_ASYNC_FORWARD=1`) asserting **plausible complete output** AND **completion within a
  timeout** (deadlock catch), not just "no CUDA error"; then enable-async smoke. Commit `p2`.

## Phase 3 — Per-token overhead diet + retire the biggest risk early
- **Real win:** restructure `tokenizer.decode` off the per-token hot path — `_ingest_token_with_stop`
  re-decodes the whole sliding window every token (GPU-idle critical path for every decode loop).
- **Consistency tidy:** merge `batch_infer_stream_state`'s two offload units/token into the
  single `_gpu_decode_step` pattern (state endpoint; bytes-identical output with fixed seed).
- **Demote `await asyncio.sleep(0)`:** remove ONLY where a mandatory per-token await already
  yields; under default-off `_offload_gpu` is synchronous so `sleep(0)` may be the only yield —
  blind removal can stall SSE. Gate with a stress check.
- **Remove `gc.collect()`** from the blocking cores (threadpool `gc.collect()` can stall the
  in-flight streaming decode; refs aren't in cycles → pure overhead).
- **NEW in this phase (from old Phase 5): the 1-line CUDA seq-stream fix** —
  `cuda/rwkv7_state_fwd_fp16.cu:315` seq/batch kernel launch omits the stream; add
  `at::cuda::getCurrentCUDAStream()` (the `one` kernel at `:321` passes it). Add a small
  seq-vs-one kernel test proving they agree, THEN a **graph-vs-eager smoke** (the in-code NOTE
  already shows a `forward_one` capture passes) to confirm the launch-tax hypothesis that Phase 4
  depends on. Kernel change → rebuild.
- **Gate:** full suite; real-GPU numeric recipe before/after (returning the baseline we need for
  Phase 4); byte-identity checks use an **injected/fixed sampling seed** (per-request seeds are
  `random.randint(0,2**63-1)`, so unsolid identity is not reproducible otherwise). Commit `p3`.

## Phase 4 — Dynamic decode batching (the "general + both" occupancy win) [HIGH RISK — CONFIRM]
- **Corrected premise:** `big_batch_stream` uses ONE batch-wide `temperature`/`max_length`/
  `stop_tokens` via `sampler_gumbel_batch` (gumbel temp-only); `_V1/_V2` samplers are also
  batch-wide (only penalty counts are per-row). So "any-request-mix, per-row sampling" is NOT an
  extension of `big_batch` — it requires a **new per-row sampler** (per-row temperature minimum;
  decide top_k/top_p/penalty policy). Without it, throughput scales only for the homogeneous
  all-default subset (the homogeneity gate's ceiling persists) — the plan's central promise
  silently won't happen.
- **Compute-vs-launch model:** batching scales *compute*; CUDA graphs remove *launches*; they
  are independent and stack. On the 3090 Ti decode stays launch-bound until ~B≈16; B=8 lifts
  occupancy to maybe ~40–60%, not "full". Say this explicitly so an occupancy-only Phase 4 gate
  doesn't read as failure.
- **Goal:** a running bsz=N decode that rows join/leave as requests arrive/finish (extend
  `big_batch_stream`'s existing row-compaction + incremental release), with the new per-row
  sampler, so any request mix keeps the GPU batch full.
- **Files:** `infer/big_batch.py`, `infer/batch_inference.py`, `infer/inference.py`,
  `API_servers/router/*`. NOT parallel with Phase 5 (shared core).
- **Gate:** full suite; the numeric recipe (i)(ii)(iii); solo parity is mandatory. If no
  measured gain → STOP and reassess. **Pause for user confirmation before starting.** Commit `p4`.

## Phase 5 — CUDA-graph capture of the decode step [DEPENDS ON PHASE 4 — CONFIRM]
- **Mechanics (required, else solo latency regresses):** capture **per-block-size graphs**
  B ∈ {1,2,4,…,N} and replay the smallest one covering the live prefix — a single max-B graph
  would run all B rows (grid B·H) every token and break the solo-latency gate. Split
  `.tolist()`/GPU→CPU reads OUT of the captured region (host sync inside capture is illegal);
  use **static input/output buffers** overwritten each step; sampler RNG (`setup_rand`,
  `sample_rand_states`) must be static/captured, not reallocated per token; `forward_batch`
  mutates recurrent `state` in place → same static addresses each step. Capture under `no_grad`.
  Revive the `_init_cuda_graph_state` skeleton (removed in Phase 1C).
- **Padding caveat:** WKV grid = B·H, so padding rows still cost compute — you always pay for B
  rows in a max-B replay; hence the per-size pool.
- **Gate:** graph-vs-eager byte-identical (fixed seed) + numeric recipe gain (occupancy and
  latency); else rollback (which here = revert `.cu` + rebuild). **Pause for user confirmation.**
  Commit `p5`.

---

## Parallelism summary
- **Phase 1 tracks (1A/1B/1C):** parallel, disjoint files; ONE coordinator commit at the gate.
- **Phases 1→2→3→4→5:** strictly sequential, each gated; 2 & 3 overlap `batch_inference.py`, and
  4 & 5 share the decode core — no parallel editing of shared core.
- Every phase committed locally (never pushed), green-verified, reversible via the rollback rule.

## Decisions deferred to the user (not done silently)
- Unifying sampler *values* (V1 50 / V2 500 / fuse 20 → one number) — behavior change, deferred.
- Whether opt-ins auto-enable on GPU after Phase 4/5 validation, or stay behind flags.
- Endpoint-path / model-name "configurability" — public API contract, not changed.
- Homogeneity-vs-per-row-sampling tradeoff (Phase 4) — surfaced, not silently chosen.
- Cross-request prefill/decode pipelining (2-stream overlap) — considered, REJECTED as it
  conflicts with the single-CUDA-thread invariant Phase 2 adds. `torch.compile` — plausible but
  high-variance; bounded spike to assess once, not a phase.