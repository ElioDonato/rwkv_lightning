# CLAUDE.md

Guidance for working in this repo (`RWKV-Vibe/rwkv_lightning`, cloned 2026-07-11 at
commit `dce31345`). This is a FastAPI batch-inference server for RWKV-7 models,
built on custom JIT-compiled CUDA/HIP kernels ("Albatross" kernels). It backs
the `lm.small` endpoint in `~/search/backend/config.yaml` (see "Relationship
to the search project" below).

## Environment setup (already done in this checkout)

Deps are managed with `uv` in `.venv` (Python 3.12). **Always run via
`uv run`, never bare `python`.**

The tricky part is CUDA: the server JIT-compiles two custom kernels
(`infer/rwkv_batch/cuda/rwkv7_state_fwd_fp16.cu`, `.../sampling.cu`) via
`torch.utils.cpp_extension.load()` at import time, which needs `nvcc` — this
machine has the NVIDIA driver but no system CUDA toolkit. Rather than
`sudo pacman -S cuda`, the toolkit is pip-installed straight into the venv
(`nvidia-cuda-nvcc`, `nvidia-cuda-runtime`, `nvidia-cuda-cccl`, `nvidia-curand`,
all pinned to matching `13.0.8x` builds — torch's own `cu130` wheel pulls in
mismatched newer versions transitively, which produces PTX the bundled
`ptxas` can't read; keep these pinned together if you bump torch).

System `gcc` (16.1.1) is newer than CUDA 13.0's `nvcc` supports (caps at
gcc 15), so `gcc15`/`gcc15-libs` are installed system-wide via pacman and
used as the host compiler.

Also: the pip `nvidia-cuda-runtime` package ships only the versioned
`libcudart.so.13`, not the unversioned `libcudart.so` the linker wants for
`-lcudart` — there's a symlink for it in
`.venv/lib/python3.12/site-packages/nvidia/cu13/lib/`.

**Before running anything**, `source env.sh` (repo root) — sets `CUDA_HOME`,
`PATH`, `TORCH_CUDA_ARCH_LIST=8.6` (RTX 3090 Ti / Ampere), and `CC`/`CXX`/
`CUDAHOSTCXX` to `gcc-15`/`g++-15`.

First run after any kernel-source change or `~/.cache/torch_extensions`
wipe takes longer (JIT compile); subsequent runs are `ninja: no work to do`
and start in seconds.

## Models

Not checked into the repo. Downloaded from `BlinkDL/rwkv7-g1` on
HuggingFace (public, no auth) into `models/`:
- `rwkv7-g1h-2.9b-20260710-ctx10240.pth` (5.9GB on disk, ~6GB VRAM loaded)
- `rwkv7-g1h-7.2b-20260710-ctx10240.pth` (14.4GB on disk, ~14.1GB VRAM loaded)

Both fit comfortably on the 3090 Ti's 24GB, with more headroom for batching
on the 2.9b (max_prefill_bsz≈269 for 2.9b vs ≈138-402 for 7.2b depending on
free VRAM at the time — it's computed dynamically in
`RWKV_x070.refresh_max_prefill_bsz()`).

Launch: `uv run python app.py --model-path models/<name-without-.pth> --port <p> --password <pw>`
(one model per server process — there's no hot-swap/multi-model routing).

## Bugs found while testing (both fixed 2026-07-12)

1. **`test/test_prefix_state_cache.py` was stale against `state_manager/state_pool.py`.**
   The test called `put_prefix_state()` with 64/128-token prefixes, but
   `PREFIX_CACHE_BUCKETS = (1024, 2048, ..., 8192)` (state_pool.py:19) and
   `put_prefix_state` rejects any length not exactly in that tuple
   (state_pool.py:420). Fixed by rewriting the test to use
   `PREFIX_CACHE_BUCKETS[:2]` (1024/2048) instead of hardcoded 64/128 —
   pure test bug, the bucket-only restriction in state_pool.py was correct
   as-is. Passes now.

2. **`POST /openai/v1/chat/completions` (non-streaming) intermittently hung
   indefinitely when `use_prefix_cache` was true** (the server-side default
   when the field is omitted — `openai_routes.py:159`). Not model-size
   dependent (reproduced on both 2.9b and 7.2b). Root-caused by restarting
   the live service with a temporary `SIGUSR1` handler that walked
   `asyncio.all_tasks()` / `cr_await` chains during a live hang (a plain
   `py-spy dump` only showed the main thread idle in `select()`, since a
   suspended-on-await Task isn't visible on any OS thread's stack — this
   live-introspection step was the missing piece last time this was
   investigated). The dump showed the request task stuck in
   `common.py:cleanup_disconnect_watcher`'s `await task`, while the
   `watch_disconnect` task it was awaiting was still alive and cycling
   normally through its poll loop — i.e. `task.cancel()` had already been
   called and simply didn't stop it. Cause: Starlette's
   `Request.is_disconnected()` wraps its receive call in an
   **already-cancelled** `anyio.CancelScope` (`cs.cancel()` then
   `await self._receive()`); if the external `task.cancel()` lands while
   the watcher is inside that scope, anyio can treat the injected
   `CancelledError` as satisfying its own scope's cancellation and swallow
   it, leaving the watcher looping forever instead of exiting — so the
   unbounded `await task` after `cancel()` blocks until the client
   genuinely disconnects (matching the old observation that it "only
   resolves when the client disconnects"). Separately, every
   `reserve_prefill_capacity` call site was also spinning up a *second*,
   redundant `watch_disconnect` task on the same request instead of
   reusing the `cancel_token` param the function already exposed for this
   — doubling the exposure to the same race. Fixed in
   `API_servers/router/common.py`: `cleanup_disconnect_watcher` now polls
   with bounded, repeated `cancel()` attempts instead of an unbounded
   `await task` (so a swallowed cancellation can never hang the response —
   an abandoned watcher is harmless, it exits on its own on real
   disconnect), and `openai_routes.py` / `v1_routes.py` / `v2_routers.py`
   / `state_routes.py` now all pass a shared `cancel_token` into
   `reserve_prefill_capacity` instead of creating a duplicate watcher.
   Verified against the live 2.9b service: 20/20 concurrent and 20/20
   sequential `use_prefix_cache:true` non-stream requests succeeded with
   no hangs (previously ~50% hung under the same repro), plus a full
   `test/test_curl.sh` pass.

## Testing

- `uv run pytest test/test_prefix_state_cache.py` — CPU-only, no model
  needed. Passes (see bug #1 above).
- `uv run pytest test/test_state_pool_l1_l2_cache.py` (added 2026-07-24) —
  CPU-only except one CUDA-skipped case, no model needed. Covers
  `state_manager/state_pool.py`'s in-memory L1 (VRAM)/L2 (RAM) session
  cache — `put_state`/`get_state` round-trip, LRU (not FIFO) eviction
  order (`move_to_end` on L1 hit matters), overwrite-of-existing-session
  not double-evicting, L2-hit promotion back to L1 (including the cascade
  eviction that can trigger), and L2-overflow persisting to disk. This was
  previously untested — only the prefix-cache's L2/disk paths had
  coverage via `test_prefix_state_cache.py`. Passes (5/5, run against
  this checkout 2026-07-24).
- `bash test/test_curl.sh` (env vars `BASE_URL`, `PASSWORD`,
  `STATE_MEMORY_ROUNDS`, etc. — see the script header) — full HTTP smoke
  suite: batch translate, `/v1/chat/completions` batch (stream + non-stream),
  FIM batch, session-based `/state/chat/completions` (verified actual
  cross-turn memory recall, not just non-crashing, by planting a code word
  and recalling it after distractor turns), `/multi_state`, `/state/status`,
  `/state/delete`, `/openai/v1/*`. Passes cleanly on both models, including
  the `use_prefix_cache:true` non-stream call that used to intermittently
  hang (bug #2 above, now fixed) — no need to pass `RUN_OPENAI=0` anymore.
- `test/test_state_reuse.py` / `test/test_batch_state_reuse.py` (shipped)
  hardcoded another machine's absolute model path
  (`/mnt/3f7ab3b2-.../rwkv7-g1b-1.5b-...`) and had inconsistent import
  assumptions (one expected cwd=repo root, the other cwd=`infer/`) — both
  confirmed to fail at runtime (`ModuleNotFoundError` and then
  `FileNotFoundError` on the hardcoded path once the import is worked
  around) on 2026-07-24. **Deleted** rather than "fixed", since
  `test/test_local_state_and_batch.py` (added 2026-07-11, kept) is a
  strict superset — same single-sequence + batched (bsz=2) state-reuse
  coverage, argv-driven model path, runs from the repo root. Verified
  passing against the local 2.9b model 2026-07-24 (`state[2]` token count
  advances correctly across turns for both the single-sequence and
  bsz=2 cases).
  Note its generated *text* isn't as clean as the HTTP endpoints' — it
  doesn't replicate the `<think>\n</think>` formatting cue or stop-token
  handling the production routes use, so don't read too much into
  turn-2 content quality from that script specifically; the HTTP-level
  `test_curl.sh` state-memory-recall round is the stronger proof of
  correctness.
- `uv run python test/verify_batch_v2_decode.py models/<model-dir>`
  (added 2026-07-24) — needs a real model + GPU. Covers
  `infer/batch_inference.py`'s V2 batch decode loop
  (`batch_generate_v2` and `batch_infer_stream_v2`), which previously had
  zero test coverage (distinct code path from `infer/big_batch.py`, which
  `test/verify_batch_compaction.py` already covers). Follows that script's
  misattribution-testing convention: 8 concurrent prompts each required to
  repeat a distinctive unique number, so any per-row indexing bug in the
  occurrence-penalty tensors / `active_mask` / per-index stop-state lists
  would show up as one prompt's output containing a *different* prompt's
  number — categorically distinguishable from ordinary sampling variation.
  Verified passing against the local 2.9b model 2026-07-24 (all 3 checks:
  blocking `batch_generate_v2`, streaming `batch_infer_stream_v2`, and a
  staggered-finish-length variant of the streaming case — zero
  cross-contamination in all 8×3 outputs).

Only one server process/model can run at a time on this GPU realistically
(2.9b + 7.2b weights alone would be ~20GB, no headroom left for state/batch
workspace) — stop one (`SIGTERM`, it persists all sessions to
`rwkv_sessions.db` on shutdown) before starting the other.

## Relationship to the search project (`~/search`)

`~/search/backend/config.yaml`'s `lm.small` currently points at a remote
endpoint (`10.0.0.10:8081`, likely another instance of this same server on
the "strix halo" box) with:
- `api_base: http://10.0.0.10:8081/openai/v1` — used by DSPy for
  **single, non-streaming** requests
- a `rwkv_inline` batching strategy (`backend/batching.py`) that POSTs
  batched requests to `http://10.0.0.10:8081/big_batch/completions` and
  parses an SSE stream of `{"choices":[{"index":..,"delta":{"content":..}}]}`
  chunks — this exactly matches this server's `/big_batch/completions`
  contract, confirming it's the same backend software.

  **Per-item `finish_reason` consumption (2026-07-25):** since commit
  `7e2e315` this server emits a per-item `finish_reason` (`"stop"`/
  `"length"`) in each item's terminal SSE chunk and compacts finished rows
  out of the GPU batch. `RWKVInlineBatcher._send_batch` was updated
  (`~/search` commit `0f756ec`) to resolve each request's future as soon as
  its own `finish_reason` chunk arrives — rather than buffering the whole
  stream to `[DONE]` — and to drop the stream once every slot is released.
  This closes the "zero perceived latency benefit" caveat noted in that
  commit's message: early-finishing requests are now handed back to callers
  immediately (measured ~25s earlier on a staggered short/long batch against
  the live 2.9b service). A `[DONE]`-time fallback still resolves any slot
  that never received a `finish_reason`, so older servers without the signal
  keep working.

**Nuance on bug #2 (now fixed, see above) and `lm.small`'s actual traffic:**
`batching.py::install_hooks()` monkey-patches `httpx.AsyncClient.send`
globally; while `create_batch_context(config)` is active (true for
`collect_data.py`, the working extraction pipeline), *any* outgoing call
to a `/chat/completions`-shaped path on `lm.small`'s host — including
whatever DSPy/litellm would normally send to `api_base`'s
`/openai/v1/chat/completions` — gets intercepted client-side and
redirected into `RWKVInlineBatcher`, which always POSTs to
`batch.endpoint` (`/big_batch/completions`), batch-of-1 or not. So code
running inside that context never actually touched the endpoint bug #2
lived on, regardless of which model is behind it — this was true even
before the fix, and remains true now. The only code that ever called
`lm.small` *outside* `create_batch_context` was `local_research_agent/agent.py`,
which `~/search/CLAUDE.md` already documents as broken/unwired for
unrelated import-path reasons; now that bug #2 is fixed, there's nothing
special to do if/when that path gets wired up.

As of 2026-07-11, `lm.small` in `~/search/backend/config.yaml` points at
`http://127.0.0.1:8081` (2.9b, chosen over 7.2b mainly for the extra VRAM
headroom for batching — bug #2 turned out not to be model-size-specific,
so it wasn't the deciding factor in the end), served by the `rwkv-lightning`
runit service (`/etc/runit/sv/rwkv-lightning`, `sv status rwkv-lightning`
/ `sv up|down rwkv-lightning`). The `run` script does
`exec chpst -u donato:donato service_run.sh` — deliberately not
`su -l donato -c '...'`, which forks multiple layers and can leave an
orphaned grandchild still holding the port if the tracked PID gets
SIGKILLed (this happened once during setup: a `kill -9` on the `su`-tracked
PID left the actual Python process alive on port 8081, and every
subsequent runit restart attempt then failed to bind and got killed and
respawned in a tight loop, reloading the model each time, until the
orphan was found and killed manually). With `chpst`, the tracked PID
*is* the Python process — a single exec chain, no orphan risk. Server
output goes to `~/rwkv_lightning/service.log`.

## Git workflow: fork `main` gets direct commits, no PRs

`origin` is upstream (`RWKV-Vibe/rwkv_lightning`); `fork` is this user's own
fork (`ElioDonato/rwkv_lightning`). **Deliberate choice, made 2026-07-13:**
work on this repo is committed straight to `main` and pushed to `fork`
(`git push fork main`) without going through a feature-branch → PR → merge
cycle. This is a personal fork, not a shared/reviewed upstream — the PR
machinery would add process overhead with no corresponding review benefit
here. Feature branches (e.g. `fix/openai-prefix-cache-hang`) may still get
used for in-progress work within a session, but they get fast-forwarded
into `main` and pushed directly rather than opened as PRs against either
remote. If this repo ever needs to contribute back upstream to
`RWKV-Vibe/rwkv_lightning`, that would be a real PR against `origin` — a
different, deliberate decision, not the default flow described here.

## `/big_batch/completions` now supports templated `chats` (2026-07-13)

See `INVESTIGATION_2026-07-13_small_lm_reliability.md` for the full
root-cause writeup. Summary: `/big_batch/completions` previously only
accepted `contents: list[str]` — raw, untemplated prompt strings, no chat
formatting applied. `~/search/backend/batching.py::RWKVInlineBatcher`
(the sole real caller) was sending bare `user`-message text through this
path with the `system` message silently dropped, which reliably produced
hallucinated/off-topic output from this instruction-tuned model (0/9 clean
in testing) — not a concurrency bug, confirmed via serial vs. concurrent
and raw vs. templated A/B tests.

Fix: `ChatRequest.chats: list[dict]` (`API_servers/router/schemas.py`) —
each item is an OpenAI-style `{"messages": [...], "system": ...}` dict.
`big_batch_completions` (`API_servers/router/v1_routes.py`) now builds each
prompt via `format_openai_prompt()` — the same function
`/openai/v1/chat/completions` already used — when `req.chats` is set,
falling back to the legacy raw `req.contents` path otherwise. This makes
this repo the single source of truth for the chat-template string; a
future finetune that changes the expected prompt format only requires
touching `format_openai_prompt()` here, not any client repo.
`~/search/backend/batching.py::_send_batch` was updated correspondingly to
send `chats` (full `messages`, system included) instead of an extracted
`contents` string.

## Second swarm review round (2026-07-24): dead code, webui auth, DoS fix, CUDA graph investigation

Six parallel workstreams, each independently controller-reviewed before
merge (see `/tmp/rwkv_swarm_reports/*.md` on this dev machine for the full
write-ups and independent re-verification transcripts — not part of the
repo, dev-machine-local only).

- **Dead code removed**: `app_big_batch.py` (superseded standalone entry
  point, zero references anywhere), `sample_logits`/`sampler_simple_batch`
  in `infer/rwkv_batch/utils.py`, and an unwired `self.model_lock`/
  `self.executor` in `infer/inference.py` (assigned, never used).
- **`webui_rwkv.py` hardening**: opt-in Gradio login via `RWKV_WEBUI_AUTH`
  (unset = prior no-auth behavior, unchanged) and a default-on backend-URL
  SSRF allowlist (`backend_url_error()`, extend via
  `RWKV_WEBUI_ALLOWED_HOSTS`, escape hatch `RWKV_WEBUI_ALLOW_ANY_BACKEND=1`).
  Adversarially re-tested (cloud metadata IP, userinfo/fragment tricks,
  IP-literal encodings) with no bypass found.
- **Real DoS fix**: `max_tokens` had no upper bound — a few concurrent
  huge-`max_tokens` requests could hold prefill-queue slots indefinitely,
  starving other clients. Fixed with `MAX_ALLOWED_TOKENS = 32768` on
  `ChatRequest` plus a `parse_request_model()` helper (`common.py`) that
  turns an uncaught `pydantic.ValidationError` into a clean 400 across
  `v1_routes.py`/`v2_routers.py`/`state_routes.py`
  (`openai_routes.py` already had equivalent handling). Also guards against
  a non-dict top-level JSON body (list/string), which independent review
  found still 500'd via an uncaught `TypeError` before that guard was added.
- **CUDA graph investigation** (comment-only change, no functional impact):
  `_init_cuda_graph_state` in `infer/inference_utils.py` is confirmed dead
  code, but a naive graph-capture repro found it would also be numerically
  **unsafe** to revive as-is: `forward_seq_batch_tensor`/`cuda_forward_seq`
  diverges under CUDA graph replay (root cause: its kernel launch omits an
  explicit CUDA stream, unlike `cuda_forward_one`/`cuda_spmv_forward`,
  which both correctly pass `getCurrentCUDAStream()`). The single-sequence
  `forward_one`/`cuda_forward_one` decode path, once correctly tested, is
  numerically fine under graph capture — an earlier draft of this
  investigation's finding that it also diverged was traced (by independent
  controller review) to a state-cloning bug in the test harness, not a real
  issue. See the code comment above `_init_cuda_graph_state` for the
  self-contained summary and suggested fix if anyone revives this later.
- **Test coverage**: added `test/test_state_pool_l1_l2_cache.py` and
  `test/verify_batch_v2_decode.py` (see Testing section above), deleted
  the two broken `test_*_state_reuse.py` files.
- **Docs**: removed stale `README.md.bak`, documented
  `/v2/chat/completions` and `/multi_state/chat/completions`.

All 6 opened as separate upstream PRs against `RWKV-Vibe/rwkv_lightning`
(#20-25) in addition to the direct `main`/`fork` commits described in the
git-workflow section above — upstream contribution is the one case where
this repo does go through real PRs (see that section's last line).

## Critical follow-up (2026-07-24): hardcoded constants -> configurable env vars

A later review of all 13 open upstream PRs found a real pattern worth
remembering: several "fixes" for genuine bugs shipped their remedy as an
unconditional hardcoded constant instead of an operator-configurable
default. The underlying bug/vulnerability was real in each case, but the
specific number picked is a deployment-specific policy choice, not a
universal constant, and this codebase already has a precedent
(`RWKV_WEBUI_*` env vars in `webui_rwkv.py`) for exposing tuning knobs this
way instead of baking them into source.

Fixed three instances, all keeping the previous value as the default (pure
additive, non-breaking):

- `MAX_ALLOWED_TOKENS` (`API_servers/router/schemas.py`, was a bare
  `32768`) -> `RWKV_MAX_ALLOWED_TOKENS` env var.
- `_prefill_bsz_refresh_interval_s` (`infer/rwkv_batch/rwkv7.py`, was a
  bare `2.0`) -> `RWKV_PREFILL_BSZ_REFRESH_INTERVAL_S` env var.
- `cleanup_disconnect_watcher`'s `timeout`/`poll_interval` defaults
  (`API_servers/router/common.py`, were bare `1.0`/`0.05`) ->
  `RWKV_DISCONNECT_WATCHER_CLEANUP_TIMEOUT_S` /
  `RWKV_DISCONNECT_WATCHER_CLEANUP_POLL_INTERVAL_S` env vars.

Documented in README.md's new "Server tuning env vars" section. Applied to
`main`/`fork` and propagated to the 6 affected upstream PR branches
(#13, #16, #18, #19, #23, #24) as follow-up commits, plus updated each of
those PRs' descriptions.

**Lesson for future work in this repo**: when a fix for a DoS/fairness/
performance issue involves picking a specific numeric threshold or
interval, default to making it an env-var-overridable module constant from
the start, following the `RWKV_WEBUI_*` / `RWKV_MAX_ALLOWED_TOKENS` /
`RWKV_PREFILL_BSZ_REFRESH_INTERVAL_S` naming convention (`RWKV_` prefix,
`_S` suffix for seconds), rather than a bare hardcoded value that later
needs a follow-up PR to fix.

The same review pass also found and removed several dangling references in
code comments and PR bodies to internal-only investigation files/process
names (a private report file path, an internal review-process worktree
path, generic "controller review"/"second reviewer session" phrasing) that
meant nothing to an outside reader of the public repo/PRs. None of these
were security-sensitive (no secrets, no private machine paths beyond a
`/tmp` report location and one stray Claude Code session URL in a PR body,
both removed), but they were dead references that should never have been
in code meant to be read by anyone outside this dev machine. Lesson: before
writing "see X for details" in a code comment or PR body, check that X is
actually reachable by whoever will read it.

## Incremental prefill-admission release (2026-07-25)

Follow-up to the `/big_batch/completions` finish_reason/compaction feature
(see the earlier entry above), which left "freeing a finished row's
capacity back into the prefill-queue's admission accounting" as explicit
future work. That gap is now closed: `infer/inference.py`'s
`acquire_prefill_permit` returns a permit carrying a mutable
`outstanding_bsz` box, and a new `release_prefill_capacity(amount, permit)`
lets `big_batch_stream`'s existing decode-time row compaction
(`infer/big_batch.py`) give back freed admission slots to the shared queue
*as each row finishes*, instead of only at the very end of the whole
batch request. A bsz=100 batch where 90 rows finish almost immediately no
longer blocks other queued requests from that freed capacity until the
last 10 slow rows also finish.

Wiring: `prefill_sse_response` (`API_servers/router/common.py`) gained an
`on_permit` callback so a route can hand the lazily-acquired permit into a
generator's closure (`permit_box`, a `[None]` list populated post-admission)
before the generator itself starts running. `release_prefill_permit` gained
an optional `permit=` param so the final end-of-request release only gives
back whatever's left outstanding, not the original full amount (avoiding
double-release if compaction already released some of it). Every route
except `/big_batch/completions` is unaffected — none of them pass
`permit_box`/`on_permit`, so their behavior is byte-identical to before.

Verified live: a bsz=8 batch (7 fast rows, 1 slow), admission-capped at 8,
admits a concurrent bsz=4 second batch ~0.6-1.1s before the first batch's
slow row finishes (reproducible across reruns, real 2.9b model on GPU) --
without this change the second batch would have to wait for the *entire*
first request to complete. Independently controller-reviewed: traced the
shared-lock concurrency safety (including the specific "late release call
after the permit's outstanding_bsz was already zeroed" race), an
independently-designed A/B (feature on vs off) reproduction confirming the
effect is real, a from-scratch adversarial test suite (concurrent
`asyncio.gather` races, negative amounts, malformed state), and 10
consecutive leak-check iterations with zero drift. Verdict: APPROVE.

Opened as PR #26 upstream, stacked on #19 (the compaction feature it
depends on) — same stacking pattern as #19 on #18.

**Done (2026-07-30)**: `/v2/chat/completions` now has both per-item
`finish_reason` and decode-time row compaction (commit `32c1206`). The V2
streaming path (`batch_infer_stream_v2`) uses `active_indices` remapping
(same pattern as `/big_batch/completions`): finished rows are removed from
state/logits/occurrence tensors via index_select, and `active_indices` maps
compacted positions back to original prompt indices for correct SSE `index`
fields. Non-streaming `batch_generate_v2` returns `(decoded_texts,
finish_reasons)` tuple. `/v1/chat/completions` also has per-item
`finish_reason` in both streaming and non-streaming paths (commit `c2ccda7`),
but V1 streaming does NOT have batch compaction yet (V1's sampler uses
`sample_rand_states` and `penalties` tensors that need GPU-validated
compaction logic).

GPU-validated against 2.9B model (RTX 3090 Ti): `verify_batch_v2_decode.py`
(3/3 checks, zero cross-contamination across 8×3 outputs with compaction
active), `verify_batch_compaction.py` (2/2), `test_local_state_and_batch.py`
(2/2). Production service restarted and serving with new code.


