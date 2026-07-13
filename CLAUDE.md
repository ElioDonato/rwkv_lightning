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
  hardcode another machine's absolute model path and have inconsistent
  import assumptions (one expects cwd=repo root, the other cwd=`infer/`) —
  don't run as-is. `test/test_local_state_and_batch.py` (added here)
  is an adapted equivalent that takes a model path as `argv[1]` and runs
  from the repo root; validates state token-count advances correctly
  across turns for both single-sequence and batched (bsz=2) generation.
  Note its generated *text* isn't as clean as the HTTP endpoints' — it
  doesn't replicate the `<think>\n</think>` formatting cue or stop-token
  handling the production routes use, so don't read too much into
  turn-2 content quality from that script specifically; the HTTP-level
  `test_curl.sh` state-memory-recall round is the stronger proof of
  correctness.

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
