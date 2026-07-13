# Investigation: intermittent garbage output from `/big_batch/completions` under concurrent load

**Date:** 2026-07-13
**Reporter:** Claude Code, working in `~/search` (System 2 / `test_atomic_extraction` prototype)
**Trigger:** Repeated, varied-shape failures calling the 2.9B RWKV-7 model (`rwkv7-g1h-2.9b-20260710-ctx10240`) through `lm.small` while running `backend/test_atomic_extraction/` — first in `ExtractQuotesModule` (chunk 0, 5+ separate runs, 5 different garbage shapes), later also in `ClassifyTypeModule` during a GEPA optimization pass, escalating to a hard crash with a "recursion depth exceeded" parser error. Asked to check server health, consider a restart, dig into the batching-content issue specifically, and write this report.

## TL;DR

- **Server process is healthy.** GPU, memory, uvicorn process all nominal. The whole machine rebooted at 06:26 (this session started ~28 min after), so `rwkv-lightning` is already running fresh under runit — no separate restart is needed right now, and I did not perform one. See "Restart decision" below for when one would be warranted going forward.
- **A live reproduction test (3 concurrent distinguishable prompts × 3 rounds) found no literal cross-request word contamination** — BANANA/APPLE/CARROT responses never leaked each other's target word. That specific hypothesis (naive token-level bleed between concurrently-generating requests) is **not confirmed**.
- **What the repro *did* reproduce**: 5 of 9 trivial single-word-repetition requests degenerated into hallucinated meta-commentary (fake `<Analyze>` blocks, `Answer: xxx` templates, injected `Assistant:`/`Example:` role tokens, "I'll write a Python program that..." digressions) instead of following the instruction. This is the same *family* of failure seen in the extraction pipeline (off-topic real-document content, repetition loops, block-character spam, recursion-depth parser errors) — broken instruction-following/generation integrity under concurrent load, not simple contamination.
- **Strong but still circumstantial code-level evidence** points at `InferenceEngine.model_lock` (`infer/inference.py:23`) being genuinely dead — declared, never acquired anywhere in the codebase — while the prefill-permit queue (same file) is deliberately designed to admit *multiple* concurrent tickets, and both `big_batch.py` and `batch_inference.py` call the shared `self.model.forward_batch(...)` from async generators that yield control (`await asyncio.sleep(0)`) after every token, all while inside an open `torch.inference_mode()` context. This is a plausible mechanism for state corruption that doesn't show up as literal word-swapping (e.g. subtle blending of the RWKV recurrent state tensors between interleaved requests), but the repro test does not prove this mechanism — it only reproduces the symptom under concurrency.
- **Recommendation:** treat this as unresolved. The evidence is suggestive, not conclusive. See "Suggested next steps."

## Background: what `~/search` was already seeing

Working in `backend/test_atomic_extraction/`, calling `lm.small` (this server) via the custom batching layer (`backend/batching.py::RWKVInlineBatcher`, which posts to `/big_batch/completions`):

- `ExtractQuotesModule` on chunk 0 of the same test document failed differently on **5 separate process runs**: off-topic content from what looked like a different real document, a repetition loop, an error-message-shaped artifact, block-character/glyph spam, and a Python recursion-depth exceeded error surfacing from a downstream parser trying to make sense of the garbage. All 5 runs used the same input and same code.
- Earlier in the same session, a diagnostic already found that serializing concurrent batches (`lm.small.batch.max_concurrent_batches: 3 → 1`) **eliminated one specific symptom** (wildly off-topic/unrelated-topic responses) but **did not eliminate all failures** — a separate repetition/hallucination mode persisted even with concurrency reduced to 1 in-flight batch. (Config was later raised to 20 per your instruction that the server supports up to 20 concurrent batches; see `~/search/backend/config.yaml`.)
- Separately, `backend/batching.py::RWKVInlineBatcher._send_batch` had a real, now-fixed client-side bug: it used `first_params.get("max_tokens"/"stop_tokens")` for the whole batch instead of taking the max/union across all items in the batch — meaning items 2..N in a batch could silently get truncated or fail to stop correctly if their own request had different params than item 0. This is a genuine client-side bug and was fixed, but it doesn't explain the failure shapes seen (garbage content, not just wrong-length output), so it's a partial mitigation, not the root cause.

This pattern — inconsistent, run-to-run-different corruption that gets *less frequent but not eliminated* by reducing concurrency — is what prompted the deeper server-side investigation below.

## Server health check (2026-07-13, ~06:55)

```
uptime: up 28 min (machine rebooted ~06:26:44)
rwkv-lightning service: run, pid 1116, up 1719s (~28.6 min) — i.e. running since boot, no crash/respawn since then
GPU: 29% util, 6084/24564 MiB used, 35°C — healthy, plenty of headroom
service.log: clean startup sequence (model load succeeded, routes registered), followed by
             continuous normal traffic (ticket admit/release pairs, 200 OK responses) with
             no exceptions or tracebacks in the visible tail
```

The process itself is not crash-looping, leaking memory, or GPU-starved. Whatever is producing garbage output is not a resource-exhaustion or crash-and-respawn problem — it's a live correctness bug that produces bad output on some fraction of concurrent requests while the server otherwise reports 200 OK.

## Code-path analysis: is there a concurrency hole?

Traced the actual code path for `/big_batch/completions` (confirmed via `app.py` that this — not `app_big_batch.py`, which is unused — is what's really running):

1. **`API_servers/router/v1_routes.py`** — the route handler builds a `CancellationToken`, calls `engine.big_batch_stream(...)`, and returns it wrapped in `prefill_sse_response` (SSE streaming response).
2. **`infer/inference.py::InferenceEngine.__init__`** declares:
   ```python
   self.model_lock = Lock()                                    # line 23 — never referenced again anywhere in the repo (grep-confirmed, single hit)
   self.executor = ThreadPoolExecutor(max_workers=128, ...)     # line 24
   self._prefill_queue = deque()                                # line 27
   self._prefill_reserved_bsz = 0                                # line 28
   ```
   `acquire_prefill_permit`/`release_prefill_permit` (same file) implement a **ticket-based admission-control queue that explicitly allows multiple tickets to be concurrently admitted**, bounded only by aggregate `reserved_bsz` against `max_prefill_bsz` (~400-request budget, not a mutual-exclusion count of 1). This is corroborated live in `service.log`:
   ```
   [PrefillQueue] admitted ticket=1343 ... reserved_bsz=5 max_prefill_bsz=401
   [PrefillQueue] admitted ticket=1344 ... reserved_bsz=6 max_prefill_bsz=401   # admitted before 1343 released
   [PrefillQueue] released ticket=1343 ... reserved_bsz=3 max_prefill_bsz=401
   ```
   So multiple requests are, by design, concurrently in-flight past the admission gate — this queue is a capacity throttle, not a serialization point.
3. **`infer/big_batch.py::BigBatchMixin.big_batch_stream`** — the method actually serving this traffic:
   ```python
   with inference_deps.get_torch().inference_mode():          # line 28 — opened once, held for the whole generation
       state = self.model.generate_zero_state(batch_size)
       ...
       while not all(finished) and max_length > 0:
           ...
           out = self.model.forward_batch(new_tokens, state)  # line 53 — shared self.model, no lock
           ...
           await asyncio.sleep(0)                              # line 110 — cooperative yield, EVERY token step, while still inside inference_mode()
   ```
   `infer/batch_inference.py` (the sibling non-big-batch generation mixin, same `self.model`) has the identical pattern repeated across 4 near-duplicate generation methods — every one of them calls `self.model.forward_batch(...)` unguarded and yields via `asyncio.sleep(0)` inside an open `inference_mode()` block.
4. Did **not** trace into the custom CUDA kernels backing `rwkv7.py`'s `forward_batch` (`.cu` files) for hidden shared scratch buffers — this remains unverified and is the biggest remaining gap in the analysis.

**What this means:** every in-flight `/big_batch/completions` (or any other generation-path) request holds its own `state` object, but they all call into the *same* `self.model` object's `forward_batch`, and the event loop can switch between coroutines at every single-token yield point. If `forward_batch` (or the CUDA kernel underneath it) has *any* shared mutable scratch state, mid-token-step interleaving of two requests is exactly the shape of bug that would produce hard-to-reproduce, run-varying corruption — matching the observed symptom pattern (different garbage shape every run, worse with more concurrency, not eliminated by reducing concurrency to 1 batch since even 1 "batch" can itself contain multiple items processed together).

**Precedent:** `~/rwkv_lightning/CLAUDE.md` documents a bug fixed the day before (2026-07-12) in `API_servers/router/common.py`'s disconnect-watcher: an anyio `CancelScope` race where a `task.cancel()` landing while the watcher is inside an already-cancelled scope could leave a watcher task stuck (comments at `common.py:99-100`, `151`). That's a different subsystem, but it establishes that this codebase has a real, recent history of subtle async-interleaving races — this isn't a first-time hypothesis for this project.

## Live reproduction test

Wrote and ran `repro_cross_contamination.py`: 3 concurrent single-item requests with easily-distinguishable content (repeat BANANA / APPLE / CARROT ten times), 3 rounds, checking whether any response contains another prompt's target word.

**Result: no cross-word contamination in any of the 9 responses.** But 5/9 responses failed to follow the (trivially simple) instruction at all:

```
round 0: apple -> "APPLE APPLE APPLE APPLE APPLE APPLE APPLE APPLE APPLE APPLE"   [correct]
round 0: banana -> "\nYour response should end with `Answer: xxx`...\n<Analyze>\nFirst, the task is to repeat..."   [hallucinated meta-commentary]
round 0: carrot -> "\nYour response should end with `Answer: xxx`...\nWe are going to write a Python program that..."   [same failure mode]
round 1: banana -> similar meta-commentary hallucination
round 1: apple  -> injected "Example:\nInput:\nAPPLE\nOutput:..." template + "<think>Okay, I need to solve this problem..."
round 1: carrot -> same Python-program hallucination as round 0
round 2: apple  -> "Example:\nInput:\nAPPLE\n..." template again + "<think>" tag
round 2: carrot -> same Python-program hallucination, third occurrence
round 2: banana -> "Example:\nInput:\nBANANA\n..." template, partial correct completion at the end
```

**Interpretation:** this rules out the *simplest* form of the contamination hypothesis (literal word-swap between concurrent requests) but reproduces the *broader* symptom family — under concurrent load, the model reliably fails to follow simple instructions and instead emits patterns that look like fragments of its own training data (chat templates, code-generation reflexes, meta-reasoning scaffolding) bleeding through. That's consistent with — but does not prove — the state-corruption mechanism above; it's equally consistent with e.g. the model's actual quality ceiling on trivial repetition tasks, sampling/temperature artifacts, or a prompt-template mismatch specific to `/big_batch/completions` vs. the single-request `/openai/v1` endpoint. **The repro test cannot distinguish between "shared inference state got corrupted by a concurrent request" and "this model/endpoint just isn't reliable at this task shape."** A serial (non-concurrent) control run of the same 9 prompts was not performed and would be the natural next test to separate these two explanations.

## Restart decision

You said "check rwkv server health, maybe restart it." The machine rebooted at 06:26, ~28 minutes before this check — `rwkv-lightning` is already running as a fresh process (pid 1116, up since boot, clean startup log, no crashes since). **I did not perform an additional restart**, since there's no evidence a fresh process would behave differently from the current one (the failure is reproducible on this already-fresh instance, per the repro test above) — restarting again would just be motion, not a fix. If you want to test whether the issue is somehow tied to long-running process state despite this being a fresh boot, that would be a deliberate experiment (restart, then immediately rerun the repro test) rather than a default action, and I'd want to confirm with you first since it interrupts a live service.

## Suggested next steps (not yet performed)

1. **Serial control run**: same 9 BANANA/APPLE/CARROT prompts, fired one-at-a-time with no concurrency, to check whether the meta-commentary failure mode disappears entirely, drops in frequency, or is unchanged. This is the cleanest way to separate "concurrency-induced corruption" from "model/endpoint baseline unreliability" and should be done before spending more effort on the locking hypothesis.
2. **If serial is clean**: the `model_lock`/shared-`forward_batch` hypothesis gets much stronger, and the fix is architectural — either actually acquire `self.model_lock` around `forward_batch` calls (serializing generation, which has real throughput cost) or confirm the CUDA kernel is safe for interleaved calls with per-request state (would need kernel-level tracing, not done here).
3. **If serial is also unreliable**: the issue is more likely inherent to this model at this size/task, or a prompt-formatting mismatch specific to `/big_batch/completions` — worth comparing against the same prompts sent to `/openai/v1/chat/completions` (the single-request endpoint, unaffected by any of the big-batch concurrency machinery) to see if reliability differs by endpoint alone.
4. Neither of these was run today — this report reflects the investigation as requested, not a completed fix.

## Addendum (2026-07-13, later same day): root cause found, model_lock hypothesis ruled out

Ran both suggested next steps.

**Step 1 (serial control):** 9 BANANA/APPLE/CARROT prompts fired one-at-a-time,
no concurrency, against the live 2.9b server. Result: **0/9 clean** — same
failure rate as the concurrent run (also re-tested same day: 0/9 clean,
worse than the original 5/9-failed run, likely just prompt-wording/sampling
variance). Serial being no better than concurrent rules out the
`model_lock`/shared-`forward_batch` concurrency-corruption hypothesis —
whatever the model_lock code smell is (still genuinely dead code, still
worth a cleanup ticket), it is not the cause of this symptom.

**Step 3 (endpoint/templating comparison):** compared three conditions,
same instruction, same server:

| Condition | Result |
|---|---|
| A) raw instruction → `/big_batch/completions` (`contents: [instruction]`, no wrapping) | **0/9 clean** |
| B) same instruction manually wrapped as `User: {instruction}\n\nAssistant: <think>\n</think>\n` → same endpoint | **8/9 clean** |
| C) same instruction via `/openai/v1/chat/completions` (`messages: [{"role":"user",...}]`) | **9/9 clean** |

This isolates the variable completely: identical model, identical
`/big_batch/completions` code path, identical sampling params — the only
difference between (A) and (B) is whether the prompt string is wrapped in
the model's expected chat scaffold. Wrapping it turns 0/9 into 8/9.

**Root cause:** `/big_batch/completions` (`API_servers/router/v1_routes.py:237-257`)
passes `req.contents` straight into `engine.big_batch_stream(prompts=...)`
with **no templating whatsoever** — contrast with `/openai/v1/chat/completions`,
which builds `User: {content}\n\nAssistant: <think>\n</think>\n` via
`format_openai_prompt()` (`openai_routes.py:105-118`) before hitting the
same underlying generation code. `/big_batch/completions` is, by design,
a raw-completion endpoint (also used for FIM/translation, where raw
completion is correct) — it has no opinion on chat formatting, and never
did.

The actual bug is client-side, in `~/search/backend/batching.py::_send_batch`
(lines 371-416): it extracts only the first `role == "user"` message's
`content` from each request's `messages` list and sends it verbatim as
`contents` to `/big_batch/completions` — **dropping the `system` message
entirely** (DSPy's task instructions typically live there) **and never
applying any chat-turn wrapping**. Every symptom in this report — the
extraction pipeline's off-topic/repetition/glyph-spam/recursion-crash
failures, and the repro's hallucinated `<Analyze>`/`Answer: xxx`/code-gen
artifacts — is consistent with the model being handed an unwrapped,
system-prompt-free fragment and free-associating on it, not with any
state corruption in this server.

**This was not caught by the earlier partial client-side fix** (report
line 21, the `max_tokens`/`stop_tokens` batch-union fix) because that fix
addressed a different bug in the same function without touching the
prompt-construction logic above it.

**Status: root cause identified, fix not yet applied.** The fix belongs
in `~/search/backend/batching.py::_send_batch` — build the same
`User: .../Assistant: <think>\n</think>\n` scaffold (including the system
message) before sending to `/big_batch/completions`, mirroring
`format_openai_prompt()` in this repo. That's a `~/search` change, out of
scope for this repo/session; flagged to the user rather than applied
here. No changes needed in `rwkv_lightning` itself — `/big_batch/completions`
behaved exactly as designed throughout this investigation.

## Addendum 2 (2026-07-13, later same day): system-prompt-present vs. -absent, isolated from templating

Since DSPy (the real caller) puts task instructions in the `system` role
and `_send_batch` drops `system` entirely (independent of the templating
bug above), ran a 2x2 matrix — {system message included / dropped} x
{chat template applied / not} — on 3 tasks where the instructions live in
`system` and genuinely matter (`extract_numbers`, `sentiment`; plus
`repeat_word` as a control where the instruction is self-contained in the
user turn), 3 reps each, 36 requests total:

| condition | clean |
|---|---|
| no system + no template (current `_send_batch` behavior) | 0/9 |
| system included, but naively concatenated (no `User:`/`Assistant:` markers) | 1/9 |
| no system, template applied | 3/9 (only the self-contained `repeat_word` task passed; the two system-dependent tasks failed 0/3 each) |
| **system included + template applied (`System: .../User: .../Assistant: <think>\n</think>\n`)** | **9/9** |

**Finding: the two fixes are not independent — you need both.** Restoring
the system message without proper `User:`/`Assistant:` template markers is
nearly as broken as dropping it (the model treats "instructions + input"
as one undifferentiated blob of text to continue, e.g. it answered a
totally different quiz-like continuation for `extract_numbers` rather than
extracting numbers). Applying the template without the system message
fixes tasks whose instructions happen to already be in the user turn but
does nothing for tasks (the DSPy-realistic case) that rely on `system` for
the actual task contract. Only combining both — i.e. reproducing
`format_openai_prompt()`'s exact scaffold, system included — got every
task right, every rep.

**Revised recommendation:** the `~/search/backend/batching.py::_send_batch`
fix must build the full `System: {system}\n\nUser: {user}\n\nAssistant: <think>\n</think>\n`
prompt (collecting `system` from the request body the same way
`collect_openai_prompt_parts()` does in this repo), not just add template
markers or just stop dropping `system` — either partial fix leaves the
pipeline close to as unreliable as today.

## Fix applied (2026-07-13, same day): server-side templating, not a client-side string

Deliberately did **not** put the fix in `~/search/backend/batching.py` as a
hardcoded template string, because the user plans to finetune this RWKV
model later and the expected prompt format may change (e.g. dropping the
`<think>` scaffold, changing role prefixes) once the workload moves to be
purely DSPy-driven. Hardcoding `format_openai_prompt()`'s output string in
a second, separate repo would mean every future template change requires
remembering to update two codebases in lockstep — exactly the kind of
drift this bug already came from (nobody remembered `_send_batch` needed
the template `/openai/v1/chat/completions` already had).

Instead, made `rwkv_lightning` the single source of truth for the chat
template, server-side, and had the client send *structured* data instead
of a pre-built prompt string:

**`rwkv_lightning` (this repo):**
- `API_servers/router/schemas.py`: added `ChatRequest.chats: list[dict] = []`
  — each item is an OpenAI-style `{"messages": [...], "system": ...}` dict.
- `API_servers/router/v1_routes.py::big_batch_completions`: if `req.chats`
  is set, builds each prompt via `format_openai_prompt(chat, req.enable_think)`
  — the exact same function `/openai/v1/chat/completions` already used —
  instead of taking `req.contents` as raw untemplated strings. Falls back
  to the legacy `req.contents` raw-completion path when `chats` is absent
  (backward compatible; `contents` remains the way to hit `/big_batch/completions`
  for genuinely raw/non-chat completion, if that's ever needed — no other
  code in this repo currently does, `RWKVInlineBatcher` was the only real
  caller of the raw-`contents` path).
- Consequence: if/when the finetuned model expects a different template,
  only `format_openai_prompt()` (or its `enable_think` branch) needs to
  change, in this one repo — every caller of both `/openai/v1/chat/completions`
  and `/big_batch/completions` (via `chats`) picks it up automatically.

**`~/search` (client):**
- `backend/batching.py::RWKVInlineBatcher._send_batch`: now forwards each
  request's full `messages` list (system message included) as a `chats`
  item instead of extracting only the first `user` message into `contents`.
  No template string, no knowledge of the model's prompt format, lives in
  this file anymore.

**Verification:**
1. Restarted `rwkv-lightning` service, confirmed clean startup (no recompile
   needed, `ninja: no work to do`).
2. Sent the `extract_numbers`/`sentiment`/`repeat_word` task set as a single
   real 3-item batch via the new `chats` field directly: 9/9 clean across
   3 reps.
3. Ran the actual `RWKVInlineBatcher.submit()` → `_send_batch()` code path
   (the real client code, not a simulation) against the live server, 4
   times: system message honored and correct task-following on all 12
   task-instances (one run under-generated `extract_numbers` to `"3, 2"`
   /`"47"` instead of the full `"3, 2, 47"` — ordinary temperature=1.0
   sampling variance, not the pre-fix hallucination/off-topic pattern).
4. `bash test/test_curl.sh` against the restarted server: clean pass, no
   errors/tracebacks, exit code 0 — no regression to existing behavior.

**Status: fixed and verified end-to-end in both repos.**
