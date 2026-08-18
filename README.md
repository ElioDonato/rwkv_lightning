# rwkv_lightning 🕊️ ⚡
RWKV Batch infer backend Base on [Albatross](https://github.com/BlinkDL/Albatross) 🕊️ and [fastapi](https://github.com/fastapi/fastapi)
- Thanks to [Rapid-Sampling](https://github.com/Triang-jyed-driung/Rapid-Sampling) Kernel From [Triang-jyed-driung](https://github.com/Triang-jyed-driung), it also have native HIP kerel compatible with ROCm😎

## 🔀 What this fork adds (differences from upstream)

This is an actively-developed fork that hardens the serving path and adds major
capabilities on top of the upstream RWKV-Vibe server. Everything is opt-in and
**off by default**, so the out-of-the-box behavior matches upstream byte-for-byte;
you flip env vars / flags to turn on the new features.

**GPU throughput & utilization**
- **Dynamic decode batching** — `RWKV_DYNAMIC_BATCH=1`: merges concurrent, even
  *heterogeneous*, chat requests into one shared multi-row decode with per-row
  sampling (`sample_batch_per_row`). Measured **~4–5× concurrent tokens/s** on a
  single 3090 Ti vs the bsz-1 baseline; flat scaling to N=128 clients.
- **CUDA-graph decode** — `RWKV_CUDA_GRAPH=1`: replays `forward_batch` as a CUDA
  graph (per-size pool, static buffers) to cut per-token host launch overhead.
  Requires the fork's fix that launches the seq/batch WKV kernel on the current
  CUDA stream (upstream's omits it, which blocked graph capture).
- **Process-wide CUDA serialization** — one global lock guarantees only one thread
  is ever inside CUDA, so the (also opt-in) async GPU-worker offload
  (`RWKV_ASYNC_FORWARD`) is safe to enable even with the blocking/embed paths.
- Aggregated embedding prefill (`RWKV_EMBED_AGGREGATE`) and code cleanups /
  allocation dedup throughout.

**Multi-model serving (one process, many models)**
- Declare a catalog in `models.json` (`app.py --models-config models.json`):
  arbitrary RWKV `.pth` checkpoints, identified by an `id` (e.g. two different
  2.9b models, or a 2.9b + 7.2b).
- Requests pick a model with the OpenAI `model` field; models **load lazily** on
  first use and can **co-exist in VRAM** (limited by the card).
- Designate an **embed model** (`--embed-model` / JSON `embed_model`) so the
  `/embedding` endpoints use one model while chat/state use the default.
- **Auth-gated runtime admin API**: `GET /admin/models`,
  `POST /admin/models/load|unload|unload_all` (free VRAM, LRU-evict under
  `RWKV_MAX_RESIDENT_BYTES`).
- `/openai/v1/models` lists the whole catalog with resident/default status.

**Universal env-driven configuration**
- A single `settings.py` reads every tuning knob from env vars (`RWKV_*`) with
  defaults added — ports, host, model path, auth, batch/window/ceiling sizes,
  sampler defaults — so nothing machine-specific is hardcoded. See
  *Server tuning env vars* below.
- `embedding_run.sh` serves the head-less embedding endpoint; `env.sh` carries
  local model/auth values (kept out of git).

Continue with the install/setup instructions below; for the fork-specific
endpoints and multi-model quick start, see
[*Fork additions quick start*](#fork-additions-quick-start).

## Install requirements
**For Nvidia CUDA**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu132
pip install fastapi pydantic ninja numpy 
```
**For AMD ROCm**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm7.2
pip install fastapi pydantic ninja numpy 
```

## Quantized inference backends

Only the large attention, FFN, and output-head matrices are quantized. Small
RWKV low-rank matrices and embeddings remain FP16 to limit accuracy loss.

### CUDA W8A16

The custom CUDA extension uses CUTLASS headers. Clone CUTLASS into the exact
path below before the first import (the extension is compiled lazily by
`torch.utils.cpp_extension`):

```bash
mkdir -p infer/rwkv_batch/cuda/third_party
git clone --depth 1 --branch v3.9.2 \
  https://github.com/NVIDIA/cutlass.git \
  infer/rwkv_batch/cuda/third_party/cutlass
```

Export a W8A16 checkpoint:

```bash
python -m infer.rwkv_batch.quant.export_quant \
  /path/to/model.pth \
  /path/to/model-w8a16.pth \
  --bits 8
```

Load it for inference (`MODEL_NAME` does not include the `.pth` suffix):

```python
from infer.rwkv_batch.rwkv7_w8a16 import RWKV_x070

args.MODEL_NAME = "/path/to/model-w8a16"
model = RWKV_x070(args)
```

### GemLite A16 weight-only formats

Install [GemLite](https://github.com/dropbox/gemlite), then select one of
`A16W8_FP8`, `A16W8_INT8`, or `A16W4_HQQ_INT`. Conversion writes already
packed GemLite tensors, so model loading does not quantize or repack weights.
The packed metadata schema and optimized loader currently target GemLite 0.6.x.

```bash
pip install gemlite==0.6.0

# For the virtual environment used by this repository, if pip is unavailable:
uv pip install \
  --python /mnt/pc411_data/python_env/nv-py312/bin/python \
  gemlite==0.6.0

# Recommended accuracy / speed default
python -m infer.rwkv_batch.quant.export_quant_gemlite \
  /path/to/model.pth \
  /path/to/model-gemlite-int8.pth \
  --format A16W8_INT8

# FP8 weight-only
python -m infer.rwkv_batch.quant.export_quant_gemlite \
  /path/to/model.pth \
  /path/to/model-gemlite-fp8.pth \
  --format A16W8_FP8

# HQQ-compatible INT4, grouped along the input dimension
python -m infer.rwkv_batch.quant.export_quant_gemlite \
  /path/to/model.pth \
  /path/to/model-gemlite-w4.pth \
  --format A16W4_HQQ_INT \
  --group-size 64
```

GemLite checkpoints must be exported with the current conversion script; old
checkpoint layouts are not supported by the loader.

GemLite compiles and may autotune Triton kernels when it first sees each matrix
and batch shape. Warm up every batch size before timing it; the first pass can
take seconds and is not representative of decode throughput. W4 has a
substantially larger accuracy cost than W8, so prefer `A16W8_INT8` unless
memory capacity is the primary constraint. At larger batch sizes these A16
weight-only kernels still perform FP16 tensor-core work after unpacking or
dequantization, so INT4 and INT8 can converge to similar compute-bound speed.

## Usage
```bash
# FP16 (default)
python app.py --model-path /path/to/model --inference-engine fp16 \
  --port 8000 --password rwkv7_7.2b

# GemLite packed checkpoint
python app.py --model-path /path/to/model-gemlite-int8 \
  --inference-engine gemlite --port 8000 --password rwkv7_7.2b

# CUTLASS W8A16 checkpoint
python app.py --model-path /path/to/model-w8a16 \
  --inference-engine cutlass --port 8000 --password rwkv7_7.2b
```

`--backend` is accepted as a shorter alias for `--inference-engine`. GemLite
and CUTLASS checkpoints use different layouts and cannot be interchanged.
- if no password, you can do not add ```--password``` flag

### Server tuning env vars

A few internal server limits/timeouts have sane defaults but can be
overridden per-deployment without editing source:

- `RWKV_MAX_ALLOWED_TOKENS` (default `32768`): hard ceiling on `max_tokens`
  per request. This is a fairness/DoS guard, not a GPU-memory limit (RWKV's
  recurrent state is O(1) in sequence length) — it exists because a request
  holds its prefill-admission slot for its entire generation, so an
  unbounded `max_tokens` lets a handful of concurrent requests starve every
  other client indefinitely. Lower it for a smaller/more defensive
  deployment, or raise it if you legitimately need longer generations.
- `RWKV_PREFILL_BSZ_REFRESH_INTERVAL_S` (default `2.0`): how often the
  server re-measures free VRAM to estimate the max safe prefill batch size.
  Lower values react faster to VRAM freed by other processes on a shared
  GPU, at the cost of more frequent blocking CUDA syncs (`torch.cuda.
  empty_cache()` + `mem_get_info()`) on the request-handling path.
- `RWKV_DISCONNECT_WATCHER_CLEANUP_TIMEOUT_S` (default `1.0`) /
  `RWKV_DISCONNECT_WATCHER_CLEANUP_POLL_INTERVAL_S` (default `0.05`): how
  long/how often the server retries cancelling a per-request
  disconnect-watcher task during cleanup before giving up (an abandoned
  watcher is harmless — it exits on its own once the client actually
  disconnects). Rarely needs tuning; exposed mainly for deployments running
  under unusually heavy event-loop load.
- `RWKV_CORS_ORIGINS` (default `*`): comma-separated list of allowed CORS
  origins, e.g. `http://localhost:3000,https://app.example.com`. The
  default `*` allows any origin (suitable for trusted-LAN deployments).
  Set a specific allowlist if the server is reachable from untrusted
  networks.


## Test API quickly
```bash
bash ./test/test_curl.sh
```

## WebUI (`webui_rwkv.py`)

A Gradio-based demo/ops UI for talking to a running `rwkv_lightning` backend
(chat, batch generation, HTML-wall demos, etc). Start it with:

```bash
python webui_rwkv.py
```

By default it binds `0.0.0.0:7860` **with no login**, matching prior
behavior. This is intended for a trusted LAN / single-user setup only. The
webui process makes outbound HTTP requests to whatever "API URL" / "Delete
URL" is configured in the UI, so anyone who can reach it can use your GPU
backend and (if unrestricted) point it at other internal hosts. Two
independent hardening controls are available, both **opt-in / off by
default** to avoid breaking existing deployments:

- **Login (`RWKV_WEBUI_AUTH`)**: set to `"user:password"` (or a
  comma-separated list, `"user1:pass1,user2:pass2"`) to require a login
  screen before the UI is usable. If unset, the webui starts open (as
  before) and prints a startup warning to the console.
  ```bash
  RWKV_WEBUI_AUTH="admin:change-me" python webui_rwkv.py
  ```
- **Bind address / port**: `RWKV_WEBUI_HOST` (default `0.0.0.0`) and
  `RWKV_WEBUI_PORT` (default `7860`). Set `RWKV_WEBUI_HOST=127.0.0.1` to
  restrict access to localhost only.
- **Backend URL allowlist**: the "API URL" / "Delete URL" textboxes in the
  UI are restricted by default to `127.0.0.1` / `localhost` / `::1` (the
  hosts baked into the `DEFAULT_*_URL` constants at the top of
  `webui_rwkv.py`), to prevent the webui being used as an open SSRF relay
  toward arbitrary hosts if it's reachable by untrusted users. To point the
  webui at a backend on another trusted host, set
  `RWKV_WEBUI_ALLOWED_HOSTS` to a comma-separated allowlist of extra
  hostnames/IPs. To disable this restriction entirely (not recommended if
  the webui itself is reachable by anyone you don't fully trust), set
  `RWKV_WEBUI_ALLOW_ANY_BACKEND=1`.

Recommended for anything beyond a fully trusted, single-user LAN:

```bash
RWKV_WEBUI_AUTH="admin:change-me" RWKV_WEBUI_HOST=127.0.0.1 python webui_rwkv.py
```

## API Docs 


### **1. Batch synchronous Translate**

<details>
<summary><strong><em>curl examples</em></strong></summary>

**Compatible with immersive translation custom API**
**--- Very stable 🚀 ---** 
```bash
curl -X POST http://localhost:8000/translate/v1/batch-translate \
         -H "Content-Type: application/json" \
         -d '{
           "source_lang": "en",
           "target_lang": "zh-CN",
           "text_list": ["Hello world!", "Good morning"]
         }'
```
```bash
curl -X POST http://localhost:8000/translate/v1/batch-translate \
         -H "Content-Type: application/json" \
         -d '{
           "source_lang": "zh-CN",
           "target_lang": "en",
           "text_list": ["你好世界", "早上好"]
         }'
```
</details>

___
### **2. ```v1/chat/completions```  [Support all decode parameters]**

<details>
<summary><strong><em>curl examples</em></strong></summary>

**--- Very stable 🚀 ---** 
- Streaming synchronous batch processing 
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "contents": [
      "English: After a blissful two weeks, Jane encounters Rochester in the gardens. He invites her to walk with him, and Jane, caught off guard, accepts. Rochester confides that he has finally decided to marry Blanche Ingram and tells Jane that he knows of an available governess position in Ireland that she could take.\n\nChinese:",
      "English: That night, a bolt of lightning splits the same chestnut tree under which Rochester and Jane had been sitting that evening.\n\nChinese:"
    ],
    "max_tokens": 1024,
    "stop_tokens": ["\nUser:"],
    "temperature": 0.8,
    "top_k": 50,
    "top_p": 0.6,
    "alpha_presence": 1.0,
    "alpha_frequency": 0.1,
    "alpha_decay": 0.99,
    "stream": true,
    "password": "rwkv7_7.2b"
  }'
```
- Non-streaming synchronous batch processing
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "contents": [
      "English: After a blissful two weeks, Jane encounters Rochester in the gardens. He invites her to walk with him, and Jane, caught off guard, accepts. Rochester confides that he has finally decided to marry Blanche Ingram and tells Jane that he knows of an available governess position in Ireland that she could take.\n\nChinese:",
      "English: That night, a bolt of lightning splits the same chestnut tree under which Rochester and Jane had been sitting that evening.\n\nChinese:"
    ],
    "max_tokens": 1024,
    "stop_tokens": ["\nUser:"],
    "temperature": 0.8,
    "top_k": 50,
    "top_p": 0.6,
    "alpha_presence": 1.0,
    "alpha_frequency": 0.1,
    "alpha_decay": 0.99,
    "stream": false,
    "password": "rwkv7_7.2b"
  }'
```

</details>


___
### **3. ```/v2/chat/completions``` [GPU-native sampler, used by the webui]**

Same request/response shape as `/v1/chat/completions` (`contents` list in,
one `chat.completion`-style choice out per prompt, `session_id`/`stop_tokens`/
`chunk_size` all supported the same way), but decoding runs through a
different, GPU-native top-k/top-p sampling kernel (`sample_logits_batch_cuda`
in `infer/inference_utils.py`) instead of the `/v1` sampler, and the default
decode parameters differ: `top_k=500`, `top_p=0.5`, `alpha_presence=1.0`,
`alpha_frequency=0.1`, `alpha_decay=0.99` (vs. `/v1`'s `top_k=50`, `top_p=0.6`,
`alpha_presence=2`, `alpha_frequency=0.2`, `alpha_decay=0.996`). Auth,
back-pressure (`bsz overflow` 400 response), and disconnect handling are the
same as `/v1/chat/completions`.

**Per-item `finish_reason` and batch compaction:** like `/big_batch/completions`,
the streaming path emits a per-item `finish_reason` (`"stop"` or `"length"`) in
each row's terminal SSE chunk, and compacts finished rows out of the GPU batch
mid-decode so remaining active rows get faster per-step compute. Non-streaming
responses include a per-choice `finish_reason` field. Clients can resolve an
individual request as soon as its `finish_reason` arrives rather than waiting
for `[DONE]`.

This is the endpoint `webui_rwkv.py` uses by default for its batch-generation
tabs (`DEFAULT_BATCH_API_URL`).

<details>
<summary><strong><em>curl examples</em></strong></summary>

- Streaming synchronous batch processing
```bash
curl -X POST http://localhost:8000/v2/chat/completions \
  -H "Content-Type: application/json" \
  -N \
  -d '{
    "contents": ["Hi there!", "Tell me a joke."],
    "max_tokens": 1024,
    "chunk_size": 128,
    "stream": true,
    "password": "rwkv7_7.2b"
  }'
```
- Non-streaming synchronous batch processing
```bash
curl -X POST http://localhost:8000/v2/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "contents": ["Hi there!", "Tell me a joke."],
    "max_tokens": 1024,
    "stream": false,
    "password": "rwkv7_7.2b"
  }'
```

</details>

___
### **4. ```state/chat/completions``` [Support state cache manager] 😜**

#### Have 3 Levels Cache design 🤓
- **L1 cache(VRAM) 16**
- **L2 cache(RAM) 32**
- **L3 cache(Sqlite3 database)**
#### The all cached state will be stored in the database when shout down the server 😋
- could modify the cache size in ```./state_pool.py``` in line 14-16

***Need to add a unique "session_id": "XXX" in the request body as a unique identifier for each session***👆

**ONLY support for bsz = 1 one session** 🤫

<details>
<summary><strong><em>curl examples</em></strong></summary>

- Streaming asynchronous batch processing With CUDA Graph For Bsz=1
```bash
curl -X POST http://localhost:8000/state/chat/completions \
  -H "Content-Type: application/json" \
  -N \
  -d '{
    "contents": [
      "User: What should we eat for dinner? Any brief suggestions?\n\nAssistant: <think>\n</think>\n"
    ],
    "max_tokens": 1024,
    "stop_tokens": ["\nUser:"],
    "temperature": 0.8,
    "top_k": 50,
    "top_p": 0.6,
    "alpha_presence": 1.0,
    "alpha_frequency": 0.1,
    "alpha_decay": 0.99,
    "stream": true,
    "chunk_size": 128,
    "password": "rwkv7_7.2b",
    "session_id": "session_one"
  }'
```
- Non-streaming asynchronous batch processing With CUDA Graph For Bsz=1
```bash
curl -X POST http://localhost:8000/state/chat/completions \
      -H "Content-Type: application/json" \
      -d '{
    "contents": [
      "User: What should we eat for dinner? Any brief suggestions?\n\nAssistant: <think>\n</think>\n"
    ],
    "max_tokens": 1024,
    "stop_tokens": ["\nUser:"],
    "temperature": 0.8,
    "top_k": 50,
    "top_p": 0.6,
    "alpha_presence": 1.0,
    "alpha_frequency": 0.1,
    "alpha_decay": 0.99,
    "stream": false,
    "password": "rwkv7_7.2b",
    "session_id": "session_one"
  }'
```

</details>

___
### **5. State Management API [Support state cache manager] 😜**

#### Use ```state/status```  Interface to check the state pool status of a session

<details>
<summary><strong><em>curl examples</em></strong></summary>

```bash
curl -X POST http://localhost:8000/state/status \
  -H "Content-Type: application/json" \
  -d '{
    "password": "rwkv7_7.2b"
  }'
```

</details>

#### Use ```state/delete```  Interface to delete the state of a session

Set ```"delete_prefix": true``` to also delete every ```/multi_state``` branch
whose session id starts with ```"<session_id>:"``` (see below), not just the
```/state``` session itself.

<details>
<summary><strong><em>curl examples</em></strong></summary>


```bash
curl -X POST http://localhost:8000/state/delete \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "your_session_id_to_delete",
    "delete_prefix": true,
    "password": "rwkv7_7.2b"
  }'
```

</details>

___
### **6. ```/multi_state/chat/completions``` [Branching state sessions]**

Like ```/state/chat/completions```, but instead of one mutable session it
keeps a tree of numbered dialogue turns: give it ```session_id``` +
```dialogue_idx``` (the turn you're continuing from, ```0``` for a fresh
tree), and it stores the result under the *next* free ```dialogue_idx``` for
that session as ```"<session_id>:<new_dialogue_idx>"```, returned in the
response body (and, for streaming, as an extra
```{"object": "multi_state.dialogue_idx", ...}``` SSE event before the token
chunks). This lets a client branch/replay conversation history by requesting
the same ```dialogue_idx``` again instead of always continuing linearly.
Only supports single-session (bsz=1) requests, same as ```/state```.

<details>
<summary><strong><em>curl examples</em></strong></summary>

```bash
curl -X POST http://localhost:8000/multi_state/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "session_one",
    "dialogue_idx": 0,
    "contents": ["User: What should we eat for dinner?\n\nAssistant:"],
    "max_tokens": 1024,
    "stream": false,
    "password": "rwkv7_7.2b"
  }'
```

</details>

___
### **7. ```/openai/v1/chat/completions``` [Open AI format support]**

<details>
<summary><strong><em>curl examples</em></strong></summary>

- Streaming asynchronous Open AI API
```bash
curl -X POST 'http://localhost:8000/openai/v1/chat/completions' \
  --header 'Content-Type: application/json' \
  --header 'Authorization: Bearer your-password-if-set' \
  --data '{
    "model": "rwkv7",
    "messages": [
      {"role": "user", "content": "please tell me about the history of artificial intelligence"}
    ],
    "top_p": 0.6,
    "max_tokens": 2048,
    "temperature": 0.8,
    "stream": true
  }'
```
- Non-streaming asynchronous Open AI API
```bash
curl -X POST 'http://localhost:8000/openai/v1/chat/completions' \
  --header 'Content-Type: application/json' \
  --header 'Authorization: Bearer your-password-if-set' \
  --data '{
    "model": "rwkv7",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "please tell me about the history of artificial intelligence"}
    ],
    "top_p": 0.6,
    "max_tokens": 2048,
    "temperature": 1,
    "stream": false
  }'
```

- Stateful incremental Open AI API with `session_id`
```bash
curl -X POST 'http://localhost:8000/openai/v1/chat/completions' \
  --header 'Content-Type: application/json' \
  --header 'Authorization: Bearer your-password-if-set' \
  --data '{
    "model": "rwkv7",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Please continue from our last turn and give me 3 short ideas."}
    ],
    "top_p": 0.6,
    "max_tokens": 2048,
    "temperature": 1,
    "stream": false
  }'
```

</details>

Also exposes ```GET /v1/models``` and ```GET /openai/v1/models``` (the
latter honoring the same `Authorization: Bearer` password check), both
returning an OpenAI-style `{"object": "list", "data": [{"id": <model name>, ...}]}`
body for client auto-discovery.

___
### **8. ```/big_batch/completions```  [Only Support temperature decode parameters]**

<details>
<summary><strong><em>curl examples</em></strong></summary>

**The Fastest Batch Processing API 🚀** 
- Streaming synchronous batch processing 
```bash
curl -X POST 'http://localhost:8000/big_batch/completions' \
  --header 'Content-Type: application/json' \
  --data '{
    "contents": [
      "English: That night, a bolt of lightning splits the same chestnut tree under which Rochester and Jane had been sitting that evening.\n\nChinese:",
      "English: That night, a bolt of lightning splits the same chestnut tree under which Rochester and Jane had been sitting that evening.\n\nChinese:"
    ],
    "max_tokens": 1024,
    "stop_tokens": ["\nUser:"],
    "temperature": 1.0,
    "chunk_size": 8,
    "stream": true,
    "password": "rwkv7_7.2b"
  }'
```
</details>

**SSE response format:** the endpoint streams `data: {...}\n\n` chunks
followed by a final `data: [DONE]\n\n`. Each chunk is an OpenAI-style
`chat.completion.chunk` whose `choices` carry per-item deltas keyed by
`index` (the original prompt position in the request's `contents`/`chats`
array):

```json
{"object": "chat.completion.chunk",
 "choices": [{"index": 0, "delta": {"content": "partial text"}}]}
```

When an item finishes, its terminal chunk includes a per-item
`finish_reason` — `"stop"` (hit a stop token) or `"length"` (exhausted
`max_tokens`) — and that index receives no further deltas:

```json
{"object": "chat.completion.chunk",
 "choices": [{"index": 0, "delta": {"content": "final text"},
              "finish_reason": "stop"}]}
```

Every index gets exactly one `finish_reason`. Clients can resolve an
individual request as soon as its `finish_reason` arrives rather than
waiting for `[DONE]` (which only fires once the whole batch completes).
Finished rows are also compacted out of the GPU batch mid-decode, so
early-finishing items free compute for the still-active ones.

___
### **9. FIM ( For RWKV7_G1c series model )**

<details>
<summary><strong><em>curl examples</em></strong></summary>

**Batch stream inference using [FIM/v1/batch-FIM interface]**

```bash
curl -X POST http://localhost:8000/FIM/v1/batch-FIM \
  -H "Content-Type: application/json" \
  -d '{
    "prefix": [
      "The rain had stopped, but the street still glistened like a river of broken glass.",
      "She wasn’t sure why she’d come back.",
      "A cat darted from the alley,"
    ],
    "suffix": [
      "though everyone knew Mr. Ellis hadn’t opened that door in three years.",
      "sounding almost like her name.",
      "And then, from inside, a single lamp clicked on."
    ],
    "max_tokens": 1024,
    "stop_tokens": ["✿"],
    "temperature": 0.8,
    "top_k": 50,
    "top_p": 0.6,
    "alpha_presence": 1.0,
    "alpha_frequency": 0.1,
    "alpha_decay": 0.99,
    "stream": true,
    "password": "rwkv7_7.2b"
  }'
```

**Batch inference using [FIM/v1/batch-FIM interface]**

```bash
curl -X POST http://localhost:8000/FIM/v1/batch-FIM \
  -H "Content-Type: application/json" \
  -d '{
    "prefix": [
      "The rain had stopped, but the street still glistened like a river of broken glass.",
      "She wasn’t sure why she’d come back.",
      "A cat darted from the alley,"
    ],
    "suffix": [
      "though everyone knew Mr. Ellis hadn’t opened that door in three years.",
      "sounding almost like her name.",
      "And then, from inside, a single lamp clicked on."
    ],
    "max_tokens": 1024,
    "stop_tokens": ["✿"],
    "temperature": 0.8,
    "top_k": 50,
    "top_p": 0.6,
    "alpha_presence": 1.0,
    "alpha_frequency": 0.1,
    "alpha_decay": 0.99,
    "stream": false,
    "password": "rwkv7_7.2b"
  }'
```

</details>

___
### **10. `/v1/responses` [OpenAI Responses API — Stateful multi-turn]**

The [Responses API](https://platform.openai.com/docs/api-reference/responses) is a stateful interface that maps naturally to RWKV's recurrent state. Use `previous_response_id` to chain multi-turn conversations — each turn resumes from the stored RWKV state (O(1) per turn, no re-processing history).

<details>
<summary><strong><em>curl examples</em></strong></summary>

- Single turn
```bash
curl -X POST http://localhost:8081/v1/responses \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-password-if-set" \
  -d '{
    "model": "rwkv7",
    "input": "Write a one-sentence bedtime story about a unicorn.",
    "max_output_tokens": 256
  }'
```

- Multi-turn (stateful via `previous_response_id`)
```bash
# Turn 1
curl -X POST http://localhost:8081/v1/responses \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-password-if-set" \
  -d '{
    "model": "rwkv7",
    "input": "My name is Alice. Remember it.",
    "instructions": "You are a helpful assistant.",
    "max_output_tokens": 100
  }'
# Response includes "id": "resp_..." — use it in the next turn

# Turn 2
curl -X POST http://localhost:8081/v1/responses \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-password-if-set" \
  -d '{
    "model": "rwkv7",
    "input": "What is my name?",
    "previous_response_id": "resp_...",
    "max_output_tokens": 50
  }'
```

- Streaming
```bash
curl -N -X POST http://localhost:8081/v1/responses \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-password-if-set" \
  -d '{
    "model": "rwkv7",
    "input": "Hello!",
    "stream": true,
    "max_output_tokens": 256
  }'
```

</details>

**Parameters:**
- `input` (required): string or message array (`[{"role": "user", "content": "..."}]`)
- `instructions`: system-level guidance (like a system prompt)
- `previous_response_id`: resume from a previous response's stored RWKV state
- `max_output_tokens`: max tokens to generate (default 1024)
- `temperature`, `top_p`: sampling parameters
- `stream`: SSE streaming with `response.output_text.delta` events
- `store`: whether to store state for multi-turn (default `true`)

**Response format** matches the OpenAI spec: `output[]` array with typed message items, `usage` stats, and a `resp_...` ID for chaining.

---

## Fork additions quick start

Everything here is a fork addition (see the *What this fork adds* section at the top).

### Embedding endpoints (head-less)
Serve embeddings from any model without a WebUI:
```bash
# /embedding            -> "project-compatible" shape (top-level list of vectors)
# /v1/embeddings        -> OpenAI-compatible shape
curl http://localhost:8000/embedding -H 'Content-Type: application/json' \
  -d '{"input": "hello world"}'
```
Optional aggregating prefill batching: `RWKV_EMBED_AGGREGATE=1`.

### Multi-model serving
Declare a catalog (`models.json`) and start the server, then pick models by id on
every request; models load lazily on first use and can co-reside in VRAM:
```bash
# models.json
# { "models": [
#     {"id":"small","path":"models/rwkv7-2.9b.pth"},
#     {"id":"big",  "path":"models/rwkv7-7.2b.pth"}
#   ] }
app.py --models-config models.json
```
- `POST /openai/v1/chat/completions {"model":"big", ...}` serves the 7.2b; omit or
  use an unknown `model` to get the default (first-declared, or `--default-model`).
- **Embed-vs-chat split:** set `--embed-model <id>` (or `"embed_model"` in the
  JSON) so `/embedding` & `/v1/embeddings` use that model while chat/state use the
  default; an explicit `model` field still overrides.
- **Runtime management** (bearer-auth, same password as the serving routes):
  ```bash
  GET  /admin/models                      # catalog + residency + VRAM footprint
  POST /admin/models/load   {"id":"big"}  # load now
  POST /admin/models/unload {"id":"big"}  # stop using + free VRAM
  POST /admin/models/unload_all
  ```
- Cap resident VRAM with `RWKV_MAX_RESIDENT_BYTES=<bytes>`; the least-recently-used
  non-default model is evicted automatically to make room.
- `GET /openai/v1/models` lists the whole catalog with `resident`/`default` flags.

The single-model path is unchanged (`app.py --model-path models/<file>.pth`) and
behaves byte-for-byte like upstream.

### Performance flags (all opt-in, all default-off)
| Env | What it does |
|-----|--------------|
| `RWKV_DYNAMIC_BATCH` | merge concurrent heterogeneous chat requests into one shared multi-row decode (~4–5× concurrent throughput) |
| `RWKV_CUDA_GRAPH` | replay the decode forward as a CUDA graph (needs the fork's stream fix) |
| `RWKV_ASYNC_FORWARD` | offload the heavy GPU forward to a single worker thread |
| `RWKV_FUSE_CHAT_BATCH` | older combine-queue for homogeneous chat requests |
| `RWKV_EMBED_AGGREGATE` | batched embedding prefill |
| `RWKV_MAX_RESIDENT_BYTES` | multi-model VRAM budget -> LRU eviction |
| `RWKV_EMBED_MODEL` | model id/`id` used by the embedding endpoints by default |

Every configurable knob (ports, host, model path, auth, batch/window/ceiling
sizes, sampler defaults, ...) is centralized in `settings.py`, read from
`RWKV_*` env vars with defaults.
