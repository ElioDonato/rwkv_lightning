# rwkv_lightning 🕊️ ⚡
RWKV Batch infer backend Base on [Albatross](https://github.com/BlinkDL/Albatross) 🕊️ and [fastapi](https://github.com/fastapi/fastapi)
- Thanks to [Rapid-Sampling](https://github.com/Triang-jyed-driung/Rapid-Sampling) Kernel From [Triang-jyed-driung](https://github.com/Triang-jyed-driung), it also have native HIP kerel compatible with ROCm😎
## Install requirements
**For Nvidia CUDA**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install fastapi pydantic ninja numpy 
```
**For AMD ROCm**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.4
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

## Tips
If you want to the max performance optimization, you can use the ```torch.compile(mode='max-autotune-no-cudagraphs')```  

you can modify the code in the ```rwkv_batch/rwkv7.py``` line 30, 31
```python
MyFunction = torch.compile(mode='max-autotune-no-cudagraphs')
MyStatic = torch.compile(mode='max-autotune-no-cudagraphs')
```
**But it will be slow in first inference request, Because it needs to compile the Triton kernel firstly.**

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
