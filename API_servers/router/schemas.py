from typing import Optional

from pydantic import BaseModel, field_validator

# Hard ceiling on max_tokens (decode-step budget per request). This is not a
# GPU-memory limit (RWKV's recurrent state is O(1) in sequence length, unlike
# a transformer KV-cache), it's a DoS/fairness guard: reserve_prefill_capacity
# holds a request's slot against the shared max_prefill_bsz admission budget
# for the *entire* generation, not just prefill. Without a cap, a handful of
# concurrent requests with max_tokens=999999999 (bsz=1 each, so individually
# admitted instantly) can each occupy one budget slot indefinitely, starving
# every other client even though the shared prefill-bsz admission control is
# working exactly as designed. See auth-sweep.md for the full writeup.
MAX_ALLOWED_TOKENS = 32768


class ChatRequest(BaseModel):
    model: str = "rwkv7"
    contents: list[str] = []
    # Batched alternative to `contents` for /big_batch/completions: each item
    # is an OpenAI-style {"messages": [...], "system": ...} dict, chat-templated
    # server-side the same way as /openai/v1/chat/completions (single source of
    # truth for the prompt format, so it stays correct across model finetunes).
    chats: list[dict] = []
    messages: list[dict] = []
    system: Optional[str] = None
    prefix: list[str] = []
    suffix: list[str] = []
    max_tokens: int = 8192
    stop_tokens: list[str] = ["\nUser:"]
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.6
    noise: float = 1.5
    stream: bool = False
    pad_zero: bool = True
    alpha_presence: float = 2
    alpha_frequency: float = 0.2
    alpha_decay: float = 0.996
    enable_think: bool = False
    chunk_size: int = 4
    password: Optional[str] = None
    session_id: Optional[str] = None
    dialogue_idx: Optional[int] = 0
    use_prefix_cache: bool = True

    @field_validator("max_tokens")
    @classmethod
    def _clamp_max_tokens(cls, value: int) -> int:
        if value > MAX_ALLOWED_TOKENS:
            raise ValueError(
                f"max_tokens={value} exceeds the server limit of {MAX_ALLOWED_TOKENS}"
            )
        return value


class TranslateRequest(BaseModel):
    source_lang: str = "auto"
    target_lang: str
    text_list: list[str]
    placeholders: Optional[list[str]] = None
    password: Optional[str] = None


class TranslateResponse(BaseModel):
    translations: list[dict]
