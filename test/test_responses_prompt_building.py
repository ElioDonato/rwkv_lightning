"""Regression tests for the /v1/responses prompt-building helper
(API_servers/router/responses_routes.py::_build_prompt_from_input).

Covers: string vs message-array input, instructions only applied on the
first turn (not continuations, since a resumed state already encodes any
system prompt from turn 1), multimodal content-array flattening, and the
blank-line prefix used for continuation turns (mirrors normalize_state_prompts
for /state/).

CPU-only, no model needed:
    uv run pytest test/test_responses_prompt_building.py -v
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from API_servers.router.responses_routes import _build_prompt_from_input


def test_string_input_first_turn():
    prompt = _build_prompt_from_input("Hello there!", instructions=None, is_continuation=False)
    assert prompt == "User: Hello there!\n\nAssistant:"


def test_instructions_included_on_first_turn():
    prompt = _build_prompt_from_input(
        "Hi", instructions="You are a pirate.", is_continuation=False
    )
    assert prompt.startswith("System: You are a pirate.\n\nUser: Hi")


def test_instructions_excluded_on_continuation():
    """A resumed state already encodes turn 1's system prompt -- repeating it
    on every subsequent turn would duplicate it in the model's context."""
    prompt = _build_prompt_from_input(
        "What's my name?", instructions="You are a pirate.", is_continuation=True
    )
    assert "System:" not in prompt
    assert "You are a pirate." not in prompt


def test_continuation_prefixes_blank_line():
    prompt = _build_prompt_from_input("Follow-up question", instructions=None, is_continuation=True)
    assert prompt.startswith("\n\nUser: Follow-up question")


def test_first_turn_no_leading_blank_line():
    prompt = _build_prompt_from_input("First question", instructions=None, is_continuation=False)
    assert not prompt.startswith("\n\n")


def test_message_array_input():
    messages = [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "4"},
        {"role": "user", "content": "And 3+3?"},
    ]
    prompt = _build_prompt_from_input(messages, instructions=None, is_continuation=False)
    assert "User: What is 2+2?" in prompt
    assert "Assistant: 4" in prompt
    assert "User: And 3+3?" in prompt
    assert prompt.endswith("\n\nAssistant:")


def test_multimodal_content_array_flattened_to_text():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this: "},
                {"type": "image_url", "image_url": {"url": "http://example.com/x.png"}},
                {"type": "text", "text": "please"},
            ],
        }
    ]
    prompt = _build_prompt_from_input(messages, instructions=None, is_continuation=False)
    assert "Describe this: please" in prompt


def test_empty_input_list_produces_bare_assistant_marker():
    prompt = _build_prompt_from_input([], instructions=None, is_continuation=False)
    assert prompt == "\n\nAssistant:"


def test_role_capitalized():
    messages = [{"role": "system", "content": "Be terse."}]
    prompt = _build_prompt_from_input(messages, instructions=None, is_continuation=False)
    assert "System: Be terse." in prompt


# -- ResponsesRequest schema validation --------------------------------------

from pydantic import ValidationError

from API_servers.router.schemas import MAX_ALLOWED_TOKENS, ResponsesRequest


def test_responses_request_defaults():
    req = ResponsesRequest(input="hi")
    assert req.max_output_tokens == 1024
    assert req.store is True
    assert req.stream is False


def test_responses_request_rejects_empty_input():
    for bad_input in ("", [], None):
        try:
            ResponsesRequest(input=bad_input)
            assert False, f"expected ValidationError for input={bad_input!r}"
        except ValidationError:
            pass


def test_responses_request_caps_max_output_tokens():
    try:
        ResponsesRequest(input="hi", max_output_tokens=MAX_ALLOWED_TOKENS + 1)
        assert False, "expected ValidationError for max_output_tokens over the cap"
    except ValidationError as exc:
        assert "exceeds the server limit" in str(exc)


def test_responses_request_accepts_message_array_input():
    req = ResponsesRequest(input=[{"role": "user", "content": "hi"}])
    assert isinstance(req.input, list)


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
