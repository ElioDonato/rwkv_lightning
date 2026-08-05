"""CPU-only unit tests for create_translation_prompt
(API_servers/router/v1_routes.py), used by POST /translate/v1/batch-translate.
Zero prior test coverage -- pure function, easy to verify without a model.

Run with: uv run pytest test/test_translation_prompt.py -v
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from API_servers.router.v1_routes import create_translation_prompt


def test_known_language_codes_map_to_display_names():
    prompt = create_translation_prompt("en", "zh-CN", "hello world")
    assert prompt == "English: hello world\n\nChinese:"


def test_both_zh_variants_map_to_chinese():
    """zh-CN and zh-TW are distinct locale codes but both display as
    'Chinese' -- the model doesn't distinguish simplified/traditional in
    the prompt itself, only via the actual generated characters."""
    assert create_translation_prompt("zh-CN", "en", "x").startswith("Chinese:")
    assert create_translation_prompt("zh-TW", "en", "x").startswith("Chinese:")


def test_all_documented_language_codes():
    expected = {
        "zh-CN": "Chinese",
        "zh-TW": "Chinese",
        "en": "English",
        "ja": "Japanese",
        "fr": "French",
        "de": "German",
        "es": "Spanish",
        "ru": "Russian",
    }
    for code, name in expected.items():
        prompt = create_translation_prompt(code, "en", "text")
        assert prompt.startswith(f"{name}:"), f"code={code!r} expected {name!r} prefix, got {prompt!r}"


def test_unknown_language_code_passes_through_verbatim():
    """A code not in the lookup table (e.g. 'auto', or a locale this table
    doesn't know about) must fall back to the raw code itself rather than
    raising or silently mapping to something wrong -- lang_names.get(code, code)."""
    prompt = create_translation_prompt("auto", "it", "ciao")
    assert prompt == "auto: ciao\n\nit:"


def test_source_and_target_resolved_independently():
    prompt = create_translation_prompt("fr", "ja", "bonjour")
    assert prompt == "French: bonjour\n\nJapanese:"


def test_preserves_text_content_verbatim_including_newlines():
    """The text to translate is interpolated as-is, including any newlines
    or special characters it contains -- create_translation_prompt does no
    sanitization itself (unlike sanitize_text_block used elsewhere)."""
    text = "line one\nline two"
    prompt = create_translation_prompt("en", "de", text)
    assert prompt == "English: line one\nline two\n\nGerman:"


def test_empty_text():
    prompt = create_translation_prompt("en", "es", "")
    assert prompt == "English: \n\nSpanish:"


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
