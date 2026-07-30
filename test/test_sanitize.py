"""Regression test for sanitize_text_block (API_servers/router/openai_routes.py).

The function must preserve interior indentation and blank lines (critical for
code-containing prompts) while normalizing line endings and stripping
leading/trailing blank lines. This guards the fix from commit 4397328.

CPU-only, no model needed:
    uv run pytest test/test_sanitize.py -v
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from API_servers.router.openai_routes import sanitize_text_block


def test_preserves_code_indentation():
    """Indented code blocks must keep their leading whitespace."""
    code = "def foo():\n    return 42\n\nclass Bar:\n    pass"
    assert sanitize_text_block(code) == code


def test_preserves_interior_blank_lines():
    """Blank lines between paragraphs must survive."""
    text = "First paragraph.\n\nSecond paragraph."
    assert sanitize_text_block(text) == text


def test_strips_leading_trailing_blank_lines():
    """Leading/trailing blank lines are removed."""
    text = "\n\n  hello  \n\n"
    assert sanitize_text_block(text) == "  hello"


def test_normalizes_crlf():
    """Windows line endings are normalized to \\n."""
    text = "line1\r\nline2\r\nline3"
    assert sanitize_text_block(text) == "line1\nline2\nline3"


def test_strips_trailing_whitespace_per_line():
    """Trailing spaces/tabs on each line are removed."""
    text = "hello   \nworld\t\t"
    assert sanitize_text_block(text) == "hello\nworld"


def test_none_and_empty():
    """None and empty string produce empty output."""
    assert sanitize_text_block(None) == ""
    assert sanitize_text_block("") == ""


def test_multimodal_content_list():
    """OpenAI multimodal content (list of dicts) extracts text parts."""
    content = [{"type": "text", "text": "  indented code:\n    x = 1"}]
    result = sanitize_text_block(content)
    assert "    x = 1" in result
