"""
Hand-written tests for `extract_python_code` — the LLM-output sanitizer.

Coverage focus: the three-tier fallback strategy
  1. fenced ```python``` block (preferred)
  2. prose-strip fallback (LLM forgot the fence)
  3. raw-text fallback (last resort)
"""
from __future__ import annotations

from auto_sdet.graph.nodes.generator import extract_python_code


# ── Tier 1: fenced code block ──────────────────────────────────────

def test_python_fence_extracts_block():
    """Standard ```python block should be extracted, fence stripped."""
    text = """Here is the test:
```python
import pytest
def test_foo():
    assert 1 == 1
```
"""
    result = extract_python_code(text)
    assert result == "import pytest\ndef test_foo():\n    assert 1 == 1"


def test_unlabeled_fence_extracts_block():
    """A bare ``` fence (no language tag) should still be matched."""
    text = """```
import os
```"""
    result = extract_python_code(text)
    assert result == "import os"


def test_multiple_fences_returns_longest():
    """When multiple code blocks exist, the longest one wins."""
    text = """```python
print("short")
```
some prose
```python
import pytest
def test_a(): assert 1 == 1
def test_b(): assert 2 == 2
```
"""
    result = extract_python_code(text)
    assert "test_a" in result
    assert "test_b" in result
    assert "short" not in result


def test_fence_with_surrounding_prose_ignores_prose():
    """Prose around the fence must not leak into the result."""
    text = """Sure, here is the code:
```python
import pytest
```
Hope that helps!"""
    result = extract_python_code(text)
    assert result == "import pytest"
    assert "Sure" not in result
    assert "helps" not in result


# ── Tier 2: prose-strip fallback ────────────────────────────────────

def test_prose_then_import_strips_prose():
    """No fence but prose precedes a real import line — prose dropped."""
    text = """Now I have a good understanding of the code.
Let me write the tests.

import pytest
def test_foo(): assert True
"""
    result = extract_python_code(text)
    assert result.startswith("import pytest")
    assert "Now I have" not in result


def test_prose_then_from_strips_prose():
    """Same as above but starting with `from`."""
    text = """Reasoning: I need to import the module.
from foo import bar

def test(): pass"""
    result = extract_python_code(text)
    assert result.startswith("from foo import bar")
    assert "Reasoning" not in result


def test_prose_then_def_strips_prose():
    """Direct `def` start (no import) should also be detected."""
    text = """Let me write a function.

def helper():
    return 42
"""
    result = extract_python_code(text)
    assert result.startswith("def helper")
    assert "Let me" not in result


def test_prose_then_class_strips_prose():
    """`class` start should be detected."""
    text = """Here's the class:

class TestFoo:
    def test_x(self): pass
"""
    result = extract_python_code(text)
    assert result.startswith("class TestFoo")


def test_prose_then_decorator_strips_prose():
    """`@decorator` line should also be a valid Python start."""
    text = """Now writing parametrized tests:

@pytest.mark.parametrize("x", [1, 2])
def test_x(x): assert x > 0
"""
    result = extract_python_code(text)
    assert result.startswith("@pytest")


def test_prose_then_shebang_strips_prose():
    """A shebang line counts as Python source start."""
    text = """Note this is a script:

#!/usr/bin/env python
import sys
"""
    result = extract_python_code(text)
    assert result.startswith("#!/usr/bin/env python")


# ── Tier 3: raw-text fallback ───────────────────────────────────────

def test_no_python_marker_returns_raw_text():
    """Pure English with no Python-looking line — return whole text trimmed."""
    text = "I cannot generate code for this request."
    result = extract_python_code(text)
    assert result == "I cannot generate code for this request."


def test_empty_string_returns_empty():
    """Empty input stays empty."""
    assert extract_python_code("") == ""


def test_only_whitespace_returns_empty():
    """Whitespace-only input is normalized to empty."""
    assert extract_python_code("   \n\n  \t  ") == ""


# ── Edge cases ──────────────────────────────────────────────────────

def test_strips_trailing_whitespace():
    """Result must be .strip()-ed."""
    text = """```python

import os

```"""
    result = extract_python_code(text)
    assert result == "import os"
    assert not result.endswith("\n")
    assert not result.startswith("\n")


def test_indented_import_not_matched_as_start():
    """An indented import inside a function shouldn't be the prose-strip anchor."""
    # Top of file is prose; only an indented import exists. Since prose-strip
    # uses ^(import|...) with re.MULTILINE, indented `import` won't match.
    # The function should fall through to raw-text fallback.
    text = """Some prose here.
    import inside_function   # indented, not at line start
"""
    result = extract_python_code(text)
    # Falls through to raw-text — full text is returned trimmed
    assert "Some prose here." in result