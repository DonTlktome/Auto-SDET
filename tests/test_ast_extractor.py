"""
Hand-written tests for the AST top-level symbol extractor.

This module is the foundation of the parallel-generation flow, so we
test it thoroughly: extraction rules, edge cases, signature formatting,
and the build_signature_index helper.
"""
from __future__ import annotations

import textwrap

import pytest

from auto_sdet.models.schemas import SymbolSpec
from auto_sdet.tools.ast_extractor import (
    extract_top_level_symbols,
    build_signature_index,
)


def _src(code: str) -> str:
    """Dedent + strip leading newline so tests can use triple-quoted blocks."""
    return textwrap.dedent(code).lstrip("\n")


# ══════════════════════════════════════════════════════════════════
# Basic extraction
# ══════════════════════════════════════════════════════════════════

class TestBasicExtraction:
    def test_empty_source(self):
        assert extract_top_level_symbols("") == []

    def test_only_comments_and_imports(self):
        code = _src("""
            # just a comment
            import os
            from typing import Optional
        """)
        assert extract_top_level_symbols(code) == []

    def test_single_function(self):
        code = _src("""
            def add(a: int, b: int) -> int:
                return a + b
        """)
        symbols = extract_top_level_symbols(code)
        assert len(symbols) == 1
        assert symbols[0].name == "add"
        assert symbols[0].kind == "function"

    def test_single_class(self):
        code = _src("""
            class Calculator:
                def __init__(self):
                    self.history = []
        """)
        symbols = extract_top_level_symbols(code)
        assert len(symbols) == 1
        assert symbols[0].name == "Calculator"
        assert symbols[0].kind == "class"

    def test_multiple_top_level_symbols(self):
        code = _src("""
            def foo(): pass
            def bar(): pass
            class Baz: pass
        """)
        symbols = extract_top_level_symbols(code)
        names = [s.name for s in symbols]
        assert names == ["foo", "bar", "Baz"]


# ══════════════════════════════════════════════════════════════════
# Privacy rules
# ══════════════════════════════════════════════════════════════════

class TestPrivacyRules:
    def test_underscore_prefixed_function_excluded(self):
        code = _src("""
            def _private_helper(): pass
            def public_func(): pass
        """)
        symbols = extract_top_level_symbols(code)
        assert [s.name for s in symbols] == ["public_func"]

    def test_underscore_prefixed_class_excluded(self):
        code = _src("""
            class _Internal: pass
            class Public: pass
        """)
        symbols = extract_top_level_symbols(code)
        assert [s.name for s in symbols] == ["Public"]

    def test_dunder_method_inside_class_kept_in_signature(self):
        """__init__ is private-looking but defines the construction API."""
        code = _src('''
            class Box:
                def __init__(self, x: int):
                    self.x = x
                def get_x(self) -> int:
                    return self.x
                def _internal(self):
                    pass
        ''')
        symbols = extract_top_level_symbols(code)
        sig = symbols[0].signature
        assert "__init__" in sig
        assert "get_x" in sig
        assert "_internal" not in sig


# ══════════════════════════════════════════════════════════════════
# Nesting / scope rules
# ══════════════════════════════════════════════════════════════════

class TestNesting:
    def test_nested_function_not_extracted_as_top_level(self):
        code = _src("""
            def outer():
                def inner():
                    pass
                return inner
        """)
        symbols = extract_top_level_symbols(code)
        assert [s.name for s in symbols] == ["outer"]

    def test_method_not_extracted_as_top_level(self):
        code = _src("""
            class A:
                def method(self): pass
            def free_function(): pass
        """)
        symbols = extract_top_level_symbols(code)
        assert [s.name for s in symbols] == ["A", "free_function"]


# ══════════════════════════════════════════════════════════════════
# Source preservation
# ══════════════════════════════════════════════════════════════════

class TestSourcePreservation:
    def test_function_source_includes_body(self):
        code = _src('''
            def hello() -> str:
                """Greet the world."""
                return "hello"
        ''')
        symbols = extract_top_level_symbols(code)
        src = symbols[0].source
        assert "def hello()" in src
        assert "return \"hello\"" in src
        assert "Greet the world." in src

    def test_decorators_preserved_in_source(self):
        code = _src("""
            @staticmethod
            @cached
            def helper(x):
                return x
        """)
        symbols = extract_top_level_symbols(code)
        src = symbols[0].source
        assert "@staticmethod" in src
        assert "@cached" in src
        assert "def helper" in src

    def test_class_source_includes_methods(self):
        code = _src("""
            class Box:
                def __init__(self, x):
                    self.x = x
                def get(self):
                    return self.x
        """)
        symbols = extract_top_level_symbols(code)
        src = symbols[0].source
        assert "def __init__" in src
        assert "def get" in src


# ══════════════════════════════════════════════════════════════════
# Signature formatting
# ══════════════════════════════════════════════════════════════════

class TestFunctionSignature:
    def test_simple_function_signature(self):
        code = _src("def add(a: int, b: int) -> int: return a + b\n")
        sig = extract_top_level_symbols(code)[0].signature
        assert sig.startswith("def add(")
        assert "a: int" in sig
        assert "b: int" in sig
        assert "-> int" in sig
        # No body should leak into the signature
        assert "return" not in sig

    def test_signature_excludes_decorators(self):
        """Decorators belong to source, not to API shape."""
        code = _src("""
            @cached
            def foo(x: int) -> int:
                return x
        """)
        sig = extract_top_level_symbols(code)[0].signature
        assert "@cached" not in sig
        assert sig.startswith("def foo(")

    def test_signature_includes_docstring_summary(self):
        code = _src('''
            def fib(n: int) -> int:
                """Return the n-th Fibonacci number.

                Uses an iterative approach.
                """
                return 0
        ''')
        sig = extract_top_level_symbols(code)[0].signature
        assert "Return the n-th Fibonacci number" in sig
        # Only first line should be in signature, not the second paragraph
        assert "Uses an iterative approach" not in sig

    def test_signature_no_docstring(self):
        code = _src("def foo(): return 1\n")
        sig = extract_top_level_symbols(code)[0].signature
        assert sig == "def foo():"


class TestClassSignature:
    def test_class_signature_with_docstring_and_methods(self):
        code = _src('''
            class Calc:
                """A simple calculator."""
                def add(self, a: int, b: int) -> int:
                    return a + b
                def sub(self, a, b):
                    return a - b
        ''')
        sig = extract_top_level_symbols(code)[0].signature
        assert sig.startswith("class Calc:")
        assert '"""A simple calculator."""' in sig
        assert "def add(" in sig
        assert "def sub(" in sig

    def test_class_with_bases(self):
        code = _src("""
            class Child(Parent, Mixin):
                def f(self): pass
        """)
        sig = extract_top_level_symbols(code)[0].signature
        assert "class Child(Parent, Mixin):" in sig

    def test_class_signature_excludes_private_methods(self):
        code = _src("""
            class Box:
                def public(self): pass
                def _private(self): pass
        """)
        sig = extract_top_level_symbols(code)[0].signature
        assert "def public" in sig
        assert "_private" not in sig

    def test_class_with_no_methods(self):
        code = _src('''
            class Empty:
                """Just a marker class."""
                pass
        ''')
        sig = extract_top_level_symbols(code)[0].signature
        assert sig.startswith("class Empty:")


# ══════════════════════════════════════════════════════════════════
# Docstring extraction
# ══════════════════════════════════════════════════════════════════

class TestDocstringField:
    def test_function_with_docstring(self):
        code = _src('''
            def foo():
                """First line summary.

                Detailed explanation here.
                """
                pass
        ''')
        sym = extract_top_level_symbols(code)[0]
        assert sym.docstring == "First line summary."

    def test_function_without_docstring(self):
        code = _src("def foo(): return 1\n")
        sym = extract_top_level_symbols(code)[0]
        assert sym.docstring is None

    def test_class_docstring_first_line(self):
        code = _src('''
            class C:
                """Heading.

                Body text.
                """
                pass
        ''')
        sym = extract_top_level_symbols(code)[0]
        assert sym.docstring == "Heading."


# ══════════════════════════════════════════════════════════════════
# Failure modes
# ══════════════════════════════════════════════════════════════════

class TestFailureModes:
    def test_syntax_error_returns_empty_list(self):
        code = "def broken(:\n    pass\n"   # invalid
        assert extract_top_level_symbols(code) == []

    def test_only_invalid_top_level_statements(self):
        code = _src("""
            x = 1
            y = 2
            print('hello')
        """)
        # No functions or classes — should be empty
        assert extract_top_level_symbols(code) == []


# ══════════════════════════════════════════════════════════════════
# build_signature_index
# ══════════════════════════════════════════════════════════════════

class TestBuildSignatureIndex:
    @pytest.fixture
    def sample_symbols(self) -> list[SymbolSpec]:
        code = _src("""
            def foo(x: int) -> int:
                return x
            def bar(y: str) -> str:
                return y
            class Baz:
                def m(self): pass
        """)
        return extract_top_level_symbols(code)

    def test_excludes_target_symbol(self, sample_symbols):
        index = build_signature_index(sample_symbols, exclude_name="foo")
        assert "def bar(" in index
        assert "class Baz:" in index
        assert "def foo(" not in index

    def test_excludes_nothing_when_name_absent(self, sample_symbols):
        index = build_signature_index(sample_symbols, exclude_name="nonexistent")
        assert "def foo(" in index
        assert "def bar(" in index
        assert "class Baz:" in index

    def test_excludes_nothing_when_exclude_is_none(self, sample_symbols):
        index = build_signature_index(sample_symbols, exclude_name=None)
        assert "def foo(" in index
        assert "def bar(" in index

    def test_empty_symbols_returns_empty_string(self):
        assert build_signature_index([], exclude_name="anything") == ""

    def test_single_symbol_excluded_returns_empty_string(self, sample_symbols):
        only_foo = [sample_symbols[0]]   # just `foo`
        assert build_signature_index(only_foo, exclude_name="foo") == ""

    def test_signatures_separated_by_blank_lines(self, sample_symbols):
        """Visual separation makes the LLM-facing context easier to parse."""
        index = build_signature_index(sample_symbols)
        # We join with two newlines between signatures
        assert "\n\n" in index


# ══════════════════════════════════════════════════════════════════
# Integration: realistic algorithm-style file
# ══════════════════════════════════════════════════════════════════

class TestRealistic:
    def test_fibonacci_like_module(self):
        """Smoke test resembling TheAlgorithms/fibonacci.py shape."""
        code = _src('''
            """Module-level docstring."""
            from math import sqrt

            CONSTANT = 42

            def fib_iterative(n: int) -> list[int]:
                """Iterative Fibonacci."""
                seq = [0, 1]
                for _ in range(n - 1):
                    seq.append(seq[-1] + seq[-2])
                return seq[:n]

            def fib_binet(n: int) -> list[int]:
                """Binet's formula."""
                phi = (1 + sqrt(5)) / 2
                return [int(phi**i) for i in range(n)]

            def _helper():
                """Private utility."""
                pass

            class FibCache:
                """Memoization cache."""
                def get(self, n: int) -> int:
                    return n
        ''')
        symbols = extract_top_level_symbols(code)
        names = [s.name for s in symbols]
        # CONSTANT is a top-level assignment, ignored. _helper is private.
        assert names == ["fib_iterative", "fib_binet", "FibCache"]

        # Verify each kind
        kinds = {s.name: s.kind for s in symbols}
        assert kinds == {
            "fib_iterative": "function",
            "fib_binet": "function",
            "FibCache": "class",
        }

        # Spot-check signatures don't leak bodies
        fib_iter_sig = next(s for s in symbols if s.name == "fib_iterative").signature
        assert "for _ in range" not in fib_iter_sig
        assert "Iterative Fibonacci." in fib_iter_sig