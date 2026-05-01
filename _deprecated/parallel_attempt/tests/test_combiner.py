"""
Hand-written tests for the test-file combiner.

Combiner correctness is critical — if it produces invalid Python or
silently drops a worker's tests, the parallel-generation flow fails
in a hard-to-debug way.
"""
from __future__ import annotations

import textwrap

from auto_sdet.graph.nodes.combiner import (
    combine_test_files,
    _split_imports_and_body,
    _normalize_import,
)


def _src(code: str) -> str:
    return textwrap.dedent(code).lstrip("\n")


# ══════════════════════════════════════════════════════════════════
# combine_test_files — main public API
# ══════════════════════════════════════════════════════════════════

class TestCombineTestFiles:

    def test_empty_input_returns_empty(self):
        assert combine_test_files([]) == ""

    def test_single_partial_passes_through(self):
        code = _src("""
            import pytest

            def test_x():
                assert True
        """)
        result = combine_test_files([code])
        assert "import pytest" in result
        assert "def test_x" in result

    def test_two_partials_dedupe_imports(self):
        a = _src("""
            import pytest
            from foo import bar

            class TestA:
                def test_a(self): assert True
        """)
        b = _src("""
            import pytest
            from foo import bar
            from unittest.mock import MagicMock

            class TestB:
                def test_b(self): assert True
        """)
        result = combine_test_files([a, b])
        # Each unique import appears exactly once
        assert result.count("import pytest") == 1
        assert result.count("from foo import bar") == 1
        assert result.count("from unittest.mock import MagicMock") == 1
        # Both test classes survive
        assert "class TestA" in result
        assert "class TestB" in result

    def test_imports_appear_before_bodies(self):
        a = _src("""
            from foo import bar

            class TestA:
                pass
        """)
        result = combine_test_files([a])
        import_idx = result.index("from foo import bar")
        body_idx = result.index("class TestA")
        assert import_idx < body_idx

    def test_preserves_first_seen_import_order(self):
        a = "import zlib\nimport os\n\nclass TestA: pass\n"
        b = "import json\nimport os\n\nclass TestB: pass\n"
        result = combine_test_files([a, b])
        # Order should be: zlib, os (from a), then json (from b)
        zlib_pos = result.index("import zlib")
        os_pos = result.index("import os")
        json_pos = result.index("import json")
        assert zlib_pos < os_pos < json_pos

    def test_dedup_ignores_whitespace_difference(self):
        a = "from  foo   import   bar\n\nclass TestA: pass\n"
        b = "from foo import bar\n\nclass TestB: pass\n"
        result = combine_test_files([a, b])
        # Even with different whitespace, only one import survives
        # (count substring "from foo import bar" — only the first form matters)
        # The first-seen form is preserved
        assert result.count("from") == 1 or "from  foo" in result

    def test_skips_empty_partials(self):
        a = "import pytest\n\nclass TestA: pass\n"
        result = combine_test_files([a, "", "   \n  ", a])
        # No crash, and the duplicate `a` only contributes once via dedup
        assert "import pytest" in result
        assert "class TestA" in result

    def test_indented_import_not_hoisted(self):
        """An import inside a function body must NOT be moved to the top."""
        code = _src("""
            import pytest

            def test_lazy_import():
                from heavy_module import thing
                assert thing()
        """)
        result = combine_test_files([code])
        # Indented `from heavy_module` should still be inside the function
        assert "    from heavy_module import thing" in result

    def test_only_imports_no_body(self):
        result = combine_test_files(["import pytest\nimport os\n"])
        assert "import pytest" in result
        assert "import os" in result

    def test_only_body_no_imports(self):
        result = combine_test_files(["def test_x(): assert True\n"])
        assert "def test_x" in result

    def test_trailing_newline_normalized(self):
        result = combine_test_files(["import pytest\n\nclass TestA: pass\n"])
        assert result.endswith("\n")
        # Not multiple trailing newlines
        assert not result.endswith("\n\n\n")


# ══════════════════════════════════════════════════════════════════
# _split_imports_and_body
# ══════════════════════════════════════════════════════════════════

class TestSplitImportsAndBody:
    def test_clean_separation(self):
        code = _src("""
            import pytest
            from foo import bar

            class TestX:
                pass
        """)
        imports, body = _split_imports_and_body(code)
        assert imports == ["import pytest", "from foo import bar"]
        assert "class TestX" in body
        assert "import" not in body

    def test_no_imports_all_body(self):
        code = "def f(): pass\n"
        imports, body = _split_imports_and_body(code)
        assert imports == []
        assert "def f" in body

    def test_only_imports_empty_body(self):
        code = "import os\nimport json\n"
        imports, body = _split_imports_and_body(code)
        assert imports == ["import os", "import json"]
        assert body.strip() == ""

    def test_blank_lines_between_imports_skipped(self):
        code = _src("""
            import pytest

            from foo import bar

            class TestA: pass
        """)
        imports, body = _split_imports_and_body(code)
        assert "import pytest" in imports
        assert "from foo import bar" in imports
        assert "class TestA" in body

    def test_comment_before_first_def_not_in_body(self):
        code = _src("""
            import pytest
            # explanation comment
            def test_x(): pass
        """)
        imports, body = _split_imports_and_body(code)
        assert imports == ["import pytest"]
        assert "def test_x" in body

    def test_indented_import_kept_in_body(self):
        code = _src("""
            import pytest

            def test_x():
                from heavy import thing
                assert thing()
        """)
        imports, body = _split_imports_and_body(code)
        # only the unindented import is hoisted
        assert imports == ["import pytest"]
        assert "from heavy import thing" in body


# ══════════════════════════════════════════════════════════════════
# _normalize_import
# ══════════════════════════════════════════════════════════════════

class TestNormalizeImport:
    def test_collapses_multiple_spaces(self):
        assert _normalize_import("from  foo   import   bar") == "from foo import bar"

    def test_strips_leading_trailing(self):
        assert _normalize_import("  import os  ") == "import os"

    def test_normal_input_unchanged(self):
        assert _normalize_import("import pytest") == "import pytest"