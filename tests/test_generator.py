"""
Tests for the auto_sdet.prompts.generator module.

Covers:
- Module-level prompt constants
- build_generator_prompt() function
"""

import pytest

from auto_sdet.prompts.generator import (
    GENERATOR_SYSTEM_PROMPT,
    GENERATOR_USER_PROMPT,
    build_generator_prompt,
)


# ---------------------------------------------------------------------------
# Tests for module-level constants
# ---------------------------------------------------------------------------

class TestModuleConstants:
    """Verify that the prompt template constants are well-formed."""

    def test_system_prompt_is_non_empty_string(self):
        """GENERATOR_SYSTEM_PROMPT should be a non-empty string."""
        assert isinstance(GENERATOR_SYSTEM_PROMPT, str)
        assert len(GENERATOR_SYSTEM_PROMPT) > 0

    def test_user_prompt_is_non_empty_string(self):
        """GENERATOR_USER_PROMPT should be a non-empty string."""
        assert isinstance(GENERATOR_USER_PROMPT, str)
        assert len(GENERATOR_USER_PROMPT) > 0

    def test_user_prompt_contains_expected_placeholders(self):
        """GENERATOR_USER_PROMPT must contain the three format placeholders."""
        assert "{source_path}" in GENERATOR_USER_PROMPT
        assert "{source_code}" in GENERATOR_USER_PROMPT
        assert "{context_files_section}" in GENERATOR_USER_PROMPT


# ---------------------------------------------------------------------------
# Tests for build_generator_prompt()
# ---------------------------------------------------------------------------

class TestBuildGeneratorPrompt:
    """Tests for the build_generator_prompt function."""

    # -- Happy path ----------------------------------------------------------

    def test_returns_tuple_of_two_strings(self):
        """Return type is a 2-tuple of strings."""
        result = build_generator_prompt(
            source_path="/fake/path.py",
            source_code="def foo(): pass",
            context_files={},
        )
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], str)
        assert isinstance(result[1], str)

    def test_system_prompt_is_unchanged(self):
        """The first element is exactly GENERATOR_SYSTEM_PROMPT."""
        system, _ = build_generator_prompt(
            source_path="/a/b.py",
            source_code="x = 1",
            context_files={},
        )
        assert system == GENERATOR_SYSTEM_PROMPT

    def test_user_prompt_contains_source_path(self):
        """The user prompt embeds the source_path argument."""
        _, user = build_generator_prompt(
            source_path="/my/project/module.py",
            source_code="print('hello')",
            context_files={},
        )
        assert "/my/project/module.py" in user

    def test_user_prompt_contains_source_code(self):
        """The user prompt embeds the source_code argument."""
        code = "def add(a, b): return a + b"
        _, user = build_generator_prompt(
            source_path="/x.py",
            source_code=code,
            context_files={},
        )
        assert code in user

    def test_user_prompt_with_single_context_file(self):
        """A single context file appears inside <file> tags."""
        _, user = build_generator_prompt(
            source_path="/src/main.py",
            source_code="pass",
            context_files={"dep.py": "DEP_CONTENT"},
        )
        assert '<file path="dep.py">' in user
        assert "DEP_CONTENT" in user
        assert "</file>" in user

    def test_user_prompt_with_multiple_context_files(self):
        """Multiple context files each get their own <file> block."""
        context = {
            "a.py": "AAA",
            "b.py": "BBB",
        }
        _, user = build_generator_prompt(
            source_path="/s.py",
            source_code="42",
            context_files=context,
        )
        assert '<file path="a.py">' in user
        assert "AAA" in user
        assert '<file path="b.py">' in user
        assert "BBB" in user
        # Order is insertion-order (Python 3.7+), so a.py before b.py
        assert user.index("a.py") < user.index("b.py")

    # -- Edge cases ----------------------------------------------------------

    def test_empty_context_files_produces_placeholder_message(self):
        """When context_files is empty, a fallback message is inserted."""
        _, user = build_generator_prompt(
            source_path="/p.py",
            source_code="",
            context_files={},
        )
        assert "(No dependency files found)" in user

    def test_empty_source_path(self):
        """Empty source_path is accepted and embedded literally."""
        _, user = build_generator_prompt(
            source_path="",
            source_code="x=1",
            context_files={},
        )
        # The placeholder {source_path} is replaced with empty string
        assert 'source_file path=""' in user

    def test_empty_source_code(self):
        """Empty source_code is accepted and embedded literally."""
        _, user = build_generator_prompt(
            source_path="/f.py",
            source_code="",
            context_files={},
        )
        # The empty source_code appears between the <source_file> tags
        assert '<source_file path="/f.py">\n\n</source_file>' in user

    def test_special_characters_in_source_code(self):
        """Special characters (braces, angle brackets) are preserved."""
        tricky_code = "def f(): return '<{test}>'"
        _, user = build_generator_prompt(
            source_path="/t.py",
            source_code=tricky_code,
            context_files={},
        )
        assert tricky_code in user

    def test_special_characters_in_context_file_names_and_content(self):
        """Context file names and content with special chars are preserved."""
        context = {
            "path/to/file.py": "content with <tags> & {braces}",
        }
        _, user = build_generator_prompt(
            source_path="/s.py",
            source_code="pass",
            context_files=context,
        )
        assert 'path/to/file.py' in user
        assert "content with <tags> & {braces}" in user

    def test_large_context_files(self):
        """Function handles a large number of context files without error."""
        many_files = {f"file_{i}.py": f"content_{i}" for i in range(100)}
        _, user = build_generator_prompt(
            source_path="/big.py",
            source_code="pass",
            context_files=many_files,
        )
        for i in range(100):
            assert f"file_{i}.py" in user
            assert f"content_{i}" in user

    @pytest.mark.parametrize(
        "source_path,source_code,context_files",
        [
            ("/a.py", "x=1", {}),
            ("/a.py", "x=1", {"d.py": "d"}),
            ("", "", {}),
            ("/p", "", {"a.py": "", "b.py": "b"}),
        ],
    )
    def test_various_input_combinations(
        self, source_path, source_code, context_files
    ):
        """Parametrized smoke test: function returns without error for varied inputs."""
        system, user = build_generator_prompt(source_path, source_code, context_files)
        assert isinstance(system, str)
        assert isinstance(user, str)
        assert len(system) > 0
        assert len(user) > 0