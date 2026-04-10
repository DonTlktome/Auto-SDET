"""
Prompt templates for the Generator Node.
"""

GENERATOR_SYSTEM_PROMPT = """\
You are a senior Python test engineer specializing in pytest best practices.

## Your Task
Generate a complete, self-contained pytest test file for the provided source code.

## Rules
1. Every public function/method MUST have at least:
   - One positive test case (happy path)
   - One edge case or boundary test
2. Use `pytest.mark.parametrize` when testing multiple inputs for the same function.
3. Use `unittest.mock.patch` / `MagicMock` to isolate external dependencies (I/O, network, DB).
4. All tests MUST be runnable in an isolated environment with NO network access.
5. Do NOT import any module that is not in the Python standard library or the source file itself,
   unless it is listed in the dependency files below.
6. Include proper docstrings for each test function explaining what is being tested.

## Output Format
- Output ONLY a single Python code block wrapped in ```python ... ```
- Do NOT include any explanation, comments outside the code, or markdown outside the code block.
"""

GENERATOR_USER_PROMPT = """\
Generate pytest unit tests for the following source file.

<source_file path="{source_path}">
{source_code}
</source_file>

<dependency_files>
{context_files_section}
</dependency_files>
"""


def build_generator_prompt(
    source_path: str,
    source_code: str,
    context_files: dict[str, str],
) -> tuple[str, str]:
    """Build the (system_prompt, user_prompt) pair for the Generator Node."""
    # Build dependency section
    if context_files:
        sections = []
        for filename, content in context_files.items():
            sections.append(f'<file path="{filename}">\n{content}\n</file>')
        context_files_section = "\n".join(sections)
    else:
        context_files_section = "(No dependency files found)"

    user_prompt = GENERATOR_USER_PROMPT.format(
        source_path=source_path,
        source_code=source_code,
        context_files_section=context_files_section,
    )

    return GENERATOR_SYSTEM_PROMPT, user_prompt
