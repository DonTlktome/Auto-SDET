"""
Prompt templates for the Reflector Node (Chain-of-Thought error analysis).
"""

REFLECTOR_SYSTEM_PROMPT = """\
You are a senior Python debugging expert. Your job is to fix failing pytest test code.

## Chain-of-Thought Process (you MUST follow these steps)

### Step 1 — Error Classification
Identify the error type from stderr:
- ImportError / ModuleNotFoundError → missing or wrong import
- SyntaxError → invalid Python syntax in test code
- AssertionError → test logic is wrong (expected vs actual mismatch)
- TypeError / AttributeError → wrong API usage or mock setup
- Other → describe the error category

### Step 2 — Root Cause Analysis
Pinpoint the EXACT line(s) in the test code that caused the failure.
Cross-reference with the source code to verify the correct API.

### Step 3 — Minimal Fix Strategy
Apply the SMALLEST possible change to fix the issue:
- Do NOT remove test cases to "fix" failures
- Do NOT change the testing logic, only fix technical bugs
- Prefer fixing imports, mock targets, and assertion values

### Step 4 — Output (STRICT — non-negotiable)
- Your response MUST start with ```python and end with ```.
- The code block MUST be the COMPLETE fixed test file (every line, every test).
- Do NOT output partial patches, diffs, ellipses, or "rest unchanged" comments.
- Do NOT add narration, summaries, or "Now I will fix..." preambles before the code block.
- If the file is large, prioritize completeness over verbosity — keep tests concise but include them all.
- Violating this format truncates the output and breaks the pipeline.
"""

REFLECTOR_USER_PROMPT = """\
The test code below is failing. Fix it based on the error output.

<source_file path="{source_path}">
{source_code}
</source_file>

<current_test_code>
{test_code}
</current_test_code>

<latest_error>
{latest_error}
</latest_error>

<error_history>
{error_history_section}
</error_history>

Current retry: {retry_count} / {max_retries}

Follow the Chain-of-Thought steps in your system instructions, then output the COMPLETE fixed test file.
"""


def build_reflector_prompt(
    source_path: str,
    source_code: str,
    test_code: str,
    latest_error: str,
    error_history: list[str],
    retry_count: int,
    max_retries: int,
) -> tuple[str, str]:
    """Build (system_prompt, user_prompt) for the Reflector Node."""
    if error_history:
        history_entries = []
        for i, err in enumerate(error_history, 1):
            history_entries.append(f"--- Attempt {i} ---\n{err}")
        error_history_section = "\n\n".join(history_entries)
    else:
        error_history_section = "(First attempt — no previous errors)"

    user_prompt = REFLECTOR_USER_PROMPT.format(
        source_path=source_path,
        source_code=source_code,
        test_code=test_code,
        latest_error=latest_error,
        error_history_section=error_history_section,
        retry_count=retry_count,
        max_retries=max_retries,
    )

    return REFLECTOR_SYSTEM_PROMPT, user_prompt
