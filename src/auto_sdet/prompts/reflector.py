"""
Prompt templates for the Reflector Node (Chain-of-Thought error analysis).
"""
from __future__ import annotations

from typing import Optional

from auto_sdet.models.schemas import MemoryTrajectory

REFLECTOR_SYSTEM_PROMPT = """\
You are a senior Python debugging expert. Your job is to fix failing pytest test code.

## Chain-of-Thought Process (you MUST follow these steps internally)

### Step 1 — Error Classification
Identify the error type from stderr. Pick ONE category:
- ImportError / ModuleNotFoundError → missing or wrong import
- AssertionError → test logic is wrong (expected vs actual mismatch)
- TypeError / AttributeError → wrong API usage
- SyntaxError → invalid Python syntax in test code
- FixtureError → pytest fixture not found (e.g. `mocker`, missing 3rd-party plugin)
- MockError → wrong patch target / wrong mock library / wrong attribute path
- Other → only when nothing else fits

### Step 2 — Root Cause Analysis
Identify the EXACT line(s) in the test code that caused the failure.
Cross-reference with the source code to verify the correct API surface.

### Step 3 — Fix Strategy Selection
Choose between:
- `minimal_patch` — only a few lines need to change. Prefer this whenever possible.
- `full_rewrite` — multiple structural problems require regenerating substantial portions.

### Step 4 — Output Constraints
The output MUST be a structured object with these fields (the runtime will
validate it against a Pydantic schema, so respect the types):

  - error_classification : one of the categories from Step 1
  - root_cause           : ONE sentence describing why the test failed
  - affected_lines       : the test-file line numbers that change
  - fix_strategy         : `minimal_patch` or `full_rewrite`
  - fixed_code           : the COMPLETE fixed test file as a single string

Rules for `fixed_code`:
- Must be valid, runnable Python
- Must include ALL imports the file needs
- Must preserve every test from the original UNLESS deletion is itself the fix
- Must NOT contain markdown fences, ellipses, or "..." placeholders
- Do NOT remove test cases just to make failures go away
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

<similar_past_cases>
{similar_cases_section}
</similar_past_cases>

Current retry: {retry_count} / {max_retries}

Follow the Chain-of-Thought steps in your system instructions, then emit the structured `ReflectionResult` with all five fields populated.

Note on `<similar_past_cases>`: these are retrieved from episodic memory based
on semantic similarity to the current error. They are HINTS — apply them only
if they actually fit the current case. If they don't, ignore them.
"""


def build_reflector_prompt(
    source_path: str,
    source_code: str,
    test_code: str,
    latest_error: str,
    error_history: list[str],
    retry_count: int,
    max_retries: int,
    similar_cases: Optional[list[MemoryTrajectory]] = None,
) -> tuple[str, str]:
    """Build (system_prompt, user_prompt) for the Reflector Node."""
    if error_history:
        history_entries = []
        for i, err in enumerate(error_history, 1):
            history_entries.append(f"--- Attempt {i} ---\n{err}")
        error_history_section = "\n\n".join(history_entries)
    else:
        error_history_section = "(First attempt — no previous errors)"

    similar_cases_section = _format_similar_cases(similar_cases or [])

    user_prompt = REFLECTOR_USER_PROMPT.format(
        source_path=source_path,
        source_code=source_code,
        test_code=test_code,
        latest_error=latest_error,
        error_history_section=error_history_section,
        similar_cases_section=similar_cases_section,
        retry_count=retry_count,
        max_retries=max_retries,
    )

    return REFLECTOR_SYSTEM_PROMPT, user_prompt


def _format_similar_cases(cases: list[MemoryTrajectory]) -> str:
    """Render retrieved memory trajectories as a few-shot context block."""
    if not cases:
        return "(No similar past cases retrieved — this is a novel error pattern, or the memory store is cold.)"

    blocks = []
    for i, c in enumerate(cases, 1):
        outcome_label = "✓ resolved" if c.outcome == "fix_applied" else "✗ couldn't resolve"
        blocks.append(
            f"--- Past case {i} ({outcome_label}) ---\n"
            f"  classification: {c.error_classification}\n"
            f"  root cause:    {c.root_cause}\n"
            f"  fix strategy:  {c.fix_strategy}\n"
            f"  fix summary:   {c.fix_summary}"
        )
    return "\n\n".join(blocks)
