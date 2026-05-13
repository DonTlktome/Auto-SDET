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

Respond with a SINGLE JSON object (no markdown fence, no prose before/after)
matching this schema exactly:

```
{
  "error_classification": <one of: "ImportError" | "AssertionError" |
                          "TypeError" | "AttributeError" | "SyntaxError" |
                          "FixtureError" | "MockError" | "Other">,
  "root_cause":           <ONE sentence describing why the test failed>,
  "affected_lines":       <array of test-file line numbers, e.g. [12, 14]>,
  "fix_strategy":         <"minimal_patch" | "full_rewrite">,
  "fixed_code":           <the COMPLETE fixed test file as a JSON string>
}
```

The runtime parses this JSON and validates it against a Pydantic schema —
unknown values for `error_classification` or `fix_strategy` will be rejected.

Rules for `fixed_code`:
- Must be valid, runnable Python
- Must include ALL imports the file needs
- Must preserve every test from the original UNLESS deletion is itself the fix
- Must NOT contain markdown fences, ellipses, or "..." placeholders
- Do NOT remove test cases just to make failures go away
- Escape newlines as \\n when serializing the file into the JSON string
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

Follow the Chain-of-Thought steps in your system instructions, then emit the JSON object with all five fields populated.

Note on `<similar_past_cases>`: these are retrieved from episodic memory based
on semantic similarity to the current error. They are HINTS — apply them only
if they actually fit the current case. If they don't, ignore them.
"""


# Sliding-window cap on how many prior errors get injected into the prompt.
# Rationale: pytest stderr is ~2-4K tokens per attempt, mostly header noise.
# After 3 retries the history section can balloon to 10K+ tokens, and the
# Reflector mostly needs the LATEST error — older ones are only there to
# detect A→B→A oscillation, for which 2 entries is plenty. State still holds
# the full history (preserves LangSmith trace fidelity); we slice only when
# rendering the prompt.
ERROR_HISTORY_PROMPT_WINDOW = 2


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
    recent_errors = error_history[-ERROR_HISTORY_PROMPT_WINDOW:] if error_history else []
    if recent_errors:
        history_entries = []
        # Use absolute attempt numbers (counting from the start of the run)
        # so the LLM can see retry depth even when older entries are clipped.
        start_idx = max(0, len(error_history) - len(recent_errors))
        for offset, err in enumerate(recent_errors):
            attempt_num = start_idx + offset + 1
            history_entries.append(f"--- Attempt {attempt_num} ---\n{err}")
        clipped_note = (
            f"\n\n[Note: showing last {len(recent_errors)} of {len(error_history)} attempts; "
            f"earlier ones omitted to keep prompt focused]"
            if len(error_history) > len(recent_errors)
            else ""
        )
        error_history_section = "\n\n".join(history_entries) + clipped_note
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
