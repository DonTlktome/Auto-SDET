"""
ruff static-analysis adapter — produces an objective second signal for the
Evaluator alongside the LLM-as-Judge's fuzzy semantic scoring.

Why this exists:
  LLM-as-Judge (Generator + Evaluator running on the same LLM) has a known
  "self-evaluation homogeneity" blind spot — subtle bugs the model commonly
  makes, the same model is bad at scoring. Adding `ruff` gives an independent,
  deterministic signal: syntax errors / unused imports / undefined names /
  obviously broken patterns get caught objectively in <100ms with zero LLM
  cost, fed into the Evaluator prompt as a sidecar fact set.

Boundary:
  ruff catches the OBJECTIVE-STATIC layer. LLM judge still owns the SEMANTIC
  layer (assertion quality / mock correctness / coverage gaming). Together
  they cover what either one misses alone.
"""
from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RuffIssue:
    """One issue reported by ruff."""
    code: str        # rule code, e.g. "F401" (unused import) or "E999" (syntax)
    message: str
    line: int
    column: int
    is_fatal: bool   # True for syntax errors / undefined names — code can't run


# Rule codes that mean "the file won't even parse or run". Treated as
# blocking so the Evaluator can route straight back to Reflector without
# burning a sandbox cycle on guaranteed-broken code.
FATAL_PREFIXES = ("E9",)        # E9xx — syntax errors (older ruff format)
FATAL_CODES = {
    "F821",          # undefined name
    "F811",          # redefined name
    "F823",          # local var referenced before assignment
    "invalid-syntax", # ruff 0.6+ uses this string for parser errors
    "syntax-error",   # alternate spelling some versions use
}


# Ruff CLI flags:
#   --select E,F  — pycodestyle errors + pyflakes (the high-signal core)
#   --output-format=json  — structured output for parsing
#   --stdin-filename       — affects rule selection only, not actual reads
#   -                      — read source from stdin
RUFF_ARGS = [
    "check",
    "--select", "E,F",
    "--output-format", "json",
    "--stdin-filename", "test_generated.py",
    "-",
]


def check_code(code: str, *, timeout: float = 10.0) -> list[RuffIssue]:
    """
    Run ruff against a source string. Returns a list of RuffIssue.

    Failures (ruff not on PATH, subprocess hang, malformed JSON) return an
    empty list rather than raising — static analysis is a sidecar signal,
    not a gate. The Evaluator continues with just the LLM judge if ruff
    can't be reached.
    """
    try:
        result = subprocess.run(
            ["ruff", *RUFF_ARGS],
            input=code,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,             # ruff exits 1 when there are findings — that's normal
            encoding="utf-8",
        )
    except FileNotFoundError:
        logger.warning("ruff not on PATH; skipping static analysis signal")
        return []
    except subprocess.TimeoutExpired:
        logger.warning(f"ruff timed out after {timeout}s; skipping")
        return []
    except Exception as e:
        logger.warning(f"ruff invocation failed: {e}; skipping")
        return []

    if not result.stdout.strip():
        return []

    try:
        raw = json.loads(result.stdout)
    except json.JSONDecodeError as e:
        logger.warning(f"ruff returned malformed JSON: {e}; skipping")
        return []

    issues: list[RuffIssue] = []
    for entry in raw:
        # ruff JSON shape (stable since 0.4): each entry is a dict with
        # "code", "message", "location": {"row", "column"}, "filename", ...
        location = entry.get("location") or {}
        code = entry.get("code") or "UNKNOWN"
        issues.append(RuffIssue(
            code=code,
            message=str(entry.get("message", "")),
            line=int(location.get("row", 0)),
            column=int(location.get("column", 0)),
            is_fatal=_is_fatal(code),
        ))
    return issues


def _is_fatal(code: Optional[str]) -> bool:
    """A rule code is fatal if it means 'this code won't even run'."""
    if not code:
        return False
    if code in FATAL_CODES:
        return True
    return any(code.startswith(prefix) for prefix in FATAL_PREFIXES)


def format_issues_for_prompt(issues: list[RuffIssue], *, max_entries: int = 10) -> str:
    """
    Render issues as a compact bullet list suitable for injection into the
    Evaluator system prompt. Caps at `max_entries` to keep the prompt small.
    """
    if not issues:
        return "(ruff found no static-analysis issues — code parses cleanly with no unused imports / undefined names)"

    fatals = [i for i in issues if i.is_fatal]
    nonfatals = [i for i in issues if not i.is_fatal]

    # Show all fatals first (they're blocking), then top non-fatals.
    shown = fatals[:max_entries] + nonfatals[: max(0, max_entries - len(fatals))]
    lines = []
    for issue in shown:
        tag = "[FATAL]" if issue.is_fatal else "[lint]"
        lines.append(f"  {tag} L{issue.line} {issue.code}: {issue.message}")

    suffix = ""
    omitted = max(0, len(issues) - len(shown))
    if omitted:
        suffix = f"\n  ... and {omitted} more issue(s) omitted"

    summary = (
        f"ruff found {len(fatals)} fatal + {len(nonfatals)} lint issue(s):\n"
        + "\n".join(lines)
        + suffix
    )
    return summary