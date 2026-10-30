"""
Normalized error signatures — the shared representation for memory
documents AND retrieval queries in the `signature` memory variant.

The legacy variant embedded "<classification>: <root_cause first sentence>
[in file]" as the document but queried with the first 300 chars of pytest
output (mostly the session header), so the two sides never lived in the same
space. Here both sides are produced by `extract_error_signature()` from the
raw failure output:

  pytest      → messages of FAILED / ERROR summary lines, plus the first and
                last `E ` line of each failure block (exception type + the
                tail of the traceback)
  ruff fatal  → "ruff <code>: <message>" per distinct finding
  evaluator   → "evaluator: <actionable weakness>"
  timeout     → a fixed "Timeout" signature

Normalization removes what is specific to one module or one run (test ids,
paths, line numbers, numeric literals, object addresses) so that the same
failure mechanism in two different modules maps to similar text.
"""
from __future__ import annotations

import re

MAX_LINES = 6
MAX_LINE_CHARS = 160

_BLOCK_HEADER = re.compile(r"^_{3,} .+ _{3,}$")
_SUMMARY_LINE = re.compile(r"^(FAILED|ERROR) (\S+)(?: - (.*))?$")
_RUFF_LINE = re.compile(r"^\s*L\d+ ([\w-]+): (.*)$")
_WEAKNESS_LINE = re.compile(r"^\s*\[(?:minor|moderate|blocking)\] (.*)$")

_TEST_ID = re.compile(r"\btest_[\w]*\.py(?:::[\w\[\]\-.,= ]+)?")
_PATH = re.compile(r"(?:/[\w.\-]+)+/")
_LOCATION = re.compile(r"\b[\w.\-]+\.py:\d+(?::\d+)?:?")
_HEX = re.compile(r"0x[0-9a-fA-F]+")
_NUMBER = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?(?:e-?\d+)?(?![\w])")
_SPACE = re.compile(r"\s+")


def normalize_line(line: str) -> str:
    """Strip run/module-specific detail from one error line."""
    line = _TEST_ID.sub("<test>", line)
    line = _LOCATION.sub("", line)
    line = _PATH.sub("", line)
    line = _HEX.sub("0x?", line)
    line = _NUMBER.sub("<n>", line)
    line = _SPACE.sub(" ", line).strip(" :")
    return line[:MAX_LINE_CHARS]


def _dedupe(lines: list[str]) -> list[str]:
    seen, out = set(), []
    for line in lines:
        if line and line not in seen:
            seen.add(line)
            out.append(line)
    return out


def _pytest_lines(output: str) -> list[str]:
    lines = output.splitlines()
    summary: list[str] = []
    block_e: list[list[str]] = []
    current: list[str] | None = None

    for raw in lines:
        stripped = raw.strip()
        if _BLOCK_HEADER.match(stripped):
            current = []
            block_e.append(current)
            continue
        if stripped.startswith("E ") or stripped == "E":
            if current is None:
                current = []
                block_e.append(current)
            text = stripped[1:].strip()
            if text:
                current.append(text)
            continue
        m = _SUMMARY_LINE.match(stripped)
        if m and m.group(3):
            summary.append(m.group(3))

    picked: list[str] = list(summary)
    for e_lines in block_e:
        if e_lines:
            picked.append(e_lines[0])
            if len(e_lines) > 1:
                picked.append(e_lines[-1])
    return picked


def extract_error_signature(output: str) -> str:
    """Build the normalized signature of a failure output (pytest / ruff / evaluator)."""
    output = output or ""

    if "STATIC ANALYSIS (ruff)" in output:
        picked = [f"ruff {m.group(1)}: {m.group(2)}" for m in map(_RUFF_LINE.match, output.splitlines()) if m]
    elif "EVALUATOR REJECTION" in output:
        picked = [f"evaluator: {m.group(1)}" for m in map(_WEAKNESS_LINE.match, output.splitlines()) if m]
    elif "pytest did not finish within" in output:
        picked = ["Timeout: pytest did not finish (a test is hanging: infinite loop, runaway recursion or input())"]
    else:
        picked = _pytest_lines(output)

    if not picked:
        # Unknown shape: fall back to the tail, skipping coverage-table noise.
        tail = [ln for ln in output.splitlines() if ln.strip() and "%" not in ln and not set(ln.strip()) <= set("=-_ ")]
        picked = tail[-3:]

    return "\n".join(_dedupe([normalize_line(p) for p in picked])[:MAX_LINES])
