"""
Combiner Node — merges per-symbol partial test files (produced by parallel
generator_worker invocations) into a single coherent pytest file.

Responsibilities:
  1. Deduplicate top-level imports across workers
  2. Concatenate test bodies (classes / functions) preserving each worker's contribution
  3. Strip empty / accidental partials

The combiner does NO LLM call — it's a pure-string/AST utility node.
"""
from __future__ import annotations

import logging

from rich.console import Console

from auto_sdet.models.schemas import AgentState

logger = logging.getLogger(__name__)
console = Console()


def combiner_node(state: AgentState) -> dict:
    """Merge `partial_test_codes` into a single `test_code` string."""
    partial_codes = state.get("partial_test_codes", [])
    console.print(
        f"[bold blue]🪡 [Combiner][/]  Merging {len(partial_codes)} partial test file(s)..."
    )

    combined = combine_test_files(partial_codes)
    line_count = len(combined.splitlines())

    console.print(
        f"[bold green]✓ [Combiner][/]  Combined into {line_count} lines"
    )

    return {
        "test_code": combined,
        "status": "executing",
    }


# ──────────────────────────────────────────────────────────────────
# Pure helper (testable in isolation)
# ──────────────────────────────────────────────────────────────────

def combine_test_files(partial_codes: list[str]) -> str:
    """
    Merge multiple per-symbol pytest files into one.

    Strategy:
      - Walk each partial top-down
      - Collect leading import / from lines into a deduplicated ordered set
      - Everything from the first non-import statement onward is the "body"
      - Final layout: <imports sorted by first-seen> + blank line + bodies separated by blank lines

    Limitations (acceptable for v1):
      - Multi-line `from x import (a, b, c)` is not split — kept as a single block
      - Indented imports (inside functions) are left in place
      - Conflicting fixture / class names are not auto-renamed
    """
    if not partial_codes:
        return ""

    seen_imports: list[str] = []   # preserve first-seen order
    seen_set: set[str] = set()
    bodies: list[str] = []

    for code in partial_codes:
        if not code.strip():
            continue
        imports, body = _split_imports_and_body(code)
        for imp in imports:
            key = _normalize_import(imp)
            if key not in seen_set:
                seen_set.add(key)
                seen_imports.append(imp)
        if body.strip():
            bodies.append(body.rstrip())

    if not seen_imports and not bodies:
        return ""

    parts: list[str] = []
    if seen_imports:
        parts.append("\n".join(seen_imports))
    if bodies:
        parts.append("\n\n\n".join(bodies))

    return "\n\n\n".join(parts) + "\n"


def _split_imports_and_body(code: str) -> tuple[list[str], str]:
    """
    Split a Python source string into top-level import lines and "the rest".

    Walks line-by-line; once a non-import, non-blank, non-comment top-level
    line is seen, the remainder is treated as body. Indentation matters
    here — only column-0 imports are hoisted.
    """
    lines = code.split("\n")
    imports: list[str] = []
    body_start_idx = 0

    for i, line in enumerate(lines):
        stripped = line.lstrip()
        # Only consider unindented imports (those at module top level)
        if line and not line[0].isspace() and (
            stripped.startswith("import ") or stripped.startswith("from ")
        ):
            imports.append(line.rstrip())
            continue
        if stripped == "" or stripped.startswith("#"):
            # Skip blank / comment lines while still in the import header
            continue
        # First "real" line — body starts here.
        body_start_idx = i
        break
    else:
        # File was 100% imports / blanks
        body_start_idx = len(lines)

    body = "\n".join(lines[body_start_idx:])
    return imports, body


def _normalize_import(imp: str) -> str:
    """Whitespace-normalize an import line for deduplication."""
    return " ".join(imp.split())