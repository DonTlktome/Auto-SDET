"""
AST-based extraction of top-level public symbols (functions and classes).

Used by the parallel test-generation flow: each extracted SymbolSpec
becomes one fan-out target, letting the LLM generate tests for that
symbol in parallel with siblings.
"""
from __future__ import annotations

import ast

from auto_sdet.models.schemas import SymbolSpec


# ──────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────

def extract_top_level_symbols(source_code: str) -> list[SymbolSpec]:
    """
    Parse `source_code` and return all top-level public functions and classes.

    Rules:
    - Only symbols at module top level are included (no nested defs)
    - Symbols whose name starts with `_` are treated as private and skipped
    - Decorators are included in the captured source
    - Syntax errors return an empty list (caller decides what to do)

    The returned `source` field preserves the original text (comments,
    spacing) by line-slicing the input rather than re-printing the AST.
    """
    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        return []

    lines = source_code.splitlines(keepends=True)
    symbols: list[SymbolSpec] = []

    for node in tree.body:  # iterate only top-level nodes
        if isinstance(node, ast.FunctionDef):
            if _is_private(node.name):
                continue
            symbols.append(_build_function_spec(node, lines))
        elif isinstance(node, ast.ClassDef):
            if _is_private(node.name):
                continue
            symbols.append(_build_class_spec(node, lines))
        # All other top-level statements (imports, assignments, if-blocks)
        # are intentionally ignored — they're not testable units.

    return symbols


def build_signature_index(
    symbols: list[SymbolSpec],
    exclude_name: str | None = None,
) -> str:
    """
    Render a context block listing OTHER symbols' signatures (no bodies).

    Given the worker is generating tests for `exclude_name`, this provides
    awareness of what sibling symbols exist so the worker can write
    cross-implementation consistency tests without seeing full bodies.

    Returns an empty string when no other symbols exist.
    """
    others = [s for s in symbols if s.name != exclude_name]
    if not others:
        return ""
    return "\n\n".join(s.signature for s in others)


# ──────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────

def _is_private(name: str) -> bool:
    """Names starting with a single underscore are treated as private."""
    return name.startswith("_")


def _slice_source(node: ast.AST, lines: list[str]) -> str:
    """
    Extract the original source text for `node`, including any decorators.

    `node.lineno` is 1-indexed and points to the `def`/`class` line, but
    decorators sit on earlier lines. We back up to the earliest decorator.
    """
    start_line = node.lineno - 1  # to 0-indexed
    decorators = getattr(node, "decorator_list", [])
    if decorators:
        start_line = min(d.lineno - 1 for d in decorators)
    end_line = node.end_lineno  # already exclusive when used as slice end
    return "".join(lines[start_line:end_line])


def _docstring_summary(node: ast.AST) -> str | None:
    """First non-empty line of the docstring, stripped. None if no docstring."""
    raw = ast.get_docstring(node)
    if not raw:
        return None
    for line in raw.split("\n"):
        line = line.strip()
        if line:
            return line
    return None


def _build_function_spec(node: ast.FunctionDef, lines: list[str]) -> SymbolSpec:
    return SymbolSpec(
        name=node.name,
        kind="function",
        source=_slice_source(node, lines),
        signature=_format_function_signature(node),
        docstring=_docstring_summary(node),
    )


def _build_class_spec(node: ast.ClassDef, lines: list[str]) -> SymbolSpec:
    return SymbolSpec(
        name=node.name,
        kind="class",
        source=_slice_source(node, lines),
        signature=_format_class_signature(node),
        docstring=_docstring_summary(node),
    )


def _format_function_signature(node: ast.FunctionDef) -> str:
    """
    Build a string like `def name(args) -> ret:` plus an indented one-line
    docstring summary if present. No function body.
    """
    # Reconstruct a stripped-body FunctionDef and unparse to get the signature.
    # We strip decorators here too — signature index is about the API shape,
    # not how callers wire it up.
    stripped = ast.FunctionDef(
        name=node.name,
        args=node.args,
        body=[ast.Pass()],
        decorator_list=[],
        returns=node.returns,
        type_comment=None,
    )
    ast.fix_missing_locations(stripped)
    sig_line = ast.unparse(stripped).split("\n", 1)[0]   # take only the def line

    summary = _docstring_summary(node)
    if summary:
        return f'{sig_line}\n    """{summary}"""'
    return sig_line


def _format_class_signature(node: ast.ClassDef) -> str:
    """
    Build a class header + docstring summary + signatures of public methods
    (and `__init__` since it defines the construction API).
    """
    bases = ", ".join(ast.unparse(b) for b in node.bases) if node.bases else ""
    header = f"class {node.name}({bases}):" if bases else f"class {node.name}:"

    out = [header]

    summary = _docstring_summary(node)
    if summary:
        out.append(f'    """{summary}"""')

    for item in node.body:
        if not isinstance(item, ast.FunctionDef):
            continue
        # Include public methods + __init__ (the construction contract);
        # skip everything else dunder or private.
        if item.name.startswith("_") and item.name != "__init__":
            continue
        method_sig = _format_function_signature(item)
        # indent every line by 4 spaces to nest under the class
        indented = "\n".join("    " + line for line in method_sig.split("\n"))
        out.append(indented)

    return "\n".join(out)