"""
Counterfactual check for the one mechanism where relevant memory hits occur:
"module location assumptions" (Windows absolute path, dotted package path in
import / patch / runpy, repo-layout file path). For every Reflector attempt
whose diagnosis is in that class, did the next pytest run pass?
Compared across A (legacy memory), B (no memory), C (signature memory).

Usage (repo root): python scripts/memory_counterfactual.py
"""
import json
import re
from pathlib import Path

KEYWORDS = re.compile(
    r"Windows|absolute path|hard-?codes? .*path|dotted|run_module|runpy|import_module\(|"
    r"patch target|string patch|resolve_name|not importable|ModuleNotFoundError|No module named|"
    r"FileNotFoundError|SOURCE_PATH|source path|Path\(__file__\)|spec_from_file_location",
    re.IGNORECASE,
)
root = Path("benchmark_results/flash/runs")
for config in "ABC":
    total = fixed = 0
    with_relevant_mem = []
    for i in (1, 2, 3):
        events = [json.loads(line) for line in (root / f"2026-10-30_{config}_r{i}" / "trace.jsonl")
                  .read_text(encoding="utf-8").splitlines()]
        execs = {(e["module"], e["retry"]): e for e in events if e["kind"] == "execution"}
        for e in events:
            if e["kind"] != "reflection" or not e.get("root_cause"):
                continue
            if not KEYWORDS.search(e["root_cause"]):
                continue
            nxt = execs.get((e["module"], e["retry"]))
            ok = bool(nxt and nxt["passed"])
            total += 1
            fixed += ok
            mem_hit = any(KEYWORDS.search(m["root_cause"]) for m in e.get("injected") or []
                          if m["source_module"] != e["module"])
            with_relevant_mem.append((ok, mem_hit))
    hit = [ok for ok, m in with_relevant_mem if m]
    miss = [ok for ok, m in with_relevant_mem if not m]
    print(f"{config}: module-location attempts={total}, fixed by next run={fixed} ({fixed / total:.0%})"
          + (f" | with same-class cross-module memory injected: {sum(hit)}/{len(hit)}"
             f" | without: {sum(miss)}/{len(miss)}" if config != "B" else ""))
