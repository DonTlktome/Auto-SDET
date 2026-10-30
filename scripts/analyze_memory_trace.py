"""
Analyze memory retrieval traces: did a memory from module X help fix module Y?

For every Reflector call (`reflection` event) the trace records which memories
were injected into the prompt and which module each came from. The execution
event with the same (module, retry) tells whether that fix made pytest pass.

A "cross-module hit that was followed by a fix" is a reflection where
  - at least one injected memory came from a different module, and
  - the following pytest run passed.
That is a necessary condition for cross-module reuse, not proof that the
memory caused the fix — the case listing keeps the evidence (error, injected
memory, diff) so each case can be judged by reading it.

Usage:
    python scripts/analyze_memory_trace.py benchmark_results/runs/2026-10-30_A_r1
    python scripts/analyze_memory_trace.py benchmark_results/runs/*_A_r* --out benchmark_results/<name>.json
"""
from __future__ import annotations

import argparse
import difflib
import json
from collections import Counter
from pathlib import Path
from statistics import mean, median


def load_events(run_dir: Path) -> list[dict]:
    path = run_dir / "trace.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def unified_diff(before: str | None, after: str | None) -> str:
    return "".join(difflib.unified_diff(
        (before or "").splitlines(keepends=True), (after or "").splitlines(keepends=True),
        fromfile="before", tofile="after", n=2,
    ))


def analyze_run(run_dir: Path) -> dict:
    events = load_events(run_dir)
    executions = {(e["module"], e["retry"]): e for e in events if e["kind"] == "execution"}
    reflections = [e for e in events if e["kind"] == "reflection"]
    memory_ops = [e for e in events if e["kind"] == "memory_op"]

    rows = []
    for r in reflections:
        nxt = executions.get((r["module"], r["retry"]))
        injected = r.get("injected") or []
        sims = {c["id"]: c["similarity"] for c in r.get("candidates") or []}
        cross = [m for m in injected if (m.get("source_module") or m.get("target_file")) != r["module"]]
        rows.append({
            "module": r["module"],
            "retry": r["retry"],
            "trigger": r.get("trigger"),
            "fix_produced": r.get("fix_produced", False),
            "fixed": bool(nxt and nxt.get("passed")),
            "next_execution_found": nxt is not None,
            "n_candidates": len(r.get("candidates") or []),
            "n_injected": len(injected),
            "n_cross_module": len(cross),
            "injected_similarities": [sims.get(m["trajectory_id"]) for m in injected],
            "event": r,
            "next_execution": nxt,
            "cross": cross,
        })

    def _rate(subset: list[dict]) -> dict:
        return {"n": len(subset), "fixed": sum(x["fixed"] for x in subset),
                "fix_rate": round(sum(x["fixed"] for x in subset) / len(subset), 3) if subset else None}

    with_memory = [x for x in rows if x["event"].get("memory_enabled")]
    kept_sims_same, kept_sims_cross = [], []
    for x in with_memory:
        for m, s in zip(x["event"].get("injected") or [], x["injected_similarities"]):
            if s is None:
                continue
            ((kept_sims_cross if m in x["cross"] else kept_sims_same)).append(s)

    cases = []
    for x in rows:
        if not (x["n_cross_module"] and x["fixed"]):
            continue
        ev = x["event"]
        cases.append({
            "module": x["module"],
            "retry": x["retry"],
            "trigger": x["trigger"],
            "query": ev.get("query"),
            "latest_error": ev.get("latest_error"),
            "current_diagnosis": {"classification": ev.get("classification"), "root_cause": ev.get("root_cause"),
                                  "fix_strategy": ev.get("fix_strategy")},
            "injected": [
                {**m, "similarity": dict(zip([i["trajectory_id"] for i in ev.get("injected") or []],
                                              x["injected_similarities"])).get(m["trajectory_id"]),
                 "cross_module": m in x["cross"]}
                for m in ev.get("injected") or []
            ],
            "diff": unified_diff(ev.get("test_before"), ev.get("test_after")),
        })

    def _dist(values: list[float]) -> dict | None:
        if not values:
            return None
        return {"n": len(values), "min": min(values), "median": round(median(values), 4),
                "mean": round(mean(values), 4), "max": max(values)}

    return {
        "run_dir": run_dir.as_posix(),
        "reflections": len(rows),
        "by_trigger": dict(Counter(x["trigger"] for x in rows)),
        "memory_ops": dict(Counter(e["operation"] for e in memory_ops)),
        "with_memory_enabled": len(with_memory),
        "fix_rate_with_any_injected": _rate([x for x in with_memory if x["n_injected"]]),
        "fix_rate_with_cross_module_injected": _rate([x for x in with_memory if x["n_cross_module"]]),
        "fix_rate_with_none_injected": _rate([x for x in with_memory if not x["n_injected"]]),
        "fix_rate_overall": _rate(rows),
        "injected_similarity_same_module": _dist(kept_sims_same),
        "injected_similarity_cross_module": _dist(kept_sims_cross),
        "cross_module_hit_then_fixed": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Auto-SDET memory retrieval traces")
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=None, help="Write the full analysis JSON here")
    args = parser.parse_args()

    report = [analyze_run(d) for d in args.run_dirs]
    for r in report:
        print(f"\n== {r['run_dir']}")
        print(f"  reflections: {r['reflections']}  by trigger: {r['by_trigger']}  memory ops: {r['memory_ops']}")
        print(f"  fix rate | any injected: {r['fix_rate_with_any_injected']}")
        print(f"           | cross-module injected: {r['fix_rate_with_cross_module_injected']}")
        print(f"           | none injected: {r['fix_rate_with_none_injected']}")
        print(f"  injected similarity same-module: {r['injected_similarity_same_module']}")
        print(f"  injected similarity cross-module: {r['injected_similarity_cross_module']}")
        print(f"  cross-module hit → fixed: {len(r['cross_module_hit_then_fixed'])} case(s)")
        for c in r["cross_module_hit_then_fixed"]:
            srcs = [(m["source_module"], m["similarity"]) for m in c["injected"] if m["cross_module"]]
            print(f"    - {c['module']} retry {c['retry']} ({c['trigger']}): from {srcs}")
    if args.out:
        args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
