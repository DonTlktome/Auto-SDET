"""
Benchmark script — runs Auto-SDET against a fixed list of modules
and reports one-shot pass rate, final pass rate, retry distribution,
average duration, and average coverage.

Usage:
    python scripts/benchmark.py
    python scripts/benchmark.py --bench-dir D:/path/to/algorithms-bench
    python scripts/benchmark.py --output results.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from statistics import mean

from dotenv import load_dotenv
load_dotenv()

from rich.console import Console
from rich.table import Table

from auto_sdet.graph.graph import run_agent

console = Console()

# ── Default 20 benchmark targets (relative to --bench-dir) ──
DEFAULT_TARGETS = [
    "maths/prime_check.py",
    "maths/fibonacci.py",
    "maths/factorial.py",
    "maths/gcd_of_n_numbers.py",
    "maths/sigmoid.py",
    "sorts/bubble_sort.py",
    "sorts/quick_sort.py",
    "sorts/merge_sort.py",
    "sorts/insertion_sort.py",
    "searches/binary_search.py",
    "searches/linear_search.py",
    "strings/palindrome.py",
    "strings/reverse_words.py",
    "strings/word_occurrence.py",
    "data_structures/stacks/stack.py",
    "data_structures/queues/queue_by_list.py",
    "data_structures/linked_list/singly_linked_list.py",
    "conversions/decimal_to_binary.py",
    "conversions/celsius_to_fahrenheit.py",
    "dynamic_programming/fibonacci.py",
]


def _memory_size_safe() -> int | None:
    """Read live trajectory count without crashing if the store is unavailable."""
    try:
        from auto_sdet.tools.memory_store import MemoryStore
        return MemoryStore().count()
    except Exception:
        return None


def run_one(target: Path, output_dir: Path, max_retries: int) -> dict:
    """Run auto-sdet on one file and return a metrics dict."""
    mem_before = _memory_size_safe()
    start = time.time()
    try:
        final_state = run_agent(
            target_path=target,
            output_path=output_dir / f"test_{target.name}",
            max_retries=max_retries,
            verbose=False,
        )
        wall_time = time.time() - start
        er = final_state.get("execution_result")
        retry_count = final_state.get("retry_count", 0)
        eval_rejects = final_state.get("evaluator_reject_count", 0)
        mem_after = _memory_size_safe()
        return {
            "file": str(target.name),
            "status": final_state.get("status"),
            "retry_count": retry_count,
            # Breakdown of the retry_count: how many were Evaluator pushbacks vs
            # real pytest failures. Lets us tell "Evaluator over-rejected" apart
            # from "Reflector failed to fix a real bug" in the per-file table.
            "evaluator_rejects": eval_rejects,
            "executor_retries": max(retry_count - eval_rejects, 0),
            "exit_code": er.exit_code if er else -1,
            "coverage_pct": er.coverage_pct if er else None,
            "wall_time_s": round(wall_time, 1),
            # Memory growth in this run — directly visible CRUD activity.
            "memory_size_before": mem_before,
            "memory_size_after": mem_after,
            "memory_delta": (mem_after - mem_before) if (mem_before is not None and mem_after is not None) else None,
            "error": None,
        }
    except Exception as e:
        return {
            "file": str(target.name),
            "status": "error",
            "retry_count": 0,
            "evaluator_rejects": 0,
            "executor_retries": 0,
            "exit_code": -1,
            "coverage_pct": None,
            "wall_time_s": round(time.time() - start, 1),
            "memory_size_before": mem_before,
            "memory_size_after": _memory_size_safe(),
            "memory_delta": None,
            "error": str(e)[:200],
        }


def summarize(results: list[dict]) -> dict:
    n = len(results)
    if n == 0:
        return {}

    done = [r for r in results if r["status"] == "done"]
    one_shot = [r for r in done if r["retry_count"] == 0]
    coverages = [r["coverage_pct"] for r in done if r["coverage_pct"] is not None]
    times = [r["wall_time_s"] for r in results]
    eval_rejects = [r.get("evaluator_rejects", 0) for r in results]
    exec_retries = [r.get("executor_retries", 0) for r in results]

    # Snapshot memory store endpoints (first available "before" / last "after")
    mem_befores = [r.get("memory_size_before") for r in results if r.get("memory_size_before") is not None]
    mem_afters = [r.get("memory_size_after") for r in results if r.get("memory_size_after") is not None]
    mem_initial = mem_befores[0] if mem_befores else None
    mem_final = mem_afters[-1] if mem_afters else None

    return {
        "total": n,
        "one_shot_pass_rate": round(len(one_shot) / n * 100, 1),
        "final_pass_rate": round(len(done) / n * 100, 1),
        "avg_retries": round(mean(r["retry_count"] for r in results), 2),
        # Decomposed view of retries: Evaluator pushbacks vs real Executor failures
        "avg_evaluator_rejects": round(mean(eval_rejects), 2),
        "avg_executor_retries": round(mean(exec_retries), 2),
        "total_evaluator_rejects": sum(eval_rejects),
        # Memory store growth across the run — directly observable CRUD activity
        "memory_initial": mem_initial,
        "memory_final": mem_final,
        "memory_net_growth": (mem_final - mem_initial) if (mem_initial is not None and mem_final is not None) else None,
        "avg_coverage_pct": round(mean(coverages), 1) if coverages else None,
        "avg_wall_time_s": round(mean(times), 1),
    }


def print_table(results: list[dict]) -> None:
    table = Table(title="Per-file results")
    table.add_column("File", style="cyan")
    table.add_column("Status")
    table.add_column("Retries", justify="right")
    table.add_column("EvalRej", justify="right")
    table.add_column("ExecRetry", justify="right")
    table.add_column("MemΔ", justify="right")
    table.add_column("Coverage", justify="right")
    table.add_column("Time(s)", justify="right")

    for r in results:
        status_color = {
            "done": "[green]done[/]",
            "failed": "[red]failed[/]",
            "error": "[yellow]error[/]",
        }.get(r["status"], r["status"])
        cov = f"{r['coverage_pct']}%" if r["coverage_pct"] is not None else "-"
        mem_delta = r.get("memory_delta")
        mem_delta_str = f"+{mem_delta}" if mem_delta and mem_delta > 0 else (str(mem_delta) if mem_delta is not None else "-")
        table.add_row(
            r["file"],
            status_color,
            str(r["retry_count"]),
            str(r.get("evaluator_rejects", 0)),
            str(r.get("executor_retries", 0)),
            mem_delta_str,
            cov,
            str(r["wall_time_s"]),
        )
    console.print(table)


def print_summary(summary: dict) -> None:
    table = Table(title="Aggregate summary", show_header=False)
    table.add_column("Metric", style="bold")
    table.add_column("Value")

    table.add_row("Total modules", str(summary["total"]))
    table.add_row("One-shot pass rate", f"[green]{summary['one_shot_pass_rate']}%[/]")
    table.add_row("Final pass rate (after self-healing)", f"[green]{summary['final_pass_rate']}%[/]")
    table.add_row("Avg retries (total)", str(summary["avg_retries"]))
    table.add_row("  ├─ avg Evaluator rejects", str(summary.get("avg_evaluator_rejects", 0)))
    table.add_row("  └─ avg real Executor retries", str(summary.get("avg_executor_retries", 0)))
    table.add_row("Total Evaluator rejects across run", str(summary.get("total_evaluator_rejects", 0)))
    if summary.get("memory_initial") is not None:
        table.add_row(
            "Memory store size (initial → final)",
            f"{summary['memory_initial']} → {summary['memory_final']}  (Δ {summary['memory_net_growth']:+d})",
        )
    avg_cov = summary["avg_coverage_pct"]
    table.add_row("Avg coverage", f"{avg_cov}%" if avg_cov is not None else "-")
    table.add_row("Avg wall time per module", f"{summary['avg_wall_time_s']}s")
    console.print(table)


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto-SDET benchmark runner")
    parser.add_argument(
        "--bench-dir",
        type=Path,
        default=Path("D:/Documents/Study/Projects/algorithms-bench"),
        help="Root directory containing benchmark target files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark_results/tests"),
        help="Where to save generated test files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark_results/results.json"),
        help="JSON file to save raw results",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Max self-healing retries per file",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run only the first N targets (for quick testing)",
    )
    args = parser.parse_args()

    if not args.bench_dir.exists():
        console.print(f"[bold red]Error:[/] bench dir does not exist: {args.bench_dir}")
        console.print("[dim]Hint: clone https://github.com/TheAlgorithms/Python to that path[/]")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    targets = DEFAULT_TARGETS[: args.limit] if args.limit else DEFAULT_TARGETS

    console.print(f"\n[bold]Running benchmark on {len(targets)} modules[/]\n")
    results = []

    for i, rel_path in enumerate(targets, 1):
        target = args.bench_dir / rel_path
        if not target.exists():
            console.print(f"[yellow]({i}/{len(targets)}) SKIP {rel_path} (not found)[/]")
            continue

        console.print(f"\n[bold cyan]({i}/{len(targets)}) {rel_path}[/]")
        result = run_one(target, args.output_dir, args.max_retries)
        results.append(result)

        # Persist after each run so partial progress is saved + summary is
        # always queryable mid-run for live monitoring.
        partial = {"summary": summarize(results), "results": results}
        args.output.write_text(json.dumps(partial, indent=2, ensure_ascii=False), encoding="utf-8")

    console.print("\n")
    print_table(results)
    summary = summarize(results)
    if summary:
        console.print("\n")
        print_summary(summary)
        # Append summary to JSON
        full = {"summary": summary, "results": results}
        args.output.write_text(json.dumps(full, indent=2, ensure_ascii=False), encoding="utf-8")
        console.print(f"\n[bold green]Results saved to:[/] {args.output}")


if __name__ == "__main__":
    main()
