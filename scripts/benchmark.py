"""
Benchmark script — runs Auto-SDET against a fixed, ordered list of modules
under one named configuration and reports one-shot pass rate, final pass
rate, retry distribution, coverage, time breakdown and infra failures.

Every run writes a self-describing result file (git commit, model, thinking,
memory / ruff switches, retry cap, module order, start / end time) to
`benchmark_results/<date>_<config>_r<n>.json`; existing files are never
overwritten. Per-run artifacts (generated tests, memory store, event trace)
live under `benchmark_results/runs/<date>_<config>_r<n>/`.

Configurations:
    A  final architecture, episodic memory ON (legacy retrieval), empty store
    B  same as A with memory OFF (no retrieval, no writes)
    C  A + memory fixes (signature retrieval, cosine threshold, verified outcome)

Usage:
    python scripts/benchmark.py --config A --run-index 1
    python scripts/benchmark.py --config B --run-index 1 --limit 3     # smoke
    python scripts/benchmark.py --config A --run-index 1 --resume      # after a crash
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

from dotenv import load_dotenv
load_dotenv()

from rich.console import Console
from rich.table import Table

from auto_sdet.graph.graph import run_agent
from auto_sdet.tools.llm_usage import LLMUsageTracker

console = Console()

REPO_ROOT = Path(__file__).resolve().parent.parent

# ── Benchmark targets (relative to --bench-dir), fixed order ──
# 19 modules. `conversions/celsius_to_fahrenheit.py` from the original list
# does not exist in the pinned TheAlgorithms checkout and was always skipped;
# it is removed so the list matches what actually runs. Modules are keyed by
# relative path because two of them share the basename `fibonacci.py`.
BENCH_MODULES = [
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
    "dynamic_programming/fibonacci.py",
]

# Held-out modules for calibrating the signature-memory threshold (config C).
# Same repo and kinds of modules as BENCH_MODULES, none of them in it, all
# stdlib-only with absolute imports. Never used for reported benchmark numbers.
CALIBRATION_MODULES = [
    "maths/perfect_number.py",
    "maths/lucas_series.py",
    "sorts/selection_sort.py",
    "sorts/heap_sort.py",
    "sorts/counting_sort.py",
    "searches/jump_search.py",
    "searches/ternary_search.py",
    "strings/is_pangram.py",
    "conversions/roman_numerals.py",
    "conversions/hexadecimal_to_decimal.py",
    "data_structures/queues/circular_queue.py",
    "data_structures/linked_list/doubly_linked_list.py",
]

# Environment each configuration runs under (read by auto_sdet.config.Settings).
CONFIGS = {
    "A": {
        "description": "5-node FSM + ruff dual-source gate + asymmetric routing + Memory-R1 memory (legacy retrieval), empty store",
        "env": {"MEMORY_ENABLED": "true", "MEMORY_VARIANT": "legacy"},
    },
    "B": {
        "description": "Same as A with episodic memory OFF (no retrieval, no trajectory writes)",
        "env": {"MEMORY_ENABLED": "false", "MEMORY_VARIANT": "legacy"},
    },
    "C": {
        "description": "A + memory fixes: error-signature query/document, cosine threshold, outcome after re-run, module-path keys",
        "env": {"MEMORY_ENABLED": "true", "MEMORY_VARIANT": "signature"},
    },
}


def _memory_size_safe() -> int | None:
    """Read live trajectory count without crashing if the store is unavailable."""
    try:
        from auto_sdet.tools.memory_store import get_memory_store
        store = get_memory_store()
        return store.count() if store else None
    except Exception:
        return None


def _time_breakdown(timings: dict, llm: dict, wall_time: float, memory_flush_s: float) -> dict:
    """
    Per-module time accounting: sandbox create / pip / pytest / LLM.

    `llm_critical_s` excludes the MemoryManager's background-thread calls,
    which overlap other work; `other_s` is the remainder of wall time not
    covered by the critical-path segments (graph overhead, embeddings,
    file upload, infra retry backoff). `memory_flush_s` is the end-of-module
    wait for background memory writes.
    """
    sandbox_create_s = timings.get("sandbox_create_ms", 0) / 1000
    pip_install_s = timings.get("pip_install_ms", 0) / 1000
    pytest_s = timings.get("pytest_ms", 0) / 1000
    llm_total_s = llm.get("time_s", 0.0)
    llm_background_s = llm.get("by_node", {}).get("memory_manager", {}).get("time_s", 0.0)
    llm_critical_s = llm_total_s - llm_background_s
    return {
        "sandbox_create_s": round(sandbox_create_s, 1),
        "pip_install_s": round(pip_install_s, 1),
        "pytest_s": round(pytest_s, 1),
        "llm_total_s": round(llm_total_s, 1),
        "llm_critical_s": round(llm_critical_s, 1),
        "upload_s": round(timings.get("upload_ms", 0) / 1000, 1),
        "infra_backoff_s": round(timings.get("infra_backoff_ms", 0) / 1000, 1),
        "executions": timings.get("executions", 0),
        "memory_flush_s": round(memory_flush_s, 1),
        "other_s": round(wall_time - sandbox_create_s - pip_install_s - pytest_s - llm_critical_s - memory_flush_s, 1),
    }


def run_one(bench_dir: Path, module: str, output_dir: Path, max_retries: int) -> dict:
    """Run auto-sdet on one module (path relative to bench_dir) and return a metrics dict."""
    target = bench_dir / module
    mem_before = _memory_size_safe()
    tracker = LLMUsageTracker()
    start = time.time()
    try:
        final_state = run_agent(
            target_path=target,
            # Path-derived name: maths/fibonacci.py and dynamic_programming/fibonacci.py
            # must not overwrite each other.
            output_path=output_dir / f"test_{module.replace('/', '__')}",
            max_retries=max_retries,
            verbose=False,
            callbacks=[tracker],
            module_id=module,
        )
        wall_time = time.time() - start
        er = final_state.get("execution_result")
        retry_count = final_state.get("retry_count", 0)
        eval_rejects = final_state.get("evaluator_reject_count", 0)
        mem_after = _memory_size_safe()
        llm = tracker.summary()
        return {
            "module": module,
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
            # Sandbox failures retried inside the executor — NOT in retry_count.
            "infra_failures": final_state.get("infra_failures", 0),
            "time_breakdown": _time_breakdown(
                final_state.get("timings") or {}, llm, wall_time, final_state.get("memory_flush_s", 0.0),
            ),
            "memory_unflushed_ops": final_state.get("memory_unflushed_ops", 0),
            # retry_count counts loop iterations; these say what actually happened in them.
            "reflector_attempts": final_state.get("reflector_attempts", 0),
            "reflector_failed_attempts": final_state.get("reflector_failed_attempts", 0),
            "structured_output_failures": final_state.get("structured_output_failures") or {},
            "evaluator_structured_failures": final_state.get("evaluator_structured_failures") or {},
            "llm_usage": llm,
            "error": None,
        }
    except Exception as e:
        return {
            "module": module,
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
            "infra_failures": 0,
            "time_breakdown": None,
            "llm_usage": tracker.summary(),
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

    breakdowns = [r["time_breakdown"] for r in results if r.get("time_breakdown")]
    avg_breakdown = {
        key: round(mean(b[key] for b in breakdowns), 1)
        for key in ("sandbox_create_s", "pip_install_s", "pytest_s", "llm_total_s",
                    "llm_critical_s", "upload_s", "infra_backoff_s", "memory_flush_s", "other_s")
    } if breakdowns else None
    llm_rows = [r["llm_usage"] for r in results if r.get("llm_usage")]

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
        # Sandbox-side failures: retried, never counted as agent retries.
        "total_infra_failures": sum(r.get("infra_failures", 0) for r in results),
        "total_reflector_attempts": sum(r.get("reflector_attempts", 0) for r in results),
        "total_reflector_failed_attempts": sum(r.get("reflector_failed_attempts", 0) for r in results),
        "total_reflector_truncated": sum((r.get("structured_output_failures") or {}).get("truncated", 0) for r in results),
        "total_reflector_parse_errors": sum(
            (r.get("structured_output_failures") or {}).get(k, 0) for r in results for k in ("parse_error", "empty")),
        "total_reflector_recovered": sum(
            (r.get("structured_output_failures") or {}).get("recovered_with_thinking_off", 0) for r in results),
        "total_evaluator_structured_failures": sum(
            sum((r.get("evaluator_structured_failures") or {}).values()) for r in results),
        "infra_failed_modules": sum(1 for r in results if r["status"] == "infra_failed"),
        "avg_time_breakdown_s": avg_breakdown,
        "total_llm_calls": sum(u["calls"] for u in llm_rows),
        "total_prompt_tokens": sum(u["prompt_tokens"] for u in llm_rows),
        "total_completion_tokens": sum(u["completion_tokens"] for u in llm_rows),
        "total_cache_hit_tokens": sum(u["cache_hit_tokens"] for u in llm_rows),
    }


def print_table(results: list[dict]) -> None:
    table = Table(title="Per-module results")
    table.add_column("Module", style="cyan")
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
            "infra_failed": "[yellow]infra_failed[/]",
        }.get(r["status"], r["status"])
        cov = f"{r['coverage_pct']}%" if r["coverage_pct"] is not None else "-"
        mem_delta = r.get("memory_delta")
        mem_delta_str = f"+{mem_delta}" if mem_delta and mem_delta > 0 else (str(mem_delta) if mem_delta is not None else "-")
        table.add_row(
            r.get("module", r["file"]),
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
    table.add_row("Sandbox infra failures (retried)", str(summary.get("total_infra_failures", 0)))
    table.add_row(
        "Reflector attempts (failed) / truncated / parse errors / recovered",
        f"{summary.get('total_reflector_attempts', 0)} ({summary.get('total_reflector_failed_attempts', 0)}) / "
        f"{summary.get('total_reflector_truncated', 0)} / {summary.get('total_reflector_parse_errors', 0)} / "
        f"{summary.get('total_reflector_recovered', 0)}",
    )
    table.add_row("Modules ended by infra failure", str(summary.get("infra_failed_modules", 0)))
    breakdown = summary.get("avg_time_breakdown_s")
    if breakdown:
        table.add_row(
            "Avg time split (create / pip / pytest / LLM)",
            f"{breakdown['sandbox_create_s']}s / {breakdown['pip_install_s']}s / "
            f"{breakdown['pytest_s']}s / {breakdown['llm_total_s']}s",
        )
    table.add_row("LLM calls / prompt / completion tokens",
                  f"{summary.get('total_llm_calls', 0)} / {summary.get('total_prompt_tokens', 0)} / "
                  f"{summary.get('total_completion_tokens', 0)}")
    console.print(table)


def _git(*args: str, cwd: Path = REPO_ROOT) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=cwd, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def _package_versions() -> dict[str, str]:
    versions = {}
    for pkg in ("langgraph", "langchain-core", "langchain-deepseek", "chromadb", "e2b", "e2b-code-interpreter", "ruff"):
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            versions[pkg] = None
    return versions


def build_metadata(args: argparse.Namespace, run_name: str, modules: list[str], memory_dir: Path) -> dict:
    """Everything needed to tie a result file back to the code and settings that produced it."""
    from auto_sdet.config import get_settings
    from auto_sdet.graph.nodes.evaluator import PASS_THRESHOLD, WARN_THRESHOLD
    from auto_sdet.tools import e2b_sandbox
    from auto_sdet.tools.llm_factory import describe_llm_config

    settings = get_settings()
    dirty = [line.split(maxsplit=1)[1] for line in _git("status", "--porcelain", "--untracked-files=no").splitlines() if line.strip()]
    memory = {
        "enabled": settings.memory_enabled,
        "variant": settings.memory_variant if settings.memory_enabled else None,
        "initial_state": "empty" if settings.memory_enabled else None,
        "dir": memory_dir.as_posix() if settings.memory_enabled else None,
    }
    if settings.memory_enabled and settings.memory_variant == "legacy":
        memory["similarity"] = {"metric": "1/(1+squared_l2)", "threshold": 0.4, "top_k": 3}
    elif settings.memory_enabled:
        memory["similarity"] = {"metric": "cosine", "threshold": settings.memory_cosine_threshold, "top_k": 3,
                                "calibration_file": args.calibration_file.as_posix() if args.calibration_file else None}

    return {
        "run_name": run_name,
        "config": args.config,
        "config_description": CONFIGS[args.config]["description"],
        "run_index": args.run_index,
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty_files": dirty,
        "llm": describe_llm_config(),
        "langsmith_tracing": os.environ.get("LANGCHAIN_TRACING_V2", "").lower() == "true",
        "memory": memory,
        "ruff_enabled": True,
        "evaluator": {"pass_threshold": PASS_THRESHOLD, "warn_threshold": WARN_THRESHOLD, "reject_cap": 2},
        "max_retries": args.max_retries,
        "sandbox": {
            "pytest_timeout_s": settings.e2b_sandbox_timeout,
            "sandbox_lifetime_s": settings.e2b_sandbox_timeout + e2b_sandbox.SANDBOX_LIFETIME_MARGIN_S,
            "infra_max_attempts": e2b_sandbox.INFRA_MAX_ATTEMPTS,
            "base_packages": list(e2b_sandbox.BASE_TEST_PACKAGES),
            "template": "e2b code-interpreter default",
        },
        "module_set": args.module_set,
        "module_order": modules,
        "bench_dir": args.bench_dir.as_posix(),
        "bench_commit": _git("rev-parse", "HEAD", cwd=args.bench_dir),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": _package_versions(),
        "note": args.note,
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "finished_at": None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto-SDET benchmark runner")
    parser.add_argument("--config", choices=sorted(CONFIGS), required=True, help="Experiment configuration")
    parser.add_argument("--run-index", type=int, required=True, help="Repeat number (1, 2, 3, ...)")
    parser.add_argument(
        "--bench-dir",
        type=Path,
        default=Path("D:/Documents/Study/Projects/algorithms-bench"),
        help="Root directory containing benchmark target files",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=REPO_ROOT / "benchmark_results",
        help="Where result files and per-run artifacts are written",
    )
    parser.add_argument("--date", default=datetime.now().strftime("%Y-%m-%d"),
                        help="Date prefix of the result file name (default: today)")
    parser.add_argument("--max-retries", type=int, default=3, help="Max self-healing retries per file")
    parser.add_argument("--limit", type=int, default=None, help="Run only the first N modules (smoke test)")
    parser.add_argument("--resume", action="store_true", help="Continue an interrupted run, skipping finished modules")
    parser.add_argument("--note", default="", help="Free-form note stored in metadata (e.g. concurrency)")
    parser.add_argument("--module-set", choices=["benchmark", "calibration"], default="benchmark",
                        help="calibration = held-out modules for threshold calibration (not benchmark numbers)")
    parser.add_argument("--cosine-threshold", type=float, default=None,
                        help="Config C only: calibrated cosine threshold for signature memory")
    parser.add_argument("--calibration-file", type=Path, default=None,
                        help="Config C only: calibration report the threshold comes from")
    args = parser.parse_args()

    if not args.bench_dir.exists():
        console.print(f"[bold red]Error:[/] bench dir does not exist: {args.bench_dir}")
        console.print("[dim]Hint: clone https://github.com/TheAlgorithms/Python to that path[/]")
        sys.exit(1)

    if CONFIGS[args.config]["env"]["MEMORY_VARIANT"] == "signature":
        if args.cosine_threshold is None or args.calibration_file is None or not args.calibration_file.exists():
            console.print("[bold red]Error:[/] config C needs --cosine-threshold and an existing --calibration-file")
            sys.exit(1)
        os.environ["MEMORY_COSINE_THRESHOLD"] = str(args.cosine_threshold)

    module_list = CALIBRATION_MODULES if args.module_set == "calibration" else BENCH_MODULES
    modules = module_list[: args.limit] if args.limit else module_list
    missing = [m for m in modules if not (args.bench_dir / m).exists()]
    if missing:
        # A silently skipped module changes the denominator — refuse to run.
        console.print(f"[bold red]Error:[/] benchmark modules missing: {missing}")
        sys.exit(1)

    suffix = f"_limit{args.limit}" if args.limit else ""
    prefix = "calib_" if args.module_set == "calibration" else ""
    run_name = f"{args.date}_{prefix}{args.config}_r{args.run_index}{suffix}"
    output = args.results_dir / f"{run_name}.json"
    run_dir = args.results_dir / "runs" / run_name
    tests_dir = run_dir / "tests"
    memory_dir = run_dir / "memory"

    # ── Configure the agent for this run (read lazily by Settings) ──
    os.environ.update(CONFIGS[args.config]["env"])
    os.environ["MEMORY_DIR"] = str(memory_dir)
    os.environ["TRACE_FILE"] = str(run_dir / "trace.jsonl")

    results: list[dict] = []
    if output.exists():
        if not args.resume:
            console.print(f"[bold red]Error:[/] {output} already exists — refusing to overwrite (use --resume)")
            sys.exit(1)
        previous = json.loads(output.read_text(encoding="utf-8"))
        metadata = previous["metadata"]
        if metadata.get("git_commit") != _git("rev-parse", "HEAD"):
            console.print("[bold red]Error:[/] code changed since this run started — cannot resume")
            sys.exit(1)
        results = previous["results"]
        metadata.setdefault("resumed_at", []).append(datetime.now(timezone.utc).isoformat(timespec="seconds"))
    else:
        if memory_dir.exists() and any(memory_dir.iterdir()):
            console.print(f"[bold red]Error:[/] memory dir {memory_dir} is not empty — runs must start from an empty store")
            sys.exit(1)
        metadata = build_metadata(args, run_name, modules, memory_dir)

    tests_dir.mkdir(parents=True, exist_ok=True)
    done_modules = {r["module"] for r in results}

    console.print(f"\n[bold]Run {run_name}: {len(modules)} modules — {CONFIGS[args.config]['description']}[/]\n")

    def _persist() -> None:
        payload = {"metadata": metadata, "summary": summarize(results), "results": results}
        output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    for i, module in enumerate(modules, 1):
        if module in done_modules:
            console.print(f"[dim]({i}/{len(modules)}) {module} — already done, skipping[/]")
            continue
        console.print(f"\n[bold cyan]({i}/{len(modules)}) {module}[/]")
        results.append(run_one(args.bench_dir, module, tests_dir, args.max_retries))
        # Persist after every module so a crash loses at most one module.
        _persist()

    metadata["finished_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    _persist()

    console.print("\n")
    print_table(results)
    summary = summarize(results)
    if summary:
        console.print("\n")
        print_summary(summary)
    console.print(f"\n[bold green]Results saved to:[/] {output}")


if __name__ == "__main__":
    main()
