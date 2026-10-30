"""
Aggregate repeated benchmark runs into one report.

Reads `benchmark_results/<date>_<config>_r<n>.json` files, then reports per
configuration the mean / min / max across runs, and per-module paired tests
between configurations (pairs are (module, run_index), i.e. A_r1 vs B_r1
— those runs executed concurrently under the same API conditions).

Statistics (exact, no scipy needed):
  - pass / fail outcomes: exact McNemar test on discordant pairs
  - per-module mean retries / coverage: exact two-sided sign test
Modules that ended in a sandbox infra failure or a harness error are not
agent outcomes and are excluded from the paired tests (count reported).

Usage:
    python scripts/summarize_benchmarks.py
    python scripts/summarize_benchmarks.py --out benchmark_results/2026-10-31_summary
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from math import comb
from pathlib import Path
from statistics import mean

REPO_ROOT = Path(__file__).resolve().parent.parent
RUN_FILE = re.compile(r"^(?P<date>\d{4}-\d{2}-\d{2})_(?P<config>[A-Z])_r(?P<run>\d+)\.json$")
NON_AGENT_STATUSES = {"infra_failed", "error"}

# USD per 1M tokens. Assumed list prices for deepseek-v4-pro (peak), taken from
# DeepSeek's pricing page as reported on 2026-10-17 — verify against the
# console bill; the token counts themselves are measured.
DEFAULT_PRICES = {"input_cache_miss": 1.32, "input_cache_hit": 0.044, "output": 3.96}

METRICS = [
    ("one_shot_pass_rate", "一次通过率 %"),
    ("final_pass_rate", "最终通过率 %"),
    ("avg_retries", "平均重试次数"),
    ("avg_coverage_pct", "平均覆盖率 %（通过的模块）"),
    ("avg_wall_time_s", "单模块平均耗时 s"),
    ("total_infra_failures", "基础设施故障次数（已重试）"),
    ("infra_failed_modules", "因基础设施故障结束的模块数"),
    ("total_llm_calls", "LLM 调用次数"),
    ("cost_usd", "LLM 估算费用 USD"),
]
BREAKDOWN = [
    ("sandbox_create_s", "沙箱创建"),
    ("pip_install_s", "pip 安装"),
    ("pytest_s", "pytest 执行"),
    ("llm_total_s", "LLM 调用（全部）"),
    ("llm_critical_s", "LLM 调用（关键路径，不含后台 MemoryManager）"),
    ("other_s", "其他（图调度 / 嵌入 / 上传 / 退避）"),
]


# ── Statistics ───────────────────────────────────────────

def _binom_two_sided(k: int, n: int) -> float:
    """Exact two-sided binomial test p-value against p=0.5."""
    if n == 0:
        return 1.0
    tail = sum(comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def mcnemar(pairs: list[tuple[bool, bool]]) -> dict:
    """pairs of (x_pass, y_pass). b = x only, c = y only."""
    b = sum(1 for x, y in pairs if x and not y)
    c = sum(1 for x, y in pairs if y and not x)
    return {"n_pairs": len(pairs), "x_only": b, "y_only": c, "p_value": round(_binom_two_sided(min(b, c), b + c), 4)}


def sign_test(diffs: list[float]) -> dict:
    """Exact two-sided sign test on paired differences (zeros dropped)."""
    pos = sum(1 for d in diffs if d > 0)
    neg = sum(1 for d in diffs if d < 0)
    return {
        "n_modules": len(diffs),
        "x_higher": pos,
        "y_higher": neg,
        "ties": len(diffs) - pos - neg,
        "mean_diff": round(mean(diffs), 3) if diffs else None,
        "p_value": round(_binom_two_sided(min(pos, neg), pos + neg), 4),
    }


# ── Loading ──────────────────────────────────────────────

def load_runs(results_dir: Path) -> dict[str, list[dict]]:
    runs: dict[str, list[dict]] = defaultdict(list)
    for path in sorted(results_dir.glob("*.json")):
        m = RUN_FILE.match(path.name)
        if not m:
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        if "metadata" not in data:
            continue
        data["_file"] = path.relative_to(REPO_ROOT).as_posix()
        data["_run_index"] = int(m["run"])
        runs[m["config"]].append(data)
    return runs


def run_cost(results: list[dict], prices: dict) -> float:
    total = 0.0
    for r in results:
        u = r.get("llm_usage") or {}
        hit = u.get("cache_hit_tokens", 0)
        miss = u.get("prompt_tokens", 0) - hit
        total += (miss * prices["input_cache_miss"] + hit * prices["input_cache_hit"]
                  + u.get("completion_tokens", 0) * prices["output"]) / 1e6
    return total


# ── Aggregation ──────────────────────────────────────────

def _stats(values: list[float]) -> dict:
    values = [v for v in values if v is not None]
    if not values:
        return {"mean": None, "min": None, "max": None}
    return {"mean": round(mean(values), 2), "min": round(min(values), 2), "max": round(max(values), 2)}


def aggregate_config(runs: list[dict], prices: dict) -> dict:
    per_run = []
    for run in sorted(runs, key=lambda r: r["_run_index"]):
        s = dict(run["summary"])
        s["cost_usd"] = round(run_cost(run["results"], prices), 2)
        per_run.append({"file": run["_file"], "finished": run["metadata"].get("finished_at") is not None, **s})
    agg = {key: _stats([r.get(key) for r in per_run]) for key, _ in METRICS}
    agg["time_breakdown_s"] = {
        key: _stats([(r.get("avg_time_breakdown_s") or {}).get(key) for r in per_run]) for key, _ in BREAKDOWN
    }
    return {"n_runs": len(per_run), "runs": per_run, "metrics": agg}


def paired_tests(x_runs: list[dict], y_runs: list[dict]) -> dict:
    x_by_run = {r["_run_index"]: {m["module"]: m for m in r["results"]} for r in x_runs}
    y_by_run = {r["_run_index"]: {m["module"]: m for m in r["results"]} for r in y_runs}

    final_pairs, one_shot_pairs, excluded = [], [], 0
    retries_x, retries_y = defaultdict(list), defaultdict(list)
    cov_x, cov_y = defaultdict(list), defaultdict(list)
    for idx in sorted(set(x_by_run) & set(y_by_run)):
        for module in sorted(set(x_by_run[idx]) & set(y_by_run[idx])):
            mx, my = x_by_run[idx][module], y_by_run[idx][module]
            if mx["status"] in NON_AGENT_STATUSES or my["status"] in NON_AGENT_STATUSES:
                excluded += 1
                continue
            px, py = mx["status"] == "done", my["status"] == "done"
            final_pairs.append((px, py))
            one_shot_pairs.append((px and mx["retry_count"] == 0, py and my["retry_count"] == 0))
            retries_x[module].append(mx["retry_count"])
            retries_y[module].append(my["retry_count"])
            if px and mx.get("coverage_pct") is not None:
                cov_x[module].append(mx["coverage_pct"])
            if py and my.get("coverage_pct") is not None:
                cov_y[module].append(my["coverage_pct"])

    retry_diffs = [mean(retries_x[m]) - mean(retries_y[m]) for m in retries_x]
    cov_diffs = [mean(cov_x[m]) - mean(cov_y[m]) for m in cov_x if cov_y.get(m)]
    return {
        "run_indices": sorted(set(x_by_run) & set(y_by_run)),
        "excluded_non_agent_pairs": excluded,
        "final_pass_mcnemar": mcnemar(final_pairs),
        "one_shot_pass_mcnemar": mcnemar(one_shot_pairs),
        "retries_sign_test": sign_test(retry_diffs),
        "coverage_sign_test": sign_test(cov_diffs),
    }


# ── Rendering ────────────────────────────────────────────

def _fmt(stat: dict) -> str:
    if stat["mean"] is None:
        return "—"
    return f"{stat['mean']} ({stat['min']}–{stat['max']})"


def render_markdown(report: dict) -> str:
    configs = sorted(report["configs"])
    lines = ["# Benchmark summary", "", "均值（最小–最大），每组 n 次运行。", ""]
    lines.append("| 指标 | " + " | ".join(f"{c} (n={report['configs'][c]['n_runs']})" for c in configs) + " |")
    lines.append("|---|" + "---|" * len(configs))
    for key, label in METRICS:
        lines.append(f"| {label} | " + " | ".join(_fmt(report["configs"][c]["metrics"][key]) for c in configs) + " |")
    lines += ["", "## 耗时拆账（单模块平均，秒）", ""]
    lines.append("| 段 | " + " | ".join(configs) + " |")
    lines.append("|---|" + "---|" * len(configs))
    for key, label in BREAKDOWN:
        lines.append(f"| {label} | " + " | ".join(
            _fmt(report["configs"][c]["metrics"]["time_breakdown_s"][key]) for c in configs) + " |")
    lines += ["", "## 按模块配对检验", ""]
    for name, t in report["paired_tests"].items():
        x, y = name.split("_vs_")
        f, o, r, cv = t["final_pass_mcnemar"], t["one_shot_pass_mcnemar"], t["retries_sign_test"], t["coverage_sign_test"]
        lines += [
            f"### {x} vs {y}（runs {t['run_indices']}，排除非 Agent 结果 {t['excluded_non_agent_pairs']} 对）",
            "",
            f"- 最终通过（McNemar 精确检验）：{f['n_pairs']} 对，仅 {x} 通过 {f['x_only']}，仅 {y} 通过 {f['y_only']}，p = {f['p_value']}",
            f"- 一次通过（McNemar，记忆不影响首次生成，可作噪声基线）：仅 {x} {o['x_only']}，仅 {y} {o['y_only']}，p = {o['p_value']}",
            f"- 每模块平均重试（符号检验）：{x} 更多 {r['x_higher']}，{y} 更多 {r['y_higher']}，持平 {r['ties']}，均差 {r['mean_diff']}，p = {r['p_value']}",
            f"- 每模块平均覆盖率（符号检验）：{x} 更高 {cv['x_higher']}，{y} 更高 {cv['y_higher']}，持平 {cv['ties']}，均差 {cv['mean_diff']}，p = {cv['p_value']}",
            "",
        ]
    lines += ["## 来源文件", ""]
    for c in configs:
        for run in report["configs"][c]["runs"]:
            lines.append(f"- {c}: `{run['file']}`" + ("" if run["finished"] else "（未完成）"))
    lines += ["", f"费用按假设单价估算：{report['prices_usd_per_1m']}（token 数为实测）。"]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate repeated Auto-SDET benchmark runs")
    parser.add_argument("--results-dir", type=Path, default=REPO_ROOT / "benchmark_results")
    parser.add_argument("--configs", nargs="*", default=None, help="Subset of configs to include")
    parser.add_argument("--out", type=Path, default=None, help="Write <out>.json and <out>.md")
    args = parser.parse_args()

    runs = load_runs(args.results_dir)
    if args.configs:
        runs = {c: r for c, r in runs.items() if c in args.configs}
    report = {
        "prices_usd_per_1m": DEFAULT_PRICES,
        "configs": {c: aggregate_config(r, DEFAULT_PRICES) for c, r in sorted(runs.items())},
        "paired_tests": {},
    }
    for x, y in [("A", "B"), ("C", "A"), ("C", "B")]:
        if x in runs and y in runs:
            report["paired_tests"][f"{x}_vs_{y}"] = paired_tests(runs[x], runs[y])

    md = render_markdown(report)
    print(md)
    if args.out:
        args.out.with_suffix(".json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        args.out.with_suffix(".md").write_text(md, encoding="utf-8")


if __name__ == "__main__":
    main()
