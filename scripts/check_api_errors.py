"""
Scan benchmark runs for LLM API failures (402 balance, 429 rate limit, 5xx,
connection / timeout errors).

The agent catches LLM exceptions in several places (Reflector counts them as
a failed repair attempt, Evaluator falls through to the Executor, the
MemoryManager defaults to ADD), so an API outage can hide inside otherwise
normal-looking results. A run is accepted only if this scan finds nothing.

Usage:
    python scripts/check_api_errors.py benchmark_results/flash/runs/2026-10-30_A_r1 ...
Exit code 1 if any API error is found.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

API_ERROR = re.compile(
    r"Error code: (?:4\d\d|5\d\d)|Insufficient Balance|APIConnectionError|APITimeoutError|"
    r"RateLimitError|InternalServerError|ServiceUnavailable|Connection error|Request timed out"
)


def scan_run(run_dir: Path) -> list[str]:
    hits: list[str] = []
    trace = run_dir / "trace.jsonl"
    if trace.exists():
        for line in trace.read_text(encoding="utf-8").splitlines():
            if API_ERROR.search(line):
                e = json.loads(line)
                hits.append(f"trace {e.get('ts')} {e.get('kind')} {e.get('module')}: {API_ERROR.search(line).group(0)}")
    console = run_dir / "console.log"
    if console.exists():
        for n, line in enumerate(console.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            m = API_ERROR.search(line)
            if m:
                hits.append(f"console.log:{n}: {m.group(0)}")
    result = run_dir.parent.parent / f"{run_dir.name}.json"
    if result.exists():
        for r in json.loads(result.read_text(encoding="utf-8"))["results"]:
            if r.get("error") and API_ERROR.search(r["error"]):
                hits.append(f"result {r['module']}: status={r['status']} {API_ERROR.search(r['error']).group(0)}")
    return hits


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan benchmark runs for LLM API errors")
    parser.add_argument("run_dirs", nargs="+", type=Path)
    args = parser.parse_args()
    dirty = False
    for run_dir in args.run_dirs:
        hits = scan_run(run_dir)
        print(f"{run_dir.name}: {'CLEAN' if not hits else f'{len(hits)} API error line(s)'}")
        for h in hits[:10]:
            print(f"   {h}")
        dirty |= bool(hits)
    sys.exit(1 if dirty else 0)


if __name__ == "__main__":
    main()
