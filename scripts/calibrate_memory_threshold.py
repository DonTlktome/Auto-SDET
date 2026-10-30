"""
Calibrate the cosine-similarity threshold of the `signature` memory variant.

Two steps, so the labeling is a reviewable artifact rather than hidden logic:

  1. collect   — pull every failure the agent saw from held-out run traces:
                 the raw output → normalized error signature, plus the
                 Reflector's diagnosis of it (classification / root cause),
                 which is what a human labeler reads.
  2. evaluate  — given a hand-written label per sample (the failure
                 *mechanism*, e.g. "expects-exception-not-raised"), embed all
                 signatures with the same model the store uses, and sweep the
                 threshold over every cross-module pair:
                     relevant(i, j)  ⇔  same mechanism label
                 Reports precision / recall / F1 per threshold and picks the
                 F1-maximizing one (ties → the higher, more conservative value).

Only cross-module pairs count: same-module recall is not what memory is for.

Usage:
    python scripts/calibrate_memory_threshold.py collect <run_dir>... --out benchmark_results/calibration/samples.json
    # write benchmark_results/calibration/labels.json  {sample_id: mechanism}
    python scripts/calibrate_memory_threshold.py evaluate --samples ... --labels ... --out benchmark_results/calibration/threshold.json
"""
from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from auto_sdet.tools.error_signature import extract_error_signature  # noqa: E402

GRID = [round(0.30 + 0.025 * i, 3) for i in range(27)]   # 0.30 … 0.95


def collect(run_dirs: list[Path]) -> list[dict]:
    samples = []
    for run_dir in run_dirs:
        events = [json.loads(line) for line in (run_dir / "trace.jsonl").read_text(encoding="utf-8").splitlines()
                  if line.strip()]
        for e in events:
            if e["kind"] != "reflection":
                continue
            # The Reflector saw this failure output; its diagnosis helps labeling.
            samples.append({
                "id": f"{run_dir.name}:{e['module']}:{e['retry']}",
                "run": run_dir.name,
                "module": e["module"],
                "retry": e["retry"],
                "trigger": e.get("trigger"),
                "signature": extract_error_signature(e.get("latest_error") or ""),
                "classification": e.get("classification"),
                "root_cause": e.get("root_cause"),
            })
    return samples


def embed(texts: list[str]) -> np.ndarray:
    from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

    vecs = np.asarray(DefaultEmbeddingFunction()(texts), dtype=float)
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


def evaluate(samples: list[dict], labels: dict[str, str]) -> dict:
    labeled = [s for s in samples if s["id"] in labels and labels[s["id"]] != "exclude"]
    vecs = embed([s["signature"] for s in labeled])
    sims = vecs @ vecs.T

    pairs = []
    for i, j in combinations(range(len(labeled)), 2):
        if labeled[i]["module"] == labeled[j]["module"]:
            continue
        pairs.append((float(sims[i, j]), labels[labeled[i]["id"]] == labels[labeled[j]["id"]]))
    n_relevant = sum(rel for _, rel in pairs)

    curve = []
    for t in GRID:
        retrieved = [rel for s, rel in pairs if s >= t]
        tp = sum(retrieved)
        precision = tp / len(retrieved) if retrieved else None
        recall = tp / n_relevant if n_relevant else None
        f1 = (2 * precision * recall / (precision + recall)) if precision and recall else 0.0
        curve.append({"threshold": t, "retrieved_pairs": len(retrieved), "true_pairs": tp,
                      "precision": None if precision is None else round(precision, 3),
                      "recall": None if recall is None else round(recall, 3), "f1": round(f1, 3)})

    best = max(curve, key=lambda c: (c["f1"], c["threshold"]))
    rel_sims = [s for s, rel in pairs if rel]
    irr_sims = [s for s, rel in pairs if not rel]
    return {
        "n_samples": len(labeled),
        "n_modules": len({s["module"] for s in labeled}),
        "label_counts": {lab: sum(1 for s in labeled if labels[s["id"]] == lab) for lab in sorted(set(labels.values()))},
        "n_cross_module_pairs": len(pairs),
        "n_relevant_pairs": n_relevant,
        "relevant_similarity": _describe(rel_sims),
        "irrelevant_similarity": _describe(irr_sims),
        "curve": curve,
        "chosen_threshold": best["threshold"],
        "chosen_point": best,
        "rule": "argmax F1 over cross-module pairs; ties → higher threshold",
    }


def _describe(values: list[float]) -> dict | None:
    if not values:
        return None
    arr = np.asarray(values)
    return {"n": len(values), "min": round(float(arr.min()), 3), "p25": round(float(np.percentile(arr, 25)), 3),
            "median": round(float(np.median(arr)), 3), "p75": round(float(np.percentile(arr, 75)), 3),
            "max": round(float(arr.max()), 3)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate the signature-memory cosine threshold")
    sub = parser.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("run_dirs", nargs="+", type=Path)
    c.add_argument("--out", type=Path, required=True)
    e = sub.add_parser("evaluate")
    e.add_argument("--samples", type=Path, required=True)
    e.add_argument("--labels", type=Path, required=True)
    e.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if args.cmd == "collect":
        samples = collect(args.run_dirs)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(samples, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"{len(samples)} samples from {len({s['module'] for s in samples})} modules → {args.out}")
    else:
        samples = json.loads(args.samples.read_text(encoding="utf-8"))
        labels = json.loads(args.labels.read_text(encoding="utf-8"))
        report = evaluate(samples, labels)
        args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        for row in report["curve"]:
            print(row)
        print("chosen:", report["chosen_point"])


if __name__ == "__main__":
    main()
