# Memory threshold calibration (config C)

Calibrates the cosine-similarity threshold of the `signature` memory variant on
**held-out** data: 12 TheAlgorithms modules that are not among the 19 benchmark
modules (`CALIBRATION_MODULES` in `scripts/benchmark.py`).

## Data

- Runs: `benchmark_results/flash/2026-10-30_calib_B_r{1,2,3}.json` — config B
  (memory off, so failures are not influenced by memory), `deepseek-v4-flash`,
  base commit `275ff17`. Repeats 2 and 3 were added because repeat 1 produced
  only 9 failure samples.
- Samples: `samples.json` — every failure the Reflector was given, with the
  normalized error signature (`extract_error_signature`, commit `1400d93`) and
  the Reflector's diagnosis (classification / root cause).

## Labeling rubric

Each sample gets one **failure-mechanism** label, chosen by the knowledge that
fixes it (the root cause in the diagnosis), not by the surface symptom. Two
samples from different modules are *relevant* to each other iff they share a
label — i.e. a memory of fixing one would plausibly help fix the other.

The categories were written after reading the 9 samples of repeat 1 and before
looking at repeats 2–3:

| label | meaning |
|---|---|
| `import-path` | test imports / patches / runs the module by a package path that does not exist in the flat sandbox (`maths.x`, `runpy.run_module("pkg.mod")`, patch target) |
| `expects-exception-not-raised` | test asserts an exception (`pytest.raises`) that the source does not raise for that input |
| `unexpected-exception` | test expects graceful handling (a value, or no error) for an input the source does not handle, so the source raises |
| `wrong-expected-value` | test's expected value is wrong for the input (miscalculated, or behavior invented for invalid input) |
| `api-contract-misread` | test calls the API with the wrong argument convention / violated precondition (inclusive vs exclusive bounds, pre-sorted input, heap property) |
| `test-code-bug` | the test's own code is wrong (bad input construction, helper bug, NameError / TypeError inside the test) |
| `mutation-semantics` | test assumes the input is (not) mutated / in-place vs returned copy |
| `float-precision` | exact comparison on floating-point results |
| `hang-timeout` | infinite loop, runaway recursion, blocking `input()` |
| `missing-plugin-fixture` | uses a fixture / plugin / package not installed (`mocker`, `hypothesis`) |
| `output-capture` | asserts on printed output / doctest output formatting |
| `static-syntax` | ruff fatal / syntax error in the generated test |
| `other` | none of the above (never counted as relevant to anything, including other `other`) |
| `exclude` | sample has no usable failure output (removed from the evaluation) |

Conventions applied while labeling:

- If the diagnosis lists several causes, the **first** one decides the label.
- A sample whose repair attempt produced no diagnosis is `exclude`
  (1 sample: `calib_B_r3:searches/ternary_search.py:3`).
- `calib_B_r3:sorts/heap_sort.py:1` loads the module by repository file path
  (`Path(__file__).parent.parent / "sorts/heap_sort.py"`), which fails in the
  flat sandbox for the same reason as package-path imports → `import-path`.

## Procedure

`scripts/calibrate_memory_threshold.py evaluate` embeds every signature with
the same model the memory store uses (Chroma default, all-MiniLM-L6-v2), takes
all cross-module pairs, and sweeps the threshold 0.30–0.95 (step 0.025):
precision / recall / F1 of "similarity ≥ threshold" vs "same label". The chosen
threshold is the F1 maximum (ties → the higher threshold). Results:
`threshold.json`. Labels: `labels.json`.

## Result

25 labeled samples from 9 modules (3 of the 12 held-out modules never
failed), 273 cross-module pairs, 31 of them relevant (base rate 0.114).

**Chosen threshold: cosine ≥ 0.40** — precision 0.276, recall 0.871, F1 0.419.

The F1 curve is flat (0.40 → 0.419, 0.55 → 0.414 with precision 0.444 /
recall 0.387): relevant pairs have median similarity 0.45, irrelevant 0.32,
with heavy overlap. The pre-stated rule (argmax F1) was applied as written.

Same labeled samples scored the legacy way (config A: query = first 300 chars
of the output + filename, document = "<classification>: <root cause up to the
first '.'> [in file]", 1/(1+squared L2) ≥ 0.4), ordered cross-module pairs:

| representation | threshold | retrieved | precision | recall | median sim. relevant / irrelevant |
|---|---|---|---|---|---|
| legacy (A) | 1/(1+L2²) ≥ 0.4 | 510 / 546 | 0.106 | 0.871 | 0.439 / 0.435 |
| signature (C) | cosine ≥ 0.40 | 98 / 273 | 0.276 | 0.871 | 0.45 / 0.32 |

At the same recall the legacy representation's precision equals the base rate
(retrieval is effectively random); the signature representation is ~2.4× the
base rate but still mostly retrieves irrelevant pairs. Small sample: 25
failures, 9 modules, one labeler (no inter-rater check).
