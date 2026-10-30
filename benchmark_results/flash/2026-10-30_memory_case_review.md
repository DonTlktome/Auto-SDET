# Cross-module memory hits — manual relevance review (flash, 3 rounds)

Source data: `runs/2026-10-30_{A,C}_r{1,2,3}/trace.jsonl`; candidate list:
`2026-10-30_memory_trace_analysis.json` (`cross_module_hit_then_fixed`).

A candidate is a Reflector attempt where ≥1 injected memory came from another
module **and** the next pytest run passed. That is necessary, not sufficient,
for "memory helped". Each candidate was judged by reading the current
diagnosis next to the full injected memory (root cause + fix summary), using
the mechanism labels of `calibration/README.md`:

- **relevant** — same failure mechanism, and the memory's fix applies
- **partial** — the memory lists several causes; a secondary one matches
- **not relevant** — everything else (including same surface symptom with a
  fix that would not apply)

## Counts

| config | candidates (hit → fixed) | relevant | partial | not relevant |
|---|---|---|---|---|
| A (legacy retrieval) | 34 | 4 | 2 | 28 |
| C (signature retrieval) | 32 | 8 | 1 | 23 |

Consistent with the calibration precision (legacy 0.106 vs signature 0.276):
most injected memories are irrelevant in both configs.

## Relevant / partial cases

| run | module (retry) | ← memory from | sim. | mechanism | verdict |
|---|---|---|---|---|---|
| A_r1 | sorts/quick_sort.py (1) | maths/gcd_of_n_numbers.py | 0.450 | function mutates caller's input | relevant |
| A_r1 | data_structures/stacks/stack.py (1) | searches/binary_search.py | 0.489 | hard-coded Windows absolute path | relevant |
| A_r2 | sorts/quick_sort.py (1) | sorts/bubble_sort.py | 0.487 | string patch target re-imports missing package | relevant |
| A_r2 | sorts/insertion_sort.py (1) | sorts/bubble_sort.py | 0.479 | module location assumption | relevant |
| A_r2 | sorts/bubble_sort.py (1) | maths/factorial.py | 0.446 | package path not importable (memory's 2nd cause: runpy) | partial |
| A_r3 | maths/gcd_of_n_numbers.py (2) | maths/factorial.py | 0.468 | package path not importable (memory's 2nd cause: runpy) | partial |
| C_r1 | sorts/insertion_sort.py (1) | sorts/quick_sort.py | 0.665 | module location assumption | relevant |
| C_r1 | data_structures/stacks/stack.py (1) | sorts/insertion_sort.py | 0.530 | test loads module by a file path absent in sandbox | relevant |
| C_r1 | data_structures/linked_list/singly_linked_list.py (1) | data_structures/stacks/stack.py | 0.811 | hard-coded Windows absolute path | relevant |
| C_r1 | sorts/quick_sort.py (2) | maths/factorial.py | 0.480 | package path not importable (memory's 2nd cause: runpy) | partial |
| C_r2 | sorts/quick_sort.py (1) | sorts/bubble_sort.py | 0.633 | string patch target re-imports missing package | relevant |
| C_r2 | sorts/insertion_sort.py (1) | sorts/bubble_sort.py | 0.898 | dotted package path not importable | relevant |
| C_r2 | data_structures/stacks/stack.py (1) | sorts/insertion_sort.py | 0.403 | module location assumption | relevant |
| C_r3 | sorts/insertion_sort.py (2) | sorts/quick_sort.py | 0.728 | module location assumption | relevant |
| C_r3 | data_structures/stacks/stack.py (1) | sorts/insertion_sort.py | 0.479 | module location assumption | relevant |

Notable non-relevant high-similarity hits (signature variant false positives):
`C_r1 dynamic_programming/fibonacci.py ← sorts/quick_sort.py` (0.941) and
`C_r3 … ← sorts/quick_sort.py` (0.801): normalizing numbers to `<n>` makes any
list-inequality assertion look identical. `C_r3 searches/binary_search.py ←
maths/gcd_of_n_numbers.py` (1.000): both RecursionError, but gcd's fix (raise
the recursion limit for a deep finite recursion) does not apply to
binary_search's infinite recursion — judged not relevant.

## Counterfactual: is the relevant class fixed without memory?

All relevant hits fall in one class: **module-location assumptions** in the
generated test (hard-coded host path, dotted package path in import / patch /
runpy, repo-layout file path). Reflector attempts whose diagnosis is in that
class (keyword match on the root cause; `scripts/memory_counterfactual.py`), fixed by the next pytest run:

| config | attempts | fixed by next run |
|---|---|---|
| B (no memory) | 15 | 11 (73%) |
| A (legacy memory) | 11 | 8 (73%) |
| C (signature memory) | 16 | 12 (75%) |
| C, same-class cross-module memory injected | 11 | 9 (82%) |
| C, no same-class memory injected | 5 | 3 (60%) |

Without memory the Reflector fixes this class at the same rate; the C split
(9/11 vs 3/5) is far too small to distinguish. Example: `stack.py` hit the
same hard-coded-Windows-path failure in B_r1 and B_r2 and was fixed on the
first attempt both times, with no memory.

## Best case (full evidence)

`C_r1`, `data_structures/linked_list/singly_linked_list.py`, retry 1.

- Error: collection fails —
  `FileNotFoundError: [Errno 2] No such file or directory: '/workspace/D:\\Documents\\...\\singly_linked_list.py'`
  (the generated test hard-codes the host's Windows path to the module).
- Query (signature): `FileNotFoundError: [Errno <n>] No such file or directory: 'D:\\Documents\\...\\singly_linked_list.py`
- Top candidate, injected: trajectory `0976d275…` from
  `data_structures/stacks/stack.py`, cosine 0.811, outcome `fix_applied` —
  root cause "The test hard-codes a Windows absolute path to the module under
  test … exec_module raises FileNotFoundError during collection"; fix summary
  "+# Locate the module under test (no hard-coded absolute paths)."
- Applied fix (diff in the trace's `test_before` / `test_after`): replaces
  `_SOURCE_PATH = Path(r"D:\...\singly_linked_list.py")` with a `_find_source()`
  helper documented "Locate ``singly_linked_list.py`` without hard-coding an
  absolute path", and corrects one invalid-index parametrization.
- Next run: passed. Caveat: coverage is `None` — the module is loaded under
  the name `singly_linked_list_under_test`, which `--cov=singly_linked_list`
  does not measure.

Verdict: a genuine cross-module retrieval of the right experience, and the fix
follows it; but the counterfactual above shows the same class is fixed on the
first attempt without memory, so the memory did not change the outcome.
