"""
Tests for normalized error signatures (memory variant `signature`).
"""
from __future__ import annotations

from auto_sdet.tools.error_signature import extract_error_signature, normalize_line

HEADER = """============================= test session starts ==============================
platform linux -- Python 3.13.16, pytest-8.4.2, pluggy-1.6.0 -- /usr/local/bin/python
cachedir: .pytest_cache
rootdir: /workspace
plugins: cov-7.0.0, anyio-4.11.0
collecting ... collected 12 items
"""

DID_NOT_RAISE = HEADER + """
test_prime_check.py::test_primes[2] PASSED                               [  8%]
test_prime_check.py::test_invalid[True] FAILED                           [ 16%]

=================================== FAILURES ===================================
____________________________ test_invalid[True] ________________________________

value = True

    @pytest.mark.parametrize("value", [True, -1])
    def test_invalid(value):
>       with pytest.raises(ValueError):
E       Failed: DID NOT RAISE <class 'ValueError'>

test_prime_check.py:42: Failed
=========================== short test summary info ============================
FAILED test_prime_check.py::test_invalid[True] - Failed: DID NOT RAISE <class 'ValueError'>
========================= 1 failed, 11 passed in 0.21s =========================
"""

DID_NOT_RAISE_OTHER_MODULE = HEADER + """
=================================== FAILURES ===================================
______________________ TestFactorial.test_negative[-3] _________________________

    def test_negative(self, n):
>       with pytest.raises(ValueError):
E       Failed: DID NOT RAISE <class 'ValueError'>

test_factorial.py:88: Failed
=========================== short test summary info ============================
FAILED test_factorial.py::TestFactorial::test_negative[-3] - Failed: DID NOT RAISE <class 'ValueError'>
========================= 1 failed, 30 passed in 0.40s =========================
"""

COLLECTION_ERROR = HEADER + """
==================================== ERRORS ====================================
____________________ ERROR collecting test_stack.py _____________________
ImportError while importing test module '/workspace/test_stack.py'.
Traceback:
/usr/local/lib/python3.13/importlib/__init__.py:88: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
test_stack.py:3: in <module>
    from data_structures.stacks.stack import Stack
E   ModuleNotFoundError: No module named 'data_structures'
=========================== short test summary info ============================
ERROR test_stack.py
!!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
"""

ASSERTION = HEADER + """
=================================== FAILURES ===================================
_________________________________ test_sorted __________________________________

    def test_sorted():
>       assert merge_sort([3, 1, 2]) == [3, 2, 1]
E       AssertionError: assert [1, 2, 3] == [3, 2, 1]
E
E         At index 0 diff: 1 != 3

test_merge_sort.py:10: AssertionError
=========================== short test summary info ============================
FAILED test_merge_sort.py::test_sorted - AssertionError: assert [1, 2, 3] == [3, 2, 1]
"""

RUFF = """==================== STATIC ANALYSIS (ruff) — FATAL ====================
The generated test code cannot run — ruff detected blocking issues.

  L128 invalid-syntax: Expected a statement
  L128 invalid-syntax: Expected a statement
  L12 F821: Undefined name `mocker`

Fix these before the code can even be parsed/imported by pytest.
==================== END STATIC ANALYSIS ===================="""

EVALUATOR = """==================== EVALUATOR REJECTION ====================
overall_score = 0.42 (below threshold)

ACTIONABLE WEAKNESSES (must be addressed):
  [blocking] Tests assert on printed output instead of return values
    → suggested: assert on the returned list
==================== END EVALUATOR REJECTION ===================="""


def test_header_alone_is_not_the_signature():
    """The legacy query was mostly this header — the signature must not be."""
    sig = extract_error_signature(DID_NOT_RAISE)
    assert "test session starts" not in sig
    assert "platform linux" not in sig


def test_did_not_raise_extracted_and_test_id_removed():
    sig = extract_error_signature(DID_NOT_RAISE)
    assert "Failed: DID NOT RAISE <class 'ValueError'>" in sig
    assert "test_prime_check" not in sig


def test_same_mechanism_in_two_modules_gives_same_signature():
    assert extract_error_signature(DID_NOT_RAISE) == extract_error_signature(DID_NOT_RAISE_OTHER_MODULE)


def test_collection_error_uses_e_line():
    sig = extract_error_signature(COLLECTION_ERROR)
    assert sig == "ModuleNotFoundError: No module named 'data_structures'"


def test_assertion_numbers_normalized_and_tail_kept():
    sig = extract_error_signature(ASSERTION)
    assert "AssertionError: assert [<n>, <n>, <n>] == [<n>, <n>, <n>]" in sig
    assert "At index <n> diff: <n> != <n>" in sig     # traceback tail line


def test_ruff_findings_deduplicated_without_line_numbers():
    sig = extract_error_signature(RUFF)
    assert sig.splitlines() == ["ruff invalid-syntax: Expected a statement", "ruff F821: Undefined name `mocker`"]


def test_evaluator_weaknesses():
    sig = extract_error_signature(EVALUATOR)
    assert sig == "evaluator: Tests assert on printed output instead of return values"


def test_timeout_signature():
    sig = extract_error_signature("pytest did not finish within 60s — a test is likely hanging")
    assert sig.startswith("Timeout:")


def test_unknown_output_falls_back_to_tail():
    sig = extract_error_signature("something odd happened\nRuntimeError: boom\n")
    assert "RuntimeError: boom" in sig


def test_line_cap():
    many = HEADER + "\n".join(f"FAILED test_x.py::t{i} - ValueError: case {chr(97 + i)}" for i in range(10))
    assert len(extract_error_signature(many).splitlines()) == 6


def test_normalize_line_strips_paths_and_addresses():
    line = "/usr/local/lib/python3.13/site.py:12: <obj at 0x7f3a2b> failed 3 times"
    assert normalize_line(line) == "<obj at 0x?> failed <n> times"
