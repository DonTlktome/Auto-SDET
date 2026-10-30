# INVALID — round 1 aborted by DeepSeek "402 Insufficient Balance"

Not used in any reported number. Kept as evidence only.

- Runs: `2026-10-30_{A,B,C}_r1` (deepseek-v4-flash; A/B at `275ff17`, C at `80f9cd9`),
  started 12:27–12:28 UTC concurrently.
- The account balance ran out at about **12:44:30 UTC**; every LLM call after
  that returned `Error code: 402 - Insufficient Balance`.
- Damage:
  - A_r1: last 3 modules `error`; the MemoryManager call for
    `data_structures/queues/queue_by_list.py` failed with 402 and fell back to ADD.
  - B_r1: the Reflector call for `conversions/decimal_to_binary.py` got 402,
    which the base code counts as a failed repair attempt (`parse_error`), so
    the module was recorded as `failed` — not a real agent failure;
    `dynamic_programming/fibonacci.py` `error`.
  - C_r1: 14 modules finished; stopped manually during the 15th
    (`data_structures/stacks/stack.py`, Reflector got 402).
- Round 1 was rerun from scratch after the balance was topped up (same
  commits, empty memory stores). From then on, every round's traces and logs
  are scanned for API errors (402 / 429 / 5xx / connection) before it is
  accepted.
