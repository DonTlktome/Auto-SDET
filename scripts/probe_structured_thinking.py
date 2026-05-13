"""
Minimal probe — does `with_structured_output` survive thinking-on in a
single-turn call against DeepSeek V4?

If this prints a parsed ReflectionResult with non-empty fields, the
"structured output forces thinking off" assumption is false and we can
have both features simultaneously.

If this 400s or returns None, the assumption holds and we need a fallback.
"""
from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_deepseek import ChatDeepSeek

from auto_sdet.config import get_settings
from auto_sdet.models.schemas import ReflectionResult


def main() -> None:
    settings = get_settings()
    if not settings.deepseek_api_key:
        raise SystemExit("DEEPSEEK_API_KEY missing")

    # Build the LLM ourselves — bypass get_llm() so we can flip thinking ON
    # despite the "for tool use" semantics that the factory enforces.
    llm = ChatDeepSeek(
        model=settings.deepseek_model,
        api_key=settings.deepseek_api_key.get_secret_value(),
        api_base=settings.deepseek_base_url,
        temperature=0.0,
        max_tokens=4096,
        extra_body={"thinking": {"type": "enabled"}},
    )
    structured = llm.with_structured_output(ReflectionResult)

    # Use the real production Reflector prompt to verify end-to-end.
    from auto_sdet.prompts.reflector import build_reflector_prompt
    system_prompt, user_prompt = build_reflector_prompt(
        source_path="examples/calculator.py",
        source_code="def add(a, b):\n    return a + b\n",
        test_code="def test_add():\n    assert add(2, 3) == 6\n",
        latest_error="AssertionError: assert 5 == 6",
        error_history=[],
        retry_count=1,
        max_retries=3,
        similar_cases=None,
    )
    # json_mode requires the prompt to literally contain the word "json".
    # The production Reflector prompt does not — patch it here for the probe.
    system_prompt += (
        "\n\n## Output Format\n"
        "Respond with a single JSON object containing exactly these fields: "
        "error_classification, root_cause, affected_lines, fix_strategy, fixed_code. "
        "error_classification MUST be one of: ImportError, AssertionError, TypeError, "
        "AttributeError, SyntaxError, FixtureError, MockError, Other. "
        "fix_strategy MUST be one of: minimal_patch, full_rewrite."
    )
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]

    # Attempt 1 — default method ("function_calling", uses tool_choice)
    print(">>> [1/2] default method (function_calling / tool_choice) ...")
    try:
        result = structured.invoke(messages)
    except Exception as e:
        print(f"    !! invoke raised: {type(e).__name__}: {str(e)[:200]}")
    else:
        if result is None:
            print("    !! invoke returned None")
        else:
            print(f"    ok: classification={result.error_classification}, "
                  f"fixed_code={len(result.fixed_code)} chars")

    # Attempt 2 — json_mode (bypasses tool_choice, uses response_format)
    print("\n>>> [2/2] method='json_mode' (response_format) ...")
    try:
        structured_json = llm.with_structured_output(ReflectionResult, method="json_mode")
        result2 = structured_json.invoke(messages)
    except Exception as e:
        print(f"    !! invoke raised: {type(e).__name__}: {str(e)[:200]}")
    else:
        if result2 is None:
            print("    !! invoke returned None")
        else:
            print(f"    ok: classification={result2.error_classification}, "
                  f"fixed_code={len(result2.fixed_code)} chars")


if __name__ == "__main__":
    main()