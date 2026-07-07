"""Basic offline evaluation harness for the weather agent.

Runs a fixed dataset of cases through the real agent and scores each on
property-based metrics (we can't string-match a non-deterministic LLM, so we
assert on *properties* of the answer instead):

  1. structured_ok   - the agent returned a valid ResponseFormat
  2. output_valid     - the answer passes the output guardrails (ranges, C/F)
  3. city_grounded    - the correct city (from user_id) shows up in the tool trace

Prints a per-metric pass rate. Run when you change the model, prompt, or tools
to catch regressions:

    uv run python evaluation.py

Needs a valid GROQ_API_KEY (real model calls).
"""

import asyncio

from dotenv import load_dotenv, find_dotenv
from langgraph.checkpoint.memory import InMemorySaver

from agent import build_agent
from guardrails import GuardrailViolation, check_output
from schemas import Context

load_dotenv(find_dotenv())

# input -> expected properties. expect_city ties user_id to locate_user's answer.
CASES = [
    {"user_id": "user_1", "message": "what is the weather like", "expect_city": "pune"},
    {"user_id": "user_2", "message": "how's the weather today?", "expect_city": "hyd"},
    {"user_id": "user_3", "message": "is it hot outside right now?", "expect_city": "mumbai"},
]


def _serialize_messages(messages) -> str:
    """Flatten a run's messages (incl. tool calls/results) into lowercase text."""
    parts = []
    for m in messages:
        parts.append(str(getattr(m, "content", "")))
        for call in getattr(m, "tool_calls", None) or []:
            parts.append(str(call.get("args", "")))
    return " ".join(parts).lower()


async def run_case(agent, case: dict, index: int) -> dict:
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": case["message"]}]},
        config={"configurable": {"thread_id": f"eval-{index}"}},
        context=Context(user_id=case["user_id"]),
    )

    response = result.get("structured_response")
    structured_ok = response is not None

    output_valid = False
    if structured_ok:
        try:
            check_output(response)
            output_valid = True
        except GuardrailViolation:
            output_valid = False

    trace = _serialize_messages(result.get("messages", []))
    city_grounded = case["expect_city"] in trace

    return {
        "case": case["message"],
        "structured_ok": structured_ok,
        "output_valid": output_valid,
        "city_grounded": city_grounded,
    }


async def main() -> None:
    agent = build_agent(InMemorySaver())

    results = []
    for i, case in enumerate(CASES):
        try:
            results.append(await run_case(agent, case, i))
        except Exception as exc:  # a crash is itself a failed case
            print(f"[error] case {i} raised: {exc}")
            results.append(
                {
                    "case": case["message"],
                    "structured_ok": False,
                    "output_valid": False,
                    "city_grounded": False,
                }
            )

    metrics = ["structured_ok", "output_valid", "city_grounded"]
    n = len(results)

    print("\n=== Evaluation results ===")
    for r in results:
        flags = " ".join(f"{m}={'PASS' if r[m] else 'FAIL'}" for m in metrics)
        print(f"- {r['case'][:40]:40}  {flags}")

    print("\n=== Pass rates ===")
    for m in metrics:
        passed = sum(1 for r in results if r[m])
        print(f"{m:16} {passed}/{n}  ({passed / n:.0%})")

    overall = sum(1 for r in results if all(r[m] for m in metrics))
    print(f"\nfully-passing cases: {overall}/{n} ({overall / n:.0%})")


if __name__ == "__main__":
    asyncio.run(main())
