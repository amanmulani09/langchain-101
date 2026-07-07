"""LangSmith-native evaluation for the weather agent.

Same property-based metrics as evaluation.py, but run through LangSmith so the
dataset, each run's trace, and the scores are versioned in the UI and
comparable across experiments (e.g. before/after a model or prompt change).

Reuses check_output as the output-validity evaluator, so runtime guardrails and
eval share one definition of "valid answer".

Requires (in .env or environment):
    GROQ_API_KEY, LANGSMITH_API_KEY   (and LANGSMITH_TRACING=true for traces)

Run:
    uv run python evaluation_langsmith.py
"""

import uuid

from dotenv import load_dotenv, find_dotenv
from langgraph.checkpoint.memory import InMemorySaver
from langsmith import Client, evaluate

from agent import build_agent
from guardrails import GuardrailViolation, check_output
from schemas import Context, ResponseFormat

load_dotenv(find_dotenv())

DATASET_NAME = "weather-agent-eval"

# inputs -> reference outputs (what we expect to be true about the answer)
EXAMPLES = [
    ({"user_id": "user_1", "message": "what is the weather like"}, {"expect_city": "pune"}),
    ({"user_id": "user_2", "message": "how's the weather today?"}, {"expect_city": "hyd"}),
    ({"user_id": "user_3", "message": "is it hot outside right now?"}, {"expect_city": "mumbai"}),
]

_agent = build_agent(InMemorySaver())


def _ensure_dataset(client: Client) -> None:
    if client.has_dataset(dataset_name=DATASET_NAME):
        return
    dataset = client.create_dataset(DATASET_NAME)
    client.create_examples(
        inputs=[inp for inp, _ in EXAMPLES],
        outputs=[out for _, out in EXAMPLES],
        dataset_id=dataset.id,
    )


def target(inputs: dict) -> dict:
    """The system under test: one agent turn. Traced automatically."""
    result = _agent.invoke(
        {"messages": [{"role": "user", "content": inputs["message"]}]},
        config={"configurable": {"thread_id": f"eval-{uuid.uuid4().hex}"}},
        context=Context(user_id=inputs["user_id"]),
    )
    response = result.get("structured_response")

    trace_parts = []
    for m in result.get("messages", []):
        trace_parts.append(str(getattr(m, "content", "")))
        for call in getattr(m, "tool_calls", None) or []:
            trace_parts.append(str(call.get("args", "")))

    return {
        "response": response.model_dump() if response else None,
        "tool_trace": " ".join(trace_parts).lower(),
    }


# --- evaluators: (outputs, reference_outputs) -> {"key", "score"} ---------
def structured_ok(outputs: dict) -> dict:
    return {"key": "structured_ok", "score": outputs.get("response") is not None}


def output_valid(outputs: dict) -> dict:
    data = outputs.get("response")
    ok = False
    if data is not None:
        try:
            check_output(ResponseFormat(**data))
            ok = True
        except GuardrailViolation:
            ok = False
    return {"key": "output_valid", "score": ok}


def city_grounded(outputs: dict, reference_outputs: dict) -> dict:
    trace = outputs.get("tool_trace", "")
    return {"key": "city_grounded", "score": reference_outputs["expect_city"] in trace}


def main() -> None:
    client = Client()
    _ensure_dataset(client)
    results = evaluate(
        target,
        data=DATASET_NAME,
        evaluators=[structured_ok, output_valid, city_grounded],
        experiment_prefix="weather-agent",
    )
    print("Experiment done — view results in LangSmith.")
    print(results)


if __name__ == "__main__":
    main()
