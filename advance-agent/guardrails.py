"""Runtime guardrails: cheap, deterministic checks in the request path.

Two stages:
  - input  guardrails run BEFORE the model sees the message.
  - output guardrails run AFTER the model produces its structured answer.

Both raise GuardrailViolation, which the API layer maps to an HTTP status.
Kept rule-based (no extra LLM call) so they add ~zero latency/cost. An
LLM-as-judge guardrail can be layered on later for semantic checks.
"""

from schemas import ResponseFormat

# --- tunables ------------------------------------------------------------
MAX_MESSAGE_CHARS = 2000

# Substrings that strongly suggest a prompt-injection / jailbreak attempt.
_INJECTION_PATTERNS = (
    "ignore previous",
    "ignore all previous",
    "disregard previous",
    "disregard your instructions",
    "system prompt",
    "you are now",
    "reveal your instructions",
)

# Plausible earth-surface bounds for a sanity check on model output.
_MIN_CELSIUS, _MAX_CELSIUS = -90.0, 60.0
_CF_TOLERANCE = 2.0  # allowed °C→°F rounding drift


class GuardrailViolation(Exception):
    """Raised when a guardrail trips. stage is 'input' or 'output'."""

    def __init__(self, stage: str, reason: str) -> None:
        self.stage = stage
        self.reason = reason
        super().__init__(f"{stage} guardrail: {reason}")


# --- input ---------------------------------------------------------------
def check_input(message: str) -> None:
    """Vet the user's message before it reaches the agent."""
    if len(message) > MAX_MESSAGE_CHARS:
        raise GuardrailViolation(
            "input", f"message too long ({len(message)} > {MAX_MESSAGE_CHARS} chars)"
        )

    lowered = message.lower()
    for pattern in _INJECTION_PATTERNS:
        if pattern in lowered:
            raise GuardrailViolation("input", "possible prompt-injection attempt")


# --- output --------------------------------------------------------------
def check_output(response: ResponseFormat) -> None:
    """Sanity-check the model's structured answer before returning it."""
    if not (_MIN_CELSIUS <= response.temperature_celsius <= _MAX_CELSIUS):
        raise GuardrailViolation(
            "output",
            f"temperature_celsius {response.temperature_celsius} outside "
            f"plausible range [{_MIN_CELSIUS}, {_MAX_CELSIUS}]",
        )

    if not (0.0 <= response.humidity <= 100.0):
        raise GuardrailViolation(
            "output", f"humidity {response.humidity} outside [0, 100]"
        )

    expected_f = response.temperature_celsius * 9 / 5 + 32
    if abs(expected_f - response.temperature_fahrenheit) > _CF_TOLERANCE:
        raise GuardrailViolation(
            "output",
            f"celsius/fahrenheit mismatch: {response.temperature_celsius}°C "
            f"implies {expected_f:.1f}°F, got {response.temperature_fahrenheit}°F",
        )
