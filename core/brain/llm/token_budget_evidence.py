"""core/brain/llm/token_budget_evidence.py — a chars-per-token ratio that says where it came from.

The prompt assembler sizes every budget in characters and converts with one
number: ``max_tokens * 4``, annotated "Rough estimation: 1 token ~= 4 chars".
Four is roughly right for English prose and wrong in both directions for the
text this runtime actually carries. Code, JSON receipts, file paths, and CJK
run nearer two to three characters per token, so a prompt built to fit can be
half again over the real window. Overflow is not the symmetric failure: the
backend drops from the head, and the head is the identity lock and the
structural constraint block. The prompt keeps its shape and loses what binds
it.

The estimate cannot simply be replaced here. Loading a tokenizer in the
orchestrator process is the thing that must not happen on this path — it is
model work in the process that serves conversation, contending with the
resident worker for the same hardware.

So the same move as ``context_window_evidence``: the ratio travels with its
provenance, the assumption is conservative rather than average, and the
component that already knows both numbers reports them.

``MEASURED``
    Derived from prompts the worker actually tokenized: it holds the string it
    encoded and the token count it got, and reporting the pair costs nothing.
``ASSUMED``
    Nothing has been observed yet. The value is deliberately below the prose
    average, because under-filling the window wastes context and over-filling
    it deletes the constraints.

``ASSUMED`` records a degradation the first time it is used, so a runtime that
never receives an observation says so once instead of budgeting on a guess for
the life of the process.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from core.runtime.errors import record_degradation
from core.runtime.lockdep import checked_lock

__all__ = [
    "RatioSource",
    "CharsPerToken",
    "CALIBRATION_SCHEMA",
    "MIN_OBSERVATIONS",
    "assumed_chars_per_token",
    "calibration_batch_errors",
    "chars_per_token",
    "observe_calibration_batch",
    "observe_prompt_tokenization",
    "reset_for_test",
]

#: Below the ~4.0 English-prose average on purpose. The cost of being low is a
#: prompt that carries less than it could; the cost of being high is a prompt
#: whose head is silently deleted by the backend. Those are not comparable.
ASSUMED_CHARS_PER_TOKEN = 3.0

#: Observations needed before the measured ratio is trusted over the assumption.
#: One prompt is a sample of one document; a handful spans a conversation's mix
#: of prose, code, and receipts.
MIN_OBSERVATIONS = 8
CALIBRATION_SCHEMA = "aura.token_budget_calibration.v1"

#: Bounds any single observation must satisfy to be counted. A ratio outside
#: this range means the caller paired a string with a token count from a
#: different string, and averaging that in would corrupt every later budget.
_MIN_PLAUSIBLE_RATIO = 0.5
_MAX_PLAUSIBLE_RATIO = 12.0


class RatioSource(StrEnum):
    MEASURED = "measured"
    ASSUMED = "assumed"


@dataclass(frozen=True, slots=True)
class CharsPerToken:
    """A ratio and the evidence behind it."""

    ratio: float
    source: RatioSource
    observations: int
    detail: str = ""

    @property
    def measured(self) -> bool:
        return self.source is RatioSource.MEASURED

    def tokens_to_chars(self, tokens: int) -> int:
        return max(0, int(float(tokens) * self.ratio))


_LOCK = checked_lock("core.brain.llm.token_budget_evidence")
_TOTAL_CHARS = 0
_TOTAL_TOKENS = 0
_OBSERVATIONS = 0
_ASSUMPTION_REPORTED = False


def _coerce_observation(chars: Any, tokens: Any) -> tuple[int, int] | None:
    try:
        char_count = int(chars)
        token_count = int(tokens)
    except (TypeError, ValueError):
        return None
    if char_count <= 0 or token_count <= 0:
        return None
    ratio = char_count / token_count
    if not _MIN_PLAUSIBLE_RATIO <= ratio <= _MAX_PLAUSIBLE_RATIO:
        return None
    return char_count, token_count


def calibration_batch_errors(payload: Any) -> list[str]:
    """Validate a worker-init calibration batch without changing evidence state."""

    if not isinstance(payload, Mapping):
        return ["token_budget_calibration_not_mapping"]
    if payload.get("schema") != CALIBRATION_SCHEMA:
        return ["token_budget_calibration_schema_invalid"]
    observations = payload.get("observations")
    if not isinstance(observations, list):
        return ["token_budget_calibration_observations_not_list"]
    if not MIN_OBSERVATIONS <= len(observations) <= 16:
        return [
            "token_budget_calibration_observation_count_invalid:"
            f"{len(observations)}"
        ]
    errors: list[str] = []
    for index, item in enumerate(observations):
        if not isinstance(item, Mapping):
            errors.append(f"token_budget_calibration_item_not_mapping:{index}")
            continue
        if _coerce_observation(item.get("chars"), item.get("tokens")) is None:
            errors.append(f"token_budget_calibration_item_invalid:{index}")
    return errors


def observe_calibration_batch(payload: Any) -> int:
    """Atomically admit a validated worker-init calibration batch.

    Validation happens before any global counter changes, so a malformed worker
    receipt cannot partly poison an established runtime ratio.
    """

    global _TOTAL_CHARS, _TOTAL_TOKENS, _OBSERVATIONS

    if calibration_batch_errors(payload):
        return 0
    observations = payload["observations"]
    pairs = [
        _coerce_observation(item.get("chars"), item.get("tokens"))
        for item in observations
    ]
    if any(pair is None for pair in pairs):
        return 0
    admitted = [(chars, tokens) for chars, tokens in pairs if chars and tokens]
    with _LOCK:
        _TOTAL_CHARS += sum(chars for chars, _tokens in admitted)
        _TOTAL_TOKENS += sum(tokens for _chars, tokens in admitted)
        _OBSERVATIONS += len(admitted)
    return len(admitted)


def observe_prompt_tokenization(chars: int, tokens: int) -> bool:
    """Report one prompt's real character and token counts.

    Returns True when the observation was counted. Callers are components that
    tokenized a prompt anyway; nothing here tokenizes on its own.
    """

    global _TOTAL_CHARS, _TOTAL_TOKENS, _OBSERVATIONS

    pair = _coerce_observation(chars, tokens)
    if pair is None:
        try:
            char_count = int(chars)
            token_count = int(tokens)
            ratio = char_count / token_count if token_count else float("inf")
        except (TypeError, ValueError, ZeroDivisionError):
            return False
        if char_count <= 0 or token_count <= 0:
            return False
        # Two different strings, not a surprising one.
        record_degradation(
            "llm.token_budget_evidence",
            ValueError(
                f"implausible chars-per-token observation {ratio:.2f} "
                f"({char_count} chars / {token_count} tokens); discarded"
            ),
            severity="warning",
            action="kept the previous chars-per-token evidence",
        )
        return False
    char_count, token_count = pair

    with _LOCK:
        _TOTAL_CHARS += char_count
        _TOTAL_TOKENS += token_count
        _OBSERVATIONS += 1
    return True


def assumed_chars_per_token() -> CharsPerToken:
    return CharsPerToken(
        ratio=ASSUMED_CHARS_PER_TOKEN,
        source=RatioSource.ASSUMED,
        observations=0,
        detail="no prompt tokenization has been reported to this process",
    )


def chars_per_token() -> CharsPerToken:
    """The ratio to budget with, carrying how it was arrived at."""

    global _ASSUMPTION_REPORTED

    with _LOCK:
        observations = _OBSERVATIONS
        total_chars = _TOTAL_CHARS
        total_tokens = _TOTAL_TOKENS

    if observations >= MIN_OBSERVATIONS and total_tokens > 0:
        return CharsPerToken(
            ratio=total_chars / total_tokens,
            source=RatioSource.MEASURED,
            observations=observations,
            detail=f"{total_chars} chars over {total_tokens} tokens",
        )

    with _LOCK:
        first_time = not _ASSUMPTION_REPORTED
        _ASSUMPTION_REPORTED = True
    if first_time:
        record_degradation(
            "llm.token_budget_evidence",
            RuntimeError(
                "prompt budgets are using the assumed chars-per-token ratio "
                f"({ASSUMED_CHARS_PER_TOKEN}); {observations} of "
                f"{MIN_OBSERVATIONS} observations reported"
            ),
            severity="warning",
            action="budgeted the prompt against a stated assumption",
        )
    return CharsPerToken(
        ratio=ASSUMED_CHARS_PER_TOKEN,
        source=RatioSource.ASSUMED,
        observations=observations,
        detail=f"{observations}/{MIN_OBSERVATIONS} observations",
    )


def reset_for_test() -> None:
    global _TOTAL_CHARS, _TOTAL_TOKENS, _OBSERVATIONS, _ASSUMPTION_REPORTED

    with _LOCK:
        _TOTAL_CHARS = 0
        _TOTAL_TOKENS = 0
        _OBSERVATIONS = 0
        _ASSUMPTION_REPORTED = False
