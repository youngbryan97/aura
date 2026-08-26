"""Exact evidence available to the turn that authored one delivered answer.

This is not an explanation of transformer internals.  It records the inputs the
runtime can prove reached the answer's turn: exact-turn tool receipts, sensory
observations, and admitted grounding.  When none of those exist, the remaining
supported account is model-native inference from the checkpoint and ordinary
conversation context.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from core.conversation.discourse_repair_pursuit import question_focus

__all__ = [
    "AnswerProvenance",
    "answer_provenance_from_turn",
    "answer_provenance_reply",
    "asks_for_prior_answer_provenance",
    "provenance_grounding_json",
    "select_prior_answer_provenance",
]

_SCHEMA = "aura.answer_provenance.v1"
_MAX_RECEIPTS = 12
_MAX_GROUNDING = 12
_MAX_SENSES = 3
_EPISTEMIC_TERMS = frozenset(
    {
        "answer",
        "believe",
        "conclude",
        "conclusion",
        "came",
        "come",
        "evidence",
        "find",
        "found",
        "get",
        "knew",
        "know",
        "learn",
        "learned",
        "source",
        "say",
        "tell",
        "think",
    }
)
_REFERENTIAL_TERMS = frozenset({"answer", "it", "that", "this"})


def _text(value: Any, *, limit: int) -> str:
    return " ".join(str(value or "").split())[:limit]


def _sha256(value: Any) -> str:
    return hashlib.sha256(str(value or "").encode("utf-8")).hexdigest()


def _bounded_receipt(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    receipt_id = _text(value.get("receipt_id"), limit=64)
    tool = _text(value.get("tool"), limit=128)
    if not receipt_id or not tool:
        return None
    try:
        recorded_at = float(value.get("recorded_at") or 0.0)
    except (TypeError, ValueError):
        recorded_at = 0.0
    return {
        "receipt_id": receipt_id,
        "tool": tool,
        "action": _text(value.get("action"), limit=160),
        "object_ref": _text(value.get("object_ref"), limit=320),
        "ok": bool(value.get("ok")),
        "effect_observed": bool(value.get("effect_observed")),
        "verification": _text(value.get("verification"), limit=240),
        "recorded_at": recorded_at,
        "session_id": _text(value.get("session_id"), limit=160),
        "turn_id": _text(value.get("turn_id"), limit=160),
    }


def _bounded_sense(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    channel = _text(value.get("channel"), limit=32).casefold()
    if channel not in {"camera", "microphone", "screen"}:
        return None
    try:
        observed_at = float(value.get("observed_at") or 0.0)
    except (TypeError, ValueError):
        observed_at = 0.0
    return {
        "channel": channel,
        "ok": bool(value.get("ok")),
        "observation": _text(value.get("observation"), limit=480),
        "observed_at": observed_at,
        "session_id": _text(value.get("session_id"), limit=160),
        "turn_id": _text(value.get("turn_id"), limit=160),
    }


@dataclass(frozen=True, slots=True)
class AnswerProvenance:
    """Bounded evidence contract for one delivered answer."""

    answer_sha256: str
    session_id: str
    turn_id: str
    captured_at: float
    response_path: str = ""
    tool_receipts: tuple[dict[str, Any], ...] = ()
    sensory_evidence: tuple[dict[str, Any], ...] = ()
    grounding_digests: tuple[str, ...] = ()
    model_native_inference: bool = True
    causal_scope: str = "available_inputs_not_hidden_state_attribution"
    schema: str = _SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_value(cls, value: Any) -> AnswerProvenance | None:
        if not isinstance(value, Mapping) or value.get("schema") != _SCHEMA:
            return None
        answer_sha256 = _text(value.get("answer_sha256"), limit=64)
        if len(answer_sha256) != 64:
            return None
        raw_receipts = value.get("tool_receipts")
        receipt_values = raw_receipts if isinstance(raw_receipts, (list, tuple)) else ()
        receipts = tuple(
            receipt
            for raw in receipt_values[:_MAX_RECEIPTS]
            if (receipt := _bounded_receipt(raw)) is not None
        )
        raw_senses = value.get("sensory_evidence")
        sense_values = raw_senses if isinstance(raw_senses, (list, tuple)) else ()
        senses = tuple(
            sense
            for raw in sense_values[:_MAX_SENSES]
            if (sense := _bounded_sense(raw)) is not None
        )
        raw_grounding = value.get("grounding_digests")
        grounding_values = raw_grounding if isinstance(raw_grounding, (list, tuple)) else ()
        grounding = tuple(
            digest
            for raw in grounding_values[:_MAX_GROUNDING]
            if len(digest := _text(raw, limit=64)) == 64
        )
        try:
            captured_at = float(value.get("captured_at") or 0.0)
        except (TypeError, ValueError):
            captured_at = 0.0
        return cls(
            answer_sha256=answer_sha256,
            session_id=_text(value.get("session_id"), limit=160),
            turn_id=_text(value.get("turn_id"), limit=160),
            captured_at=captured_at,
            response_path=_text(value.get("response_path"), limit=120),
            tool_receipts=receipts,
            sensory_evidence=senses,
            grounding_digests=grounding,
            model_native_inference=bool(value.get("model_native_inference", True)),
            causal_scope="available_inputs_not_hidden_state_attribution",
        )


def answer_provenance_from_turn(
    answer_text: Any,
    *,
    response_path: Any = "",
    model_native_inference: bool = True,
) -> AnswerProvenance:
    """Snapshot the exact-turn evidence still under active custody."""

    from core.conversation.surface_disposition import turn_tool_receipts
    from core.conversation.turn_evidence_custody import (
        current_turn_evidence_custody,
        turn_grounding_evidence,
        turn_sensory_evidence,
    )

    custody = current_turn_evidence_custody()
    receipts = tuple(
        receipt
        for raw in turn_tool_receipts()[:_MAX_RECEIPTS]
        if (receipt := _bounded_receipt(raw)) is not None
    )
    senses = tuple(
        sense
        for raw in turn_sensory_evidence()[:_MAX_SENSES]
        if (sense := _bounded_sense(raw)) is not None
    )
    grounding_digests = tuple(
        _sha256(text)
        for raw in turn_grounding_evidence()[:_MAX_GROUNDING]
        if (text := str(raw or "").strip())
    )
    return AnswerProvenance(
        answer_sha256=_sha256(answer_text),
        session_id=_text(getattr(custody, "session_id", ""), limit=160),
        turn_id=_text(getattr(custody, "turn_id", ""), limit=160),
        captured_at=time.time(),
        response_path=_text(response_path, limit=120),
        tool_receipts=receipts,
        sensory_evidence=senses,
        grounding_digests=grounding_digests,
        model_native_inference=bool(model_native_inference),
    )


def asks_for_prior_answer_provenance(value: Any) -> bool:
    """Whether a question asks how the immediately prior answer was known."""

    focus = question_focus(value)
    if focus is None or focus.kind not in {
        "cause",
        "entity_or_description",
        "mechanism_or_manner",
        "location",
    }:
        return False
    terms = set(focus.terms)
    if not (terms & _EPISTEMIC_TERMS):
        return False
    normalized = " ".join(str(value or "").casefold().split())
    surface_terms = set(re.findall(r"[a-z0-9]+", normalized))
    implicit_short_epistemic = bool(
        focus.kind == "mechanism_or_manner"
        and terms <= {"know", "knew"}
    )
    return bool(surface_terms & _REFERENTIAL_TERMS) or implicit_short_epistemic or normalized.startswith("what is your source") or (
        normalized.startswith("what's your source")
    )


def select_prior_answer_provenance(
    current_question: Any,
    recent_exchanges: Sequence[Mapping[str, Any]] | Any,
) -> AnswerProvenance | None:
    """Resolve an epistemic follow-up to the latest delivered answer."""

    if not asks_for_prior_answer_provenance(current_question) or not isinstance(
        recent_exchanges, Sequence
    ):
        return None
    for exchange in reversed(list(recent_exchanges)):
        if not isinstance(exchange, Mapping):
            continue
        provenance = AnswerProvenance.from_value(exchange.get("answer_provenance"))
        if provenance is None:
            continue
        # A provenance projection explains an earlier answer; it does not
        # become the new epistemic subject when the user repeats or emphasizes
        # the same question. Walk through it to the authored answer beneath.
        if provenance.response_path == "verified_answer_provenance":
            continue
        aura_text = str(exchange.get("aura") or "")
        if aura_text and provenance.answer_sha256 == _sha256(aura_text):
            return provenance
    return None


def answer_provenance_reply(provenance: AnswerProvenance) -> str:
    """Render only facts established by the answer's bound provenance."""

    successful_tools = [row for row in provenance.tool_receipts if row.get("ok")]
    live_senses = [row for row in provenance.sensory_evidence if row.get("ok")]
    parts: list[str] = []
    if successful_tools:
        names = ", ".join(dict.fromkeys(str(row["tool"]) for row in successful_tools))
        parts.append(f"That answer's turn actually used {names}; the receipts are bound to that turn.")
    if live_senses:
        channels = ", ".join(dict.fromkeys(str(row["channel"]) for row in live_senses))
        parts.append(f"It also had a current {channels} observation available.")
    if provenance.grounding_digests:
        parts.append("It had authenticated grounding attached to the same turn.")
    if not successful_tools and not live_senses and not provenance.grounding_digests:
        parts.append(
            "That answer came from the cortex's learned parameters and inference over the conversation; "
            "there was no tool lookup or sensor reading bound to that turn."
        )
    elif not successful_tools:
        parts.append("There was no tool lookup bound to that turn.")
    parts.append(
        "I can establish which inputs reached the turn, but not attribute an individual token to a "
        "particular hidden representation."
    )
    return " ".join(parts)


def provenance_grounding_json(provenance: AnswerProvenance) -> str:
    """Stable serialization for exact-turn evidence custody."""

    return json.dumps(provenance.to_dict(), ensure_ascii=True, sort_keys=True, separators=(",", ":"))
