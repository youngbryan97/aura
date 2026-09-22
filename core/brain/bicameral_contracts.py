"""Immutable contracts and intent evidence for bicameral advisory frames."""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import re
import secrets
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from core.phases.action_intent import detect_action_intent
from core.runtime.turn_analysis import analyze_turn

MAX_OBJECTIVE_SCAN_CHARS = 8192
MAX_OBJECTIVE_PREVIEW_CHARS = 300
MAX_HISTORY_LIMIT = 1024
HISTORY_RETENTION_S = 3600.0

_PERSPECTIVES = ("narrator", "protector", "explorer", "critic", "social")
_PERSPECTIVE_DIMENSION = MappingProxyType(
    {
        "narrator": "coherence",
        "protector": "effect_integrity",
        "explorer": "novelty_utility",
        "critic": "factuality",
        "social": "continuity",
    }
)
_LEARNABLE_OUTCOMES = frozenset(
    {"verified_success", "verified_failure", "explicit_user_positive", "explicit_user_negative"}
)
_OUTCOME_SOURCES = frozenset(
    {"effect_receipt", "response_verifier", "task_verifier", "explicit_user_feedback"}
)
_GOVERNANCE = MappingProxyType(
    {
        "advisory_only": True,
        "no_external_effects": True,
        "will_authority_required_for_effects": True,
        "narrator_reconciles_internal_proposals": True,
        "behavior_controls_allowlisted": True,
    }
)
_FRAME_KEY = secrets.token_bytes(32)

_WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_'-]{2,}")
_DIRECTED_ACTION_RE = re.compile(
    r"^(?:please\s+)?(?:open|launch|run|execute|click|type|write|save|export|search|"
    r"download|install|delete|commit|push|browse|navigate|create|send)\b|"
    r"\b(?:can|could|would|will)\s+you\s+(?:please\s+)?(?:open|launch|run|execute|"
    r"click|type|write|save|export|search|download|install|delete|commit|push|"
    r"browse|navigate|create|send)\b",
    re.IGNORECASE,
)
_TOOL_MEDIATED_ACTION_RE = re.compile(
    r"\b(?:can|could|would|will)\s+you\s+use\s+(?:your\s+)?(?:tools?|browser|desktop|"
    r"terminal)\s+to\s+(?:verify|check|search|open|write|save|export|download|run|execute)\b",
    re.IGNORECASE,
)
_EPISTEMIC_UNCERTAINTY_RE = re.compile(
    r"\b(?:i(?:'m| am)\s+(?:confused|uncertain|unsure)|we\s+do(?:n't| not)\s+know|"
    r"uncertainty|unknown|unclear|ambiguous|verify|prove|test\s+whether|check\s+whether|"
    r"what\s+evidence|how\s+can\s+we\s+know)\b",
    re.IGNORECASE,
)
_CREATIVE_INTENT_RE = re.compile(
    r"\b(?:brainstorm|invent|imagine|design|synthesize|generate\s+(?:ideas|options)|"
    r"novel\s+(?:idea|approach|alternative)|creative\s+(?:idea|approach)|metaphor)\b",
    re.IGNORECASE,
)
_SELF_CONCEPT_RE = re.compile(
    r"\b(?:self|feel|introspect|reflect|conscious|sentient|aware|identity|experience|"
    r"subjective|mind|preferences?|capabilit(?:y|ies)|tools?|skills?|abilities)\b",
    re.IGNORECASE,
)
_SELF_PRINCIPAL_RE = re.compile(r"\b(?:you|your|yourself|aura|i|my|myself)\b", re.IGNORECASE)
_RELATIONSHIP_RE = re.compile(
    r"\b(?:our\s+(?:relationship|conversation|history)|between\s+us|remember\s+me|"
    r"what\s+i\s+told\s+you|my\s+preference|together)\b",
    re.IGNORECASE,
)

_STOPWORDS = {
    "about",
    "again",
    "also",
    "and",
    "are",
    "can",
    "could",
    "does",
    "for",
    "from",
    "have",
    "how",
    "into",
    "just",
    "like",
    "make",
    "more",
    "need",
    "not",
    "now",
    "out",
    "that",
    "the",
    "then",
    "this",
    "through",
    "want",
    "what",
    "when",
    "where",
    "with",
    "would",
    "you",
    "your",
}


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    if not math.isfinite(value):
        return lower
    return max(lower, min(upper, value))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return parsed if math.isfinite(parsed) else default


def _keywords(text: str, limit: int = 6) -> list[str]:
    found: list[str] = []
    seen: set[str] = set()
    for match in _WORD_RE.finditer(text.lower()):
        token = match.group(0).strip("'_-")
        if token in _STOPWORDS or token in seen:
            continue
        seen.add(token)
        found.append(token)
        if len(found) >= limit:
            break
    return found


def _emotion_value(affect: Any, *names: str, default: float = 0.0) -> float:
    values: list[float] = []
    emotions = getattr(affect, "emotions", None)
    for name in names:
        values.append(_safe_float(getattr(affect, name, default), default))
        if isinstance(emotions, dict):
            values.append(_safe_float(emotions.get(name), default))
    return max(values) if values else default


def _emotion_observed(affect: Any, *names: str) -> bool:
    if affect is None:
        return False
    emotions = getattr(affect, "emotions", None)
    return any(
        hasattr(affect, name) or (isinstance(emotions, Mapping) and name in emotions)
        for name in names
    )


def _stable_id(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:16]


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_thaw(item) for item in value]
    return value


def _canonical(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        _thaw(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sign(payload: Mapping[str, Any]) -> str:
    return hmac.new(_FRAME_KEY, _canonical(payload), hashlib.sha256).hexdigest()


def _single_line(value: Any, limit: int) -> str:
    text = " ".join(str(value or "").split())
    text = "".join(char for char in text if char.isprintable())
    return text[:limit]


def _scope_id(context: Mapping[str, Any] | None, origin: str) -> str:
    context = context if isinstance(context, Mapping) else {}
    parts = [
        str(context.get(key) or "").strip()
        for key in ("principal_id", "user_id", "conversation_id", "session_id")
    ]
    material = ":".join(part for part in parts if part) or f"local:{origin.split(':', 1)[0]}"
    return _stable_id(material)


@dataclass(frozen=True)
class AdvisoryIntentEvidence:
    tool: bool
    uncertain: bool
    creative: bool
    self_reflective: bool
    social: bool
    intent_type: str
    semantic_mode: str
    sources: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "uncertain": self.uncertain,
            "creative": self.creative,
            "self_reflective": self.self_reflective,
            "social": self.social,
            "intent_type": self.intent_type,
            "semantic_mode": self.semantic_mode,
            "sources": list(self.sources),
        }


def classify_advisory_intent(
    text: str,
    *,
    context: Mapping[str, Any] | None,
    origin: str,
    confusion: float,
    curiosity: float,
) -> AdvisoryIntentEvidence:
    context = context if isinstance(context, Mapping) else {}
    turn = analyze_turn(text, matched_skills=context.get("matched_skills", False))
    action = detect_action_intent(text)
    lower = text.casefold()
    structured_action = bool(
        context.get("user_requested_action")
        or context.get("desktop_task")
        or context.get("tool_execution_required")
        or isinstance(context.get("action_intent"), Mapping)
    )
    tool_mediated_action = bool(_TOOL_MEDIATED_ACTION_RE.search(lower))
    directed_action = bool(_DIRECTED_ACTION_RE.search(lower) or tool_mediated_action)
    tool = structured_action or bool(
        directed_action and (action.has_action_request or turn.intent_type in {"SKILL", "TASK"})
    ) or tool_mediated_action
    uncertain = confusion >= 0.25 or bool(_EPISTEMIC_UNCERTAINTY_RE.search(lower))
    creative = curiosity >= 0.35 or bool(_CREATIVE_INTENT_RE.search(lower))
    self_reflective = bool(
        _SELF_PRINCIPAL_RE.search(lower)
        and (_SELF_CONCEPT_RE.search(lower) or turn.requires_live_aura_voice)
    )
    social = bool(origin.startswith(("user", "desktop", "voice")))
    social = social or bool(_RELATIONSHIP_RE.search(lower))
    sources: list[str] = ["turn_analysis"]
    if structured_action:
        sources.append("structured_context")
    if action.has_action_request:
        sources.append("action_intent")
    if confusion >= 0.25 or curiosity >= 0.35:
        sources.append("measured_affect")
    return AdvisoryIntentEvidence(
        tool=tool,
        uncertain=uncertain,
        creative=creative,
        self_reflective=self_reflective,
        social=social,
        intent_type=turn.intent_type,
        semantic_mode=turn.semantic_mode,
        sources=tuple(dict.fromkeys(sources)),
    )


@dataclass(frozen=True)
class AdvisoryProposal:
    perspective: str
    salience: float
    stance: str
    directive: str
    attention_targets: tuple[str, ...] = field(default_factory=tuple)
    routing_bias: Mapping[str, bool] = field(default_factory=lambda: MappingProxyType({}))
    sampling_bias: Mapping[str, float] = field(default_factory=lambda: MappingProxyType({}))
    state_pressure: Mapping[str, float] = field(default_factory=lambda: MappingProxyType({}))

    def to_dict(self) -> dict[str, Any]:
        return {
            "perspective": self.perspective,
            "salience": self.salience,
            "stance": self.stance,
            "directive": self.directive,
            "attention_targets": list(self.attention_targets),
            "routing_bias": _thaw(self.routing_bias),
            "sampling_bias": _thaw(self.sampling_bias),
            "state_pressure": _thaw(self.state_pressure),
        }


@dataclass(frozen=True)
class BicameralFrame:
    frame_id: str
    objective: str
    narrator_summary: str
    consensus: float
    dissent: float
    salience: float
    proposals: tuple[AdvisoryProposal, ...]
    attention_targets: tuple[str, ...]
    routing_bias: Mapping[str, bool]
    sampling_bias: Mapping[str, float]
    causal_effects: Mapping[str, Any]
    reconciliation: Mapping[str, Any]
    state_evidence: Mapping[str, Any]
    intent_evidence: Mapping[str, Any]
    scope_id: str
    governance: Mapping[str, Any] = field(default_factory=lambda: _GOVERNANCE)
    created_at: float = field(default_factory=lambda: time.time())
    integrity: str = ""

    def to_dict(self, *, include_integrity: bool = True) -> dict[str, Any]:
        payload = {
            "frame_id": self.frame_id,
            "objective": self.objective,
            "narrator_summary": self.narrator_summary,
            "consensus": self.consensus,
            "dissent": self.dissent,
            "salience": self.salience,
            "attention_targets": list(self.attention_targets),
            "routing_bias": _thaw(self.routing_bias),
            "sampling_bias": _thaw(self.sampling_bias),
            "causal_effects": _thaw(self.causal_effects),
            "reconciliation": _thaw(self.reconciliation),
            "state_evidence": _thaw(self.state_evidence),
            "intent_evidence": _thaw(self.intent_evidence),
            "scope_id": self.scope_id,
            "governance": _thaw(self.governance),
            "created_at": self.created_at,
        }
        payload["proposals"] = [proposal.to_dict() for proposal in self.proposals]
        if include_integrity:
            payload["integrity"] = self.integrity
        return payload


_FRAME_FIELDS = frozenset(BicameralFrame.__dataclass_fields__)


def validate_bicameral_frame(frame: Mapping[str, Any] | BicameralFrame) -> bool:
    if isinstance(frame, BicameralFrame):
        payload = frame.to_dict()
    elif isinstance(frame, Mapping):
        payload = _thaw(frame)
    else:
        return False
    if set(payload) != _FRAME_FIELDS:
        return False
    supplied = str(payload.pop("integrity", "") or "")
    if payload.get("governance") != _thaw(_GOVERNANCE):
        return False
    try:
        expected = _sign(payload)
    # not a failure: a payload that will not sign does not verify, and False is the
    # refusing direction.
    except (TypeError, ValueError, OverflowError):
        return False
    return bool(supplied) and hmac.compare_digest(supplied, expected)
