"""Social cognition layer for timing, subtext, memory selection, and restraint."""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

from .schemas import clamp, stable_hash


@dataclass
class SocialCognitionDecision:
    decision_id: str
    subtext: str
    timing: str
    response_mode: str
    memory_policy: str
    vulnerability_allowed: bool
    restraint_level: float
    trust_risk: float
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SocialCognitionLayer:
    """Grounds social naturalness in real conversational state, not style prompts."""

    VALID_SUBTEXTS = {
        "information_request",
        "validation_request",
        "challenge",
        "reassurance",
        "brainstorming",
        "frustration",
        "pride_or_excitement",
        "project_meaning",
    }

    def evaluate(
        self,
        message: str,
        *,
        relationship_memory: Sequence[Mapping[str, Any]] = (),
        runtime_state: Mapping[str, Any] | None = None,
        confidence: float = 0.7,
    ) -> SocialCognitionDecision:
        text = str(message or "")
        lower = text.lower()
        reasons: list[str] = []
        subtext = "information_request"
        if any(w in lower for w in ("worth", "believe", "real", "matter", "good is", "how good")):
            subtext = "validation_request"
            reasons.append("question implies project validation need")
        if any(w in lower for w in ("frustrated", "angry", "annoyed", "not working", "broken", "can't")):
            subtext = "frustration"
            reasons.append("frustration language detected")
        if any(w in lower for w in ("proud", "excited", "amazing", "love this")):
            subtext = "pride_or_excitement"
            reasons.append("positive accomplishment signal")
        if "?" not in text and len(text.split()) < 8:
            subtext = "reassurance" if subtext == "information_request" else subtext
            reasons.append("short prompt may prefer presence before detail")
        if any(w in lower for w in ("prove", "challenge", "be honest", "really")):
            subtext = "challenge"
            reasons.append("challenge/honesty framing")
        if any(w in lower for w in ("what if", "could we", "idea", "brainstorm")):
            subtext = "brainstorming"
            reasons.append("open-ended ideation signal")

        trust_risk = 0.0
        if any(w in lower for w in ("dismiss", "ignored", "validation", "fake", "roleplay")):
            trust_risk += 0.35
            reasons.append("trust/dismissal risk present")
        if len(text.split()) > 120:
            trust_risk += 0.15
            reasons.append("large ask needs structured restraint")
        if confidence < 0.45:
            trust_risk += 0.2
            reasons.append("low subtext confidence")

        state = dict(runtime_state or {})
        genuine_state_available = any(k in state for k in ("uncertainty", "degraded_mode", "affect", "memory_salience", "confidence"))
        vulnerability_allowed = bool(genuine_state_available and trust_risk < 0.75)
        if not genuine_state_available:
            reasons.append("no grounded internal state for vulnerability claims")

        if subtext in {"validation_request", "reassurance", "frustration"}:
            timing = "validate_first"
            response_mode = "two_layer"
            restraint = 0.7
        elif subtext == "challenge":
            timing = "direct_then_evidence"
            response_mode = "precise"
            restraint = 0.45
        elif subtext == "brainstorming":
            timing = "ask_or_offer_paths"
            response_mode = "collaborative"
            restraint = 0.35
        else:
            timing = "answer_now"
            response_mode = "technical_if_requested"
            restraint = 0.5

        if trust_risk > 0.55:
            response_mode = "short_empathic_then_optional_detail"
            restraint = max(restraint, 0.8)
        memory_policy = self._memory_policy(relationship_memory, subtext, trust_risk)
        decision_id = stable_hash(
            {
                "message": re.sub(r"\s+", " ", text[:400]),
                "subtext": subtext,
                "timing": timing,
                "mode": response_mode,
                "risk": trust_risk,
            },
            prefix="soc_",
        )
        return SocialCognitionDecision(
            decision_id=decision_id,
            subtext=subtext if subtext in self.VALID_SUBTEXTS else "information_request",
            timing=timing,
            response_mode=response_mode,
            memory_policy=memory_policy,
            vulnerability_allowed=vulnerability_allowed,
            restraint_level=clamp(restraint),
            trust_risk=clamp(trust_risk),
            reasons=reasons,
        )

    @staticmethod
    def _memory_policy(memories: Sequence[Mapping[str, Any]], subtext: str, trust_risk: float) -> str:
        if not memories:
            return "no_memory_reference"
        if trust_risk > 0.6:
            return "use_relationship_memory_only_if_user_recently_referenced_it"
        if subtext in {"validation_request", "reassurance", "project_meaning"}:
            return "use_one_high-salience_relationship_memory"
        return "retrieve_relevant_memory_with_creepiness_penalty"
