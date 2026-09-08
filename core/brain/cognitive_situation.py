"""Unified semantic, analogical, and sensorimotor situation frame.

This module is a side-effect-free cognitive organ. It does not capture the
screen, click, type, write files, or call models. It reads the objective,
current runtime state, and already-owned perception/embodiment telemetry, then
emits a compact causal frame consumed by CognitiveEngine and response
generation. The point is to make semantic flexibility, analogical leaps, and
embodied grounding affect routing, sampling, attention, verification pressure,
and tool-governance posture through the same live path.
"""

from __future__ import annotations

import hashlib
import math
import re
import time
from typing import Any

from core.brain.cognitive_situation_frame import (
    CognitiveSituationFrame,
    clamp_unit_interval,
)
from core.container import ServiceContainer
from core.conversation.word_markers import names_any
from core.runtime.errors import record_degradation
from core.runtime.service_access import optional_service

_WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_'-]{2,}")
_SEMANTIC_RE = re.compile(
    r"\b(mean|meaning|interpret|understand|explain|define|why|how|what is|"
    r"what are|what would|could|should|ambiguous|nuance|frame|concept|abstract|"
    r"metaphor|symbol|implication|criteria|label|ontology|paradigm)\b",
    re.IGNORECASE,
)
_ANALOGY_RE = re.compile(
    r"\b(analogy|analogical|metaphor|like|compare|pattern|connection|bridge|"
    r"leap|novel|creative|imagine|what would .* look like|model out|synthesize|"
    r"cross-domain|transfer|fictional|inspiration)\b",
    re.IGNORECASE,
)
_SENSORIMOTOR_RE = re.compile(
    r"\b(screen|see|look|visible|desktop|window|cursor|click|type|keyboard|mouse|"
    r"open|close|scroll|drag|drop|select|copy|paste|write|save|export|download|"
    r"upload|folder|file|pdf|notes|docs|chrome|browser|app|application|wallpaper|"
    r"image|camera|microphone|voice|hear|speak|tool|tools|external|real[- ]world)\b",
    re.IGNORECASE,
)
_ACTION_RE = re.compile(
    r"\b(open|click|type|write|save|export|download|upload|create|delete|move|"
    r"rename|install|run|execute|search|browse|change|set|send|commit|push)\b",
    re.IGNORECASE,
)
_UNCERTAINTY_RE = re.compile(
    r"\b(confus|unsure|uncertain|maybe|probably|hypothetical|prove|verify|test|"
    r"check|investigate|review|audit|does this work|is this true)\b",
    re.IGNORECASE,
)
_SOCIAL_RE = re.compile(
    r"\b(feel|feeling|frustrat|angry|upset|tired|trust|relationship|rapport|"
    r"boundary|consent|private|personal|sensitive|apolog|sorry|repair|tone|"
    r"rude|kind|empathy|respect|pressure|persuad|convince|manipulat|"
    r"misunderst|hurt|comfortable|uncomfortable)\b",
    re.IGNORECASE,
)

_STOPWORDS = {
    "about",
    "again",
    "also",
    "and",
    "are",
    "because",
    "been",
    "being",
    "but",
    "can",
    "could",
    "does",
    "doing",
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
    "there",
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
    # `float(...)` rather than a bare return: the helper comes from a module
    # outside the strict-typed set, so mypy sees its result as Any and a
    # function that promises a float would be quietly returning anything.
    # The conversion is what makes the annotation true rather than asserted.
    return float(clamp_unit_interval(value, lower, upper))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return parsed if math.isfinite(parsed) else default


def _compact(value: Any, *, limit: int = 180) -> str:
    text = " ".join(str(value or "").strip().split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def _keywords(text: str, *, limit: int = 10) -> list[str]:
    seen: set[str] = set()
    words: list[str] = []
    for match in _WORD_RE.finditer(text.lower()):
        token = match.group(0).strip("'_-")
        if len(token) < 3 or token in _STOPWORDS or token in seen:
            continue
        seen.add(token)
        words.append(token)
        if len(words) >= limit:
            break
    return words


def _stable_id(*parts: Any) -> str:
    payload = "\n".join(str(part or "") for part in parts)
    return hashlib.sha256(payload.encode("utf-8", errors="replace")).hexdigest()[:16]


def _affect_value(state: Any, *names: str, default: float = 0.0) -> float:
    affect = getattr(state, "affect", None)
    emotions = getattr(affect, "emotions", None)
    values: list[float] = []
    for name in names:
        values.append(_safe_float(getattr(affect, name, default), default))
        if isinstance(emotions, dict):
            values.append(_safe_float(emotions.get(name), default))
    return max(values) if values else default


def _read_status(service: Any) -> dict[str, Any]:
    if service is None:
        return {}
    for attr in ("get_status", "status", "snapshot", "to_dict"):
        fn = getattr(service, attr, None)
        if not callable(fn):
            continue
        try:
            value = fn()
        except (OSError, ConnectionError, TimeoutError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
            record_degradation(
                "cognitive_situation",
                exc,
                severity="warning",
                action=f"skipped unreadable {type(service).__name__}.{attr} status",
            )
            continue
        if isinstance(value, dict):
            return dict(value)
    return {}


def _service_state(name: str) -> tuple[bool, dict[str, Any]]:
    try:
        service = ServiceContainer.get(name, default=None)
    except (OSError, ConnectionError, TimeoutError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
        record_degradation(
            "cognitive_situation",
            exc,
            severity="warning",
            action=f"treated {name} as unavailable while building situation frame",
        )
        return False, {}
    return service is not None, _read_status(service)


def _social_affect(summary: dict[str, Any], name: str) -> tuple[float, float]:
    hypotheses = summary.get("affect_hypotheses")
    if not isinstance(hypotheses, dict):
        return 0.0, 0.0
    hypothesis = hypotheses.get(name)
    if not isinstance(hypothesis, dict):
        return 0.0, 0.0
    return (
        _clamp(_safe_float(hypothesis.get("value"), 0.0)),
        _clamp(_safe_float(hypothesis.get("confidence"), 0.0)),
    )


class CognitiveSituationEngine:
    """Build a unified live-turn situation frame without side effects."""

    def __init__(self) -> None:
        self.frames_built = 0
        self.last_frame: CognitiveSituationFrame | None = None

    def get_status(self) -> dict[str, Any]:
        latest = self.last_frame.to_dict() if self.last_frame is not None else None
        return {
            "running": True,
            "frames_built": self.frames_built,
            "latest": latest,
            "governance": {
                "side_effect_free": True,
                "external_effects_require_authority_gateway": True,
                "claims_require_receipts": True,
            },
        }

    status = get_status

    def frame(
        self,
        objective: str,
        *,
        state: Any = None,
        context: dict[str, Any] | None = None,
        origin: str = "system",
        is_background: bool = False,
    ) -> CognitiveSituationFrame:
        text = " ".join(str(objective or "").split())
        lower = text.lower()
        words = _keywords(text)
        context = context if isinstance(context, dict) else {}

        semantic_hits = len(_SEMANTIC_RE.findall(lower))
        analogy_hits = len(_ANALOGY_RE.findall(lower))
        sensor_hits = len(_SENSORIMOTOR_RE.findall(lower))
        action_hits = len(_ACTION_RE.findall(lower))
        uncertainty_hits = len(_UNCERTAINTY_RE.findall(lower))
        social_hits = len(_SOCIAL_RE.findall(lower))

        curiosity = _affect_value(state, "curiosity", "wonder", "interest", default=0.0)
        confusion = _affect_value(state, "confused", "uncertainty", default=0.0)
        frustration = _affect_value(state, "frustration", "upset", default=0.0)

        live_desktop = bool(
            context.get("desktop_cognitive_engine_required")
            or context.get("desktop_quick_reply_contract")
            or str(origin or "").startswith(("desktop", "voice", "user"))
        )
        live_mind_required = bool(context.get("live_mind_context_required"))
        has_screen_context = bool(
            context.get("screen_context")
            or context.get("desktop_task_contract")
            or context.get("desktop_execution_contract")
        )
        perception_summary = self._perception_summary()
        social_summary = self._social_summary(context)
        fusion_summary = perception_summary.get("multimodal_fusion")
        fusion_summary = fusion_summary if isinstance(fusion_summary, dict) else {}
        fusion_frame_available = bool(fusion_summary.get("frame_id"))
        fusion_confidence = _clamp(_safe_float(fusion_summary.get("confidence"), 0.0))
        fusion_uncertainty = _clamp(_safe_float(fusion_summary.get("uncertainty"), 1.0))
        unresolved_sensor_conflicts = max(
            0,
            int(_safe_float(fusion_summary.get("unresolved_contradictions"), 0.0)),
        )
        social_confidence = _clamp(_safe_float(social_summary.get("confidence"), 0.0))
        social_uncertainty = 1.0 - social_confidence
        social_rupture = _clamp(
            _safe_float(social_summary.get("social_rupture_risk"), 0.0)
        )
        frustration_estimate, frustration_confidence = _social_affect(
            social_summary,
            "frustration",
        )
        urgency_estimate, urgency_confidence = _social_affect(social_summary, "urgency")
        fatigue_estimate, fatigue_confidence = _social_affect(social_summary, "fatigue")
        recommendation = social_summary.get("recommendation")
        recommendation = recommendation if isinstance(recommendation, dict) else {}
        social_repair = _clamp(
            max(
                social_rupture * (0.5 + 0.5 * social_confidence),
                frustration_estimate * frustration_confidence,
                0.65 if recommendation.get("slow_down") and social_confidence >= 0.25 else 0.0,
            )
        )
        social_brevity = bool(
            recommendation.get("be_concise")
            and max(urgency_confidence, fatigue_confidence) >= 0.20
        )

        semantic = _clamp(
            0.10
            + 0.10 * min(4, semantic_hits)
            + 0.10 * min(2, uncertainty_hits)
            + 0.15 * curiosity
            + 0.20 * confusion
            + (0.10 if len(words) >= 6 else 0.0)
        )
        analogy = _clamp(
            0.04
            + 0.14 * min(4, analogy_hits)
            + 0.18 * curiosity
            + (0.12 if semantic >= 0.42 else 0.0)
            + (0.08 if "fictional" in lower or "inspiration" in lower else 0.0)
        )
        sensorimotor = _clamp(
            0.04
            + 0.12 * min(5, sensor_hits)
            + 0.10 * min(3, action_hits)
            + (0.14 if live_desktop else 0.0)
            + (0.12 if has_screen_context else 0.0)
            + (0.12 * fusion_confidence if fusion_frame_available else 0.0)
        )
        ambiguity = _clamp(
            0.08
            + 0.12 * min(3, uncertainty_hits)
            + 0.16 * confusion
            + (0.10 if len(words) >= 8 else 0.0)
            + (0.08 if "or" in lower or "maybe" in lower else 0.0)
            + (0.25 * fusion_uncertainty if fusion_frame_available else 0.0)
            + 0.06 * min(3, unresolved_sensor_conflicts)
            + (0.12 if live_desktop and sensor_hits and not fusion_frame_available else 0.0)
            + (0.12 * social_uncertainty if social_hits else 0.0)
        )
        abstraction = _clamp(0.15 + 0.16 * min(4, semantic_hits) + 0.12 * min(3, analogy_hits))
        verification = _clamp(
            max(
                0.10 + 0.15 * min(3, uncertainty_hits) + 0.20 * sensorimotor,
                0.35 if action_hits else 0.0,
                0.42 if live_mind_required and sensorimotor >= 0.25 else 0.0,
                (
                    0.30 + 0.45 * fusion_uncertainty
                    if fusion_frame_available and (action_hits or sensor_hits)
                    else 0.0
                ),
                0.58 if action_hits and unresolved_sensor_conflicts else 0.0,
                0.62 if action_hits and social_repair >= 0.50 else 0.0,
            )
        )
        metacognition = _clamp(
            0.20
            + 0.35 * ambiguity
            + 0.15 * semantic
            + 0.10 * frustration
            + 0.20 * social_repair
        )

        embodied_affordances = self._embodied_affordances(lower, sensorimotor, perception_summary)
        interpretations = self._semantic_interpretations(text, words, semantic, sensorimotor)
        bridges = self._analogy_bridges(words, analogy, sensorimotor)
        attention_targets = self._attention_targets(
            words,
            semantic=semantic,
            analogy=analogy,
            sensorimotor=sensorimotor,
            ambiguity=ambiguity,
            embodied_affordances=embodied_affordances,
        )
        if social_repair >= 0.45:
            attention_targets.extend(["interaction-repair", "user-boundaries"])
        elif social_hits and social_uncertainty >= 0.55:
            attention_targets.append("social-ambiguity")
        fusion_directives = fusion_summary.get("directives")
        fusion_directives = fusion_directives if isinstance(fusion_directives, dict) else {}
        perception_attention = fusion_directives.get("attention_targets")
        if isinstance(perception_attention, list):
            attention_targets.extend(str(item)[:160] for item in perception_attention[:8] if item)
        attention_targets = list(dict.fromkeys(attention_targets))[:12]
        perception_planning = fusion_directives.get("planning_constraints")
        perception_planning = (
            [str(item)[:160] for item in perception_planning[:8] if item]
            if isinstance(perception_planning, list)
            else []
        )
        perception_repairs = fusion_directives.get("repair_requirements")
        perception_repairs = (
            [str(item)[:160] for item in perception_repairs[:8] if item]
            if isinstance(perception_repairs, list)
            else []
        )
        supplied_social_constraints = social_summary.get("planning_constraints")
        social_constraints = (
            [str(item)[:160] for item in supplied_social_constraints[:8] if item]
            if isinstance(supplied_social_constraints, list)
            else []
        )
        if action_hits and social_repair >= 0.50:
            social_constraints.append(
                "confirm consequential or irreversible action while evidence-backed interaction caution is active"
            )
        if social_hits and social_uncertainty >= 0.65:
            social_constraints.append(
                "treat inferred feelings and intent as hypotheses and clarify only material ambiguity"
            )
        social_constraints = list(dict.fromkeys(social_constraints))[:8]

        routing_bias = {
            "raise_metacognition": metacognition >= 0.35,
            "seek_verification": verification >= 0.35,
            "use_tool_gateway": sensorimotor >= 0.35 or action_hits > 0,
            "preserve_conversation_context": semantic >= 0.35 or ambiguity >= 0.35,
            "use_imagination": analogy >= 0.30,
            "use_analogy": analogy >= 0.35,
            "bind_sensorimotor_evidence": sensorimotor >= 0.30,
            "requires_memory_grounding": semantic >= 0.45 or ambiguity >= 0.42,
            "deliberate_mode": not is_background and (semantic >= 0.52 or ambiguity >= 0.48 or sensorimotor >= 0.62),
            "perception_repair_required": bool(perception_repairs),
            # Abstention keyed on action_hits alone let READ-ONLY perception
            # questions through: "what is on my screen?" has sensor hits and no
            # action hit, so with no fusion frame Aura answered a question about
            # what she could see without being able to see anything. Asking
            # about the world is exactly when the absence of a percept must
            # force abstention — the answer would otherwise be invented.
            "perception_abstention_required": bool(
                unresolved_sensor_conflicts
                or (
                    (action_hits or sensor_hits)
                    and (not fusion_frame_available or fusion_confidence < 0.40)
                )
            ),
            "social_repair_required": social_repair >= 0.50,
            "social_confirmation_required": bool(action_hits and social_repair >= 0.50),
            "social_state_clarification_required": bool(
                social_hits and social_uncertainty >= 0.65
            ),
            "social_response_brevity": social_brevity,
        }
        if is_background:
            routing_bias["deliberate_mode"] = False

        sampling_bias = {
            "temperature_delta": _clamp(
                0.05 * analogy
                + 0.03 * semantic
                - 0.06 * sensorimotor
                - 0.06 * fusion_uncertainty,
                -0.12,
                0.12,
            ),
            "max_tokens_factor": _clamp(
                1.0
                + 0.08 * semantic
                + 0.06 * analogy
                + 0.05 * ambiguity
                - 0.04 * sensorimotor
                - (0.18 if social_brevity else 0.0),
                0.80,
                1.18,
            ),
            "top_p_delta": _clamp(0.03 * analogy - 0.02 * sensorimotor, -0.05, 0.05),
        }

        causal_effects = {
            "semantic_flexibility_pressure": round(semantic, 4),
            "analogical_leap_pressure": round(analogy, 4),
            "sensorimotor_grounding_pressure": round(sensorimotor, 4),
            "verification_pressure": round(verification, 4),
            "metacognition_depth": round(metacognition, 4),
            "attention_focus": attention_targets,
            "tool_governance_pressure": bool(routing_bias["use_tool_gateway"]),
            "multimodal_fusion_frame_id": fusion_summary.get("frame_id", ""),
            "multimodal_confidence": round(fusion_confidence, 4),
            "multimodal_uncertainty": round(fusion_uncertainty, 4),
            "unresolved_sensor_conflicts": unresolved_sensor_conflicts,
            "perception_planning_constraints": perception_planning,
            "perception_repair_requirements": perception_repairs,
            "social_confidence": round(social_confidence, 4),
            "social_uncertainty": round(social_uncertainty, 4),
            "social_repair_pressure": round(social_repair, 4),
            "social_planning_constraints": social_constraints,
            "social_inference_is_hypothesis": True,
            "screen_or_body_evidence_available": bool(
                fusion_summary.get("observed_modalities")
                or perception_summary.get("screen_perception_available")
                or perception_summary.get("embodiment_available")
            ),
        }

        frame = CognitiveSituationFrame(
            frame_id=_stable_id(text, origin, time.time() // 60),
            objective=_compact(text, limit=240),
            semantic_flexibility=semantic,
            analogical_leap_pressure=analogy,
            sensorimotor_grounding=sensorimotor,
            abstraction_level=abstraction,
            ambiguity=ambiguity,
            verification_pressure=verification,
            metacognition_pressure=metacognition,
            social_uncertainty=social_uncertainty,
            social_repair_pressure=social_repair,
            agent_id=_compact(social_summary.get("agent_id"), limit=160),
            keywords=words,
            semantic_interpretations=interpretations,
            analogy_bridges=bridges,
            embodied_affordances=embodied_affordances,
            perception_summary=perception_summary,
            social_summary=social_summary,
            predicted_consequences=(
                dict(social_summary.get("predicted_impacts") or {})
                if isinstance(social_summary.get("predicted_impacts"), dict)
                else {}
            ),
            attention_targets=attention_targets,
            routing_bias=routing_bias,
            sampling_bias=sampling_bias,
            causal_effects=causal_effects,
        )
        self.frames_built += 1
        self.last_frame = frame
        return frame

    def _social_summary(self, context: dict[str, Any]) -> dict[str, Any]:
        """Read the canonical estimator; never trust caller-supplied social claims."""
        try:
            service = optional_service("other_agent_model")
            if service is None or not hasattr(service, "cognitive_snapshot"):
                return {}
            # Fail CLOSED on identity. Falling back to the service's mutable
            # active_agent_id meant that, with no user_id in context, a prior or
            # concurrent user's social model could steer this frame — and be
            # copied into it. A social model is about a specific person; without
            # an identified subject there is no one it is about.
            requested_agent = _compact(context.get("user_id"), limit=160)
            if not requested_agent:
                return {}
            snapshot = service.cognitive_snapshot(requested_agent)
            if not isinstance(snapshot, dict):
                return {}
            if _compact(snapshot.get("agent_id"), limit=160) != requested_agent:
                raise RuntimeError("social estimator returned the wrong agent identity")
            return {
                key: snapshot.get(key)
                for key in (
                    "schema_version",
                    "agent_id",
                    "identity_verified",
                    "identity_scoped",
                    "abstained",
                    "confidence",
                    "freshness_s",
                    "observations",
                    "response_feedback_context",
                    "repair_evidence",
                    "evidence_digest",
                    "affect_hypotheses",
                    "likely_goals",
                    "beliefs_about_aura",
                    "belief_confidence",
                    "social_rupture_risk",
                    "recommendation",
                    "planning_constraints",
                    "predicted_impacts",
                    "inference_limitations",
                    "culture",
                    "power_context",
                    "privacy",
                )
                if key in snapshot
            }
        except (OSError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
            record_degradation(
                "cognitive_situation.social",
                exc,
                severity="warning",
                action="continued without untrusted or unavailable social state",
            )
            return {}

    def _perception_summary(self) -> dict[str, Any]:
        pump_available, pump_status = _service_state("perceptual_pump")
        screen_available, screen_status = _service_state("screen_perception")
        daemon_available, daemon_status = _service_state("perception_daemon")
        runtime_available, runtime_status = _service_state("perception_runtime")
        embodiment_available, embodiment_status = _service_state("embodiment")
        if not embodiment_available:
            embodiment_available, embodiment_status = _service_state("embodiment_system")
        world_available, world_status = _service_state("world_bridge")

        def compact_status(status: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
            return {
                key: status.get(key)
                for key in keys
                if status.get(key) not in (None, "", [], {})
            }

        fusion_status = pump_status.get("fusion")
        fusion_status = fusion_status if isinstance(fusion_status, dict) else {}
        observations = fusion_status.get("observations")
        observations = observations if isinstance(observations, dict) else {}
        missing = fusion_status.get("missing")
        missing = missing if isinstance(missing, dict) else {}
        directives = fusion_status.get("directives")
        directives = directives if isinstance(directives, dict) else {}
        multimodal_fusion = {
            "frame_id": _compact(fusion_status.get("frame_id"), limit=192),
            "confidence": _clamp(_safe_float(fusion_status.get("confidence"), 0.0)),
            "uncertainty": _clamp(_safe_float(fusion_status.get("uncertainty"), 1.0)),
            "observed_modalities": sorted(str(key)[:40] for key in observations)[:12],
            "missing": {
                str(key)[:40]: _compact(value, limit=80)
                for key, value in list(missing.items())[:12]
            },
            "unresolved_contradictions": max(
                0,
                int(_safe_float(fusion_status.get("unresolved_contradictions"), 0.0)),
            ),
            "directives": {
                key: [str(item)[:160] for item in value[:8] if item]
                for key in (
                    "attention_targets",
                    "memory_candidates",
                    "planning_constraints",
                    "repair_requirements",
                )
                if isinstance((value := directives.get(key)), list)
            },
        }

        return {
            "perceptual_pump_available": pump_available,
            "perceptual_pump": compact_status(
                pump_status,
                ("running", "frames_produced", "substrate_injections", "errors", "pump_hz"),
            ),
            "multimodal_fusion": multimodal_fusion,
            "screen_perception_available": screen_available,
            "screen": compact_status(
                screen_status,
                ("active_app", "frontmost_app", "focused_control", "last_capture_at", "permissions_ok"),
            ),
            "perception_daemon_available": daemon_available,
            "perception_daemon": compact_status(
                daemon_status,
                ("running", "healthy", "last_observation_at", "active_app", "last_error"),
            ),
            "perception_runtime_available": runtime_available,
            "perception_runtime": compact_status(
                runtime_status,
                ("running", "healthy", "last_frame_at", "last_error"),
            ),
            "embodiment_available": embodiment_available,
            "embodiment": compact_status(
                embodiment_status,
                ("running", "healthy", "permissions_ok", "last_action_at", "last_error"),
            ),
            "world_bridge_available": world_available,
            "world_bridge": compact_status(
                world_status,
                ("running", "healthy", "last_observation_at", "last_error"),
            ),
        }

    @staticmethod
    def _semantic_interpretations(
        text: str,
        words: list[str],
        semantic: float,
        sensorimotor: float,
    ) -> list[dict[str, Any]]:
        focus = ", ".join(words[:4]) if words else _compact(text, limit=80)
        interpretations = [
            {
                "label": "literal_request",
                "focus": focus or "current user objective",
                "weight": round(_clamp(0.45 + 0.25 * (1.0 - semantic)), 3),
            },
            {
                "label": "conceptual_frame",
                "focus": "meaning, criteria, and implications",
                "weight": round(_clamp(0.20 + 0.50 * semantic), 3),
            },
        ]
        if sensorimotor >= 0.30:
            interpretations.append(
                {
                    "label": "operational_frame",
                    "focus": "visible environment, tool state, and effect evidence",
                    "weight": round(_clamp(0.25 + 0.50 * sensorimotor), 3),
                }
            )
        return interpretations

    @staticmethod
    def _analogy_bridges(
        words: list[str],
        analogy: float,
        sensorimotor: float,
    ) -> list[dict[str, str]]:
        if analogy < 0.22:
            return []
        seed = words[0] if words else "objective"
        bridges = [
            {
                "source": seed,
                "target": "navigation",
                "relation": "treat ambiguous intent like a route with landmarks and checkpoints",
            },
            {
                "source": seed,
                "target": "engineering",
                "relation": "turn claims into interfaces, tests, receipts, and rollback paths",
            },
        ]
        if sensorimotor >= 0.30:
            bridges.append(
                {
                    "source": "screen",
                    "target": "body",
                    "relation": "bind perception to action only through observable affordances",
                }
            )
        return bridges

    @staticmethod
    def _embodied_affordances(
        lower: str,
        sensorimotor: float,
        perception_summary: dict[str, Any],
    ) -> list[str]:
        if sensorimotor < 0.22:
            return []
        affordances: list[str] = []
        if names_any(lower, ("screen", "see", "visible", "look")):
            affordances.append("inspect screen state before claiming what is visible")
        if names_any(lower, ("open", "click", "type", "write", "save", "export")):
            affordances.append("route external actions through governed desktop/tool execution")
        if names_any(lower, ("browser", "chrome", "docs", "notes", "pdf", "folder")):
            affordances.append("verify target app, focus, and artifact path after each step")
        if perception_summary.get("screen_perception_available"):
            affordances.append("use existing screen perception telemetry when available")
        if perception_summary.get("embodiment_available"):
            affordances.append("bind action planning to embodiment permission state")
        return affordances[:6]

    @staticmethod
    def _attention_targets(
        words: list[str],
        *,
        semantic: float,
        analogy: float,
        sensorimotor: float,
        ambiguity: float,
        embodied_affordances: list[str],
    ) -> list[str]:
        targets: list[str] = []
        if semantic >= 0.30:
            targets.append("current-intent-semantics")
        if ambiguity >= 0.30:
            targets.append("ambiguity-resolution")
        if analogy >= 0.30:
            targets.append("cross-domain-bridge")
        if sensorimotor >= 0.30:
            targets.append("sensorimotor-evidence")
        if embodied_affordances:
            targets.append("tool-effect-verification")
        targets.extend(words[:3])
        seen: set[str] = set()
        ordered: list[str] = []
        for target in targets:
            if target and target not in seen:
                seen.add(target)
                ordered.append(target)
        return ordered[:8]


_ENGINE: CognitiveSituationEngine | None = None


def get_cognitive_situation_engine() -> CognitiveSituationEngine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = CognitiveSituationEngine()
    try:
        current = optional_service("cognitive_situation")
        if current is not _ENGINE:
            ServiceContainer.register_instance(
                "cognitive_situation",
                _ENGINE,
                required=False,
                owner="core/brain/cognitive_situation.py",
                registered_by="core.brain.cognitive_situation.get_cognitive_situation_engine",
                required_for="semantic flexibility, analogical routing, and sensorimotor grounding",
                failure_policy="degrade_with_receipt",
            )
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
        record_degradation(
            "cognitive_situation",
            exc,
            severity="warning",
            action="continued without ServiceContainer registration for cognitive situation engine",
        )
    return _ENGINE


#: Sequences that would let a caller-supplied frame field stop being content
#: and start acting as prompt structure or a role turn.
_SITUATION_STRUCTURE_RE = re.compile(
    r"(?i)(?:(?:(?<=\s)|^)#{1,6}\s|```|~~~|<\|[^|]*\|>|"
    r"\b(?:system|assistant|user|human)\s*:)"
)


def _directive_safe(value: Any, limit: int = 200) -> str:
    """Render one frame field as inert prompt text.

    render_cognitive_situation_prompt_block accepts ANY dictionary — there is
    no producer identity or signature on a frame — and interpolates its
    interpretations, affordances, and constraints into text the model reads as
    directives. Anything reaching this function is therefore untrusted.
    """
    text = " ".join(str(value or "").split())
    text = "".join(ch for ch in text if ch == " " or ord(ch) >= 32)
    text = _SITUATION_STRUCTURE_RE.sub(" ", text)
    return " ".join(text.split())[:limit]


def render_cognitive_situation_prompt_block(frame: dict[str, Any], *, compact: bool = False) -> str:
    if not isinstance(frame, dict):
        return ""
    try:
        salience = _safe_float(frame.get("salience"), 0.0)
        if salience < 0.16:
            return ""
        semantic = _safe_float(frame.get("semantic_flexibility"), 0.0)
        analogy = _safe_float(frame.get("analogical_leap_pressure"), 0.0)
        sensorimotor = _safe_float(frame.get("sensorimotor_grounding"), 0.0)
        ambiguity = _safe_float(frame.get("ambiguity"), 0.0)
        social_uncertainty = _safe_float(frame.get("social_uncertainty"), 1.0)
        social_repair = _safe_float(frame.get("social_repair_pressure"), 0.0)
        if compact:
            directives = [
                "COGNITIVE SITUATION FRAME: use semantic alternatives, analogies, and embodiment as causal grounding only.",
            ]
            if semantic >= 0.35 or ambiguity >= 0.35:
                directives.append("Resolve current-turn meaning before continuing older topics.")
            if analogy >= 0.35:
                directives.append("Use a relevant analogy when it helps.")
            if sensorimotor >= 0.30:
                directives.append("Ground screen/tool claims in observed state or receipts.")
            routing = frame.get("routing_bias")
            if isinstance(routing, dict) and routing.get("perception_abstention_required"):
                directives.append(
                    "Perception is incomplete or contested: abstain from unsupported scene claims and gather evidence first."
                )
            if isinstance(routing, dict) and routing.get("social_repair_required"):
                directives.append(
                    "Acknowledge the concrete failure or boundary before advancing the task."
                )
            if isinstance(routing, dict) and routing.get(
                "social_state_clarification_required"
            ):
                directives.append(
                    "Treat user-state inferences as uncertain and clarify only material ambiguity."
                )
            return " ".join(directives) + "\n\n"

        lines = [
            "## COGNITIVE SITUATION FRAME",
            (
                f"Semantic flexibility={semantic:.2f}; analogical pressure={analogy:.2f}; "
                f"sensorimotor grounding={sensorimotor:.2f}; ambiguity={ambiguity:.2f}; "
                f"social uncertainty={social_uncertainty:.2f}; repair pressure={social_repair:.2f}."
            ),
        ]
        interpretations = frame.get("semantic_interpretations") or []
        if isinstance(interpretations, list) and interpretations:
            rendered = "; ".join(
                f"{_directive_safe(item.get('label'), 80)}: "
                f"{_directive_safe(item.get('focus'), 120)}"
                for item in interpretations[:3]
                if isinstance(item, dict)
            )
            if rendered:
                lines.append(f"Candidate interpretations: {rendered}.")
        affordances = frame.get("embodied_affordances") or []
        if isinstance(affordances, list) and affordances:
            lines.append("Embodied affordances: " + ", ".join(
                _directive_safe(a, 100) for a in affordances[:5]) + ".")
        causal = frame.get("causal_effects") or {}
        if isinstance(causal, dict):
            constraints = causal.get("perception_planning_constraints")
            if isinstance(constraints, list) and constraints:
                lines.append("Perception constraints: " + ", ".join(
                    _directive_safe(c, 120) for c in constraints[:5]) + ".")
            repairs = causal.get("perception_repair_requirements")
            if isinstance(repairs, list) and repairs:
                lines.append("Perception repair: " + ", ".join(
                    _directive_safe(r, 120) for r in repairs[:5]) + ".")
            social_constraints = causal.get("social_planning_constraints")
            if isinstance(social_constraints, list) and social_constraints:
                lines.append(
                    "Social constraints: "
                    + ", ".join(_directive_safe(c, 120) for c in social_constraints[:5])
                    + "."
                )
        routing = frame.get("routing_bias") or {}
        if isinstance(routing, dict) and routing.get("perception_abstention_required"):
            lines.append(
                "Perception is incomplete or contested: abstain from unsupported scene claims and gather evidence first."
            )
        if frame.get("social_summary"):
            lines.append(
                "Social state is an uncertain hypothesis: do not diagnose, stereotype, manipulate, or assert hidden intent."
            )
        lines.append(
            "This frame changes routing, sampling, attention, and verification; do not recite it."
        )
        return "\n".join(lines) + "\n\n"
    except (TypeError, ValueError, AttributeError) as exc:
        record_degradation(
            "cognitive_situation",
            exc,
            severity="warning",
            action="skipped malformed cognitive situation prompt block",
        )
        return ""
