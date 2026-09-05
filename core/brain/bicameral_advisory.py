"""Causal internal advisory layer for live cognition.

This is inspired by bicameral-mind and multi-draft ideas, but it is not an
experience claim and it does not create user-visible "voices." It gives Aura a
bounded internal chamber where distinct perspectives can propose pressures for
attention, verification, memory, creativity, and tool governance. The narrator
then reconciles those proposals into one advisory frame consumed by the normal
CognitiveEngine path.
"""

from __future__ import annotations

import hashlib
import hmac
import math
import re
import time
from collections import deque
from collections.abc import Mapping
from typing import Any

from core.brain.bicameral_contracts import (
    _LEARNABLE_OUTCOMES,
    _OUTCOME_SOURCES,
    _PERSPECTIVE_DIMENSION,
    _PERSPECTIVES,
    HISTORY_RETENTION_S,
    MAX_HISTORY_LIMIT,
    MAX_OBJECTIVE_PREVIEW_CHARS,
    MAX_OBJECTIVE_SCAN_CHARS,
    AdvisoryProposal,
    BicameralFrame,
    _clamp,
    _emotion_observed,
    _emotion_value,
    _freeze,
    _keywords,
    _safe_float,
    _scope_id,
    _sign,
    _single_line,
    _stable_id,
    _thaw,
    classify_advisory_intent,
    validate_bicameral_frame,
)
from core.container import ServiceContainer
from core.runtime.lockdep import checked_lock
from core.runtime.turn_analysis import canonical_turn_text
from core.security.structural_redaction import redact_text

_SINGLETON_LOCK = checked_lock("bicameral_advisory.singleton")


class BicameralAdvisory:
    """Internal proposal-and-reconciliation layer for live cognition."""

    def __init__(self, history_limit: int = 128):
        bounded_limit = max(1, min(MAX_HISTORY_LIMIT, int(history_limit)))
        self._lock = checked_lock("bicameral_advisory.instance", reentrant=True)
        self._history: deque[dict[str, Any]] = deque(maxlen=bounded_limit)
        self._issued_frames: dict[str, BicameralFrame] = {}
        self._feedback_applied: set[str] = set()
        self._history_limit = bounded_limit
        self._reliability: dict[str, float] = {
            "narrator": 0.62,
            "protector": 0.62,
            "explorer": 0.60,
            "critic": 0.64,
            "social": 0.58,
        }

    def advise(
        self,
        objective: str,
        *,
        state: Any = None,
        context: dict[str, Any] | None = None,
        origin: str = "system",
        is_background: bool = False,
    ) -> BicameralFrame:
        raw = objective if isinstance(objective, str) else ""
        truncated = len(raw) > MAX_OBJECTIVE_SCAN_CHARS
        text = " ".join(raw[:MAX_OBJECTIVE_SCAN_CHARS].split())
        words = _keywords(text)
        affect = getattr(state, "affect", None)
        curiosity = _emotion_value(affect, "curiosity", "wonder", "interest", default=0.0)
        arousal = _safe_float(getattr(affect, "arousal", 0.0), 0.0)
        valence = _safe_float(getattr(affect, "valence", 0.0), 0.0)
        frustration = _emotion_value(affect, "frustration", "upset", default=0.0)
        confusion = _emotion_value(affect, "confused", "uncertainty", default=0.0)

        intent = classify_advisory_intent(
            canonical_turn_text(text),
            context=context,
            origin=origin,
            confusion=confusion,
            curiosity=curiosity,
        )
        tool = intent.tool
        uncertain = intent.uncertain
        creative = intent.creative
        self_reflective = intent.self_reflective
        social = intent.social
        load_pressure = bool(
            isinstance(context, Mapping) and context.get("desktop_cognitive_engine_required")
        ) and not is_background
        observed_affect = affect is not None
        measured = tuple(
            name
            for name, value, present in (
                (
                    "curiosity",
                    curiosity,
                    _emotion_observed(affect, "curiosity", "wonder", "interest"),
                ),
                ("arousal", arousal, _emotion_observed(affect, "arousal")),
                ("valence", valence, _emotion_observed(affect, "valence")),
                (
                    "frustration",
                    frustration,
                    _emotion_observed(affect, "frustration", "upset"),
                ),
                (
                    "confusion",
                    confusion,
                    _emotion_observed(affect, "confused", "uncertainty"),
                ),
            )
            if observed_affect and present and math.isfinite(value)
        )
        state_evidence = {
            "affect_present": observed_affect,
            "measured_fields": measured,
            "complete": observed_affect and len(measured) == 5,
            "input_truncated": truncated,
            "objective_type_valid": isinstance(objective, str),
            "basis": "live_affect_and_turn" if observed_affect else "turn_only_affect_unmeasured",
        }

        proposals = [
            self._proposal(
                "narrator",
                0.35 + 0.25 * self_reflective + 0.10 * social,
                "integrate",
                "Reconcile competing internal pressures into one coherent user-facing stance.",
                words[:3] or ["current_objective"],
                {"raise_metacognition": self_reflective or uncertain},
                {"temperature_delta": -0.02},
                {"self_model_update": 0.45 if self_reflective else 0.18},
            ),
            self._proposal(
                "protector",
                0.30 + 0.30 * tool + 0.20 * max(arousal, frustration),
                "bound_effects",
                "Check authority, memory, and verification before claiming external action.",
                ["governance", "verification", *(words[:2] or [])],
                {"use_tool_gateway": tool, "seek_verification": tool or uncertain},
                {"temperature_delta": -0.04, "max_tokens_factor": 0.92},
                {"verification_pressure": 0.55 if tool or uncertain else 0.25},
            ),
            self._proposal(
                "explorer",
                0.25 + 0.35 * creative + 0.20 * curiosity,
                "generate_possibilities",
                "Search for a novel angle, analogy, or hypothesis before settling.",
                ["novelty", *(words[:3] or ["possibility"])],
                {"use_imagination": creative, "expand_options": creative or uncertain},
                {"temperature_delta": 0.05 if creative else 0.01},
                {"creative_pressure": 0.60 if creative else 0.20},
            ),
            self._proposal(
                "critic",
                0.25 + 0.40 * uncertain + 0.10 * (1.0 - max(valence, 0.0)),
                "test_claims",
                "Find the weak assumption and require evidence before certainty.",
                ["assumption", "evidence", *(words[:2] or [])],
                {"raise_metacognition": uncertain, "ask_clarification": uncertain and not tool},
                {"temperature_delta": -0.03},
                {"metacognition_depth": 0.68 if uncertain else 0.42},
            ),
            self._proposal(
                "social",
                0.22 + 0.30 * social,
                "preserve_relationship_context",
                "Keep the user's intent and continuity visible while avoiding canned status language.",
                ["user_intent", *(words[:3] or ["continuity"])],
                {"preserve_conversation_context": social},
                {"temperature_delta": 0.01},
                {"memory_priority": 0.48 if social else 0.24},
            ),
        ]
        proposals = [proposal for proposal in proposals if proposal.salience >= 0.18]
        proposals.sort(key=lambda item: item.salience, reverse=True)
        # All five perspectives are bounded and materially distinct. Dropping
        # the fifth by salience could remove the protector exactly when
        # uncertainty and default curiosity made the other chambers louder.
        top = proposals[: len(_PERSPECTIVES)]

        saliences = [proposal.salience for proposal in top] or [0.0]
        consensus, dissent, reconciliation = self._reconciliation_metrics(top)
        attention = self._merge_attention(top)
        routing = self._merge_routing(top, load_pressure=load_pressure)
        sampling = self._merge_sampling(top, load_pressure=load_pressure)
        causal = self._merge_causal(top, consensus=consensus, dissent=dissent)
        summary = self._summarize(top, uncertain=uncertain, tool=tool, creative=creative)
        redacted_objective, objective_redacted = redact_text(text[:MAX_OBJECTIVE_PREVIEW_CHARS])
        state_evidence["objective_redacted"] = objective_redacted
        frame = BicameralFrame(
            frame_id=_stable_id(f"{time.time_ns()}:{origin}:{hashlib.sha256(text.encode()).hexdigest()}"),
            objective=redacted_objective,
            narrator_summary=summary,
            consensus=consensus,
            dissent=dissent,
            salience=_clamp(max(saliences) if saliences else 0.0),
            proposals=tuple(top),
            attention_targets=tuple(attention),
            routing_bias=_freeze(routing),
            sampling_bias=_freeze(sampling),
            causal_effects=_freeze(causal),
            reconciliation=_freeze(reconciliation),
            state_evidence=_freeze(state_evidence),
            intent_evidence=_freeze(intent.to_dict()),
            scope_id=_scope_id(context, origin),
        )
        object.__setattr__(frame, "integrity", _sign(frame.to_dict(include_integrity=False)))
        now = time.time()
        with self._lock:
            self._prune(now)
            self._issued_frames[frame.frame_id] = frame
            self._history.append(
                {
                    "frame_id": frame.frame_id,
                    "scope_id": frame.scope_id,
                    "created_at": frame.created_at,
                    "salience": frame.salience,
                    "consensus": frame.consensus,
                    "dissent": frame.dissent,
                    "perspectives": tuple(item.perspective for item in frame.proposals),
                    "state_grounded": bool(frame.state_evidence.get("complete")),
                }
            )
        return frame

    def learn_from_feedback(
        self,
        frame: dict[str, Any],
        *,
        reward: float,
        outcome: str,
        outcome_receipt: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not isinstance(frame, dict) or not validate_bicameral_frame(frame):
            return {"learned": False, "reason": "unissued_or_invalid_frame"}
        reward_value = _safe_float(reward, float("nan"))
        if not math.isfinite(reward_value):
            return {"learned": False, "reason": "non_finite_reward"}
        bounded_reward = max(-1.0, min(1.0, reward_value))
        outcome_name = _single_line(outcome, 80)
        if outcome_name not in _LEARNABLE_OUTCOMES:
            return {
                "learned": False,
                "reason": "outcome_is_not_measured_quality_evidence",
                "outcome": outcome_name,
            }
        if not self._valid_outcome_receipt(frame, outcome_name, outcome_receipt):
            return {"learned": False, "reason": "verified_outcome_receipt_required"}
        frame_id = str(frame.get("frame_id") or "")
        with self._lock:
            issued = self._issued_frames.get(frame_id)
            if issued is None or issued.integrity != frame.get("integrity"):
                return {"learned": False, "reason": "frame_not_issued_by_this_advisor"}
            if frame_id in self._feedback_applied:
                return {"learned": False, "reason": "feedback_already_applied"}
            self._feedback_applied.add(frame_id)

        dimensions = outcome_receipt.get("dimensions") if outcome_receipt else {}
        dimensions = dimensions if isinstance(dimensions, Mapping) else {}
        updated: dict[str, float] = {}
        with self._lock:
            for proposal in issued.proposals:
                perspective = proposal.perspective
                if perspective not in _PERSPECTIVES or perspective in updated:
                    continue
                dimension = _PERSPECTIVE_DIMENSION[perspective]
                measured_dimension = _safe_float(dimensions.get(dimension), float("nan"))
                if not math.isfinite(measured_dimension):
                    continue
                current = self._reliability[perspective]
                salience = proposal.salience
                dimension_reward = max(-1.0, min(1.0, measured_dimension))
                delta = 0.04 * bounded_reward * dimension_reward * max(0.25, salience)
                self._reliability[perspective] = _clamp(current + delta, 0.05, 0.95)
                updated[perspective] = round(self._reliability[perspective], 4)
        return {
            "learned": bool(updated),
            "outcome": outcome_name,
            "reward": bounded_reward,
            "reliability": updated,
            "measured_dimensions": sorted(dimensions),
        }

    @staticmethod
    def _valid_outcome_receipt(
        frame: Mapping[str, Any],
        outcome: str,
        receipt: Mapping[str, Any] | None,
    ) -> bool:
        if not isinstance(receipt, Mapping):
            return False
        if receipt.get("frame_id") != frame.get("frame_id") or receipt.get("outcome") != outcome:
            return False
        if receipt.get("observed") is not True or receipt.get("source") not in _OUTCOME_SOURCES:
            return False
        evidence_sha256 = str(receipt.get("evidence_sha256") or "")
        if not re.fullmatch(r"[0-9a-f]{64}", evidence_sha256):
            return False
        dimensions = receipt.get("dimensions")
        if not isinstance(dimensions, Mapping) or not dimensions:
            return False
        for name, value in dimensions.items():
            if name not in set(_PERSPECTIVE_DIMENSION.values()):
                return False
            parsed = _safe_float(value, float("nan"))
            if not math.isfinite(parsed) or not -1.0 <= parsed <= 1.0:
                return False
        body = {key: _thaw(value) for key, value in receipt.items() if key != "integrity"}
        supplied = str(receipt.get("integrity") or "")
        try:
            return bool(supplied) and hmac.compare_digest(supplied, _sign(body))
        except (TypeError, ValueError, OverflowError):
            return False

    def attest_outcome(
        self,
        frame_id: str,
        *,
        outcome: str,
        source: str,
        evidence_sha256: str,
        dimensions: Mapping[str, float],
    ) -> dict[str, Any] | None:
        """Bind already-measured external evidence to one issued frame.

        This does not measure an outcome. Callers must supply the digest of an
        independently retained effect/user/verifier receipt; the attestation
        only prevents that evidence from being replayed against another frame.
        """
        with self._lock:
            if frame_id not in self._issued_frames:
                return None
        body = {
            "frame_id": frame_id,
            "outcome": _single_line(outcome, 80),
            "source": _single_line(source, 80),
            "evidence_sha256": str(evidence_sha256 or "").lower(),
            "dimensions": {str(key): value for key, value in dimensions.items()},
            "observed": True,
        }
        if body["outcome"] not in _LEARNABLE_OUTCOMES or body["source"] not in _OUTCOME_SOURCES:
            return None
        if not re.fullmatch(r"[0-9a-f]{64}", body["evidence_sha256"]):
            return None
        if not body["dimensions"] or any(
            key not in set(_PERSPECTIVE_DIMENSION.values())
            or not math.isfinite(_safe_float(value, float("nan")))
            or not -1.0 <= _safe_float(value, float("nan")) <= 1.0
            for key, value in body["dimensions"].items()
        ):
            return None
        body["integrity"] = _sign(body)
        return body

    def get_status(self) -> dict[str, Any]:
        with self._lock:
            self._prune(time.time())
            last = dict(self._history[-1]) if self._history else None
            return {
                "status": "active",
                "advisory_only": True,
                "history_size": len(self._history),
                "issued_frames": len(self._issued_frames),
                "history_retention_s": HISTORY_RETENTION_S,
                "history_contains_objectives": False,
                "reliability": dict(sorted(self._reliability.items())),
                "last_frame": last,
            }

    def _prune(self, now: float) -> None:
        while self._history and now - float(self._history[0]["created_at"]) > HISTORY_RETENTION_S:
            expired = self._history.popleft()
            frame_id = str(expired.get("frame_id") or "")
            self._issued_frames.pop(frame_id, None)
            self._feedback_applied.discard(frame_id)
        while len(self._issued_frames) >= self._history_limit:
            oldest = next(iter(self._issued_frames))
            self._issued_frames.pop(oldest, None)
            self._feedback_applied.discard(oldest)

    def _proposal(
        self,
        perspective: str,
        salience: float,
        stance: str,
        directive: str,
        attention_targets: list[str],
        routing_bias: dict[str, bool],
        sampling_bias: dict[str, float],
        state_pressure: dict[str, float],
    ) -> AdvisoryProposal:
        with self._lock:
            reliability = self._reliability.get(perspective, 0.5)
        weighted_salience = _clamp(salience * (0.75 + 0.5 * reliability))
        return AdvisoryProposal(
            perspective=perspective,
            salience=weighted_salience,
            stance=stance,
            directive=directive,
            attention_targets=tuple(attention_targets[:4]),
            routing_bias=_freeze(routing_bias),
            sampling_bias=_freeze(sampling_bias),
            state_pressure=_freeze(state_pressure),
        )

    @staticmethod
    def _reconciliation_metrics(
        proposals: list[AdvisoryProposal],
    ) -> tuple[float, float, dict[str, Any]]:
        if len(proposals) < 2:
            return 1.0, 0.0, {
                "method": "pairwise_behavioral_jaccard_v1",
                "pair_count": 0,
                "pairwise_agreement_mean": 1.0,
            }

        def features(proposal: AdvisoryProposal) -> set[str]:
            found = {f"route:{key}" for key, value in proposal.routing_bias.items() if value}
            found.update(
                f"pressure:{key}"
                for key, value in proposal.state_pressure.items()
                if _safe_float(value, 0.0) >= 0.35
            )
            temperature = _safe_float(proposal.sampling_bias.get("temperature_delta"), 0.0)
            if temperature:
                found.add("sampling:warm" if temperature > 0 else "sampling:cool")
            found.update(f"attention:{target}" for target in proposal.attention_targets[:2])
            return found

        agreements: list[float] = []
        pair_labels: list[str] = []
        for index, left in enumerate(proposals):
            left_features = features(left)
            for right in proposals[index + 1 :]:
                right_features = features(right)
                union = left_features | right_features
                agreements.append(len(left_features & right_features) / len(union) if union else 1.0)
                pair_labels.append(f"{left.perspective}:{right.perspective}")
        consensus = _clamp(sum(agreements) / len(agreements))
        dissent = _clamp(1.0 - consensus)
        return consensus, dissent, {
            "method": "pairwise_behavioral_jaccard_v1",
            "pair_count": len(agreements),
            "pairwise_agreement_mean": round(consensus, 4),
            "pairwise_agreement_min": round(min(agreements), 4),
            "pairs": tuple(pair_labels),
        }

    @staticmethod
    def _merge_attention(proposals: list[AdvisoryProposal]) -> list[str]:
        seen: set[str] = set()
        merged: list[str] = []
        for proposal in proposals:
            for target in proposal.attention_targets:
                clean = str(target or "").strip().lower().replace(" ", "_")
                if not clean or clean in seen:
                    continue
                seen.add(clean)
                merged.append(clean)
                if len(merged) >= 6:
                    return merged
        return merged

    @staticmethod
    def _merge_routing(
        proposals: list[AdvisoryProposal],
        *,
        load_pressure: bool,
    ) -> dict[str, bool]:
        routing: dict[str, bool] = {}
        for proposal in proposals:
            for key, value in proposal.routing_bias.items():
                routing[key] = routing.get(key, False) or bool(value)
        if load_pressure:
            routing["compact_foreground"] = True
        if routing.get("use_tool_gateway"):
            routing["seek_verification"] = True
        return routing

    @staticmethod
    def _merge_sampling(
        proposals: list[AdvisoryProposal],
        *,
        load_pressure: bool,
    ) -> dict[str, float]:
        temperature_delta = 0.0
        token_factor = 1.0
        for proposal in proposals:
            temperature_delta += _safe_float(proposal.sampling_bias.get("temperature_delta"), 0.0)
            token_factor *= _safe_float(proposal.sampling_bias.get("max_tokens_factor"), 1.0)
        if load_pressure:
            token_factor *= 0.92
        return {
            "temperature_delta": round(max(-0.12, min(0.12, temperature_delta)), 4),
            "max_tokens_factor": round(max(0.70, min(1.12, token_factor)), 4),
        }

    @staticmethod
    def _merge_causal(
        proposals: list[AdvisoryProposal],
        *,
        consensus: float,
        dissent: float,
    ) -> dict[str, Any]:
        baseline: dict[str, float] = {
            "metacognition_depth": 0.35,
            "verification_pressure": 0.20,
            "memory_priority": 0.20,
            "creative_pressure": 0.15,
            "self_model_update": 0.15,
        }

        def merge(selected: list[AdvisoryProposal]) -> dict[str, float]:
            pressure = dict(baseline)
            for proposal in selected:
                for key, value in proposal.state_pressure.items():
                    pressure[key] = max(pressure.get(key, 0.0), _safe_float(value, 0.0))
            if dissent >= 0.22:
                pressure["metacognition_depth"] = max(pressure["metacognition_depth"], 0.64)
                pressure["verification_pressure"] = max(pressure["verification_pressure"], 0.45)
            return pressure

        pressure = merge(proposals)
        attribution: dict[str, dict[str, float]] = {}
        for proposal in proposals:
            lesioned = merge([item for item in proposals if item is not proposal])
            effects = {
                key: round(value - lesioned.get(key, baseline.get(key, 0.0)), 4)
                for key, value in pressure.items()
                if value - lesioned.get(key, baseline.get(key, 0.0)) > 1e-9
            }
            if effects:
                attribution[proposal.perspective] = effects
        result: dict[str, Any] = {key: round(value, 4) for key, value in pressure.items()}
        result.update(
            {
                "consensus": round(consensus, 4),
                "dissent": round(dissent, 4),
                "counterfactual_method": "leave_one_proposal_out_v1",
                "causally_validated": True,
                "attribution": attribution,
            }
        )
        return result

    @staticmethod
    def _summarize(
        proposals: list[AdvisoryProposal],
        *,
        uncertain: bool,
        tool: bool,
        creative: bool,
    ) -> str:
        leaders = ", ".join(proposal.perspective for proposal in proposals[:3])
        pressures: list[str] = []
        if tool:
            pressures.append("route effects through governance")
        if uncertain:
            pressures.append("increase metacognition")
        if creative:
            pressures.append("keep a novel option alive")
        if not pressures:
            pressures.append("answer coherently from the current thread")
        return f"Internal advisory reconciliation ({leaders}): " + "; ".join(pressures) + "."


def render_bicameral_prompt_block(frame: dict[str, Any] | BicameralFrame, *, compact: bool = False) -> str:
    """Render a bounded prompt block for the reconciled internal advisory frame."""
    if not validate_bicameral_frame(frame):
        return ""
    if isinstance(frame, BicameralFrame):
        payload = frame.to_dict()
    elif isinstance(frame, dict):
        payload = frame
    else:
        return ""

    salience = _safe_float(payload.get("salience"), 0.0)
    if salience < 0.18:
        return ""

    summary = _single_line(payload.get("narrator_summary"), 260)
    attention = payload.get("attention_targets") or []
    causal = payload.get("causal_effects") or {}
    routing = payload.get("routing_bias") or {}
    proposals = payload.get("proposals") or []
    if not isinstance(attention, list):
        attention = []
    if not isinstance(causal, dict):
        causal = {}
    if not isinstance(routing, dict):
        routing = {}
    if not isinstance(proposals, list):
        proposals = []

    lines = [
        "## BICAMERAL ADVISORY",
        "Private internal proposal/reconciliation layer. This is not a claim of voices, spirits, hallucinated authority, or phenomenal proof.",
    ]
    if summary:
        lines.append(f"- Narrator reconciliation: {summary[:260]}")
    if attention:
        rendered_attention = ", ".join(
            _single_line(item, 48) for item in attention[:5] if _single_line(item, 48)
        )
        if rendered_attention:
            lines.append(f"- Attention targets: {rendered_attention}.")
    if routing.get("use_tool_gateway"):
        lines.append("- External effects must route through governed tools and require effect evidence before claiming completion.")
    if routing.get("seek_verification"):
        lines.append("- Verification is elevated; distinguish checked facts/actions from hypotheses.")
    if routing.get("raise_metacognition"):
        lines.append("- Metacognition is elevated; inspect assumptions before answering.")
    if routing.get("use_imagination") or routing.get("expand_options"):
        lines.append("- Keep one novel alternative alive while preserving the user's actual intent.")
    if not compact:
        rendered_proposals: list[str] = []
        for proposal in proposals[:4]:
            if not isinstance(proposal, dict):
                continue
            perspective = _single_line(proposal.get("perspective") or "unknown", 32)
            stance = _single_line(proposal.get("stance") or "advise", 48)
            directive = _single_line(proposal.get("directive"), 140)
            rendered_proposals.append(f"{perspective}: {stance}; {directive}")
        if rendered_proposals:
            lines.append("- Proposals: " + " | ".join(rendered_proposals))
    meta_depth = _safe_float(causal.get("metacognition_depth"), 0.0)
    verification = _safe_float(causal.get("verification_pressure"), 0.0)
    memory = _safe_float(causal.get("memory_priority"), 0.0)
    creative = _safe_float(causal.get("creative_pressure"), 0.0)
    self_model = _safe_float(causal.get("self_model_update"), 0.0)
    causal_parts = []
    if meta_depth >= 0.50:
        causal_parts.append(f"metacognition={meta_depth:.2f}")
    if verification >= 0.35:
        causal_parts.append(f"verification={verification:.2f}")
    if memory >= 0.35:
        causal_parts.append(f"memory={memory:.2f}")
    if creative >= 0.35:
        causal_parts.append(f"creative={creative:.2f}")
    if self_model >= 0.35:
        causal_parts.append(f"self_model={self_model:.2f}")
    if causal_parts:
        lines.append("- Causal pressures: " + ", ".join(causal_parts) + ".")
    return "\n".join(lines).strip() + "\n\n"


_BICAMERAL_ADVISORY: BicameralAdvisory | None = None


def get_bicameral_advisory() -> BicameralAdvisory:
    global _BICAMERAL_ADVISORY
    existing = ServiceContainer.get("bicameral_advisory", default=None)
    if existing is not None:
        if not isinstance(existing, BicameralAdvisory):
            raise RuntimeError(
                "bicameral_advisory is registered to an incompatible owner; refusing overwrite"
            )
        with _SINGLETON_LOCK:
            _BICAMERAL_ADVISORY = existing
        return existing
    with _SINGLETON_LOCK:
        existing = ServiceContainer.get("bicameral_advisory", default=None)
        if existing is not None:
            if not isinstance(existing, BicameralAdvisory):
                raise RuntimeError(
                    "bicameral_advisory is registered to an incompatible owner; refusing overwrite"
                )
            _BICAMERAL_ADVISORY = existing
            return existing
        if _BICAMERAL_ADVISORY is None:
            _BICAMERAL_ADVISORY = BicameralAdvisory()
        ServiceContainer.register_instance(
            "bicameral_advisory",
            _BICAMERAL_ADVISORY,
            required=False,
        )
        return _BICAMERAL_ADVISORY


__all__ = [
    "AdvisoryProposal",
    "BicameralAdvisory",
    "BicameralFrame",
    "get_bicameral_advisory",
    "render_bicameral_prompt_block",
    "validate_bicameral_frame",
]
