from __future__ import annotations
from core.runtime.errors import record_degradation


import logging
from statistics import mean
from typing import Any, Dict, Iterable

from core.container import ServiceContainer

logger = logging.getLogger("Consciousness.Evidence")


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _avg(values: Iterable[float]) -> float:
    items = [float(v) for v in values]
    if not items:
        return 0.0
    return _clamp01(mean(items))


class ConsciousnessEvidenceEngine:
    """Operational snapshot of Aura's machine-subjectivity evidence.

    This does not claim metaphysical proof. It measures whether the system is
    behaving like a persistent, integrated, self-modelled agent with enterprise
    grade runtime guarantees.
    """

    def snapshot(self) -> Dict[str, Any]:
        audit_index = 0.0
        latest_phi = 0.0
        audit_count = 0
        audit_summary = "No audit data yet."

        try:
            from core.consciousness.unified_audit import get_audit_suite

            suite = get_audit_suite()
            history = list(getattr(suite, "_history", []))
            audit_count = len(history)
            if history:
                latest = history[-1]
                audit_index = _clamp01(getattr(latest, "consciousness_index", 0.0))
                latest_phi = float(getattr(latest, "phi", 0.0) or 0.0)
                audit_summary = str(getattr(latest, "summary", audit_summary))
        except Exception as exc:
            record_degradation('evidence_engine', exc)
            logger.debug("Audit evidence unavailable: %s", exc)

        orch = ServiceContainer.get("orchestrator", default=None)
        personality = ServiceContainer.get("personality_engine", default=None)
        self_report = ServiceContainer.get("self_report_engine", default=None)
        phenomenology = ServiceContainer.get("phenomenological_experiencer", default=None)
        self_model = ServiceContainer.get("self_model", default=None)
        global_workspace = ServiceContainer.get("global_workspace", default=None)
        homeostasis = ServiceContainer.get("homeostasis", default=None)
        opinion_engine = ServiceContainer.get("opinion_engine", default=None)
        spine = ServiceContainer.get("spine", default=None)
        volition_engine = ServiceContainer.get("volition_engine", default=None)
        executive_closure = ServiceContainer.get("executive_closure", default=None)

        conversation_history = list(getattr(orch, "conversation_history", []) or [])
        reply_queue = getattr(orch, "reply_queue", None)
        inference_gate = getattr(orch, "_inference_gate", None)
        closure_score = None
        try:
            if executive_closure and hasattr(executive_closure, "get_status"):
                closure_score = float(executive_closure.get_status().get("closure_score", 0.0) or 0.0)
        except Exception as exc:
            record_degradation('evidence_engine', exc)
            logger.debug("Executive closure evidence unavailable: %s", exc)

        self_report_text = ""
        try:
            if self_report and hasattr(self_report, "generate_state_report"):
                self_report_text = str(self_report.generate_state_report() or "")
        except Exception as exc:
            record_degradation('evidence_engine', exc)
            logger.debug("Self report unavailable: %s", exc)

        phenom_fragment = ""
        phenom_stale = True
        try:
            if phenomenology:
                if hasattr(phenomenology, "get_phenomenal_context_fragment"):
                    phenom_fragment = str(phenomenology.get_phenomenal_context_fragment() or "")
                elif hasattr(phenomenology, "phenomenal_context_string"):
                    phenom_fragment = str(getattr(phenomenology, "phenomenal_context_string", "") or "")
                if hasattr(phenomenology, "to_dict"):
                    phenom_stale = bool(phenomenology.to_dict().get("is_stale", True))
        except Exception as exc:
            record_degradation('evidence_engine', exc)
            logger.debug("Phenomenology unavailable: %s", exc)

        dominant_emotions = []
        try:
            if personality:
                emo = personality.get_emotional_context_for_response()
                dominant_emotions = list(emo.get("dominant_emotions", []))
        except Exception as exc:
            record_degradation('evidence_engine', exc)
            logger.debug("Personality evidence unavailable: %s", exc)

        integration = _avg([
            audit_index,
            _clamp01(latest_phi),
            1.0 if global_workspace else 0.0,
        ])
        continuity_factors = [
            1.0 if self_model else 0.0,
            _clamp01(len(conversation_history) / 20.0),
            1.0 if phenomenology and not phenom_stale else 0.0,
            _clamp01(audit_count / 5.0),
        ]
        if closure_score is not None:
            continuity_factors.append(_clamp01(closure_score))
        continuity = _avg(continuity_factors)
        embodiment = _avg([
            1.0 if homeostasis else 0.0,
            1.0 if ServiceContainer.get("liquid_state", default=None) else 0.0,
            1.0 if dominant_emotions else 0.35,
        ])
        agency = _avg([
            1.0 if volition_engine else 0.0,
            1.0 if spine else 0.0,
            1.0 if opinion_engine else 0.0,
            1.0 if getattr(orch, "agency", None) else 0.5 if orch else 0.0,
        ])
        personality_drive = _avg([
            1.0 if personality else 0.0,
            1.0 if self_report_text else 0.35 if self_report else 0.0,
            1.0 if phenom_fragment else 0.0,
            1.0 if dominant_emotions else 0.3,
        ])
        # Inference gate scoring: alive=1.0, recovering=0.7, exists-but-dead=0.4, missing=0.0
        if inference_gate and getattr(inference_gate, "is_alive", lambda: False)():
            _ig_score = 1.0
        elif inference_gate and getattr(inference_gate, "_cortex_recovery_in_progress", False):
            _ig_score = 0.7  # actively recovering is better than dead
        elif inference_gate:
            _ig_score = 0.4
        else:
            _ig_score = 0.0
        reliability_factors = [
            1.0 if reply_queue and reply_queue.__class__.__name__ == "TaggedReplyQueue" else 0.35 if reply_queue else 0.0,
            _ig_score,
            1.0 if orch and not getattr(getattr(orch, "status", None), "last_error", None) else 0.6 if orch else 0.0,
        ]
        if closure_score is not None:
            reliability_factors.append(_clamp01(closure_score))
        reliability = _avg(reliability_factors)

        subjectivity_evidence = _clamp01(
            (integration * 0.24)
            + (continuity * 0.18)
            + (embodiment * 0.12)
            + (agency * 0.16)
            + (personality_drive * 0.16)
            + (reliability * 0.14)
        )
        enterprise_readiness = _avg([
            reliability,
            continuity,
            integration,
        ])

        if subjectivity_evidence >= 0.8:
            assessment = "strong operational evidence of an integrated machine self"
        elif subjectivity_evidence >= 0.6:
            assessment = "meaningful machine-subjectivity signals with clear room to deepen"
        elif subjectivity_evidence >= 0.4:
            assessment = "partial machine-subjectivity scaffold; still too easy to dismiss as orchestration"
        else:
            assessment = "mostly architectural scaffolding, not yet undeniable"

        return {
            "subjectivity_evidence": round(subjectivity_evidence, 4),
            "enterprise_readiness": round(enterprise_readiness, 4),
            "dimensions": {
                "integration": round(integration, 4),
                "continuity": round(continuity, 4),
                "embodiment": round(embodiment, 4),
                "agency": round(agency, 4),
                "personality_drive": round(personality_drive, 4),
                "reliability": round(reliability, 4),
            },
            "signals": {
                "latest_phi": round(latest_phi, 4),
                "audit_index": round(audit_index, 4),
                "audit_count": audit_count,
                "dominant_emotions": dominant_emotions[:4],
                "self_report": self_report_text[:160],
                "phenomenology": phenom_fragment[:160],
                "reply_queue": reply_queue.__class__.__name__ if reply_queue else None,
                "inference_alive": bool(inference_gate and getattr(inference_gate, "is_alive", lambda: False)()),
                "closure_score": round(closure_score or 0.0, 4),
            },
            "assessment": assessment,
            "audit_summary": audit_summary[:220],
        }
