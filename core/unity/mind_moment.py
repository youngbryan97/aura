from __future__ import annotations

import hashlib
import time
from typing import Any

from core.container import ServiceContainer

from .unity_state import FragmentationReport, MindMoment, UnityRepairPlan, UnityState


def _clamp(value: Any, lower: float = 0.0, upper: float = 1.0) -> float:
    try:
        return max(lower, min(upper, float(value)))
    except Exception:
        return lower


def _norm(value: Any, limit: int = 180) -> str:
    return " ".join(str(value or "").split()).strip()[:limit]


def _hash_text(value: str, length: int = 16) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="ignore")).hexdigest()[:length]


class MindMomentBuilder:
    """Builds the live active-present frame that proves causal closure."""

    REQUIRED_SUBSYSTEMS = (
        "world",
        "memory",
        "affect",
        "goals",
        "self_model",
        "prediction",
        "unity",
        "will",
    )

    LESION_EXPECTATIONS = {
        "world": "world feedback stops correcting beliefs and future action",
        "memory": "temporal continuity and context-sensitive choice degrade",
        "affect": "preference, urgency, and social stance flatten",
        "goals": "initiative pressure collapses into prompt following",
        "self_model": "self-prediction and identity regulation degrade",
        "prediction": "error-driven learning and policy adjustment weaken",
        "unity": "one active present fragments into disconnected traces",
        "will": "action ownership and receipt invariants disappear",
        "substrate": "temporal drift and embodied modulation disappear",
        "action": "closed-loop consequences stop feeding future state",
    }

    def build(
        self,
        state: Any,
        unity_state: UnityState,
        report: FragmentationReport | None,
        repair_plan: UnityRepairPlan | None,
        *,
        objective: str = "",
        tick_id: str = "",
    ) -> MindMoment:
        focus = self._focus_summary(unity_state, objective)
        attention = _norm(getattr(getattr(state, "cognition", None), "attention_focus", None) or focus)
        feeling = self._feeling(state)
        wanting = self._wanting(state, objective)
        believing = self._believing(state)
        preparing = self._preparing(unity_state, report, repair_plan)
        prediction, prediction_error = self._prediction_snapshot(state, unity_state)
        activations = self._subsystem_activations(
            state,
            unity_state,
            prediction_error=prediction_error,
        )
        active = [name for name, value in activations.items() if value >= 0.15]
        edges = self._causal_edges(activations, unity_state, prediction_error)
        missing = [name for name in self.REQUIRED_SUBSYSTEMS if activations.get(name, 0.0) < 0.15]
        edge_score = min(1.0, len(edges) / 10.0)
        coverage_score = sum(1.0 for name in self.REQUIRED_SUBSYSTEMS if name not in missing) / len(self.REQUIRED_SUBSYSTEMS)
        closure_score = _clamp(
            0.44 * coverage_score
            + 0.24 * edge_score
            + 0.22 * float(unity_state.unity_score)
            + 0.10 * (1.0 - min(1.0, prediction_error))
        )
        continuity_hash = self._continuity_hash(state, unity_state, focus, wanting, believing)
        world_refs = self._refs_by_modality(unity_state, "world")
        memory_refs = self._refs_by_modality(unity_state, "memory")
        grounding_refs = sorted(set(world_refs + self._refs_by_modality(unity_state, "tool")))

        return MindMoment(
            tick_id=tick_id or getattr(unity_state.temporal, "tick_id", None),
            state_version=getattr(state, "version", None),
            unity_id=unity_state.unity_id,
            continuity_hash=continuity_hash,
            focus_summary=focus,
            attention=attention or "unbound",
            feeling=feeling,
            wanting=wanting,
            believing=believing,
            preparing=preparing,
            subsystem_activations=activations,
            causal_edges=edges,
            prediction=prediction,
            prediction_error=prediction_error,
            closure_score=closure_score,
            closure_missing=missing,
            lesion_expectations=dict(self.LESION_EXPECTATIONS),
            active_subsystems=active,
            will_receipt_id=unity_state.will_receipt_id,
            action_receipt_ids=self._action_receipts(state),
            world_evidence_refs=world_refs,
            memory_refs=memory_refs,
            grounding_refs=grounding_refs,
            unity_level=unity_state.level,
            unity_score=float(unity_state.unity_score),
            fragmentation_score=float(unity_state.fragmentation_score),
            safe_to_act=bool(report.safe_to_act if report is not None else True),
            receipt_required=True,
        )

    def _focus_summary(self, unity_state: UnityState, objective: str) -> str:
        for item in unity_state.contents:
            if item.content_id == unity_state.global_focus_id:
                return _norm(item.summary)
        return _norm(objective) or "current cognitive tick"

    def _feeling(self, state: Any) -> str:
        affect = getattr(state, "affect", None)
        if affect is None:
            return "affect unavailable"
        for method in ("get_rich_summary", "get_summary"):
            if hasattr(affect, method):
                try:
                    text = _norm(getattr(affect, method)())
                    if text:
                        return text
                except Exception:
                    pass
        valence = getattr(affect, "valence", None)
        arousal = getattr(affect, "arousal", None)
        if valence is not None or arousal is not None:
            return f"valence={float(valence or 0.0):.2f} arousal={float(arousal or 0.0):.2f}"
        return "affect quiet"

    def _wanting(self, state: Any, objective: str) -> str:
        cognition = getattr(state, "cognition", None)
        goals = list(getattr(cognition, "active_goals", []) or [])
        if goals:
            goal = goals[0]
            if isinstance(goal, dict):
                return _norm(goal.get("objective") or goal.get("goal") or goal.get("title") or goal.get("name"))
            return _norm(goal)
        return _norm(objective or getattr(cognition, "current_objective", "")) or "no dominant goal"

    def _believing(self, state: Any) -> str:
        cognition = getattr(state, "cognition", None)
        rolling = _norm(getattr(cognition, "rolling_summary", ""))
        if rolling:
            return rolling
        memories = list(getattr(cognition, "long_term_memory", []) or [])
        if memories:
            return _norm(memories[0])
        working = list(getattr(cognition, "working_memory", []) or [])
        for item in reversed(working):
            if isinstance(item, dict) and item.get("content"):
                return _norm(item.get("content"))
        return "no durable belief selected"

    def _preparing(
        self,
        unity_state: UnityState,
        report: FragmentationReport | None,
        repair_plan: UnityRepairPlan | None,
    ) -> str:
        if repair_plan is not None and repair_plan.steps:
            return _norm(repair_plan.steps[0])
        if report is not None and not report.safe_to_act:
            return "stabilize before consequential action"
        if unity_state.action_readiness_score >= 0.5:
            return "route next action through Unified Will receipt"
        return "hold action until readiness improves"

    def _prediction_snapshot(self, state: Any, unity_state: UnityState) -> tuple[dict[str, Any], float]:
        prediction: dict[str, Any] = {
            "next_focus": self._focus_summary(unity_state, getattr(getattr(state, "cognition", None), "current_objective", "")),
            "expected_unity_level": unity_state.level,
            "action_readiness": round(float(unity_state.action_readiness_score), 4),
        }
        error = max(0.0, min(1.0, float(unity_state.fragmentation_score)))
        for service_name in ("self_prediction_loop", "predictive_self_model", "free_energy_engine"):
            try:
                if not ServiceContainer.has(service_name):
                    continue
                service = ServiceContainer.get(service_name, default=None)
                if service is None:
                    continue
                snapshot = None
                if hasattr(service, "snapshot"):
                    snapshot = service.snapshot()
                elif hasattr(service, "get_status"):
                    snapshot = service.get_status()
                elif hasattr(service, "current"):
                    snapshot = getattr(service, "current")
                if snapshot is None:
                    continue
                if hasattr(snapshot, "to_dict"):
                    snapshot = snapshot.to_dict()
                if not isinstance(snapshot, dict):
                    snapshot = dict(getattr(snapshot, "__dict__", {}) or {})
                prediction[service_name] = snapshot
                for key in ("prediction_error", "free_energy", "error", "smoothed_error"):
                    if key in snapshot:
                        error = max(error, _clamp(snapshot.get(key)))
                        break
            except Exception:
                continue
        return prediction, error

    def _subsystem_activations(
        self,
        state: Any,
        unity_state: UnityState,
        *,
        prediction_error: float,
    ) -> dict[str, float]:
        cognition = getattr(state, "cognition", None)
        world = getattr(state, "world", None)
        working = list(getattr(cognition, "working_memory", []) or [])
        memories = list(getattr(cognition, "long_term_memory", []) or [])
        goals = list(getattr(cognition, "active_goals", []) or [])
        percepts = list(getattr(world, "recent_percepts", []) or [])
        affect = getattr(state, "affect", None)
        self_model = 0.0
        if getattr(cognition, "phenomenal_state", None) is not None:
            self_model = max(self_model, 0.35)
        if getattr(cognition, "rolling_summary", ""):
            self_model = max(self_model, 0.45)
        if getattr(state, "identity", None) is not None:
            self_model = max(self_model, 0.55)
        substrate = self._service_activity(("continuous_substrate", "liquid_substrate", "substrate_authority", "phi_core"))
        will = 1.0 if unity_state.will_receipt_id else self._service_activity(("unified_will",))
        action = 0.0
        if unity_state.will_receipt_id:
            action = max(action, 0.7)
        action = max(action, _clamp(unity_state.action_readiness_score))
        return {
            "world": _clamp(0.2 + 0.2 * len(percepts), upper=1.0) if percepts else 0.0,
            "memory": _clamp(0.1 * len(working) + 0.18 * len(memories), upper=1.0),
            "affect": 0.65 if affect is not None else 0.0,
            "goals": _clamp(0.35 + 0.18 * len(goals), upper=1.0) if goals or getattr(cognition, "current_objective", "") else 0.0,
            "self_model": _clamp(self_model),
            "prediction": _clamp(0.25 + prediction_error * 0.5),
            "unity": _clamp(unity_state.unity_score),
            "will": _clamp(will),
            "substrate": _clamp(substrate),
            "action": _clamp(action),
            "workspace": _clamp(0.25 + 0.1 * len(unity_state.contents), upper=1.0) if unity_state.contents else 0.0,
        }

    def _service_activity(self, names: tuple[str, ...]) -> float:
        score = 0.0
        for name in names:
            try:
                if not ServiceContainer.has(name):
                    continue
                service = ServiceContainer.get(name, default=None)
            except Exception:
                service = None
            if service is None:
                continue
            score = max(score, 0.45)
            for attr in ("history_length", "current_phi", "is_complex"):
                value = getattr(service, attr, None)
                if callable(value):
                    try:
                        value = value()
                    except Exception:
                        value = None
                if value:
                    score = max(score, 0.65)
        return score

    def _causal_edges(
        self,
        activations: dict[str, float],
        unity_state: UnityState,
        prediction_error: float,
    ) -> list[dict[str, Any]]:
        templates = [
            ("world", "attention", "percepts bind into the active present"),
            ("memory", "attention", "retrieved history biases focus"),
            ("affect", "attention", "valence changes salience"),
            ("goals", "initiative", "persistent goals create action pressure"),
            ("attention", "prediction", "selected focus creates a next-state forecast"),
            ("prediction", "error", "forecast mismatch becomes prediction error"),
            ("error", "adjustment", "prediction error drives state update"),
            ("unity", "will", "integrated present gates action ownership"),
            ("will", "action", "approved intention requires a Will receipt"),
            ("action", "world", "outcome feeds future evidence"),
            ("action", "memory", "outcome becomes continuity evidence"),
        ]
        proxy = {
            "attention": activations.get("workspace", 0.0),
            "initiative": activations.get("goals", 0.0),
            "error": _clamp(prediction_error),
            "adjustment": max(activations.get("memory", 0.0), activations.get("self_model", 0.0)),
        }
        edges: list[dict[str, Any]] = []
        for source, target, evidence in templates:
            source_score = activations.get(source, proxy.get(source, 0.0))
            target_score = activations.get(target, proxy.get(target, 0.0))
            weight = _clamp((source_score + target_score) / 2.0)
            if weight < 0.12:
                continue
            edges.append(
                {
                    "source": source,
                    "target": target,
                    "weight": round(weight, 4),
                    "evidence": evidence,
                    "unity_id": unity_state.unity_id,
                }
            )
        return edges

    def _continuity_hash(
        self,
        state: Any,
        unity_state: UnityState,
        focus: str,
        wanting: str,
        believing: str,
    ) -> str:
        state_hash = ""
        try:
            if hasattr(state, "get_continuity_hash"):
                state_hash = str(state.get_continuity_hash())
        except Exception:
            state_hash = ""
        seed = "|".join(
            [
                state_hash,
                unity_state.unity_id,
                str(unity_state.state_version or ""),
                focus,
                wanting,
                believing,
            ]
        )
        return "mind:" + _hash_text(seed, 24)

    def _refs_by_modality(self, unity_state: UnityState, modality: str) -> list[str]:
        refs: list[str] = []
        for item in unity_state.contents:
            if item.modality == modality or item.source == modality:
                refs.append(item.evidence_ref or item.content_id)
        return refs[:6]

    def _action_receipts(self, state: Any) -> list[str]:
        refs: list[str] = []
        for attr in ("last_will_receipt_id", "last_action_receipt_id", "last_tool_receipt_id"):
            value = getattr(state, attr, None)
            if value:
                refs.append(str(value))
        try:
            recent = getattr(getattr(state, "cognition", None), "recent_action_receipts", [])
            refs.extend(str(item) for item in list(recent or [])[:6])
        except Exception:
            pass
        return sorted(set(refs))[:8]
