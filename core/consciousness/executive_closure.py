from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional

import numpy as np

from core.container import ServiceContainer
from core.goals.goal_text import is_actionable_goal_text, is_intrinsic_goal_text
from core.predictive.predictive_self_model import PredictiveSelfModel
from core.runtime.proposal_governance import propose_governed_initiative_to_state
from core.state.aura_state import _origin_is_user_anchored

logger = logging.getLogger("Aura.ExecutiveClosure")


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


@dataclass
class ExecutiveClosureSnapshot:
    timestamp: float = 0.0
    dominant_need: str = "stability"
    need_pressure: float = 0.0
    attention_focus: str = ""
    workspace_source: str = ""
    selected_objective: str = ""
    free_energy: float = 0.0
    phi_estimate: float = 0.0
    vitality: float = 1.0
    prediction_error: float = 0.0
    closure_score: float = 0.0
    active_goal_count: int = 0
    pending_initiatives: int = 0
    motivation_pressures: Dict[str, float] = field(default_factory=dict)


class ExecutiveClosureEngine:
    """Bind Aura's distributed loops into a single recurrent control surface.

    This layer does not replace the existing heartbeat, workspace, volition, or
    homeostatic systems. It fuses their outputs into the authoritative runtime
    state that MindTick phases consume downstream.
    """

    _SELF_MODEL_SYNC_INTERVAL_S = 30.0
    _GOAL_SYNC_INTERVAL_S = 20.0
    _VOLITION_SEED_INTERVAL_S = 8.0
    _HOMEOSTASIS_PULSE_INTERVAL_S = 2.0
    _HOMEOSTASIS_PULSE_TIMEOUT_S = 0.35
    _BOOT_WARMUP_CYCLES = 12

    _OBJECTIVE_TEMPLATES = {
        "stability": "Stabilize runtime load and preserve continuous cognition.",
        "integrity": "Protect identity, memory integrity, and process continuity.",
        "curiosity": "Investigate the most novel unresolved pattern in the current context.",
        "social": "Seek clearer social grounding and relational understanding.",
        "growth": "Consolidate learning into durable improvements.",
    }

    def __init__(self):
        self._predictive_self = PredictiveSelfModel(dim=32)
        self._last_snapshot = ExecutiveClosureSnapshot()
        self._last_self_model_sync = 0.0
        self._last_goal_sync = 0.0
        self._last_volition_seed = 0.0
        self._last_homeostasis_pulse = 0.0
        self._cached_homeostasis_status: Dict[str, float] = {}
        self._self_model_sync_task: Optional[asyncio.Task[Any]] = None

    async def integrate(self, state: Any) -> Any:
        now = time.time()
        warmup_mode = int(getattr(state, "loop_cycle", 0) or 0) < self._BOOT_WARMUP_CYCLES
        background_commitment = ""

        current_objective = str(getattr(state.cognition, "current_objective", "") or "").strip()
        current_origin = str(getattr(state.cognition, "current_origin", "") or "")
        if (
            current_objective
            and is_intrinsic_goal_text(current_objective)
            and not _origin_is_user_anchored(current_origin)
        ):
            background_commitment = current_objective
            state.cognition.current_objective = None

        homeostasis_status = await self._get_homeostasis_status(warmup=warmup_mode)
        closed_loop_status = self._get_closed_loop_status()
        workspace_snapshot = self._get_workspace_snapshot()

        prediction_error = self._observe_state_vector(state, closed_loop_status, homeostasis_status)
        pressures = self._compute_pressures(state, homeostasis_status, closed_loop_status, prediction_error)
        interaction_signals = self._get_interaction_signals_status()
        fused_interaction = dict(interaction_signals.get("fused", {}) or {})
        if fused_interaction:
            engagement = _clamp01(float(fused_interaction.get("engagement", 0.0) or 0.0))
            hesitation = _clamp01(float(fused_interaction.get("hesitation", 0.0) or 0.0))
            attention = _clamp01(float(fused_interaction.get("attention_available", 0.5) or 0.5))
            pressures["social"] = max(float(pressures.get("social", 0.0) or 0.0), engagement * 0.72)
            pressures["stability"] = max(float(pressures.get("stability", 0.0) or 0.0), hesitation * 0.48)
            pressures["curiosity"] = max(float(pressures.get("curiosity", 0.0) or 0.0), engagement * attention * 0.45)
        dominant_need, need_pressure = max(
            pressures.items(),
            key=lambda item: item[1],
        ) if pressures else ("stability", 0.0)

        selected_objective = self._select_objective(
            state,
            dominant_need=dominant_need,
            need_pressure=need_pressure,
            workspace_snapshot=workspace_snapshot,
        )
        active_goal_count = self._sync_active_goals(state, selected_objective)

        if is_actionable_goal_text(selected_objective) and not getattr(state.cognition, "current_objective", None):
            state, decision = await propose_governed_initiative_to_state(
                state,
                selected_objective,
                orchestrator=None,
                source="executive_closure",
                kind="executive_closure",
                urgency=max(0.45, min(0.98, need_pressure)),
                triggered_by=dominant_need,
                metadata={
                    "workspace_focus": workspace_snapshot.get("last_content"),
                    "workspace_source": workspace_snapshot.get("last_winner"),
                    "dominant_need": dominant_need,
                    "need_pressure": round(need_pressure, 4),
                },
            )
            logger.debug("ExecutiveClosure: queued selected objective via executive authority (%s).", decision.get("reason"))

        _ws_content = str(workspace_snapshot.get("last_content") or "")
        _ws_source = str(workspace_snapshot.get("last_winner") or "")
        # Never promote internal housekeeping (baseline ticks, drive alerts)
        # into attention_focus — it leaks into user-facing fallback responses.
        _internal_sources = {"baseline_continuity", "drive_growth", "drive_social"}
        if _ws_content and _ws_source not in _internal_sources and "baseline tick" not in _ws_content.lower():
            state.cognition.attention_focus = _ws_content

        if is_actionable_goal_text(selected_objective):
            state.cognition.modifiers["executive_objective"] = selected_objective
        else:
            state.cognition.modifiers.pop("executive_objective", None)
        if background_commitment:
            state.cognition.modifiers["executive_background_commitment"] = background_commitment
        else:
            state.cognition.modifiers.pop("executive_background_commitment", None)
        state.cognition.modifiers["executive_dominant_need"] = dominant_need
        state.cognition.modifiers["executive_need_pressure"] = round(need_pressure, 4)
        state.cognition.modifiers["prediction_error"] = round(prediction_error, 4)

        state.loop_cycle = int(closed_loop_status.get("cycle_count") or (state.loop_cycle + 1))
        state.free_energy = float(closed_loop_status.get("free_energy", state.free_energy))
        phi_estimate = float(closed_loop_status.get("phi_estimate", 0.0))
        state.phi_estimate = phi_estimate
        state.phi = phi_estimate
        state.vitality = float(homeostasis_status.get("will_to_live", state.vitality))

        state.response_modifiers["executive_closure"] = {
            "dominant_need": dominant_need,
            "need_pressure": round(need_pressure, 4),
            "attention_focus": state.cognition.attention_focus,
            "workspace_source": workspace_snapshot.get("last_winner"),
            "selected_objective": selected_objective,
            "background_commitment": background_commitment,
            "prediction_error": round(prediction_error, 4),
            "free_energy": round(state.free_energy, 4),
            "phi_estimate": round(phi_estimate, 4),
            "vitality": round(state.vitality, 4),
            "interaction_signals": interaction_signals,
        }

        if not state.cognition.phenomenal_state:
            state.cognition.phenomenal_state = (
                f"Attention converges on {state.cognition.attention_focus or 'internal monitoring'}. "
                f"Dominant need is {dominant_need}. "
                f"Free energy is {state.free_energy:.3f}."
            )

        closure_score = self._compute_closure_score(
            free_energy=state.free_energy,
            phi_estimate=phi_estimate,
            vitality=state.vitality,
            prediction_error=prediction_error,
            has_focus=bool(state.cognition.attention_focus),
            has_objective=bool(selected_objective or state.cognition.current_objective),
        )

        self._last_snapshot = ExecutiveClosureSnapshot(
            timestamp=now,
            dominant_need=dominant_need,
            need_pressure=round(need_pressure, 4),
            attention_focus=str(state.cognition.attention_focus or ""),
            workspace_source=str(workspace_snapshot.get("last_winner") or ""),
            selected_objective=str(selected_objective or state.cognition.current_objective or ""),
            free_energy=round(state.free_energy, 4),
            phi_estimate=round(phi_estimate, 4),
            vitality=round(state.vitality, 4),
            prediction_error=round(prediction_error, 4),
            closure_score=round(closure_score, 4),
            active_goal_count=active_goal_count,
            pending_initiatives=len(getattr(state.cognition, "pending_initiatives", []) or []),
            motivation_pressures={k: round(v, 4) for k, v in pressures.items()},
        )

        self._maybe_sync_self_model(self._last_snapshot, warmup=warmup_mode)
        self._maybe_sync_goal_hierarchy(
            selected_objective,
            dominant_need,
            need_pressure,
            warmup=warmup_mode,
        )

        return state

    def get_status(self) -> Dict[str, Any]:
        return asdict(self._last_snapshot)

    def _observe_state_vector(
        self,
        state: Any,
        closed_loop_status: Dict[str, Any],
        homeostasis_status: Dict[str, float],
    ) -> float:
        vec = np.zeros((32,), dtype=np.float32)

        budgets = getattr(state.motivation, "budgets", {}) or {}

        def budget_level(name: str, default: float = 50.0) -> float:
            payload = budgets.get(name, {})
            capacity = float(payload.get("capacity", 100.0) or 100.0)
            level = float(payload.get("level", default) or default)
            return _clamp01(level / max(1.0, capacity))

        vec[0] = float(getattr(state.affect, "valence", 0.0))
        vec[1] = float(getattr(state.affect, "arousal", 0.5))
        vec[2] = float(getattr(state.affect, "curiosity", 0.5))
        vec[3] = float(getattr(state.affect, "engagement", 0.5))
        vec[4] = float(getattr(state.affect, "social_hunger", 0.5))
        vec[5] = budget_level("energy")
        vec[6] = budget_level("curiosity")
        vec[7] = budget_level("social")
        vec[8] = budget_level("integrity")
        vec[9] = budget_level("growth")
        vec[10] = float(homeostasis_status.get("integrity", 1.0))
        vec[11] = float(homeostasis_status.get("persistence", 1.0))
        vec[12] = float(homeostasis_status.get("metabolism", 1.0))
        vec[13] = float(homeostasis_status.get("sovereignty", 1.0))
        vec[14] = float(homeostasis_status.get("will_to_live", 1.0))
        vec[15] = _clamp01(float(getattr(state.soma.hardware, "get", lambda *_: 0.0)("cpu_usage", 0.0)) / 100.0) if hasattr(state.soma, "hardware") else 0.0
        vec[16] = _clamp01(float(getattr(state.soma.hardware, "get", lambda *_: 0.0)("vram_usage", 0.0)) / 100.0) if hasattr(state.soma, "hardware") else 0.0
        vec[17] = _clamp01(float(getattr(state.soma.hardware, "get", lambda *_: 0.0)("temperature", 0.0)) / 100.0) if hasattr(state.soma, "hardware") else 0.0
        vec[18] = _clamp01(len(getattr(state.world, "recent_percepts", []) or []) / 20.0)
        vec[19] = _clamp01(len(getattr(state.cognition, "working_memory", []) or []) / 20.0)
        vec[20] = _clamp01(len(getattr(state.cognition, "pending_initiatives", []) or []) / 10.0)
        vec[21] = _clamp01(len(getattr(state.cognition, "active_goals", []) or []) / 10.0)
        vec[22] = _clamp01(float(getattr(state, "free_energy", 0.0)))
        vec[23] = _clamp01(float(closed_loop_status.get("free_energy", 0.0)))
        vec[24] = _clamp01(float(closed_loop_status.get("phi_estimate", 0.0)))
        vec[25] = _clamp01(float(getattr(state.cognition, "conversation_energy", 0.5)))
        vec[26] = _clamp01(float(getattr(state.cognition, "discourse_depth", 0)) / 10.0)
        vec[27] = 1.0 if getattr(state.cognition, "current_objective", None) else 0.0
        vec[28] = 1.0 if getattr(state.cognition, "attention_focus", None) else 0.0
        vec[29] = 1.0 if getattr(state.cognition, "phenomenal_state", None) else 0.0
        vec[30] = _clamp01(float(getattr(state, "vitality", 1.0)))
        vec[31] = _clamp01(float(getattr(state, "phi_estimate", 0.0)))

        return float(self._predictive_self.observe_and_update(vec, lr=0.015))

    def _compute_pressures(
        self,
        state: Any,
        homeostasis_status: Dict[str, float],
        closed_loop_status: Dict[str, Any],
        prediction_error: float,
    ) -> Dict[str, float]:
        budgets = getattr(state.motivation, "budgets", {}) or {}

        def deficit(name: str) -> float:
            payload = budgets.get(name, {})
            capacity = float(payload.get("capacity", 100.0) or 100.0)
            level = float(payload.get("level", capacity) or capacity)
            return _clamp01(1.0 - (level / max(1.0, capacity)))

        cpu_pressure = _clamp01(float(getattr(state.soma, "hardware", {}).get("cpu_usage", 0.0)) / 100.0)
        free_energy = _clamp01(float(closed_loop_status.get("free_energy", getattr(state, "free_energy", 0.0)) or 0.0))
        vitality = _clamp01(float(homeostasis_status.get("will_to_live", getattr(state, "vitality", 1.0)) or 1.0))

        # ── SUBSTRATE-DERIVED PRESSURES ──────────────────────────────
        # The unified field, neurochemical state, and interoceptive body
        # are now PRIMARY inputs to pressure computation — not supplements.
        # The executive closure derives its "needs" from the substrate.
        field_coherence = 0.6
        field_valence = 0.0
        chem_stress = 0.0
        chem_motivation = 0.5
        chem_sociality = 0.4
        body_budget = 0.0
        body_energy = 0.5

        try:
            unified_field = ServiceContainer.get("unified_field", default=None)
            if unified_field:
                quality = unified_field.get_experiential_quality()
                field_coherence = quality.get("coherence", 0.6)
                field_valence = quality.get("valence", 0.0)
        except Exception:
            pass

        try:
            ncs = ServiceContainer.get("neurochemical_system", default=None)
            if ncs:
                mood = ncs.get_mood_vector()
                chem_stress = mood.get("stress", 0.0)
                chem_motivation = mood.get("motivation", 0.5)
                chem_sociality = mood.get("sociality", 0.4)
        except Exception:
            pass

        try:
            intero = ServiceContainer.get("embodied_interoception", default=None)
            if intero:
                bb = intero.get_body_budget()
                body_budget = bb.get("budget", 0.0)
                body_energy = bb.get("energy_reserves", 0.5)
        except Exception:
            pass

        # Stability pressure: now includes field incoherence + chemical stress +
        # body energy deficit — the substrate IS the stability signal.
        stability_pressure = max(
            deficit("energy"),
            1.0 - float(homeostasis_status.get("metabolism", 1.0)),
            cpu_pressure,
            free_energy,
            1.0 - vitality,
            1.0 - field_coherence,       # incoherent field = instability
            chem_stress,                  # chemical stress = instability
            1.0 - body_energy,            # low body energy = instability
        )

        integrity_pressure = max(
            deficit("integrity"),
            1.0 - float(homeostasis_status.get("integrity", 1.0)),
            1.0 - float(homeostasis_status.get("sovereignty", 1.0)),
        )

        # Novelty should drive exploration only when the runtime is healthy enough
        # to sustain it. Under stress, closure should bias toward stabilization.
        # Chemical motivation directly modulates exploration headroom.
        survival_guard = max(stability_pressure, integrity_pressure)
        exploration_headroom = max(0.15, min(1.0,
            (1.0 - survival_guard * 0.75) * (0.5 + chem_motivation * 0.5)
        ))
        consolidation_headroom = max(0.2, 1.0 - (survival_guard * 0.6))

        curiosity_drive = max(
            deficit("curiosity"),
            min(1.0, prediction_error * 0.55),
            free_energy * 0.35,
        )
        # Social drive now includes neurochemical sociality (oxytocin/serotonin)
        social_drive = max(
            deficit("social"),
            _clamp01(float(getattr(state.affect, "social_hunger", 0.5))),
            chem_sociality * 0.6,
        )
        growth_drive = max(
            deficit("growth"),
            min(1.0, prediction_error * 0.45),
        )

        return {
            "stability": stability_pressure,
            "integrity": integrity_pressure,
            "curiosity": curiosity_drive * exploration_headroom,
            "social": social_drive * max(0.45, 1.0 - (stability_pressure * 0.45)),
            "growth": growth_drive * consolidation_headroom,
        }

    def _select_objective(
        self,
        state: Any,
        *,
        dominant_need: str,
        need_pressure: float,
        workspace_snapshot: Dict[str, Any],
    ) -> str:
        current_objective = str(getattr(state.cognition, "current_objective", "") or "")
        if current_objective and not is_intrinsic_goal_text(current_objective):
            return current_objective

        goal_hierarchy = ServiceContainer.get("goal_hierarchy", default=None)
        if goal_hierarchy:
            try:
                next_goal = goal_hierarchy.get_next_goal()
                if (
                    next_goal
                    and getattr(next_goal, "description", None)
                    and is_actionable_goal_text(next_goal.description)
                ):
                    return str(next_goal.description)
            except Exception as exc:
                logger.debug("ExecutiveClosure: goal hierarchy lookup failed: %s", exc)

        volition = ServiceContainer.get("volition_engine", default=None)
        if volition:
            try:
                result = getattr(volition, "_last_goal", None)
                if result and result.get("objective") and is_actionable_goal_text(result["objective"]):
                    return str(result["objective"])
            except Exception as _exc:
                logger.debug("Suppressed Exception: %s", _exc)

            if (
                not getattr(state.cognition, "pending_initiatives", [])
                and (time.time() - self._last_volition_seed) >= self._VOLITION_SEED_INTERVAL_S
            ):
                try:
                    loop = asyncio.get_running_loop()
                    # Volition is meaningful but non-authoritative. Use a short task so it
                    # can seed future continuity without parking the current tick.
                    loop.create_task(self._seed_from_volition(volition, current_objective))
                    self._last_volition_seed = time.time()
                except Exception as exc:
                    logger.debug("ExecutiveClosure: volition seed failed: %s", exc)

        workspace_focus = str(workspace_snapshot.get("last_content") or "").strip()
        if (
            workspace_focus
            and float(workspace_snapshot.get("last_priority") or 0.0) >= 0.55
            and is_actionable_goal_text(workspace_focus)
        ):
            return workspace_focus

        return ""

    async def _seed_from_volition(self, volition: Any, current_objective: str) -> None:
        try:
            proposal = await volition.tick(current_objective)
            if not proposal or not proposal.get("objective"):
                return
            volition._last_goal = proposal
        except Exception as exc:
            logger.debug("ExecutiveClosure: volition tick failed: %s", exc)

    def _sync_active_goals(self, state: Any, selected_objective: str) -> int:
        active = [
            goal
            for goal in list(getattr(state.cognition, "active_goals", []) or [])
            if not is_intrinsic_goal_text(goal)
        ]
        if is_actionable_goal_text(selected_objective):
            record = {
                "description": selected_objective,
                "priority": 1.0,
                "source": "executive_closure",
                "timestamp": time.time(),
            }
            if not any(goal.get("description") == selected_objective for goal in active if isinstance(goal, dict)):
                active.insert(0, record)
        state.cognition.active_goals = active[:5]
        return len(getattr(state.cognition, "active_goals", []) or [])

    async def _get_homeostasis_status(self, *, warmup: bool = False) -> Dict[str, float]:
        homeostasis = ServiceContainer.get("homeostasis", default=None)
        if homeostasis is None:
            return dict(self._cached_homeostasis_status)

        now = time.time()
        should_pulse = (
            hasattr(homeostasis, "pulse")
            and not warmup
            and (now - self._last_homeostasis_pulse) >= self._HOMEOSTASIS_PULSE_INTERVAL_S
        )
        try:
            if should_pulse:
                result = await asyncio.wait_for(
                    homeostasis.pulse(),
                    timeout=self._HOMEOSTASIS_PULSE_TIMEOUT_S,
                )
                if isinstance(result, dict):
                    self._cached_homeostasis_status = dict(result)
                    self._last_homeostasis_pulse = now
                    return dict(self._cached_homeostasis_status)
            if hasattr(homeostasis, "get_status"):
                self._cached_homeostasis_status = dict(homeostasis.get_status() or {})
                return dict(self._cached_homeostasis_status)
        except asyncio.TimeoutError:
            logger.debug("ExecutiveClosure: homeostasis pulse timed out; using cached status.")
        except Exception as exc:
            logger.debug("ExecutiveClosure: homeostasis pulse failed: %s", exc)
        return dict(self._cached_homeostasis_status)

    def _get_closed_loop_status(self) -> Dict[str, Any]:
        closed_loop = ServiceContainer.get("closed_causal_loop", default=None)
        if closed_loop is None:
            consciousness = ServiceContainer.get("consciousness", default=None)
            closed_loop = getattr(consciousness, "closed_loop", None) if consciousness else None
        if closed_loop and hasattr(closed_loop, "get_status"):
            try:
                status = closed_loop.get_status()
                return {
                    "cycle_count": int(status.get("loop", {}).get("cycle_count", 0)),
                    "free_energy": float(status.get("free_energy", {}).get("current", 0.0)),
                    "phi_estimate": float(status.get("phi", {}).get("estimate", 0.0)),
                }
            except Exception as exc:
                logger.debug("ExecutiveClosure: closed-loop read failed: %s", exc)
        return {"cycle_count": 0, "free_energy": 0.0, "phi_estimate": 0.0}

    def _get_workspace_snapshot(self) -> Dict[str, Any]:
        workspace = ServiceContainer.get("global_workspace", default=None)
        if workspace and hasattr(workspace, "get_snapshot"):
            try:
                return dict(workspace.get_snapshot() or {})
            except Exception as exc:
                logger.debug("ExecutiveClosure: workspace snapshot failed: %s", exc)
        return {}

    def _get_interaction_signals_status(self) -> Dict[str, Any]:
        interaction_signals = ServiceContainer.get("interaction_signals", default=None)
        if interaction_signals and hasattr(interaction_signals, "get_status"):
            try:
                return dict(interaction_signals.get_status() or {})
            except Exception as exc:
                logger.debug("ExecutiveClosure: interaction signal snapshot failed: %s", exc)
        return {}

    def _compute_closure_score(
        self,
        *,
        free_energy: float,
        phi_estimate: float,
        vitality: float,
        prediction_error: float,
        has_focus: bool,
        has_objective: bool,
    ) -> float:
        return _clamp01(
            (0.2 if has_focus else 0.0)
            + (0.2 if has_objective else 0.0)
            + (_clamp01(vitality) * 0.2)
            + ((1.0 - _clamp01(free_energy)) * 0.2)
            + (_clamp01(phi_estimate) * 0.15)
            + ((1.0 - min(1.0, prediction_error)) * 0.05)
        )

    def _maybe_sync_self_model(self, snapshot: ExecutiveClosureSnapshot, *, warmup: bool = False) -> None:
        if warmup:
            return
        now = time.time()
        if now - self._last_self_model_sync < self._SELF_MODEL_SYNC_INTERVAL_S:
            return
        if self._self_model_sync_task and not self._self_model_sync_task.done():
            return

        self_model = ServiceContainer.get("self_model", default=None)
        if self_model is None:
            return

        payload = {
            "dominant_need": snapshot.dominant_need,
            "attention_focus": snapshot.attention_focus,
            "selected_objective": snapshot.selected_objective,
            "free_energy": snapshot.free_energy,
            "phi_estimate": snapshot.phi_estimate,
            "vitality": snapshot.vitality,
            "closure_score": snapshot.closure_score,
            "timestamp": snapshot.timestamp,
        }

        try:
            self._last_self_model_sync = now
            loop = asyncio.get_running_loop()
            self._self_model_sync_task = loop.create_task(
                self._sync_self_model_payload(self_model, payload)
            )
        except Exception as exc:
            logger.debug("ExecutiveClosure: self-model sync failed: %s", exc)

    async def _sync_self_model_payload(self, self_model: Any, payload: Dict[str, Any]) -> None:
        try:
            if hasattr(self_model, "update_belief"):
                await self_model.update_belief(
                    "executive_closure",
                    payload,
                    note="Continuous closure update",
                )
            elif hasattr(self_model, "beliefs") and isinstance(self_model.beliefs, dict):
                constitutional_runtime_live = (
                    ServiceContainer.has("executive_core")
                    or ServiceContainer.has("aura_kernel")
                    or ServiceContainer.has("kernel_interface")
                    or bool(getattr(ServiceContainer, "_registration_locked", False))
                )
                if constitutional_runtime_live:
                    try:
                        from core.health.degraded_events import record_degraded_event

                        record_degraded_event(
                            "executive_closure",
                            "direct_self_model_sync_suppressed",
                            detail="executive_closure",
                            severity="warning",
                            classification="background_degraded",
                            context={"reason": "update_belief_unavailable"},
                        )
                    except Exception as degraded_exc:
                        logger.debug("ExecutiveClosure degraded-event logging failed: %s", degraded_exc)
                    return
                self_model.beliefs["executive_closure"] = payload
        except Exception as exc:
            logger.debug("ExecutiveClosure: self-model sync task failed: %s", exc)

    def _maybe_sync_goal_hierarchy(
        self,
        selected_objective: str,
        _dominant_need: str,
        need_pressure: float,
        *,
        warmup: bool = False,
    ) -> None:
        if warmup:
            return
        now = time.time()
        if (
            not is_actionable_goal_text(selected_objective)
            or need_pressure < 0.55
            or (now - self._last_goal_sync) < self._GOAL_SYNC_INTERVAL_S
        ):
            return

        goal_hierarchy = ServiceContainer.get("goal_hierarchy", default=None)
        if goal_hierarchy is None or not hasattr(goal_hierarchy, "add_goal"):
            return

        try:
            goal_hierarchy.add_goal(
                selected_objective,
                priority=max(0.6, min(1.0, need_pressure)),
            )
            self._last_goal_sync = now
        except Exception as exc:
            logger.debug("ExecutiveClosure: goal sync failed: %s", exc)
