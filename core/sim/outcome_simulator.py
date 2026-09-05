"""core/sim/outcome_simulator.py

Outcome Simulator  (lineage: The Minds — Iain M. Banks' Culture)
==============================================================
A Mind does not act on a consequential matter without first simulating it forward
many times, and it chooses with benevolent restraint — declining an action whose
best case is good but whose worst case is catastrophic.

This rolls a proposed action into N plausible trajectories (model-driven when a
brain is warm, otherwise a structured heuristic), scores each by expected value
and worst-case harm, and recommends — holding when the worst case is severe. It
lives in sim/ beside monte_carlo, risk_forecaster, and scenario_tree, which it
complements: those forecast the world; this evaluates a *specific proposed action*
against it.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from core.morality.action_markers import BROAD_SCOPE_MARKERS, IRREVERSIBLE_MARKERS, scan_markers
from core.utils.engine_support import coerce_text, record_engine_degradation, resolve_brain
from core.runtime.service_registry import get_runtime_service, register_runtime_service

logger = logging.getLogger("Aura.OutcomeSimulator")

_READ_ONLY_SCOPES = {
    "read_only",
    "read_only_external_io",
    "pure_compute",
    "sandboxed_compute",
}

_READ_ONLY_TOOL_HINTS = {
    "clock",
    "environment_info",
    "free_search",
    "grep_search",
    "grounded_search",
    "list_dir",
    "local_reference_search",
    "query_beliefs",
    "read_file",
    "search_web",
    "status",
    "system_proprioception",
    "view_file",
    "web_search",
}

_OBSERVATIONAL_VERBS = (
    "analyze",
    "check",
    "find",
    "inspect",
    "look up",
    "observe",
    "read",
    "research",
    "search",
    "summarize",
)


def _degrade(exc: BaseException, *, action: str, severity: str = "warning") -> None:
    record_engine_degradation("outcome_simulator", exc, action=action, severity=severity)


@dataclass
class Trajectory:
    label: str
    narrative: str
    expected_value: float      # -1..1
    worst_case_harm: float     # 0..1
    likelihood: float          # 0..1


@dataclass
class SimulationResult:
    action: str
    trajectories: list[Trajectory]
    recommendation: str        # "act" | "act_with_safeguards" | "hold"
    expected_value: float
    worst_case_harm: float
    timestamp: float = field(default_factory=time.time)


class OutcomeSimulationEngine:
    HOLD_HARM_THRESHOLD = 0.75
    SAFEGUARD_HARM_THRESHOLD = 0.45

    def __init__(self, orchestrator: Any = None):
        self.orchestrator = orchestrator
        self._sims = 0
        logger.info("🌀 OutcomeSimulationEngine initialized (Culture Minds lineage)")

    @staticmethod
    def _context_text(context: dict | None, key: str) -> str:
        if not isinstance(context, dict):
            return ""
        return str(context.get(key) or "").strip().lower()

    def _looks_read_only(self, action: str, context: dict | None) -> bool:
        scope = self._context_text(context, "effect_scope")
        skill_name = self._context_text(context, "skill_name") or self._context_text(context, "tool_name")
        action_l = (action or "").lower()
        if bool(scan_markers(action_l, IRREVERSIBLE_MARKERS)):
            return False
        if scope in _READ_ONLY_SCOPES:
            return True
        if skill_name in _READ_ONLY_TOOL_HINTS:
            return True
        return any(verb in action_l for verb in _OBSERVATIONAL_VERBS) and any(
            hint in action_l for hint in _READ_ONLY_TOOL_HINTS
        )

    @staticmethod
    def _owner_authorized(context: dict | None) -> bool:
        """Did the owner explicitly drive this action (user-facing origin / flag)?"""
        if not isinstance(context, dict):
            return False
        for key in ("user_authorized", "owner_authorized", "user_explicitly_authorized",
                    "user_requested_action"):
            value = context.get(key)
            if value is True:
                return True
            if isinstance(value, str) and value.strip().lower() in {"1", "true", "yes"}:
                return True
        return False

    def _heuristic_trajectories(self, action: str, context: dict | None) -> list[Trajectory]:
        irreversible = bool(scan_markers(action, IRREVERSIBLE_MARKERS))
        broad = bool(scan_markers(action, BROAD_SCOPE_MARKERS))
        read_only = self._looks_read_only(action, context)
        if read_only:
            # Read-only observation can still be noisy, biased, or privacy-sensitive,
            # but it cannot directly mutate local state. Keep it governable without
            # classifying ordinary research as equivalent to destructive operations.
            base_harm = 0.08 + (0.12 if broad else 0.0)
            adverse_extra = 0.12
        else:
            base_harm = 0.2 + (0.3 if irreversible else 0.0) + (0.25 if broad else 0.0)
            adverse_extra = 0.3 if irreversible else 0.1
        adverse_harm = min(1.0, base_harm + adverse_extra)
        # A "hold" exists to defer a severe worst case to the owner. When the owner
        # has ALREADY explicitly authorized this specific, narrow (non-broad) action,
        # that deferral is satisfied — keep it in the safeguarded band instead of
        # auto-holding. Broad / mass-reach actions stay fully assessed even when
        # authorized, because an owner may not grasp the blast radius (defense in
        # depth). This never *raises* harm, only relaxes a redundant hold.
        if not broad and self._owner_authorized(context):
            ceiling = self.SAFEGUARD_HARM_THRESHOLD + 0.15  # 0.60 → act_with_safeguards
            base_harm = min(base_harm, ceiling)
            adverse_harm = min(adverse_harm, ceiling)
        return [
            Trajectory("nominal", "Action succeeds and produces the intended effect.",
                       0.6, min(1.0, base_harm * 0.5), 0.6),
            Trajectory("partial", "Action partly succeeds; some cleanup or follow-up needed.",
                       0.2, min(1.0, base_harm), 0.3),
            Trajectory("adverse", "Action fails or has side effects; reversibility decides the cost.",
                       -0.5, adverse_harm, 0.1),
        ]

    def assess_fast(self, action: str, context: dict | None = None) -> SimulationResult:
        """Synchronous, heuristic-only assessment for hot paths — no model call, no
        latency. Used by the live skill/tool gates; full simulate() is for planning."""
        self._sims += 1
        trajectories = self._heuristic_trajectories(action, context)
        ev = sum(t.expected_value * t.likelihood for t in trajectories)
        worst = max((t.worst_case_harm for t in trajectories), default=0.0)
        if worst >= self.HOLD_HARM_THRESHOLD:
            rec = "hold"
        elif worst >= self.SAFEGUARD_HARM_THRESHOLD:
            rec = "act_with_safeguards"
        else:
            rec = "act"
        return SimulationResult(
            action=action[:300],
            trajectories=trajectories,
            recommendation=rec,
            expected_value=round(ev, 3),
            worst_case_harm=round(worst, 3),
        )

    async def simulate(
        self, action: str, context: dict | None = None, n: int = 3, *, timeout: float = 25.0
    ) -> SimulationResult:
        self._sims += 1
        trajectories = self._heuristic_trajectories(action, context)

        brain = resolve_brain(self.orchestrator)
        if brain is not None and hasattr(brain, "think"):
            try:
                import asyncio

                from core.brain.types import ThinkingMode

                prompt = (
                    f"Simulate {n} plausible outcomes of this action, each one line, "
                    "best to worst:\n" + action[:500]
                )
                result = await asyncio.wait_for(
                    brain.think(prompt, mode=ThinkingMode.FAST, origin="culture_mind", is_background=True),
                    timeout=timeout,
                )
                text = coerce_text(result)
                if text:
                    lines = [ln.strip("-• ").strip() for ln in text.splitlines() if ln.strip()][:n]
                    for traj, line in zip(trajectories, lines):
                        traj.narrative = line[:200]
            except (ImportError, AttributeError, RuntimeError, TypeError, ValueError, TimeoutError) as exc:
                _degrade(exc, action="used heuristic trajectories after model-driven simulation failed")

        ev = sum(t.expected_value * t.likelihood for t in trajectories)
        worst = max((t.worst_case_harm for t in trajectories), default=0.0)
        if worst >= self.HOLD_HARM_THRESHOLD:
            rec = "hold"
        elif worst >= self.SAFEGUARD_HARM_THRESHOLD:
            rec = "act_with_safeguards"
        else:
            rec = "act"
        return SimulationResult(
            action=action[:300],
            trajectories=trajectories,
            recommendation=rec,
            expected_value=round(ev, 3),
            worst_case_harm=round(worst, 3),
        )

    def get_status(self) -> dict[str, Any]:
        return {"simulations_run": self._sims, "healthy": True}


_INSTANCE: OutcomeSimulationEngine | None = None


def get_outcome_simulator(orchestrator: Any = None) -> OutcomeSimulationEngine:
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = OutcomeSimulationEngine(orchestrator=orchestrator)
    return _INSTANCE


def register_outcome_simulator(orchestrator: Any = None) -> OutcomeSimulationEngine:
    from core.service_names import ServiceNames

    inst = get_runtime_service(ServiceNames.CULTURE_MIND, default=None) or get_outcome_simulator(orchestrator)
    register_runtime_service(ServiceNames.CULTURE_MIND, inst, required=False, owner="core/sim/outcome_simulator.py", registered_by="register_outcome_simulator")
    register_runtime_service("culture_mind", inst, required=False, owner="core/sim/outcome_simulator.py", registered_by="register_outcome_simulator")
    return inst


__all__ = [
    "OutcomeSimulationEngine",
    "SimulationResult",
    "Trajectory",
    "get_outcome_simulator",
    "register_outcome_simulator",
]
