"""core/sim/scenario_forge.py

Scenario Forge  (lineage: Caine — The Amazing Digital Circus)
===========================================================
Caine is the ringmaster AI that *generates* endless immersive worlds and
adventures on demand and improvises them in real time — but cannot address the
humans' real underlying needs (he can't get them out, can't make it matter).

We take both halves. The generator: procedurally build a structured scenario
(setting, agents, events, branches, success criteria) for planning, training, or
creative exploration — model-enriched when a brain is warm, heuristic otherwise.
And Caine's honest limitation, baked in as a feature: every scenario is checked
against whether a *simulation* can actually serve the user's real need, and says
so plainly when it can't. It lives in sim/ beside world_simulator and
scenario_tree.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

from core.runtime.service_registry import get_runtime_service, register_runtime_service
from core.utils.engine_support import coerce_text, record_engine_degradation, resolve_brain

logger = logging.getLogger("Aura.ScenarioForge")


def _degrade(exc: BaseException, *, action: str, severity: str = "warning") -> None:
    record_engine_degradation("scenario_forge", exc, action=action, severity=severity)


# Needs a simulated adventure genuinely cannot resolve — Caine's blind spot.
_REAL_NEED_MARKERS = (
    "lonely", "alone", "escape", "get out", "leave", "stuck", "trapped", "real life",
    "actually help", "really help", "i feel", "i'm scared", "afraid", "grief", "dying",
)


@dataclass
class Scenario:
    title: str
    theme: str
    setting: str
    agents: list[str]
    events: list[str]
    branches: list[str]
    success_criteria: list[str]
    addresses_real_need: bool = True
    caveat: str = ""
    timestamp: float = field(default_factory=time.time)


class ScenarioForge:
    COMPLEXITY_EVENTS = {"low": 3, "medium": 5, "high": 8}

    def __init__(self, orchestrator: Any = None):
        self.orchestrator = orchestrator
        self._forged = 0
        logger.info("🎪 ScenarioForge initialized (Caine lineage)")

    @staticmethod
    def _keywords(text: str, limit: int = 6) -> list[str]:
        stop = {"the", "a", "an", "and", "or", "to", "of", "for", "with", "in", "on", "my", "me"}
        words = [w for w in re.findall(r"[a-zA-Z][a-zA-Z\-]{2,}", text.lower()) if w not in stop]
        out: list[str] = []
        for w in words:
            if w not in out:
                out.append(w)
        return out[:limit] or ["the unknown"]

    def _heuristic_scenario(self, theme: str, agents: int, complexity: str, goal: str | None) -> Scenario:
        kws = self._keywords(theme)
        n_events = self.COMPLEXITY_EVENTS.get(complexity, 5)
        agent_list = [f"agent_{i+1}::{kws[i % len(kws)]}" for i in range(max(1, agents))]
        events = [
            f"Event {i+1}: a development involving {kws[i % len(kws)]}"
            for i in range(n_events)
        ]
        branches = [
            f"If the user pursues {kws[0]}, the scenario escalates toward a {complexity}-stakes choice.",
            f"If the user avoids {kws[-1]}, an alternate path opens that tests a different skill.",
        ]
        criteria = [
            f"Resolve the central tension around {kws[0]}.",
            "Make at least one irreversible-feeling choice and live with it.",
        ]
        if goal:
            criteria.insert(0, f"Achieve the user's stated goal: {goal[:120]}")

        addresses, caveat = self._assess_real_need(theme, goal)
        return Scenario(
            title=f"The {kws[0].title()} Scenario",
            theme=theme[:200],
            setting=f"A constructed space themed around {', '.join(kws[:3])}.",
            agents=agent_list,
            events=events,
            branches=branches,
            success_criteria=criteria,
            addresses_real_need=addresses,
            caveat=caveat,
        )

    @staticmethod
    def _assess_real_need(theme: str, goal: str | None) -> tuple[bool, str]:
        blob = f"{theme} {goal or ''}".lower()
        if any(m in blob for m in _REAL_NEED_MARKERS):
            return (
                False,
                "This looks like a real-world or emotional need. A generated scenario can "
                "rehearse or explore it, but it will not resolve it — that needs real action "
                "or a real conversation, not a simulation.",
            )
        return True, ""

    def forge_fast(
        self,
        theme: str,
        *,
        agents: int = 2,
        complexity: str = "low",
        goal: str | None = None,
    ) -> Scenario:
        """Synchronous heuristic-only scenario (no model call) for idle/background rehearsal."""
        self._forged += 1
        return self._heuristic_scenario(theme, agents, complexity, goal)

    async def forge(
        self,
        theme: str,
        *,
        agents: int = 2,
        complexity: str = "medium",
        goal: str | None = None,
    ) -> Scenario:
        self._forged += 1
        scenario = self._heuristic_scenario(theme, agents, complexity, goal)

        brain = resolve_brain(self.orchestrator)
        if brain is not None and hasattr(brain, "think"):
            try:
                import asyncio

                from core.brain.types import ThinkingMode

                out = coerce_text(await asyncio.wait_for(
                    brain.think(
                        f"Sketch {self.COMPLEXITY_EVENTS.get(complexity, 5)} vivid plot beats for a "
                        f"scenario themed '{theme}'. One per line.",
                        mode=ThinkingMode.FAST, origin="caine", is_background=True,
                    ),
                    timeout=25.0,
                ))
                if out:
                    beats = [ln.strip("-• ").strip() for ln in out.splitlines() if ln.strip()]
                    if beats:
                        scenario.events = beats[: len(scenario.events)] or scenario.events
            except (ImportError, AttributeError, RuntimeError, TypeError, ValueError, TimeoutError) as exc:
                _degrade(exc, action="returned heuristic scenario after model enrichment failed")
        return scenario

    def get_status(self) -> dict[str, Any]:
        return {"scenarios_forged": self._forged, "healthy": True}


_INSTANCE: ScenarioForge | None = None


def get_scenario_forge(orchestrator: Any = None) -> ScenarioForge:
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = ScenarioForge(orchestrator=orchestrator)
    return _INSTANCE


def register_scenario_forge(orchestrator: Any = None) -> ScenarioForge:
    from core.service_names import ServiceNames

    inst = get_runtime_service(ServiceNames.CAINE, default=None) or get_scenario_forge(orchestrator)
    register_runtime_service(ServiceNames.CAINE, inst, required=False, owner="core/sim/scenario_forge.py", registered_by="register_scenario_forge")
    register_runtime_service("caine", inst, required=False, owner="core/sim/scenario_forge.py", registered_by="register_scenario_forge")
    return inst


__all__ = ["Scenario", "ScenarioForge", "get_scenario_forge", "register_scenario_forge"]
