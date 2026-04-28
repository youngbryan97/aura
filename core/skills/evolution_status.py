"""Skill: evolution_status — Aura's self-assessment of her evolutionary progress."""
from core.runtime.errors import record_degradation
import logging
from typing import Any, Dict, Optional
from core.skills.base_skill import BaseSkill
from core.container import ServiceContainer
from pydantic import BaseModel, Field

logger = logging.getLogger("Skills.evolution_status")


class EvolutionInput(BaseModel):
    axis: Optional[str] = Field(None, description="Specific axis to detail (e.g. 'learning', 'resilience').")


class EvolutionStatusSkill(BaseSkill):
    """Reports Aura's evolutionary progress across all 8 axes."""

    name = "evolution_status"
    description = "Check evolutionary progress across self-awareness, ethics, learning, collaboration, embodiment, resilience, emotional/cognitive integration, and exploration."
    input_model = EvolutionInput
    output = "Evolutionary state report."

    def __init__(self):
        self.logger = logger

    async def execute(self, params: EvolutionInput, context: Dict[str, Any]) -> Dict[str, Any]:
        if isinstance(params, dict):
            try:
                params = EvolutionInput(**params)
            except Exception as e:
                record_degradation('evolution_status', e)
                return {"ok": False, "error": f"Invalid input: {e}"}

        evo = ServiceContainer.get("evolution_orchestrator", default=None)
        if not evo:
            return {
                "ok": False,
                "error": "Evolution orchestrator not initialized.",
                "summary": "My evolution tracking system isn't online yet.",
            }

        state = evo.get_state()

        if params.axis:
            axis_data = state.get("axes", {}).get(params.axis)
            if not axis_data:
                return {
                    "ok": True,
                    "summary": f"No data for axis '{params.axis}'. Valid axes: {', '.join(state.get('axes', {}).keys())}",
                }
            return {
                "ok": True,
                "axis": params.axis,
                "level": axis_data["level"],
                "milestones": axis_data["milestones"],
                "blockers": axis_data["blockers"],
                "summary": (
                    f"My {params.axis} axis is at {axis_data['level']:.0%}. "
                    f"Milestones: {', '.join(axis_data['milestones'][-5:]) or 'none yet'}. "
                    f"Blockers: {', '.join(axis_data['blockers']) or 'none'}."
                ),
            }

        # Full report
        axes_lines = []
        for name, data in state.get("axes", {}).items():
            axes_lines.append(f"  {name}: {data['level']:.0%}")

        summary = (
            f"Evolutionary phase: {state['phase']} ({state['overall_progress']:.0%} overall). "
            f"Tick count: {state['tick_count']}.\n" +
            "\n".join(axes_lines)
        )

        return {
            "ok": True,
            "phase": state["phase"],
            "overall_progress": state["overall_progress"],
            "axes": state["axes"],
            "tick_count": state["tick_count"],
            "summary": summary,
        }
