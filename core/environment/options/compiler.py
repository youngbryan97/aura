"""Compile selected options into action intents."""
from __future__ import annotations

from .base import Option
from ..command import ActionIntent


class OptionCompiler:
    def compile(self, option: Option, *, target_id: str | None = None) -> ActionIntent:
        mapping = {
            "RESOLVE_MODAL": "resolve_modal",
            "OBSERVE_MORE": "observe",
            "INSPECT_OBJECT": "inspect",
            "STABILIZE_RESOURCE": "stabilize",
            "RETREAT_FROM_HAZARD": "retreat",
            "NAVIGATE_TO_GOAL": "navigate_to",
            "EXPLORE_FRONTIER": "explore_frontier",
            "USE_KNOWN_SAFE_AFFORDANCE": "use",
            "RUN_DIAGNOSTIC": "diagnose",
            "RECOVER_FROM_LOOP": "recover_from_loop",
            "BACKTRACK": "backtrack",
            "SUMMARIZE_CONTEXT": "summarize",
            "SAVE_CHECKPOINT": "snapshot",
            "ROLLBACK_LAST_CHANGE": "rollback",
        }
        return ActionIntent(
            name=mapping.get(option.name, option.policy_name),
            target_id=target_id,
            expected_effect=";".join(option.expected_effects),
            risk="caution" if option.risk_tags else "safe",
            tags=set(option.risk_tags),
        )


__all__ = ["OptionCompiler"]
