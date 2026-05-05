"""Episode lifecycle and terminal state management."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from core.environment.parsed_state import ParsedState


@dataclass
class TerminalState:
    """Represents the end of an episode (e.g. death, crash, completion)."""
    reason: str
    is_success: bool
    context: dict[str, Any]


class LifecycleManager:
    """Treats external tasks as episodes, handling death detection, postmortems, and restarts."""

    def __init__(self, run_id: str):
        self.run_id = run_id
        self.is_terminal = False
        self.terminal_state: TerminalState | None = None
        self.log = logging.getLogger(__name__)

    def check_terminal_state(self, parsed_state: ParsedState) -> bool:
        """Evaluates the state to determine if the episode has ended."""
        if self.is_terminal:
            return True
            
        # Example: generic semantic event detection
        # Specific adapters (like NetHack) might yield "death" or "victory"
        # Browser adapters might yield "browser_crashed" or "task_completed"
        for event in parsed_state.semantic_events:
            if event.name in ("death", "system_crash", "ban"):
                self.is_terminal = True
                self.terminal_state = TerminalState(
                    reason=event.name,
                    is_success=False,
                    context=event.properties
                )
                self.log.warning(f"Episode terminated: {event.name}")
                return True
                
            if event.name in ("victory", "task_completed", "success"):
                self.is_terminal = True
                self.terminal_state = TerminalState(
                    reason=event.name,
                    is_success=True,
                    context=event.properties
                )
                self.log.info(f"Episode completed successfully: {event.name}")
                return True
                
        return False

    def trigger_postmortem(self) -> dict[str, Any]:
        """Gathers data and logs a postmortem summary of the episode."""
        if not self.is_terminal or not self.terminal_state:
            return {}
            
        summary = {
            "run_id": self.run_id,
            "success": self.terminal_state.is_success,
            "reason": self.terminal_state.reason,
            "context": self.terminal_state.context,
        }
        self.log.info(f"Postmortem generated for run {self.run_id}: {summary}")
        return summary

__all__ = ["TerminalState", "LifecycleManager"]
