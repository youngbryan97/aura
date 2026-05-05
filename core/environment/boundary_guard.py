"""Runtime information boundary enforcement."""
from __future__ import annotations

import logging
from dataclasses import dataclass


@dataclass
class BoundaryViolationError(Exception):
    """Raised when an operation attempts to breach the environment information boundary."""
    operation: str
    reason: str


class BoundaryGuard:
    """Intercepts and blocks forbidden operations (e.g. cheating, inspecting internal memory)."""

    def __init__(self):
        self.log = logging.getLogger(__name__)
        # Configurable allowed channels
        self.allowed_channels = {"stdout", "stderr", "display", "audio"}
        # Configurable blocked operations
        self.blocked_operations = {
            "read_save_file", 
            "inspect_memory", 
            "oracle_metadata",
            "inject_state"
        }

    def check_operation(self, operation: str, channel: str | None = None) -> None:
        """Validates if an operation is permitted across the boundary."""
        if operation in self.blocked_operations:
            msg = f"Blocked forbidden operation: {operation}"
            self.log.error(msg)
            raise BoundaryViolationError(operation, "Operation is explicitly blocked by boundary guard.")
            
        if channel and channel not in self.allowed_channels:
            msg = f"Blocked operation {operation} on forbidden channel: {channel}"
            self.log.error(msg)
            raise BoundaryViolationError(operation, f"Channel '{channel}' is not permitted.")

__all__ = ["BoundaryViolationError", "BoundaryGuard"]
