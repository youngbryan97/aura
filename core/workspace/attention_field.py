"""core/workspace/attention_field.py
Attention field mapping active focus scopes.
"""
from typing import Any


class AttentionField:
    """Manages active focus slots mapped to workspace sensors/goals."""

    def __init__(self):
        self._slots: dict[str, Any] = {}

    def assign_attention_slot(self, slot_name: str, target_details: Any) -> None:
        self._slots[slot_name] = target_details

    def get_focused_slots(self) -> dict[str, Any]:
        return self._slots

    def clear_slot(self, slot_name: str) -> None:
        self._slots.pop(slot_name, None)
