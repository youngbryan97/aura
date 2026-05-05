"""Modal-state schema and safe modal resolution."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

ModalKind = Literal[
    "none",
    "prompt",
    "confirmation",
    "menu",
    "form",
    "item_selection",
    "direction_selection",
    "authentication",
    "permission_request",
    "error_dialog",
    "unknown",
]


@dataclass
class ModalState:
    kind: ModalKind
    text: str
    legal_responses: set[str] = field(default_factory=set)
    safe_default: str | None = None
    dangerous_responses: set[str] = field(default_factory=set)
    requires_resolution: bool = True
    confidence: float = 1.0
    source_evidence: str = ""

    def is_known(self) -> bool:
        return self.kind != "unknown" and self.confidence >= 0.5


class ModalManager:
    """Suspends ordinary policy when a modal blocks the environment."""

    def resolve(self, modal_state: ModalState) -> str | None:
        if not modal_state.requires_resolution:
            return None
        if modal_state.safe_default is not None:
            return modal_state.safe_default
        safe_candidates = sorted(modal_state.legal_responses - modal_state.dangerous_responses)
        if safe_candidates:
            return safe_candidates[0]
        return None

    def should_block_normal_policy(self, modal_state: ModalState | None) -> bool:
        return bool(modal_state and modal_state.requires_resolution)


__all__ = ["ModalKind", "ModalState", "ModalManager"]
