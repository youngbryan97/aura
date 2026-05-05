"""Modal-state schema and safe modal resolution.

Enhancements:
- ``ModalPolicy`` for task-aware modal resolution (intent context informs response).
- ``ModalState.from_prompt_text()`` factory for auto-classifying modals.
- Prompt classification into shop, identification, direction, confirmation, etc.
"""
from __future__ import annotations

import re
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

    @staticmethod
    def from_prompt_text(text: str) -> "ModalState":
        """Auto-classify a modal from raw prompt text."""
        lower = text.lower()

        # Direction selection
        if "direction" in lower or "in what direction" in lower:
            return ModalState(
                kind="direction_selection",
                text=text,
                legal_responses={"h", "j", "k", "l", "y", "u", "b", "n", "\x1b"},
                safe_default="\x1b",
                confidence=0.9,
            )

        # Startup/setup selection menus that should choose the environment's
        # default option rather than escape the run before it starts.
        if "pick a" in lower and any(token in lower for token in ("role", "profession", "class", "option", "profile")):
            return ModalState(
                kind="item_selection",
                text=text,
                legal_responses={"\r", "\n", " "},
                safe_default="\r",
                confidence=0.78,
            )

        # Item selection
        if "what do you want to" in lower or "pick an object" in lower:
            return ModalState(
                kind="item_selection",
                text=text,
                legal_responses={"\x1b"},
                safe_default="\x1b",
                confidence=0.9,
            )

        # Confirmation / dangerous
        dangerous = set()
        if any(p in lower for p in ("really attack", "eat it?", "sacrifice", "destroy", "you are about to")):
            dangerous = {"y"}

        if "[yn]" in lower or "[ynq]" in lower:
            safe_default = "y" if not dangerous and any(token in lower for token in ("is this ok", "is this okay", "confirm setup", "confirm settings")) else "n"
            return ModalState(
                kind="confirmation",
                text=text,
                legal_responses={"y", "n", "\x1b"},
                safe_default=safe_default,
                dangerous_responses=dangerous,
                confidence=0.85,
            )

        # Shop / payment
        if "pay" in lower or "price" in lower or "sell" in lower:
            return ModalState(
                kind="prompt",
                text=text,
                legal_responses={"y", "n", "\x1b"},
                safe_default="n",
                dangerous_responses={"y"},
                confidence=0.8,
            )

        # Menu / inventory
        if "end" in lower or "menu" in lower or "inventory" in lower:
            return ModalState(
                kind="menu",
                text=text,
                legal_responses={"\x1b", " "},
                safe_default="\x1b",
                confidence=0.85,
            )

        # --More-- continuation
        if "--more--" in lower or "press return" in lower:
            return ModalState(
                kind="prompt",
                text=text,
                legal_responses={" ", "\r", "\n"},
                safe_default=" ",
                requires_resolution=True,
                confidence=0.95,
            )

        # Identification
        if "call" in lower or "name" in lower:
            return ModalState(
                kind="prompt",
                text=text,
                legal_responses={"\x1b"},
                safe_default="\x1b",
                confidence=0.7,
            )

        # Unknown
        return ModalState(
            kind="unknown",
            text=text,
            legal_responses={"\x1b"},
            safe_default="\x1b",
            confidence=0.4,
        )


class ModalManager:
    """Suspends ordinary policy when a modal blocks the environment."""

    def __init__(self):
        self.policy = ModalPolicy()

    def resolve(self, modal_state: ModalState, intent_name: str | None = None, intent_parameters: dict | None = None) -> str | None:
        return self.policy.resolve_with_intent(
            modal_state, 
            intent_name=intent_name, 
            intent_parameters=intent_parameters
        )

    def should_block_normal_policy(self, modal_state: ModalState | None) -> bool:
        return bool(modal_state and modal_state.requires_resolution)


class ModalPolicy:
    """Task-aware modal resolver that uses intent context to pick responses."""

    def resolve_with_intent(
        self,
        modal_state: ModalState,
        *,
        intent_name: str | None = None,
        intent_parameters: dict | None = None,
    ) -> str | None:
        """Resolve a modal using intent context for smarter responses."""
        if not modal_state.requires_resolution:
            return None
        params = intent_parameters or {}

        # If the intent was item-targeted and the modal asks for an item, use it
        if modal_state.kind == "item_selection" and "item_letter" in params:
            letter = str(params["item_letter"])[0]
            if letter in modal_state.legal_responses or not modal_state.legal_responses:
                return letter

        # If the intent was direction-targeted and modal asks for direction, use it
        if modal_state.kind == "direction_selection" and "direction" in params:
            from core.environments.terminal_grid.nethack_commands import NetHackCommandCompiler
            dir_key = NetHackCommandCompiler.direction_keys.get(str(params["direction"]).lower())
            if dir_key:
                return dir_key

        # Confirmation modals: approve if intent is known-safe, reject otherwise
        if modal_state.kind == "confirmation":
            if intent_name in {"eat", "pray", "pickup", "loot"} and "y" not in modal_state.dangerous_responses:
                return "y"
            if modal_state.safe_default is not None and modal_state.safe_default not in modal_state.dangerous_responses:
                return modal_state.safe_default
            return "n"

        # Fall back to safe default
        if modal_state.safe_default is not None:
            return modal_state.safe_default
        safe_candidates = sorted(modal_state.legal_responses - modal_state.dangerous_responses)
        return safe_candidates[0] if safe_candidates else None


__all__ = ["ModalKind", "ModalState", "ModalManager", "ModalPolicy"]
