"""core/goals/standing_objective.py

One authority for the question: "may this text become a STANDING objective?"

A standing objective is durable volitional state — it survives the turn that
produced it, gets promoted into initiatives and goals, drives the executive
closure's imperative display, and re-enters cognition at every restore. Live
evidence (July 2026) shows three text classes repeatedly leaking into that
role and resurrecting themselves through the persistence loop:

1. **Ephemeral conversation turns** — "Ok. Once more. You with me?" spent
   weeks as CURRENT IMPERATIVE at urgency 0.98 because a check-in question
   was bound as current_objective and every downstream organ trusted it.
2. **Control-contract scaffolds** — "[EMBODIED CONTROL CONTRACT] ..." prompt
   frames from the embodied NetHack lane persisted as active goals.
3. **Non-linguistic renders** — raw dungeon-map framebuffers became the
   committed executive objective (origin `embodied_motor_reflex`), which no
   chat-turn classifier can catch because they are not language at all.

Quarantine-after-the-fact proved insufficient: the executive loop recreated
the goal faster than provenance repair abandoned it. The durable fix is to
refuse these texts at EVERY ingress into standing state — selection,
proposal, arbitration, restore, and store repair — via this one predicate.
"""
from __future__ import annotations

import logging
import re
from typing import Any

from core.autonomy.research_goal_filter import (
    is_stale_or_prompt_scaffold_goal,
    normalize_goal_text,
)
from core.state.readings_of_her import is_a_reading_of_her

logger = logging.getLogger("Aura.StandingObjective")

# Standing imperatives are a sentence or two. Raw transcripts, contract
# frames, and screen dumps run longer; the live leaked artifacts measured
# 839 (map render) and 4096 (embodied contract) characters.
_MAX_STANDING_OBJECTIVE_CHARS = 600

# Below this alphabetic-character fraction, text of meaningful length is a
# render/diagram, not language. The live leaked map measured 0.29; ordinary
# English prose measures 0.75-0.85.
_MIN_ALPHA_RATIO = 0.35
_MIN_LENGTH_FOR_ALPHA_TEST = 80

_CONTROL_CONTRACT_MARKERS = (
    "[embodied control contract]",
    "[somatic control contract]",
    "[control contract]",
)
_ACTION_MARKER_RE = re.compile(r"\[action:\s*[a-z0-9_ .-]+\]", re.IGNORECASE)
# Frame borders: long unbroken runs of box/ruler characters.
_FRAME_RUN_RE = re.compile(r"[-+|=_~]{16,}")
# Screen rows: interior runs of 4+ spaces (framebuffer column padding).
_SCREEN_ROW_RE = re.compile(r"\S\s{4,}\S")


def _raw_text(value: Any) -> str:
    """Extract the raw (whitespace-preserving) text of a goal-like value."""
    if isinstance(value, dict):
        for key in ("goal", "description", "title", "objective", "content", "name", "text"):
            candidate = value.get(key)
            if candidate:
                return str(candidate)
        return ""
    return str(value or "")


def _alpha_ratio(text: str) -> float:
    if not text:
        return 1.0
    return sum(1 for ch in text if ch.isalpha()) / len(text)


def standing_objective_rejection_reason(value: Any) -> str:
    """Explain why text may not become durable volitional state ("" = valid).

    The raw string matters for render detection (whitespace geometry is the
    signal), so normalization happens per check rather than up front.
    """
    raw = _raw_text(value)
    text = normalize_goal_text(raw)
    if not text:
        return "empty"
    lowered = raw.casefold()
    if any(marker in lowered for marker in _CONTROL_CONTRACT_MARKERS):
        return "control_contract_scaffold"
    if _ACTION_MARKER_RE.search(raw):
        return "control_contract_scaffold"
    if _FRAME_RUN_RE.search(raw):
        return "nonlinguistic_render"
    screen_rows = len(_SCREEN_ROW_RE.findall(raw))
    if screen_rows >= 3:
        return "nonlinguistic_render"
    if len(raw) >= _MIN_LENGTH_FOR_ALPHA_TEST and _alpha_ratio(raw) < _MIN_ALPHA_RATIO:
        return "nonlinguistic_render"
    if len(text) > _MAX_STANDING_OBJECTIVE_CHARS:
        return "overlong_raw_text"
    if is_stale_or_prompt_scaffold_goal(text):
        return "prompt_scaffold"
    if is_a_reading_of_her(text):
        return "a_reading_of_her_state"
    try:
        from core.goals.objective_lifecycle import is_ephemeral_conversation_turn

        if is_ephemeral_conversation_turn(text):
            return "ephemeral_conversation_turn"
    except (ImportError, AttributeError, RecursionError) as exc:
        # Fail open on classifier availability: an unclassifiable text is not
        # thereby a chat turn, and the structural checks above already ran.
        logger.debug("Standing-objective chat classification unavailable: %s", exc)
    return ""


def is_valid_standing_objective(value: Any) -> bool:
    """True only for text that may legitimately persist as durable volition."""
    return standing_objective_rejection_reason(value) == ""


# Rejection classes that hold even for explicitly-bound durable work (task
# rows, commitments): a screen render or contract scaffold is garbage no
# matter what ids are attached. Chat-turn and prompt-scaffold texts, by
# contrast, can legitimately label a dispatched task that tracks a user
# request — explicit bindings keep those.
_STRUCTURAL_REJECTIONS = frozenset(
    {"empty", "nonlinguistic_render", "control_contract_scaffold", "overlong_raw_text", "a_reading_of_her_state"}
)


def standing_objective_rejection_for_bound_work(value: Any) -> str:
    """Rejection reason that applies even to explicitly-bound durable work."""
    reason = standing_objective_rejection_reason(value)
    return reason if reason in _STRUCTURAL_REJECTIONS else ""


__all__ = [
    "is_valid_standing_objective",
    "standing_objective_rejection_for_bound_work",
    "standing_objective_rejection_reason",
]
