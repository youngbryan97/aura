"""The evidence side of the honesty check, as a runtime fact rather than a read.

Every guard in the honesty layer needs the same input: did anything verifiably
happen in the world on THIS turn? Until now each guard answered that question
differently. ``unverified_effect_correction`` was handed the desktop task's own
receipt list, which exists only inside the desktop lane, so on every other lane
— the fast conversational path, the protected foreground route, a reply
composed after a planner declined — the answer defaulted to "no receipts" by
absence of a channel rather than by absence of an effect. And the general
reply path had no access to it at all: the only effect check that reached
``_stabilize_user_facing_reply`` was the filesystem one, which can see a
claimed *path* and nothing else.

``core/runtime/turn_outcome.py`` already defines the right structure. It has
``declare_effect`` and ``observe_effect``, a ``VerificationGrade`` that
distinguishes "a component said so" from "something looked", a ContextVar so
concurrent turns cannot read each other's evidence, and a ``TurnOutcome`` bound
around the live chat turn in core/brain/cognitive_engine.py.

Neither method had a single production caller. The structure was correct and
empty — a writer with no reader, which is the defect class this codebase has
now hit often enough to name. This module is the wiring: lanes that cause
effects record them here, and guards that need to know ask here.

The grade matters and is not cosmetic. ``receipt["ok"]`` from the desktop lane
is genuinely strong — it is true only when the tool returned success AND
``_verify_step_effect`` independently confirmed the effect — so it lands as
OBSERVED. A tool that merely returned success lands as ASSERTED, and
``is_confirmed`` refuses to count it. That distinction is the reason a false
completion claim is catchable at all.
"""

from __future__ import annotations

import logging
from typing import Any

from core.runtime.turn_outcome import VerificationGrade, current_turn

__all__ = [
    "record_verified_effects",
    "turn_effect_evidence",
    "turn_has_verified_effect",
    "request_expects_action",
]

logger = logging.getLogger(__name__)


def record_verified_effects(receipts: Any, *, lane: str = "desktop_task") -> int:
    """Put this lane's step receipts onto the current turn's effect ledger.

    Returns how many verified effects were recorded, so a caller can log the
    number without re-deriving it. Absent a bound turn this is a no-op: a
    background tick has no user-facing reply to keep honest.
    """

    outcome = current_turn()
    if outcome is None or not isinstance(receipts, list):
        return 0
    recorded = 0
    for index, receipt in enumerate(receipts):
        if not isinstance(receipt, dict):
            continue
        action = str(receipt.get("action") or "").strip()
        if not action:
            continue
        name = f"{lane}:{action}:{receipt.get('index', index)}"
        evidence = str(receipt.get("effect_evidence") or "").strip()
        result = receipt.get("result") if isinstance(receipt.get("result"), dict) else {}
        try:
            from core.epistemics.effect_registry import effect_spec

            spec = effect_spec(action)
            fields = tuple(spec.evidence_fields) if spec is not None else ()
        except (ImportError, AttributeError, TypeError, ValueError):
            fields = ()
        object_ref = ""
        for field in fields:
            candidate = result.get(field, receipt.get(field))
            if candidate not in (None, "", [], {}):
                object_ref = str(candidate)
                break
        observed_content = ""
        if action in {"read_screen_text", "inspect_screen"}:
            observed_content = str(
                result.get("text")
                or result.get("summary")
                or receipt.get("text")
                or receipt.get("summary")
                or ""
            )
        failure_detail = str(
            receipt.get("error")
            or result.get("error")
            or receipt.get("effect_evidence")
            or result.get("status")
            or "step did not verify"
        ).strip()
        try:
            from core.conversation.surface_disposition import record_tool_receipt

            record_tool_receipt(
                lane,
                action=action,
                object_ref=object_ref,
                ok=bool(receipt.get("ok")),
                effect_observed=bool(receipt.get("ok")),
                verification="postcondition_verified" if receipt.get("ok") else "failed",
                evidence=evidence or failure_detail,
                observed_content=observed_content,
            )
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            logger.debug("turn surface receipt unavailable for %s: %s", action, exc)
        try:
            if receipt.get("ok"):
                # ok is the conjunction: the tool returned success AND the
                # effect verified independently. That is what OBSERVED means.
                outcome.observe_effect(
                    name,
                    evidence or f"{action} verified",
                    verification=VerificationGrade.OBSERVED,
                )
                recorded += 1
            else:
                outcome.declare_effect(name, action)
                outcome.observe_effect(
                    name,
                    failure_detail,
                    verification=VerificationGrade.ASSERTED,
                )
                # A failed postcondition is still an observation of what
                # happened. Put its retry class on the same turn record so the
                # finalizer reports a known failure instead of "never
                # observed". Missing retryability remains retryable: another
                # attempt may succeed unless the child contract says it cannot.
                outcome.record_error(
                    failure_detail,
                    retryable=result.get("retryable") is not False,
                )
        except RuntimeError:
            # The turn finalized underneath us. Late evidence cannot change a
            # decided turn, and refusing it is the ledger working as designed.
            logger.debug("turn outcome already finalized; effect %s not recorded", name)
            break
    return recorded


def turn_effect_evidence() -> dict[str, Any]:
    """What this turn can prove it did, for a guard about to trust a reply."""

    outcome = current_turn()
    if outcome is None:
        return {
            "bound": False,
            "verified_effects": 0,
            "declared_effects": 0,
            "has_verified_effect": False,
            "confirmed": [],
        }
    effects = outcome.effects()
    confirmed = [effect for effect in effects if effect.is_confirmed]
    return {
        "bound": True,
        "verified_effects": len(confirmed),
        "declared_effects": len(effects),
        "has_verified_effect": bool(confirmed),
        "confirmed": [
            {"name": effect.name, "observed": str(effect.observed or "")[:200]}
            for effect in confirmed[:32]
        ],
    }


def turn_has_verified_effect() -> bool:
    """Whether anything on this turn is entitled to be reported as done.

    False when no turn is bound, which is the conservative direction: a guard
    asking this is about to decide whether a completion claim has evidence, and
    "we lost the ledger" is not evidence.
    """

    outcome = current_turn()
    if outcome is None:
        return False
    return any(effect.is_confirmed for effect in outcome.effects())


def request_expects_action(user_message: Any) -> bool:
    """Whether the person asked for something to be DONE, not answered.

    Delegates to the runtime's own turn classifier rather than matching
    phrasings here. ``analyze_turn`` is what routing already consults, so the
    honesty guard and the router agree about what kind of turn this is by
    construction — a second opinion in a second module is how the two drift.
    """

    text = str(user_message or "").strip()
    if not text:
        return False
    try:
        from core.runtime.turn_analysis import analyze_turn

        analysis = analyze_turn(text)
    except (ImportError, AttributeError, RuntimeError, ValueError, TypeError) as exc:
        logger.debug("turn analysis unavailable for action-expectation check: %s", exc)
        return False
    if str(getattr(analysis, "intent_type", "")).upper() in {"SKILL", "TASK"}:
        return True
    try:
        from core.conversation.request_mood import assess_request_mood

        return bool(assess_request_mood(text).asks_for_action)
    except (ImportError, AttributeError, RuntimeError, ValueError, TypeError) as exc:
        logger.debug("request mood unavailable for action-expectation check: %s", exc)
        return False
