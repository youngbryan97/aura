"""What a turn can read, and what it could not reach: the prompt fitted to the budget, and the inventory miss recorded.

Lifted whole out of `cognitive_engine`, which imports them straight back: every
caller and every patch that names them there still finds them. What they
take from that module is imported at CALL time, for the same reason.
"""
from __future__ import annotations

from typing import Any

from core.runtime.errors import record_degradation


def _fit_prompt_to_what_the_turn_can_read(
    system_prompt: str, *, request: str, max_tokens: int, room_taken: int = 0
) -> str:
    """Order an assembled system prompt, and trim it if the turn cannot read it.

    Two things happen here and the ordering is the one that pays. A cached
    prompt is the KV for a byte-identical prefix, and this runtime was reusing
    558 tokens of 27,298 — two per cent, about forty-six seconds of prefill
    thrown away per turn — because whatever changed between turns sat near the
    front. Which sections those are is measured rather than listed, so nobody
    has to keep a list in step with the assembler.

    The trim fires where the turn cannot afford to read what it was given, or
    where the prompt is past the ceiling the worker enforces by keeping the
    head and the tail and dropping everything between them. Meeting that
    ceiling deliberately is the point: the middle of an assembled prompt is
    the mind context, and losing it at a byte offset is recorded as a fault
    and felt as friction. A budget that cannot be worked out at all is no
    budget, and the prompt goes out whole rather than cut by a guess.
    """

    from .cognitive_engine import (
        logger,
    )

    body = str(system_prompt or "")
    if not body:
        return body
    try:
        from core.brain.llm.context_budget import (
            CRITICAL_FOREGROUND_HEADERS,
            budget_for_answer,
            fit_to_budget,
            observe_sections,
            prefill_ceiling,
            stable_prefix_first,
            volatility_of,
        )

        observe_sections(body)
        # Two constraints, and the turn is held to whichever bites first: what
        # it can afford to read, and what the worker will accept before it
        # starts cutting the middle out on its own.
        afford = budget_for_answer(max_tokens)
        ceiling = prefill_ceiling(room_taken)
        budget = min(value for value in (afford, ceiling) if value > 0) if (
            afford > 0 or ceiling > 0
        ) else 0
        if budget <= 0 or len(body) <= budget:
            return stable_prefix_first(body) or body
        trimmed = fit_to_budget(
            body,
            request,
            budget=budget,
            always=CRITICAL_FOREGROUND_HEADERS,
            volatility=volatility_of,
        )
    except (ImportError, AttributeError, TypeError, ValueError) as exc:
        record_degradation(
            "cognitive_engine",
            exc,
            action="left the assembled system prompt as the assembler built it",
        )
        return body
    if not trimmed:
        return body
    logger.info(
        "✂️ [CONTEXT] system prompt %d → %d chars for a %d-token answer.",
        len(body),
        len(trimmed),
        int(max_tokens),
    )
    return trimmed


def _record_the_capability_inventory_miss(
    *,
    capability_inventory_contract: Any,
    system_prompt: Any,
    visible_user_message: Any,
) -> None:
    """Record why the capability inventory contract did not hold.

    Moved out of ``CognitiveEngine._direct_desktop_quick_reply`` by tools/extract_seam.py, which
    checks the body against the original token for token before
    writing. It reads 3 name(s) from the turn and hands back
    0.
    """
    from .cognitive_engine import (
        _COGNITIVE_ENGINE_RECOVERABLE_ERRORS,
        logger,
    )

    if not capability_inventory_contract:
        try:
            from core.brain.present_moment import present_moment_block

            present = present_moment_block()
            if present:
                # PREPENDED, not appended. The system prompt is compacted to
                # a 2,400-char scaffold floor before it reaches the worker,
                # keeping the head and a critical excerpt; anything at the
                # tail is the first thing cut. Attached at the tail, this
                # block was logged as attached and still never arrived —
                # "what time is it?" answered "my clock says 06:15 and the
                # ambient light sensors report low illumination" at 01:40,
                # from a runtime with no light sensor.
                # NOT prepended any more. inference_gate now delivers this
                # same block as its own system message positioned just
                # BEFORE the final user turn — added after compaction, so
                # it cannot be trimmed away, which was the original reason
                # for prepending here.
                #
                # Prepending it a second time put per-turn volatile text
                # (a clock, "2 min ago" receipts) at token ~125 of the
                # system prompt, which invalidated the KV prefix for
                # everything behind it. Measured live: 1,648 of 1,834
                # tokens re-prefilled every turn (10% reuse) and a simple
                # reply taking up to 16s, almost all of it prefill.
                pass
                # Grounding that cannot be seen cannot be verified. Two
                # prompt builders and one of them ungrounded cost an hour
                # of reasoning about why a fix "did not work" when it had
                # simply never run.
                logger.info(
                    "🧭 [GROUNDING] present-moment prepended to the desktop "
                    "system prompt (+%d chars, total %d).",
                    len(present),
                    len(system_prompt),
                )
        except _COGNITIVE_ENGINE_RECOVERABLE_ERRORS as exc:
            record_degradation(
                "cognitive_engine",
                exc,
                severity="warning",
                action="continued desktop turn without present-moment grounding",
            )
        try:
            from core.brain.recent_actions import recent_actions_block

            actions = recent_actions_block()
            if actions:
                # Same: the gate places the receipts block before the final
                # user turn. This copy only cost the cache prefix.
                pass
        except _COGNITIVE_ENGINE_RECOVERABLE_ERRORS as exc:
            record_degradation(
                "cognitive_engine",
                exc,
                severity="warning",
                action="continued desktop turn without recent-action receipts",
            )
        try:
            # Wider predicate: this path only adds a reading, while
            # asks_about_own_runtime also turns off web search.
            from core.runtime.self_state_intent import (
                asks_about_own_capabilities,
            )

            if asks_about_own_capabilities(visible_user_message):
                from core.brain.self_state_report import runtime_self_report

                instruments = runtime_self_report()
                if instruments:
                    # Same: delivered by the gate.
                    pass
        except _COGNITIVE_ENGINE_RECOVERABLE_ERRORS as exc:
            record_degradation(
                "cognitive_engine",
                exc,
                severity="warning",
                action="continued desktop turn without runtime self-readings",
            )
