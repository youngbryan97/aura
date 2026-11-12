"""Steps of the chat turn moved out of `_api_chat_turn` and `_run_cognitive_engine_chat_turn`: keeping a retry that improves, the desktop contract that answers alone, the engine connection, the grounded-claim check and the salvaged draft.

Lifted whole out of `chat`, which imports them straight back: every
caller and every patch that names them there still finds them. What they
take from that module is imported at CALL time, for the same reason.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import math
import os
import random
import re
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any


def _keep_the_retry_if_it_improves(
    *,
    failure_reason: Any,
    retry_reply: Any,
    text: Any,
    visible: Any,
) -> Any:
    """Keep a retried reply only where it improves on the one it repairs.

    Moved out of ``_run_cognitive_engine_chat_turn`` by tools/extract_seam.py, which
    checks the body against the original token for token before
    writing. It reads 4 name(s) from the turn and hands back
    1.
    """
    from .chat import (
        _CHAT_RECOVERABLE_ERRORS,
        logger,
    )

    if retry_reply:
        # A repair has to earn the substitution. Most of these gates were
        # written for a weaker model and encode style expectations from
        # that period; a repair they trigger must not hand the person a
        # blander or shorter answer than the one it replaces.
        try:
            from core.conversation.surface_disposition import (
                repair_is_an_improvement,
            )

            # `failure_reason` is what this retry was FOR. A replacement
            # that still carries it delivered nothing the retry predicted,
            # and swapping it in trades a known answer for an equally
            # objectionable one.
            keep_retry = repair_is_an_improvement(
                text, retry_reply, visible, targeted=(failure_reason,)
            )
        except _CHAT_RECOVERABLE_ERRORS:
            keep_retry = True
        if not keep_retry:
            logger.warning(
                "Kept the original draft (%d chars): the repair for %s was "
                "not an improvement on it (%d chars).",
                len(str(text or "")),
                failure_reason,
                len(str(retry_reply or "")),
            )
            retry_reply = None
    return retry_reply


def _the_desktop_contract_answers_for_itself(
    *,
    desktop_execution_contract: Any,
    require_engine: Any,
    turn_trace: Any,
    visible: Any,
) -> Any:
    """A desktop objective the execution contract can carry alone is answered without a model.

    Moved out of ``_run_cognitive_engine_chat_turn`` by tools/extract_seam.py, which checks
    the body against the original token for token before writing. The
    block returns early, so it sits in a nested function and _SEAM_FELL_THROUGH
    means it finished instead. It reads 4 name(s) and hands back
    0.
    """
    from .chat import (
        _SEAM_FELL_THROUGH,
        _desktop_objective_self_sufficient_without_cognitive_text,
        logger,
    )
    from .chat_desktop_objective import _a_page_she_read_awaits_her_answer

    def _block() -> Any:
        # Once her browser has ended on the page for the question, the desktop work is done and the answer is hers to
        # write from it. LIVE 11 November 2026 this path served its promise to execute in place of her turn after
        # r/nosleep went to her turn; the promise failed its authorship gate, and her background model answered.
        if (
            desktop_execution_contract
            and require_engine
            and _desktop_objective_self_sufficient_without_cognitive_text(visible)
            and not _a_page_she_read_awaits_her_answer()
        ):
            logger.info(
                "Serving self-sufficient desktop execution contract without foreground model allocation."
            )
            if turn_trace is not None:
                turn_trace.update(
                    {
                        "bounded_contract_used": True,
                        "response_path": "self_sufficient_desktop_execution_contract",
                    }
                )
            return (
                "I will execute this through the governed desktop_task lane and report only "
                "receipt-verified effects. If desktop_task cannot prove the effect, I will "
                "report the blocker instead of claiming completion."
            )
        return _SEAM_FELL_THROUGH

    _seam_early_response = _block()
    return _seam_early_response


async def _take_an_engine_connection(
    *,
    engine: Any,
    pool: Any,
    require_engine: Any,
) -> tuple[Any, Any]:
    """Take a desktop-chat connection from the engine pool.

    Moved out of ``_run_cognitive_engine_chat_turn`` by tools/extract_seam.py, which checks
    the body against the original token for token before writing. The
    block returns early, so it sits in a nested function and _SEAM_FELL_THROUGH
    means it finished instead. It reads 3 name(s) and hands back
    1.
    """
    from .chat import (
        _CHAT_RECOVERABLE_ERRORS,
        _SEAM_FELL_THROUGH,
        logger,
        record_degradation,
    )

    async def _block() -> Any:
        nonlocal pool
        try:
            from core.providers.engine_connection_pool import get_engine_connection_pool

            pool = get_engine_connection_pool()
            await pool.acquire_engine_connection(engine, connection_id="desktop_chat")
        except _CHAT_RECOVERABLE_ERRORS as exc:
            pool = None
            record_degradation("chat", exc)
            if require_engine:
                logger.warning(
                    "CognitiveEngine desktop chat connection pool unavailable; "
                    "continuing with direct CognitiveEngine call under foreground timeout: %s",
                    exc,
                )
            else:
                logger.warning("CognitiveEngine desktop chat connection unavailable: %s", exc)
                return None
        return _SEAM_FELL_THROUGH

    _seam_early_response = await _block()
    return _seam_early_response, pool


def _check_the_grounded_claims(
    *,
    _final_reply: Any,
    _live_turn_trace: Any,
    _qualified_exact_delivery: Any,
) -> Any:
    """Hold the final reply's claims against what grounds them.

    Moved out of ``_api_chat_turn`` by tools/extract_seam.py, which
    checks the body against the original token for token before
    writing. It reads 3 name(s) from the turn and hands back
    1.
    """
    from .chat import (
        _CHAT_RECOVERABLE_ERRORS,
        _append_turn_text_mutation,
        logger,
        record_degradation,
    )

    try:
        from core.conversation.grounded_claim_guard import verify_grounded_claims

        _grounded = (
            verify_grounded_claims(_final_reply)
            if not _qualified_exact_delivery
            else None
        )
        if _grounded is not None and _grounded.changed:
            _append_turn_text_mutation(
                _live_turn_trace,
                stage="chat.grounded_claim_guard",
                method="measured_reading_overrides_stated_claim",
                reasons=list(_grounded.corrections),
                before=_final_reply,
                after=_grounded.text,
                deterministic=True,
                authorship_effect="augmented_by_runtime",
            )
            logger.warning(
                "🧭 [GROUNDING] reconciled a spoken claim against a real reading: %s",
                "; ".join(_grounded.corrections)[:240],
            )
            _final_reply = _grounded.text or _final_reply
    except _CHAT_RECOVERABLE_ERRORS as _exc:
        record_degradation("chat.grounded_claim_guard", _exc)
    return _final_reply


def _serve_or_withhold_the_salvaged_draft(
    *,
    salvage_contract: Any,
    salvage_output_proven: Any,
    salvaged_no_reply: Any,
) -> Any:
    """Serve unfinished authored work only where its salvage contract allows it; otherwise withhold it.

    Moved out of ``_api_chat_turn`` by tools/extract_seam.py, which
    checks the body against the original token for token before
    writing. It reads 2 name(s) from the turn and hands back
    1.
    """
    from .chat import (
        _authored_answer_can_serve_unfinished,
        logger,
    )

    if not (
        salvage_output_proven
        and _authored_answer_can_serve_unfinished(salvage_contract)
    ):
        logger.warning(
            "Preserved no-reply draft remained ineligible for delivery; "
            "withholding it (missing=%s).",
            ",".join(
                salvage_contract.get("full_mind_missing_proofs") or ()
            )
            or "unknown",
        )
        salvaged_no_reply = ""
    else:
        logger.info(
            "Serving %d characters of unfinished authored work rather "
            "than an apology (unproven=%s).",
            len(salvaged_no_reply),
            ",".join(
                salvage_contract.get("full_mind_missing_proofs") or ()
            )
            or "none",
        )
    return salvaged_no_reply


