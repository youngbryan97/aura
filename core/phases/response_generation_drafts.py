"""What becomes of a foreground draft: kept for repair or rejected, and its instruction shape repaired after voice shaping.

Lifted whole out of `response_generation`, which imports them straight back: every
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


def _keep_a_repairable_draft_or_reject_it(
    *,
    reliability: Any,
    reliability_reasons: Any,
    response_text: Any,
    response_text_s: Any,
    state: Any,
) -> Any:
    """Keep a foreground draft the final repair can fix, or reject an unsafe one and end the phase.

    Moved out of ``execute`` by tools/extract_seam.py, which checks
    the body against the original token for token before writing. The
    block returns early, so it sits in a nested function and _SEAM_FELL_THROUGH
    means it finished instead. It reads 5 name(s) and hands back
    0.
    """
    from .response_generation import (
        _DOWNSTREAM_REPAIRABLE_RESPONSE_REASONS,
        _REJECTED_DRAFT_LOG_CHARS,
        _SEAM_FELL_THROUGH,
        logger,
    )

    from core.conversation.surface_disposition import draft_is_servable
    def _block() -> Any:
        if (
            reliability_reasons
            and (
                reliability_reasons.issubset(
                    _DOWNSTREAM_REPAIRABLE_RESPONSE_REASONS
                )
                or draft_is_servable(reliability_reasons)
            )
            and len(response_text_s) >= 48
            and len(response_text_s.split()) >= 8
        ):
            # The draft itself, bounded. A reason and a length
            # describe a rejection without saying what was
            # rejected, and the two questions a reader has are
            # "was the gate right?" and "what did she nearly
            # say?" — neither answerable from a number. The
            # file sink redacts, and this stays local.
            logger.warning(
                "ResponseGeneration kept repairable foreground draft for final reply repair (%s, len=%d): %r",
                ",".join(reliability.reasons) or "unknown",
                len(response_text_s),
                response_text_s[:_REJECTED_DRAFT_LOG_CHARS],
            )
            try:
                from core.conversation.surface_disposition import (
                    preserve_draft,
                )

                preserve_draft(response_text_s)
            except (ImportError, RuntimeError, TypeError, ValueError) as exc:
                logger.debug(
                    "the rejected draft was not preserved (%s: %s)",
                    type(exc).__name__,
                    exc,
                )
        else:
            logger.warning(
                "ResponseGeneration rejected unsafe user-facing draft (%s, len=%d): %r",
                ",".join(reliability.reasons) or "unknown",
                len(str(response_text or "")),
                str(response_text or "")[:_REJECTED_DRAFT_LOG_CHARS],
            )
            return state
        return _SEAM_FELL_THROUGH

    _seam_early_response = _block()
    return _seam_early_response


def _repair_the_instruction_shape_after_voice(
    *,
    append_only_continuation_pending: Any,
    cleaned_response: Any,
    is_background: Any,
    is_test_run: Any,
    latent_response_owned: Any,
    response_mutation_receipt: Any,
    self: Any,
    state: Any,
    user_surface_validation_prompt: Any,
) -> Any:
    """Repair a substantive instruction-shape miss that voice shaping left, with its receipt.

    Moved out of ``execute`` by tools/extract_seam.py, which
    checks the body against the original token for token before
    writing. It reads 9 name(s) from the turn and hands back
    1.
    """
    from .response_generation import (
        append_text_mutation,
        logger,
    )

    if (
        not is_background
        and cleaned_response
        and not is_test_run
        and not latent_response_owned
        and not append_only_continuation_pending
    ):
        repaired_response, repaired_shape, repair_reasons = (
            self._repair_substantive_instruction_shape_miss(
                user_surface_validation_prompt, cleaned_response
            )
        )
        if repaired_shape:
            pre_post_voice_repair = cleaned_response
            cleaned_response = repaired_response
            append_text_mutation(
                response_mutation_receipt,
                stage="response_generation.post_voice_shape",
                method="deterministic_instruction_shape",
                reasons=repair_reasons,
                before=pre_post_voice_repair,
                after=cleaned_response,
                deterministic=True,
                authorship_effect="preserved",
            )
            state.response_modifiers["post_voice_shape_repair"] = {
                "reasons": list(repair_reasons),
                "method": "deterministic_instruction_shape",
            }
            logger.info(
                "ResponseGeneration repaired instruction shape after voice shaping (%s).",
                ",".join(repair_reasons) or "unknown",
            )
    return cleaned_response


