"""Warming the foreground lane before a turn's first generation attempt.

Lifted whole out of `inference_gate`, which imports them straight back: every
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


async def _warm_the_foreground_lane_before_the_first_attempt(
    *,
    lane_status: Any,
    local_label: Any,
    primary_timeout: Any,
    primary_warmup_memory_deferred: Any,
    self: Any,
    skip_initial_primary_attempt: Any,
) -> tuple[Any, Any, Any]:
    """Warm the foreground lane before the first attempt, or say the first attempt is skipped.

    Moved out of ``_generate_with_metadata_sink`` by tools/extract_seam.py, which
    checks the body against the original token for token before
    writing. It reads 4 name(s) from the turn and hands back
    3.
    """
    from .inference_gate import (
        _named_blockers,
        logger,
    )

    if not lane_status.get("conversation_ready"):
        blockers = lane_status.get("readiness_blockers") or []
        blocker_text = ", ".join(str(item) for item in blockers[:3]) or "conversation probe"
        logger.info(
            "%s lane process state=%s; conversation readiness is blocked by %s. Completing foreground warmup before first generation attempt.",
            local_label,
            lane_status.get("state", "unknown"),
            blocker_text,
        )
        try:
            # Admission control — break the cortex doom-loop.
            # A COLD first boot legitimately needs ~150s to
            # load the cortex and the user expects that one-time
            # wait. But a RECOVERY (Cortex was ready, got
            # force-killed on a first-token stall, is now
            # reloading) must NOT block every foreground turn
            # for 90-180s — that is the observed doom loop
            # (soak Jul 7: turns 21-30 crawled to 200s+ while
            # the warm window played out, memory thrashed).
            # When the lane was EVER ready, cap the preflight
            # wait short and let this turn fall to the ready
            # tier while Cortex warms in the background for the
            # next turn.
            warmup_timeout = self._foreground_warmup_timeout(
                lane_status, primary_timeout
            )
            lane_status = await self.ensure_foreground_ready(
                timeout=warmup_timeout
            )
        except (
            TimeoutError,
            RuntimeError,
            AttributeError,
            TypeError,
            ValueError,
            OSError,
        ) as warmup_exc:
            if self._note_foreground_warmup_failure(warmup_exc):
                primary_warmup_memory_deferred = True
            lane_status = self.get_conversation_status()
        if not self._lane_can_attempt_visible_conversation_turn(lane_status):
            skip_initial_primary_attempt = True
            logger.warning(
                "%s is still not ready after foreground preflight warmup (state=%s, blocked by %s). Skipping the cold first attempt and waiting for recovery before retry.",
                local_label, lane_status.get("state", "unknown"), _named_blockers(lane_status),
            )
    return lane_status, primary_warmup_memory_deferred, skip_initial_primary_attempt


