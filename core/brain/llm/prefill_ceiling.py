"""The longest prompt the resident cortex is handed, in characters.

This is the last boundary before a prompt reaches the worker. The inference
gate has per-section and total budgets, but a path that assembles its own
prompt never meets them, and on 2026-08-03 one did: 88,659 / 78,861 / 91,441
characters, prefill consumed the whole request deadline, and the answer came
back empty. Past this ceiling the client keeps the head and the tail and drops
the middle.

It was a fixed 48,000 characters, while the qualified serving profile lets the
standard lanes read 24,576 tokens. At the measured ratio of about three and a
half characters per token, 48,000 characters is under 14,000 tokens, so every
prompt between the two was cut although the cortex had been qualified to read
it. The ceiling is now the widest lane's qualified input, in characters at the
measured ratio, and the fixed figure remains only for an artifact with no
qualified profile.

The window is what the cortex can read, not what a turn should. LIVE
2026-10-03, the first turn after the window went from 32768 to 65536: a link
turn's prebuilt messages came to 130,902 characters, 35,391 tokens, read in
162 seconds for a 640-token answer, because every assembler fills to the
ceiling it is given. So an assembler is given :func:`reading_ceiling_chars`:
the fixed figure, and beyond it only as much as the answer earns
(core/brain/llm/context_budget.py, ``budget_for_answer``), up to the window.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Final

#: The client's own logger, so a bounded or declined prompt reads in the log where it always did.
logger = logging.getLogger("LLM.MLX")

#: The ceiling for a cortex with no qualified serving profile.
UNQUALIFIED_CEILING_CHARS: Final = 48_000
#: The head kept when a prompt is cut: the system contract is at the start.
KEEP_HEAD_CHARS: Final = 12_000


def prefill_ceiling_chars(model_path: str | None = None) -> int:
    """The widest qualified lane's input, in characters; the fixed figure without a profile.

    ``model_path`` names the model the prompt is for. The profile belongs to
    the active cortex, so any other model, the smaller fallback among them,
    keeps the fixed figure.
    """
    try:
        from core.brain.llm.model_registry import get_active_cortex_serving_limits
        from core.brain.llm.token_budget_evidence import chars_per_token

        limits = get_active_cortex_serving_limits(model_path or None)
        if limits is None or not limits.qualified or not limits.lanes:
            return UNQUALIFIED_CEILING_CHARS
        widest = max(int(lane.max_input_tokens) for lane in limits.lanes)
        if widest <= 0:
            return UNQUALIFIED_CEILING_CHARS
        return max(UNQUALIFIED_CEILING_CHARS, chars_per_token().tokens_to_chars(widest))
    except (ImportError, AttributeError, OSError, RuntimeError, TypeError, ValueError):
        return UNQUALIFIED_CEILING_CHARS


def reading_ceiling_chars(answer_tokens: int, model_path: str | None = None) -> int:
    """What a turn whose answer is ``answer_tokens`` long may read, in characters.

    Never below the fixed figure, so no prompt that fitted before the window
    was widened is cut now; never above the window.
    """
    window = prefill_ceiling_chars(model_path)
    try:
        from core.brain.llm.context_budget import budget_for_answer

        earned = int(budget_for_answer(int(answer_tokens)))
    except (ImportError, AttributeError, OSError, RuntimeError, TypeError, ValueError):
        earned = 0
    return min(window, max(UNQUALIFIED_CEILING_CHARS, earned))




def _record_degradation(error: BaseException, *, action: str, severity: str = "warning") -> None:
    """Recorded under the client's name, as it was before the bounding lived here."""
    from core.runtime.errors import record_degradation

    record_degradation("mlx_client", error, severity=severity, action=action)


def _prompt_within_prefill_ceiling(prompt: Any, *, model_path: str = "", origin: str = "") -> str:
    """Bound a prompt so prefill cannot eat the whole request deadline.

    Keeps the head (the system contract) and the tail (the question and the
    latest exchange), drops the middle, and says so in the text so the model
    is not silently reasoning over a gap it cannot see. The line names the
    origin: a 96,998-character prompt was bounded on 2026-09-15 and nothing
    in the record said who built it.
    """

    text, ceiling = str(prompt or ""), prefill_ceiling_chars(model_path)
    if len(text) <= ceiling:
        return text

    # The marker counts against the ceiling too — it is prompt like any other.
    marker = (
        f"\n\n…[{len(text) - ceiling} characters omitted: this prompt "
        "exceeded the prefill ceiling and would not have been answered at all]…\n\n"
    )
    tail_budget = max(0, ceiling - KEEP_HEAD_CHARS - len(marker))
    bounded = text[:KEEP_HEAD_CHARS] + marker + text[-tail_budget:]
    who = str(origin or "").strip() or "an unnamed caller"
    logger.error(
        "🪓 [MLX] Prompt %d chars from %s exceeded the %d prefill ceiling for %s — kept "
        "head+tail, dropped the middle. An unbounded prompt returns one token and no text.",
        len(text),
        who,
        ceiling,
        os.path.basename(str(model_path or "")) or "model",
    )
    _record_degradation(
        RuntimeError(
            f"prompt {len(text)} chars from {who} over prefill ceiling {ceiling}"
        ),
        action="bounded the prompt to head+tail so the turn could produce an answer",
        severity="warning",
    )
    return bounded


def cpu_lane_cannot_start(client: Any, prompt_chars: int) -> bool:
    """Whether this CPU lane has already failed to start a prompt this long, and so declines it at once.

    LIVE, seed 101's report runs, 30 and 31 October: when the 2-bit 27B
    fallback could not run, a turn's background work fell to the 1.5B
    emergency lane on the CPU with her whole prompt, 25,000 to 43,000
    characters. The CPU read it at about 8 tokens a second, so a first token
    was minutes away, and the ceiling ended every attempt after about 64
    seconds with no text: 345 times in one ten-hour run, each ending as the
    first had.

    On a CPU the rate does not come and go with a busy GPU, so a prompt at
    least as long as one this worker could not start will not start either.
    The answer is the one the lane gave before, no text, given now instead of
    after the ceiling. Nothing is published as a deliberate cancel, so the
    router reads it exactly as it read each of those attempts. A GPU lane
    never declines, because there a stall is pressure that can clear; and a
    new worker starts with no record (`mlx_client_worker_lifecycle`).
    """
    smallest = int(getattr(client, "_unstartable_from_chars", 0) or 0)
    if smallest <= 0 or str(getattr(client, "device", "")).lower() != "cpu" or int(prompt_chars) < smallest:
        return False
    logger.info(
        "⏭️ [MLX] %s on the CPU could not start a %d-char prompt within its first-token "
        "ceiling; declining a %d-char one now rather than after the same ceiling.",
        os.path.basename(str(getattr(client, "model_path", "") or "")),
        smallest,
        int(prompt_chars),
    )
    return True


def note_cpu_lane_could_not_start(client: Any) -> None:
    """On a CPU lane, remember the prompt it just failed to produce a first token for."""
    if str(getattr(client, "device", "")).lower() != "cpu" or getattr(client, "_tokens_this_request", 0):
        return
    chars = int(getattr(client, "_current_prompt_chars", 0) or 0)
    smallest = int(getattr(client, "_unstartable_from_chars", 0) or 0)
    if chars > 0 and (smallest <= 0 or chars < smallest):
        client._unstartable_from_chars = chars


__all__ = [
    "KEEP_HEAD_CHARS",
    "UNQUALIFIED_CEILING_CHARS",
    "_prompt_within_prefill_ceiling",
    "cpu_lane_cannot_start",
    "note_cpu_lane_could_not_start",
    "prefill_ceiling_chars",
    "reading_ceiling_chars",
]
