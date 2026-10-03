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

from typing import Final

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


__all__ = ["KEEP_HEAD_CHARS", "UNQUALIFIED_CEILING_CHARS", "prefill_ceiling_chars", "reading_ceiling_chars"]
