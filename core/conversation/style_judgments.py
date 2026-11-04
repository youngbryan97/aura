"""Which of her reply checks judge her wording, and which find a reply broken or a claim false.

Most of the hard checks in `response_reliability` were written while a 16,478-character system prompt told her model
who to be, against what that prompt produced. It is gone (13 October), and what the wording checks flag now is her own
wording. A substantive draft is not thrown away for these alone
(`core/phases/response_generation_unitary.py`); a draft with leaked prompt text, a clipped tail, a loop or a false
claim still is.
"""

from __future__ import annotations

from typing import Any

from core.conversation.response_reliability import _HARD_USER_FACING_REASONS

__all__ = ["STYLE_JUDGMENTS", "only_style_judgments"]

#: The hard reasons that judge how she put something, against word lists.
STYLE_JUDGMENTS: frozenset[str] = frozenset(
    {
        "generic_assistant_language",
        "pseudo_internal_jargon",
        "status_page_self_reflection",
        "social_presence_instead_of_self_reflection",
        "persona_card_deflection",
        "detail_request_deflection",
        "reliability_diagnostic_deflection",
        "vague_status_derailment",
        "unprovoked_rebuke",
        "unrequested_pop_culture_intrusion",
        "missing_current_request_recap",
    }
)


def only_style_judgments(reasons: Any) -> bool:
    """Whether every hard reason given is a judgment of her wording, so the draft is still hers to serve."""
    hard = set(reasons or ()) & _HARD_USER_FACING_REASONS
    return hard <= STYLE_JUDGMENTS
