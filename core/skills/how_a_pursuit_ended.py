"""What stopped her work on a page, said the way a person says it.

LIVE 2026-10-20 13:27, asked to find an Allen Pan video: her reply began "Worked through
https://www.youtube.com/results?search_query=Allen+Pan over 3 round(s), then stopped:
browser_interaction_incomplete." The loop's reasons are names for its log. Said to a person, each is what happened.
"""
from __future__ import annotations

__all__ = ["IN_WORDS", "how_it_ended"]

#: Each reason the page loop records for a round, as what happened.
IN_WORDS = {
    "browser_interaction_incomplete": "what I pressed did nothing",
    "interaction_failed": "what I pressed did nothing",
    "no_progress": "the page stopped changing",
    "no_executable_action": "nothing I chose could be done on the page",
    "unparsable_decision": "I lost track of what to do next",
    "empty_decision": "I lost track of what to do next",
    "llm_router_unavailable": "I could not think it through just then",
    "page_read_refused": "the page could not be read",
    "pursuit_interrupted": "I was interrupted",
    "browser_interaction_authority_unavailable": "I was not allowed to act on the page",
    "browser_interaction_origin_unobserved": "I could not see which page I was acting on",
}


def how_it_ended(where: str, rounds: int, failure: str) -> str:
    """One sentence on a run over a page: where it got to and, if it stopped short, what stopped it.

    ``failure`` is the loop's reason, or several joined by ":" with the most recent first; a reason that is already a
    sentence is said as it is.
    """
    if not failure:
        return f"Worked through {where} over {rounds} round(s)."
    said = next((IN_WORDS[one.strip()] for one in failure.split(":") if one.strip() in IN_WORDS), failure)
    return f"I got as far as {where} and stopped: {said}."
