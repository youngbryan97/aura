"""A question that asks for her view.

Part of the language substrate, beside search_request: one bounded question,
answered from the sentence, abstaining when the structure is not there.

Bryan, 5 October 2026: for questions asking her opinion he does not need
sources unless she is using sourced information; an opinion is a subjective
view, not a research question, though real evidence can support one. "Which
was better, Bam's 83 point game or Kobe's 81?" names two things and opens with
"which", so the reading of factual questions about named things took it for
one, and when the offline articles scored below the judge's matched median
it went to the web and came back with a "Source:" line under her opinion.

Two readers, as elsewhere in the substrate:

* the grammar is the floor: her view asked for in so many words ("what do you
  think", "your take", "in your opinion", "do you prefer"), or a judgement
  asked of her ("which was better", "who's the greatest");
* a learned surface over her own model's representation takes the phrasings
  the grammar does not reach, with a measured boundary and an abstention.

What it does not decide: whether evidence would help. The offline copy is
still read and still offered when it bears on the question; a view can stand
on facts. Asking her to look something up, or for sources, is an instruction
that wins over this reading.
"""

from __future__ import annotations

import logging
import re

from core.language.learned_matcher import LearnedMatcher
from core.language.model_features import model_hidden_features

logger = logging.getLogger(__name__)

__all__ = ["asks_for_her_view", "teach_her_view_from_the_floor"]

#: Her view asked for in words, or a judgement asked of her.
_ASKS_HER_VIEW = re.compile(
    r"\b(?:what|how)\s+do\s+you\s+(?:think|feel|reckon|make\s+of)\b"
    r"|\bwhat(?:'s|’s|\s+is|\s+are)\s+your\s+(?:opinion|take|view|views|thoughts|stance|verdict|pick|favou?rite)\b"
    r"|\b(?:do|would|did|don't|wouldn't)\s+you\s+(?:think|believe|like|love|hate|prefer|agree|rate|rank|consider|rather)\b"
    r"|\bin\s+your\s+(?:opinion|view|eyes|book)\b"
    r"|\byour\s+favou?rite\b"
    r"|\bwhich\s+(?:one\s+)?(?:do|would)\s+you\b"
    r"|\b(?:which|who|what)(?:'s|’s|\s+(?:is|was|are|were|would\s+be))\s+(?:the\s+)?"
    r"(?:better|best|worse|worst|greatest|favou?rite|goat|(?:more|most)\s+(?:impressive|important|beautiful|overrated|underrated))\b"
    r"|\bis\s+\S+(?:\s+\S+){0,3}\s+(?:overrated|underrated)\b",
    re.IGNORECASE,
)

_HER_VIEW = LearnedMatcher(
    name="asks_for_her_view",
    positives=(
        "which was better, Bam's 83 point game or Kobe's 81?",
        "what do you make of the new arena?",
        "is Bam underrated?",
        "would you rather watch baseball or basketball?",
        "how would you rank the Heat's title teams?",
        "does that album still hold up?",
        "who's the greatest Heat player ever?",
        "what's your take on LeBron leaving Miami?",
    ),
    negatives=(
        "when did the Kaseya Center open?",
        "who founded Hugging Face?",
        "how much did it cost to build?",
        "google it",
        "what is 17 times 4?",
        "what's the weather tomorrow?",
        "who won the game last night?",
        "how many points did Kobe score?",
    ),
    features=model_hidden_features,
)


def asks_for_her_view(text: str) -> bool | None:
    """Whether ``text`` asks for her view; None when neither reader can tell."""
    message = str(text or "").strip()
    if not message:
        return False
    if _ASKS_HER_VIEW.search(message):
        return True
    return _HER_VIEW.decide_without_waiting(message)


def teach_her_view_from_the_floor(text: str) -> None:
    """What the grammar read exactly becomes an example for the learned surface."""
    message = str(text or "").strip()
    try:
        if _ASKS_HER_VIEW.search(message):
            _HER_VIEW.observe(message, holds=True)
    except (RuntimeError, TypeError, ValueError) as exc:
        logger.debug("the floor's reading of a view was not kept as an example: %s", exc)
