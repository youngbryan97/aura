"""Whether a question asks for a fact the world holds about something named, or for her own working-out.

Her world evidence searches the web when a factual question about a named thing is answered only in part by her
offline copy (core/conversation/what_the_world_says.py, ``_answers_only_in_part``). Whether a question was factual was
read by words: a question word followed by another word, and a capitalised name. LIVE 22 November 2026, typed into
her window: "Aura, you freaking genius…. I think we can work with this! ... have this type of reasoning be applicable
in any context (But of course being, respectfully, used to kick Stockfish's ass as a proof of concept ...). What would
that look like?" has "What would" and "Stockfish", and was searched on the web whole. Two pages came back, "Essay
Flagged as AI But You Wrote It?" and "AI Excuse Generator", and went into her prompt as sources.

This reads the question by meaning, on her encoder, against declared examples of each side: a fact about a named
thing, which a search finds, and on the other side a design, a plan, a hypothetical or a judgement asked of her, which
she works out. It decides only where its own measured boundary decides, and says nothing otherwise, so a question it
cannot place is read by its words as before.
"""
from __future__ import annotations

import logging
from typing import Any

__all__ = ["asks_for_a_fact"]

logger = logging.getLogger(__name__)

#: A fact about something named: what it is, who made it, when, where, how many, what it cost.
_A_FACT = (
    "Who designed the Kaseya Center, and what did it cost to build?",
    "When did the Kaseya Center open?",
    "How many MVP awards did Nikola Jokic win?",
    "What is the population of Lisbon?",
    "Who founded Hugging Face?",
    "What is the latest stable release of Python?",
    "Where is Anthropic headquartered?",
    "Who directed the film Arrival?",
    "What year did Stockfish first beat a world champion program?",
    "How tall is the Eiffel Tower?",
)
#: Her own working-out asked for: a design, a plan, what something would look like, whether something could be done,
#: and a judgement of hers. A named thing in it is the subject of her thinking, not a fact to look up.
_HER_WORKING_OUT = (
    "What would a version of this look like that works for any game, not only chess?",
    "How would you design an engine that out-plays Stockfish at its own game?",
    "Do you think it would be possible to build something that out-strategizes Stockfish?",
    "Sketch the architecture you have in mind for this.",
    "How could we make this kind of reasoning work in any context?",
    "What would you change about the plan we just made?",
    "If you were building this for Go or poker, where would you start?",
    "Walk me through how you would approach beating Stockfish.",
    "Which was better, Bam's 83 point game or Kobe's 81?",
    "What do you make of the new arena?",
)

_SURFACE: list[Any] = []
_SURFACE_FAILED: list[bool] = []


def _surface() -> Any:
    if _SURFACE:
        return _SURFACE[0]
    if _SURFACE_FAILED:
        return None
    try:
        from core.language.learned_matcher import LearnedMatcher, her_encoder_only

        # Her encoder or nothing: this reading decides whether a question goes to the web, and a boundary measured on
        # hashed words decides nonsense.
        _SURFACE.append(LearnedMatcher(name="asks_for_a_fact", positives=_A_FACT, negatives=_HER_WORKING_OUT,
                                       features=her_encoder_only))
        return _SURFACE[0]
    except (ImportError, RuntimeError, TypeError, ValueError) as exc:
        logger.debug("the fact surface could not be built, so the question is read by its words (%s: %s)",
                     type(exc).__name__, exc)
        _SURFACE_FAILED.append(True)
        return None


def asks_for_a_fact(message: object) -> bool | None:
    """True when the question asks for a fact about something named, False when it asks for her own working-out, None
    when the surface cannot tell (no encoder yet, or the question sits where its declared examples overlap). It
    computes: ask it off her event loop."""
    text = str(message or "").strip()
    surface = _surface()
    if not text or surface is None:
        return None
    try:
        return surface.decide(text)
    except (RuntimeError, TypeError, ValueError) as exc:
        logger.debug("the fact surface would not decide (%s: %s)", type(exc).__name__, exc)
        return None
