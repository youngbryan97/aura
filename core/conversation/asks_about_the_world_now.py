"""Whether a question asks about the world as it is now, read by meaning on her encoder.

A turn about something as it is now is searched, and the pages found are read as her browser shows them, whatever a
dated copy holds (interface/routes/chat_desktop_evidence.py, ``_about_the_world_now``). Whether it was about now was read
from a list of words, each claiming every word it begins: "stock" claims "Stockfish". LIVE 22 November 2026, 00:07,
typed into her window: "How've the games been going? What's it like to play Stockfish?" asked about her own chess, was
searched as something current, and rendering a dictionary's page for "how" ran out of time; her model was never asked,
and a smaller one's 90 tokens went to the person, cut off mid-answer.

This reads the question against declared examples of each side: a state of the world that changes, which only a live
source answers, and on the other side her own doings, the conversation, and what does not change. It decides only where
its own measured boundary decides; where it says no, the turn is not taken for one about now.
"""
from __future__ import annotations

import logging
from typing import Any

__all__ = ["asks_about_the_world_now"]

logger = logging.getLogger(__name__)

#: The world as it is now: a score, a price, the weather, the news, who won, whether something is out yet.
_THE_WORLD_NOW = (
    "What's the score of the Lakers game right now?",
    "Who won the election yesterday?",
    "What is Nvidia's stock price today?",
    "What's the weather in Paris this afternoon?",
    "What's the latest news about the Federal Reserve?",
    "Is the new version of Stockfish out yet?",
    "How did the Knicks play last night?",
    "When is the next SpaceX launch scheduled?",
    "Are flights grounded at JFK today?",
    "What's happening with the outage right now?",
    "What do you make of the current situation in Sudan?",
    "What's your take on the trade war between the US and China these days?",
)
#: Not the world now: her own doings and experience, the conversation, and what does not change with the day.
_NOT_THE_WORLD_NOW = (
    "How've the games been going? What's it like to play Stockfish?",
    "How are your chess games against the engine going?",
    "What was it like for you to lose that last game?",
    "What have you been up to this evening?",
    "What did you think of what I said earlier?",
    "How does Stockfish evaluate a position?",
    "Who invented the game of chess?",
    "What does a stock market index measure?",
    "Create a small HTML page at artifacts/live_runtime/now.html.",
    "I slept badly last night, maybe four hours.",
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

        # Her encoder or nothing: this reading decides whether a turn's time goes to live pages, and a boundary measured
        # on hashed words decides nonsense.
        _SURFACE.append(LearnedMatcher(name="asks_about_the_world_now", positives=_THE_WORLD_NOW,
                                       negatives=_NOT_THE_WORLD_NOW, features=her_encoder_only))
        return _SURFACE[0]
    except (ImportError, RuntimeError, TypeError, ValueError) as exc:
        logger.debug("the now surface could not be built, so the question is read by its words (%s: %s)",
                     type(exc).__name__, exc)
        _SURFACE_FAILED.append(True)
        return None


def asks_about_the_world_now(message: object) -> bool | None:
    """True when the question asks about the world as it is now, False when it does not, None when the surface cannot
    tell (no encoder yet, or the question sits where its declared examples overlap). It computes: ask it off her event
    loop."""
    text = str(message or "").strip()
    surface = _surface()
    if not text or surface is None:
        return None
    try:
        return surface.decide(text)
    except (RuntimeError, TypeError, ValueError) as exc:
        logger.debug("the now surface would not decide (%s: %s)", type(exc).__name__, exc)
        return None
