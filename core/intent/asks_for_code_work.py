"""Whether a request asks for work on code, or asks about the world in words that name a language.

The kind of a request was read by words: ``_CODE_HINT`` in core/brain/reasoning_task_kinds.py matches "Python", so a
question about Python's release history is a code task. LIVE 19 November 2026, four times: "Can you look up what the
latest stable release of Python is right now?" was read as code, a deep search of nine candidates ran over her answer,
and the turn ran out of time before she was heard.

This reads the request by meaning, on her encoder, against declared examples of each kind: writing, running, reading,
fixing or explaining code on one side, and on the other facts about languages, tools and their makers, the things a
search finds. It decides only where its own measured boundary decides, and says nothing otherwise, so a request it
cannot place keeps the kind its words gave it.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

__all__ = ["asks_for_code_work"]

logger = logging.getLogger(__name__)

#: Work on code: something written, run, read, mended or explained, in the request's own words.
_CODE_WORK = (
    "Write a function that reverses a linked list.",
    "Why does this Python script raise a KeyError on line 12?",
    "Fix the off-by-one error in my loop.",
    "Implement binary search in Rust.",
    "Refactor this class to use dataclasses.",
    "Write a SQL query that counts orders per customer.",
    "What does this regular expression match: ^a+b$?",
    "Can you debug this stack trace for me?",
    "Make this JavaScript function async.",
    "Add type hints to this module.",
)
#: About the world, in words that name a language or a tool: what a search answers, not what code computes.
_ABOUT_THE_WORLD = (
    "What's the newest version of Java?",
    "When was Python first released?",
    "Who created the Rust language?",
    "Is JavaScript still the most used language?",
    "Which company maintains TypeScript?",
    "Where is the next PyCon being held?",
    "How many people contribute to the Linux kernel?",
    "Look up the release date of Node 24.",
    "What does a GitHub Copilot subscription cost now?",
    "Search for the latest news about the Go language.",
)

_SURFACE: list[Any] = []
_SURFACE_FAILED: list[bool] = []
#: What was decided off her event loop, by the request's text, for an ask on her loop to read without computing.
_DECIDED: dict[str, bool] = {}


def _surface() -> Any:
    if _SURFACE:
        return _SURFACE[0]
    if _SURFACE_FAILED:
        return None
    try:
        from core.language.learned_matcher import LearnedMatcher, embed_sentences

        _SURFACE.append(LearnedMatcher(name="asks_for_code_work", positives=_CODE_WORK, negatives=_ABOUT_THE_WORLD,
                                       features=embed_sentences))
        return _SURFACE[0]
    except (ImportError, RuntimeError, TypeError, ValueError) as exc:
        logger.debug("the code-work surface could not be built, so the kind stays the words' (%s: %s)",
                     type(exc).__name__, exc)
        _SURFACE_FAILED.append(True)
        return None


def _on_an_event_loop() -> bool:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


def asks_for_code_work(message: object) -> bool | None:
    """True when the request asks for work on code, False when it asks about the world, None when the surface cannot
    tell (no encoder yet, or the request sits where its declared examples overlap)."""
    text = str(message or "").strip()
    surface = _surface()
    if not text or surface is None:
        return None
    try:
        # On an event loop's thread it never computes: it answers from what it decided before, or leaves the request to
        # the matcher's warmer. Her turn asks from a worker thread first (the phases call is_amplifiable through
        # to_thread), and what is decided there is kept here, so a later ask on her loop in the same turn has it.
        if _on_an_event_loop():
            return _DECIDED[text] if text in _DECIDED else surface.decide_without_waiting(text)
        verdict = surface.decide(text)
        if verdict is not None:
            _DECIDED[text] = verdict
        return verdict
    except (RuntimeError, TypeError, ValueError) as exc:
        logger.debug("the code-work surface would not decide (%s: %s)", type(exc).__name__, exc)
        return None
