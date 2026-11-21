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
    # Code run over files or data to measure them. LIVE 20 November 2026, "Find the two largest Python files under
    # core/brain by line count, count their lines by actually running something, and tell me by what percentage the
    # larger one is bigger" was read as not code work: nothing declared here ran code to measure anything.
    "Run a script that counts the lines in every file of this folder and print the totals.",
    "Measure which files in this directory are largest by running something, and list their sizes.",
    "Compute the mean of the numbers in this CSV with code and show what it prints.",
)
#: About the world, in words that name a language or a tool: what a search answers, not what code computes. And a code
#: that opens, addresses or names something, which is no program: LIVE 19 November 2026, "What is the locker code for
#: the Brimvaltor depot?" read as work on code, and a second candidate answer was written and checked on the web.
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
    "What's the door code for the back office?",
    "What is the area code for Chicago?",
    "Which postal code is this address in?",
    "What is the airport code for Lisbon?",
    "Do you remember the alarm code I gave you?",
)
#: Acting on the screen or in an application, which is no work on code either. LIVE 20 November 2026, once this reading
#: began deciding which lane a request goes to (core/runtime/desktop_objective_intent.py), on her encoder it read
#: "increase the brightness" and "create a folder called Orca on my desktop" as work on code: it knew two kinds, code and
#: questions about the world, and screen work sat nearer code.
_ON_THE_SCREEN = (
    "Open Notes and write a line about today.",
    "Turn the screen brightness up a little.",
    "Make a new folder on my desktop named Projects.",
    "Close the window that is in front.",
    "Put this note into the Notes app.",
    "Write a short list in Pages and save the document.",
    "Click the blue button, then scroll to the bottom of the page.",
    "Play the next song in Music.",
    "Look through my documents for last month's invoice.",
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
        from core.language.learned_matcher import LearnedMatcher, her_encoder_only

        # Her encoder or nothing: this reading sends a turn to her code runner (core/runtime/desktop_objective_intent.py)
        # and keeps code_repl offered, and a boundary measured on hashed words decides nonsense.
        _SURFACE.append(LearnedMatcher(name="asks_for_code_work", positives=_CODE_WORK,
                                       negatives=(*_ABOUT_THE_WORLD, *_ON_THE_SCREEN),
                                       features=her_encoder_only))
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
