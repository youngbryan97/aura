"""Asking which app is in front is asking about the screen.

LIVE 2026-08-18: "can you tell what app I'm using right now?"

    I can capture and read this screen. I'm in my own little computational
    world, not connected to your device's sensors or UI.

Two sentences, contradicting each other, and the second one denies a
capability she demonstrably has — she reads the screen, and the frontmost
window is the easiest thing on it to report.

Nothing reached the turn. The screen-observation pattern wanted the word
"screen" or "display", and the ordinary way to ask this never says it: "what
app am I in", "what am I looking at", "which application is frontmost". A
question that reaches no reading is answered from priors, and the prior for an
assistant is that it cannot see anything.
"""

from __future__ import annotations

import pytest

from core.runtime.desktop_objective_intent import looks_like_screen_observation


@pytest.mark.parametrize(
    "question",
    [
        "can you tell what app I'm using right now?",
        "what app am I in?",
        "what window is in front?",
        "what am I looking at?",
        "which application is frontmost?",
        "what document am I working on?",
        "what's on my screen right now?",
        "can you see what's on the screen and tell me what you see?",
    ],
)
def test_a_question_about_the_front_window_reads_the_screen(question: str) -> None:
    assert looks_like_screen_observation(question)


@pytest.mark.parametrize(
    "question",
    [
        # Advice about software is not a question about this screen.
        "what app should I use for invoicing?",
        "which application would you recommend for notes?",
        # Actions still belong to the actuation lane.
        "make a file on my desktop called notes.txt",
        "look at the screen and close the window",
        "what is 2 + 2",
        "how are you",
    ],
)
def test_an_unrelated_question_is_not_a_screen_read(question: str) -> None:
    assert not looks_like_screen_observation(question)
