"""Making a file is not looking at the screen.

LIVE 2026-08-18. "make a file on my desktop called aura-live-check.txt with
one sentence about what you are doing right now" finished in 2ms and answered
"Done — the desktop steps completed and their effects verified". Nothing was
written, anywhere.

The objective went to the ambient screen-observation lane, which answers from
a cached reading and returns a receipt that is perfectly honest about what it
did — it observed the frontmost window. The request was to WRITE.

It routed there because the action vocabulary held "create" but not "make", so
the sentence named no action at all and the reading phrase at the end ("what
you are doing right now") decided the lane. Same shape as the earlier gap
where "put" and "copy" were missing while "paste" was present: an everyday
word for the act, absent from an enumeration.
"""

from __future__ import annotations

import pytest

from core.runtime.desktop_objective_intent import looks_like_screen_observation
from core.skills.desktop_task import DesktopTaskSkill


class _NoSteps:
    steps = None


@pytest.mark.parametrize(
    "objective",
    [
        "make a file on my desktop called aura-live-check.txt with one sentence about what you are doing right now",
        "build me a note on the desktop about what you see right now",
        "generate a summary file of what's on my screen",
        "draft a document about what's on screen right now",
        "look at the screen and close the window",
    ],
)
def test_a_request_to_produce_something_is_not_an_observation(objective: str) -> None:
    assert not looks_like_screen_observation(objective)


@pytest.mark.parametrize(
    "objective",
    [
        "what's on my screen right now?",
        "can you see what's on the screen and tell me what you see?",
        "look at my screen and tell me what the paper is about",
    ],
)
def test_a_real_screen_question_still_reads_the_screen(objective: str) -> None:
    assert looks_like_screen_observation(objective)


def test_the_write_objective_does_not_take_the_ambient_shortcut() -> None:
    skill = DesktopTaskSkill()
    objective = (
        "make a file on my desktop called aura-live-check.txt with one "
        "sentence about what you are doing right now"
    )

    assert skill._ambient_answer(objective, _NoSteps()) is None


def test_the_write_objective_plans_a_write() -> None:
    skill = DesktopTaskSkill()
    steps = skill._derive_steps_from_objective(
        "make a file on my desktop called aura-live-check.txt with one sentence "
        "about what you are doing right now",
        {},
    )

    assert [step.action for step in steps] == ["write_text_file"]
    assert "aura-live-check.txt" in str(steps[0].target)
