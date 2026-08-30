""""Put X on my clipboard" is the same request as "copy X to my clipboard".

LIVE 2026-08-10: "Put the text ORION-7 on my clipboard" routed nowhere, nothing
ran, and she said "The text ORION-7 is now on your clipboard" with the
clipboard empty. "copy that to my clipboard" reached the lane the whole time.

The verbs of putting something somewhere are a closed class, and they were
enumerated in two files that had drifted apart.
"""

from __future__ import annotations

import pytest

from core.conversation.request_mood import assess_request_mood
from core.runtime.desktop_objective_intent import looks_like_desktop_objective


@pytest.mark.parametrize(
    "asked",
    [
        "put BUILD-42 on my clipboard",
        "Put the text ORION-7 on my clipboard",
        "copy that to my clipboard",
        "paste it into Notes",
        "rename that file to notes.txt",
        "share that folder",
        "add a line to notes.txt",
    ],
)
def test_a_request_to_put_something_somewhere_reaches_the_lane(asked):
    assert assess_request_mood(asked).asks_for_action
    assert looks_like_desktop_objective(asked)


@pytest.mark.parametrize(
    "asked",
    [
        "put the kettle on",
        "I put the milk in the fridge",
        "add two and two",
        "what did you think of the film",
    ],
)
def test_the_same_verb_with_nothing_of_the_machine_in_it_does_not(asked):
    """The verb says the act is asked for now. The object says whose act it is."""
    assert not looks_like_desktop_objective(asked)


def test_the_two_lists_of_these_verbs_agree():
    """They were enumerated twice and drifted, which is how "put" reached one."""
    from core.runtime.desktop_objective_intent import _DESKTOP_OBJECTIVE_ACTION_TERMS
    from core.conversation.request_mood import _ACTION_VERBS

    putting = {
        "put", "place", "paste", "insert", "append", "attach", "stick",
        "rename", "print", "export", "import", "empty", "clear", "add",
        "remove", "drop", "share",
    }
    in_mood = set(_ACTION_VERBS.split("|"))
    missing_from_mood = sorted(putting - in_mood)
    missing_from_desktop = sorted(putting - set(_DESKTOP_OBJECTIVE_ACTION_TERMS))
    assert not missing_from_mood, f"request_mood is missing {missing_from_mood}"
    assert not missing_from_desktop, f"desktop intent is missing {missing_from_desktop}"
