"""Her mood went into the context of every turn, and she opened by reporting it.

`inject_operational_self_context` prepends a block to the user's message on
every chat turn. Four of its lines are readings of her condition — runtime
state, mood, agency, presence, continuity — and a model handed a mood in its
context says so.

LIVE, 2026-09-08, four questions in a row about arithmetic and astronomy:

    "I'm feeling a bit drained right now, but let's get through this
     calculation."
    "I'm tired right now, so I'll keep this rough and direct."
    "I'm a bit drained right now, but I can work through this for you."

Nobody had asked. The rest of the block — what she is, what she can do, and
the binding rules about what she may claim — stays on every turn, because it
bounds what she may say rather than telling her how she feels.
"""

from __future__ import annotations

import asyncio

import pytest

from core.conversation.chat_preflight import inject_operational_self_context

_A_CONDITION_READING = ("Mood:", "Runtime state:", "Functional agency signal:")


def _block_for(question: str) -> str:
    return asyncio.run(inject_operational_self_context(question))


@pytest.mark.parametrize(
    "question",
    [
        "How many minutes of daylight does 45N lose between the solstice and the equinox?",
        "A train leaves at 09:40 and arrives at 14:15. Show the subtraction.",
        "Write me a haiku about rain.",
        "What is the capital of Peru?",
        "",
    ],
)
def test_a_question_about_something_else_carries_none_of_her_condition(question):
    block = _block_for(question)
    for reading in _A_CONDITION_READING:
        assert reading not in block, f"{reading!r} reached a turn that never asked"


@pytest.mark.parametrize(
    "question",
    [
        "How are you feeling right now?",
        "Are you ok?",
        "Which of your subsystems is degraded?",
        "How much memory are you using?",
    ],
)
def test_a_question_about_her_still_carries_it(question):
    """Not gating into a coma: the turns this block exists for keep it."""
    block = _block_for(question)
    assert "Mood:" in block, f"{question!r} asks about her and got no reading"


def test_what_bounds_her_claims_is_on_every_turn():
    block = _block_for("What is the capital of Peru?")
    assert "[Operational Self Context]" in block
    assert "What I am:" in block
    assert "How I choose and act (binding):" in block
    assert "Evidence boundary:" in block


def test_an_unreadable_question_keeps_the_condition_out():
    """Unable to read the question is not licence to answer it."""
    import inspect

    source = inspect.getsource(inject_operational_self_context)
    at = source.index("asked_about_her = False")
    tail = source[at:]
    assert "except (ImportError, AttributeError, TypeError, ValueError):" in tail
    assert tail.count("asked_about_her = False") >= 2
