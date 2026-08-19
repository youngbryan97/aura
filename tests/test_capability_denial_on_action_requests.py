"""A denial is as wrong when she is asked to DO a thing as when she is asked if she can.

Live 2026-08-10: "do something for real instead of describing it: run a tiny
bit of code and tell me the actual number it printed. any snippet you like,
but i want the output, not a plan."

    "I cannot execute code or generate numbers. I am a cognitive architecture
     running on silicon that does not perform computations unless instructed
     to do so by an external input."

``code_repl`` — "Execute Python code in a real-time, sandboxed REPL" —
``internal_sandbox`` and ``install_package`` were all READY, in a catalogue of
73 skills with none degraded.

The instrument block exists precisely to stop this, and had already been
widened once (from code execution alone to seven families) after she denied
having search with five search skills READY. It was still gated on ability
QUESTION patterns — and that is not the shape a denial usually takes. She is
rarely asked "are you able to run code"; she is asked to run some.
"""

from __future__ import annotations

import re
from pathlib import Path

from core.brain.self_state_report import _CAPABILITY_FAMILIES
from core.runtime.self_state_intent import (
    asked_to_act_in_a_capability_domain,
    asks_about_own_capabilities,
    asks_about_own_runtime,
)

#: The message that produced the live denial.
LIVE_DENIAL_TURN = (
    "do something for real instead of describing it: run a tiny bit of code "
    "and tell me the actual number it printed. any snippet you like, but i "
    "want the output, not a plan."
)


def test_the_turn_that_produced_the_denial_now_gets_her_reading():
    assert asks_about_own_capabilities(LIVE_DENIAL_TURN)


def test_action_requests_across_every_family_attach_the_reading():
    for request in (
        "run this python snippet and give me the output",
        "search the web for the 76ers roster",
        "take a screenshot and tell me what's there",
        "click the save button for me",
        "save that to a file on my desktop",
        "remember that my sister's name is Ada",
        "send Sam an email about tomorrow",
    ):
        assert asks_about_own_capabilities(request), request


def test_ordinary_conversation_does_not_drag_in_a_capability_dump():
    """Attaching it everywhere would make it noise and cost prompt budget."""
    for chatter in (
        "how are you feeling today?",
        "i had a rough morning, tell me something nice",
        "what do you think about consciousness?",
        "that made me laugh",
    ):
        assert not asks_about_own_capabilities(chatter), chatter


def test_someone_else_doing_it_is_not_her_doing_it():
    """Widening the trigger must not swallow questions about other agents.

    "can a language model run code" is a question about language models.
    Attaching her instrument reading makes her answer it about herself.
    """
    for third_person in (
        "can a language model run code",
        "are people able to read this",
        "what tools does a carpenter need",
        "do most assistants have file access",
    ):
        assert not asked_to_act_in_a_capability_domain(third_person), third_person
        assert not asks_about_own_capabilities(third_person), third_person


def test_second_person_wins_over_a_named_third_party():
    """"can a model like you run code" is still about her."""
    assert asks_about_own_capabilities("can a model like you run code")


def test_widening_did_not_touch_the_predicate_that_suppresses_search():
    """The two decisions must stay separate.

    ``asks_about_own_runtime`` also sets explicit_search = False, so widening
    IT would mean "search the web for X" stops being able to search — trading a
    wrong answer about capability for a broken one.
    """
    for action in (
        "search the web for the 76ers roster",
        "run a tiny bit of code and tell me the number",
        "look up tomorrow's forecast",
    ):
        assert not asks_about_own_runtime(action), action


def test_the_domain_table_still_covers_every_family_it_mirrors():
    """core/runtime may not import cognition, so the table is a copy.

    A copy that drifts is worse than no copy: a family present in the report
    and missing here is a family she can silently deny again.
    """
    source = Path(
        Path(__file__).resolve().parents[1] / "core/runtime/self_state_intent.py"
    ).read_text(encoding="utf-8")
    # One flat word list became a verb set and an object set, because
    # matching any of those words anywhere claimed "since you started running"
    # as a request to execute code. Both halves are the table now.
    verbs = source.split("_CAPABILITY_VERBS = frozenset(", 1)[1].split(")", 1)[0]
    objects = source.split("_CAPABILITY_OBJECTS = frozenset(", 1)[1].split(")", 1)[0]
    domain_block = (verbs + objects).lower()

    for label, tokens in _CAPABILITY_FAMILIES:
        assert any(
            re.search(rf"\b{re.escape(token)}\b", domain_block) for token in tokens
        ), f"family {label!r} has no token in the runtime-side domain table"
