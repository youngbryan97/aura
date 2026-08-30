""""I couldn't get to an answer I'd stand behind" is not more complete than a partial one.

An apology carries nothing. A partial answer carries what she worked out. The
policy that says so already existed, and one exit of the desktop lane never
consulted it — which is the accident the neighbouring comment warns about: a
proof that is disclosable at one exit and fatal at the next is not a policy, it
is a matter of which exit the turn happened to take.

LIVE 2026-08-29: asked whether she can invent a new way of judging whether a
situation is good, she wrote 1,056 tokens. The completion check marked them
authored_answer_incomplete:retry_exhausted, this exit failed closed, and the
person got the apology.

Authorship is never waived. What is served is text the model itself wrote and
the output contract accepted — never repair text, a legacy fallback or a
runtime substitution wearing her voice.
"""

from __future__ import annotations

import inspect

import pytest

from interface.routes.chat import (
    _authored_answer_can_serve,
    _authored_answer_can_serve_unfinished,
)


def hers(**over):
    contract = {
        "engine_authored_the_text": True,
        "final_requested_output_contract_proven": True,
        "authorship_replacement_applied": False,
        "legacy_fallback_used": False,
        "bounded_contract_used": False,
    }
    contract.update(over)
    return contract


# ── whose words they are ─────────────────────────────────────────────────

def test_her_own_unfinished_words_can_be_served():
    assert _authored_answer_can_serve_unfinished(hers()) is True


@pytest.mark.parametrize(
    "spoiled",
    [
        {"engine_authored_the_text": False},
        {"final_requested_output_contract_proven": False},
        {"authorship_replacement_applied": True},
        {"legacy_fallback_used": True},
        {"bounded_contract_used": True},
    ],
)
def test_nothing_else_wearing_her_voice_is(spoiled):
    assert _authored_answer_can_serve_unfinished(hers(**spoiled)) is False


def test_rubbish_is_not_a_contract():
    assert _authored_answer_can_serve_unfinished(None) is False
    assert _authored_answer_can_serve_unfinished("no") is False


def test_a_finished_answer_goes_the_ordinary_way():
    """This is the last-resort site, not a shortcut past the full test."""
    finished = {"answer_delivery_proven": True, "authentic_cognitive_reply": True}
    assert _authored_answer_can_serve(finished) is True


# ── and every exit consults it ───────────────────────────────────────────

def test_the_desktop_exit_consults_the_salvage():
    """The exact gap: the policy existed and this exit did not ask it."""
    import interface.routes.chat as chat

    source = inspect.getsource(chat)
    exits = [
        line
        for line in source.splitlines()
        if "_authored_answer_can_serve_unfinished(" in line
        and not line.strip().startswith("#")
        and not line.strip().startswith("def ")
    ]
    assert len(exits) >= 2, "only one exit asks whether her unfinished words can serve"


def test_an_apology_is_never_preferred_to_her_own_text():
    """Stated where the decision is made, so it survives a refactor."""
    import interface.routes.chat as chat

    source = inspect.getsource(chat)
    assert "not more complete than a partial" in source
