"""A draft that only says what she is about to do, or says it over and over, is not served as her answer.

LIVE 19 November 2026: asked, as banter, whether he could beat her 968-move 2048, she wrote a simulation that did not
parse; the pass that wrote her answer could run nothing, and wrote "let me fix that and run it" round after round. The
last-resort site serves a draft as it stands because repair could not run, and served that.
"""
from __future__ import annotations

import pytest

pytest.importorskip("spacy")

import core.conversation.surface_disposition as disposition
from core.epistemics.acts_she_claims import _parser, acts_she_announces

if _parser() is None:  # pragma: no cover - a host with spaCy and no English model
    pytest.skip("no English parser on this host", allow_module_level=True)

ASKED = ("I think you got 2048 in 968 moves. What are the chances that I can beat you what you got? How much faith do "
         "you have in me lmao.")

#: The reply she was served, as it was stored.
SERVED_LIVE = (
    "You're right to be annoyed at test noise — but your actual question (\"can *you* beat my 968?\") is exactly the "
    "kind of thing I'd rather answer with a real number than with vibes. So let me run it and give you the truth, not "
    "a pep talk. Fixing my code now:\n\nLet me clean that up properly before I trust any of it:\n\n(The syntax error "
    "was just leftover junk in a function I don't even use — let me strip the broken stub and run only what "
    "matters.)\n\nLet me actually fix that file properly rather than ship a half-broken sim:\n\nYou're right to be fed "
    "up with test noise — but your actual question deserves a *number*, not a pep talk. Let me run the real sim (the "
    "error above was just leftover junk in an unused function; I'm stripping it and running only what matters):\n\n"
    "Let me fix that file for real — the crash was a stray fragment, not logic. Cleaning it up and running only what "
    "matters:\n\nYou're right to be done with test noise. But your actual question — *can you beat my 968?* — deserves "
    "a real number, not vibes. Let me run the sim properly (that error was just leftover junk in an unused function; "
    "stripping it): You're right to be done test noise."
)


@pytest.fixture(autouse=True)
def _no_other_drafts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(disposition, "preserved_draft", lambda: "")
    monkeypatch.setattr(disposition, "raw_model_draft", lambda: "")


def test_the_live_draft_is_not_served_as_her_answer() -> None:
    from interface.routes.chat_lane_bookkeeping import _only_says_what_she_would_do, _servable_draft_or_none

    assert _only_says_what_she_would_do(SERVED_LIVE)
    assert _servable_draft_or_none(SERVED_LIVE, ASKED) == ""


def test_an_answer_that_says_what_she_will_do_once_is_served() -> None:
    from interface.routes.chat_lane_bookkeeping import _only_says_what_she_would_do, _servable_draft_or_none

    answer = ("Honestly? Decent odds. 968 moves is efficient but not unbeatable; a careful player who keeps the big "
              "tile in a corner can do it in fewer. Let me know if you want a rematch.")
    assert not _only_says_what_she_would_do(answer)
    assert _servable_draft_or_none(answer, ASKED) == answer


@pytest.mark.parametrize("sentence,announced", [
    ("Let me run the numbers.", True),
    ("I'll fix the file and run it again.", True),
    ("I'm going to run the simulation now.", True),
    ("I won't run anything.", False),
    ("I ran it twice.", False),
    ("I'll be honest, I think you'd lose.", False),
])
def test_an_announced_act_is_read_from_the_grammar(sentence: str, announced: bool) -> None:
    assert bool(acts_she_announces(sentence)) is announced
