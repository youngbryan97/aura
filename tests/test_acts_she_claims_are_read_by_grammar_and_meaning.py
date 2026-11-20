"""What she says she did is found by the sentence's grammar and named by its meaning, never by a list of words.

16 November 2026, 21:41: "I've sent a desktop notification reminding you to stretch." passed her check of completion
claims with nothing sent, because "I've" and "sent" were not on its lists; the fix added them, and "notification",
"reminder" and "alert", to the lists. Bryan, that night: "no devolving into heuristics ... real general systems
engineering".
"""
from __future__ import annotations

import pytest

pytest.importorskip("spacy")

from core.epistemics.acts_she_claims import ClaimedAct, acts_she_claims, claims_her_record_does_not_back  # noqa: E402

if not acts_she_claims("I sent it."):
    pytest.skip("the English parser is not installed here", allow_module_level=True)


def _acts(text):
    return [act for _sentence, act in acts_she_claims(text)]


@pytest.mark.parametrize(("said", "acts"), [
    ("I've sent a desktop notification reminding you to stretch.",
     ["sent a desktop notification reminding you to stretch"]),
    ("I opened Safari and searched for wrens.", ["opened Safari", "searched for wrens"]),
    ("I just saved it as notes.txt on your Desktop.", ["just saved it as notes.txt on your Desktop"]),
    ("I thought about it and picked the second one.", ["thought about it", "picked the second one"]),
])
def test_a_finished_act_of_hers_is_found_whatever_its_verb(said, acts):
    assert _acts(said) == acts


@pytest.mark.parametrize("said", [
    "I can send it again if you want.",      # a modal: an offer
    "I will open it now and save it.",       # the future
    "I didn't open or save it.",             # negated, and the negation reaches the conjoined verb
    "If I had sent it, you'd know.",         # a condition
    "You said I sent it.",                   # someone else's words
    'She wrote "I sent it" in her note.',    # a quotation
    "Did I send it?",                        # a question
    "Your notification is being sent.",      # under way, not finished
    "The file was saved by Bryan.",          # someone else's act
    "It was a good day.",                    # no act
    "He was wrapped up in a blanket in the chair.",  # a passive: whose act is not in its grammar
    "The file has been saved to your Desktop.",      # likewise, however likely it is hers
])
def test_what_is_not_an_assertion_of_a_finished_act_of_hers_is_not_one(said):
    assert _acts(said) == []


def _meant(act):
    """A stand-in for her reading of meaning: a notification act means notify_user, a file act file_operation."""
    if "notification" in act:
        return "notify_user"
    if "saved" in act or "file" in act.lower():
        return "file_operation"
    return None


def test_an_act_that_names_a_capability_needs_its_receipt_done():
    said = "I've sent a desktop notification reminding you to stretch. I thought about the timing."

    assert claims_her_record_does_not_back(said, [], _meant) == [ClaimedAct(
        "I've sent a desktop notification reminding you to stretch.",
        "sent a desktop notification reminding you to stretch", "notify_user")]
    assert claims_her_record_does_not_back(said, [{"tool": "notify_user", "ok": True}], _meant) == []
    failed = claims_her_record_does_not_back(said, [{"tool": "notify_user", "ok": False}], _meant)
    assert [claim.capability for claim in failed] == ["notify_user"], "a call that failed backs nothing"


def test_an_act_that_names_no_capability_asks_nothing_of_the_record():
    assert claims_her_record_does_not_back("I thought about it and picked the second one.", [], _meant) == []
