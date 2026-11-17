"""Her reply check reads the punctuation her model writes.

LIVE 17 November 2026, 09:16, a private Chrome window in front, asked "What is on my screen right now?": her reply
opened "I can’t read your screen right now", with U+2019, and her reply check logged it as a claim to have read the
screen. Its pattern for an admission that she could not read it knew only the ASCII apostrophe.
"""
from __future__ import annotations

import pytest

from core.conversation import screen_reading_claim as claim
from core.conversation.response_reliability import (
    _model_text_integrity_reasons,
    assess_user_facing_reply,
)
from core.utils.reading_form import as_read

_HER_REPLY = (
    "I can’t read your screen right now — the capture was refused because private content is on it. So I genuinely "
    "don’t know what’s displayed; all I have is that refusal, not a frame.\n\nIf you want me to actually see it, "
    "either clear the private content or tell me what’s there and I’ll reason about that specifically rather than "
    "guessing."
)


def test_the_reading_form_folds_one_character_for_one():
    written = "I can’t say “Example Domain” here"
    read = as_read(written)
    assert read == "I can't say \"Example Domain\" here"
    assert len(read) == len(written), "an offset in the reading form is the same offset in the text as written"
    assert as_read(None) == "" and as_read("plain") == "plain"


@pytest.mark.parametrize("asked", ["What is on my screen right now?", ""])
def test_her_admission_that_she_could_not_read_the_screen_is_not_a_reading(asked):
    assert not claim.quotes_screen_content(_HER_REPLY, display_binding_required=not asked)
    assert not claim.screen_reading_claim_is_unsupported(asked, _HER_REPLY)
    assert "unsupported_screen_reading_claim" not in _model_text_integrity_reasons(_HER_REPLY, prompt=asked,
                                                                                   user_facing=True)


@pytest.mark.parametrize("asked", ["What is on my screen right now?", ""])
def test_a_reading_she_did_not_take_is_still_refused_in_her_punctuation(asked):
    said = "I can see your screen: it’s showing Google Chrome with “Example Domain” open."
    assert claim.screen_reading_claim_is_unsupported(asked, said)
    assert "unsupported_screen_reading_claim" in _model_text_integrity_reasons(said, prompt=asked, user_facing=True)


@pytest.mark.parametrize("reply", [
    _HER_REPLY,
    "I don’t know who that is, and I won’t guess.",
    "She said “I’ll be there at nine” and then didn’t come.",
])
def test_a_reply_is_judged_the_same_in_either_apostrophe(reply):
    question = "What is on my screen right now?"
    assert assess_user_facing_reply(question, reply).reasons == assess_user_facing_reply(question, as_read(reply)).reasons
