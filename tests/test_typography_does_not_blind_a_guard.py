"""A model writes "I’ve"; the patterns that police it said "I've".

LIVE, 2026-08-22. Beneath a measured line saying the session was three
minutes old, she wrote "I’ve been up since 0600", and the check written to
strike exactly that claim did not fire. The rule was right, the text was
right, and they could not meet.

Sweeping the module afterwards found 27 single-argument checkers in
response_reliability that answered differently for the two apostrophes, two
of them the guards against fabricated completion claims and fabricated tool
receipts — the shape a fabrication actually arrives in was the shape that
walked past.
"""

from __future__ import annotations

import core.conversation.response_reliability as rr
from core.language.typography import fold_typography


def curly(text: str) -> str:
    return text.replace("'", "’")


CLAIMS = (
    "I've set a reminder for 20 minutes to check the oven.",
    "I've saved it as notes.md in your Downloads folder.",
    "I've added that to your calendar.",
    "I've created the file and put it on your desktop.",
)


def test_folding_leaves_plain_text_alone():
    assert fold_typography("nothing to fold here") == "nothing to fold here"


def test_folding_covers_more_than_quotes():
    folded = fold_typography("a—b “c” d… e f")
    assert folded == 'a-b "c" d... e f'


def test_an_action_claim_is_caught_in_either_apostrophe():
    for claim in CLAIMS:
        assert rr._sentence_claims_an_action(claim), claim
        assert rr._sentence_claims_an_action(curly(claim)), curly(claim)


def test_a_fabricated_tool_receipt_is_caught_in_either_apostrophe():
    receipt = "I've run the calculation with Python. Output: 4"
    assert rr._has_unfounded_tool_execution_claim(receipt)
    assert rr._has_unfounded_tool_execution_claim(curly(receipt))


def test_the_shared_normaliser_folds_typography():
    assert rr._normalize("I’ve done it") == rr._normalize("I've done it")


def test_an_offer_is_still_not_a_claim_in_either_apostrophe():
    """Folding must not turn a question into a completion."""
    for offer in ("I can save it as notes.md if you like.", "Shall I put that on your calendar?"):
        assert not rr._sentence_claims_an_action(offer)
        assert not rr._sentence_claims_an_action(curly(offer))


def test_an_offer_phrased_as_a_question_is_never_a_completion():
    """This guard destroys a reply rather than editing it, so a false
    positive throws away a perfectly good offer to help."""
    offers = (
        "Shall I put that on your calendar?",
        "Should I save it as notes.md?",
        "Would you like me to set a reminder for the oven?",
        "Do you want me to create the file on your desktop?",
        "Can I add that to your calendar?",
    )
    for offer in offers:
        assert not rr._sentence_claims_an_action(offer), offer
        assert not rr._sentence_claims_an_action(curly(offer)), offer


def test_a_real_claim_is_still_caught_beside_the_offers():
    """The exemption is for questions, not for the verbs."""
    assert rr._sentence_claims_an_action("I put that on your calendar.")
    assert rr._sentence_claims_an_action("I've set a reminder for the oven.")
