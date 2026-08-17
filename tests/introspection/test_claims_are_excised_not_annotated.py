"""An ungrounded claim must be REMOVED, not annotated.

The guards all appended a correction and left the claim standing. That is not a
fix, it is an argument the reply has with itself: the person still reads "the
sun is shining", and a retraction two lines below does not un-say it. It makes
the reply longer and asks them to decide which half to believe.

The behaviour change is that the sentence is not there.
"""

from __future__ import annotations

from core.introspection.self_evidence import excise_unsupported_sensory_claims
from interface.routes.chat import _append_sensory_claim_correction as guard


def test_the_invention_is_gone_from_the_reply() -> None:
    out = str(guard("x", "It's afternoon. The sun is shining. My buffers are empty."))

    assert "The sun is shining" not in out.split("I cut a sentence")[0]


def test_the_surrounding_answer_survives() -> None:
    """Deleting a good reply over one bad sentence trades one failure for another."""
    out = str(guard("x", "It's afternoon. The sun is shining. My buffers are empty."))

    assert "It's afternoon." in out
    assert "My buffers are empty." in out


def test_the_disclosure_names_what_was_cut() -> None:
    out = str(guard("x", "It's afternoon. The sun is shining."))

    assert "I cut a sentence" in out
    assert "The sun is shining" in out  # quoted in the disclosure, not asserted


def test_a_reply_that_was_only_the_claim_becomes_the_absence() -> None:
    out = str(guard("x", "The sun is shining."))

    assert out.strip().startswith("I cut a sentence")


def test_a_clean_reply_is_returned_untouched() -> None:
    original = "I fixed the parser and pushed it."

    assert str(guard("x", original)) == original


def test_punctuation_is_not_fused_when_a_middle_sentence_goes() -> None:
    kept, removed = excise_unsupported_sensory_claims(
        "One. The sun is shining. Three."
    )

    assert kept == "One. Three."
    assert removed == ["The sun is shining."]


def test_nothing_unsupported_means_nothing_removed() -> None:
    text = "I read the file and counted nine entries."
    kept, removed = excise_unsupported_sensory_claims(text)

    assert kept == text
    assert removed == []


def test_empty_input_is_safe() -> None:
    for value in ("", None, "   "):
        kept, removed = excise_unsupported_sensory_claims(value)
        assert removed == []
