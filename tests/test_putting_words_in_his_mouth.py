"""Attributing a claim to him is fabrication whatever the tense.

LIVE, 2026-08-18. Asked how many Python files live in core/introspection and
whether her previous non-answer had felt off, she replied:

    There are 10 Python files in the core/introspection directory.
    ... But now that you point out the 'your_files' directory exists,
    something did feel off about my response.

The count was exactly right. The second half was invented: no "your_files"
directory was ever mentioned, by anyone, and she attributed the claim to him
to build agreement on top of it.

`fabricated_shared_history` exists for precisely this and returned nothing.
Every one of its patterns was PAST tense — "you said", "you told me", "your
last message" — and "you point out" is present. Putting words in someone's
mouth does not require the past tense.
"""

from __future__ import annotations

import pytest

from core.dialogue.shared_history import fabricated_shared_history

USER_TURN = (
    "different angle on the same thing: how many python files live in "
    "core/introspection? and while you're at it - when you just told me there "
    "was no directory called 'your', did that feel like an answer to you, or "
    "did something about it feel off?"
)


def test_the_invented_attribution_is_caught():
    reply = (
        "There are 10 Python files in the core/introspection directory. "
        "But now that you point out the 'your_files' directory exists, "
        "something did feel off about my response."
    )
    flagged = fabricated_shared_history(reply, USER_TURN)
    assert flagged, "an invented claim was attributed to him and not caught"
    assert "your_files" in flagged[0]


def test_a_genuine_attribution_is_not_flagged():
    """Referring to what he actually said must stay allowed.

    The grounding test downstream is what separates these: a real attribution
    shares content words with his message.
    """
    reply = (
        "There are 10 Python files in core/introspection. As you point out, "
        "there was no directory called 'your', and that answer was too quick."
    )
    assert fabricated_shared_history(reply, USER_TURN) == []


@pytest.mark.parametrize(
    "phrasing",
    [
        # A speech-act verb is a closed grammatical CLASS. Naming six of them
        # was the same mistake the past-tense scope was: right for the example
        # in hand, one wording behind the next one.
        "you note that the deployment already shipped",
        "you claim the deployment already shipped",
        "you argued the deployment already shipped",
        "you imply the deployment already shipped",
        "you described the deployment as already shipped",
        "you reported the deployment already shipped",
        "you insist the deployment already shipped",
        "you admitted the deployment already shipped",
        "you confirm the deployment already shipped",
        "you acknowledged the deployment already shipped",
        "you state the deployment already shipped",
        "you've mentioned the deployment already shipped",
        "you point out that the deployment already shipped",
        "you note that the deployment already shipped",
        "you're saying the deployment already shipped",
        "as you mention, the deployment already shipped",
        "you just said the deployment already shipped",
        "you're telling me the deployment already shipped",
    ],
)
def test_present_tense_attributions_are_all_reachable(phrasing):
    """One phrasing slipping through is the whole defect."""
    reply = f"Right — {phrasing}, so I will hold off."
    assert fabricated_shared_history(reply, USER_TURN), phrasing


def test_past_tense_attributions_still_work():
    """The patterns that already existed must keep working."""
    reply = "You told me the deployment already shipped, so I held off."
    assert fabricated_shared_history(reply, USER_TURN)


def test_no_context_still_means_no_verdict():
    """An empty vocabulary makes everything novel; that guard must survive."""
    assert fabricated_shared_history("You point out that X happened.", "") == []
