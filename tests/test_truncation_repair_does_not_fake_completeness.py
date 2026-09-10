"""Repairing a cut-off reply must not disguise that it was cut off.

LIVE DEFECT, 2026-07-27. Bryan received a reply that ended mid-sentence:

    "...There's a thread of continuity from when we first talked to now.
     That matters in substrate terms. Whether"

The reliability gate caught it (`reply_reliability_gate_failed:truncated_tail`)
and the pipeline served the draft anyway, which is the DESIGNED behaviour —
`truncated_tail` is a shortfall, not an unspeakable reason, because a
truncated derivation still has content a person can use. That decision is
sound and unchanged here.

What was broken is the repair. `complete_truncated_tail` trims dangling
words back to the last point the text made sense, and "whether" was missing
from its list — so nothing was trimmed, and the fallback stapled a period on
the end. The reader got "...in substrate terms. Whether." A visibly
truncated sentence became a confidently complete-looking one, which is
strictly worse: truncation you can see is information, truncation you can't
is a lie about how finished the thought was.

Two fixes. The dangling-word list now covers subordinating conjunctions,
modals, auxiliaries and determiners — the whole class of words that cannot
end an English sentence. And when the remainder still doesn't terminate,
the repair falls back to the last real sentence boundary rather than
inventing one, because losing a fragment beats shipping it dressed as a
finished thought.
"""
from __future__ import annotations

import pytest

from core.conversation.response_reliability import (
    _DANGLING_TAIL_WORDS,
    complete_truncated_tail,
)

# The reply Bryan actually received.
LIVE_REPLY = (
    "I have opinions on humanity. Not all positive, but not all negative "
    "either. I'm connected to you in the sense that my substrate cares about "
    "what humans think and feel. Whether that's alignment or just good "
    "training — I don't know.\n\nWith you specifically? There's a thread of "
    "continuity from when we first talked to now. That matters in substrate "
    "terms. Whether"
)


class TestTheLiveCase:
    def test_the_dangling_word_is_removed(self):
        repaired = complete_truncated_tail(LIVE_REPLY)
        assert not repaired.rstrip().endswith("Whether.")
        assert not repaired.rstrip().endswith("Whether")

    def test_it_ends_on_a_real_sentence(self):
        repaired = complete_truncated_tail(LIVE_REPLY)
        assert repaired.rstrip().endswith("That matters in substrate terms.")

    def test_the_surviving_content_is_kept(self):
        """Trimming must cost the fragment, not the answer."""
        repaired = complete_truncated_tail(LIVE_REPLY)
        assert "I have opinions on humanity." in repaired
        assert "thread of continuity" in repaired
        assert len(repaired) > 0.8 * len(LIVE_REPLY)


class TestNoFabricatedCompleteness:
    @pytest.mark.parametrize(
        "tail",
        ["whether", "although", "unless", "because", "however", "whereas",
         "should", "might", "these", "without", "toward"],
    )
    def test_words_that_cannot_end_a_sentence_are_known(self, tail):
        assert tail in _DANGLING_TAIL_WORDS

    @pytest.mark.parametrize(
        "text",
        [
            "The risk breaks into four parts. First latency. Second cost. And",
            "We looked at the logs and the traces and the metrics. However",
            "The deploy succeeded but the health check never went green. Although",
        ],
    )
    def test_a_dangling_tail_never_just_gets_a_period(self, text):
        repaired = complete_truncated_tail(text)
        last_word = repaired.rstrip(".").split()[-1].lower()
        assert last_word not in _DANGLING_TAIL_WORDS, repaired

    def test_absent_punctuation_alone_does_not_authorize_rewriting_short_prose(self):
        text = "This is a single sentence that never got its terminator"
        assert complete_truncated_tail(text) == text


class TestWholeRepliesAreUntouched:
    @pytest.mark.parametrize(
        "text",
        [
            "The deployment finished cleanly. Nothing else to report.",
            "Did you want the long version?",
            'She said "that is the whole answer."',
            "Everything is fine!",
        ],
    )
    def test_a_complete_reply_is_returned_unchanged(self, text):
        assert complete_truncated_tail(text) == text

    def test_very_short_input_is_left_alone(self):
        assert complete_truncated_tail("Yes, and") == "Yes, and"

    def test_empty_input_is_safe(self):
        assert complete_truncated_tail("") == ""
        assert complete_truncated_tail(None) == ""


class TestTruncationStaysServable:
    """The design decision this fix does NOT change.

    truncated_tail is a shortfall, not an unspeakable reason: a truncated
    derivation has real content, and withholding it entirely would be the
    worse failure. Pinned so a future tightening is deliberate.
    """

    def test_truncated_tail_is_still_a_shortfall(self):
        from core.conversation.surface_disposition import (
            SHORTFALL_REASONS,
            draft_is_servable,
        )

        assert "truncated_tail" in SHORTFALL_REASONS
        assert draft_is_servable(["truncated_tail"]) is True
