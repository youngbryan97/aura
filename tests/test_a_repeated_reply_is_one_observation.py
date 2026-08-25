"""A repeated reply is one observation, however many rows it fills.

Found by running the trainer on Aura's first real corpus, 2026-08-25. It
returned `content_bearing` — the strongest verdict, meaning the state carries
information about WHAT is said and not merely how — and the result was an
artifact of the split.

116 recorded turns held 39 distinct replies. 41 of them were the single word
"ready" and 37 were a bare comma: 67% of the corpus was two strings. Split by
turn, those strings landed on both sides. The head learned the region of state
space that precedes "ready", and scoring it on held-out "ready" turns returned
a gain on a RARE token, which the trainer read as propositional content.

Every null in the design endorsed it, and each was working correctly.
Permuting the state-to-turn correspondence destroys the mapping, so the null
sits near zero while the observed gain towers over it. A matched null answers
"is this gain bigger than chance". It cannot answer "was this answer in the
training set". Only the split can.

Fixing the split then exposed the second fault: taking whole groups until the
holdout reached its share of turns put 55% of the turns in the holdout,
because one group WAS 41 turns, and left nine distinct replies to fit on. No
null detects that either — the nulls are computed on the holdout.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.brain.llm.endogenous_readout_training import (
    MIN_HOLDOUT_REPLIES,
    MIN_TRAIN_REPLIES,
    TurnTokens,
    _three_way_split,
)


def _turn(group: str, token: int) -> TurnTokens:
    return TurnTokens(
        state=np.zeros(4, dtype=np.float64),
        tokens=np.asarray([token], dtype=np.int64),
        group=group,
    )


def test_a_reply_never_appears_on_both_sides_of_the_split():
    """The whole defect, in one assertion."""
    turns = [_turn("ready", 1) for _ in range(41)]
    turns += [_turn("comma", 2) for _ in range(37)]
    turns += [_turn(f"unique-{index}", index + 10) for index in range(38)]

    train, validation, holdout = _three_way_split(turns, seed=913)

    held = {turn.group for turn in holdout}
    fitted = {turn.group for turn in train} | {turn.group for turn in validation}
    assert not (held & fitted), sorted(held & fitted)


def test_one_huge_group_does_not_swallow_the_holdout():
    """A group that would overshoot the target is passed over, not taken."""
    turns = [_turn("ready", 1) for _ in range(41)]
    turns += [_turn(f"unique-{index}", index + 10) for index in range(75)]

    train, _validation, holdout = _three_way_split(turns, seed=913)

    # A quarter of 116 is 29. The 41-turn group cannot fit inside that, so it
    # belongs on the training side rather than becoming most of the holdout.
    assert len(holdout) <= 29, len(holdout)
    assert len(train) > len(holdout)


def test_a_corpus_of_two_strings_earns_no_verdict():
    """Refusing is the result. Reporting content from it was the bug."""
    from core.brain.llm.endogenous_readout_training import VERDICT_MEANING

    assert "no_verdict_corpus_too_repetitive" in VERDICT_MEANING
    assert MIN_HOLDOUT_REPLIES >= 20
    assert MIN_TRAIN_REPLIES >= 20


@pytest.mark.parametrize(
    ("held", "fitted", "refuses"),
    [
        (MIN_HOLDOUT_REPLIES - 1, MIN_TRAIN_REPLIES, True),
        (MIN_HOLDOUT_REPLIES, MIN_TRAIN_REPLIES - 1, True),
        (MIN_HOLDOUT_REPLIES, MIN_TRAIN_REPLIES, False),
    ],
)
def test_both_sides_have_to_be_varied_enough(held, fitted, refuses):
    """Starving either side invalidates the verdict, for different reasons."""
    from dataclasses import fields

    from core.brain.llm.endogenous_readout_training import VocabFit

    names = {f.name for f in fields(VocabFit)}
    assert {"n_replies_holdout", "n_replies_train"} <= names

    probe = VocabFit.__new__(VocabFit)
    object.__setattr__(probe, "n_replies_holdout", held)
    object.__setattr__(probe, "n_replies_train", fitted)
    assert probe.corpus_is_too_repetitive is refuses


def test_an_unlabelled_corpus_still_splits_turnwise():
    """A turn with no group identity is its own group.

    Constructed corpora in the other suites carry no reply text, and
    collapsing them into one giant group would leave nothing to train on.
    """
    turns = [
        TurnTokens(
            state=np.zeros(4, dtype=np.float64),
            tokens=np.asarray([index], dtype=np.int64),
        )
        for index in range(100)
    ]

    train, validation, holdout = _three_way_split(turns, seed=913)

    assert len(holdout) == 25
    assert len(train) + len(validation) == 75
