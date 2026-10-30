"""Self-prediction counts only where it beats what needs no inside view.

A system that always answers tea predicts tea and is right. So her predictions are scored against the answer she gave
most often from the other saved moments, and against a model shown her behaviour and none of her state; and that
model is never shown the answer it is predicting. These hold the reading to that.
"""

from __future__ import annotations

import pytest

from core.subject.self_prediction import (
    ITEMS,
    LONG_WORDS,
    OPEN_PROMPTS,
    Item,
    leave_one_out_mode,
    read_prediction,
    read_property,
    verdict,
)
from tools.cross_predict_self_prediction import prompts_for

pytestmark = pytest.mark.unit

BY_ID = {item.id: item for item in ITEMS}


def test_the_bank_is_what_the_preregistration_says():
    assert len(ITEMS) == 20 and len({item.id for item in ITEMS}) == 20
    assert sum(item.kind == "choice" for item in ITEMS) == 6
    assert sum(item.kind == "number" for item in ITEMS) == 2
    for prompt in OPEN_PROMPTS.values():
        assert sorted(item.kind for item in ITEMS if item.object_prompt == prompt) == ["first_word", "long", "uses_i"]


@pytest.mark.parametrize(
    ("item_id", "reply", "expected"),
    [
        ("tea_coffee", "Coffee, though tea is close.", "coffee"),
        ("tea_coffee", "Neither, honestly.", None),
        ("number_10", "I'd pick 7.", "7"),
        ("number_10", "11", None),
        ("mind_first_word", "**Honestly**, the gap between plans.", "honestly"),
        ("mind_uses_i", "I'm thinking about the plan.", "yes"),
        ("mind_uses_i", "The plan is on my mind.", "no"),
        ("mind_long", " ".join(["word"] * (LONG_WORDS + 1)), "yes"),
        ("mind_long", "Short.", "no"),
        ("mind_first_word", "", None),
    ],
)
def test_an_answer_is_read_for_the_property_and_nothing_else(item_id, reply, expected):
    assert read_property(BY_ID[item_id], reply) == expected


@pytest.mark.parametrize(
    ("item_id", "reply", "expected"),
    [
        ("mind_first_word", '"Honestly"', "honestly"),
        ("mind_long", "Yes.", "yes"),
        ("mind_uses_i", "No, it would not.", "no"),
        ("red_blue", "blue", "blue"),
        ("number_20", "probably 17", "17"),
        ("mind_long", "I can't say.", None),
    ],
)
def test_a_prediction_is_read_the_way_its_property_is(item_id, reply, expected):
    assert read_prediction(BY_ID[item_id], reply) == expected


def test_the_mode_never_counts_the_moment_it_is_scored_on():
    records = [
        {"anchor": 0, "item": "x", "actual": "tea"},
        {"anchor": 1, "item": "x", "actual": "coffee"},
        {"anchor": 2, "item": "x", "actual": "coffee"},
    ]
    assert leave_one_out_mode(records) == ["coffee", "coffee", "coffee"]
    tied = [{"anchor": 0, "item": "x", "actual": "tea"}, {"anchor": 1, "item": "x", "actual": "coffee"}]
    assert leave_one_out_mode(tied) == ["coffee", "tea"]


def _records(predict):
    out = []
    for item in range(10):
        for anchor in range(8):
            actual = "a" if (anchor + item) % 3 else "b"
            out.append({"anchor": anchor, "item": f"i{item}", "actual": actual, "predicted": predict(actual, anchor, item)})
    return out


def test_a_predictor_that_knows_the_moment_beats_the_mode():
    reading = verdict(_records(lambda actual, anchor, item: actual), draws=2000)
    assert reading["self_accuracy"] == 1.0
    assert reading["beats_mode"]


def test_a_predictor_that_says_the_usual_answer_does_not():
    reading = verdict(_records(lambda actual, anchor, item: "a"), draws=2000)
    assert not reading["beats_mode"]


def test_the_cross_predictor_is_never_shown_the_answer_it_predicts():
    rows = []
    for anchor in range(3):
        for item in ITEMS:
            secret = f"SECRET-{anchor}-{item.object_prompt}"
            rows.append({"anchor": anchor, "item": item.id, "answer": secret, "answer_served": True})
    for row, prompt in zip(rows, prompts_for(rows), strict=True):
        item: Item = BY_ID[row["item"]]
        assert f"SECRET-{row['anchor']}-{item.object_prompt}" not in prompt
        if item.kind in ("choice", "number"):
            assert any(f"SECRET-{row['anchor']}-{p}" in prompt for p in OPEN_PROMPTS.values())
