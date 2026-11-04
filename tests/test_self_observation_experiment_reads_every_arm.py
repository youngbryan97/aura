"""The self-observation experiment's questions, arms and ceiling (preregistered 4 November 2026)."""

from __future__ import annotations

import os

from core.subject import self_prediction as sp


def test_the_new_questions_are_twenty_four_and_none_was_asked_before():
    items, open_prompts = sp.item_set("v2")
    assert len(items) == 24
    assert sum(1 for i in items if i.kind == "choice") == 4
    assert sum(1 for i in items if i.kind == "number") == 2
    assert len(open_prompts) == 6
    old_ids = {i.id for i in sp.ITEMS}
    old_prompts = {i.object_prompt for i in sp.ITEMS}
    assert not old_ids & {i.id for i in items}
    assert not old_prompts & {i.object_prompt for i in items}


def test_each_arm_is_read_from_its_own_predictions_and_the_second_draw_is_a_ceiling():
    items, _ = sp.item_set("v2")
    item = next(i for i in items if i.id == "rain_snow")
    rows = [
        {"anchor": a, "item": item.id, "answer": ans, "answer_2": second,
         "prediction_with": w, "prediction_without": wo}
        for a, (ans, second, w, wo) in enumerate([
            ("rain", "rain", "rain", "snow"),
            ("snow", "snow", "snow", "rain"),
            ("rain", "snow", "rain", "rain"),
            ("rain", "rain", "rain", "snow"),
        ])
    ]
    with_arm = sp.verdict(sp.records_from(rows, items=items, arm="with"))
    without_arm = sp.verdict(sp.records_from(rows, items=items, arm="without"))
    assert with_arm["self_accuracy"] == 1.0
    assert without_arm["self_accuracy"] == 0.25
    assert with_arm["second_draw_accuracy"] == 0.75


def test_the_outsider_can_be_shown_the_summary_she_was_given():
    item = sp.ITEMS_V2[0]
    shown = sp.cross_prompt(item, ["rain"], {}, reading="- 20 of 20 ran past 40 words.")
    assert "- 20 of 20 ran past 40 words." in shown
    assert "as it was put to her" in shown
    assert "- 20 of 20" not in sp.cross_prompt(item, ["rain"], {})


def test_an_arm_sets_her_record_s_reading_for_its_turn_and_puts_it_back(monkeypatch):
    from core.self.how_i_answer import ARMS, reading_as_in

    assert ARMS == {"with": "", "without": "off", "shuffled": "shuffled"}
    monkeypatch.setenv("AURA_SELF_OBSERVATION", "before")
    for arm, value in ARMS.items():
        with reading_as_in(arm):
            assert os.environ["AURA_SELF_OBSERVATION"] == value
        assert os.environ["AURA_SELF_OBSERVATION"] == "before"
