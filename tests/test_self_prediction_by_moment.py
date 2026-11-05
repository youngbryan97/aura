"""Self-prediction against her moment's own answer, and by stated probability (preregistered 4 November, late)."""

from __future__ import annotations

from core.subject import self_prediction_by_moment as m
from core.subject.self_prediction import ITEMS_V2


def _item(item_id: str):
    return next(i for i in ITEMS_V2 if i.id == item_id)


def test_every_prediction_is_asked_with_how_sure_she_is():
    for item in ITEMS_V2:
        prompt = m.stated_prompt(item)
        assert item.object_prompt in prompt
        assert "from 0 to 100" in prompt


def test_a_stated_prediction_and_its_probability_are_read_apart():
    assert m.read_stated(_item("rain_snow"), "Rain, 80") == ("rain", 0.8)
    assert m.read_stated(_item("say_long"), "Yes. I'm about 90% sure.") == ("yes", 0.9)
    assert m.read_stated(_item("number_5"), "3, and I'd put it at 70") == ("3", 0.7)
    assert m.read_stated(_item("number_100"), "42, 65") == ("42", 0.65)
    # A prediction with no number after it has no stated probability.
    assert m.read_stated(_item("rain_snow"), "Snow.") == ("snow", None)


def test_the_moment_s_own_answer_does_not_depend_on_the_order_of_draws():
    assert m.modal(["tea", "coffee", "tea", None]) == "tea"
    assert m.modal(["tea", "coffee"]) == m.modal(["coffee", "tea"]) == "coffee"
    assert m.modal([None, None]) is None


def _rows(moments):
    """One choice question; each moment is (draws, her stated prediction)."""
    return [{"anchor": a, "item": "rain_snow", "answers": [[d, True] for d in draws], "prediction_with": said,
             "prediction_with_served": True} for a, (draws, said) in enumerate(moments)]


def test_a_self_model_that_knows_its_moment_beats_the_habit():
    # Her habit is rain; two moments lean snow, and she says so with the right confidence.
    moments = [
        (["rain"] * 4 + ["snow"], "rain, 80"),
        (["rain"] * 4 + ["snow"], "rain, 80"),
        (["snow"] * 4 + ["rain"], "snow, 80"),
        (["rain"] * 4 + ["snow"], "rain, 80"),
        (["snow"] * 4 + ["rain"], "snow, 80"),
        (["rain"] * 4 + ["snow"], "rain, 80"),
    ]
    reading = m.by_moment_verdict(m.by_moment_records(_rows(moments), ITEMS_V2, "with"), draws=2000)
    assert reading["accuracy_against_the_moment"] == 1.0
    assert reading["habit_accuracy_against_the_moment"] < 1.0
    assert reading["brier"] < reading["habit_brier"]


def test_a_self_model_that_only_repeats_the_habit_does_not():
    moments = [(["rain"] * 4 + ["snow"], "rain, 80")] * 3 + [(["snow"] * 4 + ["rain"], "rain, 80")] * 2
    reading = m.by_moment_verdict(m.by_moment_records(_rows(moments), ITEMS_V2, "with"), draws=2000)
    assert reading["accuracy_against_the_moment"] == reading["habit_accuracy_against_the_moment"]
    assert not reading["beats_habit"]


def test_her_self_model_states_how_sure_its_record_is():
    from core.self import how_i_answer as hia
    from core.state.aura_state import AuraState

    state = AuraState.default()
    for i in range(10):
        hia.note_reply(state, "Tell me what you have been thinking about today.",
                       " ".join(["Today"] + ["word"] * (60 if i < 8 else 10)), at=float(i))
    prompt = m.stated_prompt(_item("say_long"))
    said = hia.self_model_prediction(state, prompt, "long", stated=True)
    # 8 of 10 replies ran long; the rule of succession makes that (8 + 1) / (10 + 2).
    assert said == "yes, 75"
    assert m.read_stated(_item("say_long"), said) == ("yes", 0.75)


def test_her_self_model_is_never_certain_from_a_few_replies():
    from core.self import how_i_answer as hia

    assert hia._how_sure(5, 5, two_valued=True) == round(6 / 7, 4)
    assert hia._how_sure(5, 5, two_valued=False) == round(5 / 6, 4)
    assert hia._how_sure(0, 0, two_valued=True) == 0.0
