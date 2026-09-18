"""The forecast is graded against the delivered actions and pre-action belief."""

from collections import Counter
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, Mock

import pytest

from core.agency.how_far_her_model_carries import HowFarHerModelCarries
from core.cognition.how_far_to_go_before_looking import HowFarToGo
from core.perception.what_is_there import Arrangement, Cell
from core.skills import screen_pursuit_decision as decision
from core.skills.screen_pursuit_acting import _bind_delivered_forecast


def board(value, column=0):
    return Arrangement(2, 2, (Cell(0, column, str(value), (0.0, 0.0)),))


def grade(expected, after, carries, before=None):
    previous = NS(chosen=NS(name="left"))
    pending = {"arranged": before or board(1), "watched": {}, "ahead": {str(i): i for i in range(7)}}
    decision._decide_the_next_move_world_where_she(
        expected, HowFarToGo(), Mock(), after, {}, pending, previous,
        {"moving": NS(what_measures_doing_well=lambda: None)}, carries,
    )
    return pending, previous


@pytest.mark.parametrize("arrived", [1, 2, 3])
def test_partial_delivery_selects_precomputed_prefix(arrived):
    prefixes = [board(2), board(3), board(4)]
    expected = {"took": 3, "after": prefixes[-1], "prefixes": prefixes, "confidence": 0.9}
    _bind_delivered_forecast(expected, arrived)
    assert expected["after"] is prefixes[arrived - 1]
    carried = HowFarHerModelCarries()
    grade(expected, prefixes[arrived - 1], carried)
    assert carried.graded == {arrived: 1}
    assert carried.carrying.useful_horizon()["by_horizon"][0]["error"] == 0.0
    assert expected["took"] == arrived
    assert carried.calibration.n == int(arrived == 1)
    if arrived == 1:
        assert carried.calibration.confidence_sum == 0.9
    grade(expected, prefixes[arrived - 1], carried)
    assert carried.graded == {arrived: 1}, "a forecast is consumed once"


def test_grading_does_not_erase_multi_action_learning_evidence(monkeypatch):
    expected = {"took": 2, "after": board(3), "confidence": 0.9}
    carried = HowFarHerModelCarries()
    pending, previous = grade(expected, board(3), carried)
    knows = Mock()
    dropped = Counter()
    monkeypatch.setattr(decision, "_left_her_better_off", lambda *args: True)
    decision._decide_the_next_move_part_10(
        dropped, expected, knows, board(3), pending, {"held": None}, previous, {}, "",
    )
    knows.watched.assert_not_called()
    assert dropped["more than one act, one reading"] == 1


def test_single_action_still_reaches_the_learner(monkeypatch):
    expected = {"took": 1, "after": board(2), "confidence": 0.9}
    pending, previous = grade(expected, board(2), HowFarHerModelCarries())
    knows = Mock()
    monkeypatch.setattr(decision, "_left_her_better_off", lambda *args: True)
    monkeypatch.setattr(decision, "_in_the_same_grid", lambda *args: True)
    decision._decide_the_next_move_part_10(
        Counter(), expected, knows, board(2), pending, {"held": None}, previous, {"lattice": None}, "",
    )
    knows.watched.assert_called_once()


@pytest.mark.parametrize("after", [None])
def test_missing_observation_is_not_a_model_failure(after):
    expected = {"took": 1, "after": board(2), "confidence": 0.9}
    carried = HowFarHerModelCarries()
    grade(expected, after, carried)
    assert carried.graded == {}


def test_failed_delivery_cannot_be_graded_as_model_error():
    expected = {"took": 2, "after": board(3), "prefixes": [board(2), board(3)], "confidence": 0.9}
    _bind_delivered_forecast(expected, 0)
    carried = HowFarHerModelCarries()
    grade(expected, board(1), carried)
    assert carried.graded == {}
    assert expected["took"] == 0
    assert expected["after"] is None


def test_unknown_prefix_is_unmeasured_not_the_full_plan_prediction():
    expected = {"took": 3, "after": board(4)}
    _bind_delivered_forecast(expected, 2)
    assert expected == {"took": 2, "after": None}


@pytest.mark.parametrize("arrived", [-1, 4, 1.5, True])
def test_invalid_delivery_receipts_do_not_mutate_the_forecast(arrived):
    expected = {"took": 3, "after": board(4)}
    before = expected.copy()
    with pytest.raises(ValueError):
        _bind_delivered_forecast(expected, arrived)
    assert expected == before


def test_prediction_and_no_change_use_the_same_correctness_contract():
    before = board(1)
    after = Arrangement(2, 2, before.cells + board(2, column=1).cells)
    expected = {"took": 1, "after": before, "confidence": 0.8}
    carried = HowFarHerModelCarries()
    grade(expected, after, carried, before=before)
    row = carried.carrying.useful_horizon()["by_horizon"][0]
    assert row["error"] == row["baseline"] == 0.0
    assert row["beats_baseline"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize("arrived", [0, 1, 2, 3])
async def test_action_helper_binds_the_forecast_before_waiting_for_the_world(monkeypatch, arrived):
    from core.perception import where_am_i
    from core.skills import screen_pursuit_acting as acting

    prefixes = [board(2), board(3), board(4)]
    expected = {"took": 3, "after": prefixes[-1], "prefixes": prefixes, "confidence": 0.9}
    pending = {"watched": {}, "deliberation": object()}
    run = NS(
        about_to={"key": "left"}, anchor={"app": "test"}, at_rest={}, busy=Mock(),
        expected=expected, follow_on=["up", "right"], goal="test", in_flight=Mock(),
        key="left", laid_out=board(1), lattice=Mock(), made=None, moves=[], narrate=False,
        pacing={"brief": True, "choice": ""}, pending=pending,
        responds={"state": None, "lattice": None}, target_app="test", world=NS(acts_with_arrivals=0),
    )
    monkeypatch.setattr(where_am_i, "where_am_i", lambda *args, **kwargs: NS(the_thing_is_here=True))
    monkeypatch.setattr(acting, "press_many", AsyncMock(return_value=arrived))
    monkeypatch.setattr(acting, "_say_intent", Mock())
    monkeypatch.setattr(acting, "_say_it_did_not_land", Mock())
    seen = []

    def matches(forecast, *args):
        seen.append(forecast)
        return None

    monkeypatch.setattr(acting, "_looks_like", matches)
    monkeypatch.setattr(acting, "_settled_after", AsyncMock(return_value=({}, None)))
    monkeypatch.setattr(acting, "_how_long_to_wait", lambda: 0.1)
    monkeypatch.setattr("core.agency.what_she_is_doing.a_step_taken", Mock())
    result = await acting.carry_out_the_move(run)
    assert result is (arrived > 0)
    assert len(run.moves) == arrived
    assert expected["took"] == arrived
    assert seen == ([prefixes[arrived - 1]] if arrived else [])
    if arrived == 0:
        assert pending["deliberation"] is None
        run.in_flight.she_started.assert_not_called()
    else:
        assert run.in_flight.she_started.call_args.kwargs["brings"] == str(arrived)
