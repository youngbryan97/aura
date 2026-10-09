"""An action is ranked by what it is for, which the world proposing it says, not by what one world calls it.

The ranker gave credit to "use_stairs", "eat" and "pray" by name: what helped a
dungeon game helped nothing else.
"""
from __future__ import annotations

import inspect

import pytest

from core.environment.command import ActionIntent
from core.environment.policy import action_ranker
from core.environment.policy.purposes import ADVANCE, EVADE, IDLE, RESTORE, purpose_of

pytestmark = pytest.mark.unit


def test_a_purpose_comes_from_the_actions_tag_else_from_its_generic_name():
    assert purpose_of(ActionIntent(name="board_the_train", tags={ADVANCE})) == ADVANCE
    assert purpose_of(ActionIntent(name="drink_from_the_well", tags={RESTORE})) == RESTORE
    assert purpose_of(ActionIntent(name="duck", tags={"threat_response"})) == EVADE
    assert purpose_of(ActionIntent(name="move")) == ADVANCE
    assert purpose_of(ActionIntent(name="wait")) == IDLE
    assert purpose_of(ActionIntent(name="use_stairs")) == ""


def test_the_ranker_names_no_worlds_actions():
    source = inspect.getsource(action_ranker)
    for name in ("use_stairs", '"eat"', '"pray"'):
        assert name not in source
