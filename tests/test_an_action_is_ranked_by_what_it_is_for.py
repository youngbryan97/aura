"""An action is ranked by what it is for, which the world proposing it says, not by what one world calls it.

The ranker gave credit to "use_stairs", "eat" and "pray" by name: what helped a
dungeon game helped nothing else.
"""
from __future__ import annotations

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


@pytest.mark.parametrize("critical", [[], ["food"]])
def test_the_ranker_names_no_worlds_actions(critical):
    """Two worlds that call the same actions different things get the same scores, so no name of one is rewarded."""
    from core.environment.homeostasis import HomeostaticAssessment
    from core.environment.simulation import SimulationBundle, SimulationHypothesis

    def scores(named: list[tuple[str, set[str]]]) -> list[float]:
        intents = [ActionIntent(name=name, tags=tags) for name, tags in named]
        simulations = {}
        for n, intent in enumerate(intents):
            guess = SimulationHypothesis(predicted_events=[], predicted_resource_delta={"food": 0.1 * n},
                                         predicted_context_delta={}, information_gain=0.2, risk_delta=0.1 * n,
                                         confidence=0.8)
            simulations[intent.intent_id()] = SimulationBundle(intent, [guess], 0.1 * n, 0.0, 0.2)
        needs = HomeostaticAssessment(stability_score=0.5, critical_resources=list(critical),
                                      deteriorating_resources=[], recommended_goal=None)
        ranker = action_ranker.ActionRanker()
        return [ranker.score_candidate(i, simulations[i.intent_id()], needs, None) for i in intents]

    dungeon = [("use_stairs", {ADVANCE}), ("eat", {RESTORE}), ("pray", set()), ("duck", {"threat_response"})]
    elsewhere = [("board_the_train", {ADVANCE}), ("drink_from_the_well", {RESTORE}), ("hum", set()),
                 ("crouch", {"threat_response"})]
    assert scores(dungeon) == scores(elsewhere)
