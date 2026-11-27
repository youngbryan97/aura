"""She plays another player at the setting of its strength where her results vary, and what her practice keeps answers
to her games there (core/agency/where_her_results_vary.py).

LIVE 26-27 November 2026 everything built from her own search preferred weights that a stronger outside player beat
more easily, and that player at full strength beat her every game, so her results against it taught her nothing.
"""
from __future__ import annotations

import asyncio
import math
import random

import pytest

from core.agency.where_her_results_vary import GAMES_A_TRIAL, KeepOnTrial, Ladder

pytestmark = pytest.mark.unit


def test_with_no_games_she_begins_between_the_weakest_and_the_strongest_or_at_the_weakest() -> None:
    assert Ladder().next_setting(1.0, 64.0) == pytest.approx(8.0)
    assert Ladder().next_setting(1.0) == 1.0


def test_after_games_all_lost_she_halves_it_and_after_games_all_won_she_doubles_it() -> None:
    lost, won = Ladder(), Ladder()
    for _ in range(3):
        lost.came(16.0, 0.0)
        won.came(4.0, 1.0)
    assert lost.next_setting(1.0, 64.0) == 8.0
    assert won.next_setting(1.0, 64.0) == 8.0
    assert Ladder(results={1.0: [0.0, 0.0]}).next_setting(1.0, 64.0) == 1.0


def test_where_her_games_went_both_ways_she_plays_where_she_would_score_even() -> None:
    rng = random.Random(3)
    ladder = Ladder()
    even_at = 10.0
    for setting in (2.0, 4.0, 8.0, 16.0, 32.0):
        for _ in range(40):
            chance = 1.0 / (1.0 + math.exp(2.0 * (math.log(setting) - math.log(even_at))))
            ladder.came(setting, 1.0 if rng.random() < chance else 0.0)
    assert ladder.next_setting(1.0, 64.0) == pytest.approx(even_at, rel=0.25)


def test_a_ladder_comes_back_as_it_was_kept() -> None:
    ladder = Ladder(results={8.0: [1.0, 0.5], 16.0: [0.0]})
    assert Ladder.from_dict(ladder.as_dict()) == ladder


def test_a_keep_that_cost_her_much_is_put_back_and_one_that_did_not_is_kept() -> None:
    rng = random.Random(5)
    before = [rng.choice((0.0, 0.5, 1.0)) for _ in range(GAMES_A_TRIAL)]
    worse = KeepOnTrial(setting=8.0, before=list(before), weights_before={"worth": {"a": 1.0}})
    said = None
    for _ in range(GAMES_A_TRIAL):
        said = worse.came(0.0)
        if said:
            break
    assert said == "put back"
    alike = KeepOnTrial(setting=8.0, before=list(before), weights_before={})
    said = None
    for points in before:
        said = alike.came(points)
        if said:
            break
    assert said == "kept"
    assert KeepOnTrial.from_dict(worse.as_dict()) == worse


def _a_rehearsal():
    from core.agency.searching_by_kind import first_weights
    from core.agency.what_a_kind_is_worth import Rehearsal
    from tests.fixtures.a_game_of_pieces_to_learn import NOBODYS_GAME

    return Rehearsal(weights=first_weights(NOBODYS_GAME))


def test_her_practice_games_go_where_her_results_vary_and_a_trial_holds_them_at_its_setting() -> None:
    from core.agency.where_her_results_vary import came_from_a_game, open_a_trial, the_setting_to_play

    rehearsal = _a_rehearsal()
    assert the_setting_to_play(rehearsal) == 1
    came_from_a_game(rehearsal, 1, 1.0)
    assert the_setting_to_play(rehearsal) == 2
    assert not open_a_trial(rehearsal, rehearsal.weights)
    for points in (1.0, 0.0, 0.5, 0.0, 1.0):
        came_from_a_game(rehearsal, 4, points)
    assert open_a_trial(rehearsal, rehearsal.weights)
    assert the_setting_to_play(rehearsal) == 4
    assert not open_a_trial(rehearsal, rehearsal.weights)


def test_a_keep_whose_games_go_worse_is_put_back() -> None:
    from dataclasses import replace

    from core.agency.where_her_results_vary import came_from_a_game, open_a_trial

    rehearsal = _a_rehearsal()
    before = rehearsal.weights
    for points in [1.0, 0.0, 0.5, 1.0] * 10:
        came_from_a_game(rehearsal, 6, points)
    assert open_a_trial(rehearsal, before)
    rehearsal.weights = replace(before, tempo=before.tempo + 99.0)
    said = None
    for _ in range(GAMES_A_TRIAL):
        said = came_from_a_game(rehearsal, 6, 0.0)
        if said:
            break
    assert said == "put back"
    assert rehearsal.weights == before and rehearsal.on_trial is None


def test_a_practice_game_is_due_only_when_nobody_is_at_the_machine_and_never_under_a_test(monkeypatch) -> None:
    import core.agency.practising_between_games as practice
    import core.capabilities.window_server as window_server
    import core.runtime.proof_policy as proof_policy

    rehearsal = _a_rehearsal()
    rehearsal.rounds = 3
    monkeypatch.setattr(practice, "_held", lambda name, written: {"app": "A Board App"})
    monkeypatch.setattr(practice, "_HELD", {"kept": {"app": "A Board App"}, "rehearsal": rehearsal})
    monkeypatch.setattr(practice, "_GAME_AFTER_ROUND", [-1])
    monkeypatch.setattr(window_server, "seconds_since_someone_touched_it", lambda: 10_000.0)
    monkeypatch.setattr(proof_policy, "proof_run_active", lambda origin=None: True)
    assert practice._a_practice_game_due(("a-board", (1, 1))) is None
    monkeypatch.setattr(proof_policy, "proof_run_active", lambda origin=None: False)
    assert practice._a_practice_game_due(("a-board", (1, 1))) == ("a-board", "A Board App", 1)
    monkeypatch.setattr(window_server, "seconds_since_someone_touched_it", lambda: 30.0)
    assert practice._a_practice_game_due(("a-board", (1, 1))) is None


def test_a_practice_game_s_points_are_kept_at_the_setting_it_was_played_at(monkeypatch) -> None:
    import core.agency.practising_between_games as practice
    import core.skills.screen_pursuit as pursuit
    import core.skills.screen_pursuit_board_of_pieces as board

    rehearsal = _a_rehearsal()
    asked = []

    async def pursue(**options):
        from core.skills.screen_pursuit_a_setting import FOR_HER_PRACTICE

        # What a practice game changes is marked to be put back.
        assert FOR_HER_PRACTICE.get()
        asked.append(options["goal"])
        return {"handed_over": {"ended": "won", "settings_set": {"the other player's strength": "4"}}}

    monkeypatch.setattr(pursuit, "pursue_on_screen", pursue)
    monkeypatch.setattr(board, "_keep_what_she_learned", lambda *args, **kwargs: None)
    monkeypatch.setattr(practice, "_written_now", lambda name: (1, 1))
    monkeypatch.setattr(practice, "_held", lambda name, written: {"app": "A Board App"})
    monkeypatch.setattr(practice, "_the_board_she_last_played", lambda: ("a-board", (1, 2)))
    monkeypatch.setattr(practice, "_HELD", {"kept": {"app": "A Board App", "looks": None, "rules": None},
                                            "rehearsal": rehearsal})
    came = asyncio.run(practice._a_practice_game("a-board", "A Board App", 4))
    assert asked == ["Play a game against the other player, with the other player's strength set to 4."]
    assert came == {"practice game": 4.0, "points": 1.0, "trial": None}
    assert rehearsal.results_vary == {"4.0": [1.0]}
