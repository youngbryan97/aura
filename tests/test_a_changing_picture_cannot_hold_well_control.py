"""Changing scenery cannot keep a candidate well in charge or manufacture results."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from core.agency import playing_a_well as playing
from core.perception.a_well_seen import WellSeen, _well_geometry_invariant, the_piece_in


def test_provisional_motion_has_no_piece_or_clear_authority():
    assert playing._provisional_well_invariant() == ()


@pytest.mark.asyncio
@pytest.mark.parametrize('branch', ['changing', 'missing', 'resting', 'no_piece'])
@pytest.mark.parametrize('recalled', [False, True])
async def test_every_provisional_continuation_spends_the_same_admission_budget(monkeypatch, branch, recalled):
    clock = [0.]
    well = WellSeen(0., 0., 8., 8., 8, 16, (30, 30, 30))

    async def look():
        clock[0] += .04
        return np.full((128, 64, 3), 30, np.uint8), clock[0]

    async def wait(seconds):
        clock[0] += seconds

    async def seen(game):
        if branch == 'missing':
            return None
        return {(0, 15): 1}, None, branch == 'changing'

    monkeypatch.setattr(playing, '_now', lambda: clock[0])
    monkeypatch.setattr(playing, '_wait', wait)
    monkeypatch.setattr(playing._Game, 'seen', seen)
    monkeypatch.setattr(playing, '_what_stayed_put_is_taken_in', lambda *args: branch == 'resting')
    monkeypatch.setattr('core.perception.a_well_seen.a_well_in', lambda *args: well)
    kept = {'well_clears': playing.WhatClears(), 'well_looks': []}
    if recalled:
        kept['well_keys'] = {'left': 'a', 'right': 'd'}
    before = dict(kept)
    result = await playing.play_a_well(look, SimpleNamespace(), well, seconds=60., known=kept)
    assert result['ended'] == playing.NOTHING_MOVES_IT
    assert result['controls_confirmed'] is False
    assert result['pieces'] == result['cleared'] == 0
    assert clock[0] <= playing.ACROSS_FOUND_WITHIN_S+.2
    assert kept == before  # Prior hypotheses survive; this run supplies no new learned result.
    assert 'well_control_verified' not in kept


def test_a_provisional_attempt_cannot_mutate_verified_learning_through_an_alias():
    prior = playing.WhatClears()
    prior.foretold['full rows'] = 4
    looks = [np.ones((6, 6, 3))]
    kept = dict(well_clears=prior, well_looks=looks, well_said={'one'}, well_control_verified=True)
    game = playing._Game(WellSeen(0., 0., 8., 8., 8, 16, (30, 30, 30)), None, None, lambda line: None, ())
    playing._recalled(game, kept)
    game.clears.foretold['full rows'] = 99
    game.well.looks[0][:] = 0
    game.said.add('two')
    assert prior.foretold['full rows'] == 4 and looks[0].all() and kept['well_said'] == {'one'}


@pytest.mark.parametrize('field,value', [(0, float('nan')), (1, float('inf')), (2, 0.), (3, -1.),
                                         (2, 1e308), (4, 8.5), (5, 5.), (6, 256.), (7, True)])
def test_bad_saved_geometry_cannot_enter_the_visual_grid(field, value):
    kept = WellSeen(4., 4., 8., 8., 8, 16, (30, 30, 30)).kept()
    kept[field] = value
    assert WellSeen.from_kept(kept) is None
    assert _well_geometry_invariant() == ()


@pytest.mark.parametrize('left,top', [(80., 80.), (-80., 0.), (0., -80.), (40., 0.), (0., 40.)])
def test_an_offscreen_candidate_cannot_be_treated_as_a_measured_piece(left, top):
    well = WellSeen(left, top, 8., 8., 8, 16, (30, 30, 30))
    assert the_piece_in(np.full((32, 32, 3), 30, np.uint8), well, {}) is None


@pytest.mark.asyncio
async def test_a_different_piece_cannot_supply_a_horizontal_control_response(monkeypatch):
    from core.perception.a_well_seen import PieceSeen

    game = playing._Game(SimpleNamespace(rows=16), None, None, lambda line: None, ('left', 'right'))
    game.stack = {}
    before = PieceSeen(((0, 0, 1),), 3, 4., True)
    after = PieceSeen(((0, 0, 2),), 2, 4., True)

    async def press(*args):
        return None

    async def steady(*args):
        return {}, after, False

    monkeypatch.setattr(playing._Game, 'press', press)
    monkeypatch.setattr(playing._Game, 'steady', steady)
    await playing._find_out(game, 'left', before)
    assert not game.current_controls and not game.keys.left


@pytest.mark.asyncio
async def test_the_running_actor_cannot_count_an_unconfirmed_specialist_result(monkeypatch):
    from core.skills import screen_pursuit_as_it_happens as pursuit

    well = WellSeen(0., 0., 8., 8., 8, 16, (30, 30, 30))

    async def look():
        return np.zeros((128, 64, 3), np.uint8), 100.

    async def candidate(*args):
        return well

    async def unconfirmed(*args, **kwargs):
        return dict(ended='out of time', controls_confirmed=False, pieces=99, cleared=999)

    reflexes = SimpleNamespace(keep={}, look=look, words=[], said_now='', ends_at=1e9,
                               no_well_here=False, waited_at_a_well=0., stretches=[])
    monkeypatch.setattr(playing, 'a_well_with_a_piece_coming_down', candidate)
    monkeypatch.setattr(playing, 'play_a_well', unconfirmed)
    assert not await pursuit._played_a_well(reflexes, 100.)
    assert not reflexes.stretches and not getattr(reflexes, 'play_stopped', False)
