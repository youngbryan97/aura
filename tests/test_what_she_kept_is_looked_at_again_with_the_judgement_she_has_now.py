"""The positions she keeps for learning how the pieces stand together are looked at again with the judgement she has
now (core/agency/looking_again_at_what_she_kept.py): every first look before each fit, and the oldest longer looks a
round at a time.

LIVE 27 November 2026 every kept position's first look was the one taken when it was gathered, with weights her
practice had since changed, and how the pieces stand together was fitted to a gap she no longer had.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

import core.agency.what_a_kind_is_worth as worth
from core.agency.looking_again_at_what_she_kept import a_position_from_its_parts, first_looks_now, looked_at_again
from core.agency.searching_by_kind import _compiled, described, first_weights, placed
from tests.fixtures.a_game_of_pieces_to_learn import CHESS, NOBODYS_GAME

pytestmark = pytest.mark.unit

engine = _compiled()
needs_the_compiled_search = pytest.mark.skipif(
    engine is None or not hasattr(engine, "seen_as_practice_sees"), reason="the compiled search is not built")


def _kept(family, weights, games: int = 2) -> worth.LookedAtLonger:
    played = engine.self_play(described(family), placed(family.the_start()), weights.as_json(), weights.as_json(),
                              games, 300, random_plies=4, seed=7, max_plies=60, sample_every=1,
                              deep_nodes=1000, deep_quiet_only=True)
    store = worth.LookedAtLonger()
    store.add(played, 3, family.rows, family.columns)
    return store


@needs_the_compiled_search
@pytest.mark.parametrize("family", [CHESS, NOBODYS_GAME], ids=["chess", "nobody's game"])
def test_a_kept_position_set_out_again_is_judged_as_it_was_when_kept(family) -> None:
    weights = first_weights(family)
    store = _kept(family, weights)
    assert store.parts
    again = first_looks_now(store, family, weights, dict(family.start))
    assert again is not None and np.allclose(again, store.glance)
    position = a_position_from_its_parts(store.parts[0], store.kinds, store.squares, store.columns, dict(family.start))
    assert len(position.cells) == len(store.parts[0])


@needs_the_compiled_search
def test_with_weights_changed_the_first_looks_are_hers_now() -> None:
    weights = first_weights(CHESS)
    store = _kept(CHESS, weights)
    changed = replace(weights, tempo=weights.tempo + 40.0)
    again = first_looks_now(store, CHESS, changed, dict(CHESS.start))
    assert again is not None and np.allclose(np.array(again) - np.array(store.glance), 40.0)


@needs_the_compiled_search
def test_the_oldest_longer_looks_are_taken_again_first_and_marked() -> None:
    weights = first_weights(CHESS)
    store = _kept(CHESS, weights)
    store.labelled = [1] * len(store.parts)
    store.labelled[-1] = 0
    looked = looked_at_again(store, CHESS, weights, dict(CHESS.start), now=9, seconds=0.0, nodes=500)
    assert looked == 1 and store.labelled[-1] == 9 and store.labelled[0] == 1


def test_a_position_looked_at_again_is_kept_once_with_what_the_later_look_found() -> None:
    parts = [np.array([1, 70], dtype=np.int16), np.array([2, 71], dtype=np.int16)]
    kept = worth.LookedAtLonger(kinds=["a", "b"], squares=64, columns=8, parts=list(parts), glance=[0.0, 0.0],
                                slow=[10.0, 20.0], game=[30_001, 30_002], scales=[200.0], labelled=[3, 3])
    mine = worth.LookedAtLonger.from_bytes(kept.as_bytes())
    mine.slow[0], mine.labelled[0] = 55.0, 9
    saved = mine.with_what_was_kept(kept)
    again = saved.with_what_was_kept(worth.LookedAtLonger.from_bytes(saved.as_bytes()))
    assert len(saved.parts) == len(again.parts) == 2
    assert again.slow == [55.0, 20.0] and again.rounds_labelled() == [9, 3]
