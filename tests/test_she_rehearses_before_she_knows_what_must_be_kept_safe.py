"""She rehearses a game from her first turn, before she knows which piece must be kept safe.

LIVE 20-21 November 2026 she played two games against Stockfish and rehearsed in neither: a rehearsal waited for a
piece she had learned must be kept safe, which she learns only when the game withholds a move that would leave it
attacked. Until then every game in her model could only stop level, and there was nothing to learn from. A game that
stops without a result is now judged by how it stood in material when it stopped.
"""
from __future__ import annotations

from dataclasses import replace

import core.agency.playing_pieces_by_kind as playing
from core.agency.what_a_kind_is_worth import a_first_rehearsal, rehearse
from tests.fixtures.a_game_of_pieces_to_learn import CHESS

#: The test world's game, with no piece known to be the one that must be kept safe: as she holds it early on.
NOTHING_KNOWN_ROYAL = replace(CHESS, kinds={letter: replace(kind, royal=False) for letter, kind in CHESS.kinds.items()})


def test_games_that_stop_without_a_result_still_teach_her_something() -> None:
    rehearsal = rehearse(NOTHING_KNOWN_ROYAL, NOTHING_KNOWN_ROYAL.the_start(), a_first_rehearsal(NOTHING_KNOWN_ROYAL),
                         games=8, nodes=300, seed=2)
    assert rehearsal.rounds == 1
    assert rehearsal.samples_y, "no positions were gathered"
    judged = [y for y in rehearsal.samples_y if y not in (0.0, 0.5, 1.0)]
    assert judged, "every position was labelled level: nothing to fit"
    assert all(0.0 < y < 1.0 for y in judged)


def test_her_rehearsal_starts_before_she_knows_what_must_be_kept_safe() -> None:
    submitted: list = []

    class _Pool:
        def submit(self, *args, **kwargs):
            submitted.append(args)
            return None

    rehearsing = playing._Rehearsing()
    rehearsing.start(_Pool(), NOTHING_KNOWN_ROYAL)
    assert submitted, "no round was started for a game whose royal piece she does not know yet"


def test_only_games_that_ended_with_a_result_say_how_long_a_game_goes() -> None:
    """LIVE 21 November 2026 rehearsed games that stopped on a repetition after a dozen plies told her two or three moves
    were left, and she gave each move a third of her clock."""
    rehearsal = rehearse(NOTHING_KNOWN_ROYAL, NOTHING_KNOWN_ROYAL.the_start(), a_first_rehearsal(NOTHING_KNOWN_ROYAL),
                         games=8, nodes=300, seed=2)
    # With no piece known to be the one kept safe, no game in her model can be won or lost.
    assert rehearsal.lengths == []


def test_she_never_expects_fewer_moves_to_come_than_she_has_made() -> None:
    from core.agency.what_a_kind_is_worth import Rehearsal
    from core.agency.searching_by_kind import first_weights

    rehearsing = playing._Rehearsing(Rehearsal(weights=first_weights(CHESS), lengths=[12] * 9))
    assert rehearsing.moves_left(0) == 6
    assert rehearsing.moves_left(10) == 10


def test_a_fresh_rehearsal_keeps_how_long_its_games_go() -> None:
    from core.agency.what_a_kind_is_worth import Rehearsal
    from core.agency.searching_by_kind import first_weights

    class _Pool:
        def submit(self, *args, **kwargs):
            return None

    old = Rehearsal(weights=first_weights(CHESS), lengths=[50, 54, 60])
    old.weights.worth["gone"] = 1.0  # a kind she no longer reads: the rehearsal starts fresh
    rehearsing = playing._Rehearsing(old)
    rehearsing.start(_Pool(), NOTHING_KNOWN_ROYAL)
    assert rehearsing.rehearsal is not old and rehearsing.rehearsal.lengths == [50, 54, 60]


def test_weights_fitted_in_another_game_are_measured_again() -> None:
    """LIVE 21 November 2026 weights fitted while she held the pawns' kind as the one kept safe valued her rooks at
    -16.7 and her pawns at -68.4, and she played every game after by them."""
    from core.agency.searching_by_kind import Weights, first_weights
    from core.agency.what_a_kind_is_worth import Rehearsal

    class _Pool:
        def submit(self, *args, **kwargs):
            return None

    royal_known = CHESS  # the test world's game, its royal kind known
    kept_then = Rehearsal(weights=Weights(worth={**first_weights(royal_known).worth, "R": -16.7, "P": -68.4},
                                          mobility=dict(first_weights(royal_known).mobility)),
                          earned=["worth"], lengths=[50, 54], royal=None)
    rehearsing = playing._Rehearsing(kept_then)
    rehearsing.start(_Pool(), royal_known)
    assert rehearsing.rehearsal is not kept_then and rehearsing.rehearsal.earned == []
    assert rehearsing.rehearsal.lengths == [50, 54]
    assert rehearsing.weights_for(royal_known).worth["R"] > 0


def test_weights_fitted_in_the_same_game_are_kept() -> None:
    from core.agency.what_a_kind_is_worth import a_first_rehearsal

    class _Pool:
        def submit(self, *args, **kwargs):
            return None

    same = a_first_rehearsal(CHESS)
    same.earned = ["worth"]
    rehearsing = playing._Rehearsing(same)
    rehearsing.start(_Pool(), CHESS)
    assert rehearsing.rehearsal is same
