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
