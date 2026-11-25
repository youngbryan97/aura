"""A move left alone by the moves set aside is searched as deep as any (core/agency/searching_by_kind.py): set aside to
learn what one move is worth, it is not answered by where it stands. LIVE 24 November 2026 two takes worked out two
moves deep on their time were held against her look by their standing before the answer that takes back, and kept: a
bishop for a man, then a queen for a bishop."""
from __future__ import annotations

from dataclasses import replace

from core.agency.pieces_by_their_kind import moves
from core.agency.playing_pieces_by_kind import _checked
from core.agency.searching_by_kind import Found, first_weights, search
from tests.fixtures.a_game_of_pieces_to_learn import CHESS, from_fen

# Her bishop on f2 can take the man on d4, which the man on c5 takes back.
TAKEN_BACK = from_fen("4k3/8/8/2p5/3p4/8/5B2/4K3 w - -")
# A bishop worth three men: what the first weights hold each kind worth puts a bishop near one man.
WORTH = replace(first_weights(CHESS), worth={"P": 100.0, "N": 300.0, "B": 300.0, "R": 500.0, "Q": 900.0, "K": 0.0})


def _the_take(legal):
    return next(m for m in legal if (m.start, m.end) == ((1, 5), (3, 3)))


def test_a_take_left_alone_is_searched_past_the_answer_that_takes_back() -> None:
    legal = moves(CHESS, TAKEN_BACK)
    take = _the_take(legal)
    best = search(CHESS, TAKEN_BACK, seconds=0.5, nodes=20_000, weights=WORTH)
    alone = search(CHESS, TAKEN_BACK, seconds=0.5, nodes=20_000, weights=WORTH, set_aside=[m for m in legal if m != take])
    assert alone.move == take and alone.depth > 1
    assert alone.score < best.score, (alone.score, best.score)


def test_a_hint_that_loses_a_piece_to_the_answer_is_not_kept_against_the_look() -> None:
    legal = moves(CHESS, TAKEN_BACK)
    take = _the_take(legal)
    now = search(CHESS, TAKEN_BACK, seconds=0.5, nodes=20_000, weights=WORTH)
    assert now.move != take
    hint = Found(take, now.score, 2, 0, [take], by="worked out on their time, to check")
    held = _checked(hint, now, legal, lambda aside: search(CHESS, TAKEN_BACK, seconds=0.5, nodes=20_000,
                                                           weights=WORTH, set_aside=aside))
    assert held.move == now.move and held.by == "worked out on their time, and a look now found better"
