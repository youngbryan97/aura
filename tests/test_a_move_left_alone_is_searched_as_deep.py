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


def test_unmatched_search_depth_does_not_certify_an_alternative() -> None:
    legal = moves(CHESS, TAKEN_BACK)
    primary, alternate = legal[:2]
    main = Found(primary, 80, 12, 1000, [primary], unsteady=30)
    hint = Found(alternate, 140, 2, 0, [alternate], by="worked out ahead, to check")
    shallow = Found(alternate, 140, 8, 100, [alternate])

    held = _checked(hint, main, legal, lambda aside: shallow)
    assert held.move == primary and held.by.endswith("not checked at the same depth")

    deeper = Found(alternate, 120, 14, 2000, [alternate],
                   accounting={"iterations": [(12, 20, False, 13), (14, 120, False, 15)]})
    held = _checked(hint, main, legal, lambda aside: deeper)
    assert held.move == primary
    deeper.accounting["iterations"][0] = (12, 80, False, 13)
    assert _checked(hint, main, legal, lambda aside: deeper).move == alternate


def test_a_restricted_search_reports_depths_and_scores_from_the_original_position(monkeypatch) -> None:
    from core.agency import searching_by_kind

    legal = moves(CHESS, TAKEN_BACK)
    take = _the_take(legal)
    answer = Found(None, -78, 5, 10, [], accounting={"iterations": [(4, -16, False, 3)], "seldepth": 8})
    monkeypatch.setattr(searching_by_kind, "search", lambda *args, **kwargs: answer)

    result = searching_by_kind._the_one_left(CHESS, TAKEN_BACK, take, seconds=0.1)
    assert result.depth == 6 and result.score_at_depth(6) == 78
    assert result.score_at_depth(5) == 16
    assert result.score_at_depth(4) is None
    assert result.accounting["seldepth"] == 9
    assert answer.accounting["iterations"][0] == (4, -16, False, 3)
