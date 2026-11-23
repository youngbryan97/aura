"""While the other side thinks, she searches the position she expects to face, and remembers what they answered where.

core/agency/thinking_on_their_time.py. Bryan asked on 22 November 2026 for a model of the other side that carries
across games of any kind: here, what they answered in each position she has met, used to choose what she looks at on
their time.
"""
from __future__ import annotations

import asyncio

import core.agency.playing_pieces_by_kind as playing
from core.agency.how_the_pieces_move import HowThePiecesMove
from core.agency.pieces_by_their_kind import FIRST, Move, Position, after, moves, outcome
from core.agency.thinking_on_their_time import KEPT_AT_MOST, Pondering, WhatTheyAnsweredWhere
from tests.fixtures.a_board_application import BoardApplication
from tests.fixtures.a_game_of_pieces_to_learn import CHESS, NOBODYS_GAME


def test_what_they_answered_most_often_where_is_remembered_and_kept() -> None:
    memory = WhatTheyAnsweredWhere()
    for answer in ("e7-e5", "c7-c5", "e7-e5"):
        memory.they_answered("a place", answer)
    assert memory.most_often("a place") == "e7-e5" and memory.most_often("another place") is None
    again = WhatTheyAnsweredWhere.from_bytes(memory.as_bytes())
    assert again.most_often("a place") == "e7-e5"
    assert WhatTheyAnsweredWhere.from_bytes(b"not a memory").answered == {}


def test_the_positions_kept_are_the_latest_met() -> None:
    memory = WhatTheyAnsweredWhere()
    memory.they_answered("the first", "a")
    for n in range(KEPT_AT_MOST):
        memory.they_answered(f"place {n}", "b")
    assert memory.most_often("the first") is None and memory.most_often(f"place {KEPT_AT_MOST - 1}") == "b"


def test_a_look_on_their_time_lasts_about_as_long_as_they_take() -> None:
    pondering = Pondering()
    assert pondering.look_for([], 2.0) == 2.0
    assert pondering.look_for([0.4, 0.6, 0.5], 2.0) == 0.5
    assert pondering.look_for([9.0, 8.0], 2.0) == 2.0, "never longer than her own turn"


def test_she_expects_what_they_answered_before_in_the_same_position() -> None:
    learned = HowThePiecesMove.from_the_start(8, 8, CHESS.start)
    game = playing._Game(learned=learned, believed=CHESS.the_start())
    mine = Move((1, 4), (3, 4))
    made = after(CHESS, game.believed, mine)
    theirs = next(m for m in moves(CHESS, made) if (m.start, m.end) == ((6, 2), (4, 2)))
    game.they_answered.they_answered(playing._a_place(made), playing._named(theirs))
    # Her search expected something else; what they answered before in this position is what she expects.
    other = next(m for m in moves(CHESS, made) if (m.start, m.end) == ((6, 4), (4, 4)))

    class _Found:
        line = [mine, other]
        by = "compiled"

    playing._expect_back(game, CHESS, mine, _Found())
    assert game.expected is not None and game.expected[0] == theirs
    assert game.expected[1].turn == FIRST


class _AnOpponentThatThinksApart(BoardApplication):
    """The other side thinks in a thread of its own, as an application's engine thinks in a process of its own: the
    test board's own answering searches on the loop, and she can do nothing while it thinks."""

    async def answers(self, stop: asyncio.Event) -> None:
        from core.agency.searching_by_kind import search

        while not stop.is_set():
            await asyncio.sleep(0.02)
            if self.position.turn != FIRST and not outcome(self.family, self.position):
                found = await asyncio.to_thread(search, self.family, self.position, seconds=self.answers_in_s)
                if found.move is not None:
                    self.make(found.move)


def test_in_a_game_she_looks_on_their_time_and_remembers_their_answers(monkeypatch) -> None:
    monkeypatch.setattr(playing, "STILL_S", 0.02)
    monkeypatch.setattr(playing, "OFFERS_SHOW_S", 0.01)
    monkeypatch.setattr(playing._Rehearsing, "start", lambda self, pool, family: None)
    looked: list[Position] = []
    starting = Pondering.start

    def noted(self, pool, family, position, **kw):
        looked.append(position)
        return starting(self, pool, family, position, **kw)

    monkeypatch.setattr(Pondering, "start", noted)
    app = _AnOpponentThatThinksApart(NOBODYS_GAME, seed=5, answers_in_s=0.3)

    async def run() -> dict:
        seat = playing.a_board_of_two_sides_at_its_start(app.draw())
        stop = asyncio.Event()
        answering = asyncio.create_task(app.answers(stop))
        try:
            return await playing.play_pieces_by_kind(seat, app.look, app.click, seconds=15.0, thinks_s=0.1)
        finally:
            stop.set()
            await answering

    came = asyncio.run(run())
    assert came["moves"] >= 3
    assert looked, "she did not look at anything on their time"
    assert all(position.turn == FIRST for position in looked), "each look is at a position where she is to move"
    assert sum(sum(counts.values()) for counts in came["they_answered"].answered.values()) >= came["theirs"] > 0


def test_what_her_eyes_and_hands_cost_is_told_with_the_game(monkeypatch) -> None:
    monkeypatch.setattr(playing, "STILL_S", 0.02)
    monkeypatch.setattr(playing, "OFFERS_SHOW_S", 0.01)
    monkeypatch.setattr(playing, "WAITS_FOR_THE_FIRST_MOVE_S", 0.5)
    monkeypatch.setattr(playing._Rehearsing, "start", lambda self, pool, family: None)
    app = _AnOpponentThatThinksApart(NOBODYS_GAME, seed=5, answers_in_s=0.1)

    async def run() -> dict:
        seat = playing.a_board_of_two_sides_at_its_start(app.draw())
        stop = asyncio.Event()
        answering = asyncio.create_task(app.answers(stop))
        try:
            return await playing.play_pieces_by_kind(seat, app.look, app.click, seconds=10.0, thinks_s=0.1)
        finally:
            stop.set()
            await answering

    told = asyncio.run(run())["eyes_and_hands"]
    assert told["looks"]["n"] > 0 and told["clicks"]["n"] >= app.clicks > 0
    assert told["looks"]["median_s"] is not None and told["looks"]["total_s"] >= 0


def test_she_also_looks_at_their_strongest_answer_where_it_is_not_the_one_she_expects(monkeypatch) -> None:
    """What they tend to answer schedules where she looks first; what they could answer is looked at as well
    (suggestions of 23 November 2026): a strong side's best answer is not the one it gave before because it gave it."""
    from concurrent.futures import ThreadPoolExecutor

    import core.agency.searching_by_kind as searching

    learned = HowThePiecesMove.from_the_start(8, 8, CHESS.start)
    game = playing._Game(learned=learned, believed=CHESS.the_start())
    mine = Move((1, 4), (3, 4))
    made = after(CHESS, game.believed, mine)
    usual = next(m for m in moves(CHESS, made) if (m.start, m.end) == ((6, 2), (4, 2)))
    game.they_answered.they_answered(playing._a_place(made), playing._named(usual))
    strongest = next(m for m in moves(CHESS, made) if (m.start, m.end) == ((6, 4), (4, 4)))

    class _Found:
        line = [mine, strongest]
        by = "compiled"

    playing._expect_back(game, CHESS, mine, _Found())
    assert game.expected[0] == usual and game.also_expected[0] == strongest
    looked: list = []
    monkeypatch.setattr(searching, "search", lambda family, position, **k: looked.append((position, k["seconds"])))
    with ThreadPoolExecutor(max_workers=1) as pool:
        game.pondering.start(pool, CHESS, game.expected[1], seconds=2.0, weights=None, earlier=[], threads=1,
                             also=game.also_expected[1])
        game.pondering.running.result()
    assert [position for position, _seconds in looked] == [game.expected[1], game.also_expected[1]]
    assert all(seconds == 1.0 for _position, seconds in looked)
