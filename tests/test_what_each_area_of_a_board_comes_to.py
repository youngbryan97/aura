"""A board read as fights apart from each other: the most urgent fight first, and what each move comes to in the whole
(core/agency/what_each_area_of_a_board_comes_to.py)."""
from __future__ import annotations

from core.agency.searching_by_kind import first_weights
from core.agency.what_each_area_of_a_board_comes_to import area_of, areas_of, read_board
from tests.fixtures.a_game_of_pieces_to_learn import CHESS, from_fen


def test_pieces_are_in_one_area_where_they_can_take_or_guard_each_other_and_apart_where_they_cannot() -> None:
    position = from_fen("7k/8/8/3p3p/2P3P1/8/8/K7 w - -")
    areas = areas_of(CHESS, position)
    c4, d5, g4, h5 = (3, 2), (4, 3), (3, 6), (4, 7)
    assert area_of(areas, c4) == area_of(areas, d5) and area_of(areas, g4) == area_of(areas, h5)
    assert area_of(areas, c4) != area_of(areas, g4)
    assert len(areas) == 4, "the two kings stand apart"


def test_with_two_pieces_attacked_in_two_fights_she_saves_the_one_worth_more() -> None:
    # Her queen on a3 is attacked by the pawn on b4 and her knight on h6 by the pawn on g7: she can save one.
    position = from_fen("4k3/6p1/7N/2p5/1p6/Q7/8/4K3 w - -")
    worth = first_weights(CHESS).worth
    read = read_board(CHESS, position, worth, plies=2)
    queen_area = area_of(areas_of(CHESS, position), (2, 0))
    knight_area = area_of(areas_of(CHESS, position), (5, 7))
    assert queen_area != knight_area
    urgency = {reading.part: reading.urgency for reading in read.readings}
    assert urgency[tuple(sorted(queen_area))] > urgency[tuple(sorted(knight_area))] > 0
    chosen = read.choice.weighed
    assert chosen.act.start == (2, 0), "the queen is moved"
    assert read.threats[0].act.end == (2, 0), "their worst threat is to take the queen"


def test_her_answer_to_each_move_they_could_make_is_worked_out_before_they_make_it() -> None:
    """Their pawn on b4 can take her knight on c3; her pawn on b2 takes back. Worked out on their time, before they
    move: what she answers to each move they could make."""
    from core.agency.what_each_area_of_a_board_comes_to import possible_on_a_board

    position = from_fen("4k3/8/8/8/1p6/2N5/1P6/4K3 b - -")
    worth = {"P": 100.0, "N": 300.0, "B": 300.0, "R": 500.0, "Q": 900.0, "K": 0.0}
    counters, found = possible_on_a_board(CHESS, position, worth, plies_at_most=4, seconds=5.0)
    b4, c3, b2 = (3, 1), (2, 2), (1, 1)
    answer, comes_to, deep = counters[(b4, c3, "")]
    assert (answer.start, answer.end) == (b2, c3) and deep >= 2
    assert comes_to == -200.0, "a knight for a pawn is still a loss, and she knows how much before they take"
    assert any(possible.deep >= 2 for possible in found)


def test_in_her_game_her_answer_worked_out_on_their_time_is_her_hint_for_the_move_they_made() -> None:
    from concurrent.futures import Future

    from core.agency.how_the_pieces_move import HowThePiecesMove
    from core.agency.pieces_by_their_kind import Move, after, moves
    from core.agency.playing_pieces_by_kind import _answer_worked_out_on_their_time, _checked, _Game
    from core.agency.searching_by_kind import Found

    before = from_fen("4k3/8/8/8/1p6/2N5/1P6/4K3 b - -")
    theirs = next(m for m in moves(CHESS, before) if (m.start, m.end) == ((3, 1), (2, 2)))
    now = after(CHESS, before, theirs)
    answer = next(m for m in moves(CHESS, now) if (m.start, m.end) == ((1, 1), (2, 2)))
    work: Future = Future()
    work.set_result(({(theirs.start, theirs.end, theirs.becomes): (answer, -200.0, 4)}, ()))
    game = _Game(learned=HowThePiecesMove.from_the_start(8, 8, dict(CHESS.start)), believed=now)
    game.earlier, game.last_followed, game.mapping = [before], theirs, (before.key(), work)
    hint = _answer_worked_out_on_their_time(game, CHESS, moves(CHESS, now))
    assert hint.move == answer and hint.by == "worked out on their time, to check" and game.counter_deep == 4
    # Their move was another: nothing she worked out answers it.
    game.last_followed = Move((7, 4), (7, 3))
    assert _answer_worked_out_on_their_time(game, CHESS, moves(CHESS, now)) is None
    # Her look now agreeing on the move keeps where it came from.
    agreed = _checked(hint, Found(answer, -190, 12, 1000, [answer]), moves(CHESS, now), lambda aside: None)
    assert agreed.by == "worked out on their time, checked"


def test_a_choice_to_wait_is_her_hint_kept_where_her_look_finds_it_no_worse_and_let_go_where_it_does() -> None:
    from dataclasses import replace

    from core.agency.how_the_pieces_move import HowThePiecesMove
    from core.agency.pieces_by_their_kind import moves
    from core.agency.playing_pieces_by_kind import _Game, _held_to_the_areas
    from core.agency.searching_by_kind import Found
    from core.agency.what_each_area_of_a_board_comes_to import BoardReading
    from core.cognition.what_each_area_comes_to import Choice, Weighed, Whole

    position = from_fen("4k3/8/8/8/8/8/8/R3K3 w - -")
    legal = moves(CHESS, position)
    searched, waiting = legal[0], legal[-1]
    choice = Choice(Weighed(part=("a1",), act=waiting, whole=0.0, acting_nowhere=0.0, leaves_them=None), "wait", True)
    areas = BoardReading(readings=(), whole=Whole(0.0, ()), choice=choice, threats=())
    game = _Game(learned=HowThePiecesMove.from_the_start(8, 8, dict(CHESS.start)), believed=position)
    found = Found(searched, 40, 12, 1000, [searched], unsteady=30)
    held = _held_to_the_areas(game, areas, found, legal, lambda aside: Found(waiting, 20, 12, 1000, [waiting]))
    assert held.move == waiting and held.by == "worked out area by area, checked" and game.area_kind == "wait"
    # Her look finds the waiting move worse than its own unsteadiness: her look's move stands.
    worse = _held_to_the_areas(game, areas, found, legal, lambda aside: Found(waiting, -100, 12, 1000, [waiting]))
    assert worse.move == searched
    # A choice of another kind leaves her look's move as it is.
    acted = replace(areas, choice=replace(choice, kind="most good"))
    assert _held_to_the_areas(game, acted, found, legal, lambda aside: None) is found


def test_what_each_side_can_do_on_a_board_stops_when_their_time_is_over_and_leaves_areas_not_begun() -> None:
    from core.agency.what_each_area_of_a_board_comes_to import possible_on_a_board

    position = from_fen("r3k2r/ppp2ppp/2n5/3pp3/3PP3/2N5/PPP2PPP/R3K2R w KQkq -")
    counters, found = possible_on_a_board(CHESS, position, {"P": 100, "N": 300, "B": 300, "R": 500, "Q": 900,
                                                            "K": 0}, plies_at_most=10, seconds=60.0,
                                          stop=lambda: True)
    assert counters == {} and found == ()


def test_a_reading_kept_under_one_set_of_rules_is_not_given_back_under_another() -> None:
    """The same pieces standing the same way, read under the rules and then under rules where the knight may also step
    one square forward onto an empty square: nothing it can take or guard changes, so the board looks the same to
    everything the reading kept rests on, and the second reading is still worked out under its own rules."""
    from dataclasses import replace

    from core.agency.pieces_by_their_kind import Step

    position = from_fen("4k3/8/8/3p4/8/2N5/8/4K3 w - -")
    worth = {"P": 100.0, "N": 300.0, "B": 300.0, "R": 500.0, "Q": 900.0, "K": 0.0}
    read_board(CHESS, position, worth, plies=2)
    knight = CHESS.kinds["N"]
    stepping = replace(knight, steps=(*knight.steps, Step(offsets=((1, 0),), may="moves")))
    widened = replace(CHESS, kinds={**CHESS.kinds, "N": stepping})
    read = read_board(widened, position, worth, plies=2)
    assert any((move.start, move.end) == ((2, 2), (3, 2)) for reading in read.readings
               for move, _comes_to in reading.acts), "under the widened rules the knight may step to c4"


def test_at_the_start_of_a_game_nothing_counted_tells_her_moves_apart_and_nothing_is_chosen() -> None:
    """At the start, no move of either side changes what is counted two moves on: the reading chooses nothing, and her
    own look decides."""
    position = CHESS.the_start()
    worth = {"P": 100.0, "N": 300.0, "B": 300.0, "R": 500.0, "Q": 900.0, "K": 0.0}
    read = read_board(CHESS, position, worth, plies=2)
    assert read.choice.kind == "even" and read.choice.weighed is None
