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


def test_judged_by_her_weights_a_quiet_move_that_leaves_a_piece_fewer_ways_is_not_one_that_changes_nothing() -> None:
    """Her knight in the middle of an empty board: every move keeps every piece, so judged by what is held alone every
    move changes nothing. Judged by weights that also count each piece's ways to go, the knight's move to the edge
    costs her, and is not a move that only passes the turn (a review Bryan passed on, 24 November 2026: "A waiting move
    might preserve all pieces while weakening mobility, safety, or defensive coverage")."""
    from dataclasses import replace

    from core.agency.what_each_area_of_a_board_comes_to import _WORKED_OUT
    from core.cognition.what_each_area_comes_to import changes_nothing, weigh_acts

    position = from_fen("4k3/8/8/8/3N4/8/8/4K3 w - -")
    held = first_weights(CHESS)
    ways = replace(held, mobility={kind: 10.0 for kind in held.worth})
    d4, c2 = (3, 3), (1, 2)
    _WORKED_OUT.clear()
    by_worth = read_board(CHESS, position, held.worth, plies=2)
    assert all(changes_nothing(w, by_worth.readings) for w in weigh_acts(by_worth.readings))
    by_weights = read_board(CHESS, position, held.worth, plies=2, weights=ways)
    to_the_edge = next(w for w in weigh_acts(by_weights.readings) if (w.act.start, w.act.end) == (d4, c2))
    assert not changes_nothing(to_the_edge, by_weights.readings)
    knight = next(r for r in by_weights.readings if d4 in r.part)
    assert to_the_edge.act in knight.weakens
    # The knight's moves that keep its ways are still moves that change nothing.
    assert any(changes_nothing(w, by_weights.readings) for w in weigh_acts(by_weights.readings) if w.act.start == d4)


def test_her_choice_is_weighed_by_the_areas_played_as_one_game_where_there_was_time() -> None:
    """Two pieces attacked in two fights: the areas are also played as one game, her to move and them to move, and the
    move she chooses is weighed by what that game comes to after it, not by the quick sum (a review Bryan passed on,
    24 November 2026: the exact sum was not yet deciding anything)."""
    position = from_fen("4k3/6p1/7N/2p5/1p6/Q7/8/4K3 w - -")
    read = read_board(CHESS, position, first_weights(CHESS).worth, plies=2)
    assert read.played is not None
    hers, theirs = read.played
    exact = {(part, act): comes_to for part, act, comes_to in hers.each}
    chosen = read.choice.weighed
    assert chosen.whole == exact[(chosen.part, chosen.act)] and chosen.acting_nowhere == theirs.comes_to
    assert chosen.act.start == (2, 0), "the queen is moved"


def test_a_judgement_held_once_judges_each_board_as_judging_it_afresh_does() -> None:
    """The compiled search's judgement, parsed once for many boards, says what its judgement of each board alone says."""
    import pytest

    from core.agency.searching_by_kind import _compiled, described, placed

    engine = _compiled()
    if engine is None or not hasattr(engine, "Judgement"):
        pytest.skip("the compiled search is not built here, or is a build from before it could hold a judgement")
    weights = first_weights(CHESS).as_json()
    boards = [from_fen(fen) for fen in ("4k3/6p1/7N/2p5/1p6/Q7/8/4K3 w - -", "4k3/8/8/8/3N4/8/8/4K3 b - -")]
    boards.append(CHESS.the_start())
    held = engine.Judgement(described(CHESS), weights)
    assert held.of_each([placed(board) for board in boards]) == [
        engine.evaluate(described(CHESS), placed(board), weights) for board in boards]


def test_the_areas_played_as_one_game_come_to_what_searching_the_whole_board_does() -> None:
    """Boards where a move in one area changes another: what it can do, by leaving a royal piece open or closing a way,
    and what stands in it, by taking from it. Played as one game on one board, each move's whole is what a plain search
    of the whole board finds two moves deep, with her to move and with them to move (a review Bryan passed on,
    24 November 2026: "Verify the composed results against exhaustive global search on small interacting worlds")."""
    import math
    import random
    from dataclasses import replace

    import core.agency.what_each_area_of_a_board_comes_to as areas
    from core.agency.pieces_by_their_kind import after, moves
    from core.cognition.what_each_area_comes_to import HERS, THEIRS, play_the_sum

    worth = first_weights(CHESS).worth
    boards = []
    for seed in (1, 10, 4):
        rng = random.Random(seed)
        board = CHESS.the_start()
        for _ in range(rng.randrange(14, 34)):
            board = after(CHESS, board, rng.choice(moves(CHESS, board)))
        boards.append(board)
    for board in boards:
        for first in (HERS, THEIRS):
            hers = board.turn if first == HERS else 1 - board.turn
            base = areas._standing(board, worth, hers)

            def searched(position, side, left, hers=hers, base=base):
                mover = hers if side == HERS else 1 - hers
                legal = moves(CHESS, replace(position, turn=mover)) if left > 0 else []
                if not legal:
                    return areas._standing(position, worth, hers) - base
                found = [searched(after(CHESS, replace(position, turn=mover), m), 1 - side, left - 1) for m in legal]
                return max(found) if side == HERS else min(found)

            mover = board.turn
            truth = {(m.start, m.end, m.becomes): searched(after(CHESS, board, m), 1 - first, 1)
                     for m in moves(CHESS, board)}
            acts, moved, judge = areas._the_world_of(CHESS, board, worth, hers)
            parts = [(areas._named(area), (board, area)) for area in areas.areas_of(CHESS, board)]
            played = play_the_sum(parts, acts=acts, after=moved, judge=judge, plies_at_most=2, seconds=math.inf,
                                  name_of=lambda state: (state[0].key(), state[1]), first=first,
                                  joined=areas._held_together, whole=lambda states, judge=judge: judge(states[0]))
            got = {(act.start, act.end, act.becomes): comes_to for _part, act, comes_to in played.each}
            assert mover == board.turn and got == truth
