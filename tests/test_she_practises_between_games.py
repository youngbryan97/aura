"""Between games, when she is idle, she rehearses the last board she played and keeps what the round earns.

A game gives her rehearsal a round or two, while the other side thinks; how the pieces stand together needs tens of
thousands of positions her longer look has searched. Practice between games is the same rehearsal, run when no one
has spoken to her for a while and nothing of hers is at a board.
"""
from __future__ import annotations

import asyncio

import core.agency.practising_between_games as practice
import core.agency.what_a_kind_is_worth as worth
import core.skills.screen_pursuit_board_of_pieces as pursuit
from core.agency.how_the_pieces_move import HowThePiecesMove
from core.perception.pieces_by_their_look import Looks
from tests.fixtures.a_game_of_pieces_to_learn import CHESS


def _a_board_she_played(app: str) -> None:
    rules = HowThePiecesMove.from_the_start(CHESS.rows, CHESS.columns, CHESS.start)
    position = CHESS.the_start()
    family = rules.family()
    for square, (side, _kind) in position.cells.items():
        if side == 0:
            from core.agency.pieces_by_their_kind import moves

            rules.offered(position, square, {m.end for m in moves(CHESS, position) if m.start == square}, family=family)
    rehearsal = worth.a_first_rehearsal(CHESS)
    rehearsal.earned = ["worth"]
    pursuit._keep_what_she_learned(app, Looks(), rules, rehearsal)


def _rounds_run(monkeypatch) -> list[int]:
    ran: list[int] = []

    def a_round(family, start, rehearsal, **options):
        rehearsal.rounds += 1
        rehearsal.last = {"kept": False, "tried": "moves"}
        ran.append(rehearsal.rounds)
        return rehearsal

    monkeypatch.setattr(worth, "rehearse", a_round)
    # Every round here is a round of rehearsal: preparing has its own test.
    monkeypatch.setattr(practice, "_prepare_some", lambda name: 0)
    monkeypatch.setattr(practice, "_HELD", {})
    monkeypatch.setattr(practice, "_RUNNING", None)
    monkeypatch.setattr(practice, "_on_mains", lambda: True)
    return ran


def _practise(orch=None) -> bool:
    async def go() -> bool:
        begun = practice.practise_when_idle(orch)
        if practice._RUNNING is not None:
            await practice._RUNNING
        return begun

    return asyncio.run(go())


def test_idle_she_plays_a_round_of_the_last_board_and_keeps_it(monkeypatch) -> None:
    import core.runtime.background_policy as policy

    monkeypatch.setattr(policy, "background_activity_reason", lambda *a, **k: "")
    ran = _rounds_run(monkeypatch)
    _a_board_she_played("A Practice Board")
    assert _practise()
    assert ran == [1]
    assert pursuit._what_she_kept("A Practice Board")["rehearsal"].rounds == 1
    # The next round carries on the rehearsal she holds.
    assert _practise()
    assert ran == [1, 2]


def test_a_game_played_since_is_what_practice_carries_on_from(monkeypatch) -> None:
    import core.runtime.background_policy as policy

    monkeypatch.setattr(policy, "background_activity_reason", lambda *a, **k: "")
    ran = _rounds_run(monkeypatch)
    _a_board_she_played("A Practice Board")
    assert _practise()
    # A game writes its own record: its rehearsal, not the one in hand, is the next round's.
    _a_board_she_played("A Practice Board")
    assert _practise()
    assert ran == [1, 1]


def test_not_at_a_board_nor_when_she_is_not_idle_nor_on_battery(monkeypatch) -> None:
    import core.runtime.background_policy as policy

    ran = _rounds_run(monkeypatch)
    _a_board_she_played("A Practice Board")
    monkeypatch.setattr(policy, "background_activity_reason", lambda *a, **k: "")
    with practice.at_a_board():
        assert not _practise()
    monkeypatch.setattr(policy, "background_activity_reason", lambda *a, **k: "user active 4s ago")
    assert not _practise()
    monkeypatch.setattr(policy, "background_activity_reason", lambda *a, **k: "")
    monkeypatch.setattr(practice, "_on_mains", lambda: False)
    assert not _practise()
    assert ran == []


def test_she_practises_though_no_one_has_spoken_to_her_since_she_woke(monkeypatch) -> None:
    import core.runtime.background_policy as policy

    asked: list[dict] = []

    def reason(*args, **options):
        asked.append(options)
        return "" if options.get("allow_no_user_anchor") else "no_user_anchor"

    monkeypatch.setattr(policy, "background_activity_reason", reason)
    ran = _rounds_run(monkeypatch)
    _a_board_she_played("A Practice Board")
    assert _practise()
    assert ran == [1] and asked[-1]["allow_no_user_anchor"] is True


def test_a_round_that_ends_during_a_game_leaves_the_record_to_the_game(monkeypatch) -> None:
    import core.runtime.background_policy as policy

    monkeypatch.setattr(policy, "background_activity_reason", lambda *a, **k: "")
    monkeypatch.setattr(practice, "_HELD", {})
    monkeypatch.setattr(practice, "_RUNNING", None)
    monkeypatch.setattr(practice, "_on_mains", lambda: True)
    _a_board_she_played("A Practice Board")

    def a_round_that_a_game_interrupts(family, start, rehearsal, **options):
        practice._PLAYING += 1  # a game sits down while the round runs
        rehearsal.rounds += 1
        rehearsal.last = {"kept": False}
        return rehearsal

    monkeypatch.setattr(worth, "rehearse", a_round_that_a_game_interrupts)
    monkeypatch.setattr(practice, "_prepare_some", lambda name: 0)
    try:
        assert _practise()
    finally:
        practice._PLAYING = 0
    assert pursuit._what_she_kept("A Practice Board")["rehearsal"].rounds == 0
