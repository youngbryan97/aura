"""She keeps her own count of her clock, and reads it while it stands, on their time.

LIVE 21 November 2026, game 16 against Stockfish: reading her clock took half a second of every turn, on her clock, and
by her thirtieth move her turns had a second and a half to think in. A clock read once says what it will show next from
what she does: her turn's time comes off it, and what each move adds besides (time a game gives back after a move, the
moment it takes her to see theirs) is learned from the next reading. She reads it again only when her count may have
drifted by more than a turn's share, and then while it stands still on their time.
"""
from __future__ import annotations

import asyncio
import time

import core.agency.playing_pieces_by_kind as playing
from core.agency.pieces_by_their_kind import FIRST, outcome
from core.agency.searching_by_kind import search
from core.cognition.keeping_her_own_count import KeptCount
from tests.fixtures.a_board_application import BoardApplication
from tests.fixtures.a_game_of_pieces_to_learn import NOBODYS_GAME


def test_what_each_step_adds_that_she_did_not_count_is_learned() -> None:
    count = KeptCount()
    count.read(100.0)
    shown = 100.0
    for _ in range(6):
        shown += -3.0 + 2.0  # her turn took three seconds, and the instrument gave two back
        count.step(-3.0)
        count.read(shown)
    assert abs(count.per_step - 2.0) < 1e-9
    count.step(-3.0)
    assert abs(count.value - (shown - 1.0)) < 1e-9


def test_she_reads_again_only_when_her_count_may_be_off_by_more_than_she_can_afford() -> None:
    count = KeptCount()
    assert count.must_read(1.0), "with no count she reads"
    count.read(60.0)
    count.step(-2.0)
    assert count.must_read(1.0), "one reading has not shown what a step misses"
    for shown, change in ((57.9, -2.0), (55.8, -2.0), (53.8, -2.0)):
        count.read(shown)
        count.step(change)
    assert not count.must_read(1.0)
    for _ in range(40):
        count.step(-0.5)
    assert count.must_read(1.0), "steps enough since the last reading and her count may have drifted"


class _ClockedApplication(BoardApplication):
    """Her clock runs while it is her turn and stands while it is theirs, and gives back half a second after each of
    her moves. Each reading is counted, with whether her clock was running as it was read. The other side thinks
    somewhere else, as another program does, so she can look at the board while it thinks."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.left = 9.0
        self.running_since: float | None = time.monotonic()
        self.reads_on_her_time = 0
        self.reads_on_theirs = 0

    def shows(self) -> float:
        running = time.monotonic() - self.running_since if self.running_since is not None else 0.0
        return round(self.left - running, 1)

    def make(self, move) -> None:
        hers = self.position.turn == FIRST
        super().make(move)
        if hers and self.running_since is not None:
            self.left -= time.monotonic() - self.running_since
            self.left += 0.5
            self.running_since = None
        elif not hers:
            self.running_since = time.monotonic()

    async def answers(self, stop: asyncio.Event) -> None:
        while not stop.is_set():
            await asyncio.sleep(0.02)
            if self.position.turn != FIRST and not outcome(self.family, self.position):
                await asyncio.sleep(self.answers_in_s)
                found = search(self.family, self.position, seconds=0.01)
                if found.move is not None:
                    self.make(found.move)

    async def clock(self) -> float:
        await asyncio.sleep(0.05)
        if self.running_since is not None:
            self.reads_on_her_time += 1
        else:
            self.reads_on_theirs += 1
        return self.shows()


def test_her_clock_is_read_on_their_time_and_seldom(monkeypatch) -> None:
    monkeypatch.setattr(playing, "STILL_S", 0.02)
    monkeypatch.setattr(playing, "OFFERS_SHOW_S", 0.01)
    monkeypatch.setattr(playing, "WAITS_FOR_THE_FIRST_MOVE_S", 0.2)
    monkeypatch.setattr(playing._Rehearsing, "start", lambda self, pool, family: None)
    monkeypatch.setattr(playing._Rehearsing, "moves_left", lambda self, made: 30)
    app = _ClockedApplication(NOBODYS_GAME, seed=5, answers_in_s=0.6)
    counts: list[KeptCount] = []
    real_read = playing._read_her_clock_while_it_stands

    async def watched(clock, rehearsing, game):
        counts.append(game.count)
        await real_read(clock, rehearsing, game)

    monkeypatch.setattr(playing, "_read_her_clock_while_it_stands", watched)

    async def run() -> dict:
        seat = playing.a_board_of_two_sides_at_its_start(app.draw())
        assert seat is not None
        stop = asyncio.Event()
        answering = asyncio.create_task(app.answers(stop))
        try:
            return await playing.play_pieces_by_kind(seat, app.look, app.click, seconds=14.0, thinks_s=0.1,
                                                     clock=app.clock)
        finally:
            stop.set()
            await answering

    came = asyncio.run(run())
    hers = came["moves"]
    assert hers >= 8, came
    # Read on her own time once, where she had no count to begin from.
    assert app.reads_on_her_time <= 1, (app.reads_on_her_time, app.reads_on_theirs)
    assert app.reads_on_her_time + app.reads_on_theirs < hers
    # Her count is where the clock stood when it last stopped or began again, within a look's lag.
    count = counts[-1]
    assert count.value is not None
    assert abs(count.value - app.left) < 0.5, (count.value, app.left)
