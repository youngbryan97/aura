"""A board of two sides is seen in a picture, known for its family, played by the rules its words give, and searched.

Read offline 2026-10-11 from a game that "plays like checkers but with a huge twist": a draughts board drawn in two
pinks with no writing on it, two sides of chips, a figure standing on one chip of each side, and rules that change the
family's (a special piece that swaps with its own and wins at the far row). The picture in tests/fixtures is that
board as the game first drew it.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from core.agency.playing_a_board import a_move_said, draughts_said, is_draughts, position_of, search
from core.agency.rules_of_a_board import Draughts, Position
from core.perception.a_board_seen import a_board_in

pytestmark = pytest.mark.unit

PICTURE = Path(__file__).parent / "fixtures" / "a_board_of_two_sides.png"
WORDS = ("This game plays like checkers but with a huge twist: the Big Shot! Three Ways to Win: 1. Capture all of your "
         "opponent's pieces. 2. Block in your opponent so they have no moves left. 3. Get your Big Shot to king row. "
         "Regular pieces can move only forward. If you're able to jump over an opponent's piece and capture it, you've got "
         "to do it. Get a piece to king row, and it becomes a king, which can move forward or backward. Your Big Shot can "
         "always move forward or backward. Or your Big Shot can swap places with any of your pieces next to it, as long as "
         "you don't have any captures you've got to do.")


def test_the_board_is_seen_square_by_square_with_its_two_sides_and_the_figure_standing_on_each():
    board = a_board_in(np.asarray(Image.open(PICTURE).convert("RGB")))
    assert (board.rows, board.columns, board.width, board.height) == (8, 8, 56.0, 42.0)
    assert board.says().splitlines() == ["a.a.A.a.", ".a.a.a.a", "a.a.a.a.", "........",
                                         "........", ".b.b.b.b", "b.b.b.b.", ".b.B.b.b"]
    assert is_draughts(board) and is_draughts(board, WORDS)


def test_its_words_change_the_familys_rules_as_they_say():
    rules = draughts_said(WORDS, up="b", down="a")
    assert rules.men_forward_only and rules.must_take and rules.crowned
    assert rules.special_swaps and rules.special_wins_at_far_row
    plain = draughts_said("Checkers. Jump your opponent's pieces.", up="b", down="a")
    assert not plain.special_swaps and not plain.special_wins_at_far_row


def _position(rows: list[str], turn: str) -> Position:
    return Position.from_rows([[{".": "", "A": "a+", "B": "b+"}.get(c, c) for c in row] for row in rows], turn)


def test_a_jump_must_be_made_where_the_words_say_so_and_runs_on():
    rules = Draughts(up="b", down="a", must_take=True)
    position = _position(["........", "........", "...a....", "........", ".a......", "b.......", "........",
                          "........"], "b")
    moves = rules.moves(position)
    assert moves == [((5, 0), (3, 2), (1, 4))]
    after = rules.after(position, moves[0])
    assert not after.pieces("a") and rules.winner(after) == "b"


def test_the_search_takes_a_win_it_can_see_and_a_move_is_said_as_players_say_it():
    rules = Draughts(up="b", down="a", special_swaps=True, special_wins_at_far_row=True)
    position = _position(["........", "..B.....", "........", "......a.", "........", "........", "a.......",
                          "........"], "b")
    found = search(rules, position, seconds=0.5)
    assert found.move is not None and found.move[-1][0] == 0
    assert rules.winner(rules.after(position, found.move)) == "b"
    assert a_move_said(position, ((5, 1), (4, 2))) == "b3 to c4"


def test_each_move_she_makes_is_told_and_heard_as_news():
    from core.agency.playing_a_board import Searched, a_move_told
    from core.language.what_a_watcher_hears import WhatAWatcherHears

    position = _position(["........", "........", "........", "........", "........", "........", ".b.b....",
                          "........"], "b")
    first, second = ((6, 1), (5, 0)), ((6, 3), (5, 4))
    told = [a_move_told(position, first, Searched(first, 0.4, 6, 900)),
            a_move_told(position, second, Searched(second, 1001.0, 7, 900))]
    assert told == ["My move: b2 to a3, looking 6 moves ahead.", "My move: d2 to e3, and I can see it winning."]
    watcher = WhatAWatcherHears()
    assert all(watcher.heard(line, now=10.0 + k) for k, line in enumerate(told))


def test_the_real_board_is_searched_within_its_time():
    board = a_board_in(np.asarray(Image.open(PICTURE).convert("RGB")))
    position, up, down = position_of(board, turn="")
    rules = draughts_said(WORDS, up, down)
    found = search(rules, Position(position.cells, up), seconds=0.6)
    assert found.move in rules.moves(Position(position.cells, up)) and found.depth >= 3


class _NoughtsAndCrosses:
    """A drawn three-by-three board, ruled in grey lines, red pieces hers and blue the other side's: each click of hers
    on an empty square puts a red piece there, and the other side answers in the first empty square."""

    def __init__(self):
        self.cells = [["" for _ in range(3)] for _ in range(3)]
        self.at = 0.0

    def picture(self):
        image = np.full((300, 300, 3), 235, np.uint8)
        for k in range(4):
            image[:, min(299, k * 100):min(300, k * 100 + 3)] = 90
            image[min(299, k * 100):min(300, k * 100 + 3), :] = 90
        for r in range(3):
            for c in range(3):
                if self.cells[r][c]:
                    colour = (200, 40, 40) if self.cells[r][c] == "red" else (40, 60, 200)
                    image[r * 100 + 28:r * 100 + 72, c * 100 + 28:c * 100 + 72] = colour
        return image

    async def look(self):
        self.at += 0.1
        if self.at < getattr(self, "covered_until", 0.0):
            covered = self.picture()
            covered[60:240] = (250, 220, 40)          # a word put up over the board for a moment
            return covered, self.at
        return self.picture(), self.at

    async def click(self, x, y):
        r, c = min(2, int(y * 3)), min(2, int(x * 3))
        if self.cells[r][c]:
            return
        self.cells[r][c] = "red"
        self.covered_until = self.at + getattr(self, "covers_for", 0.0)
        empty = [(rr, cc) for rr in range(3) for cc in range(3) if not self.cells[rr][cc]]
        if empty and not _three(self.cells, "red"):
            rr, cc = empty[0]
            self.cells[rr][cc] = "blue"


def _three(cells, who):
    lines = [[(r, c) for c in range(3)] for r in range(3)] + [[(r, c) for r in range(3)] for c in range(3)]
    lines += [[(i, i) for i in range(3)], [(i, 2 - i) for i in range(3)]]
    return any(all(cells[r][c] == who for r, c in line) for line in lines)


def test_a_placing_game_is_known_by_its_shape_and_won_by_search_through_her_eyes_and_hands(monkeypatch):
    import asyncio

    from core.agency import playing_a_board as playing
    from core.agency.playing_a_board import a_board_game, play_a_board

    monkeypatch.setattr(playing, "STILL_S", 0.0)
    game = _NoughtsAndCrosses()
    board = a_board_in(game.picture())
    assert (board.rows, board.columns) == (3, 3) and a_board_game(board) == "placing"
    said = []
    result = asyncio.run(play_a_board(game.look, game, "", seconds=20.0, say=said.append))
    assert result["ended"] == "won" and _three(game.cells, "red"), (result, game.cells, said)


def test_a_board_hidden_a_moment_is_waited_for(monkeypatch):
    """A word put up over the board after each of her moves (LIVE 2026-10-11 "YOU MOVE A PIECE") hides it for a
    moment; play waits for it to come back rather than ending."""
    import asyncio

    from core.agency import playing_a_board as playing
    from core.agency.playing_a_board import play_a_board

    monkeypatch.setattr(playing, "STILL_S", 0.0)
    game = _NoughtsAndCrosses()
    game.covers_for = 1.5
    result = asyncio.run(play_a_board(game.look, game, "", seconds=30.0, say=None))
    assert result["ended"] == "won" and _three(game.cells, "red"), (result, game.cells)


def test_a_places_rules_are_kept_from_one_stretch_of_play_to_the_next(monkeypatch):
    """LIVE 2026-10-11 a second stretch of a draughts game, its rules screen long gone, was played without the piece its
    rules had changed: the words a place said are kept with what she keeps of it."""
    import asyncio
    from types import SimpleNamespace

    from core.agency import playing_a_board as playing
    from core.skills import screen_pursuit_as_it_happens as pursuit

    played_by: list[str] = []

    async def play(look, hands, words, *, seconds, say=None):
        played_by.append(words)
        return {"moves": 1, "ended": "out of time"}

    async def look():
        return np.asarray(Image.open(PICTURE).convert("RGB")), 0.0

    async def declined(*_args, **_kwargs):
        return None

    monkeypatch.setattr(playing, "play_a_board", play)
    # The words are kept for a board her general player cannot sit at, which goes to the player of typed rules.
    monkeypatch.setattr(pursuit, "_played_as_pieces_by_their_kind", declined)
    keep: dict = {}
    for words in ([WORDS], ["YOU MOVE A PIECE"]):
        reflexes = SimpleNamespace(keep=keep, words=words, look=look, ends_at=1e9, stretches=[], over_because="")
        assert asyncio.run(pursuit._played_a_board(reflexes, 100.0))
    assert draughts_said(played_by[-1], "b", "a").special_swaps


def test_a_board_of_two_sides_at_its_start_is_played_as_a_game_of_pieces(monkeypatch):
    """The web page's board of 11 October 2026 is a board of two sides at its start: it is played by her general
    player, its rules learned from the game, and what she learned is there for the next stretch."""
    import asyncio
    from types import SimpleNamespace

    import core.agency.playing_pieces_by_kind as general
    from core.agency import playing_a_board as playing
    from core.skills import screen_pursuit_as_it_happens as pursuit

    asked: list[dict] = []

    async def play_pieces(seat, look, click, *, seconds, **kwargs):
        asked.append({"seat": seat, "look": await look(), **kwargs})
        return {"ended": "lost", "moves": 3, "learned": "what she learned", "rehearsal": None, "went_wrong": None,
                "they_answered": None}

    async def typed(*_args, **_kwargs):
        raise AssertionError("the player of typed rules was asked")

    async def look():
        return np.asarray(Image.open(PICTURE).convert("RGB")), 0.0

    async def click(_x, _y):
        return None

    monkeypatch.setattr(general, "play_pieces_by_kind", play_pieces)
    monkeypatch.setattr(playing, "play_a_board", typed)
    keep: dict = {}
    reflexes = SimpleNamespace(keep=keep, words=[WORDS], look=look, click=click, ends_at=1e9, stretches=[],
                               over_because="")
    assert asyncio.run(pursuit._played_a_board(reflexes, 100.0))
    assert asked and (asked[0]["seat"].rows, asked[0]["seat"].columns) == (8, 8)
    assert asked[0]["look"].shape[2] == 3, "her general player is given the picture alone"
    assert reflexes.stretches[-1]["losses"] == 1
    assert keep["pieces_by_their_kind"]["learned"] == "what she learned"


def test_a_move_answered_before_she_looks_is_a_move_made():
    from core.agency.playing_a_board import _made_and_answered

    wanted = {(4, 1), (5, 2), (6, 3)}                      # where her move from (5, 0) to (4, 1) leaves her pieces
    assert _made_and_answered({(4, 1), (6, 3)}, wanted, ((5, 0), (4, 1)))          # answered by taking (5, 2)
    assert _made_and_answered({(5, 2), (6, 3)}, wanted, ((5, 0), (4, 1)))          # answered by taking the piece moved
    assert not _made_and_answered({(5, 0), (5, 2), (6, 3)}, wanted, ((5, 0), (4, 1)))   # not made
    assert not _made_and_answered({(4, 1), (5, 2), (6, 3), (7, 0)}, wanted, ((5, 0), (4, 1)))


def test_a_board_in_play_keeps_its_squares_and_is_lost_from_sight_when_covered():
    """A board does not move or change its squares in play (LIVE 2026-10-11 eight by eight read once as eight by ten,
    and a win claimed from it): the board seen first is kept, and is there where its squares show their own ground."""
    picture = np.asarray(Image.open(PICTURE).convert("RGB"))
    known = a_board_in(picture)
    assert len(known.ground) == 2
    again = a_board_in(picture, known.sides, known)
    assert again.says() == known.says() and (again.rows, again.columns) == (8, 8)
    covered = picture.copy()
    covered[100:300] = (250, 220, 40)
    assert a_board_in(covered, known.sides, known) is None


def test_a_pieces_standing_is_kept_by_its_history_not_its_look_alone():
    from core.agency.playing_a_board import _more_than_there_were, _standings_kept

    before = _position(["........", "..b.....", "........", "....B...", "........", "........", "......a.",
                        "........"], "b")
    # The man came to the far row and was crowned, and looks apart there as the special piece does.
    now = _position([".B......", "........", "........", "....B...", "........", "........", "......a.",
                     "........"], "a")
    rules = Draughts(up="b", down="a", special_wins_at_far_row=True)
    assert rules.winner(now) == "b"                                   # as the picture alone has it: won
    kept = _standings_kept(now, before)
    assert kept.at((0, 1)) == "bK" and kept.at((3, 4)) == "b+" and rules.winner(kept) == ""
    # A crowned piece that looks like the rest of its side is crowned still.
    crowned = Position(tuple(tuple("bK" if (r, c) == (0, 1) else cell for c, cell in enumerate(row))
                             for r, row in enumerate(before.cells)), "b")
    looks_a_man = _position([".b......", "........", "........", "....B...", "........", "........", "........",
                             "........"], "b")
    assert _standings_kept(looks_a_man, crowned).at((0, 1)) == "bK"
    assert _more_than_there_were(_position(["b.b.....", *["........"] * 7], "b"), _position(["b.......", *["........"] * 7], "b"), "b", "a")


class _DraughtsGame:
    """Draughts drawn eight by eight, her side blue at the bottom: a piece clicked and then each square it goes to
    moves it; the other side thinks a while before it answers, and clicks while it thinks do nothing."""

    SQUARE, MARGIN = 40, 10

    def __init__(self, think_s: float = 1.4, wants_between_s: float = 0.0):
        import time as _time

        self.time = _time
        # A piece set down sooner than this after it was taken up is not set down.
        self.wants_between_s, self.picked_at = wants_between_s, 0.0
        self.rules = Draughts(up="b", down="a", must_take=True)
        rows = [".a.a.a.a", "a.a.a.a.", ".a.a.a.a", "........", "........", "b.b.b.b.", ".b.b.b.b", "b.b.b.b."]
        self.position = Position.from_rows([["" if c == "." else c for c in row] for row in rows], "b")
        self.think_s, self.answer_at, self.picked, self.clicked_out_of_turn, self.made = think_s, None, [], 0, 0

    def _settle(self):
        if self.answer_at is not None and self.time.monotonic() >= self.answer_at:
            moves = self.rules.moves(self.position)
            if moves:
                self.position = self.rules.after(self.position, moves[0])
            self.answer_at = None

    def picture(self):
        size = 8 * self.SQUARE + 2 * self.MARGIN
        image = np.full((size, size, 3), 60, np.uint8)
        for r in range(8):
            for c in range(8):
                y, x = self.MARGIN + r * self.SQUARE, self.MARGIN + c * self.SQUARE
                image[y:y + self.SQUARE, x:x + self.SQUARE] = (235, 205, 205) if (r + c) % 2 == 0 else (190, 110, 110)
                cell = self.position.cells[r][c]
                if cell:
                    image[y + 10:y + 30, x + 10:x + 30] = (40, 60, 200) if cell[0] == "b" else (200, 40, 40)
        return image

    async def look(self):
        self._settle()
        return self.picture(), self.time.monotonic()

    async def click(self, x, y):
        self._settle()
        if self.position.turn != "b" or self.answer_at is not None:
            self.clicked_out_of_turn += 1
            return
        size = 8 * self.SQUARE + 2 * self.MARGIN
        square = (int((y * size - self.MARGIN) // self.SQUARE), int((x * size - self.MARGIN) // self.SQUARE))
        if self.picked and self.time.monotonic() - self.picked_at < self.wants_between_s:
            return
        self.picked.append(square)
        self.picked_at = self.time.monotonic()
        moves = [m for m in self.rules.moves(self.position) if m[:len(self.picked)] == tuple(self.picked)]
        if not moves:
            self.picked = [square] if any(m[0] == square for m in self.rules.moves(self.position)) else []
            return
        done = [m for m in moves if len(m) == len(self.picked)]
        if done and len(done[0]) > 1:
            self.position, self.picked, self.made = self.rules.after(self.position, done[0]), [], self.made + 1
            self.answer_at = self.time.monotonic() + self.think_s


def test_her_turn_comes_when_the_other_side_has_answered_not_when_the_board_stands_still(monkeypatch):
    """Offline 2026-10-11 moves clicked while the other side was thinking were refused by the game, and read as moves
    the game does not allow."""
    import asyncio

    from core.agency import playing_a_board as playing
    from core.agency.playing_a_board import play_a_board

    monkeypatch.setattr(playing, "STILL_S", 0.3)
    monkeypatch.setattr(playing, "THINKS_S", 0.2)
    game, said = _DraughtsGame(), []
    result = asyncio.run(play_a_board(game.look, game, "Checkers.", seconds=14.0, say=said.append))
    assert game.made >= 3 and game.clicked_out_of_turn == 0, (result, game.made, game.clicked_out_of_turn, said)
    assert not any("wouldn't take" in line for line in said), said


def test_a_swap_is_seen_made_only_where_the_special_piece_now_stands(monkeypatch):
    """A swap leaves her pieces on the squares they were on (offline 2026-10-11 a swap the game had not made was taken
    as made, and asked for again)."""
    import asyncio

    from core.agency import playing_a_board as playing

    monkeypatch.setattr(playing, "MOVE_SHOWS_S", 0.4)
    picture = np.asarray(Image.open(PICTURE).convert("RGB"))
    known = a_board_in(picture)
    position, up, _down = position_of(known, turn="")
    rules = Draughts(up=up, down=_down, special_swaps=True)
    swap = ((7, 3), (6, 2), (7, 3))
    expected = rules.after(Position(position.cells, up), swap)

    async def look():
        return picture, 0.0

    assert not asyncio.run(playing._shown(look, known.sides, expected, up, swap, known))
    assert asyncio.run(playing._shown(look, known.sides, Position(position.cells, up), up, (), known))


def test_a_move_not_taken_is_made_again_slower_before_it_is_given_up(monkeypatch):
    """A game that wants a moment between a piece taken up and set down takes her move the second time, slower;
    and a move not taken is never the end of the game while the rules give one."""
    import asyncio

    from core.agency import playing_a_board as playing
    from core.agency.playing_a_board import play_a_board

    monkeypatch.setattr(playing, "STILL_S", 0.3)
    monkeypatch.setattr(playing, "THINKS_S", 0.2)
    game, said = _DraughtsGame(think_s=0.6, wants_between_s=0.6), []
    # Each move here takes about 6.5 s (set down too soon, waited on, made again slower, then answered): measured 23
    # November 2026, the second was made at 13.8 s, and 14 s left it no room under a loaded machine.
    result = asyncio.run(play_a_board(game.look, game, "Checkers.", seconds=20.0, say=said.append))
    assert game.made >= 2 and result["ended"] == "out of time", (result, game.made, said)
    assert not any("wouldn't take" in line for line in said), said


def test_a_grid_of_three_by_three_on_a_place_whose_words_speak_of_no_contest_is_no_board_game():
    """Offline 2026-10-12 an alien maker's grid of nine parts to pick from was taken for three in a row."""
    from core.agency.playing_a_board import a_board_game

    board = a_board_in(_NoughtsAndCrosses().picture())
    assert a_board_game(board, "TEXTURES SUIT THEMES RANDOMIZE I'M DONE! START OVER") == ""
    assert a_board_game(board, "Get three in a row before the computer does!") == "placing"
    assert a_board_game(board) == "placing"
