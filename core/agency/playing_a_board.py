"""Playing a board game: knowing its family from the board and its words, reading what it changes, and searching moves.

A board she sees (core/perception/a_board_seen.py) is a position, and a
family of rules (core/agency/rules_of_a_board.py) makes it a game she can
search: every move she could make, every answer to it, as deep as the time a
move is worth allows, each line judged at its end by how it stands. That is how
a finite game is played well: not argued, searched. What the game says it
changes is read off its own words and set as the family's variations; where a
move she makes is one the game will not take, the rules she holds were wrong
there, and are changed.

Which family it is comes from the board and the words together: pieces of two
sides on one colour of square, facing each other from their own ends, is
draughts, whether or not it says so; "plays like checkers" says so outright.
Nothing here knows a game.
"""
from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass
from typing import Any

from core.agency.rules_of_a_board import KING, SPECIAL, Draughts, Move, Position
from core.runtime.wall_clock import monotonic_time

__all__ = ["Searched", "a_board_game", "a_move_told", "draughts_said", "is_draughts", "placing_said", "position_of", "search"]

#: What a game's words say of its draughts, each a variation of the family, said where a word begins.
_SAYS = {
    "men_forward_only": (re.compile(r"\b(?:regular |ordinary |normal )?(?:pieces|men|checkers) (?:can |may )?(?:only )?"
                                    r"move (?:only )?forward\b", re.I), True),
    "must_take": (re.compile(r"\b(?:jump|capture|take)\b.{0,80}\b(?:you(?:'ve| have)? got to|you must|have to|must) "
                             r"(?:do it|jump|capture|take)", re.I), True),
    "crowned": (re.compile(r"\b(?:becomes? a king|kinged|crowned|king row)\b", re.I), True),
    "special_swaps": (re.compile(r"\bswap(?:s)? places? with\b|\bchange places with\b", re.I), True),
    "special_wins_at_far_row": (re.compile(r"\b(?:get|bring|move) (?:your |the )?\w+(?: \w+)? to (?:the )?(?:king|far|"
                                           r"last|end) row\b", re.I), True),
}
#: Words that name the family outright.
_NAMED = re.compile(r"\b(?:checkers|draughts|dama)\b", re.I)


def draughts_said(words: str, up: str, down: str) -> Draughts:
    """Draughts as a game's words say it is played: the family's usual rules, changed where its words change them."""
    rules = Draughts(up=up, down=down)
    changes = {name: value for name, (said, value) in _SAYS.items() if said.search(words or "")}
    # Where nothing says men keep to forward, the family's usual way holds; and a special piece moves either way.
    return rules.with_changes(**changes) if changes else rules


#: What names a placing game, and the line that wins it; and a board's own shape where its words say nothing.
_PLACING_NAMED = (
    (re.compile(r"\b(?:tic[- ]?tac[- ]?toe|noughts and crosses|x'?s and o'?s|three in a row)\b", re.I), 3, False),
    (re.compile(r"\b(?:connect (?:4|four)|four in a row)\b", re.I), 4, True),
    (re.compile(r"\b(?:five in a row|gomoku|connect (?:5|five))\b", re.I), 5, False),
)
_PLACING_SHAPES = {(3, 3): (3, False), (6, 7): (4, True), (15, 15): (5, False), (19, 19): (5, False)}


#: Words that say a board is played against someone in turns.
_A_CONTEST = re.compile(r"\b(?:turns?|your move|opponent|computer|versus|vs\.?|player (?:1|2|one|two)|in a row|"
                        r"x'?s and o'?s|noughts|line up|beat|win)\b", re.I)


def placing_said(board: Any, words: str = "") -> Any:
    """The placing family a board is, by its words first and else by its shape: a line of three on three by three, four
    dropped down the columns of six by seven, five on a large open board; None where it is no placing game."""
    from core.agency.rules_of_a_board import Placing

    if board is None:
        return None
    for named, line, falls in _PLACING_NAMED:
        if named.search(words or ""):
            return Placing(line=line, falls=falls or bool(re.search(r"\bdrop\b|\bcolumn", words or "", re.I)))
    shaped = _PLACING_SHAPES.get((board.rows, board.columns))
    if shaped is None:
        return None
    # By its shape where nothing is said, or where what is said speaks of a contest in turns; not where the place has
    # words and none of them do: offline 2026-10-12 an alien maker's grid of nine parts to pick from, its words
    # "TEXTURES SUIT THEMES ... I'M DONE!", was taken for three in a row, and the time left went on searching moves on
    # it, none made.
    if (words or "").strip() and not _A_CONTEST.search(words or ""):
        return None
    # A placing game begins nearly empty, and a draughts board full of two facing sides is not one.
    held = sum(1 for row in board.cells for cell in row if cell)
    return Placing(line=shaped[0], falls=shaped[1]) if held <= 2 else None


def a_board_game(board: Any, words: str = "") -> str:
    """Which family of board game a board seen is ("draughts", "placing"), or "" where it is none she knows."""
    if is_draughts(board, words):
        return "draughts"
    if placing_said(board, words) is not None:
        return "placing"
    return ""


def is_draughts(board: Any, words: str = "") -> bool:
    """Whether a board seen is draughts: its words say so, or two sides stand on one colour of square, each from its own
    end of a board of six squares a side or more."""
    if board is None or board.rows < 6 or board.columns < 6:
        return False
    sides = [s for s in board.sides]
    if len(sides) < 2:
        return False
    if _NAMED.search(words or ""):
        return True
    places = [(r, c) for r in range(board.rows) for c in range(board.columns) if board.cells[r][c]]
    one_colour = len({(r + c) % 2 for r, c in places}) == 1
    rows_of = {side: [r for r, c in places if board.cells[r][c].rstrip("+") == side] for side in sides[:2]}
    apart = all(rows_of[s] for s in sides[:2]) and (
        max(rows_of[sides[0]]) < min(rows_of[sides[1]]) or max(rows_of[sides[1]]) < min(rows_of[sides[0]]))
    return one_colour and apart


def position_of(board: Any, turn: str) -> tuple[Position, str, str]:
    """A board seen as a position, and which side moves up the board and which down (by where each side stands)."""
    rows = [[cell for cell in row] for row in board.cells]
    sides = list(board.sides)[:2]
    middle = {s: sum(r for r in range(board.rows) for c in range(board.columns) if rows[r][c].rstrip("+") == s)
              / max(1, sum(1 for r in range(board.rows) for c in range(board.columns) if rows[r][c].rstrip("+") == s))
              for s in sides}
    up = max(sides, key=lambda s: middle[s])
    down = next(s for s in sides if s != up)
    return Position.from_rows(rows, turn), up, down


@dataclass
class Searched:
    move: Move | None
    worth: float
    depth: int
    looked_at: int


def search(rules: Draughts, position: Position, *, seconds: float = 1.0, deepest: int = 12) -> Searched:
    """The best move for the side whose turn it is, searched as deep as ``seconds`` allows (alpha-beta, deepening one
    move at a time, the best line of the last depth tried first)."""
    side = position.turn
    ends_at = time.monotonic() + seconds
    best = Searched(None, -math.inf, 0, 0)
    moves = rules.moves(position)
    if not moves:
        return best
    if len(moves) == 1:
        return Searched(moves[0], 0.0, 0, 1)
    looked = [0]
    ordered = list(moves)
    for depth in range(1, deepest + 1):
        alpha, chosen, chosen_worth = -math.inf, None, -math.inf
        try:
            for move in ordered:
                worth = -_negamax(rules, rules.after(position, move), depth - 1, -math.inf, -alpha, side, ends_at, looked)
                if worth > chosen_worth:
                    chosen, chosen_worth = move, worth
                alpha = max(alpha, worth)
        except TimeoutError:
            break
        best = Searched(chosen, chosen_worth, depth, looked[0])
        ordered = [chosen, *(m for m in ordered if m != chosen)]
        if abs(chosen_worth) >= _WON:
            break
    return best


_WON = 1000.0


def _negamax(rules: Draughts, position: Position, depth: int, alpha: float, beta: float, root: str, ends_at: float,
             looked: list[int]) -> float:
    looked[0] += 1
    if looked[0] % 256 == 0 and time.monotonic() > ends_at:
        raise TimeoutError
    winner = rules.winner(position)
    if winner == "-":
        return 0.0  # a board filled with no winner is drawn
    if winner:
        return (_WON + depth) if winner == position.turn else -(_WON + depth)
    if depth <= 0:
        return rules.worth(position, position.turn)
    best = -math.inf
    for move in rules.moves(position):
        worth = -_negamax(rules, rules.after(position, move), depth - 1, -beta, -alpha, root, ends_at, looked)
        if worth > best:
            best = worth
        alpha = max(alpha, worth)
        if alpha >= beta:
            break
    return best


#: How long the board must stand still before a turn is taken to have ended, and the longest a move is waited on.
STILL_S = 0.8
MOVE_SHOWS_S = 2.5
#: How long a move is searched for.
THINKS_S = 1.2
#: How long a board may be out of sight before play is given up as over.
HIDDEN_AT_MOST_S = 4.0
#: How long the other side's answer to her move is waited for before the turn is taken to be hers again.
ANSWER_AT_MOST_S = 15.0
#: The time between the clicks of a move, and how many times a move is made before the game is taken not to allow it.
CLICKS_APART_S = 0.35
TRIES_OF_A_MOVE = 2


def a_square_named(position: Position, square: tuple[int, int]) -> str:
    """A square as a board's players name it: a column letter from the left, a row number from the bottom ("c3")."""
    row, column = square
    return f"{'abcdefghijklmnopqrstuvwxyz'[column]}{position.rows - row}"


def a_move_said(position: Position, move: Move) -> str:
    """A move as a watcher reads it: "c3 to d4", "c3 takes, to e5", "c1 swaps with d2"."""
    if len(move) == 3 and move[2] == move[0]:
        return f"{a_square_named(position, move[0])} swaps with {a_square_named(position, move[1])}"
    names = [a_square_named(position, square) for square in move]
    if abs(move[1][0] - move[0][0]) == 2:
        return f"{names[0]} takes, to " + ", then ".join(names[1:])
    return f"{names[0]} to {names[-1]}"


def a_move_told(position: Position, move: Move, found: Searched) -> str:
    """Her move as she tells it, with what her search saw of it: "My move: c3 to d4, looking 6 moves ahead." Each move
    is news to a watcher, so the line is about the move (LIVE 2026-10-11 "Moving d3 to c4, ..." was one shape over and
    over to the chat, and most of her moves went unheard)."""
    line = f"My move: {a_move_said(position, move)}"
    if found.move == move and found.worth >= _WON:
        return line + ", and I can see it winning."
    return line + (f", looking {found.depth} moves ahead." if found.depth > 1 and found.move == move else ".")


async def play_a_board(look: Any, hands: Any, words: str, *, seconds: float, say: Any = None) -> dict[str, Any]:
    """Play the board in front of her for ``seconds``: wait for her turn, search the move, make it by clicking its
    squares, see it made, and wait for the answer. ``look`` gives a picture and its time; ``hands.click(x, y)`` clicks
    at shares of it. What came of it, for whoever handed the game over."""
    import asyncio

    from core.perception.a_board_seen import a_board_in

    began = monotonic_time()
    tell = say or (lambda _line: None)
    seen = await look()
    if seen is not None:
        first = a_board_in(seen[0])
        if a_board_game(first, words) == "placing":
            return await _play_placing(look, hands, placing_said(first, words), seconds=seconds, tell=tell)
    sides: dict[str, Any] | None = None
    known: Any = None
    rules: Draughts | None = None
    mine = ""
    refused: set[tuple[str, Move]] = set()
    tries: dict[tuple[str, Move], int] = {}
    made = 0
    tally = _Tally()
    believed: Position | None = None
    answer_awaited: float | None = None
    won_seen = ""
    last_seen, still_since = "", time.monotonic()
    hidden_since: float | None = None
    while monotonic_time() - began < seconds:
        seen = await look()
        if seen is None:
            return tally.came_of(made, "the picture could not be taken")
        picture = seen[0]
        board = a_board_in(picture, sides, known)
        if board is None or not is_draughts(board, words):
            hidden_since, gone = _out_of_sight(hidden_since)
            if gone:
                return tally.came_of(made, "no board is in front of her")
            await asyncio.sleep(0.2)
            continue
        hidden_since = None
        sides, known = sides or board.sides, known or board
        said = board.says()
        if said != last_seen:
            last_seen, still_since = said, time.monotonic()
            await asyncio.sleep(0.1)
            continue
        if time.monotonic() - still_since < STILL_S:
            await asyncio.sleep(0.1)
            continue
        position, up, down = position_of(board, turn="")
        if rules is None:
            mine = up
            rules = draughts_said(words, up, down)
            tell(_the_rules_said(rules))
        position = Position(position.cells, mine)
        if believed is not None and _more_than_there_were(position, believed, mine, rules.other(mine)):
            # No piece comes onto a draughts board in play: a picture with more is misread, and looked at again.
            last_seen = ""
            await asyncio.sleep(0.2)
            continue
        kept = _standings_kept(position, believed)
        if answer_awaited is not None and not _answered(kept, believed, rules.other(mine)) \
                and time.monotonic() - answer_awaited < ANSWER_AT_MOST_S:
            # Her turn comes when the other side has answered hers, not when the board stands still: offline 2026-10-11
            # moves clicked while the other side was thinking were refused by the game.
            await asyncio.sleep(0.2)
            continue
        answer_awaited = None
        position = believed = kept
        tally.saw(position, mine, rules.other(mine))
        winner = rules.winner(position)
        if winner and won_seen != said:
            # An end is believed when it is seen again a moment later, not from one picture.
            won_seen, last_seen = said, ""
            await asyncio.sleep(STILL_S)
            continue
        if winner:
            return tally.came_of(made, "won" if winner == mine else "lost", board=said)
        found = search(rules, position, seconds=THINKS_S)
        move = found.move
        if move is None or (said, move) in refused:
            options = [m for m in rules.moves(position) if (said, m) not in refused]
            move = options[0] if options else None
        if move is None and not rules.moves(position):
            return tally.came_of(made, "no move is left to her", board=said)
        if move is None:
            # Every move the rules give was not taken: the game is not ready for one yet (a word put up over play), and
            # they are tried again in a moment. Offline 2026-10-11 her one jump was not taken while "YOUR OPPONENT
            # CAPTURES 1 PIECE" was up, and she gave the game up with no move left to her.
            refused = {one for one in refused if one[0] != said}
            last_seen = ""
            await asyncio.sleep(MOVE_SHOWS_S)
            continue
        tried = tries.get((said, move), 0)
        if not tried:
            tell(a_move_told(position, move, found))
        for square in (move[:2] if len(move) == 3 and move[2] == move[0] else move):
            x, y = board.centre(*square)
            await hands.click(x / picture.shape[1], y / picture.shape[0])
            # Slower each time the move was not taken: a game may want a moment between a piece taken up and set down.
            await asyncio.sleep(CLICKS_APART_S * (1 + 2 * tried))
        expected = rules.after(position, move)
        if not await _shown(look, sides, expected, mine, move, known):
            tries[(said, move)] = tried + 1
            if tried + 1 >= TRIES_OF_A_MOVE:
                # The game would not take it: the rules she holds allowed a move it does not.
                refused.add((said, move))
                tell("The game wouldn't take that move; I'll go by what it takes.")
            last_seen = ""
            continue
        believed, answer_awaited = expected, time.monotonic()
        made += 1
        last_seen, still_since = "", time.monotonic()
    return tally.came_of(made, "out of time")


@dataclass
class _Tally:
    """How many pieces each side had when she first came to move in a stretch of play, and when she last did: what she
    took and what she lost, for whoever asks whether the game is getting anywhere."""

    first: tuple[int, int] | None = None
    now: tuple[int, int] | None = None

    def saw(self, position: Position, mine: str, theirs: str) -> None:
        self.now = (len(position.pieces(mine)), len(position.pieces(theirs)))
        self.first = self.first or self.now

    def came_of(self, made: int, ended: str, **more: Any) -> dict[str, Any]:
        first, now = self.first or (0, 0), self.now or (0, 0)
        return {"moves": made, "ended": ended, "taken": max(0, first[1] - now[1]), "lost": max(0, first[0] - now[0]),
                **more}


def _answered(position: Position, before: Position | None, theirs: str) -> bool:
    """Whether the other side has moved since ``before``: its pieces stand otherwise, or one of hers is gone."""
    if before is None:
        return True
    return ({sq: position.at(sq) for sq in position.pieces(theirs)} != {sq: before.at(sq) for sq in before.pieces(theirs)}
            or len(position.pieces(position.turn)) < len(before.pieces(position.turn)))


def _more_than_there_were(position: Position, before: Position, mine: str, theirs: str) -> bool:
    """Whether a side has more pieces than it had: in draughts none ever comes onto the board, so the picture is
    misread."""
    return any(len(position.pieces(side)) > len(before.pieces(side)) for side in (mine, theirs))


def _standings_kept(position: Position, before: Position | None) -> Position:
    """A piece's standing as its history says, not its look alone: a side has no more special pieces than it had, the
    ones that stand where its special ones stood before (or nearest) are they, and any other that looks apart has been
    crowned; a piece crowned before is crowned still where it stands. LIVE 2026-10-11 a piece crowned on the far row
    looked like a special one there, and that wins."""
    if before is None:
        return position
    cells = [list(row) for row in position.cells]
    for side in {cell[:1] for row in position.cells for cell in row if cell}:
        apart = [square for square in position.pieces(side) if position.at(square)[1:] == SPECIAL]
        kept: set[tuple[int, int]] = set()
        for were in (square for square in before.pieces(side) if before.at(square)[1:] == SPECIAL):
            left = [square for square in apart if square not in kept]
            if left:
                kept.add(min(left, key=lambda square, were=were: abs(square[0] - were[0]) + abs(square[1] - were[1])))
        for r, c in position.pieces(side):
            if (r, c) in apart and (r, c) not in kept or position.at((r, c)) == side and before.at((r, c)) == side + KING:
                cells[r][c] = side + KING
    return Position(tuple(tuple(row) for row in cells), position.turn)


def _out_of_sight(since: float | None) -> tuple[float, bool]:
    """Since when the board has been out of sight, and whether that is too long to wait for it. A board hidden a moment
    (a word put up over it, a piece taken with a flourish) is waited for: LIVE 2026-10-11 "YOU MOVE A PIECE" put up
    over the board ended her play two moves in."""
    since = time.monotonic() if since is None else since
    return since, time.monotonic() - since > HIDDEN_AT_MOST_S


def _waited_longer_while_hidden(board: Any, ends_at: float, latest: float) -> float:
    """How long a move is waited on to show: time the board is out of sight is not time it failed to show in."""
    return ends_at if board is not None else min(latest, max(ends_at, time.monotonic() + MOVE_SHOWS_S / 2))


def _made_and_answered(here: set[tuple[int, int]], wanted: set[tuple[int, int]], move: Move) -> bool:
    """Her move made, and the other side's answer come before she looked: the square it left is empty of hers, none of
    hers stands where the move leaves none, and where it went is hers or one of hers was taken. LIVE 2026-10-11 a quick
    answer that took one of her pieces read as the game not taking her move."""
    if not move or (len(move) == 3 and move[2] == move[0]):
        return False
    return move[0] not in here and here <= wanted and (move[-1] in here or len(here) < len(wanted))


async def _shown(look: Any, sides: Any, expected: Position, mine: str, move: Move = (), known: Any = None) -> bool:
    """Whether the board comes to show the move she made: her pieces where the move leaves them, or the move made and
    answered already."""
    import asyncio

    from core.perception.a_board_seen import a_board_in

    wanted = {square for square in expected.pieces(mine)}
    # A swap leaves her pieces on the squares they were on: it shows by where her special piece stands (offline
    # 2026-10-11 a swap the game had not made was taken as made, and the same swap asked for again).
    special = {square for square in wanted if expected.at(square)[1:] == SPECIAL}
    ends_at = monotonic_time() + MOVE_SHOWS_S
    latest = ends_at + HIDDEN_AT_MOST_S
    while monotonic_time() < ends_at:
        seen = await look()
        if seen is None:
            return False
        board = a_board_in(seen[0], sides, known)
        ends_at = _waited_longer_while_hidden(board, ends_at, latest)
        if board is not None:
            here = {(r, c) for r in range(board.rows) for c in range(board.columns) if board.cells[r][c].rstrip("+") == mine}
            apart = {(r, c) for (r, c) in here if board.cells[r][c].endswith("+")}
            if here == wanted and special <= apart or _made_and_answered(here, wanted, move):
                return True
        await asyncio.sleep(0.15)
    return False


async def _play_placing(look: Any, hands: Any, rules: Any, *, seconds: float, tell: Any) -> dict[str, Any]:
    """A placing game played: her side is whatever her first piece is, it is her turn when the count of pieces says so,
    and each move is a click on the place (or the column) the search chooses."""
    import asyncio
    from dataclasses import replace

    from core.perception.a_board_seen import a_board_in

    began = monotonic_time()
    tell(f"It's {rules.line} in a row" + (", dropped down the columns" if rules.falls else "") + ". I'll search my moves.")
    sides: dict[str, Any] | None = None
    mine, theirs, she_began = "", "", None
    my_colour: Any = None
    made, refused = 0, set()
    hidden_since: float | None = None
    known: Any = None
    last_seen, still_since = "", time.monotonic()
    while monotonic_time() - began < seconds:
        seen = await look()
        if seen is None:
            return {"moves": made, "ended": "the picture could not be taken"}
        picture = seen[0]
        board = a_board_in(picture, sides, known)
        if board is None:
            hidden_since, gone = _out_of_sight(hidden_since)
            if gone:
                return {"moves": made, "ended": "no board is in front of her"}
            await asyncio.sleep(0.2)
            continue
        hidden_since, known = None, known or board
        # The sides are named by their colours once both are on the board, and hers is the one her piece was.
        if sides is None and len(board.sides) >= 2:
            sides = board.sides
        if my_colour is not None and board.sides:
            mine = min(board.sides, key=lambda side: _apart(board.sides[side], my_colour))
            theirs = next((side for side in board.sides if side != mine), theirs)
        said = board.says()
        if said != last_seen:
            last_seen, still_since = said, time.monotonic()
            await asyncio.sleep(0.1)
            continue
        if time.monotonic() - still_since < STILL_S:
            await asyncio.sleep(0.1)
            continue
        cells = [[cell.rstrip("+") for cell in row] for row in board.cells]
        counts = {side: sum(row.count(side) for row in cells) for side in {c for row in cells for c in row if c}}
        if she_began is None:
            she_began = not counts
            theirs = next(iter(counts), "")
        her_turn = counts.get(mine, 0) == counts.get(theirs, 0) - (0 if she_began else 1) if mine else \
            sum(counts.values()) == (0 if she_began else 1)
        me, them = mine or "me", theirs or "them"
        rows = [[me if c == mine and mine else them if c and c != mine else "" for c in row] for row in cells]
        position = Position.from_rows(rows, me)
        game = replace(rules, first=me, second=them)
        ended = game.winner(position)
        if ended:
            return {"moves": made, "ended": "drawn" if ended == "-" else "won" if ended == me else "lost", "board": said}
        if not her_turn:
            await asyncio.sleep(0.2)
            continue
        found = search(game, position, seconds=THINKS_S)
        move = found.move if found.move not in refused else next((m for m in game.moves(position) if m not in refused), None)
        if move is None:
            return {"moves": made, "ended": "no move is left to her", "board": said}
        tell(f"Placing at {a_square_named(position, move[0])}.")
        x, y = board.centre(*move[0])
        await hands.click(x / picture.shape[1], y / picture.shape[0])
        placed = await _a_piece_comes(look, sides, move[0], mine, known)
        if placed is None:
            refused.add(move)
            continue
        my_colour = my_colour if my_colour is not None else placed
        made += 1
        last_seen, still_since = "", time.monotonic()
    return {"moves": made, "ended": "out of time"}


def _apart(one: Any, other: Any) -> float:
    return float(sum((float(a) - float(b)) ** 2 for a, b in zip(one, other, strict=False)))


async def _a_piece_comes(look: Any, sides: Any, square: tuple[int, int], mine: str, known: Any = None) -> Any:
    """The colour of the side of the piece that comes onto ``square`` after her click; None where none comes in time."""
    import asyncio

    from core.perception.a_board_seen import a_board_in

    ends_at = monotonic_time() + MOVE_SHOWS_S
    latest = ends_at + HIDDEN_AT_MOST_S
    while monotonic_time() < ends_at:
        seen = await look()
        board = a_board_in(seen[0], sides, known) if seen is not None else None
        ends_at = _waited_longer_while_hidden(board, ends_at, latest)
        if board is not None and board.rows > square[0] and board.columns > square[1]:
            piece = board.cells[square[0]][square[1]].rstrip("+")
            if piece and (not mine or piece == mine):
                return board.sides.get(piece)
        await asyncio.sleep(0.15)
    return None


def _the_rules_said(rules: Draughts) -> str:
    parts = ["men go forward only" if rules.men_forward_only else "men go either way"]
    if rules.must_take:
        parts.append("a jump has to be made")
    if rules.special_swaps:
        parts.append("the special piece can swap with one of its own beside it")
    if rules.special_wins_at_far_row:
        parts.append("bringing it to the far row wins")
    return "It's draughts, played as its words say: " + "; ".join(parts) + ". I'll search my moves."
