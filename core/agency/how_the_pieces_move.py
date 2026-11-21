"""How the pieces of a game move, learned from the game: what it offers when a piece is picked up, what is seen played.

Nobody hands her the rules. A person new to a board game learns them the same way: they pick a piece up and see where
the game lets it go; they watch the other side move; a move the game will not take tells them something too. What they
learn they hold in a few ideas (core/agency/pieces_by_their_kind.py): this kind steps one square, that one goes along a
line until something is in the way, this one only takes on the diagonal, that one may go two the first time.

Evidence is kept per kind (pieces that look alike move alike, each side towards its own front) and per direction, in
the piece's own frame (forward, rightward), as distances:

* offered: the game showed the square as somewhere the piece may go;
* withheld: the square was empty, the way to it clear, and the game did not offer it;
* took / would not take: a piece of the other side stood there and was offered, or was not;
* watched: a move made, by either side, that was the kind going there.

From these the simplest description that fits is drawn: a direction a kind has gone is one it goes, as far as it has
been seen to go and no further where it has been seen to stop with the way clear, and along the whole line where it has
gone more than one square and never been seen to stop; a direction seen to take but never to move to an empty square is
one it only takes along, and one seen to stop at a piece of the other side is one it only moves along; a reach seen only
from a piece that had not moved yet is one only such a piece has. A direction never seen is one it does not go: what the
game has not shown is not assumed. Mirror images of a direction (left for right) are taken to go alike unless the game
shows otherwise, because pieces are drawn alike on both sides; forward and backward are not, because many games' pieces
only go one way.

Rarer things are learned when they happen: a kind whose squares the game withholds when the other side could take on
them is royal; two of a side's pieces moving at once is a crossing; a piece changing its look on the far row becomes the
other kind; a take on an empty square that removes the piece beside it is a take in passing. Nothing here knows a game.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

from core.agency.pieces_by_their_kind import FIRST, Family, Kind, Move, Position, Step, attacked

logger = logging.getLogger("Aura.HowThePiecesMove")

__all__ = ["HowThePiecesMove", "a_move_seen", "in_its_frame", "moves_seen"]

Square = tuple[int, int]
Direction = tuple[int, int]
#: Times a royal kind must be seen kept off squares the other side could take on before it is taken to be royal.
ROYAL_AFTER = 2
#: How rare the evidence against a reading may be and the reading still stand: what she sees passes through what she
#: holds of the rest of the game, which is itself still being learned, so a reading that one sighting in ten
#: contradicts is the game's and the sighting is her model's.
AGAINST_AT_MOST = 0.1
#: The longest single step looked for along a way: a leap further than three squares is not looked for.
MAX_STEP = 3


#: Every line from a square, as far as the way is clear: what a kind she has not seen move might do.
_EVERY_LINE = (Step(((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)), rides=0),)


def in_its_frame(side: int, start: Square, end: Square) -> tuple[Direction, int]:
    """The way from ``start`` to ``end`` as the piece of ``side`` faces: a direction with no common factor, and how many
    of it."""
    ahead = 1 if side == FIRST else -1
    forward, right = (end[0] - start[0]) * ahead, (end[1] - start[1]) * ahead
    times = math.gcd(abs(forward), abs(right)) or 1
    return (forward // times, right // times), times


def _square_along(side: int, start: Square, direction: Direction, times: int) -> Square:
    ahead = 1 if side == FIRST else -1
    return start[0] + direction[0] * ahead * times, start[1] + direction[1] * ahead * times


@dataclass
class _Seen:
    """What has been seen of one kind going one way."""

    offered: set[int] = field(default_factory=set)
    offered_unmoved: set[int] = field(default_factory=set)
    withheld: set[int] = field(default_factory=set)
    withheld_unmoved: set[int] = field(default_factory=set)
    took: int = 0
    would_not_take: int = 0
    moved_to_empty: int = 0
    jumped: int = 0

    def went_to_empty(self) -> bool:
        return self.moved_to_empty > 0

    def any_seen_going(self) -> bool:
        return bool(self.offered or self.offered_unmoved or self.took)


@dataclass
class HowThePiecesMove:
    """What she has learned of one game's pieces, and the family it makes."""

    rows: int
    columns: int
    start: dict[Square, tuple[int, str]]
    seen: dict[str, dict[Direction, _Seen]] = field(default_factory=dict)
    kept_off_attacked: dict[str, int] = field(default_factory=dict)
    went_onto_attacked: dict[str, int] = field(default_factory=dict)
    crosses: dict[str, str] = field(default_factory=dict)
    becomes: dict[str, list[str]] = field(default_factory=dict)
    passes: set[str] = field(default_factory=set)
    takes_passing: set[str] = field(default_factory=set)
    names: dict[str, str] = field(default_factory=dict)
    #: Moves the game would not take that the rules she held allowed, and how many of them left each kind of hers where
    #: the other side could take it.
    refusals: int = 0
    refused_leaving_open: dict[str, int] = field(default_factory=dict)
    #: Kinds seen taken with the game going on after: none of them is the kind the game keeps safe, whatever else
    #: seemed to say so. The kind last taken waits for the next move to count.
    taken_and_played_on: dict[str, int] = field(default_factory=dict)
    last_taken: str = ""

    @classmethod
    def from_the_start(cls, rows: int, columns: int, start: dict[Square, tuple[int, str]]) -> HowThePiecesMove:
        return cls(rows=rows, columns=columns, start=dict(start))

    def knows_of(self, kinds: set[str]) -> None:
        """Kinds seen on the board, whether or not they have been seen to move."""
        for kind in kinds:
            self.seen.setdefault(kind, {})

    def _of(self, kind: str, direction: Direction) -> _Seen:
        return self.seen.setdefault(kind, {}).setdefault(direction, _Seen())

    def offered(self, position: Position, square: Square, squares: set[Square], family: Family | None = None) -> None:
        """The game offered ``squares`` for the piece on ``square``: every way it could have gone is evidence."""
        side, kind = position.cells[square]
        unmoved = square in position.unmoved
        for forward in range(-MAX_STEP, MAX_STEP + 1):
            for right in range(-MAX_STEP, MAX_STEP + 1):
                if (forward, right) == (0, 0) or math.gcd(abs(forward), abs(right)) != 1:
                    continue
                direction = (forward, right)
                times = 1
                while True:
                    there = _square_along(side, square, direction, times)
                    if not (0 <= there[0] < self.rows and 0 <= there[1] < self.columns):
                        break
                    held = position.cells.get(there)
                    is_offered = there in squares
                    if held is None:
                        if is_offered:
                            way = self._of(kind, direction)
                            (way.offered_unmoved if unmoved else way.offered).add(times)
                            way.moved_to_empty += 1
                        elif kind in self.seen and direction in self.seen[kind]:
                            way = self.seen[kind][direction]
                            (way.withheld_unmoved if unmoved else way.withheld).add(times)
                    else:
                        if held[0] != side:
                            if is_offered:
                                way = self._of(kind, direction)
                                way.took += 1
                                (way.offered_unmoved if unmoved else way.offered).add(times)
                            elif self._goes(kind, direction, times, unmoved):
                                self.seen[kind][direction].would_not_take += 1
                        break
                    times += 1
        if family is not None:
            self._what_the_offer_keeps_safe(position, square, squares, family)
        # A square offered with a piece on the way to it was leapt to.
        for there in squares:
            direction, times = in_its_frame(side, square, there)
            between = [_square_along(side, square, direction, t) for t in range(1, times)]
            if any(b in position.cells for b in between):
                self._of(kind, direction).jumped += 1

    def _what_the_offer_keeps_safe(self, position: Position, square: Square, squares: set[Square],
                                   family: Family) -> None:
        """The moves her rules give this piece, held against what the game offered: each move leaves some kinds of hers
        where the other side could take them. A kind left open by moves the game withheld, and by none it offered, is one
        the game keeps safe. Pins and attacks on it show this without being looked for; a square withheld for a reason of
        the piece's own (a step it may only take unmoved) leaves nothing of hers newly open and says nothing."""
        from core.agency.pieces_by_their_kind import _possible, after

        side = position.cells[square][0]
        if position.turn != side:
            return
        for move in _possible(family, position):
            if move.start != square:
                continue
            made = after(family, position, move)
            left_open = {kind for there, (owner, kind) in made.cells.items()
                         if owner == side and attacked(family, made, there, 1 - side)}
            tally = self.went_onto_attacked if move.end in squares else self.kept_off_attacked
            for kind in left_open:
                tally[kind] = tally.get(kind, 0) + 1

    def _goes(self, kind: str, direction: Direction, times: int, unmoved: bool) -> bool:
        way = self.seen.get(kind, {}).get(direction)
        if way is None:
            return False
        reach = self._reach(way, unmoved)
        return reach == 0 or 0 < times <= reach

    def watched(self, position: Position, move: Move) -> None:
        """A move seen made: the kind went there, and what happened with it tells what more it may do."""
        side, kind = position.cells[move.start]
        if self.last_taken:
            # A move after a piece was taken: the game went on without it.
            self.taken_and_played_on[self.last_taken] = self.taken_and_played_on.get(self.last_taken, 0) + 1
            self.last_taken = ""
        if move.taken_at is not None and move.taken_at in position.cells:
            self.last_taken = position.cells[move.taken_at][1]
        direction, times = in_its_frame(side, move.start, move.end)
        unmoved = move.start in position.unmoved
        way = self._of(kind, direction)
        if move.partner is not None:
            partner_kind = position.cells[move.partner[0]][1]
            self.crosses[kind] = partner_kind
            return
        between = [_square_along(side, move.start, direction, t) for t in range(1, times)]
        if any(b in position.cells for b in between):
            way.jumped += 1
        (way.offered_unmoved if unmoved else way.offered).add(times)
        if move.taken_at is not None:
            way.took += 1
            if move.taken_at != move.end:
                self.takes_passing.add(kind)
                victim = position.cells[move.taken_at][1]
                self.passes.add(victim)
        else:
            way.moved_to_empty += 1
        if move.becomes and move.becomes != kind:
            options = self.becomes.setdefault(kind, [])
            if move.becomes not in options:
                options.append(move.becomes)
            self.seen.setdefault(move.becomes, {})

    def refused(self, position: Position, move: Move, family: Family | None = None) -> None:
        """A move the game would not take. Where the rules she held allowed it, either the kind does not go there, or
        the move would leave a kind of hers open to be taken that the game will not let her leave open: the kind left
        open by every such refusal is royal."""
        side, kind = position.cells[move.start]
        direction, times = in_its_frame(side, move.start, move.end)
        unmoved = move.start in position.unmoved
        way = self._of(kind, direction)
        if family is not None:
            from core.agency.pieces_by_their_kind import after

            made = after(family, position, move)
            open_kinds = {k for sq, (s, k) in made.cells.items() if s == side and attacked(family, made, sq, 1 - side)}
            if open_kinds and self._goes(kind, direction, times, unmoved):
                # The kind goes there, as she has seen: what stopped it is what the move left open.
                self.refusals += 1
                for open_kind in open_kinds:
                    self.refused_leaving_open[open_kind] = self.refused_leaving_open.get(open_kind, 0) + 1
                return
        if move.taken_at is not None:
            way.would_not_take += 1
        else:
            (way.withheld_unmoved if unmoved else way.withheld).add(times)

    def stopped_with_open(self, position: Position, family: Family) -> None:
        """The game ended her turn with no move: the kinds of the side to move that the other side could take are what
        it was keeping safe, as a refusal shows it."""
        side = position.turn
        open_kinds = {k for sq, (s, k) in position.cells.items() if s == side and attacked(family, position, sq, 1 - side)}
        if open_kinds:
            self.refusals += 1
            for kind in open_kinds:
                self.refused_leaving_open[kind] = self.refused_leaving_open.get(kind, 0) + 1

    @staticmethod
    def _reach(way: _Seen, unmoved: bool) -> int:
        """How far a kind goes one way: 0 for the whole line, else the number of squares."""
        offered = way.offered | (way.offered_unmoved if unmoved else set())
        if not offered:
            return -1
        furthest = max(offered)
        withheld = {t for t in (way.withheld | (way.withheld_unmoved if unmoved else set())) if t > furthest}
        if withheld:
            return furthest
        if furthest >= 2 and not way.jumped:
            return 0
        return furthest

    def _steps_of(self, kind: str) -> list[Step]:
        steps: list[Step] = []
        ways = self.seen.get(kind, {})
        for direction, way in sorted(ways.items()):
            if not way.any_seen_going():
                continue
            moved_reach = self._reach(way, unmoved=False)
            unmoved_reach = self._reach(way, unmoved=True)
            # Seen taking, and seldom or never going to an empty square that way though kept off one: it only takes
            # along it. Seen stopped at a piece of the other side it could have reached, and seldom or never taking:
            # it only moves.
            if way.took and way.moved_to_empty <= AGAINST_AT_MOST * way.took and (way.withheld or way.withheld_unmoved):
                may = "takes"
            elif way.would_not_take and way.took <= AGAINST_AT_MOST * way.would_not_take:
                may = "moves"
            else:
                may = "either"
            offered = way.offered | way.offered_unmoved
            kept_nearer = offered and any(t < min(offered) for t in way.withheld | way.withheld_unmoved)
            if offered and 1 not in offered and (way.jumped or kept_nearer):
                # Never to the next square, only further: a leap to each distance seen, over whatever is between.
                for times in sorted(offered):
                    only = times not in way.offered
                    steps.append(Step(((direction[0] * times, direction[1] * times),), rides=1, may=may,
                                      only_unmoved=only))
                continue
            if moved_reach >= 0:
                steps.append(Step((direction,), rides=1 if way.jumped and moved_reach == 1 else moved_reach, may=may,
                                  takes_in_passing=kind in self.takes_passing and may == "takes"))
            if unmoved_reach != moved_reach and (unmoved_reach == 0 or unmoved_reach > max(moved_reach, 0)):
                steps.append(Step((direction,), rides=unmoved_reach, may=may, only_unmoved=True,
                                  leaves_passable=kind in self.passes))
        return _mirrored(steps, ways)

    def as_memory(self) -> dict:
        """What she learned, in a form kept on disk with what she worked out about this world."""
        def key(square: Square) -> str:
            return f"{square[0]},{square[1]}"

        def way(seen: _Seen) -> dict:
            return {"offered": sorted(seen.offered), "offered_unmoved": sorted(seen.offered_unmoved),
                    "withheld": sorted(seen.withheld), "withheld_unmoved": sorted(seen.withheld_unmoved),
                    "took": seen.took, "would_not_take": seen.would_not_take,
                    "moved_to_empty": seen.moved_to_empty, "jumped": seen.jumped}

        return {
            "rows": self.rows, "columns": self.columns,
            "start": {key(sq): [side, kind] for sq, (side, kind) in self.start.items()},
            "seen": {kind: {f"{d[0]},{d[1]}": way(w) for d, w in ways.items()} for kind, ways in self.seen.items()},
            "kept_off_attacked": dict(self.kept_off_attacked), "went_onto_attacked": dict(self.went_onto_attacked),
            "crosses": dict(self.crosses), "becomes": {k: list(v) for k, v in self.becomes.items()},
            "passes": sorted(self.passes), "takes_passing": sorted(self.takes_passing),
            "refusals": self.refusals, "refused_leaving_open": dict(self.refused_leaving_open),
            "taken_and_played_on": dict(self.taken_and_played_on),
        }

    @classmethod
    def from_memory(cls, kept: dict) -> HowThePiecesMove | None:
        try:
            def square(text: str) -> Square:
                row, column = text.split(",")
                return int(row), int(column)

            learned = cls(rows=int(kept["rows"]), columns=int(kept["columns"]),
                          start={square(k): (int(v[0]), str(v[1])) for k, v in kept["start"].items()})
            for kind, ways in kept.get("seen", {}).items():
                for d, w in ways.items():
                    seen = learned._of(kind, square(d))
                    seen.offered, seen.offered_unmoved = set(w["offered"]), set(w["offered_unmoved"])
                    seen.withheld, seen.withheld_unmoved = set(w["withheld"]), set(w["withheld_unmoved"])
                    seen.took, seen.would_not_take = int(w["took"]), int(w["would_not_take"])
                    seen.moved_to_empty, seen.jumped = int(w["moved_to_empty"]), int(w["jumped"])
                learned.seen.setdefault(kind, {})
            learned.kept_off_attacked = {k: int(v) for k, v in kept.get("kept_off_attacked", {}).items()}
            learned.went_onto_attacked = {k: int(v) for k, v in kept.get("went_onto_attacked", {}).items()}
            learned.crosses = dict(kept.get("crosses", {}))
            learned.becomes = {k: list(v) for k, v in kept.get("becomes", {}).items()}
            learned.passes, learned.takes_passing = set(kept.get("passes", [])), set(kept.get("takes_passing", []))
            learned.refusals = int(kept.get("refusals", 0))
            learned.refused_leaving_open = {k: int(v) for k, v in kept.get("refused_leaving_open", {}).items()}
            learned.taken_and_played_on = {k: int(v) for k, v in kept.get("taken_and_played_on", {}).items()}
            return learned
        except (KeyError, TypeError, ValueError) as why:
            logger.info("what she learned of this game could not be read back (%s); learning it afresh", why)
            return None

    def royal(self, kind: str) -> bool:
        """Whether the game keeps this kind safe: left open by moves it withheld and almost never by moves it offered,
        and more often than any other kind; or left open by (nearly) every move it refused that the kind's own going
        does not explain, at least twice. Never a kind seen taken with the game going on after: that is the one thing a
        piece the game keeps safe cannot be, and it outweighs counts that a misreading of the board can inflate. LIVE 21
        November 2026 her record of En Croissant's board, kept through misreadings since mended, held the pawns' kind as
        the one to keep safe, and she called nearly every move an attack on it."""
        if self.taken_and_played_on.get(kind, 0):
            return False
        kept_off = self.kept_off_attacked.get(kind, 0)
        if kept_off >= ROYAL_AFTER and self.went_onto_attacked.get(kind, 0) <= AGAINST_AT_MOST * kept_off and all(
                other == kind or count < kept_off or self.went_onto_attacked.get(other, 0) > AGAINST_AT_MOST * count
                for other, count in self.kept_off_attacked.items()):
            return True
        left_open = self.refused_leaving_open.get(kind, 0)
        return self.refusals >= ROYAL_AFTER and left_open >= (1 - AGAINST_AT_MOST) * self.refusals and all(
            other == kind or count < left_open for other, count in self.refused_leaving_open.items())

    def family(self, *, unseen_go_every_way: bool = False) -> Family:
        """The game as she understands it now. With ``unseen_go_every_way``, for looking ahead: a kind she has not
        seen move is taken to go along every line, as far as the way is clear, so a piece of the other side she has not
        seen move yet is not taken for one that cannot reach her. What it does is checked against the game before she
        relies on it for a move of her own, and her reading of which kind is royal never uses it."""
        # What a kind may become, among kinds there are: on the board at the start, or seen to move. A kind that is
        # neither is a look she no longer reads that way.
        there = {k for _s, k in self.start.values()} | {k for k, ways in self.seen.items() if ways}
        kinds = {kind: Kind(kind, steps=tuple(self._steps_of(kind) or (
                                _EVERY_LINE if unseen_go_every_way else ())), royal=self.royal(kind),
                            becomes=tuple(b for b in self.becomes.get(kind, ()) if b in there),
                            crosses_with=self.crosses.get(kind, ""),
                            name=self.names.get(kind, ""))
                 for kind in sorted({k for _s, k in self.start.values()} | set(self.seen))}
        return Family(name="the game in front of me", rows=self.rows, columns=self.columns, kinds=kinds,
                      start=dict(self.start))

    def what_it_does(self, kind: str) -> str:
        """A kind as she has seen it go, in words, for naming it where she has no other name for it."""
        steps = self._steps_of(kind)
        if not steps:
            return "a piece I haven't seen move"
        lines = {s.offsets[0] for s in steps if s.rides != 1}
        leaps = {s.offsets[0] for s in steps if s.rides == 1}
        if leaps and all(abs(f) + abs(r) == 3 and f and r for f, r in leaps):
            return "the piece that leaps two-and-one"
        straight = {d for d in lines if 0 in d}
        slanted = {d for d in lines if 0 not in d}
        if straight and slanted:
            return "the piece that runs any way"
        if slanted:
            return "the piece that runs on the slant"
        if straight:
            return "the piece that runs straight"
        if self.royal(kind):
            return "the piece that must be kept safe"
        if all(f > 0 for f, _r in leaps | lines):
            return "the piece that steps forward"
        return "the piece that steps one square"


def _mirrored(steps: list[Step], ways: dict[Direction, _Seen]) -> list[Step]:
    """Each step and its left-right image, unless the image's way has been seen and not gone."""
    out = list(steps)
    have = {(s.offsets[0], s.only_unmoved) for s in steps}
    for step in steps:
        forward, right = step.offsets[0]
        image = (forward, -right)
        if (image, step.only_unmoved) in have:
            continue
        way = ways.get(image)
        if way is not None and (way.withheld or way.withheld_unmoved) and not way.any_seen_going():
            continue
        out.append(Step((image,), rides=step.rides, may=step.may, only_unmoved=step.only_unmoved,
                        leaves_passable=step.leaves_passable, takes_in_passing=step.takes_in_passing))
        have.add((image, step.only_unmoved))
    return out


def a_move_seen(before: Position, after_cells: dict[Square, tuple[int, str | None]], side: int) -> Move | None:
    """The move of ``side`` that turned ``before`` into what is seen, read from the squares that changed, where the
    rules she holds cannot say it: a piece gone from one square and come to another, a piece of the other side gone
    from the square it came to or from beside it, two pieces of the side moving at once."""
    gone = [sq for sq, (s, _k) in before.cells.items() if s == side and after_cells.get(sq, (None,))[0] != side]
    came = [sq for sq, (s, _k) in after_cells.items() if s == side and before.cells.get(sq, (None,))[0] != side]
    # The other side's pieces gone: taken, unless as many came somewhere new, which is the other side's own move.
    their_gone = [sq for sq, (s, _k) in before.cells.items() if s != side and after_cells.get(sq, (None,))[0] != 1 - side]
    their_came = [sq for sq, (s, _k) in after_cells.items() if s != side and before.cells.get(sq, (None,))[0] != s]
    lost = their_gone if len(their_gone) > len(their_came) else []
    if len(gone) == 1 and len(came) == 1:
        start, end = gone[0], came[0]
        if end in before.cells and before.cells[end][0] != side:
            taken_at: Square | None = end
        elif lost:
            taken_at = min(lost, key=lambda sq: abs(sq[0] - end[0]) + abs(sq[1] - end[1]))
        else:
            taken_at = None
        letter_before = before.cells[start][1]
        letter_after = after_cells[end][1]
        becomes = letter_after if letter_after is not None and letter_after != letter_before else ""
        return Move(start, end, becomes=becomes, taken_at=taken_at)
    if len(gone) == 2 and len(came) == 2:
        # Two pieces at once: the one that went further is the one that moved, the other its partner.
        pairs = [(a, b) for a in gone for b in came]
        best = None
        for a, b in pairs:
            other_a = next(g for g in gone if g != a)
            other_b = next(c for c in came if c != b)
            if before.cells[a][1] == after_cells[b][1] and before.cells[other_a][1] == after_cells[other_b][1]:
                spread = abs(a[1] - b[1]) + abs(a[0] - b[0])
                if best is None or spread > best[0]:
                    best = (spread, a, b, other_a, other_b)
        if best is not None:
            _spread, a, b, other_a, other_b = best
            return Move(a, b, partner=(other_a, other_b))
    return None


def moves_seen(before: Position, after_cells: dict[Square, tuple[int, str | None]], side: int,
               family: Family) -> list[Move] | None:
    """The moves that turned ``before`` into what is seen, ``side`` first: one move, or that side's move and the
    other side's answer, made before she looked. Each is made on the board as she holds it and kept only where it makes
    exactly what is seen; where more than one does, the one the rules she has learned allow is believed, and one in
    which nothing becomes another kind before one in which something does. LIVE-like 20 November 2026 a knight that took
    on a square and was taken back there left only two squares emptied, and "the knight took the far rook and a pawn
    became a rook there" fitted the picture as well as what happened. None where nothing she can read fits."""
    from core.agency.pieces_by_their_kind import after
    from core.agency.pieces_by_their_kind import moves as legal_moves

    def made_shows(position: Position) -> bool:
        return all(after_cells.get(sq, (None, None))[0] == s and after_cells[sq][1] in (k, None)
                   for sq, (s, k) in position.cells.items()) and set(position.cells) == set(after_cells)

    def allowed(position: Position, move: Move) -> bool:
        return any((m.start, m.end) == (move.start, move.end) for m in legal_moves(family, position))

    one = a_move_seen(before, after_cells, side)
    if one is not None and made_shows(after(family, before, one)):
        return [one]
    gone = [sq for sq, (s, _k) in before.cells.items() if s == side and after_cells.get(sq, (None,))[0] != side]
    landed = [sq for sq in after_cells if before.cells.get(sq, (None,))[0] != side]
    fitting: list[tuple[tuple[int, int, int], list[Move]]] = []
    for start in gone:
        for end in landed:
            if end == start:
                continue
            taken = end if end in before.cells and before.cells[end][0] != side else None
            mine = Move(start, end, taken_at=taken)
            middle = after(family, before, mine)
            answer = a_move_seen(middle, after_cells, 1 - side)
            if answer is not None and made_shows(after(family, middle, answer)):
                rank = (int(not allowed(before, mine)), int(not allowed(middle, answer)), int(bool(answer.becomes)))
                fitting.append((rank, [mine, answer]))
    if not fitting:
        return None
    return min(fitting, key=lambda pair: pair[0])[1]
