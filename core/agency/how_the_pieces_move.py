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
other kind; a take on an empty square that removes the piece beside it is a take in passing. A square offered or played
beyond exactly one piece of the other side, which goes, is a take over; several pieces gone in one move, along straight
lines from one landing to the next, are take overs one after another; a game that keeps back a piece's quiet moves
whenever a take is there takes first; and a game that ends when a side has no move, read as lost, is one in which having
no move loses. Nothing here knows a game.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field, replace
from typing import Any

from core.agency.pieces_by_their_kind import (
    CROSSING_STEP,
    FIRST,
    SECOND,
    Family,
    Kind,
    Move,
    Position,
    Step,
    attacked,
)

logger = logging.getLogger("Aura.HowThePiecesMove")

__all__ = ["HowThePiecesMove", "a_move_seen", "in_its_frame", "moves_seen", "turned_round_from"]

Square = tuple[int, int]
Direction = tuple[int, int]


def turned_round_from(start: dict[Square, tuple[int, str]], rows: int, columns: int) -> dict[Square, tuple[int, str]]:
    """A board's pieces as the other side sees them: her rows count from her own back row and her columns from her own
    left, so the board turns half round and each side's pieces become the other's."""
    return {(rows - 1 - row, columns - 1 - column): (SECOND if side == FIRST else FIRST, kind)
            for (row, column), (side, kind) in start.items()}

#: Times a royal kind must be seen kept off squares the other side could take on before it is taken to be royal.
ROYAL_AFTER = 2
#: How rare the evidence against a reading may be and the reading still stand: what she sees passes through what she
#: holds of the rest of the game, which is itself still being learned, so a reading that one sighting in ten
#: contradicts is the game's and the sighting is her model's.
AGAINST_AT_MOST = 0.1
#: The longest single step looked for along a way: a leap further than three squares is not looked for.
MAX_STEP = 3


#: Every line from a square, as far as the way is clear: what a kind she has not seen move might do.
_EVERY_LINE = (Step(((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)), rides=0, sure=False),)
#: Sightings of a kind going a way, offered by the game or seen made, below which the way is not relied on: one
#: misread look can show two. LIVE 23 November 2026 one look gave her royal kind two squares far up the board.
SEEN_TO_BELIEVE = 3
#: Which reading of a board's answers gathered what is kept: 2 since an answer is read from the board as it was, once
#: it holds still (f60209bb0); 3 since a change of kind is learned only once a second look shows it; 4 since a take the
#: game did not offer counts against taking that way only where nothing else explains it (e9745c00a). What an earlier
#: reading gathered weighs as kept knowledge does when it comes back (core/runtime/what_she_learned.py,
#: TRUST_CARRIED_OVER), once, the first time the record is read after.
READING = 4
#: What one look counts for against SEEN_TO_BELIEVE, which counted squares: three squares were to make sure of two
#: looks, since one misread look shows two at most, so two looks that agree are what it asked for.
A_LOOK_IS_WORTH = SEEN_TO_BELIEVE / 2


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


#: The counts of a way kept by distance.
_COUNTED = ("went_at", "took_at", "refused_going_at", "refused_taking_at", "looks_at", "before_looks_at", "jumped_at")


def _summed(one: dict[int, int], other: dict[int, int]) -> dict[int, int]:
    return {t: one.get(t, 0) + other.get(t, 0) for t in set(one) | set(other)}


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
    #: Distances it was offered or seen to go to an empty square beyond exactly one piece of the other side; how often
    #: that piece then went; and how often the game refused such a move.
    over: set[int] = field(default_factory=set)
    over_took: int = 0
    over_refused: int = 0
    #: Distances it was offered, or seen to go, onto a square held by a piece of its own side, the two changing places.
    swapped: set[int] = field(default_factory=set)
    #: For each distance, how many times it was offered or seen gone to an empty square and onto a piece of the other
    #: side, and how many times the game refused her a move that far of each sort (a refusal that what the move left
    #: open does not explain). The game's word against the game's word: a distance stands while going or taking that
    #: far was offered more often than refused. LIVE 23 November 2026, game 26, one misread offer gave her bishop a leap
    #: six forward and four across, and the game refused it four times without it going.
    went_at: dict[int, int] = field(default_factory=dict)
    took_at: dict[int, int] = field(default_factory=dict)
    refused_going_at: dict[int, int] = field(default_factory=dict)
    refused_taking_at: dict[int, int] = field(default_factory=dict)
    #: For each distance, how many separate looks (an offer read, a move seen made) showed this way going that far and
    #: no further: what it rests on is counted by looks, not by squares, since one misread look can show several
    #: (suggestions of 23 November 2026: one corrupted look with three far squares met a threshold of three). What was
    #: counted by squares before looks were counted is kept as it was, in ``before_looks_at``.
    looks_at: dict[int, int] = field(default_factory=dict)
    before_looks_at: dict[int, int] = field(default_factory=dict)
    #: For each distance, how often this way was offered, or seen gone, that far over a piece standing between: whether
    #: a far distance passes over what stands between is what was seen at that distance. LIVE 23 November 2026 one
    #: misread look had her royal kind take three squares forward over a piece, and its far distances, six and seven,
    #: each read once with nothing seen between, became leaps over every piece on the board.
    jumped_at: dict[int, int] = field(default_factory=dict)

    def _counted(self, times: int, taking: bool) -> int:
        counts = self.took_at if taking else self.went_at
        if times in counts:
            return counts[times]
        # A distance kept before offers were counted was offered once, of whichever sort the way was seen doing.
        kept = times in self.offered or times in self.offered_unmoved
        return int(kept and not (self.took_at or self.went_at) and bool(self.took if taking else self.moved_to_empty))

    def sightings(self, distances: Any = None) -> float:
        """How often this way was offered or seen gone, at ``distances`` (all it was seen at, where None), counted by
        squares: what her search relies on. A run of squares in one look is good evidence of a line, and squares
        beyond a gap are kept as leaps of their own distance (``HowThePiecesMove._reach``)."""
        at = set(self.went_at) | set(self.took_at) | self.offered | self.offered_unmoved if distances is None \
            else set(distances)
        return sum(self._by_squares(t) for t in at)

    def looks(self, distances: Any = None) -> float:
        """How many separate looks showed this way going as far as the nearest of ``distances`` or further, each
        counting for A_LOOK_IS_WORTH sightings, with what was counted by squares before looks were: what a move is
        checked by before it is made. One misread look is one look, however many squares it shows."""
        if not self.looks_at:
            return self.sightings(distances)
        nearest = min(distances) if distances else 0
        before = sum(n for t, n in self.before_looks_at.items() if distances is None or t in set(distances))
        return before + A_LOOK_IS_WORTH * sum(n for t, n in self.looks_at.items() if t >= nearest)

    def _by_squares(self, times: int) -> int:
        """How often this distance was counted before looks were: a distance kept as offered before distances were
        counted at all was offered at least once."""
        counted = self._counted(times, False) + self._counted(times, True)
        return max(counted, int(times in self.offered or times in self.offered_unmoved))

    def a_look(self, furthest: int) -> None:
        """One look that showed this way going ``furthest`` and no further. The first keeps what was counted by
        squares before it."""
        if not self.looks_at:
            every = set(self.went_at) | set(self.took_at) | self.offered | self.offered_unmoved
            self.before_looks_at = {t: n for t in every if (n := self._by_squares(t))}
        self.looks_at[furthest] = self.looks_at.get(furthest, 0) + 1

    def _stands(self, times: int) -> bool:
        return (self._counted(times, False) > self.refused_going_at.get(times, 0)
                or self._counted(times, True) > self.refused_taking_at.get(times, 0))

    def an_offer(self, times: int, unmoved: bool, *, taking: bool) -> None:
        """The game offered this distance, or it was seen gone, onto a piece of the other side where ``taking``."""
        went, took = self._counted(times, False), self._counted(times, True)
        self.went_at[times], self.took_at[times] = went + (not taking), took + taking
        if self._stands(times):
            (self.offered_unmoved if unmoved else self.offered).add(times)

    def a_refusal(self, times: int, *, taking: bool) -> None:
        """The game refused her a move this far: where neither going nor taking that far was offered more often than
        refused, it is not a distance this kind goes."""
        self.went_at[times], self.took_at[times] = self._counted(times, False), self._counted(times, True)
        refused = self.refused_taking_at if taking else self.refused_going_at
        refused[times] = refused.get(times, 0) + 1
        if not self._stands(times):
            self.offered.discard(times)
            self.offered_unmoved.discard(times)

    def went_to_empty(self) -> bool:
        return self.moved_to_empty > 0

    def any_seen_going(self) -> bool:
        return bool(self.offered or self.offered_unmoved or self.took)

    def together_with(self, other: _Seen) -> _Seen:
        """What was seen of this way and of another taken as seen of one."""
        return _Seen(offered=self.offered | other.offered, offered_unmoved=self.offered_unmoved | other.offered_unmoved,
                     withheld=self.withheld | other.withheld, withheld_unmoved=self.withheld_unmoved | other.withheld_unmoved,
                     took=self.took + other.took, would_not_take=self.would_not_take + other.would_not_take,
                     moved_to_empty=self.moved_to_empty + other.moved_to_empty, jumped=self.jumped + other.jumped,
                     over=self.over | other.over, over_took=self.over_took + other.over_took,
                     over_refused=self.over_refused + other.over_refused, swapped=self.swapped | other.swapped,
                     went_at=_summed(self.went_at, other.went_at), took_at=_summed(self.took_at, other.took_at),
                     refused_going_at=_summed(self.refused_going_at, other.refused_going_at),
                     refused_taking_at=_summed(self.refused_taking_at, other.refused_taking_at),
                     looks_at=_summed(self.looks_at, other.looks_at),
                     before_looks_at=_summed(self.before_looks_at, other.before_looks_at),
                     jumped_at=_summed(self.jumped_at, other.jumped_at))


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
    #: Whether her side moved first when she sat down to ``start``: which way round she sees the board, since her
    #: rows count from her own back row. None where it was not recorded.
    first_was_hers: bool | None = None
    #: Kinds seen to take over again in the same move.
    takes_again: set[str] = field(default_factory=set)
    #: Her pieces picked up while her side had a take, by the rules she holds: how often the game kept back every quiet
    #: move the piece had, and how often it offered one.
    quiet_kept_back_while_taking: int = 0
    quiet_offered_while_taking: int = 0
    #: Whether a game of this board ended lost for a side that had no move, as its window said.
    no_move_loses: bool = False
    #: Kinds whose arrival on their far row ended a game, won for their side, as its window said.
    wins_arriving: set[str] = field(default_factory=set)
    #: Moves seen that the rules she held did not allow, by kind, way and distance, waiting to be seen again before they
    #: teach her anything: a board misread once explained by a move nobody made taught her men to step sideways.
    unconfirmed: dict[str, int] = field(default_factory=dict)
    #: Every answer the game gave, kept whole so her rules can be checked against all of it
    #: (core/agency/what_the_game_showed.py); kept beside the record, not in it.
    shown: Any = field(default=None, repr=False, compare=False)
    #: How often each kind was seen, or offered, making a move of each sort that is not a plain step: crossing with
    #: a partner, changing places, taking over, taking in passing. What a rule allowing such a move rests on is how
    #: often it was seen, not that the rule can express it (suggestions of 23 November 2026).
    special_seen: dict[str, int] = field(default_factory=dict)
    #: Which reading gathered what is kept here (READING).
    read_by: int = READING
    #: Takes counted as not offered by a reading before READING 4, to be read again against the answers kept once they
    #: are attached (``withheld_takes_read_again``).
    withheld_takes_unchecked: bool = field(default=False, repr=False, compare=False)

    def _saw_special(self, kind: str, sort: str) -> None:
        key = f"{kind}|{sort}"
        self.special_seen[key] = self.special_seen.get(key, 0) + 1

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
        if self.shown is not None:
            self.shown.offered(position, square, squares)
        side, kind = position.cells[square]
        unmoved = square in position.unmoved
        # A partner's square offered to a piece her rules have crossing with it there is the crossing offered: a game
        # may make a crossing by the mover taken onto its partner. LIVE 23 November 2026 En Croissant's offer of the
        # rook's square to her king was kept as the two changing places, a move she then found open to her king in
        # check, where the crossing is not.
        crossing_to = set()
        crossing_ends: set[Square] = set()
        if family is not None:
            from core.agency.pieces_by_their_kind import _possible

            # A crossing kept as a pair, not yet relied on, is asked of this offer too: the offer is how it comes to be.
            candidate = self.crosses.get(kind)
            if candidate and kind in family.kinds and not family.kinds[kind].crosses_with:
                family = replace(family, kinds={**family.kinds,
                                                kind: replace(family.kinds[kind], crosses_with=candidate)})
            crossings = [move for move in _possible(family, position)
                         if move.start == square and move.partner is not None and move.partner[0] != move.end]
            crossing_to = {move.partner[0] for move in crossings}
            # The square a crossing of hers lands on, offered, is the crossing offered too, not a plain way that far:
            # the plain way outlives what the crossing needs. LIVE 23-24 November 2026 her royal kind held two squares
            # sideways from where it began as a plain way, learned from the crossing's offers, and her search could
            # still "cross" after its partner had moved: giving the crossing up looked free.
            crossing_ends = {move.end for move in crossings}
            if (crossing_to | crossing_ends) & set(squares):
                self._saw_special(kind, "crosses")
        # This look, as one look: how far it showed each way going.
        shown: dict[Direction, int] = {}
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
                    is_offered = there in squares and there not in crossing_ends
                    if held is None:
                        if is_offered:
                            way = self._of(kind, direction)
                            way.an_offer(times, unmoved, taking=False)
                            shown[direction] = max(shown.get(direction, 0), times)
                            way.moved_to_empty += 1
                        elif squares and kind in self.seen and direction in self.seen[kind]:
                            # An offer of nothing may be a piece not picked up, or a board not yet taking moves
                            # (LIVE 23 November 2026, a game's setup board, picked up piece by piece): it withholds
                            # nothing.
                            way = self.seen[kind][direction]
                            (way.withheld_unmoved if unmoved else way.withheld).add(times)
                    else:
                        if held[0] != side:
                            if is_offered:
                                way = self._of(kind, direction)
                                way.took += 1
                                way.an_offer(times, unmoved, taking=True)
                                shown[direction] = max(shown.get(direction, 0), times)
                            elif self._goes(kind, direction, times, unmoved) and squares and not self._kept_safe_explains(
                                    position, Move(square, there, taken_at=there), family):
                                # A take the game did not offer counts against taking that way only where nothing
                                # else explains it: leaving the kind it keeps safe where it can be taken is reason
                                # enough. LIVE 24 November 2026 her royal kind was held never to take on its right,
                                # from takes withheld because the square was defended.
                                self.seen[kind][direction].would_not_take += 1
                        elif is_offered and there not in crossing_to:
                            # A square of its own side offered: the two change places.
                            self._of(kind, direction).swapped.add(times)
                            self._saw_special(kind, "swaps")
                        break
                    times += 1
        if family is not None:
            self._what_the_offer_keeps_safe(position, square, squares, family)
            self._what_the_offer_says_of_taking_first(position, square, squares, family)
            self._what_the_offer_withholds(position, square, squares, family)
            self._what_the_offer_shows_of_their_reach(position, square, squares, family)
        # A square offered with a piece on the way to it was leapt to: over exactly one piece of the other side onto an
        # empty square, it is a take over, unless the same distance is offered with nothing in the way. Any other is a
        # leap that far, and the game offering it is evidence the kind goes there, as much as a square with nothing on
        # the way: the walk along each line above stops at the first piece and never reaches it. LIVE-like 22 November
        # 2026 a kind that leaps two squares over its own pieces was offered its leap again and again, learned nothing
        # from it, and went on being searched as a kind that goes every way, until the game refused its moves.
        for there in squares:
            direction, times = in_its_frame(side, square, there)
            between = [_square_along(side, square, direction, t) for t in range(1, times)]
            standing = [b for b in between if b in position.cells]
            if len(standing) == 1 and position.cells[standing[0]][0] != side and there not in position.cells:
                self._of(kind, direction).over.add(times)
            elif standing:
                way = self._of(kind, direction)
                way.jumped += 1
                way.jumped_at[times] = way.jumped_at.get(times, 0) + 1
                way.an_offer(times, unmoved, taking=there in position.cells)
                shown[direction] = max(shown.get(direction, 0), times)
                if there in position.cells and position.cells[there][0] != side:
                    way.took += 1
                elif there not in position.cells:
                    way.moved_to_empty += 1
        for direction, furthest in shown.items():
            self._of(kind, direction).a_look(furthest)

    def _what_the_offer_withholds(self, position: Position, square: Square, squares: set[Square],
                                  family: Family) -> None:
        """The moves her rules give this piece that the game did not offer, each the game's word against the distance it
        goes, as a refusal is, unless it would leave the kind the game keeps safe open: that is reason enough to
        withhold it. LIVE 23 November 2026, after game 29: her rules had her royal kind leap two forward and three
        across, and ride forward the whole board, each on one offer seen once, and none of the times the game withheld
        those squares counted against them. Her search gave her king, and theirs, ways out of attacks that were none.

        Until she knows which kind the game keeps safe, a move withheld may be withheld for it, and says nothing; an
        offer of nothing may be a piece not picked up, and says nothing. A distance the game has offered this kind only
        unmoved, withheld once it has moved, is one only an unmoved piece of it has; withheld from it unmoved, it is
        counted against nothing, since what such a move needs besides is not in her rules."""
        from core.agency.pieces_by_their_kind import _possible, after

        side, kind = position.cells[square]
        if position.turn != side or not squares:
            return
        kept_safe = {k for k in {k for _s, k in self.start.values()} | set(self.seen) if self.royal(k)}
        if not kept_safe:
            return
        unmoved = square in position.unmoved
        weighed: set[Square] = set()
        for move in _possible(family, position):
            swap = move.partner is not None and move.partner[0] == move.end
            if (move.start != square or move.end in squares or move.end in weighed
                    or (move.partner is not None and not swap) or move.via or move.taken_at not in (None, move.end)):
                continue
            weighed.add(move.end)
            made = after(family, position, move)
            # Of two explanations for a withholding, it counts against the one resting on less: a way she is not sure
            # of is less than what the move would leave open (as a refusal is weighed, in refused). LIVE 24 November
            # 2026 her royal kind's three-square jumps outlived every look that withheld them, each put down to the
            # square it would land on being attacked.
            if self.sure_of(position, move, family) and any(
                    owner == side and held in kept_safe and attacked(family, made, there, 1 - side)
                    for there, (owner, held) in made.cells.items()):
                continue
            direction, times = in_its_frame(side, square, move.end)
            way = self._of(kind, direction)
            if swap:
                # Two pieces changing places, withheld: seen once, it was most likely a crossing offered, or misread.
                way.swapped.discard(times)
                continue
            if times in way.offered_unmoved:
                if not unmoved:
                    way.offered.discard(times)
                continue
            way.a_refusal(times, taking=move.taken_at is not None)

    def _what_the_offer_shows_of_their_reach(self, position: Position, square: Square, squares: set[Square],
                                             family: Family) -> None:
        """A move the game offered that leaves her kind it keeps safe where, by her rules, the other side could take it:
        the game would not have offered it. Where every way of theirs making that attack is one she is not sure of, the
        offer is the game's word against each of those takes that far, once a look, as a refused take is. Where a way
        she is sure of makes it, her reading of the board is what is in doubt, and nothing is counted.

        LIVE 23 November 2026 her rules had their royal kind take hers from across the board at the start, and the game
        offered her pawns' moves, each of which left hers so attacked; none of those offers counted against the take."""
        from core.agency.pieces_by_their_kind import _reach, after

        side = position.cells[square][0]
        if position.turn != side or not squares:
            return
        kept_safe = {k for k in {k for _s, k in self.start.values()} | set(self.seen) if self.royal(k)}
        if not kept_safe:
            return
        against: set[tuple[str, Direction, int]] = set()
        # The moves as the game offered them, whether or not her rules have them yet: a piece of hers onto an empty
        # square or one of theirs (a square of her own side offered is a move of another sort, and says nothing here).
        for end in squares:
            held = position.cells.get(end)
            if held is not None and held[0] == side:
                continue
            made = after(family, position, Move(square, end, taken_at=end if held is not None else None))
            royals = {there for there, (owner, held) in made.cells.items() if owner == side and held in kept_safe}
            attacks: list[tuple[str, Direction, int, bool]] = []
            for there, (owner, letter) in made.cells.items():
                if owner == side or letter not in family.kinds:
                    continue
                for end, step, _passed in _reach(family, made, there, owner, family.kinds[letter], only_takes=True):
                    if end in royals and step.may not in ("moves", "swaps"):
                        attacks.append((letter, *in_its_frame(owner, there, end), step.sure))
            if attacks and not any(sure for *_way, sure in attacks):
                against |= {(letter, direction, times) for letter, direction, times, _sure in attacks}
        for letter, (forward, right), times in against:
            ways = self.seen.get(letter, {})
            way = ways.get((forward, right)) or ways.get((forward, -right))
            if way is not None:
                way.a_refusal(times, taking=True)

    def _what_the_offer_says_of_taking_first(self, position: Position, square: Square, squares: set[Square],
                                             family: Family) -> None:
        """Where her side had a take by the rules she holds, whether the game offered this piece a quiet move it has."""
        from core.agency.pieces_by_their_kind import _possible

        side = position.cells[square][0]
        if position.turn != side:
            return
        every = _possible(replace(family, takes_first=False), position)
        # Only a take she has seen made, or offered, counts: one her rules allow on no more than having seen the kind go
        # that way to an empty square may be no take at all, and the game offering a quiet move beside it says nothing.
        if not any(self._seen_taking(position, move) for move in every):
            return
        quiet = [move for move in every if move.start == square and move.taken_at is None and move.partner is None]
        if not quiet:
            return
        if any(move.end in squares for move in quiet):
            self.quiet_offered_while_taking += 1
        else:
            self.quiet_kept_back_while_taking += 1

    def _seen_taking(self, position: Position, move: Move) -> bool:
        if not move.taken():
            return False
        side, kind = position.cells[move.start]
        direction, _times = in_its_frame(side, move.start, (move.via or (move.end,))[0])
        way = self.seen.get(kind, {}).get(direction)
        return way is not None and (way.over_took > 0 or bool(way.over) if move.via or move.taken_at != move.end
                                    and _takes_over_seen(side, move) else way.took > 0)

    def the_window_said(self, ended: str, final: Position, last: Move | None) -> None:
        """A game that stopped with no reason her rules could give, and the window saying who won (``ended`` "won" or
        "lost" for her, the first side). Where the loser was to move and her rules give it no move, a side with no move
        loses: that is the reason her rules already have. Only where they have none, and the winner's last move brought
        a piece to its far row as the kind it was, did that kind's arrival win."""
        from core.agency.pieces_by_their_kind import moves

        if ended not in ("won", "lost"):
            return
        winner = FIRST if ended == "won" else SECOND
        # A kind never seen to go is taken to go every way: knowing nothing of it is no reason it could not move.
        if final.turn != winner and not moves(self.family(unseen_go_every_way=True), final):
            self.no_move_loses = True
            return
        far = self.rows - 1 if winner == FIRST else 0
        arrived = final.cells.get(last.end) if last is not None else None
        if (arrived is not None and arrived[0] == winner and last.end[0] == far and last.start[0] != far
                and not last.becomes):
            self.wins_arriving.add(arrived[1])

    def kinds_of(self, side: int) -> set[str]:
        """The kinds a side has held: those it began with, and what they have been seen to become."""
        kinds = {kind for owner, kind in self.start.values() if owner == side}
        more = kinds
        while more:
            more = {b for kind in more for b in self.becomes.get(kind, [])} - kinds
            kinds |= more
        return kinds

    def takes_first(self) -> bool:
        """Whether the game makes a side that can take do so: its quiet moves kept back whenever a take was there."""
        kept = self.quiet_kept_back_while_taking
        return kept >= ROYAL_AFTER and self.quiet_offered_while_taking <= AGAINST_AT_MOST * kept

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

    def watched(self, position: Position, move: Move, *, allowed: bool = True) -> None:
        """A move seen made: the kind went there, and what happened with it tells what more it may do. A move the rules
        she holds did not allow (``allowed`` False) teaches her a new way only once it is seen again, or the game has
        offered it: a board read wrongly once can be explained by a move nobody made."""
        side, kind = position.cells[move.start]
        if not allowed and move.partner is None and not move.becomes:
            direction, times = in_its_frame(side, move.start, move.end)
            way = self.seen.get(kind, {}).get(direction)
            # A take over offered at a distance shows nothing of a plain move that far, nor a plain move of a take over.
            shown_before = way is not None and times in (way.over if _takes_over_seen(side, move)
                                                         else way.offered | way.offered_unmoved)
            key = f"{kind}|{direction[0]},{direction[1]}|{times}|{_sort_of(side, move)}"
            if not shown_before:
                self.unconfirmed[key] = self.unconfirmed.get(key, 0) + 1
                if self.unconfirmed[key] < 2:
                    return
        if self.last_taken:
            # A move after a piece was taken: the game went on without it.
            self.taken_and_played_on[self.last_taken] = self.taken_and_played_on.get(self.last_taken, 0) + 1
            self.last_taken = ""
        if move.taken_at is not None and move.taken_at in position.cells:
            self.last_taken = position.cells[move.taken_at][1]
        direction, times = in_its_frame(side, move.start, move.end)
        unmoved = move.start in position.unmoved
        way = self._of(kind, direction)
        if move.partner is not None and move.partner == (move.end, move.start):
            # Two of a side changed places: the one that moved goes that way onto its own side's piece.
            self._of(kind, direction).swapped.add(times)
            self._saw_special(kind, "swaps")
            return
        if move.partner is not None:
            # Seen made, a move that changes two things says which two changed and not which of them made it: both
            # readings fit what was seen. It is kept as a pair and credited to neither; the game offering it to one of
            # them, when that one is picked up, says which (offered). LIVE 24 November 2026 eight crossings watched
            # had been credited to the partner, and her straight-runner was held to cross with her royal kind.
            partner_kind = position.cells[move.partner[0]][1]
            self.crosses.setdefault(kind, partner_kind)
            return
        if _takes_over_seen(side, move):
            self._saw_special(kind, "takes over")
            # Each landing from the last, over a piece that went: a take over, and more than one a take again.
            path = (move.start, *move.via, move.end)
            for a, b in zip(path, path[1:], strict=False):
                hop, length = in_its_frame(side, a, b)
                hopped = self._of(kind, hop)
                hopped.over.add(length)
                hopped.over_took += 1
            if move.via:
                self.takes_again.add(kind)
            self._becomes(kind, move)
            return
        between = [_square_along(side, move.start, direction, t) for t in range(1, times)]
        if any(b in position.cells for b in between):
            way.jumped += 1
            way.jumped_at[times] = way.jumped_at.get(times, 0) + 1
        way.an_offer(times, unmoved, taking=move.taken_at is not None)
        # A move seen made is one look at the way it went.
        way.a_look(times)
        if move.taken_at is not None:
            way.took += 1
            if move.taken_at != move.end:
                self.takes_passing.add(kind)
                self._saw_special(kind, "takes in passing")
                victim = position.cells[move.taken_at][1]
                self.passes.add(victim)
        else:
            way.moved_to_empty += 1
        self._becomes(kind, move)

    def watched_becoming(self, kind: str, became: str) -> None:
        """A piece of ``kind`` seen to have become ``became`` where its move ended."""
        self._becomes(kind, Move((0, 0), (0, 0), becomes=became))

    def _becomes(self, kind: str, move: Move) -> None:
        if move.becomes and move.becomes != kind:
            options = self.becomes.setdefault(kind, [])
            if move.becomes not in options:
                options.append(move.becomes)
            self.seen.setdefault(move.becomes, {})

    def refused(self, position: Position, move: Move, family: Family | None = None) -> None:
        """A move the game would not take. Where the rules she held allowed it, either the kind does not go there, or
        the move would leave a kind of hers open to be taken that the game will not let her leave open: the kind left
        open by every such refusal is royal."""
        if self.shown is not None:
            self.shown.refused(position, move)
        side, kind = position.cells[move.start]
        direction, times = in_its_frame(side, move.start, move.end)
        unmoved = move.start in position.unmoved
        way = self._of(kind, direction)
        if _takes_over_seen(side, move):
            way.over_refused += 1
            return
        if family is not None:
            from core.agency.pieces_by_their_kind import after

            made = after(family, position, move)
            open_kinds = {k for sq, (s, k) in made.cells.items() if s == side and attacked(family, made, sq, 1 - side)}
            # A take refused where the kind was only ever seen going to empty squares says it does not take that way:
            # what it leaves open explains a refusal only of a move the kind has been seen to make. LIVE-like 22 November
            # 2026 a draughts man refused a take onto the next square three times, each put down to the man left open.
            goes = self._goes(kind, direction, times, unmoved) and (move.taken_at is None or way.took > 0)
            # Once she knows which kind the game keeps safe, only leaving that kind open explains a refusal: in the
            # middle of a game nearly every move leaves something of hers attacked. LIVE 23 November 2026, game 26, four
            # refusals of a leap her bishop does not have were each put down to a pawn left attacked, and the leap
            # stayed.
            kept_safe = {k for k in {k for _s, k in self.start.values()} | set(self.seen) if self.royal(k)}
            if kept_safe and not open_kinds & kept_safe:
                open_kinds = set()
            # Of two explanations, the refusal counts against the one resting on less: a way she is not sure of is
            # less than what the move left open. LIVE 23 November 2026, measured on her record after game 31: her
            # royal kind's six squares forward, read once, refused six times in one game, each put down to the royal
            # kind left open where it landed, and chosen again the next turn.
            if open_kinds and goes and not self.sure_of(position, move, family):
                open_kinds = set()
            if open_kinds and goes:
                # The kind goes there, as she has seen: what stopped it is what the move left open.
                self.refusals += 1
                for open_kind in open_kinds:
                    self.refused_leaving_open[open_kind] = self.refused_leaving_open.get(open_kind, 0) + 1
                return
        way.a_refusal(times, taking=move.taken_at is not None)
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
        # A line is what a run of offers from the next square shows. A piece that goes past a square on its line can
        # stop on it, and every square it passes is offered in the same look. Offers beyond the run, with squares
        # never offered between, have two explanations: a line whose squares between were misread as not offered,
        # or a run as long as it is, with leaps that far or misreadings beyond it. The one that needs fewer squares
        # misread is taken, and the line where they need as many (one rule, not two). The far distances of a run are
        # kept by _steps_of as leaps she is not sure of. LIVE 23 November 2026 her royal kind, offered one square
        # forward many times and six and seven squares once, was held to ride the whole file: a line needs the four
        # squares between misread, a run of one the two far offers.
        run = 0
        while run + 1 in offered:
            run += 1
        # Against the line, too: squares short of the furthest that a look showed empty and not offered, which a line
        # would have offered (a corrupted look with three far squares, beyond squares other looks withheld, is not a
        # line: suggestions of 23 November 2026).
        held_back = way.withheld | (way.withheld_unmoved if unmoved else set())
        missed = sum(1 for t in range(run + 1, furthest) if t not in offered or t in held_back)
        if 0 < run < furthest and missed > sum(1 for t in offered if t > run):
            return run
        withheld = {t for t in (way.withheld | (way.withheld_unmoved if unmoved else set())) if t > furthest}
        if withheld:
            return furthest
        # A square short of the furthest that the game withheld or refused more often than it offered stops the line
        # before it: a piece that goes past a square can stop on it.
        short = [t for t in set(way.refused_going_at) | set(way.refused_taking_at) if 1 < t < furthest
                 and not way._stands(t)]
        if short:
            nearer = [t for t in offered if t < min(short)]
            if nearer:
                return max(nearer)
        if furthest >= 2 and not way.jumped:
            return 0
        return furthest

    def _kept_safe_explains(self, position: Position, move: Move, family: Family | None) -> bool:
        """Whether, by her rules, ``move`` would leave the kind the game keeps safe where the other side could take it:
        reason enough for the game not to offer it, whatever the way it goes by."""
        if family is None:
            return False
        from core.agency.pieces_by_their_kind import after

        side = position.cells[move.start][0]
        kept_safe = {k for k in {k for _s, k in self.start.values()} | set(self.seen) if self.royal(k)}
        if not kept_safe:
            return False
        made = after(family, position, move)
        return any(owner == side and held in kept_safe and attacked(family, made, there, 1 - side)
                   for there, (owner, held) in made.cells.items())

    def _crosses_surely(self, kind: str) -> bool:
        """Whether this kind crosses with a partner, seen often enough to be relied on. Two kinds each held to cross with
        the other are two explanations of the same moves, of which one is so: the one with more behind it stands."""
        partner = self.crosses.get(kind)
        if not partner:
            return False
        seen = self.special_seen.get(f"{kind}|crosses", 0)
        if self.crosses.get(partner) == kind and self.special_seen.get(f"{partner}|crosses", 0) >= seen:
            return False
        return seen >= SEEN_TO_BELIEVE

    def _sure_as_far_as(self, way: _Seen, rides: int) -> int:
        """How far along a way, of at most ``rides`` squares (0 the whole line), it can be relied on: up to the first
        distance the game has also withheld or refused, unless enough looks have shown it going that far or further;
        ``rides`` itself where nothing stops it. A run of squares in one look, with nothing against it, is a line."""
        furthest = rides if rides > 0 else max(self.rows, self.columns)
        held = 1
        while held < furthest:
            further = held + 1
            # The game's word against the game's word: what was seen there against what was withheld or refused there.
            against = (int(further in way.withheld or further in way.withheld_unmoved)
                       + way.refused_going_at.get(further, 0) + way.refused_taking_at.get(further, 0))
            if against and (way.looks([further]) < SEEN_TO_BELIEVE or way.looks([further]) <= against):
                break
            held = further
        return rides if held >= furthest else held

    def seen_going_there(self, position: Position, move: Move) -> int:
        """How often this kind has been offered, or seen to go, the way this move goes, as far as it goes or further:
        what her rules allowing it rests on. A crossing, two pieces changing places, and a take over or in passing rest
        on how often a move of that sort was seen of the kind: that her rules can express one is no evidence it was
        seen. A kind's way of taking over, seen before such moves were counted, rests on its own count of them."""
        side, kind = position.cells[move.start]
        if move.partner is not None:
            sort = "swaps" if move.partner == (move.end, move.start) else "crosses"
            return self.special_seen.get(f"{kind}|{sort}", 0)
        if move.via or move.taken_at not in (None, move.end):
            if move.taken_at is not None and move.taken_at != move.end and not move.via \
                    and not _takes_over_seen(side, move):
                return self.special_seen.get(f"{kind}|takes in passing", 0)
            ways = self._ways_of(kind)
            counted = sum(way.over_took for way in ways.values())
            return max(self.special_seen.get(f"{kind}|takes over", 0), counted)
        (forward, right), times = in_its_frame(side, move.start, move.end)
        ways = self._ways_of(kind)
        seen = 0
        for direction in {(forward, right), (forward, -right)}:
            way = ways.get(direction)
            if way is None:
                continue
            distances = set(way.went_at) | set(way.took_at) | way.offered | way.offered_unmoved
            seen = max(seen, sum(way._by_squares(t) for t in distances if t >= times))
        return seen

    def sure_of(self, position: Position, move: Move, family: Family | None = None) -> bool:
        """Whether the way her rules allow this move by is one she is sure of (``Step.sure``): a move by a way she is
        not sure of is checked before it is made. Far squares one look showed beyond a gap are leaps she is not sure of
        until looks agree (suggestions of 23 November 2026: one corrupted look with three far squares met the count
        of three sightings)."""
        from core.agency.pieces_by_their_kind import _reach

        side, kind = position.cells[move.start]
        family = family or self.family(unseen_go_every_way=True)
        held = family.kinds.get(kind)
        if held is None or move.partner is not None or move.via:
            return True
        by = [step for end, step, _passed in _reach(family, position, move.start, side, held) if end == move.end]
        return not by or any(step.sure for step in by)

    def knows_how_it_goes(self, kind: str) -> bool:
        """Whether she has seen this kind go anywhere: whether its steps are her own reading, not a guess."""
        return bool(self._steps_of(kind))

    def counterparts(self) -> dict[str, tuple[str, bool]]:
        """Kinds of the two sides that began on each other's squares, the board turned round or seen in a mirror, each
        with the other and whether its right is the other's left. Two sides that begin as images of each other play the
        same way, so what one side's kind was seen to do is evidence of how its image goes; where the two sides' kinds
        look alike they are one kind already. LIVE-like 22 November 2026 a board drew draughts men a little differently
        by colour, and her men never learned from theirs that a man takes again."""
        rows, columns = self.rows, self.columns
        squares: dict[tuple[int, str], set[Square]] = {}
        for square, (side, kind) in self.start.items():
            squares.setdefault((side, kind), set()).add(square)
        mine = {kind: at for (side, kind), at in squares.items() if side == FIRST}
        theirs = {frozenset(at): kind for (side, kind), at in squares.items() if side == SECOND}
        # Two kinds are images of each other only where each is a look of its own side alone: where both sides have a
        # kind's look, it is one kind on both. Turned round, a chess board puts one side's queen on the other's king.
        shared = set(mine) & set(theirs.values())
        for image, flips in (((lambda sq: (rows - 1 - sq[0], columns - 1 - sq[1])), False),
                             ((lambda sq: (rows - 1 - sq[0], sq[1])), True)):
            pairs = {kind: theirs.get(frozenset(image(sq) for sq in at)) for kind, at in mine.items()}
            if mine and all(pairs.values()) and all(other == kind or not {kind, other} & shared
                                                    for kind, other in pairs.items()):
                out: dict[str, tuple[str, bool]] = {}
                for kind, other in pairs.items():
                    if other != kind:
                        out[kind], out[str(other)] = (str(other), flips), (kind, flips)
                # What the images become are images of each other, where each becomes one kind.
                for kind, (other, _flips) in list(out.items()):
                    one, two = self.becomes.get(kind, []), self.becomes.get(other, [])
                    if len(one) == 1 and len(two) == 1 and one[0] != two[0] and one[0] not in out:
                        out[one[0]], out[two[0]] = (two[0], flips), (one[0], flips)
                return out
        return {}

    def _ways_of(self, kind: str) -> dict[Direction, _Seen]:
        """What was seen of a kind going each way, with what was seen of its image on the other side."""
        ways = dict(self.seen.get(kind, {}))
        image = self.counterparts().get(kind)
        if image is None:
            return ways
        other, flips = image
        for (forward, right), way in self.seen.get(other, {}).items():
            direction = (forward, -right) if flips else (forward, right)
            ways[direction] = ways[direction].together_with(way) if direction in ways else way
        return ways

    def _steps_of(self, kind: str) -> list[Step]:
        steps: list[Step] = []
        ways = self._ways_of(kind)
        for direction, way in sorted(ways.items()):
            # A move of a special sort is relied on as far as moves of that sort were seen of the kind, as
            # seen_going_there counts them: that her rules can express one is no evidence it was seen. LIVE 23
            # November 2026 her royal kind held three ways of changing places with its own pieces, each read once.
            swaps_sure = self.special_seen.get(f"{kind}|swaps", 0) >= SEEN_TO_BELIEVE
            for times in sorted(way.swapped):
                steps.append(Step(((direction[0] * times, direction[1] * times),), rides=1, may="swaps",
                                  sure=swaps_sure))
            over = sorted(way.over - way.offered - way.offered_unmoved) if not way.over_took else sorted(way.over)
            if over and way.over_refused <= AGAINST_AT_MOST * (way.over_took + len(way.over)):
                # Beyond one piece of the other side: at one distance a leap of it, at several along the line.
                over_sure = max(self.special_seen.get(f"{kind}|takes over", 0), way.over_took) >= SEEN_TO_BELIEVE
                steps.append(Step(((direction[0] * over[0], direction[1] * over[0]),), rides=1, may="takes over",
                                  sure=over_sure)
                             if len(over) == 1 else Step((direction,), rides=0, may="takes over", sure=over_sure))
            if not way.any_seen_going():
                continue
            moved_reach = self._reach(way, unmoved=False)
            if direction[0] == 0 and self._crosses_surely(kind) and CROSSING_STEP in way.offered_unmoved:
                # Offered only from where it began, as far as its crossing lands: what the crossing explains, with
                # everything it needs, is not also a plain way.
                way = replace(way, offered_unmoved=way.offered_unmoved - {CROSSING_STEP})
            unmoved_reach = self._reach(way, unmoved=True)
            # Seen taking, and seldom or never going to an empty square that way though kept off one: it only takes
            # along it. Seen stopped at a piece of the other side it could have reached, and seldom or never taking:
            # it only moves. Taking seen fewer times than a way is believed on, and stopped short of a piece more often
            # than that, is a misreading, not a way of taking: LIVE 25 November 2026 one capture read two squares
            # straight ahead stood against four times the kind would not take, and she tried to take straight ahead;
            # at one in ten against, ten refusals would have been needed to let go of what one look had shown.
            thinly_taken = way.took < SEEN_TO_BELIEVE and way.would_not_take > way.took
            if (way.took and not thinly_taken and way.moved_to_empty <= AGAINST_AT_MOST * way.took
                    and (way.withheld or way.withheld_unmoved)):
                may = "takes"
            elif way.would_not_take and (thinly_taken or way.took <= AGAINST_AT_MOST * way.would_not_take):
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
                                      only_unmoved=only, sure=way.sightings([times]) >= SEEN_TO_BELIEVE))
                continue
            sure = way.sightings([t for t in way.offered if moved_reach <= 0 or t <= moved_reach]) >= SEEN_TO_BELIEVE
            if moved_reach >= 0:
                rides = 1 if way.jumped and moved_reach == 1 else moved_reach
                passing = kind in self.takes_passing and may == "takes"
                held = self._sure_as_far_as(way, rides) if sure else 0
                if sure and held != rides:
                    # Relied on as far as looks have shown it going that far or further, and not past that: sightings
                    # near it are no evidence of how far it goes. LIVE 23 November 2026 one misread look had her royal
                    # kind take two and three squares forward, and its twenty sightings one square forward made a
                    # three-square way of it she was sure of.
                    steps.append(Step((direction,), rides=held, may=may, takes_in_passing=passing, sure=True))
                    sure = False
                steps.append(Step((direction,), rides=rides, may=may, takes_in_passing=passing, sure=sure))
            # Offered further than the run of offers that makes its line: a leap that far or a misreading, each kept as
            # a way she is not sure of until the game has shown it often enough, or withholds it. A way never seen
            # passing over a piece goes that far only where the way is clear, as a line goes: a leap goes over whatever
            # stands between, and nothing she saw showed that. LIVE 23 November 2026, the first game after these far
            # distances were kept, as leaps: her royal kind, offered six and seven squares forward in one misread look,
            # leapt the length of the board, theirs stood attacking hers through both rows of pieces at the start, and
            # she said she had lost before making a move. Nor does it take that far where the game has shown it does
            # not (_what_the_offer_shows_of_their_reach).
            for times in sorted(offered):
                reach = moved_reach if times in way.offered else unmoved_reach
                if reach > 0 and times > reach:
                    far_may = ("moves" if may == "either"
                               and way.refused_taking_at.get(times, 0) > way._counted(times, True) else may)
                    leaps = way.jumped_at.get(times, 0) > 0
                    steps.append(Step(((direction[0] * times, direction[1] * times),) if leaps else (direction,),
                                      rides=1 if leaps else times, may=far_may,
                                      only_unmoved=times not in way.offered,
                                      sure=way.sightings([times]) >= SEEN_TO_BELIEVE))
            if unmoved_reach != moved_reach and (unmoved_reach == 0 or unmoved_reach > max(moved_reach, 0)):
                # What it may do only from where it began rests on what was seen of it from there.
                first_only = way.offered_unmoved - way.offered
                steps.append(Step((direction,), rides=unmoved_reach, may=may, only_unmoved=True,
                                  leaves_passable=kind in self.passes,
                                  sure=(way.sightings(first_only) if first_only else way.sightings())
                                  >= SEEN_TO_BELIEVE))
        return _mirrored(steps, ways)

    def as_memory(self) -> dict:
        """What she learned, in a form kept on disk with what she worked out about this world."""
        def key(square: Square) -> str:
            return f"{square[0]},{square[1]}"

        def way(seen: _Seen) -> dict:
            return {"offered": sorted(seen.offered), "offered_unmoved": sorted(seen.offered_unmoved),
                    "withheld": sorted(seen.withheld), "withheld_unmoved": sorted(seen.withheld_unmoved),
                    "took": seen.took, "would_not_take": seen.would_not_take,
                    "moved_to_empty": seen.moved_to_empty, "jumped": seen.jumped,
                    "over": sorted(seen.over), "over_took": seen.over_took, "over_refused": seen.over_refused,
                    "swapped": sorted(seen.swapped),
                    **{name: {str(t): n for t, n in sorted(getattr(seen, name).items())} for name in _COUNTED}}

        return {
            "rows": self.rows, "columns": self.columns,
            "start": {key(sq): [side, kind] for sq, (side, kind) in self.start.items()},
            "seen": {kind: {f"{d[0]},{d[1]}": way(w) for d, w in ways.items()} for kind, ways in self.seen.items()},
            "kept_off_attacked": dict(self.kept_off_attacked), "went_onto_attacked": dict(self.went_onto_attacked),
            "crosses": dict(self.crosses), "becomes": {k: list(v) for k, v in self.becomes.items()},
            "passes": sorted(self.passes), "takes_passing": sorted(self.takes_passing),
            "refusals": self.refusals, "refused_leaving_open": dict(self.refused_leaving_open),
            "taken_and_played_on": dict(self.taken_and_played_on), "first_was_hers": self.first_was_hers,
            "takes_again": sorted(self.takes_again), "quiet_kept_back_while_taking": self.quiet_kept_back_while_taking,
            "quiet_offered_while_taking": self.quiet_offered_while_taking, "no_move_loses": self.no_move_loses,
            "unconfirmed": dict(self.unconfirmed), "wins_arriving": sorted(self.wins_arriving),
            "special_seen": dict(self.special_seen), "read_by": self.read_by,
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
                    seen.over = {int(t) for t in w.get("over", [])}
                    seen.over_took, seen.over_refused = int(w.get("over_took", 0)), int(w.get("over_refused", 0))
                    seen.swapped = {int(t) for t in w.get("swapped", [])}
                    for name in _COUNTED:
                        setattr(seen, name, {int(t): int(n) for t, n in (w.get(name) or {}).items()})
                learned.seen.setdefault(kind, {})
            learned.kept_off_attacked = {k: int(v) for k, v in kept.get("kept_off_attacked", {}).items()}
            learned.went_onto_attacked = {k: int(v) for k, v in kept.get("went_onto_attacked", {}).items()}
            learned.crosses = dict(kept.get("crosses", {}))
            learned.becomes = {k: list(v) for k, v in kept.get("becomes", {}).items()}
            learned.passes, learned.takes_passing = set(kept.get("passes", [])), set(kept.get("takes_passing", []))
            learned.refusals = int(kept.get("refusals", 0))
            learned.refused_leaving_open = {k: int(v) for k, v in kept.get("refused_leaving_open", {}).items()}
            learned.taken_and_played_on = {k: int(v) for k, v in kept.get("taken_and_played_on", {}).items()}
            learned.takes_again = {str(k) for k in kept.get("takes_again", [])}
            learned.quiet_kept_back_while_taking = int(kept.get("quiet_kept_back_while_taking", 0))
            learned.quiet_offered_while_taking = int(kept.get("quiet_offered_while_taking", 0))
            learned.no_move_loses = bool(kept.get("no_move_loses", False))
            learned.unconfirmed = {str(k): int(v) for k, v in kept.get("unconfirmed", {}).items()}
            learned.wins_arriving = {str(k) for k in kept.get("wins_arriving", [])}
            learned.special_seen = {str(k): int(v) for k, v in kept.get("special_seen", {}).items()}
            first = kept.get("first_was_hers")
            learned.first_was_hers = first if isinstance(first, bool) else None
            read_by = int(kept.get("read_by", 1))
            if read_by < 2:
                learned._weighed_as_read_before_mending()
            if read_by < 3:
                learned._unlisted_as_read_before_mending()
            if read_by < 4:
                learned.withheld_takes_unchecked = True
            learned.read_by = READING
            return learned
        except (KeyError, TypeError, ValueError) as why:
            logger.info("what she learned of this game could not be read back (%s); learning it afresh", why)
            return None

    def _weighed_as_read_before_mending(self) -> None:
        """What a reading since mended gathered, weighed as kept knowledge is when it comes back: every count it made
        at TRUST_CARRIED_OVER of itself, and a distance whose own counts come to nothing no longer offered. LIVE 23
        November 2026, game 31: 31 of the squares she read in 55 answers were never offered, most of them the marks of
        the piece looked at before; thirty games read that way left her rules with ways seen once that no later look
        could test (a leap onto a square her own piece always held). Ways seen many times keep their standing."""
        from core.runtime.what_she_learned import TRUST_CARRIED_OVER

        def less(n: int) -> int:
            return int(n * TRUST_CARRIED_OVER)

        dropped = 0
        for ways in self.seen.values():
            for way in ways.values():
                counted_by_distance = bool(way.went_at or way.took_at)
                for name in _COUNTED:
                    setattr(way, name, {t: less(n) for t, n in getattr(way, name).items() if less(n) > 0})
                way.took, way.would_not_take = less(way.took), less(way.would_not_take)
                way.moved_to_empty, way.jumped = less(way.moved_to_empty), less(way.jumped)
                way.over_took, way.over_refused = less(way.over_took), less(way.over_refused)
                if counted_by_distance:
                    for kept in (way.offered, way.offered_unmoved):
                        # A distance in an unbroken run of two or more from the next square rests on the run, as a
                        # line does, where nothing contests it; any other stands on its own counts.
                        run = 0
                        while run + 1 in kept:
                            run += 1
                        line = run if run >= 2 else 0
                        contested = (way.withheld | way.withheld_unmoved | set(way.refused_going_at)
                                     | set(way.refused_taking_at))
                        gone = {t for t in kept if (t > line or t in contested)
                                and not way.went_at.get(t) and not way.took_at.get(t)}
                        kept -= gone
                        dropped += len(gone)
        self.special_seen = {k: less(v) for k, v in self.special_seen.items() if less(v) > 0}
        logger.info("what an earlier reading of this board gathered now weighs %.1f of itself; %d distance(s) it alone "
                    "held are let go", TRUST_CARRIED_OVER, dropped)

    def withheld_takes_read_again(self) -> None:
        """Takes an earlier reading counted as not offered, read again against the answers kept (``shown``): each way's
        count stands as far as those answers, read as a take not offered is read now, bear it out, and the rest is let
        go. Before READING 4 every take not offered counted against taking that way, those withheld because taking would
        leave the kind the game keeps safe where it could be taken among them, and nothing kept tells the two apart but
        the answers themselves. LIVE 24 November 2026 her royal kind was held never to take on its right from three
        such takes, older than every answer kept; measured with true moves for that kind alone, its learned ways cost
        her more against a stronger program than any other kind's (e9745c00a). What is let go and so is learned again
        the next time a take that way is not offered with nothing to explain it."""
        if not self.withheld_takes_unchecked or self.shown is None:
            return
        import copy

        from core.agency.what_the_game_showed import _read

        family = self.family()
        again = copy.deepcopy(self)
        again.shown = None
        for ways in again.seen.values():
            for way in ways.values():
                way.would_not_take = 0
        for answer in self.shown.offers:
            try:
                position = _read(answer["at"])
            # not a failure: an answer kept by an older reader that cannot be read back bears out nothing.
            except (KeyError, TypeError, ValueError):
                continue
            start = tuple(answer["from"])
            if start not in position.cells or position.cells[start][0] != position.turn:
                continue
            again.offered(position, start, {tuple(sq) for sq in answer["to"]}, family=family)
        for answer in self.shown.refusals:
            try:
                position = _read(answer["at"])
            # not a failure: as above.
            except (KeyError, TypeError, ValueError):
                continue
            start, end = tuple(answer["from"]), tuple(answer["to"])
            if start not in position.cells:
                continue
            taken = end if end in position.cells and position.cells[end][0] != position.cells[start][0] else None
            again.refused(position, Move(start, end, taken_at=taken), family=family)
        let_go = 0
        for kind, ways in self.seen.items():
            for direction, way in ways.items():
                borne_out = again.seen.get(kind, {}).get(direction)
                kept = min(way.would_not_take, borne_out.would_not_take if borne_out is not None else 0)
                let_go += way.would_not_take - kept
                way.would_not_take = kept
        self.withheld_takes_unchecked = False
        logger.info("takes an earlier reading counted as not offered, read again against %d answers kept: %d let go",
                    len(self.shown.offers) + len(self.shown.refusals), let_go)

    def _unlisted_as_read_before_mending(self) -> None:
        """What an earlier reading kept as a bare list with no count, of kinds a piece was read as having become: each
        stands for at least one reading of no known number, and weighed as every count of that reading is
        (TRUST_CARRIED_OVER of one) it comes to less than one, and is let go. A kind's becoming is learned again the
        next time two looks show it (core/agency/playing_pieces_by_kind.py, became_on_landing). LIVE 24 November 2026,
        measured on her record: two kinds held to become a third cost every game of 42 against a stronger program, and
        taking what each kind becomes from the true game alone lifted that to 0.119."""
        let_go = sum(len(v) for v in self.becomes.values())
        self.becomes = {}
        logger.info("what an earlier reading kept of kinds becoming others is let go: %d entries", let_go)

    def under_other_names(self, names: dict[str, str], *, turned_round: bool = False) -> HowThePiecesMove:
        """What she learned here, for a board that begins as this one does with each kind called what ``names`` maps
        it to (core/cognition/the_same_start_under_other_names.py), seen from the other side of it where
        ``turned_round``. How a kind goes is seen from its own side, so it reads the same from either. A kind with no
        name there is left out: she has not seen it at that board."""
        def counts(of: dict[str, int]) -> dict[str, int]:
            return {names[kind]: count for kind, count in of.items() if kind in names}

        start = turned_round_from(self.start, self.rows, self.columns) if turned_round else self.start
        first = self.first_was_hers
        return HowThePiecesMove(
            rows=self.rows, columns=self.columns,
            start={square: (side, names[kind]) for square, (side, kind) in start.items() if kind in names},
            seen={names[kind]: {d: replace(w, offered=set(w.offered), offered_unmoved=set(w.offered_unmoved),
                                           withheld=set(w.withheld), withheld_unmoved=set(w.withheld_unmoved),
                                           over=set(w.over), swapped=set(w.swapped),
                                           **{name: dict(getattr(w, name)) for name in _COUNTED})
                                for d, w in ways.items()}
                  for kind, ways in self.seen.items() if kind in names},
            kept_off_attacked=counts(self.kept_off_attacked), went_onto_attacked=counts(self.went_onto_attacked),
            crosses={names[k]: names[v] for k, v in self.crosses.items() if k in names and v in names},
            becomes={names[k]: [names[v] for v in vs if v in names] for k, vs in self.becomes.items() if k in names},
            passes={names[k] for k in self.passes if k in names},
            takes_passing={names[k] for k in self.takes_passing if k in names},
            names={names[k]: said for k, said in self.names.items() if k in names},
            refusals=self.refusals, refused_leaving_open=counts(self.refused_leaving_open),
            taken_and_played_on=counts(self.taken_and_played_on),
            takes_again={names[k] for k in self.takes_again if k in names},
            quiet_kept_back_while_taking=self.quiet_kept_back_while_taking,
            quiet_offered_while_taking=self.quiet_offered_while_taking, no_move_loses=self.no_move_loses,
            first_was_hers=None if first is None else first != turned_round,
            special_seen={f"{names[k]}|{sort}": n for key, n in self.special_seen.items()
                          for k, sort in [key.split("|", 1)] if k in names})

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
                            # A move that changes two things at once belongs to the kind the game offered it to, or
                            # was seen making it, often enough; not to whichever of the two a reading named as the
                            # mover. LIVE 24 November 2026 her record held the crossing both ways, from a reading
                            # that once named the partner as the mover, so her straight-runner "crossed" by going
                            # two squares toward her royal kind, and she played it, a plain move on the board,
                            # giving the crossing up (7.Rf1). The royal kind had been offered it twelve times, the
                            # straight-runner never.
                            crosses_with=self.crosses.get(kind, "") if self._crosses_surely(kind) else "",
                            takes_again=kind in self.takes_again or self.counterparts().get(kind, ("",))[0]
                            in self.takes_again,
                            wins_arriving=kind in self.wins_arriving,
                            name=self.names.get(kind, ""))
                 for kind in sorted({k for _s, k in self.start.values()} | set(self.seen))}
        return Family(name="the game in front of me", rows=self.rows, columns=self.columns, kinds=kinds,
                      start=dict(self.start), takes_first=self.takes_first(),
                      no_move_unattacked="loses" if self.no_move_loses else "draw")

    def what_it_does(self, kind: str) -> str:
        """A kind as she has seen it go, in words, for naming it where she has no other name for it."""
        steps = self._steps_of(kind)
        if not steps:
            return "a piece I haven't seen move"
        # A longer step a piece may take only from its first square is not how it runs. LIVE 21 November 2026 the
        # pawns' first double step made them "the piece that runs straight", the rooks' words, so "their piece that
        # runs straight from d7 to d5" did not say which.
        lines = {s.offsets[0] for s in steps if s.rides != 1 and not s.only_unmoved}
        leaps = {s.offsets[0] for s in steps if s.rides == 1 or s.only_unmoved}
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
        # The image's own way: a leap's offset is that way taken some number of times.
        times = math.gcd(abs(forward), abs(right)) or 1
        way = ways.get((forward // times, -right // times))
        if way is not None and (way.withheld or way.withheld_unmoved) and not way.any_seen_going():
            continue
        # The image the game refused her as often as it was offered there is not one, whatever the step it is the
        # image of. LIVE 23 November 2026, game 27: her bishop's misread leap, refused as e2 to a8, came back as its
        # image, d2 to h8, and was refused twice more.
        if way is not None and _refused_there(way, times if step.rides == 1 else None):
            continue
        # An image is an inference: relied on as far as the game has shown the image itself.
        seen = way.sightings([times] if step.rides == 1 else None) if way is not None else 0
        out.append(Step((image,), rides=step.rides, may=step.may, only_unmoved=step.only_unmoved,
                        leaves_passable=step.leaves_passable, takes_in_passing=step.takes_in_passing,
                        sure=step.sure and seen >= SEEN_TO_BELIEVE))
        have.add((image, step.only_unmoved))
    return out


def _refused_there(way: _Seen, times: int | None) -> bool:
    """Whether the game refused her moves along a way, at a distance (or, for a line, anywhere along it), at least as
    often as it offered them there."""
    refused = set(way.refused_going_at) | set(way.refused_taking_at)
    distances = [times] if times is not None else sorted(refused)
    return any(t in refused and not way._stands(t) for t in distances) and not (
        times is None and any(way._stands(t) for t in set(way.went_at) | set(way.took_at)))


def _takes_over_seen(side: int, move: Move) -> bool:
    """Whether a move took over: it landed more than once, or took more than one piece, or took a piece standing on
    its way rather than where it landed or beside it."""
    if move.via or move.also_taken:
        return True
    if move.taken_at is None or move.taken_at == move.end:
        return False
    direction, times = in_its_frame(side, move.start, move.end)
    return move.taken_at in {_square_along(side, move.start, direction, t) for t in range(1, times)}


def _a_chain_seen(before: Position, start: Square, end: Square, lost: list[Square], side: int,
                  size: tuple[int, int]) -> tuple[tuple[Square, ...], tuple[Square, ...]] | None:
    """How a piece went from ``start`` to ``end`` taking every one of ``lost``: from each landing along a straight line,
    over the way clear to one of them, onto an empty square beyond, until all are taken and it stands on ``end``. The
    squares it landed on before the end, and the pieces in the order taken; None where no such way is there. Pieces it
    takes stand until the move ends."""
    def empty(square: Square) -> bool:
        return square == start or square not in before.cells

    rows, columns = size

    def walk(at: Square, left: frozenset[Square], via: tuple[Square, ...],
             taken: tuple[Square, ...]) -> tuple[tuple[Square, ...], tuple[Square, ...]] | None:
        if not left:
            return (via[:-1], taken) if at == end else None
        for victim in sorted(left):
            dr, dc = victim[0] - at[0], victim[1] - at[1]
            if not dr and not dc or (dr and dc and abs(dr) != abs(dc)):
                continue
            length = max(abs(dr), abs(dc))
            ur, uc = dr // length, dc // length
            if any(not empty((at[0] + ur * k, at[1] + uc * k)) for k in range(1, length)):
                continue
            beyond = (victim[0] + ur, victim[1] + uc)
            while 0 <= beyond[0] < rows and 0 <= beyond[1] < columns and empty(beyond):
                found = walk(beyond, left - {victim}, (*via, beyond), (*taken, victim))
                if found is not None:
                    return found
                beyond = (beyond[0] + ur, beyond[1] + uc)
        return None

    return walk(start, frozenset(lost), (), ())


def a_move_her_rules_allow(family: Family, position: Position, move: Move) -> bool:
    """Whether her rules have this move: from the same square to the same square, taking the same pieces."""
    from core.agency.pieces_by_their_kind import moves as legal_moves

    taken = set(move.taken())
    return any((m.start, m.end) == (move.start, move.end) and set(m.taken()) == taken
               for m in legal_moves(family, position))


def _sort_of(side: int, move: Move) -> str:
    """What sort of move it is: a take over, a take where it lands, a take beside where it lands, or a move to an
    empty square."""
    if _takes_over_seen(side, move):
        return "over"
    if move.taken_at is None:
        return "quiet"
    return "takes" if move.taken_at == move.end else "passing"


def _sorts_of(kind: Kind) -> set[str]:
    sorts: set[str] = set()
    for step in kind.steps:
        if step.may == "takes over":
            sorts.add("over")
        if step.may in ("moves", "either"):
            sorts.add("quiet")
        if step.may in ("takes", "either"):
            sorts.add("takes")
        if step.takes_in_passing:
            sorts.add("passing")
    return sorts


def _how_new(family: Family, position: Position, move: Move, allowed: bool) -> int:
    """How much a move seen asks her to believe that she does not: nothing where her rules allow it; a new way for a
    sort of move its kind makes; a sort of move only other kinds make; a sort no kind makes."""
    if allowed:
        return 0
    side, kind = position.cells[move.start]
    sort = _sort_of(side, move)
    if kind in family.kinds and sort in _sorts_of(family.kinds[kind]):
        return 1
    return 2 if any(sort in _sorts_of(other) for other in family.kinds.values()) else 3


def a_move_seen(before: Position, after_cells: dict[Square, tuple[int, str | None]], side: int,
                size: tuple[int, int] | None = None, royal: frozenset[str] = frozenset()) -> Move | None:
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
        letter_before = before.cells[start][1]
        letter_after = after_cells[end][1]
        becomes = letter_after if letter_after is not None and letter_after != letter_before else ""
        if len(lost) > 1 and end not in before.cells:
            # Several of the other side's pieces gone and one of hers moved: take overs, one after another.
            seen_at = [*before.cells, *after_cells]
            board = size or (max(r for r, _c in seen_at) + 1, max(c for _r, c in seen_at) + 1)
            chain = _a_chain_seen(before, start, end, lost, side, board)
            if chain is not None:
                via, taken = chain
                return Move(start, end, becomes=becomes, taken_at=taken[0], via=via, also_taken=taken[1:])
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
        # Two pieces at once, only in the one shape a crossing has: on one row, the one that moved two squares toward a
        # partner beyond, the partner onto the square crossed. LIVE-like 22 November 2026, two pieces of a draughts side
        # read as moving at once taught her men a crossing, and her model made moves that put one piece where another
        # stood. Each pairing is held to the shape, not only the piece that went further: LIVE 22 November 2026 the rook
        # of a long crossing went three squares to the royal's two, was taken for the mover, and the board went
        # unexplained. Where both fit, as in a short crossing, the royal kind is the one that moved.
        fits = []
        for a in gone:
            for b in came:
                other_a = next(g for g in gone if g != a)
                other_b = next(c for c in came if c != b)
                if before.cells[a][1] != after_cells[b][1] or before.cells[other_a][1] != after_cells[other_b][1]:
                    continue
                way = 1 if b[1] > a[1] else -1
                if (a[0] == b[0] == other_a[0] == other_b[0] and abs(b[1] - a[1]) == 2
                        and other_b == (a[0], a[1] + way) and (other_a[1] - b[1]) * way > 0):
                    fits.append((before.cells[a][1] not in royal, a, b, other_a, other_b))
        if fits:
            _not_royal, a, b, other_a, other_b = min(fits)
            return Move(a, b, partner=(other_a, other_b))
    if not gone and not came:
        # Two of the side's pieces changed places, nothing else moving: a swap. The one that moved is the kind the side
        # holds fewer of, the piece set apart; the other its partner.
        changed = [sq for sq, (s, k) in before.cells.items()
                   if s == side and after_cells.get(sq, (None, None))[1] not in (None, k)]
        if len(changed) == 2:
            a, b = changed
            if after_cells[a][1] == before.cells[b][1] and after_cells[b][1] == before.cells[a][1]:
                held = [k for s, k in before.cells.values() if s == side]
                mover, other = sorted((a, b), key=lambda sq: (held.count(before.cells[sq][1]), sq))
                return Move(mover, other, partner=(other, mover))
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
        return a_move_her_rules_allow(family, position, move)

    # A move her rules allow that makes exactly what is seen, the only one that does, is the move.
    exactly = [move for move in legal_moves(family, Position(cells=before.cells, turn=side, unmoved=before.unmoved,
                                                             passable=before.passable))
               if made_shows(after(family, before, move))]
    if len(exactly) == 1:
        return exactly
    royal = frozenset(letter for letter, kind in family.kinds.items() if kind.royal)
    fitting: list[tuple[tuple[int, int, int], list[Move]]] = []
    one = a_move_seen(before, after_cells, side, (family.rows, family.columns), royal)
    if one is not None and made_shows(after(family, before, one)):
        if allowed(before, one):
            return [one]
        # A move her rules do not allow is weighed against her move and their answer, which may be made of moves she
        # knows. LIVE-like 22 November 2026 a man read as stepping two squares, taking a piece beside where it landed,
        # taught every man a two-square step.
        fitting.append(((_how_new(family, before, one, False), 0, int(bool(one.becomes))), [one]))
    gone = [sq for sq, (s, _k) in before.cells.items() if s == side and after_cells.get(sq, (None,))[0] != side]
    landed = [sq for sq in after_cells if before.cells.get(sq, (None,))[0] != side]

    def as_shown(move: Move) -> Move:
        # Her piece shown where it landed as another kind than it was: the move made it that kind. LIVE-like 22
        # November 2026 her man crowned on b8 and their answer, both made before she looked, explained nothing.
        there = after_cells.get(move.end)
        if there is not None and there[0] == side and there[1] not in (None, before.cells[move.start][1]):
            return replace(move, becomes=there[1])
        return move
    for start in gone:
        for end in landed:
            if end == start:
                continue
            taken = end if end in before.cells and before.cells[end][0] != side else None
            mine = as_shown(Move(start, end, taken_at=taken))
            middle = after(family, before, mine)
            answer = a_move_seen(middle, after_cells, 1 - side, (family.rows, family.columns), royal)
            if answer is not None and made_shows(after(family, middle, answer)):
                rank = (_how_new(family, before, mine, allowed(before, mine)),
                        _how_new(family, middle, answer, allowed(middle, answer)), int(bool(answer.becomes)))
                fitting.append((rank, [mine, answer]))
    # Her move by the rules she holds, and their answer: a move of hers that took over, or a piece of hers taken in the
    # answer, has no square it landed on that the board still shows. LIVE-like 22 November 2026, b4xd6 answered by
    # c7xe5 before she looked again left the board unexplained; and a man of hers stepping forward and taken over at
    # once was read as a two-square step sideways taken where it landed, which taught her men to step sideways.
    for mine in legal_moves(family, Position(cells=before.cells, turn=side, unmoved=before.unmoved,
                                             passable=before.passable)):
        if not mine.taken() and after_cells.get(mine.end, (None,))[0] == side:
            continue
        mine = as_shown(mine)
        middle = after(family, before, mine)
        answer = a_move_seen(middle, after_cells, 1 - side, (family.rows, family.columns), royal)
        if answer is not None and made_shows(after(family, middle, answer)):
            fitting.append(((0, _how_new(family, middle, answer, allowed(middle, answer)), int(bool(answer.becomes))),
                            [mine, answer]))
    if not fitting:
        return None
    return min(fitting, key=lambda pair: pair[0])[1]
