"""How this thing responds, worked out from watching herself act in it.

Every move she makes yields a triple — what was there, what she did, what was
there afterwards — and until now all three were thrown away after one glance.
So she could never try a move without making it, and every plan was a bet
placed blind.

Anyone can write a solver for a thing once somebody hands them the rules,
because the rules ARE the solver: turn them into a transition function and
search over it. The hard part was never the search. It was that somebody had
already done the abstracting, and what they handed over was the answer to the
only question worth asking.

So the hypotheses here are not a list of games. They are a space, and she
composes a rule out of it rather than picking one off a menu somebody wrote.
Three independent facts about how a laid-out thing answers a push —

    how far a thing carries      all the way to the end, or one place
    whether equals combine       two the same become one worth both
    how many things move         everything that can, or one thing

— which is eight ways for a thing to move, plus not moving at all. Nothing in
that space is about any particular screen: the facts are the ways a set of
things in rows and columns can respond to being pushed, and a rule made of
them is a transition function she arrived at rather than one she was given.
When nothing in the space predicts what she sees, that is the answer, and she
goes back to acting and looking rather than taking the nearest fit.

The scoring allows for what the world adds on its own. A board that deals a
new tile, a page that updates a clock, a list that gains a row from somebody
else — a rule is not wrong because something arrived that it never claimed to
know about. What it must get right is what it said would move.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

from core.perception.what_is_there import Arrangement, Cell

__all__ = [
    "HowItMoves",
    "prediction_held",
    "Rule",
    "RULES",
    "CARRIES",
    "HOW_MANY",
    "composed",
    "shifted",
    "shifted_and_combined",
]

logger = logging.getLogger("Aura.HowItMoves")

#: How many of her own moves a rule has to have survived before it is worth
#: predicting from. Below this the leader is whichever rule happened to match
#: the first thing she did.
ENOUGH_TO_TRUST = 4

#: The share of recent moves a rule has to have got right. A rule that is
#: wrong a third of the time is not a rule she can plan on.
OFTEN_ENOUGH = 0.7

#: How many of her own moves are kept. Enough to notice a rule going wrong
#: when the thing she is acting in changes under her.
REMEMBERED = 24

#: Directions a laid-out thing can be pushed, and what that does to an index.
_TOWARD: dict[str, tuple[int, int]] = {
    "left": (0, -1),
    "right": (0, 1),
    "up": (-1, 0),
    "down": (1, 0),
}


def _lines(arrangement: Arrangement, action: str) -> list[list[Cell | None]] | None:
    """The rows or columns a push runs along, in the order it runs."""
    step = _TOWARD.get(str(action or "").strip().lower())
    if step is None or not arrangement.rows or not arrangement.columns:
        return None
    down, across = step
    if across:
        lines = [list(arrangement.row_at(row)) for row in range(arrangement.rows)]
        return [line[::-1] if across > 0 else line for line in lines]
    lines = [list(arrangement.column_at(column)) for column in range(arrangement.columns)]
    return [line[::-1] if down > 0 else line for line in lines]


def _rebuild(
    arrangement: Arrangement, action: str, lines: list[list[str | None]]
) -> Arrangement:
    """Put worked-on lines back where they came from."""
    down, across = _TOWARD[action]
    cells: list[Cell] = []
    for index, line in enumerate(lines):
        restored = line[::-1] if (across > 0 or down > 0) else line
        for position, said in enumerate(restored):
            if said is None:
                continue
            row, column = (index, position) if across else (position, index)
            cells.append(Cell(row=row, column=column, says=said, at=(0.0, 0.0)))
    return Arrangement(rows=arrangement.rows, columns=arrangement.columns, cells=tuple(cells))


def _packed(line: Sequence[Cell | None]) -> list[str]:
    return [cell.says for cell in line if cell is not None]


def unchanged(arrangement: Arrangement, action: str) -> Arrangement | None:
    """The push did nothing. A real answer, and often the right one."""
    if _TOWARD.get(str(action or "").strip().lower()) is None:
        return None
    return arrangement


def shifted(arrangement: Arrangement, action: str) -> Arrangement | None:
    """Everything slides as far as it can go that way, and nothing else."""
    lines = _lines(arrangement, action)
    if lines is None:
        return None
    width = len(lines[0]) if lines else 0
    moved: list[list[str | None]] = []
    for line in lines:
        said = _packed(line)
        moved.append(said + [None] * (width - len(said)))
    return _rebuild(arrangement, str(action).strip().lower(), moved)


def shifted_and_combined(arrangement: Arrangement, action: str) -> Arrangement | None:
    """Things slide, and two equal neighbours become one thing worth both.

    What "worth both" means is read off the things themselves: where they are
    numbers, the pair becomes their sum. Where they are not, a pair of equals
    becomes one of them, which is what stacking or de-duplicating looks like.
    """
    lines = _lines(arrangement, action)
    if lines is None:
        return None
    width = len(lines[0]) if lines else 0
    moved: list[list[str | None]] = []
    for line in lines:
        said = _packed(line)
        joined: list[str] = []
        index = 0
        while index < len(said):
            if index + 1 < len(said) and said[index] == said[index + 1]:
                joined.append(_worth_both(said[index]))
                index += 2
            else:
                joined.append(said[index])
                index += 1
        moved.append(joined + [None] * (width - len(joined)))
    return _rebuild(arrangement, str(action).strip().lower(), moved)


def _worth_both(said: str) -> str:
    """What two of these together come to."""
    try:
        value = float(said.replace(",", ""))
    except (ValueError, AttributeError):
        return said
    doubled = value * 2
    return f"{doubled:g}"


#: How far a thing carries when it is pushed.
CARRIES: tuple[str, ...] = ("all the way", "one place")

#: How much of what could move does move.
HOW_MANY: tuple[str, ...] = ("everything", "one thing")


def _stepped(
    arrangement: Arrangement, action: str, *, combines: bool, only_one: bool
) -> Arrangement | None:
    """Things move one place the way they were pushed, and no further.

    A thing goes forward if the place ahead of it is free — or, where equals
    combine, if the place ahead holds one the same. With ``only_one`` a single
    thing moves and the rest stay put, which is what a puzzle with one space
    in it looks like from the outside.
    """
    step = _TOWARD.get(str(action or "").strip().lower())
    if step is None or not arrangement.rows or not arrangement.columns:
        return None
    down, across = step
    held = {(cell.row, cell.column): cell.says for cell in arrangement.cells}
    going: list[tuple[tuple[int, int], tuple[int, int], str]] = []
    for (row, column), said in held.items():
        ahead = (row + down, column + across)
        if not (0 <= ahead[0] < arrangement.rows and 0 <= ahead[1] < arrangement.columns):
            continue
        there = held.get(ahead)
        if there is None or (combines and there == said):
            going.append((ahead, (row, column), said))
    if not going:
        return arrangement
    # Whichever space comes first, read the way anything laid out is read.
    going.sort(key=lambda move: (move[0], move[1]))
    if only_one:
        going = going[:1]
    landed = dict(held)
    for ahead, came_from, said in going:
        if landed.get(came_from) != said:
            continue
        was_there = landed.get(ahead)
        landed.pop(came_from, None)
        landed[ahead] = _worth_both(said) if (combines and was_there == said) else said
    cells = tuple(
        Cell(row=row, column=column, says=said, at=(0.0, 0.0))
        for (row, column), said in landed.items()
    )
    return Arrangement(rows=arrangement.rows, columns=arrangement.columns, cells=cells)


def _carried(
    arrangement: Arrangement, action: str, *, combines: bool, only_one: bool
) -> Arrangement | None:
    """Things go as far that way as they can, packing to the end."""
    whole = shifted_and_combined(arrangement, action) if combines else shifted(arrangement, action)
    if whole is None or not only_one:
        return whole
    # One thing moving all the way, and everything else where it was.
    was = {(cell.row, cell.column): cell.says for cell in arrangement.cells}
    now = {(cell.row, cell.column): cell.says for cell in whole.cells}
    landed = sorted(place for place, said in now.items() if was.get(place) != said)
    if not landed:
        return arrangement
    first = landed[0]
    kept = dict(was)
    kept[first] = now[first]
    for place in was:
        if place != first and place not in now:
            kept.pop(place, None)
            break
    cells = tuple(
        Cell(row=row, column=column, says=said, at=(0.0, 0.0))
        for (row, column), said in kept.items()
    )
    return Arrangement(rows=arrangement.rows, columns=arrangement.columns, cells=cells)


@dataclass(frozen=True)
class Rule:
    """One way a thing might answer to being pushed.

    Made of the facts it is made of, so what she worked out can be said in
    terms of the world rather than as the name of something off a list.
    """

    name: str
    apply: Callable[[Arrangement, str], Arrangement | None]
    carries: str = ""
    combines: bool = False
    how_many: str = ""

    def as_facts(self) -> str:
        """The transition she arrived at, in the terms it was arrived at in."""
        if not self.carries:
            return "nothing she does moves anything"
        joins = "two the same become one worth both" if self.combines else "nothing combines"
        return f"{self.how_many} carries {self.carries}, {joins}"


def composed(carries: str, combines: bool, how_many: str) -> Rule:
    """One rule out of the space, built from the three facts that make it."""
    all_the_way = carries == "all the way"
    only_one = how_many == "one thing"
    build = _carried if all_the_way else _stepped

    def apply(arrangement: Arrangement, action: str) -> Arrangement | None:
        return build(arrangement, action, combines=combines, only_one=only_one)

    moves = "slides" if all_the_way else "steps"
    name = f"{'one thing' if only_one else 'everything'} {moves}"
    if not only_one and all_the_way:
        # The two she met first, kept in the words she already reports them in.
        name = "slides"
    if combines:
        name = f"{name} and combines"
    return Rule(name, apply, carries=carries, combines=combines, how_many=how_many)


def _every_way() -> tuple[Rule, ...]:
    """The whole space, strongest claim first, so a tie goes to the fuller one.

    A rule that says everything moves as far as it can and combines on the way
    claims the most about the most places; one that says nothing moves claims
    the least. Ordering that way lets a tie be broken toward the rule that
    claims least, which is the only honest way to break one: where two
    hypotheses have never disagreed about anything she has seen, the extra
    thing the fuller one claims is the part she has no evidence for.
    """
    made = [
        composed(carries, combines, how_many)
        for carries in CARRIES
        for how_many in HOW_MANY
        for combines in (True, False)
    ]
    made.sort(
        key=lambda rule: (
            rule.carries == "all the way",
            rule.how_many == "everything",
            rule.combines,
        ),
        reverse=True,
    )
    return tuple(made) + (Rule("does not move", unchanged),)


#: Every way this can compose. Nothing here was written for any world.
RULES: tuple[Rule, ...] = _every_way()


@dataclass
class HowItMoves:
    """What she has worked out about the thing she is acting in.

    Kept per run rather than per world, because a rule that held on one thing
    is a guess about the next one and she should find out rather than assume.
    """

    right: dict[str, int] = field(default_factory=dict)
    tried: dict[str, int] = field(default_factory=dict)
    seen: int = 0
    #: The last few moves, so a rule that stops working is noticed.
    recent: list[tuple[str, frozenset[str]]] = field(default_factory=list)
    #: Pairs of readings that could not be compared at all.
    unreadable: int = 0
    #: Times something turned up that she did not put there.
    arrivals: int = 0
    #: What she saw, what she did, what happened — as text, in order. The one
    #: record that can prove a quantity she is not reading.
    _record: list[tuple[str, str, str]] = field(default_factory=list)
    #: Acts that actually moved something, and how each rule did on those.
    #: Only these tell one hypothesis from another.
    moved: int = 0
    right_when_it_moved: dict[str, int] = field(default_factory=dict)
    tried_when_it_moved: dict[str, int] = field(default_factory=dict)
    #: Places that sit still and change what they say — a score, a clock, a
    #: count of moves. They answer to her, so they are inside the part of the
    #: screen that responds, and no rule about what her act MOVES can predict
    #: one. Left in the comparison they make every hypothesis wrong forever.
    counters: set[tuple[int, int]] = field(default_factory=set)
    #: How often each place has been seen holding something, and how often
    #: what it holds has changed.
    _seen_at: dict[tuple[int, int], int] = field(default_factory=dict)
    _changed_at: dict[tuple[int, int], int] = field(default_factory=dict)
    #: Places she has seen empty, which settles them as part of the thing.
    _a_place: set[tuple[int, int]] = field(default_factory=set)
    _looks: int = 0
    #: Whether what she had learned has already been thrown away once, for
    #: being about the surroundings rather than the thing.
    _started_again: bool = False
    #: How full the thing has been, added up across looks.
    _how_full: float = 0.0
    #: Every place that has ever held anything. What has not is not part of
    #: the thing, and something sitting beside it is at the edge of the thing.
    _ever_held: set[tuple[int, int]] = field(default_factory=set)
    #: The shape of the grid everything here was read through. Remembered
    #: evidence carries the conditions it was gathered under, so that it can
    #: be dropped when they stop holding.
    read_through: tuple[int, int] = (0, 0)

    # ── learning ─────────────────────────────────────────────────────────

    def _note_counters(self, before: Arrangement, after: Arrangement) -> None:
        """Places that sit still and change what they say.

        A score, a clock, a count of moves. They answer to her — they change
        when she acts — so the part of the screen that responds to her
        includes them, and every rule about what her act MOVES is wrong about
        one every single time. LIVE 2026-08-26: seventeen readings watched,
        every hypothesis discredited, on a board she was reading correctly.

        Nothing here knows what a score is. What it knows is that a thing
        which never goes anywhere and keeps saying something different is not
        a thing her moves move. Both halves are load-bearing: staying put is
        what an ordinary thing does whenever she pushes the other way, and
        saying something new is what nothing wedged in place ever does.
        """
        self._looks += 1
        now = {(cell.row, cell.column): cell.says for cell in after.cells}
        # Somewhere she has seen empty is a place, and never furniture again.
        #
        # One glimpse settles it. Everything in the thing she is acting on is
        # empty sometimes — that is what makes it a place rather than a
        # readout — and nothing that surrounds it ever is.
        for row in range(after.rows):
            for column in range(after.columns):
                if (row, column) not in now:
                    self._a_place.add((row, column))
                    self.counters.discard((row, column))
        if not self._a_place:
            # Nothing has been empty yet, so there is nothing to tell apart.
            return
        # How long a place has to have gone without ever being empty before
        # that means anything — which depends entirely on how full the thing is.
        #
        # In something half empty, a place that is occupied every single time
        # she looks stands out, and a few looks settle it. In something with
        # fifteen things on sixteen places, every place is occupied nearly
        # always, and never having caught one empty is what she would expect of
        # an ordinary place. The same observation carries information in the
        # first case and none in the second.
        #
        # So: enough looks that an ordinary place would almost certainly have
        # been caught empty by now. A place free a fifteenth of the time is
        # missed forty-six looks running about one time in twenty. Below that
        # she has no evidence, and cropping on no evidence takes the thing
        # apart. Measured 2026-08-27 on a puzzle with one space: a rule that
        # was exactly right about 200 of 200 moves scored 78%, because most of
        # the board had been called furniture and cut out of the comparison.
        # And she is not watching one place. She is watching all of them, and
        # asking of each whether it has been surprisingly reliable. The more
        # places she asks that of, the more certain it becomes that one of them
        # answers yes for no reason — sixteen places, each missed one time in a
        # hundred, is one wrongly-called place in six runs. So the bar for any
        # single place rises with how many are being held to it.
        self._how_full += after.occupied() / max(1, after.places())
        enough = max(
            STILL_ENOUGH_TO_JUDGE,
            _looks_to_expect_an_empty(self.fullness(), among=after.places()),
        )
        was = {(cell.row, cell.column): cell.says for cell in before.cells}
        self._ever_held.update(now)
        for place in now:
            if place in self._a_place:
                continue
            self._seen_at[place] = self._seen_at.get(place, 0) + 1
            if place in was and was[place] != now[place]:
                self._changed_at[place] = self._changed_at.get(place, 0) + 1
            times = self._seen_at[place]
            if times < enough or self._looks < enough:
                continue
            if times / max(1, self._looks) < ALWAYS_THERE:
                continue
            # Sitting still is not enough, and it never was.
            #
            # Never having seen a place empty cannot settle this on its own, and
            # no number of looks makes it: she picks her own moves, so where
            # things end up is a sample she biased herself, and in something
            # nearly full a place she has never caught free is the ordinary
            # case. Measured 2026-08-27 on a puzzle with one space, where two
            # ordinary places were called furniture after ninety looks and
            # cutting them out opened gaps that were not there.
            #
            # What settles it is one of two things, and either will do.
            #
            # A readout keeps saying something new while going nowhere: 240,
            # 244, 252, in the same place every time. Nothing wedged in a
            # corner does that.
            #
            # Or it stands apart. A title, a label, the word SCORE — those never
            # change at all, and what marks them is that the places around them
            # are not places anything has ever been. The thing itself is packed:
            # everywhere in it has held something. Furniture sits at the edge of
            # it with blank beside it.
            if not self._stands_apart(place, after) and (
                self._changed_at.get(place, 0) < TWICE_IS_THE_WORLD
            ):
                continue
            if place in self.counters:
                continue
            self.counters.add(place)
            if self.rule() is not None:
                # Not once she knows.
                #
                # This correction is for the confusion at the start, when
                # every reading still has a score stuck to it and the counts
                # are about a board that does not exist. Once she has a rule
                # she is willing to name, that is over. The bar is having one
                # rather than a level of certainty somebody picked: naming one
                # is already the judgement that the evidence is worth acting
                # on, and a second bar beside it would only be a worse copy.
                # Finding one more place that is furniture says the crop got a
                # little tighter; it does not say that three hundred agreeing
                # observations were about something else.
                #
                # LIVE 2026-09-02, the sitting after one that ended knowing the
                # rule at eighty six per cent: she carried it in, looked ahead
                # from the first move — and this wiped it at move twenty nine,
                # on a board she was reading perfectly. The sitting after that
                # inherited the wreckage and looked ahead on eighteen moves of
                # a hundred and forty seven.
                self._started_again = True
                continue
            if not self._started_again:
                # What she learned was about a different thing.
                #
                # Every observation until now compared a board with a score
                # stuck to it against rules about a board. Those are not
                # evidence about the board, and left in the counts they hold a
                # correct rule below the bar for as long as they outnumber
                # what came after.
                #
                # ONCE. Furniture is found a place at a time — a title, then a
                # label, then a score — and starting again at each of them
                # means the count never gets past one or two before it is wiped
                # again. LIVE 2026-08-29: "1 place(s) smaller ... 2 ... 3 ... 4
                # — starting again" through a whole run, "2 move(s) watched"
                # after eighteen moves, and no rule ever formed. The correction
                # is worth making the first time and never again.
                self._started_again = True
                logger.info(
                    "the thing itself is %s, not what surrounds it — starting again",
                    f"{len(self.counters)} place(s) smaller",
                )
                self.right.clear()
                self.tried.clear()
                self.right_when_it_moved.clear()
                self.tried_when_it_moved.clear()
                self.seen = 0
                self.moved = 0
                self.recent.clear()


    def learned_through_a_different_reading(self) -> None:
        """Forget what was counted while she was reading the thing wrongly.

        Evidence is only evidence under the conditions it was gathered in. A
        pair of arrangements is a claim about what her act does to the thing,
        and if the grid those arrangements were laid into was the wrong shape
        then the pair is about something else — not weak evidence, not old
        evidence, but a statement about a thing that does not exist.

        LIVE 2026-09-02: half a game read through a grid four by seven for a
        board four by four. The grid corrected itself mid-game, everything
        after it was right, and the rule still finished at sixty per cent
        because the wrong half outnumbered it — so she looked ahead on not one
        move of a hundred and thirty eight, on a board she was by then reading
        perfectly.

        Kept apart from starting again because the thing turned out smaller,
        which happens a place at a time and must only be done once. A change
        of shape is a different thing: it settles, so acting on it every time
        cannot thrash.
        """
        if not (self.seen or self.right):
            return
        logger.info(
            "what she learned was read through a different grid — starting again"
        )
        self.right.clear()
        self.tried.clear()
        self.right_when_it_moved.clear()
        self.tried_when_it_moved.clear()
        self.seen = 0
        self.moved = 0
        self.recent.clear()
        # And everything she settled about the SHAPE of the thing, which is
        # the part that made this necessary. Left standing, the places the
        # old grid taught her to expect make every reading in the new one
        # unreadable, and she counts nothing at all rather than counting the
        # wrong thing — which is the same amount of use.
        self.unreadable = 0
        self.arrivals = 0
        self._record.clear()
        self._how_full = 0.0
        self._ever_held.clear()
        self.counters.clear()
        self._started_again = False
        self.read_through = (0, 0)

    def _stands_apart(self, place: tuple[int, int], seen: Arrangement) -> bool:
        """Whether nothing has ever been beside this, in a thing that is packed.

        A place next to somewhere nothing has ever been is at the edge of the
        thing rather than in it. Where most of the layout has never held
        anything the test says nothing — a sparse thing is all edges — so it
        only answers where the thing is packed enough for a blank neighbour to
        be remarkable.
        """
        row, column = place
        room = max(1, seen.rows * seen.columns)
        if len(self._ever_held) * 2 < room:
            return False
        for beside in ((row - 1, column), (row + 1, column), (row, column - 1), (row, column + 1)):
            if not (0 <= beside[0] < seen.rows and 0 <= beside[1] < seen.columns):
                continue
            if beside not in self._ever_held:
                return True
        return False

    def watched(self, before: Arrangement, action: str, after: Arrangement) -> None:
        """One of her own moves, and what it did."""
        if not before.cells and not after.cells:
            return
        self.read_through = (before.rows, before.columns)
        if (before.rows, before.columns) != (after.rows, after.columns):
            # Two readings that disagree about the shape of the thing cannot
            # be compared, and a comparison that cannot be made is not
            # evidence against anything. Scored as a miss, one reading that
            # dropped a faint tile discredits every hypothesis at once and the
            # model never forms. LIVE 2026-08-26: nineteen moves in, nothing
            # worked out, on a board she was reading perfectly well.
            self.unreadable += 1
            return
        self._note_counters(before, after)
        # The part that behaves like one thing, with its furniture cropped out.
        here, there = self.the_thing(before), self.the_thing(after)
        if not here.cells or (here.rows, here.columns) != (there.rows, there.columns):
            self.unreadable += 1
            return
        self._note_arrivals(here, there)
        self._record.append((here.as_text(), str(action), there.as_text()))
        # Whether this act told one hypothesis from another.
        #
        # An act that changed nothing is agreed about by every rule there is:
        # they all predict what is already there. Counting that as evidence
        # lets a rule that is wrong about every act that MOVED something ride
        # to near-certainty on the many that did not. Measured 2026-08-26 in a
        # sliding puzzle, where the board is nearly full and most directions
        # do nothing: "slides and combines", 99% sure, in a world that does
        # nothing of the kind.
        told_apart = here.as_text() != there.as_text()
        if told_apart:
            self.moved += 1
        agreed: set[str] = set()
        for rule in RULES:
            predicted = rule.apply(here, action)
            if predicted is None:
                continue
            self.tried[rule.name] = self.tried.get(rule.name, 0) + 1
            right = _near_enough(predicted, there)
            if right:
                self.right[rule.name] = self.right.get(rule.name, 0) + 1
                agreed.add(rule.name)
            # An act that changed nothing is agreed about by every rule that
            # ALSO said nothing would change, and counting that would let a
            # rule ride to certainty on the many acts that did nothing.
            #
            # A rule that said something WOULD change has been refuted by the
            # same act, and a refutation is as informative as any move. The
            # gate threw those away with the rest: pressing a direction into
            # a wall, over and over, could not overturn a rule that claimed
            # the board would slide — measured on a carried rule that
            # survived fourteen straight contradictions of exactly that kind.
            claimed_a_change = predicted.as_text() != here.as_text()
            if told_apart or claimed_a_change:
                self.tried_when_it_moved[rule.name] = (
                    self.tried_when_it_moved.get(rule.name, 0) + 1
                )
                if right:
                    self.right_when_it_moved[rule.name] = (
                        self.right_when_it_moved.get(rule.name, 0) + 1
                    )
        self.seen += 1
        self.recent.append((str(action), frozenset(agreed)))
        del self.recent[:-REMEMBERED]

    # ── using it ─────────────────────────────────────────────────────────

    def rule(self) -> Rule | None:
        """The one that has been right most often, once there is enough to say."""
        if self.seen < ENOUGH_TO_TRUST:
            return None
        # Where things do move, a rule has to be right about the acts that
        # moved them. Where nothing has ever moved, there is nothing to be
        # right about and the absence of movement is itself the finding.
        anything_moves = self.moved >= ENOUGH_TO_TRUST
        if 0 < self.moved < ENOUGH_TO_TRUST:
            # It moves, and not yet often enough to tell one rule from
            # another.
            #
            # An act that changed nothing is agreed about by every rule there
            # is, so the raw counts in between elect whichever rule claims
            # least — and on a board where two of four directions do nothing
            # from the opening position, "this does not move" wins three of
            # the first four comparisons and is adopted.
            #
            # A rule that says nothing ever changes takes her search off the
            # board: every move looks identical, looking ahead returns
            # nothing, and she plays blind for the rest of the run. LIVE
            # 2026-09-04: "I can see what my moves do here now — this does not
            # move — right 75% of 4", on the fifth move of a game.
            #
            # Nothing is a fine answer here. She acts and looks instead, which
            # is what fills these counts.
            return None
        best: tuple[float, Rule] | None = None
        for rule in RULES:
            tried = (
                self.tried_when_it_moved.get(rule.name, 0)
                if anything_moves
                else self.tried.get(rule.name, 0)
            )
            if tried < ENOUGH_TO_TRUST:
                continue
            right = (
                self.right_when_it_moved.get(rule.name, 0)
                if anything_moves
                else self.right.get(rule.name, 0)
            )
            share = right / tried
            if share < OFTEN_ENOUGH:
                continue
            # Ties go to the rule that claims the LEAST.
            #
            # RULES runs strongest claim first, so taking a later one on equal
            # evidence takes the weaker. Where two hypotheses have never once
            # disagreed about anything she has seen, the extra thing the fuller
            # one claims is the part nothing supports. Measured 2026-08-27 in a
            # puzzle whose things are all different: "steps and combines" and
            # "steps" predict identically, because no two equal things ever
            # meet, and combining was a claim about a thing that never happened.
            if best is None or share >= best[0]:
                best = (share, rule)
        return best[1] if best else None

    def _closest(self) -> tuple[str, int, int] | None:
        """The rule with the best share so far, whether or not it is enough."""
        moving = self.moved >= ENOUGH_TO_TRUST
        best: tuple[float, str, int, int] | None = None
        for rule in RULES:
            tried = (
                self.tried_when_it_moved.get(rule.name, 0)
                if moving
                else self.tried.get(rule.name, 0)
            )
            if not tried:
                continue
            right = (
                self.right_when_it_moved.get(rule.name, 0)
                if moving
                else self.right.get(rule.name, 0)
            )
            share = right / tried
            if best is None or share > best[0]:
                best = (share, rule.name, right, tried)
        return (best[1], best[2], best[3]) if best else None

    def still_standing(self) -> tuple[Rule, ...]:
        """Every rule her evidence has not yet ruled out.

        A rule is out when it has been tried enough to judge and has been
        wrong too often. One with too little against it is still in, because
        the thing she does not have evidence about is exactly the thing worth
        acting to find out.
        """
        moving = self.moved >= ENOUGH_TO_TRUST
        standing: list[Rule] = []
        for rule in RULES:
            tried = (
                self.tried_when_it_moved.get(rule.name, 0)
                if moving
                else self.tried.get(rule.name, 0)
            )
            if tried < ENOUGH_TO_TRUST:
                standing.append(rule)
                continue
            right = (
                self.right_when_it_moved.get(rule.name, 0)
                if moving
                else self.right.get(rule.name, 0)
            )
            if right / tried >= OFTEN_ENOUGH:
                standing.append(rule)
        return tuple(standing)

    def what_this_would_settle(self, arrangement: Arrangement, action: str) -> float:
        """How much of the question this act would answer, from 0 to 1.

        The rules still standing disagree about some acts and agree about
        others. An act they all foretell the same way can be right or wrong
        but cannot tell her WHICH of them was right, so doing it leaves her
        exactly as unsure as before. An act they split over settles something
        whatever happens.

        This is the whole of acting to find out, and it is the reason
        a person pushes a thing one way early on and then never needs to
        again: nought when one rule is left or they all agree, and largest
        when the surviving rules are split evenly over what comes next.
        """
        standing = self.still_standing()
        if len(standing) < 2:
            return 0.0
        seen: list[Arrangement] = []
        counts: list[int] = []
        for rule in standing:
            try:
                foretold = rule.apply(self.the_thing(arrangement), action)
            except (AttributeError, TypeError, ValueError):
                foretold = None
            if foretold is None:
                continue
            for index, already in enumerate(seen):
                if _near_enough(foretold, already):
                    counts[index] += 1
                    break
            else:
                seen.append(foretold)
                counts.append(1)
        if len(counts) < 2:
            return 0.0
        # How split they are, not how many there are. Two rules that disagree
        # settle the question outright; nine that agree and one that does not
        # barely narrow it.
        total = sum(counts)
        largest = max(counts)
        return (total - largest) / total

    def confidence(self) -> float:
        """How often the rule she is using has been right about a real move."""
        rule = self.rule()
        if rule is None:
            return 0.0
        if self.moved >= ENOUGH_TO_TRUST:
            tried = self.tried_when_it_moved.get(rule.name, 0)
            right = self.right_when_it_moved.get(rule.name, 0)
        else:
            tried = self.tried.get(rule.name, 0)
            right = self.right.get(rule.name, 0)
        return (right / tried) if tried else 0.0

    def _note_arrivals(self, before: Arrangement, after: Arrangement) -> None:
        """Whether anything turned up that she did not put there.

        A rule about what her own act moves is not wrong because something
        arrived it never claimed to know about, and this already allows for
        that when scoring one. Allowing for a thing is not the same as knowing
        it: whether the world puts something new in front of her between her
        acts is a fact about the world, and it decides what kind of problem
        she is in.
        """
        if after.occupied() > before.occupied():
            self.arrivals += 1

    def fullness(self) -> float:
        """How full the thing has been, on average, across every look."""
        return self._how_full / self._looks if self._looks else 0.0

    def world_adds_things(self) -> bool:
        """Whether the world puts something new in front of her between acts.

        Not how often. A world either does this or it does not, and the share
        is misleading where her own acts take things away at the same time: on
        a board that merges two things into one and is then dealt a third, the
        count comes back level and nothing looks to have arrived. Twice is the
        world; once could be a misreading.

        Counting is the weaker of the two tests and it is what perception can
        answer on its own. The stronger one reads the whole record — the same
        board, the same key, two different results — and that is an inference
        rather than an observation, so it is asked from the layer that does
        inference. :meth:`what_she_saw_happen` hands it over.
        """
        return self.arrivals >= TWICE_IS_THE_WORLD

    def what_she_saw_happen(self) -> tuple[tuple[str, str, str], ...]:
        """Every move of hers: what was there, what she did, what happened.

        Watching is perception's job and reading what the watching proves is
        not. The one thing that can show a quantity she is not reading is this
        record in order, so it leaves here whole and
        :func:`core.cognition.something_she_cannot_see.what_she_cannot_see`
        asks the question of it.
        """
        return tuple(self._record)

    def told_these_report(self, places: Iterable[tuple[int, int]]) -> int:
        """Places she has established elsewhere are readouts rather than things.

        This works out for itself which places sit still and keep saying
        something different, and it needs many observations to do it — while
        every observation it is learning from is scored against a rule that is
        wrong about a score every single time. So it cannot get the evidence
        until it has the answer.

        The split between what rearranges and what reports is worked out
        elsewhere, from the same acts, and settles far sooner. Told, this
        starts from what she already knows rather than from nothing. Measured
        on a board with a score above it: the right rule scored nought out of
        five, because the score changed under it on every one of them.
        """
        told = {
            (int(row), int(column))
            for row, column in places or ()
            if (int(row), int(column)) not in self._a_place
        }
        fresh = told - self.counters
        self.counters |= told
        return len(fresh)

    def the_thing(self, arrangement: Arrangement) -> Arrangement:
        """The part of a reading that behaves like one thing.

        Furniture surrounds a thing; it is not most of it. Where what has
        never been empty is most of what is there, nothing has been told apart
        yet and cropping would leave a handful of cells that every rule agrees
        about for want of anything to disagree over.
        """
        if not self.counters or len(self.counters) * 2 >= max(1, arrangement.occupied()):
            return arrangement
        return arrangement.without(self.counters)

    def expect(self, arrangement: Arrangement, action: str) -> Arrangement | None:
        """What this thing would look like after that, without doing it.

        ``None`` when she has not worked out how it moves, which is an answer
        and not a failure: she acts and looks instead.
        """
        rule = self.rule()
        if rule is None:
            return None
        return rule.apply(self.the_thing(arrangement), action)

    def expect_all(
        self, arrangement: Arrangement, actions: Sequence[str]
    ) -> dict[str, Arrangement]:
        """Every way this could go from here, for the moves really available."""
        futures: dict[str, Arrangement] = {}
        for action in actions:
            imagined = self.expect(arrangement, action)
            if imagined is not None:
                futures[action] = imagined
        return futures

    # ── keeping it ───────────────────────────────────────────────────────

    def as_memory(self) -> dict[str, Any]:
        """What she worked out, in a form that survives the process."""
        return {
            "right": dict(self.right),
            "tried": dict(self.tried),
            "seen": self.seen,
            "arrivals": self.arrivals,
            "moved": self.moved,
            "right_when_it_moved": dict(self.right_when_it_moved),
            "tried_when_it_moved": dict(self.tried_when_it_moved),
            "read_through": list(self.read_through),
        }

    @classmethod
    def from_memory(cls, held: dict[str, Any], trust: float = 1.0) -> "HowItMoves":
        """What she worked out last time, discounted.

        Under full trust on purpose. Something she worked out yesterday is
        evidence about today, not a fact about it, and a handful of moves that
        disagree should be able to overturn it — which they can only do if
        what came back is not overwhelming.
        """
        if not isinstance(held, dict):
            return cls()
        share = max(0.0, min(1.0, float(trust)))

        def carried(counts: Any) -> dict[str, int]:
            if not isinstance(counts, dict):
                return {}
            return {
                str(name): int(round(float(value) * share))
                for name, value in counts.items()
                if isinstance(value, (int, float))
            }

        shape = held.get("read_through") or ()
        try:
            through = (int(shape[0]), int(shape[1])) if len(shape) == 2 else (0, 0)
        except (TypeError, ValueError, IndexError):
            # not a failure: a shape that is not two numbers is not a shape.
            through = (0, 0)
        return cls(
            right=carried(held.get("right")),
            tried=carried(held.get("tried")),
            seen=int(round(float(held.get("seen") or 0) * share)),
            arrivals=int(round(float(held.get("arrivals") or 0) * share)),
            moved=int(round(float(held.get("moved") or 0) * share)),
            right_when_it_moved=carried(held.get("right_when_it_moved")),
            tried_when_it_moved=carried(held.get("tried_when_it_moved")),
            read_through=through,
        )

    def says(self) -> str:
        """What she has worked out, in a line, for whoever has to answer for it."""
        rule = self.rule()
        if rule is None:
            said = f"how this moves is not worked out yet ({self.seen} move(s) watched"
            if self.unreadable:
                said = f"{said}, {self.unreadable} unreadable"
            # Which one came closest, and how close.
            #
            # "Not worked out yet" is true and says nothing about why: whether
            # nothing fits, or one thing nearly fits and is being held out by a
            # handful of misreadings. Those want different answers and looked
            # identical from outside. LIVE 2026-08-29: twenty-four moves
            # watched on a board she was reading correctly, and no way to tell
            # which it was without attaching a debugger to a live run.
            closest = self._closest()
            if closest is not None:
                name, right, tried = closest
                said = f"{said}; closest is {name} at {right}/{tried}"
            return f"{said})"
        return f"this {rule.name} — right {self.confidence():.0%} of {self.tried.get(rule.name, 0)}"


def prediction_held(predicted: Any, seen: Any) -> bool:
    """Whether a foretold arrangement is what turned up.

    The same test the rules are scored by, offered to whoever holds a
    prediction. A claim about what a move does is only interesting if being
    wrong about it means something, and the meaning has to be the same
    wherever it is checked — a rule discredited here and credited there is two
    models wearing one name.
    """

    try:
        return _near_enough(predicted, seen)
    except (AttributeError, TypeError):
        return False


def _near_enough(
    predicted: Arrangement, seen: Arrangement, ignoring: set[tuple[int, int]] | None = None
) -> bool:
    """Whether a rule got right the part it claimed to know about.

    A world that adds something of its own — a dealt tile, a new row, a clock
    ticking — has not falsified a rule about what her own act moves. What the
    rule must get right is every place it said something would be.
    """
    if predicted.rows != seen.rows or predicted.columns != seen.columns:
        return False
    skip = ignoring or set()
    claimed = {
        (cell.row, cell.column): cell.says
        for cell in predicted.cells
        if (cell.row, cell.column) not in skip
    }
    actual = {(cell.row, cell.column): cell.says for cell in seen.cells}
    for place, said in claimed.items():
        if actual.get(place) != said:
            return False
    # And nothing may vanish that the rule kept: an extra arrival is the
    # world's business, a disappearance would be the rule's mistake.
    return True


#: Once could be a misreading of a faint thing. Twice is the world.
TWICE_IS_THE_WORLD = 2

#: The least number of looks before anything can be called furniture, for a
#: thing so small that its own size would be a lower bar.
STILL_ENOUGH_TO_JUDGE = 6

#: How often a place has to be occupied before it counts as never empty. Not
#: quite one, because a reading can drop something faint for a frame and that
#: is not the same as the place being free.
ALWAYS_THERE = 0.95

#: How unlikely missing an ordinary place's empty moments has to be before
#: never having seen one empty is evidence about the place rather than about
#: how full the thing is. Twenty to one.
UNLIKELY_TO_HAVE_MISSED = 0.05

#: How full a thing can get before "never seen empty" stops being able to mean
#: anything at all. At this point an ordinary place is free so rarely that no
#: number of looks settles it, and nothing is called furniture.
TOO_FULL_TO_TELL = 0.995


def _looks_to_expect_an_empty(fullness: float, among: int = 1) -> int:
    """How many looks before an ordinary place would surely have been caught free.

    A place in a thing that is ``fullness`` full is free the rest of the time,
    so the chance of missing every one of its free moments falls off by that
    factor each look. This is the look at which missing them all would be a
    twenty-to-one result — before it, never having seen a place empty says
    nothing about the place.

    ``among`` is how many places are being asked the same question. Asking it
    of many at once makes a rare answer common, so the odds each one has to
    clear are shared out between them.
    """
    full = max(0.0, min(1.0, float(fullness or 0.0)))
    if full <= 0.0:
        return STILL_ENOUGH_TO_JUDGE
    if full >= TOO_FULL_TO_TELL:
        return _NEVER
    from math import ceil, log

    odds = UNLIKELY_TO_HAVE_MISSED / max(1, int(among))
    return max(STILL_ENOUGH_TO_JUDGE, int(ceil(log(odds) / log(full))))


#: A bar no run reaches, for a thing so full that occupancy tells her nothing.
_NEVER = 1 << 30
