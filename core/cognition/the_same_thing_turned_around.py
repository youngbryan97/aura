"""Two situations that are the same situation, and one experience for both.

The opening of a game between two very strong Go players: two stones on a
board of three hundred and sixty one points. Nothing can be searched here —
the branching is three hundred and fifty nine and the game is two hundred
moves long — and both of them play at once, confidently.

They are not looking further than anybody else. They are looking at far fewer
things. A board with two stones has four corners on it and the four corners
are the same corner: whatever is true of one is true of the others turned
round. So a choice among three hundred and fifty nine points is a choice among
a handful of kinds of point, and everything learned about one corner for the
rest of their lives is about all four.

That is the difference between a rule of thumb and an abstraction. A rule of
thumb says corners are good. This says these situations ARE that situation,
so every hour spent on one was spent on all of them.

Nothing here is told which turnings exist. A turning is a way of relabelling
where things are, and the ones worth having are the ones under which what she
has already worked out still holds. Both halves come from her own record:

    where they come from   which places things move BETWEEN, which she learns
                           by watching them move. A turning has to leave that
                           alone, so the relabellings that do are the only
                           ones worth testing.
    which ones are real    a suggested map is kept only if every transition
                           she has watched still holds after it is applied —
                           which means finding, for each of her acts, which
                           act it turns into. Checked against what she has
                           worked out the world does, because checking it
                           against her record alone asks her to have already
                           SEEN the turned situation, and in any world worth
                           the name she never has.

The second half is what makes it an abstraction rather than a coincidence. A
map that lines up one pair of situations is nothing. A map under which
everything she knows about how the world moves is still true is a fact about
the world.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from itertools import permutations
from typing import Any

__all__ = ["ATurning", "turnings_that_hold", "which_are_the_same"]

Places = Mapping[Any, Any]


@dataclass(frozen=True)
class ATurning:
    """A way of relabelling where things are, and what it does to her acts."""

    where: tuple[tuple[Any, Any], ...]
    acts_become: tuple[tuple[str, str], ...]
    held_over: int = 0

    @property
    def sends(self) -> dict[Any, Any]:
        return dict(self.where)

    @property
    def turns_acts_into(self) -> dict[str, str]:
        return dict(self.acts_become)

    def applied_to(self, places: Places) -> dict[Any, Any]:
        sends = self.sends
        return {sends[where]: what for where, what in places.items() if where in sends}

    def describe(self) -> str:
        acts = ", ".join(f"{was}->{now}" for was, now in self.acts_become)
        return f"a turning that holds over {self.held_over} of them ({acts})"


def _what_moves_between(
    watched: Sequence[tuple[Places, str, Places]],
) -> set[frozenset[Any]]:
    """Which places things have been seen to move between.

    Learned rather than declared. A thing that was here and is now there means
    here and there are joined, whatever the world is — a board, a set of
    folders, a rota. What comes out is the shape of the space she is acting
    in, and a turning of that space has to leave it alone.
    """
    joined: set[frozenset[Any]] = set()
    for before, _act, after in watched:
        for where, what in before.items():
            if after.get(where) == what:
                continue
            for other, landed in after.items():
                if landed == what and other != where:
                    joined.add(frozenset((where, other)))
    return joined


def _turnings_of_the_space(
    every: Sequence[Any], joined: set[frozenset[Any]]
) -> Iterator[dict[Any, Any]]:
    """Relabellings that leave which-places-join-which alone.

    All of them, where there are few enough places to look at all of them, and
    none at all where there are not — because a sample of the turnings of a
    space is not the turnings of a space, and saying nothing is better than
    saying some of it.

    Handed back one at a time rather than as a list, because most of what
    preserves the joins is not a real turning and only trying it says so. Cut
    the list short and what gets cut is arbitrary: with a row of three places
    all joined to each other, every shuffle within a row comes first and the
    mirror never gets looked at.
    """
    if not joined or len(every) > 8:
        return
    for order in permutations(every):
        sends = dict(zip(every, order, strict=True))
        if all(frozenset(map(sends.get, pair)) in joined for pair in joined):
            yield sends


def _suggested_by(one: Places, other: Places) -> dict[Any, Any] | None:
    """The map between two situations holding the same things elsewhere.

    Only where it is forced. Where two places hold the same value there is
    more than one way to line them up, and guessing which is how a coincidence
    becomes a belief.
    """
    if len(one) != len(other) or not one:
        return None
    by_value_here: dict[Any, list[Any]] = {}
    by_value_there: dict[Any, list[Any]] = {}
    for where, what in one.items():
        by_value_here.setdefault(what, []).append(where)
    for where, what in other.items():
        by_value_there.setdefault(what, []).append(where)
    if set(by_value_here) != set(by_value_there):
        return None
    sends: dict[Any, Any] = {}
    for what, here in by_value_here.items():
        there = by_value_there[what]
        if len(here) != len(there) or len(here) != 1:
            return None
        sends[here[0]] = there[0]
    return sends


def _grown_to_cover(sends: Mapping[Any, Any], every: Sequence[Any]) -> dict[Any, Any] | None:
    """A map over some places extended to all of them, when one extension fits.

    A turning has to say where EVERY place goes, and a pair of situations only
    ever shows the occupied ones. Where what is left over is small enough to
    settle by trying, it is settled; where it is not, this says so rather than
    picking one.
    """
    sends = dict(sends)
    missing = [one for one in every if one not in sends]
    landing = [one for one in every if one not in set(sends.values())]
    if len(missing) != len(landing):
        return None
    if not missing:
        return sends
    if len(missing) > 6:
        # Too many ways to fill it in for any of them to mean anything.
        return None
    # The first filling that keeps it a bijection, which is all of them here.
    for order in permutations(landing):
        grown = dict(sends)
        grown.update(dict(zip(missing, order, strict=True)))
        if len(set(grown.values())) == len(grown):
            return grown
    return None


def turnings_that_hold(
    watched: Sequence[tuple[Places, str, Places]],
    *,
    every_place: Sequence[Any],
    acts: Sequence[str],
    expect: Callable[[Places, str], Places | None] | None = None,
    most: int = 8,
) -> list[ATurning]:
    """The relabellings under which everything she has watched is still true.

    ``watched`` is her own record: what things looked like, what she did, what
    they looked like after. A turning is kept only when, for every one of
    those, turning the before and turning the after leaves a transition she
    would also have believed — under some renaming of her acts, which is found
    here rather than supplied.
    """
    held: list[ATurning] = []
    for sends in _turnings_of_the_space(list(every_place), _what_moves_between(watched)):
        becomes = _what_the_acts_become(sends, watched, acts, expect)
        if becomes is None:
            continue
        held.append(
            ATurning(
                where=tuple(sorted(sends.items(), key=repr)),
                acts_become=tuple(sorted(becomes.items())),
                held_over=len(watched),
            )
        )
        if len(held) >= most:
            return held
    if held:
        return held

    suggested: list[dict[Any, Any]] = []
    if not suggested:
        # Nothing to go on from the shape of the space, so fall back to
        # situations that happen to be turnings of each other. That needs her
        # to have SEEN both, which in a large world she never will, so it is
        # the weaker of the two and second for that reason.
        states = [one for before, _act, after in watched for one in (before, after)]
        seen: set[tuple[tuple[Any, Any], ...]] = set()
        for at, one in enumerate(states):
            for other in states[at + 1 :]:
                forced = _suggested_by(one, other)
                if forced is None:
                    continue
                grown = _grown_to_cover(forced, every_place)
                if grown is None:
                    continue
                key = tuple(sorted(grown.items(), key=repr))
                if key in seen:
                    continue
                seen.add(key)
                suggested.append(grown)
                if len(suggested) >= most * 4:
                    break
            if len(suggested) >= most * 4:
                break

    for sends in suggested:
        becomes = _what_the_acts_become(sends, watched, acts, expect)
        if becomes is None:
            continue
        held.append(
            ATurning(
                where=tuple(sorted(sends.items(), key=repr)),
                acts_become=tuple(sorted(becomes.items())),
                held_over=len(watched),
            )
        )
        if len(held) >= most:
            break
    return held


def _what_the_acts_become(
    sends: Mapping[Any, Any],
    watched: Sequence[tuple[Places, str, Places]],
    acts: Sequence[str],
    expect: Callable[[Places, str], Places | None] | None,
) -> dict[str, str] | None:
    """Which act each act turns into, or None when no renaming works.

    This is the test. A relabelling of places that lines two situations up is
    a coincidence; one under which every move she has watched still does what
    it did is a fact about the world.

    Where she has worked out what the world does, that is what it is checked
    against: turn the situation, do the turned act, and it should come out as
    the turned result. Without it, all she can do is look for the turned
    transition in her record, which needs her to have been there already —
    so it finds far less, and says so by finding nothing rather than by
    finding something weaker.
    """

    def turned(places: Places) -> dict[Any, Any]:
        return {sends[where]: what for where, what in places.items() if where in sends}

    becomes: dict[str, str] = {}
    for act in acts:
        mine = [one for one in watched if one[1] == act]
        if not mine:
            continue
        for other in acts:
            if expect is not None:
                got = True
                for before, _act, after in mine:
                    came = expect(turned(before), other)
                    if came is None or _as_key(came) != _as_key(turned(after)):
                        got = False
                        break
                if got:
                    becomes[act] = other
                    break
                continue
            done = {
                _as_key(before): _as_key(after)
                for before, one_act, after in watched
                if one_act == other
            }
            want = {_as_key(turned(before)): _as_key(turned(after)) for before, _a, after in mine}
            shared = set(want) & set(done)
            if shared and all(want[one] == done[one] for one in shared):
                becomes[act] = other
                break
        else:
            return None
    if not becomes or len(set(becomes.values())) != len(becomes):
        return None
    return becomes


def _as_key(places: Places) -> Hashable:
    return tuple(sorted(((repr(where), repr(what)) for where, what in places.items())))


def which_are_the_same(
    places: Places,
    turnings: Sequence[ATurning],
) -> Hashable:
    """One name for a situation and every turning of it.

    Two situations with the same name are the same situation, so anything she
    worked out about either is about both. This is where the saving is: it
    turns one experience into as many as there are turnings.
    """
    every = [_as_key(places)]
    for turning in turnings:
        every.append(_as_key(turning.applied_to(places)))
    return min(every, key=repr)
