"""Building a way of making words, rather than switching on one that was written.

She could admit a way of building words when the evidence called for it, and
what she admitted was always a function somebody had already written. The set
of ways available never grew:

    M(active, t+1)    grew
    M(available, t+1) = M(available, t)

Deciding that a dormant designer-written constructor deserves admission is a
smaller thing than building one. This is the other half.

A constructor here is a RECIPE — data, in a small language of its own — and it
is interpreted rather than looked up. Three ways of making words out of words:

    in sequence     apply several words one after another
    undone          the word that puts back what a word moved
    over and over   apply one word to its own output, several times

and a recipe may be one of those followed by another, so the space is a closure
and not a list. "Apply two in sequence" is a point in it, which is why the
constructor that used to be written out in source is now one of the things she
can arrive at rather than the only thing she can be handed.

How deep a recipe goes is read off the problem. A family whose states are five
long cannot need a chain longer than five, so the depth is bounded by what is
in front of her rather than by a number somebody picked.

What is saved is the recipe. A name resolved against a source registry can only
ever name what the source already has; a recipe reconstructs something the
source never contained.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Sequence

__all__ = [
    "IN_SEQUENCE",
    "OVER_AND_OVER",
    "Recipe",
    "UNDONE",
    "a_constructor_she_built",
    "build",
    "every_recipe",
    "read_back",
    "written_down",
]

logger = logging.getLogger("Aura.AConstructorSheBuilt")

#: Apply several words one after another.
IN_SEQUENCE = "in sequence"

#: The word that puts back what a word moved.
UNDONE = "undone"

#: Apply one word to its own output, several times over.
OVER_AND_OVER = "over and over"

WAYS = (IN_SEQUENCE, UNDONE, OVER_AND_OVER)


@dataclass(frozen=True)
class Recipe:
    """How to make words out of words, written as data.

    ``then`` makes the space a closure: a recipe may be followed by another,
    so what she can build is not the three ways but everything they compose to.
    """

    kind: str
    depth: int = 2
    then: "Recipe | None" = None

    @property
    def name(self) -> str:
        if self.kind == UNDONE:
            said = "undone"
        elif self.kind == OVER_AND_OVER:
            said = f"{self.depth} times over"
        else:
            said = f"{self.depth} in sequence"
        return f"{said}, then {self.then.name}" if self.then else said

    def how_long(self) -> int:
        return 1 + (self.then.how_long() if self.then else 0)


@dataclass(frozen=True)
class _InSequence:
    """Several ways of saying where a value comes from, used in turn."""

    steps: tuple[Callable[[int, int], int], ...]

    def __call__(self, index: int, size: int) -> int:
        at = index
        for step in self.steps:
            at = step(at, size) % size
        return at


@dataclass(frozen=True)
class _Undone:
    """The word that puts back what a word moved.

    Worked out for whatever size it is asked about, so it is a rule and not a
    record. Where the word is not a rearrangement at that size there is nothing
    to undo, and refusing is the only honest answer.
    """

    word: Callable[[int, int], int]

    def __call__(self, index: int, size: int) -> int:
        where = [self.word(at, size) % size for at in range(size)]
        if len(set(where)) != size:
            raise ValueError("nothing to undo: it does not move things one for one")
        return where.index(index % size)


@dataclass(frozen=True)
class _OverAndOver:
    """One word applied to its own output, several times."""

    word: Callable[[int, int], int]
    times: int

    def __call__(self, index: int, size: int) -> int:
        at = index
        for _ in range(max(1, self.times)):
            at = self.word(at, size) % size
        return at


def _in_sequence(words: dict[str, Any], depth: int) -> dict[str, Any]:
    made: dict[str, Any] = {}
    names = list(words)

    def walk(chain: list[str]) -> None:
        if len(chain) == depth:
            made[", then ".join(chain)] = _InSequence(
                tuple(words[name] for name in chain)
            )
            return
        for name in names:
            if chain and name == chain[-1]:
                continue
            walk([*chain, name])

    walk([])
    return made


def _undone(words: dict[str, Any], _depth: int) -> dict[str, Any]:
    return {f"{name}, undone": _Undone(word) for name, word in words.items()}


def _over_and_over(words: dict[str, Any], depth: int) -> dict[str, Any]:
    return {
        f"{name}, {depth} times over": _OverAndOver(word, depth)
        for name, word in words.items()
    }


_HOW: dict[str, Callable[[dict[str, Any], int], dict[str, Any]]] = {
    IN_SEQUENCE: _in_sequence,
    UNDONE: _undone,
    OVER_AND_OVER: _over_and_over,
}


def build(recipe: Recipe) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Turn a recipe into the constructor it describes.

    The interpreter is what makes the recipe a thing she can build rather than
    a thing she can name. Nothing here can run anything that is not one of the
    three ways and a depth.
    """

    def make(words: dict[str, Any]) -> dict[str, Any]:
        step = _HOW.get(recipe.kind)
        if step is None:
            return {}
        made = step(dict(words), max(2, int(recipe.depth)))
        if recipe.then is None:
            return made
        onward = dict(words)
        onward.update(made)
        made.update(build(recipe.then)(onward))
        return made

    # Carried so that what she built can be written down as what it is. A
    # constructor saved by name resolves against the source; this one has to
    # reconstruct from its own description.
    make.recipe = recipe  # type: ignore[attr-defined]
    return make


def every_recipe(deepest: int) -> Iterator[Recipe]:
    """Every recipe, shortest first, down to a depth the problem sets.

    Shortest first for the same reason everywhere else: the shortest recipe
    that works is the one to believe, and each step deeper multiplies what has
    to be walked.
    """
    depths = tuple(range(2, max(2, int(deepest)) + 1))
    plain = [Recipe(kind=UNDONE)]
    plain += [Recipe(kind=kind, depth=depth) for depth in depths
              for kind in (IN_SEQUENCE, OVER_AND_OVER)]
    yield from plain
    for first in plain:
        for second in plain:
            yield Recipe(kind=first.kind, depth=first.depth, then=second)


def a_constructor_she_built(
    transitions: Sequence[tuple[Sequence[Any], Sequence[Any]]],
    *,
    now_sayable: Callable[[], bool],
) -> Recipe | None:
    """Build a way of making words until the family in front of her is sayable.

    Tried in order of how much it enlarges the search, and admitted only when
    it makes sayable something that was not. A constructor that changes nothing
    has multiplied what she must walk in exchange for nothing.
    """
    from core.cognition.an_invented_kind import WAYS_TO_BUILD, WHERE_FROM

    if now_sayable():
        return None
    longest = max((len(before) for before, _ in transitions), default=2)
    for recipe in _cheapest_first(every_recipe(longest), len(WHERE_FROM)):
        name = f"a way she built: {recipe.name}"
        if name in WAYS_TO_BUILD:
            continue
        WAYS_TO_BUILD[name] = build(recipe)
        try:
            if now_sayable():
                logger.info(
                    "she built a constructor nobody wrote: %s", recipe.name
                )
                return recipe
        except (TypeError, ValueError, KeyError):
            pass
        WAYS_TO_BUILD.pop(name, None)
    return None


def _how_many_it_makes(recipe: Recipe, words: int) -> int:
    """How large a vocabulary a recipe opens, without building it.

    Counting steps in the recipe is the wrong unit. Two words in sequence and
    five words in sequence are both one step, and over five words the second
    makes a vocabulary two hundred times larger — while the search over the
    meanings they compose to is quadratic in the vocabulary. Trying them in
    written order spends minutes on the expensive one before reaching the cheap
    one that works.

    Worked out rather than measured, because measuring means building, and
    building the expensive ones is most of what there is to avoid.
    """
    if recipe.kind == IN_SEQUENCE:
        made = words * max(1, words - 1) ** max(0, int(recipe.depth) - 1)
    else:
        made = words
    total = words + made
    if recipe.then is None:
        return total
    return _how_many_it_makes(recipe.then, total)


def _cheapest_first(recipes: Iterator[Recipe], words: int) -> list[Recipe]:
    """Recipes in order of how much they enlarge what has to be walked."""
    weighed = sorted(
        recipes, key=lambda recipe: (_how_many_it_makes(recipe, words), recipe.how_long())
    )
    return weighed


def written_down(recipe: Recipe) -> dict[str, Any]:
    """The recipe as plain data, so what she built survives a restart."""
    return {
        "kind": recipe.kind,
        "depth": int(recipe.depth),
        "then": written_down(recipe.then) if recipe.then else None,
    }


def read_back(row: Any) -> Recipe | None:
    """A recipe from what was written down, or nothing when it does not read."""
    if not isinstance(row, dict):
        return None
    kind = str(row.get("kind") or "")
    if kind not in WAYS:
        return None
    try:
        depth = int(row.get("depth") or 2)
    except (TypeError, ValueError):
        return None
    return Recipe(kind=kind, depth=max(2, depth), then=read_back(row.get("then")))
