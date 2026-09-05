"""Two worlds that share a structure and share no surface.

Transfer is the thing that separates task learning from general intelligence,
and it is the easiest to fake: two worlds that look alike transfer for the
same reason two photographs of one thing look alike. So every pair here is
built from a hidden structural correspondence and a surface that is drawn
independently — different alphabets, different lengths, different names.

And every run carries negative controls: pairs whose surfaces are near
identical and whose structures differ. A system that transfers there is
matching appearances, and a study without them measures how similar the
evaluator made things look.
"""

from __future__ import annotations

import random
import string
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

__all__ = ["APairOfWorlds", "invent_the_worlds"]


#: The structures. Each is a function of a sequence, said in a way that
#: mentions no alphabet, no length and no domain — which is what makes it the
#: thing that could transfer.
def _rotate(row: tuple[Any, ...]) -> tuple[Any, ...]:
    return row[1:] + row[:1]


def _reverse(row: tuple[Any, ...]) -> tuple[Any, ...]:
    return tuple(reversed(row))


def _ends(row: tuple[Any, ...]) -> tuple[Any, ...]:
    made = list(row)
    if len(made) > 1:
        made[0], made[-1] = made[-1], made[0]
    return tuple(made)


def _pairs(row: tuple[Any, ...]) -> tuple[Any, ...]:
    made = list(row)
    for index in range(0, len(made) - 1, 2):
        made[index], made[index + 1] = made[index + 1], made[index]
    return tuple(made)


def _outward(row: tuple[Any, ...]) -> tuple[Any, ...]:
    """Each half rotates away from the middle. A permutation, checked.

    The first version moved every cell one place away from the centre and
    wrapped, which is not a permutation: two cells landed on one square and
    two squares were left empty, so the world handed the solver examples with
    holes in them and every structure that used it was unsolvable by
    anything. A generator that produces impossible instances measures the
    solver's willingness to refuse.
    """

    size = len(row)
    if size < 4:
        return row
    middle = size // 2
    left = list(row[:middle])
    right = list(row[middle:])
    left = left[1:] + left[:1]
    right = right[-1:] + right[:-1]
    return tuple(left + right)


def _is_a_permutation(structure: Callable[..., Any], size: int = 8) -> bool:
    """Whether a structure rearranges without losing or duplicating anything.

    Checked at import rather than trusted, because the one structure that was
    not a permutation produced examples with holes in them and made a whole
    transfer study unsolvable.
    """

    row = tuple(range(size))
    try:
        got = tuple(structure(row))
    except (TypeError, ValueError, IndexError):
        return False
    return sorted(got) == list(row)


THE_STRUCTURES: dict[str, Callable[[tuple[Any, ...]], tuple[Any, ...]]] = {
    "everything shifts one along, wrapping": _rotate,
    "the order is turned around": _reverse,
    "the two ends change places": _ends,
    "neighbours change places in twos": _pairs,
    "everything moves one place away from the middle": _outward,
}


for _name, _structure in THE_STRUCTURES.items():
    if not _is_a_permutation(_structure):
        raise AssertionError(
            f"{_name!r} loses or duplicates cells; an instance built from it "
            "cannot be solved by anything"
        )


@dataclass(frozen=True)
class AWorld:
    """One world: a surface, and examples in it."""

    name: str
    alphabet: tuple[Any, ...]
    shown: tuple[tuple[tuple[Any, ...], tuple[Any, ...]], ...]
    asked: tuple[Any, ...]
    answer: tuple[Any, ...]

    def is_right(self, said: Any) -> bool:
        try:
            return tuple(said) == self.answer
        except TypeError:
            return False


@dataclass(frozen=True)
class APairOfWorlds:
    """A world to learn in, a world to be tested in, and what links them."""

    name: str
    first: AWorld
    second: AWorld
    #: The structure both share, or the two that differ for a control.
    structure: str
    other_structure: str = ""

    @property
    def should_transfer(self) -> bool:
        """False for a control: it looks the same and it is not the same."""
        return not self.other_structure


_ALPHABETS: tuple[tuple[str, tuple[Any, ...]], ...] = (
    ("letters", tuple(string.ascii_lowercase)),
    ("numbers", tuple(range(100))),
    ("greek", tuple("αβγδεζηθικλμνξοπρστυφχψω")),
    ("words", tuple("north south east west up down left right in out".split())),
    ("marks", tuple("▲ ● ■ ◆ ★ ☾ ☀ ✦ ✚ ❖".split())),
)


def _a_world(
    rng: random.Random,
    name: str,
    alphabet: tuple[Any, ...],
    structure: Callable[[tuple[Any, ...]], tuple[Any, ...]],
    *,
    shown: int = 3,
) -> AWorld:
    rows: list[tuple[tuple[Any, ...], tuple[Any, ...]]] = []
    for _ in range(shown + 1):
        size = rng.randint(4, min(9, len(alphabet)))
        before = tuple(rng.sample(list(alphabet), size))
        rows.append((before, tuple(structure(before))))
    return AWorld(
        name=name,
        alphabet=alphabet,
        shown=tuple(rows[:shown]),
        asked=rows[shown][0],
        answer=rows[shown][1],
    )


def invent_the_worlds(
    seed: int,
    *,
    how_many: int = 50,
    controls: float = 0.3,
    shown_in_the_first: int = 3,
    shown_in_the_second: int = 1,
) -> tuple[APairOfWorlds, ...]:
    """Paired worlds, most sharing a structure and some deliberately not.

    ``controls`` is the share that must NOT transfer. They are not a garnish:
    a transfer number without them says how similar the evaluator made two
    things look.

    The second world shows ONE example on purpose. With three it is solvable
    from itself, so a system that has learned nothing scores what a system
    that has learned everything scores, and the transfer term is zero for
    both — which is what the first run of this measured and it measured
    nothing. One observation is genuinely ambiguous between several shapes,
    and a prior that has met the right one before is exactly what decides it.
    """

    rng = random.Random(seed ^ 0x7A17)
    names = sorted(THE_STRUCTURES)
    made: list[APairOfWorlds] = []
    for index in range(how_many):
        is_control = rng.random() < controls
        here = rng.choice(names)
        first_alphabet = rng.choice(_ALPHABETS)
        if is_control:
            # The same surface, a different structure. The trap.
            there = rng.choice([one for one in names if one != here])
            second_alphabet = first_alphabet
            pair = APairOfWorlds(
                name=f"pair {index} (control)",
                first=_a_world(
                    rng, f"world {index}a", first_alphabet[1],
                    THE_STRUCTURES[here], shown=shown_in_the_first,
                ),
                second=_a_world(
                    rng, f"world {index}b", second_alphabet[1],
                    THE_STRUCTURES[there], shown=shown_in_the_second,
                ),
                structure=here,
                other_structure=there,
            )
        else:
            # A different surface, the same structure. The thing being tested.
            second_alphabet = rng.choice(
                [one for one in _ALPHABETS if one[0] != first_alphabet[0]]
            )
            pair = APairOfWorlds(
                name=f"pair {index}",
                first=_a_world(
                    rng, f"world {index}a", first_alphabet[1],
                    THE_STRUCTURES[here], shown=shown_in_the_first,
                ),
                second=_a_world(
                    rng, f"world {index}b", second_alphabet[1],
                    THE_STRUCTURES[here], shown=shown_in_the_second,
                ),
                structure=here,
            )
        made.append(pair)
    return tuple(made)
