"""An ordering worked out on Tuesday should be language on Wednesday.

The positional shapes persist: solved, admitted, refactored, written down, and
back after a restart. The orderings over the CELLS did not. They were solved
for, applied once, and dropped — so the same question a week later cost the
same examples it cost the first time, and the newest part of the language was
the one part that could not accumulate.

Using a thing and having learned it are different, and only one of them
survives a restart.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from core.cognition.primitive_invention import Transition as T
from core.cognition.relation_language import RelationLanguage
from core.cognition.value_order import Ordering, solve_ordering


def _somewhere() -> Path:
    return Path(tempfile.mkdtemp()) / "language.json"


def test_an_ordering_is_written_down_and_comes_back() -> None:
    home = _somewhere()
    language = RelationLanguage(path=home)
    language.admit_order(solve_ordering([T((3, 1, 2), (1, 2, 3)), T((5, 9, 7), (5, 7, 9))]))
    language.save()

    again = RelationLanguage.load(home)
    assert len(again.orders) == 1
    restored = next(iter(again.orders.values()))
    assert isinstance(restored, Ordering)
    assert restored.apply((40, 11, 27)) == (11, 27, 40)


def test_a_restored_ordering_still_knows_its_cells() -> None:
    """Written form keys cells by repr; the questions do not.

    An ordering that survives the restart and then misses every lookup has not
    survived it.
    """

    home = _somewhere()
    language = RelationLanguage(path=home)
    language.admit_order(
        solve_ordering(
            [
                T(("pear", "fig", "date"), ("date", "fig", "pear")),
                T(("kiwi", "apple"), ("apple", "kiwi")),
            ]
        )
    )
    language.save()

    restored = next(iter(RelationLanguage.load(home).orders.values()))
    assert restored.apply(("zebra", "melon")) == ("melon", "zebra")


def test_what_was_learned_settles_a_world_that_shows_less() -> None:
    """The point of keeping it: the second question of a kind is cheaper."""

    home = _somewhere()
    language = RelationLanguage(path=home)
    language.admit_order(solve_ordering([T((3, 1, 2), (1, 2, 3)), T((5, 9, 7), (5, 7, 9))]))
    language.save()

    later = RelationLanguage.load(home)
    thin = [T((8, 2, 5), (2, 5, 8))]
    found = later.order_that_explains(thin)
    assert found is not None
    assert found.apply((40, 11, 27)) == (11, 27, 40)


def test_a_known_ordering_is_not_forced_onto_a_world_it_does_not_explain() -> None:
    home = _somewhere()
    language = RelationLanguage(path=home)
    language.admit_order(solve_ordering([T((3, 1, 2), (1, 2, 3)), T((5, 9, 7), (5, 7, 9))]))

    # A mirror is not this ordering, and must not be answered as one.
    assert language.order_that_explains([T((1, 2, 3), (3, 2, 1))]) is None
    # Nor is noise.
    assert language.order_that_explains([T((1, 2, 3), (9, 4, 7))]) is None
