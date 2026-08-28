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
    language.admit_order(solve_ordering(
            [
            T((3, 1, 2), (1, 2, 3)),
            T((5, 9, 7), (5, 7, 9)),
            T((2, 8, 4), (2, 4, 8)),
            T((6, 0, 3), (0, 3, 6)),
        ]
        ))
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
                T(("plum", "cherry", "berry"), ("berry", "cherry", "plum")),
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
    language.admit_order(solve_ordering(
            [
            T((3, 1, 2), (1, 2, 3)),
            T((5, 9, 7), (5, 7, 9)),
            T((2, 8, 4), (2, 4, 8)),
            T((6, 0, 3), (0, 3, 6)),
        ]
        ))
    language.save()

    later = RelationLanguage.load(home)
    thin = [T((8, 2, 5), (2, 5, 8))]
    found = later.order_that_explains(thin)
    assert found is not None
    assert found.apply((40, 11, 27)) == (11, 27, 40)


def test_a_known_ordering_is_not_forced_onto_a_world_it_does_not_explain() -> None:
    home = _somewhere()
    language = RelationLanguage(path=home)
    language.admit_order(solve_ordering(
            [
            T((3, 1, 2), (1, 2, 3)),
            T((5, 9, 7), (5, 7, 9)),
            T((2, 8, 4), (2, 4, 8)),
            T((6, 0, 3), (0, 3, 6)),
        ]
        ))

    # A mirror is not this ordering, and must not be answered as one.
    assert language.order_that_explains([T((1, 2, 3), (3, 2, 1))]) is None
    # Nor is noise.
    assert language.order_that_explains([T((1, 2, 3), (9, 4, 7))]) is None


def test_what_was_learned_beats_a_form_that_fits_one_example(monkeypatch, tmp_path) -> None:
    """Right answer, wrong reason, is a defect that hides until the reason moves.

    LIVE, 2026-08-28: a turn taught "ascending order", and the next one, showing
    a single example, was answered "position i takes from i+1 (mod n)". The
    answer was right by luck — a rotation and an ordering agree on that one
    state — and the ordering just learned was never consulted, because the
    positional path had found something and something was enough.

    The rival on that world was of the other kind, and the probe only ever
    compared positional forms to each other, so it saw no rival and called one
    example settled.
    """

    monkeypatch.setenv("AURA_STATE_ROOT", str(tmp_path))
    from core.cognition.sequence_induction import answer_sequence_question

    # Enough distinct pairs that the order was risked rather than merely
    # fitted: three values seen three times is two pairs, and a wrong order
    # survives two tests one time in four.
    taught = answer_sequence_question(
        "[30, 10, 20] becomes [10, 20, 30]. [50, 90, 70] becomes [50, 70, 90]. "
        "[20, 80, 40] becomes [20, 40, 80]. [60, 0, 30] becomes [0, 30, 60]. "
        "What does [70, 40, 90] become?"
    )
    assert "[40, 70, 90]" in taught

    thin = answer_sequence_question("[8, 2, 5] becomes [2, 5, 8]. What does [61, 14, 37] become?")
    assert "[14, 37, 61]" in thin
    assert "worked out earlier" in thin
    assert "mod n" not in thin


def test_a_pinned_positional_world_is_not_hijacked(monkeypatch, tmp_path) -> None:
    """Preferring what was learned only where the evidence is thin.

    A world showing three states at three lengths pins its rule. Letting an
    ordering take that would trade one wrong reason for another.
    """

    monkeypatch.setenv("AURA_STATE_ROOT", str(tmp_path))
    from core.cognition.sequence_induction import answer_sequence_question

    answer_sequence_question(
        "[30, 10, 20] becomes [10, 20, 30]. [50, 90, 70] becomes [50, 70, 90]. "
        "[20, 80, 40] becomes [20, 40, 80]. [60, 0, 30] becomes [0, 30, 60]. "
        "What does [70, 40, 90] become?"
    )
    for asked, wanted, rule in (
        (
            "[1,2,3,4] becomes [4,3,2,1], [1,2,3,4,5] becomes [5,4,3,2,1], "
            "[7,8,9,10] becomes [10,9,8,7]. What does [7,8,9] become?",
            "[9, 8, 7]",
            "n-1-i",
        ),
        (
            "[1,2,3,4] becomes [2,3,4,1], [1,2,3,4,5] becomes [2,3,4,5,1], "
            "[7,8,9] becomes [8,9,7]. What does [4,1,9] become?",
            "[1, 9, 4]",
            "mod n",
        ),
    ):
        said = answer_sequence_question(asked)
        assert wanted in said
        assert rule in said
        assert "worked out earlier" not in said


def test_the_composed_representation_survives_too() -> None:
    """The newest kind of learned thing was the one that could not be kept.

    An ordering followed by a move was admitted, held for the turn, and dropped
    at the file — every time, without a line anywhere — because the language's
    save filtered on ``hasattr(entry, "to_json")`` and Composed had none. A
    filter that silently discards is how a representation gets learned every
    time and never once kept.
    """

    from core.cognition.primitive_invention import _index_forms
    from core.cognition.value_order import Composed, solve_ordering_then_move

    def sorted_then_rotated(row: tuple) -> tuple:
        ordered = tuple(sorted(row))
        return ordered[1:] + ordered[:1]

    world = [T(row, sorted_then_rotated(row)) for row in ((3, 1, 2), (9, 5, 7), (2, 8, 4))]
    composed = solve_ordering_then_move(world, _index_forms(3))
    assert composed is not None

    home = _somewhere()
    language = RelationLanguage(path=home)
    language.admit_order(composed)
    language.save()

    restored = RelationLanguage.load(home)
    assert len(restored.orders) == 1
    kept = next(iter(restored.orders.values()))
    assert isinstance(kept, Composed)
    assert kept.apply((40, 11, 27)) == sorted_then_rotated((40, 11, 27))
    assert "then" in kept.describe()


def test_a_kind_that_cannot_be_written_down_says_so(caplog) -> None:
    """It is not worth an exception, and it is worth a line.

    A thing that cannot be saved is a thing that will be learned again
    tomorrow, and the way to find out was to notice it missing.
    """

    import logging

    class _Unwritable:
        def describe(self) -> str:
            return "something learned that cannot be written down"

    home = _somewhere()
    language = RelationLanguage(path=home)
    language.admit_order(_Unwritable())
    with caplog.at_level(logging.WARNING):
        language.save()
    assert any("cannot be written down" in record.message for record in caplog.records)


def test_both_kinds_are_read_back_not_just_one() -> None:
    """A reader that knows one kind turns the others into silence."""

    from pathlib import Path

    body = Path("core/cognition/relation_language.py").read_text()
    assert "Composed.from_json(row) or Ordering.from_json(row)" in body
