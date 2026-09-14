"""Development had one route into recall, and it could not fire.

The route was the retrieval organ's discrete breadth choice. That organ starts
by agreeing with the incumbent plan and only disagrees once its decision head
has learned enough to, so over a sixty-round campaign it never disagreed: the
plan was identical in every arm, displacing development changed nothing that
came back, and N->M measured exactly 0.000 with a p-value of one.

Depth is the continuous dependence on the same reading, in the form every other
modulation in that phase already takes. The reference is the organ's own —
`novelty()` returns 0.5 until it has a distribution to compare against, so
above a half is a moment more unlike her ordinary life than an ordinary one.
"""

from __future__ import annotations

from core.phases.memory_retrieval import (
    ORDINARY_NOVELTY,
    MemoryRetrievalPhase,
    novelty_deepens,
)


def test_a_novel_moment_is_searched_deeper_than_an_ordinary_one() -> None:
    ordinary = novelty_deepens(0.2)
    novel = novelty_deepens(0.9)
    assert novel > ordinary, (
        "development does not reach how deep she looks, so displacing it "
        "cannot change what comes back"
    )


def test_the_reference_is_the_organs_own_neutral_point() -> None:
    """Half is what the organ reports before it has a distribution."""
    assert ORDINARY_NOVELTY == 0.5
    assert novelty_deepens(ORDINARY_NOVELTY) == 0
    assert novelty_deepens(0.49) == 0
    assert novelty_deepens(0.51) == 1


def test_an_absent_reading_leaves_the_depth_alone() -> None:
    """An organ that is not loaded changes nothing rather than failing."""
    assert novelty_deepens(float("nan")) == 0
    assert novelty_deepens(None) == 0  # type: ignore[arg-type]
    assert novelty_deepens("broad") == 0  # type: ignore[arg-type]


def test_the_phase_reads_the_organ_when_it_sets_the_depth() -> None:
    """The rule is wired, not merely defined."""
    import inspect

    source = inspect.getsource(MemoryRetrievalPhase.execute)
    assert "novelty_deepens(novelty_now())" in source
