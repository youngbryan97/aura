"""A decision learned from declared examples, and the discipline around it.

Every matcher in this runtime is a regex with a word list, and every one has
been wrong the same way: a phrasing nobody thought of. The labels for doing
better already exist — each Observable declares examples and counter-examples,
and the registry test fails a matcher that gets its own examples wrong.

MEASURED, 2026-08-20: across all twenty-five declarations a topical sentence
embedder separated eight, none by more than the spread inside its own classes.
The axis these decisions turn on is who acts and whether it is asserted, and
an embedder is trained to make those sentences NEAR each other. So the
feature source is a parameter, and the surface abstains rather than guessing.
"""

from __future__ import annotations

from core.language.learned_matcher import Boundary, LearnedMatcher


def _mood(sentences):
    """A feature space that does carry the axis: first person, and mood."""
    vectors = []
    for text in sentences:
        lowered = str(text).lower()
        vectors.append([1.0 if lowered.startswith("i ") else 0.0, 1.0 if "?" in text else 0.0])
    return vectors


def _declared() -> LearnedMatcher:
    return LearnedMatcher(
        name="synthetic",
        positives=("I did the thing.", "I closed the door.", "I wrote it down."),
        negatives=("Would you do the thing?", "Shall I close the door?", "Could you write it down?"),
        features=_mood,
    )


def test_it_decides_when_the_examples_separate() -> None:
    matcher = _declared()
    assert matcher.decide("I moved the file.") is True
    assert matcher.decide("Would you move the file?") is False


def test_the_boundary_is_measured_from_the_declaration() -> None:
    report = _declared().report()
    assert report["separable"] is True
    assert report["trustworthy"] is True
    assert report["gap"] > report["spread"]


def test_it_abstains_without_enough_examples() -> None:
    thin = LearnedMatcher(name="thin", positives=("I did it.",), negatives=("Did you?",), features=_mood)
    assert thin.decide("I closed it.") is None
    assert thin.report()["trustworthy"] is False


def test_it_abstains_when_the_classes_overlap() -> None:
    def one_axis(sentences):
        return [[0.5] for _ in sentences]

    muddled = LearnedMatcher(
        name="muddled",
        positives=("a", "b", "c"),
        negatives=("d", "e", "f"),
        features=one_axis,
    )
    assert muddled.decide("anything at all") is None
    assert muddled.report()["separable"] is False


def test_a_gap_narrower_than_the_noise_is_not_trusted() -> None:
    """The finding that made this rule: eight declarations separated and every
    gap was smaller than the spread of the examples it was drawn from."""
    assert Boundary(lower=0.0, upper=0.05, separable=True, spread=0.20).trustworthy is False
    assert Boundary(lower=0.0, upper=0.05, separable=True, spread=0.20).decide(0.9) is None
    assert Boundary(lower=0.0, upper=0.30, separable=True, spread=0.05).trustworthy is True


def test_a_score_inside_the_gap_is_not_a_decision() -> None:
    boundary = Boundary(lower=0.0, upper=1.0, separable=True, spread=0.1)
    assert boundary.decide(0.5) is None
    assert boundary.decide(1.0) is True
    assert boundary.decide(0.0) is False


def test_what_something_else_settled_becomes_an_example() -> None:
    """A tool receipt is a label nobody had to write."""
    matcher = _declared()
    before = len(matcher.positives)
    matcher.observe("I saved the notes to disk.", holds=True)
    assert len(matcher.positives) == before + 1
    matcher.observe("I saved the notes to disk.", holds=True)
    assert len(matcher.positives) == before + 1


def test_observing_reopens_the_boundary() -> None:
    matcher = _declared()
    matcher.report()
    matcher.observe("I filed the report.", holds=True)
    assert matcher._ready is False


def test_it_never_decides_on_an_untrusted_boundary() -> None:
    """The invariant that holds whatever the feature source does."""
    matcher = LearnedMatcher(
        name="flat",
        positives=("a", "b"),
        negatives=("c", "d"),
        features=lambda sentences: [[1.0, 0.0] for _ in sentences],
    )
    report = matcher.report()
    if not report["trustworthy"]:
        assert matcher.decide("anything") is None
