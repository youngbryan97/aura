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


def test_a_live_turn_never_waits_for_a_decision() -> None:
    """The first sighting abstains and is remembered; the next is answered."""
    matcher = _declared()
    assert matcher.decide_without_waiting("I filed the report.") is None
    assert matcher.report()["pending"] == 1

    settled = matcher.warm()
    assert settled == 1
    assert matcher.decide_without_waiting("I filed the report.") is True
    assert matcher.report()["pending"] == 0


def test_the_queue_of_unseen_phrasings_is_bounded() -> None:
    from core.language.learned_matcher import _PENDING_CEILING

    matcher = _declared()
    for index in range(_PENDING_CEILING + 40):
        matcher.decide_without_waiting(f"I did thing number {index}.")
    assert matcher.report()["pending"] <= _PENDING_CEILING


def test_warming_something_undecidable_leaves_no_verdict() -> None:
    muddled = LearnedMatcher(
        name="muddled",
        positives=("a", "b", "c"),
        negatives=("d", "e", "f"),
        features=lambda sentences: [[0.5] for _ in sentences],
    )
    muddled.decide_without_waiting("anything at all")
    assert muddled.warm() == 0
    assert muddled.decide_without_waiting("anything at all") is None


def test_the_narrow_pattern_teaches_the_learned_surface() -> None:
    """The regex is precise and narrow, which makes it a teacher.

    Everything it matches IS a claim, so its matches are labels nobody had to
    write, and the learned surface extends recall without ever removing a
    match the pattern found.
    """
    from core.conversation.response_reliability import (
        _ACTION_CLAIM_MATCHER,
        _sentence_claims_an_action,
    )

    before = len(_ACTION_CLAIM_MATCHER.positives)
    assert _sentence_claims_an_action("I saved it as ledger_2026.csv in Documents.") is True
    assert len(_ACTION_CLAIM_MATCHER.positives) > before


def test_a_phrasing_nobody_enumerated_is_remembered_not_guessed() -> None:
    from core.conversation.response_reliability import (
        _ACTION_CLAIM_MATCHER,
        _sentence_claims_an_action,
    )

    novel = "The notes are now sitting in meeting.md where you asked."
    assert _sentence_claims_an_action(novel) is False
    assert novel in _ACTION_CLAIM_MATCHER._pending


def test_the_warmer_is_registered_to_run_off_the_critical_path() -> None:
    from pathlib import Path

    main = Path("core/orchestrator/main.py").read_text(encoding="utf-8")
    assert "language_matcher_warm" in main
    status = Path("core/orchestrator/handlers/status_manager.py").read_text(encoding="utf-8")
    assert "asyncio.to_thread(warm_language_matchers" in status
