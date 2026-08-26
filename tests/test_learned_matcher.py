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

import threading

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


def test_overlapping_classes_still_leave_decisive_regions() -> None:
    """The rule this replaced demanded perfect separation and threw away a
    decision with an AUROC of 0.979: a handful of overlapping examples made
    the surface abstain on everything.

    Above every negative only positives were seen; below every positive only
    negatives. In between, both.
    """
    overlapping = Boundary(decide_true_above=0.8, decide_false_below=0.2, spread=0.3)
    assert overlapping.trustworthy is True
    assert overlapping.decide(0.9) is True
    assert overlapping.decide(0.1) is False
    assert overlapping.decide(0.5) is None


def test_a_score_inside_the_overlap_is_not_a_decision() -> None:
    boundary = Boundary(decide_true_above=1.0, decide_false_below=0.0, spread=0.1)
    assert boundary.decide(0.5) is None
    assert boundary.decide(1.5) is True
    assert boundary.decide(-0.5) is False


def test_separation_is_reported_even_though_it_is_not_required() -> None:
    clean = Boundary(decide_true_above=0.1, decide_false_below=0.9, spread=0.05, separable=True)
    assert clean.gap > 0
    assert clean.decide(0.5) is True


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


def test_a_live_turn_never_waits_for_the_warmer_mutex() -> None:
    """A hidden-state warm held this mutex for minutes while /api/chat waited.

    The live surface is an already-known verdict lookup, not a synchronization
    point with background model work. Contention therefore means abstain now.
    """
    matcher = _declared()
    lock_held = threading.Event()
    release_lock = threading.Event()
    finished = threading.Event()
    result: list[bool | None] = []

    def hold_lock() -> None:
        with matcher._lock:
            lock_held.set()
            release_lock.wait(timeout=2.0)

    def decide_live() -> None:
        result.append(matcher.decide_without_waiting("I filed the report."))
        finished.set()

    holder = threading.Thread(target=hold_lock)
    caller = threading.Thread(target=decide_live)
    holder.start()
    assert lock_held.wait(timeout=1.0)
    caller.start()
    try:
        assert finished.wait(timeout=0.1), "live decision blocked on background warming"
        assert result == [None]
    finally:
        release_lock.set()
        holder.join(timeout=1.0)
        caller.join(timeout=1.0)


def test_feature_computation_does_not_hold_the_matcher_mutex() -> None:
    """Resident hidden-state reads happen outside the matcher state lock."""
    feature_started = threading.Event()
    release_features = threading.Event()

    def slow_features(sentences):
        texts = list(sentences)
        feature_started.set()
        release_features.wait(timeout=2.0)
        return _mood(texts)

    matcher = _declared()
    matcher.features = slow_features
    worker = threading.Thread(target=matcher.decide, args=("I filed it.",))
    worker.start()
    assert feature_started.wait(timeout=1.0)
    acquired = matcher._lock.acquire(timeout=0.1)
    try:
        assert acquired, "feature computation retained the matcher mutex"
    finally:
        if acquired:
            matcher._lock.release()
        release_features.set()
        worker.join(timeout=2.0)


def test_a_boundary_from_an_obsolete_example_revision_is_not_published() -> None:
    feature_started = threading.Event()
    release_features = threading.Event()

    def slow_features(sentences):
        texts = list(sentences)
        feature_started.set()
        release_features.wait(timeout=2.0)
        return _mood(texts)

    matcher = _declared()
    matcher.features = slow_features
    result: list[bool | None] = []
    worker = threading.Thread(target=lambda: result.append(matcher.decide("I filed it.")))
    worker.start()
    assert feature_started.wait(timeout=1.0)
    matcher.observe("Shall I file it?", holds=False)
    release_features.set()
    worker.join(timeout=2.0)

    assert result == [None]
    assert matcher._ready is False


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


def test_frozen_measurement_is_not_a_live_runtime_heartbeat() -> None:
    """A scientific sweep must never recycle the conversation model at boot."""
    from pathlib import Path

    main = Path("core/orchestrator/main.py").read_text(encoding="utf-8")
    assert 'name="language_substrate_measurement"' not in main


def test_warmer_obeys_shared_background_admission(monkeypatch) -> None:
    import asyncio

    from core.conversation import response_reliability
    from core.orchestrator.handlers.status_manager import StatusManagerMixin
    from core.runtime import background_policy

    calls: list[int] = []
    monkeypatch.setattr(
        background_policy,
        "background_activity_reason",
        lambda *_args, **_kwargs: "recent_user_0",
    )
    monkeypatch.setattr(
        response_reliability,
        "warm_language_matchers",
        lambda limit: calls.append(limit) or 0,
    )

    asyncio.run(StatusManagerMixin()._warm_language_matchers())
    assert calls == []


def test_warmer_limits_each_registered_surface_to_one_phrasing(monkeypatch) -> None:
    import asyncio

    from core.conversation import response_reliability
    from core.orchestrator.handlers.status_manager import StatusManagerMixin
    from core.runtime import background_policy

    calls: list[int] = []
    monkeypatch.setattr(
        background_policy,
        "background_activity_reason",
        lambda *_args, **_kwargs: "",
    )
    monkeypatch.setattr(
        response_reliability,
        "warm_language_matchers",
        lambda limit: calls.append(limit) or 0,
    )

    asyncio.run(StatusManagerMixin()._warm_language_matchers())
    assert calls == [1]


def test_what_was_learned_survives_a_restart(tmp_path, monkeypatch) -> None:
    """The difference between a cache and learning.

    Everything lives in Python fields and this runtime restarts often — for a
    code change, a model swap, after a crash. Without a durable write every
    phrasing learned from use was discarded each time.
    """
    from core.language.substrate_store import LanguageSubstrateStore

    store = tmp_path / "language" / "restart_test.json"
    substrate_store = LanguageSubstrateStore(
        data_root=tmp_path,
        project_root=tmp_path,
    )

    first = LearnedMatcher(
        name="restart_test",
        positives=("I did it.",),
        negatives=("Did you?",),
        features=_mood,
        _store=substrate_store,
    )
    first.observe("I filed the report.", holds=True)
    first.decide_without_waiting("The notes are in meeting.md now.")
    assert first.save() is True
    assert store.is_file()

    second = LearnedMatcher(
        name="restart_test",
        positives=(),
        negatives=(),
        features=_mood,
        _store=substrate_store,
    )
    second.load()
    assert "I filed the report." in second.positives
    assert "The notes are in meeting.md now." in second._pending


def test_a_verdict_is_not_restored_across_runs(tmp_path, monkeypatch) -> None:
    """Verdicts were reached against a boundary the new process has not
    measured. Re-deciding costs one warm cycle; keeping them means serving an
    answer nothing in this run can vouch for."""
    from core.language.substrate_store import LanguageSubstrateStore

    substrate_store = LanguageSubstrateStore(
        data_root=tmp_path,
        project_root=tmp_path,
    )

    first = _declared()
    first.name = "verdicts"
    first._store = substrate_store
    first.decide_without_waiting("I filed the report.")
    first.warm()
    assert first.decide_without_waiting("I filed the report.") is True
    first._dirty = True
    first.save()

    second = LearnedMatcher(
        name="verdicts",
        positives=(),
        negatives=(),
        features=_mood,
        _store=substrate_store,
    )
    second.load()
    assert second._decided == {}


def test_a_new_example_retires_the_verdicts_it_predates() -> None:
    matcher = _declared()
    matcher.decide_without_waiting("I filed the report.")
    matcher.warm()
    assert matcher.decide_without_waiting("I filed the report.") is True

    matcher.observe("Shall I file the report?", holds=False)
    assert matcher.decide_without_waiting("I filed the report.") is None


def test_a_phrase_the_model_could_not_decide_stays_queued() -> None:
    """It was dropped whether or not a verdict was reached, so a phrase
    deferred while the model was busy had to be met again."""
    undecidable = LearnedMatcher(
        name="undecidable",
        positives=("a", "b", "c"),
        negatives=("d", "e", "f"),
        features=lambda sentences: [[0.5] for _ in sentences],
    )
    undecidable.decide_without_waiting("something new")
    assert undecidable.warm() == 0
    assert "something new" in undecidable._pending


def test_spelling_is_not_a_new_sighting() -> None:
    """Keying on the raw string made "I saved it as report.csv" and the same
    sentence with a full stop two separate first sightings."""
    matcher = _declared()
    matcher.decide_without_waiting("I filed the report")
    matcher.warm()

    for variant in ("I filed the report.", "  i filed the report  ", "I FILED THE REPORT!"):
        assert matcher.decide_without_waiting(variant) is True


def test_a_real_paraphrase_is_still_a_new_sighting() -> None:
    """Exact about what this does not fix: it collapses spelling, not wording.

    Deciding a paraphrase needs its vector, and computing that costs a
    forward pass the turn cannot spend.
    """
    matcher = _declared()
    matcher.decide_without_waiting("I filed the report")
    matcher.warm()
    assert matcher.decide_without_waiting("The report has been filed.") is None
