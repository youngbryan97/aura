"""A sealed artifact that will not admit, made legible.

`load_mathematics_memory_tissue` is fail-closed and right to be: it refuses an
artifact whose admission evidence no longer matches, which includes the SHA-256
of every source file the training run was pinned against.

What the runtime could not do was notice. `neural_objective_producer` caught the
RuntimeError, fell back to the deterministic solver, and returned a receipt of
the same shape — so the answer arrived and looked ordinary. The only visible
effect of a sealed capability going offline was a registered runtime claim
reporting False, with no way to tell "the seal is broken" from "the tissue
computed the wrong answer": different problems, different owners, one symptom.

Measured 2026-08-15: `core/learning/frontier_process_supervision.py` drifted
from its pinned hash in 8c48eec8d (CP546, schema v1→v2, 317 lines). The refusal
is correct. Re-sealing without re-running the canary would launder a real
provenance break, so nothing here re-seals anything.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from core.learning.sealed_artifact_admission import (
    SEALED_ADMISSION_SCHEMA,
    artifact_admission_status,
    mathematics_memory_admitted,
    sealed_artifact_admission_report,
)

ROOT = Path(__file__).resolve().parents[1]

#: Editing any of these drifts the artifact's own admission evidence. A
#: diagnostic must not change what it measures, which is why the probe lives
#: outside this set.
PINNED_SOURCES = (
    "core/learning/frontier_process_supervision.py",
    "core/learning/recurrent_work_memory.py",
    "core/learning/recurrent_work_memory_tissue.py",
    "core/learning/recurrent_work_memory_training.py",
    "tools/run_mathematics_memory_canary.py",
)


def test_the_probe_answers_rather_than_raising():
    admitted, detail = mathematics_memory_admitted()

    assert isinstance(admitted, bool)
    assert isinstance(detail, str)


def test_a_refusal_names_the_reason_and_the_drifted_file():
    """"Something is wrong" is what the runtime already had."""
    admitted, detail = mathematics_memory_admitted()

    if admitted:
        pytest.skip("the seal admits; there is no refusal to describe")
    assert detail, "a refusal with no reason is the silence this replaces"
    assert "drifted:" in detail or "unreadable" in detail or "missing" in detail


def test_the_report_lists_every_declared_artifact():
    report = sealed_artifact_admission_report()

    assert report["schema"] == SEALED_ADMISSION_SCHEMA
    assert report["declared"] >= 1
    assert len(report["artifacts"]) == report["declared"]
    assert report["admitted"] + len(report["refused"]) == report["declared"]


def test_the_report_is_serialisable():
    assert json.loads(json.dumps(sealed_artifact_admission_report()))


def test_an_unresolvable_artifact_is_reported_not_raised():
    status = artifact_admission_status("ghost", "core.does.not.exist", "NOPE")

    assert status["admitted"] is False
    assert "unresolvable" in status["reason"]


def test_a_missing_manifest_is_reported_not_raised(monkeypatch, tmp_path):
    import core.learning.recurrent_work_memory_tissue as tissue

    monkeypatch.setattr(
        tissue, "DEFAULT_MATHEMATICS_MEMORY_ARTIFACT", tmp_path / "absent", raising=False
    )
    status = artifact_admission_status(
        "mathematics_memory_tissue",
        "core.learning.recurrent_work_memory_tissue",
        "DEFAULT_MATHEMATICS_MEMORY_ARTIFACT",
    )

    assert status["admitted"] is False
    assert "manifest unreadable" in status["reason"]


def test_the_probe_lives_outside_the_pinned_set():
    """Adding a diagnostic to a pinned file drifts the hash the diagnostic
    reports on. This was a real mistake made once while building it."""
    from core.learning import sealed_artifact_admission

    module_path = Path(sealed_artifact_admission.__file__).resolve()
    relative = str(module_path.relative_to(ROOT))

    assert relative not in PINNED_SOURCES


def test_the_pinned_set_is_what_the_manifest_actually_pins():
    """A list of files that drifted from the manifest would make the rule above
    protect the wrong things."""
    import core.learning.recurrent_work_memory_tissue as tissue

    manifest_path = Path(tissue.DEFAULT_MATHEMATICS_MEMORY_ARTIFACT) / "manifest.json"
    if not manifest_path.is_file():
        pytest.skip("no sealed artifact on this host")
    pinned = set(
        (json.loads(manifest_path.read_text("utf-8")).get("canary") or {}).get(
            "source_sha256s"
        )
        or {}
    )

    assert pinned == set(PINNED_SOURCES)


def test_the_health_report_publishes_admission():
    from core.runtime.health_contract import runtime_health_report

    sealed = runtime_health_report()["sealed_artifacts"]

    assert sealed["schema"] == SEALED_ADMISSION_SCHEMA
    assert "refused" in sealed


def test_the_fallback_records_a_degradation_once():
    """The producer answered from the deterministic solver and said nothing.
    Recorded once, not per call: a refused seal stays refused, and per-call
    recording would bury the fact it is reporting."""
    import core.brain.llm.latent_cortex.neural_objective_producer as producer
    from core.brain.llm.latent_cortex.frontier_tasks import generate_task
    from core.runtime.errors import get_degradation_tracker

    admitted, _ = mathematics_memory_admitted()
    if admitted:
        pytest.skip("the seal admits; the fallback does not run")

    producer._FALLBACK_REPORTED.clear()
    tracker = get_degradation_tracker()
    before = len(
        [
            r
            for r in tracker.recent(limit=500)
            if r.subsystem == "latent_cortex.neural_objective_producer"
        ]
    )

    task = generate_task("mathematics", seed=1_037, difficulty=3)
    producer.solve_objective_program_neural(task.public.prompt)
    producer.solve_objective_program_neural(task.public.prompt)

    after = [
        r
        for r in tracker.recent(limit=500)
        if r.subsystem == "latent_cortex.neural_objective_producer"
    ]
    assert len(after) - before == 1, f"recorded {len(after) - before} times"


def test_the_registered_claim_reports_not_measured_rather_than_failed():
    """A provenance break is not a capability regression, and the claim suite
    reported it as one — which failed the whole runtime validation."""
    from core.organism.model_validation import (
        NothingMeasured,
        _recurrent_memory_complete_engine_contract_holds,
    )

    admitted, _ = mathematics_memory_admitted()
    if admitted:
        pytest.skip("the seal admits; the claim is measurable")

    with pytest.raises(NothingMeasured, match="not admitted"):
        _recurrent_memory_complete_engine_contract_holds()


def test_the_receipt_still_says_which_engine_ran():
    """The fallback is legitimate; what matters is that the receipt does not
    claim the neural engine produced the answer."""
    from core.brain.llm.latent_cortex.frontier_tasks import generate_task
    from core.brain.llm.latent_cortex.neural_objective_producer import (
        solve_objective_program_neural,
    )

    task = generate_task("mathematics", seed=1_037, difficulty=3)
    solved = solve_objective_program_neural(task.public.prompt)

    assert solved is not None
    _candidate, receipt = solved
    execution = receipt.get("execution", {})
    admitted, _ = mathematics_memory_admitted()
    if not admitted:
        assert execution.get("engine") != "mathematics_memory_tissue.v1"
        assert receipt.get("authority") == "public_objective_deterministic_execution"


def test_a_polled_health_query_announces_a_refusal_once() -> None:
    """The report is polled. A warning per poll buries the one that matters.

    `sealed_artifact_admission_report()` is what a health surface calls, which
    means it runs on every poll. It logs the refusal so the drift is visible at
    boot, so the announcement has to be keyed on the refusal set, not on the
    call. A new reason for the same artifact is a new set and is announced.
    """
    import core.learning.sealed_artifact_admission as admission

    admission._ANNOUNCED_REFUSALS = ()
    records: list[str] = []
    handler = logging.Handler()
    handler.emit = lambda record: records.append(record.getMessage())  # type: ignore[method-assign]
    admission.logger.addHandler(handler)
    try:
        for _ in range(5):
            report = admission.sealed_artifact_admission_report()
        announcements = [r for r in records if "Sealed artifact refused" in r]
        if report["refused"]:
            assert len(announcements) == 1, (
                f"a polled query announced the same refusal {len(announcements)} times"
            )
        else:
            assert not announcements

        # A different reason is a different fact and is announced again.
        admission._ANNOUNCED_REFUSALS = (("mathematics_memory_tissue", "something else"),)
        admission.sealed_artifact_admission_report()
        if report["refused"]:
            assert len(
                [r for r in records if "Sealed artifact refused" in r]
            ) == 2, "a changed refusal reason was swallowed by the announce-once memo"
    finally:
        admission.logger.removeHandler(handler)


def test_health_reuses_a_strict_verdict_only_while_dependencies_are_unchanged(
    monkeypatch,
    tmp_path,
) -> None:
    """Polling is cheap, but a changed seal dependency is checked immediately."""
    import core.learning.recurrent_work_memory_tissue as tissue
    import core.learning.sealed_artifact_admission as admission

    artifact = tmp_path / "artifact"
    artifact.mkdir()
    manifest = artifact / "manifest.json"
    manifest.write_text('{"canary":{"source_sha256s":{}}}', encoding="utf-8")
    monkeypatch.setattr(
        tissue,
        "DEFAULT_MATHEMATICS_MEMORY_ARTIFACT",
        artifact,
        raising=False,
    )
    admission._HEALTH_ADMISSION_CACHE.clear()
    strict = admission.artifact_admission_status
    calls = 0

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return strict(*args, **kwargs)

    monkeypatch.setattr(admission, "artifact_admission_status", counted)
    admission.sealed_artifact_admission_report()
    admission.sealed_artifact_admission_report()
    assert calls == 1, "an unchanged health poll repeated strict source hashing"

    manifest.write_text(
        '{"canary":{"source_sha256s":{}},"revision":2}',
        encoding="utf-8",
    )
    admission.sealed_artifact_admission_report()
    assert calls == 2, "a changed manifest reused the prior admission verdict"
    admission._HEALTH_ADMISSION_CACHE.clear()


def test_capability_admission_never_uses_the_health_memo(monkeypatch) -> None:
    """The optimization belongs to observability, not execution authority."""
    import core.learning.sealed_artifact_admission as admission

    strict = admission.artifact_admission_status
    calls = 0

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return strict(*args, **kwargs)

    monkeypatch.setattr(admission, "artifact_admission_status", counted)
    admission.mathematics_memory_admitted()
    admission.mathematics_memory_admitted()
    assert calls == 2


def test_a_skip_for_a_refused_seal_names_the_reason() -> None:
    """The suite may skip a capability the seal refuses, but never silently.

    Five tests exercise the sealed mathematics-memory tissue. When the seal is
    refused they cannot run, and a bare failure reads as a capability
    regression that did not happen. They skip through one shared helper whose
    reason carries the admission detail, so the skip line in the pytest report
    names the drifted file rather than saying only "skipped".
    """
    import pytest as _pytest

    from core.learning.sealed_artifact_admission import mathematics_memory_admitted
    from tests.sealed_artifact_support import require_mathematics_memory_tissue

    admitted, detail = mathematics_memory_admitted()
    if admitted:
        require_mathematics_memory_tissue()  # must not raise Skipped
        return

    with _pytest.raises(_pytest.skip.Exception) as caught:
        require_mathematics_memory_tissue()
    message = str(caught.value)
    assert detail in message, "the skip reason dropped the admission detail"
    assert "run_mathematics_memory_canary" in message, (
        "the skip reason does not say how to restore the capability"
    )
