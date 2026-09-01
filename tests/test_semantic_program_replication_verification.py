from __future__ import annotations

import copy
from dataclasses import replace

import pytest

from core.learning.semantic_program_replication import (
    SEMANTIC_PROGRAM_REPLICATION_SOURCES,
    FrozenTrainingCohort,
    evaluate_frozen_semantic_replication,
)
from core.learning.semantic_program_replication_verification import (
    SEMANTIC_PROGRAM_REPLICATION_VERIFICATION_SOURCES,
    verify_frozen_semantic_replication,
)
from core.learning.semantic_program_transducer import fit_semantic_program_transducer
from tests.test_semantic_program_replication import (
    _bundle,
    _manifest,
    _replication_battery,
    _sha_character,
    _with_basis,
    _worker_basis,
)


def _evidence():
    training_basis = _worker_basis(boot="1" * 32, pid=101)
    replication_basis = _worker_basis(boot="2" * 32, pid=202)
    training_examples = _with_basis(_replication_battery(), training_basis)
    replication_examples = _with_basis(_replication_battery(), replication_basis)
    model = fit_semantic_program_transducer(training_examples)
    raw_training = _bundle(training_examples, basis=training_basis)
    training_bundle = replace(
        raw_training,
        manifest={**raw_training.manifest, "manifest_sha256": _sha_character("e")},
        examples=tuple(
            replace(
                item,
                metadata={**item.metadata, "example_id": f"original-{index}"},
            )
            for index, item in enumerate(raw_training.examples)
        ),
    )
    replication_bundle = _bundle(replication_examples, basis=replication_basis)
    report_sources = {
        path: _sha_character("a") for path in SEMANTIC_PROGRAM_REPLICATION_SOURCES
    }
    report = evaluate_frozen_semantic_replication(
        replication_bundle,
        trained_model_payload=model.to_dict(),
        training_cohort=FrozenTrainingCohort(
            feature_manifest_sha256=training_bundle.manifest["manifest_sha256"],
            example_ids=tuple(
                item.metadata["example_id"] for item in training_bundle.examples
            ),
        ),
        training_manifest=_manifest(training_basis, character="e"),
        source_sha256s=report_sources,
    )
    verification_sources = {
        path: (
            report_sources[path]
            if path in report_sources
            else _sha_character("b")
        )
        for path in SEMANTIC_PROGRAM_REPLICATION_VERIFICATION_SOURCES
    }
    return (
        training_bundle,
        replication_bundle,
        model.to_dict(),
        report,
        verification_sources,
    )


def test_independent_verifier_replays_and_recounts_every_replication_row() -> None:
    training, replication, model, report, sources = _evidence()

    verified = verify_frozen_semantic_replication(
        training_bundle=training,
        replication_bundle=replication,
        trained_model_payload=model,
        stored_report=report,
        source_sha256s=sources,
    )

    assert verified["verified"] is True
    assert verified["frozen_replay_exact"] is True
    assert verified["task_rows_independently_recounted"] == 3 * len(
        replication.examples
    )
    assert verified["paired_tests_independently_recounted"] == 16
    assert verified["serving_authority"] is False


def test_independent_verifier_refuses_row_or_source_drift() -> None:
    training, replication, model, report, sources = _evidence()
    drifted_report = copy.deepcopy(report)
    drifted_report["arms"]["treatment:test"]["rows"][0]["answer_exact"] ^= True
    with pytest.raises(ValueError, match="envelope"):
        verify_frozen_semantic_replication(
            training_bundle=training,
            replication_bundle=replication,
            trained_model_payload=model,
            stored_report=drifted_report,
            source_sha256s=sources,
        )

    changed_sources = dict(sources)
    changed_sources[next(iter(SEMANTIC_PROGRAM_REPLICATION_SOURCES))] = (
        _sha_character("f")
    )
    with pytest.raises(ValueError, match="source differs"):
        verify_frozen_semantic_replication(
            training_bundle=training,
            replication_bundle=replication,
            trained_model_payload=model,
            stored_report=report,
            source_sha256s=changed_sources,
        )
