from __future__ import annotations

import copy

import pytest

from core.evidence.calibrated_binary import (
    MIN_CALIBRATION_OBSERVATIONS,
    MIN_FIT_OBSERVATIONS,
    VerifiedBinaryObservation,
    calibrated_binary_scorer_from_dict,
    fit_calibrated_binary_scorer,
)
from core.evidence.calibrated_candidate_selector import (
    MIN_PAIRWISE_CALIBRATION_OBSERVATIONS,
    VerifiedPairwiseObservation,
    build_calibrated_candidate_selector,
    calibrated_candidate_selector_from_dict,
)
from core.evidence.necessary_condition_selector import (
    NecessaryEvidenceCondition,
    PairwiseSelectionEvidence,
    build_necessary_condition_selector,
)
from core.evidence.packet import observe

FEATURES = ("available", "quality")


def _values(*, available: float = 1.0, quality: float) -> dict[str, float]:
    return {"available": available, "quality": quality}


def _binary_rows(prefix: str, count: int, *, offset: int = 0):
    return tuple(
        VerifiedBinaryObservation.from_mapping(
            _values(quality=0.9 if (index + offset) % 2 == 0 else 0.1),
            verified_correct=(index + offset) % 2 == 0,
            source_ref=f"{prefix}:{index}",
        )
        for index in range(count)
    )


def _scorer():
    scorer, report = fit_calibrated_binary_scorer(
        _binary_rows("fit", max(40, MIN_FIT_OBSERVATIONS)),
        _binary_rows("calibration", max(40, MIN_CALIBRATION_OBSERVATIONS), offset=1),
        epochs=500,
        learning_rate=0.2,
    )
    assert scorer is not None
    assert report["admitted"] is True
    return scorer


def _necessary():
    return build_necessary_condition_selector(
        (
            NecessaryEvidenceCondition(
                name="available",
                minimum=1.0,
                necessity_contract="candidate_must_exist",
            ),
        )
    )


def _pairwise_rows(count: int = 24):
    return tuple(
        VerifiedPairwiseObservation.from_mappings(
            incumbent=_values(quality=0.1),
            challenger=_values(quality=0.9),
            incumbent_correct=False,
            challenger_correct=True,
            source_ref=f"pair:{index}",
        )
        for index in range(max(count, MIN_PAIRWISE_CALIBRATION_OBSERVATIONS))
    )


def _admission_rows(count: int = 24):
    return tuple(
        VerifiedPairwiseObservation.from_mappings(
            incumbent=_values(quality=0.1),
            challenger=_values(quality=0.9),
            incumbent_correct=False,
            challenger_correct=True,
            source_ref=f"admission:{index}",
        )
        for index in range(max(count, MIN_PAIRWISE_CALIBRATION_OBSERVATIONS))
    )


def test_binary_scorer_is_admitted_only_on_independent_verified_outcomes() -> None:
    scorer = _scorer()

    assert scorer.predict(_values(quality=0.9)) > 0.8
    assert scorer.predict(_values(quality=0.1)) < 0.2
    assert calibrated_binary_scorer_from_dict(scorer.to_dict()).to_dict() == scorer.to_dict()

    with pytest.raises(ValueError, match="splits"):
        fit_calibrated_binary_scorer(
            _binary_rows("same", 30),
            _binary_rows("same", 30),
        )


def test_binary_scorer_refuses_noise_and_tampering() -> None:
    noise_fit = tuple(
        VerifiedBinaryObservation.from_mapping(
            _values(quality=0.5),
            verified_correct=index % 2 == 0,
            source_ref=f"noise-fit:{index}",
        )
        for index in range(40)
    )
    noise_calibration = tuple(
        VerifiedBinaryObservation.from_mapping(
            _values(quality=0.5),
            verified_correct=index % 3 == 0,
            source_ref=f"noise-calibration:{index}",
        )
        for index in range(40)
    )
    scorer, report = fit_calibrated_binary_scorer(noise_fit, noise_calibration)

    assert scorer is None
    assert report["admitted"] is False

    payload = copy.deepcopy(_scorer().to_dict())
    payload["weights"][0] += 1.0
    with pytest.raises(ValueError, match="envelope"):
        calibrated_binary_scorer_from_dict(payload)


def test_pairwise_selector_repairs_necessary_failures_then_uses_calibration() -> None:
    selector, report = build_calibrated_candidate_selector(
        necessary=_necessary(),
        scorer=_scorer(),
        calibration_rows=_pairwise_rows(),
        admission_rows=_admission_rows(),
    )
    assert selector is not None
    assert report["admission_regressions"] == 0
    replay = calibrated_candidate_selector_from_dict(selector.to_dict())

    def select(old: dict[str, float], new: dict[str, float]):
        return replay.select(
            incumbent="old",
            challenger="new",
            evidence=PairwiseSelectionEvidence.from_mappings(
                incumbent=old,
                challenger=new,
                packet=observe(1.0, origin="selector_test", ref=f"{old}:{new}"),
            ),
        )

    calibrated = select(_values(quality=0.1), _values(quality=0.9))
    assert calibrated.selected == "new"
    assert calibrated.receipt["reason"] == "challenger_exceeds_calibrated_quality_margin"

    necessary = select(
        _values(available=0.0, quality=0.9),
        _values(available=1.0, quality=0.1),
    )
    assert necessary.selected == "new"
    assert necessary.receipt["reason"] == "challenger_repairs_necessary_condition_failure"

    unavailable = select(
        _values(available=1.0, quality=0.1),
        _values(available=0.0, quality=0.9),
    )
    assert unavailable.selected == "old"
    assert unavailable.receipt["reason"] == "challenger_fails_necessary_condition"


def test_pairwise_selector_refuses_a_calibration_with_regressions() -> None:
    rows = list(_pairwise_rows())
    rows.extend(
        VerifiedPairwiseObservation.from_mappings(
            incumbent=_values(quality=0.1),
            challenger=_values(quality=0.9),
            incumbent_correct=True,
            challenger_correct=False,
            source_ref=f"regression:{index}",
        )
        for index in range(30)
    )
    selector, report = build_calibrated_candidate_selector(
        necessary=_necessary(),
        scorer=_scorer(),
        calibration_rows=rows,
        admission_rows=tuple(
            VerifiedPairwiseObservation.from_mappings(
                incumbent=_values(quality=0.1),
                challenger=_values(quality=0.9),
                incumbent_correct=True,
                challenger_correct=False,
                source_ref=f"bad-admission:{index}",
            )
            for index in range(24)
        ),
        maximum_regressions=0,
    )

    assert selector is None
    assert report["admitted"] is False


def test_pairwise_selector_rejects_tampering_and_feature_drift() -> None:
    selector, _report = build_calibrated_candidate_selector(
        necessary=_necessary(),
        scorer=_scorer(),
        calibration_rows=_pairwise_rows(),
        admission_rows=_admission_rows(),
    )
    assert selector is not None
    payload = copy.deepcopy(selector.to_dict())
    payload["switch_margin"] += 0.1
    with pytest.raises(ValueError, match="envelope"):
        calibrated_candidate_selector_from_dict(payload)

    with pytest.raises(ValueError, match="feature schema"):
        selector.select(
            incumbent="old",
            challenger="new",
            evidence=PairwiseSelectionEvidence.from_mappings(
                incumbent={"available": 1.0, "other": 0.1},
                challenger={"available": 1.0, "other": 0.9},
                packet=observe(1.0, origin="selector_test", ref="drift"),
            ),
        )
