from __future__ import annotations

import numpy as np

from core.learning.semantic_operation_transfer import (
    SemanticOperationObservation,
    _evaluate_direction,
    counterfactual_center_operation_observations,
)


def _rows(family: str, split: str) -> tuple[SemanticOperationObservation, ...]:
    labels = ("add", "sub", "mul", "idiv")
    rows = []
    for index, label in enumerate(labels):
        feature = np.zeros(4, dtype=np.float32)
        feature[index] = 1.0
        rows.append(
            SemanticOperationObservation(
                family=family,
                split=split,
                example_id=f"{family}-{split}-{index}",
                contrast_id=f"{family}-{split}",
                step=0,
                label=label,
                feature=feature,
                token_surface=(100 + index,),
                geometry_feature=np.asarray((3, 2, 2, 0, 1), dtype=np.float32),
            )
        )
    return tuple(rows)


def test_operation_transfer_measures_programs_and_surface_overlap() -> None:
    source = _rows("source", "train")
    target = (*_rows("target", "validation"), *_rows("target", "test"))

    report = _evaluate_direction(source=source, target=target)

    assert report["source_training_operation_count"] == 4
    for split in ("validation", "test"):
        result = report["splits"][split]
        assert result["program_count"] == 4
        assert result["surface_overlap_count"] == 4
        assert result["arms"]["treatment"] == {
            "operation_exact": 4,
            "program_exact": 4,
        }
        assert result["arms"]["coefficient_lesion"]["program_exact"] < 4
        assert result["arms"]["label_permutation"]["program_exact"] == 0


def test_counterfactual_centering_removes_shared_context_without_labels() -> None:
    rows = _rows("source", "train")
    nuisance = np.asarray((9.0, 7.0, 5.0, 3.0), dtype=np.float32)
    shifted = tuple(
        SemanticOperationObservation(
            family=item.family,
            split=item.split,
            example_id=item.example_id,
            contrast_id=item.contrast_id,
            step=item.step,
            label=item.label,
            feature=item.feature + nuisance,
            token_surface=item.token_surface,
            geometry_feature=item.geometry_feature,
        )
        for item in rows
    )

    centered = counterfactual_center_operation_observations(shifted)

    assert np.allclose(
        np.mean(np.stack([item.feature for item in centered]), axis=0),
        0.0,
        atol=2e-8,
    )
    assert [item.label for item in centered] == [item.label for item in rows]
