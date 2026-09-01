from __future__ import annotations

import numpy as np

from core.learning.semantic_operation_transfer import (
    SemanticOperationObservation,
    _evaluate_direction,
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
