from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Mapping

import numpy as np

from .vector_registry import RegisteredVector


@dataclass(frozen=True)
class VectorQualityReport:
    available: bool
    layer_idx: int
    vector_count: int
    vector_dim: int
    mean_norm: float
    min_norm: float
    max_norm: float
    mean_pairwise_abs_cosine: float
    exact_match_ratio: float
    extracted_ratio: float
    finite: bool
    near_zero_vectors: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_vector_quality(
    vectors: Mapping[str, RegisteredVector],
    *,
    layer_idx: int = -1,
) -> VectorQualityReport:
    items = list(vectors.values())
    if not items:
        return VectorQualityReport(
            available=False,
            layer_idx=layer_idx,
            vector_count=0,
            vector_dim=0,
            mean_norm=0.0,
            min_norm=0.0,
            max_norm=0.0,
            mean_pairwise_abs_cosine=0.0,
            exact_match_ratio=0.0,
            extracted_ratio=0.0,
            finite=False,
            near_zero_vectors=0,
        )
    matrix = np.stack([np.asarray(vector.v, dtype=np.float32).reshape(-1) for vector in items])
    norms = np.linalg.norm(matrix, axis=1)
    unit = matrix / np.clip(norms[:, None], 1e-8, None)
    pairwise: list[float] = []
    for idx in range(len(unit)):
        for jdx in range(idx + 1, len(unit)):
            pairwise.append(float(abs(np.dot(unit[idx], unit[jdx]))))
    exact = sum(1 for vector in items if vector.provenance.exact_layer_match)
    extracted = sum(1 for vector in items if vector.provenance.extracted)
    finite = bool(np.isfinite(matrix).all())
    near_zero = int(np.sum(norms < 1e-6))
    return VectorQualityReport(
        available=True,
        layer_idx=layer_idx if layer_idx >= 0 else int(items[0].layer_idx),
        vector_count=len(items),
        vector_dim=int(matrix.shape[1]),
        mean_norm=float(np.mean(norms)),
        min_norm=float(np.min(norms)),
        max_norm=float(np.max(norms)),
        mean_pairwise_abs_cosine=float(np.mean(pairwise or [0.0])),
        exact_match_ratio=float(exact / len(items)),
        extracted_ratio=float(extracted / len(items)),
        finite=finite,
        near_zero_vectors=near_zero,
    )
