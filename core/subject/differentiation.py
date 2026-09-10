"""Whether the integrated thing has more than one dimension in it.

A system where every variable copies one global scalar is perfectly unified
and says nothing. The check is the participation ratio of the state
covariance,

    D_eff = (tr S)^2 / tr(S^2),

which counts how many directions carry comparable variance: 1 when a single
component explains everything, D when all of them contribute equally.

It is computed on the correlation matrix, not the covariance of the raw
readings. Two of these columns are host temperature in degrees and a
saturating count in [0,1]; on raw covariance the temperature would be most of
the trace and the answer would be a fact about units. Standardising first
costs the measure its sensitivity to amplitude and buys back its independence
from how each feature happened to be scaled, which is the right trade when the
features were authored rather than measured in one unit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from core.subject.recording import Recording

__all__ = ["DifferentiationReport", "effective_dimension"]


@dataclass(frozen=True, slots=True)
class DifferentiationReport:
    d_eff: float
    normalised: float
    width: int
    top_share: float
    spectrum: tuple[float, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "d_eff": round(self.d_eff, 4),
            "d_eff_normalised": round(self.normalised, 4),
            "live_width": self.width,
            "largest_component_share": round(self.top_share, 4),
            "spectrum_head": [round(v, 4) for v in self.spectrum[:12]],
        }


def effective_dimension(recording: Recording, rows: np.ndarray | None = None) -> DifferentiationReport:
    """Participation ratio of the correlation spectrum over moving columns."""
    x = recording.x if rows is None else recording.x[rows]
    live = recording.live_columns()
    block = x[:, live]
    if block.shape[0] < 3 or block.shape[1] < 2:
        return DifferentiationReport(1.0, 0.0, int(block.shape[1]), 1.0, (1.0,))
    spread = block.std(axis=0)
    keep = spread > 1e-9
    block = (block[:, keep] - block[:, keep].mean(axis=0)) / spread[keep]
    width = block.shape[1]
    sigma = np.cov(block, rowvar=False)
    values = np.linalg.eigvalsh(sigma)
    values = np.clip(values, 0.0, None)[::-1]
    total = float(values.sum())
    if total <= 0.0:
        return DifferentiationReport(1.0, 1.0 / max(width, 1), width, 1.0, (0.0,))
    d_eff = float(total**2 / float((values**2).sum()))
    return DifferentiationReport(
        d_eff=d_eff,
        normalised=d_eff / float(width),
        width=width,
        top_share=float(values[0] / total),
        spectrum=tuple(float(v / total) for v in values),
    )
