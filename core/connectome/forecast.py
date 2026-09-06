"""core/connectome/forecast.py — predicting what the tissue does next.

ZAPBench asks one question of a whole brain: given what every cell just did,
what will every cell do next? Its answer so far is that learned models beat the
naive baselines but not by as much as anyone expected, that a model which knows
where cells sit in space wins at short horizons, and that models mixing
information across cells barely beat models that treat every cell on its own. The paper reads that as current architectures failing to
exploit the relationships between neurons.

There is a second reading, and Aura is the one system that can separate them.
Maybe the relationships are not there to exploit. ZAPBench cannot tell, because
the fish it recorded is not the fish anyone has a connectome for. Here the cell
that fires is the cell in the graph, so the question splits cleanly:

    Does knowing who a cell is wired to help predict what it will do?

The arms below are built so that question has an answer rather than an opinion.
Every model is fitted by ridge regression in closed form, which means no seed,
no learning rate, no early stopping and nothing to tune away a null result. The
arms differ in exactly one thing: which other cells' history each cell is
allowed to see.

``blind``
    Its own past only. The univariate control.
``global``
    Its own past plus the mean over every cell. The cheapest possible way to
    use other cells, and a surprisingly hard baseline to beat.
``connectome``
    Its own past plus the mean over the cells that drive it and the mean over
    the cells it drives.
``rewired``
    The same model with the same number of neighbours per cell, drawn from a
    degree-preserving rewiring. This is the null the connectome arm has to
    beat, and without it a win means nothing: any set of neighbours supplies
    some signal, and the question is whether *these* neighbours supply more.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("Aura.Connectome.Forecast")

__all__ = [
    "Window",
    "FeatureSpec",
    "RidgeForecaster",
    "mean_baseline",
    "condition_mean_baseline",
    "persistence_baseline",
    "neighbour_means",
    "ARM_NAMES",
]

#: The arms, in the order results are reported.
ARM_NAMES: tuple[str, ...] = ("blind", "global", "connectome", "rewired")


@dataclass(frozen=True)
class Window:
    """One supervised example: context ending at ``t``, target starting at ``t``."""

    context: int
    horizon: int

    def valid_range(self, frames: int) -> range:
        return range(self.context, max(self.context, frames - self.horizon + 1))


@dataclass(frozen=True)
class FeatureSpec:
    """Which channels a cell is allowed to see, and over how many lags."""

    name: str
    use_self: bool = True
    use_global: bool = False
    use_in_neighbours: bool = False
    use_out_neighbours: bool = False

    def channels(self) -> int:
        return sum(
            (
                self.use_self,
                self.use_global,
                self.use_in_neighbours,
                self.use_out_neighbours,
            )
        )


SPECS: dict[str, FeatureSpec] = {
    "blind": FeatureSpec("blind"),
    "global": FeatureSpec("global", use_global=True),
    "connectome": FeatureSpec("connectome", use_in_neighbours=True, use_out_neighbours=True),
    "rewired": FeatureSpec("rewired", use_in_neighbours=True, use_out_neighbours=True),
}


def neighbour_means(activity: Any, adjacency: Any) -> Any:
    """Mean activity over each cell's neighbours, at every frame.

    ``adjacency`` is a sparse matrix whose rows are already normalised, so the
    product is a mean rather than a sum. A cell with no neighbours gets zeros,
    which is the honest value: it has no neighbourhood to average.
    """
    import numpy as np

    if adjacency is None:
        return np.zeros_like(activity)
    return (adjacency @ activity.T).T


def _design(
    activity: Any,
    spec: FeatureSpec,
    window: Window,
    *,
    global_series: Any = None,
    in_means: Any = None,
    out_means: Any = None,
) -> tuple[Any, Any]:
    """Stack every (cell, time) example into one design matrix and target block.

    Rows are examples, columns are ``lags x channels`` plus a bias. Building it
    once for all cells is what makes a shared-weight fit cheap enough to run
    every arm on the full recording rather than on a sample.
    """
    import numpy as np

    frames, cells = activity.shape
    times = list(window.valid_range(frames))
    if not times:
        empty = np.zeros((0, spec.channels() * window.context + 1), dtype=np.float64)
        return empty, np.zeros((0, window.horizon), dtype=np.float64)
    rows = len(times) * cells
    width = spec.channels() * window.context + 1
    design = np.empty((rows, width), dtype=np.float64)
    target = np.empty((rows, window.horizon), dtype=np.float64)
    cursor = 0
    for t in times:
        block = slice(cursor, cursor + cells)
        column = 0
        if spec.use_self:
            design[block, column : column + window.context] = activity[
                t - window.context : t
            ].T
            column += window.context
        if spec.use_global and global_series is not None:
            design[block, column : column + window.context] = np.repeat(
                global_series[t - window.context : t][None, :], cells, axis=0
            )
            column += window.context
        if spec.use_in_neighbours and in_means is not None:
            design[block, column : column + window.context] = in_means[
                t - window.context : t
            ].T
            column += window.context
        if spec.use_out_neighbours and out_means is not None:
            design[block, column : column + window.context] = out_means[
                t - window.context : t
            ].T
            column += window.context
        design[block, width - 1] = 1.0
        target[block] = activity[t : t + window.horizon].T
        cursor += cells
    return design, target


@dataclass
class RidgeForecaster:
    """One arm: a shared linear map from context to horizon, fitted in closed form.

    Weights are shared across cells, which is what ZAPBench's univariate models
    do and what makes the arms comparable — every arm has the same number of
    parameters per channel, so a win is about which channels are available and
    not about capacity.
    """

    spec: FeatureSpec
    window: Window
    ridge: float = 1.0
    weights: Any = None

    def fit(
        self,
        activity: Any,
        *,
        global_series: Any = None,
        in_means: Any = None,
        out_means: Any = None,
    ) -> RidgeForecaster:
        import numpy as np

        design, target = _design(
            activity,
            self.spec,
            self.window,
            global_series=global_series,
            in_means=in_means,
            out_means=out_means,
        )
        if design.shape[0] == 0:
            self.weights = np.zeros((design.shape[1], self.window.horizon))
            return self
        gram = design.T @ design
        gram[np.diag_indices_from(gram)] += self.ridge
        self.weights = np.linalg.solve(gram, design.T @ target)
        return self

    def predict(
        self,
        activity: Any,
        *,
        global_series: Any = None,
        in_means: Any = None,
        out_means: Any = None,
    ) -> tuple[Any, Any]:
        import numpy as np

        design, target = _design(
            activity,
            self.spec,
            self.window,
            global_series=global_series,
            in_means=in_means,
            out_means=out_means,
        )
        if design.shape[0] == 0 or self.weights is None:
            return np.zeros((0, self.window.horizon)), target
        return design @ self.weights, target


def mean_baseline(train: Any, test: Any, window: Window) -> tuple[Any, Any]:
    """Predict each cell's training mean, for every step of the horizon."""
    import numpy as np

    per_cell = train.mean(axis=0)
    times = list(window.valid_range(test.shape[0]))
    cells = test.shape[1]
    if not times:
        return np.zeros((0, window.horizon)), np.zeros((0, window.horizon))
    prediction = np.repeat(
        np.repeat(per_cell[:, None], window.horizon, axis=1)[None, :, :], len(times), axis=0
    ).reshape(len(times) * cells, window.horizon)
    target = np.concatenate([test[t : t + window.horizon].T for t in times], axis=0)
    return prediction, target


def condition_mean_baseline(
    train: Any,
    train_conditions: Sequence[str],
    test: Any,
    test_conditions: Sequence[str],
    window: Window,
) -> tuple[Any, Any]:
    """ZAPBench's stimulus-conditioned mean: what this cell does under this condition.

    It is the baseline that embarrasses learned models when the stimulus
    explains most of the variance, which is exactly why it belongs here.
    """
    import numpy as np

    per_condition: dict[str, Any] = {}
    for condition in sorted(set(train_conditions)):
        rows = [i for i, c in enumerate(train_conditions) if c == condition]
        if rows:
            per_condition[condition] = train[rows].mean(axis=0)
    fallback = train.mean(axis=0)
    times = list(window.valid_range(test.shape[0]))
    if not times:
        return np.zeros((0, window.horizon)), np.zeros((0, window.horizon))
    predictions = []
    targets = []
    for t in times:
        condition = test_conditions[t] if t < len(test_conditions) else ""
        base = per_condition.get(condition, fallback)
        predictions.append(np.repeat(base[:, None], window.horizon, axis=1))
        targets.append(test[t : t + window.horizon].T)
    return np.concatenate(predictions, axis=0), np.concatenate(targets, axis=0)


def persistence_baseline(test: Any, window: Window) -> tuple[Any, Any]:
    """Hold the last observed frame for the whole horizon.

    Not one of ZAPBench's arms, and included anyway. It is the baseline that a
    forecasting result has to beat before it is a result at all, and on slow
    signals it is very hard to beat.
    """
    import numpy as np

    times = list(window.valid_range(test.shape[0]))
    if not times:
        return np.zeros((0, window.horizon)), np.zeros((0, window.horizon))
    predictions = []
    targets = []
    for t in times:
        last = test[t - 1]
        predictions.append(np.repeat(last[:, None], window.horizon, axis=1))
        targets.append(test[t : t + window.horizon].T)
    return np.concatenate(predictions, axis=0), np.concatenate(targets, axis=0)
