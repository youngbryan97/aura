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
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("Aura.Connectome.Forecast")

__all__ = [
    "Window",
    "FeatureSpec",
    "RidgeForecaster",
    "baseline_score",
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


def _design_chunk(
    activity: Any,
    spec: FeatureSpec,
    window: Window,
    times: Sequence[int],
    *,
    global_series: Any = None,
    in_means: Any = None,
    out_means: Any = None,
) -> tuple[Any, Any]:
    """One block of examples: every cell, at the given timepoints.

    A real recording is twelve thousand cells over two thousand frames, and a
    256-lag design over all of it is eighty gigabytes. Blocks keep the
    arithmetic identical and the memory bounded, which is the only reason this
    runs on the machine it has to run on.
    """
    import numpy as np

    frames, cells = activity.shape
    width = spec.channels() * window.context + 1
    if not times:
        return (
            np.zeros((0, width), dtype=np.float64),
            np.zeros((0, window.horizon), dtype=np.float64),
        )
    rows = len(times) * cells
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


def _time_blocks(times: Sequence[int], cells: int, budget_rows: int) -> Iterator[list[int]]:
    """Split the timepoints so no block exceeds the row budget."""
    per_block = max(1, budget_rows // max(1, cells))
    for start in range(0, len(times), per_block):
        yield list(times[start : start + per_block])


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
    #: Bytes of design matrix held at once. A row costs eight bytes per column
    #: and a 256-lag connectome arm is 769 columns wide, so a row budget alone
    #: would mean something different at every context length. Two gigabytes
    #: leaves the host to whatever else is running on it.
    block_bytes: int = 2_000_000_000

    @property
    def block_rows(self) -> int:
        width = self.spec.channels() * self.window.context + 1
        return max(1, int(self.block_bytes // (width * 8 * 2)))

    def fit(
        self,
        activity: Any,
        *,
        global_series: Any = None,
        in_means: Any = None,
        out_means: Any = None,
    ) -> RidgeForecaster:
        import numpy as np

        frames, cells = activity.shape
        times = list(self.window.valid_range(frames))
        width = self.spec.channels() * self.window.context + 1
        gram = np.zeros((width, width), dtype=np.float64)
        cross = np.zeros((width, self.window.horizon), dtype=np.float64)
        seen = 0
        for block in _time_blocks(times, cells, self.block_rows):
            design, target = _design_chunk(
                activity,
                self.spec,
                self.window,
                block,
                global_series=global_series,
                in_means=in_means,
                out_means=out_means,
            )
            if design.shape[0] == 0:
                continue
            gram += design.T @ design
            cross += design.T @ target
            seen += design.shape[0]
        if seen == 0:
            self.weights = np.zeros((width, self.window.horizon))
            return self
        gram[np.diag_indices_from(gram)] += self.ridge
        self.weights = np.linalg.solve(gram, cross)
        return self

    def score(
        self,
        activity: Any,
        *,
        global_series: Any = None,
        in_means: Any = None,
        out_means: Any = None,
    ) -> tuple[Any, Any, int]:
        """Absolute error per horizon step and per cell, without holding it all.

        Returns the mean absolute error for each step of the horizon, the mean
        absolute error for each cell, and how many examples went into both. The
        per-cell figures are what the paired bootstrap resamples, so they have
        to survive a recording too large to keep the predictions for.
        """
        import numpy as np

        frames, cells = activity.shape
        times = list(self.window.valid_range(frames))
        step_error = np.zeros(self.window.horizon, dtype=np.float64)
        cell_error = np.zeros(cells, dtype=np.float64)
        examples = 0
        if self.weights is None:
            return step_error, cell_error, 0
        for block in _time_blocks(times, cells, self.block_rows):
            design, target = _design_chunk(
                activity,
                self.spec,
                self.window,
                block,
                global_series=global_series,
                in_means=in_means,
                out_means=out_means,
            )
            if design.shape[0] == 0:
                continue
            errors = np.abs(design @ self.weights - target)
            step_error += errors.sum(axis=0)
            cell_error += errors.mean(axis=1).reshape(-1, cells).sum(axis=0)
            examples += design.shape[0]
        if examples == 0:
            return step_error, cell_error, 0
        return step_error / examples, cell_error / (examples // cells), examples


def baseline_score(
    predict: Any,
    activity: Any,
    window: Window,
    *,
    block_rows: int = 2_000_000,
) -> tuple[Any, Any, int]:
    """Score a baseline the same way, in blocks, so the arms are comparable."""
    import numpy as np

    frames, cells = activity.shape
    times = list(window.valid_range(frames))
    step_error = np.zeros(window.horizon, dtype=np.float64)
    cell_error = np.zeros(cells, dtype=np.float64)
    examples = 0
    for block in _time_blocks(times, cells, block_rows):
        if not block:
            continue
        prediction, target = predict(block)
        if prediction.shape[0] == 0:
            continue
        errors = np.abs(prediction - target)
        step_error += errors.sum(axis=0)
        cell_error += errors.mean(axis=1).reshape(-1, cells).sum(axis=0)
        examples += prediction.shape[0]
    if examples == 0:
        return step_error, cell_error, 0
    return step_error / examples, cell_error / (examples // cells), examples


def mean_baseline(train: Any, test: Any, window: Window) -> Any:
    """Predict each cell's training mean, for every step of the horizon."""
    import numpy as np

    per_cell = train.mean(axis=0)

    def _block(times: Sequence[int]) -> tuple[Any, Any]:
        if not times:
            return np.zeros((0, window.horizon)), np.zeros((0, window.horizon))
        cells = test.shape[1]
        prediction = np.repeat(
            np.repeat(per_cell[:, None], window.horizon, axis=1)[None, :, :],
            len(times),
            axis=0,
        ).reshape(len(times) * cells, window.horizon)
        target = np.concatenate([test[t : t + window.horizon].T for t in times], axis=0)
        return prediction, target

    return _block


def condition_mean_baseline(
    train: Any,
    train_conditions: Sequence[str],
    test: Any,
    test_conditions: Sequence[str],
    window: Window,
) -> Any:
    """ZAPBench's stimulus-conditioned mean: what this cell does under this condition.

    It is the baseline that embarrasses learned models when the stimulus
    explains most of the variance, which is why it belongs here.
    """
    import numpy as np

    per_condition: dict[str, Any] = {}
    for condition in sorted(set(train_conditions)):
        rows = [i for i, c in enumerate(train_conditions) if c == condition]
        if rows:
            per_condition[condition] = train[rows].mean(axis=0)
    fallback = train.mean(axis=0)

    def _block(times: Sequence[int]) -> tuple[Any, Any]:
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

    return _block


def persistence_baseline(test: Any, window: Window) -> Any:
    """Hold the last observed frame for the whole horizon.

    Not one of ZAPBench's arms, and included anyway. It is the baseline a
    forecasting result has to beat before it is a result, and on slow signals it
    is very hard to beat.
    """
    import numpy as np

    def _block(times: Sequence[int]) -> tuple[Any, Any]:
        if not times:
            return np.zeros((0, window.horizon)), np.zeros((0, window.horizon))
        predictions = []
        targets = []
        for t in times:
            predictions.append(np.repeat(test[t - 1][:, None], window.horizon, axis=1))
            targets.append(test[t : t + window.horizon].T)
        return np.concatenate(predictions, axis=0), np.concatenate(targets, axis=0)

    return _block
