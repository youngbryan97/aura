"""core/connectome/zapbench.py — the benchmark, and the question it settles.

ZAPBench fixes a task so that answers to it can be compared: given ``C`` frames
of context, predict the next ``H``. Short context is four frames, long context
is 256, the horizon is 32, the metric is mean absolute error reported per step,
and one stimulus condition is held out of training entirely so that
generalisation to an unseen condition is measured rather than assumed.

This module runs that task on Aura, with her connectome available to the models.
The comparison ZAPBench cannot make is the one the whole thing is for:

    ``connectome`` sees the cells wired to it.
    ``rewired`` sees the same number of cells, chosen by a rewiring that keeps
    every degree and destroys everything else.

If the first beats the second, wiring predicts activity. If it does not, then
either the wiring carries no information about what a cell will do next, or a
linear model over neighbour means cannot find it, and the report says which of
those two the evidence supports rather than choosing the flattering one.

Everything is reported with a paired bootstrap over cells, because two arms
differing in the fourth decimal of an average over six thousand cells is not a
difference until the interval says so.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from .activity import ActivityTrace
from .forecast import (
    ARM_NAMES,
    SPECS,
    RidgeForecaster,
    Window,
    baseline_score,
    condition_mean_baseline,
    mean_baseline,
    neighbour_means,
    persistence_baseline,
)
from .topology import DiGraphView, degree_preserving_rewire
from .types import ConnectomeSnapshot, EdgeKind

logger = logging.getLogger("Aura.Connectome.ZAPBench")

__all__ = [
    "BenchmarkConfig",
    "ArmResult",
    "BenchmarkReport",
    "build_adjacency",
    "predictability",
    "run_benchmark",
    "ZAPBENCH_HORIZON",
    "ZAPBENCH_SHORT_CONTEXT",
    "ZAPBENCH_LONG_CONTEXT",
]

#: ZAPBench's task constants, kept exactly so results are comparable.
ZAPBENCH_HORIZON: int = 32
ZAPBENCH_SHORT_CONTEXT: int = 4
ZAPBENCH_LONG_CONTEXT: int = 256
#: Its split: 70% train, 10% validation, 20% test, by time.
ZAPBENCH_SPLIT: tuple[float, float] = (0.7, 0.8)


@dataclass(frozen=True)
class BenchmarkConfig:
    """Everything the run needs, with ZAPBench's values as the defaults."""

    contexts: tuple[int, ...] = (ZAPBENCH_SHORT_CONTEXT, ZAPBENCH_LONG_CONTEXT)
    horizon: int = ZAPBENCH_HORIZON
    split: tuple[float, float] = ZAPBENCH_SPLIT
    ridge: float = 1.0
    held_out_condition: str = ""
    bootstrap: int = 400
    seed: int = 0
    min_frames_per_context: int = 3
    #: Cells that never fire carry no signal and inflate every average towards
    #: zero, so they are dropped and the count is reported.
    drop_silent_cells: bool = True
    #: Which signal to forecast. "calcium" is the trace a light sheet would have
    #: produced and is what makes a number here comparable with ZAPBench's.
    #: "spikes" is the call counts themselves, which no microscope can see and
    #: which is the right choice when the question is about Aura rather than
    #: about the comparison.
    signal: str = "calcium"
    #: Length of a contiguous block when dealing frames between train and test.
    #: Zero keeps the plain time cut. A block has to hold a context and a
    #: horizon or the examples inside it straddle the boundary.
    stratify_blocks: int = 320
    #: Standardise each cell against its own training statistics before fitting.
    #: A cell that fires a hundred thousand times a frame and one that fires ten
    #: cannot share a weight matrix, and without this the arms are competing to
    #: fit the loudest cells rather than to predict anything. ZAPBench's traces
    #: arrive already comparable across neurons; a call count does not.
    standardise: bool = True


def build_adjacency(
    snapshot: ConnectomeSnapshot,
    uids: Sequence[str],
    *,
    rewire: bool = False,
    seed: int = 0,
) -> tuple[Any, Any, dict[str, Any]]:
    """Row-normalised in- and out-neighbour operators over the recorded cells.

    The graph is induced on the cells that were actually recorded before it is
    rewired, so the null preserves the degrees of the subgraph the models see
    rather than the degrees of a graph half of which was never observed.
    """
    import numpy as np
    from scipy import sparse

    index = {uid: i for i, uid in enumerate(uids)}
    graph = DiGraphView.from_snapshot(snapshot, EdgeKind.DRIVE, drop_isolated=False)
    kept_out = {uid: {p for p in graph.out.get(uid, ()) if p in index} for uid in uids}
    induced = DiGraphView(
        nodes=tuple(uids),
        out={uid: set(targets) for uid, targets in kept_out.items()},
        inbound={
            uid: {p for p in graph.inbound.get(uid, ()) if p in index} for uid in uids
        },
        weights={},
    )
    if rewire:
        induced = degree_preserving_rewire(induced, swaps_per_edge=8, seed=seed)

    rows_in: list[int] = []
    cols_in: list[int] = []
    rows_out: list[int] = []
    cols_out: list[int] = []
    for uid in uids:
        target = index[uid]
        for pre in induced.inbound.get(uid, ()):
            source = index.get(pre)
            if source is not None:
                rows_in.append(target)
                cols_in.append(source)
        for post in induced.out.get(uid, ()):
            source = index.get(post)
            if source is not None:
                rows_out.append(target)
                cols_out.append(source)

    size = len(uids)

    def _normalised(rows: list[int], cols: list[int]) -> Any:
        if not rows:
            return sparse.csr_matrix((size, size), dtype=np.float64)
        data = np.ones(len(rows), dtype=np.float64)
        matrix = sparse.csr_matrix((data, (rows, cols)), shape=(size, size))
        degrees = np.asarray(matrix.sum(axis=1)).ravel()
        degrees[degrees == 0] = 1.0
        scale = sparse.diags(1.0 / degrees)
        return (scale @ matrix).tocsr()

    stats = {
        "cells": size,
        "in_edges": len(rows_in),
        "out_edges": len(rows_out),
        "cells_with_inputs": len({r for r in rows_in}),
        "cells_with_outputs": len({r for r in rows_out}),
        "rewired": rewire,
    }
    return _normalised(rows_in, cols_in), _normalised(rows_out, cols_out), stats


def predictability(activity: Any, conditions: Sequence[str]) -> dict[str, Any]:
    """Whether there is anything here for a forecaster to find.

    A benchmark that cannot separate two models because neither has anything to
    work with reports a tie, and a tie read as a finding is worse than no
    finding. Three numbers settle it before the arms are run.

    The autocorrelation says how much the recent past says about the next
    frame. The between-condition variance share says how much the workload says
    about it. The active-cell count says how sparse the recording is, because a
    trace where fifteen of twelve thousand cells move in a frame is mostly
    zeros and predicting zero is very hard to beat.
    """
    import numpy as np

    if activity.size == 0:
        return {"frames": 0}
    centred = activity - activity.mean(axis=0)
    spread = centred.std(axis=0)
    spread[spread <= 0] = 1.0
    scaled = centred / spread
    lags = [lag for lag in (1, 2, 4, 8, 16, 32) if lag < activity.shape[0]]
    autocorrelation = {
        str(lag): round(float((scaled[:-lag] * scaled[lag:]).mean()), 5) for lag in lags
    }
    unique = sorted(set(conditions))
    share = 0.0
    if len(unique) > 1:
        means = np.array(
            [
                activity[[i for i, c in enumerate(conditions) if c == name]].mean(axis=0)
                for name in unique
            ]
        )
        total = float(activity.var(axis=0).mean())
        share = float(means.var(axis=0).mean() / total) if total > 0 else 0.0
    active = float(np.median((activity != 0).sum(axis=1)))
    lag_one = autocorrelation.get("1", 0.0)
    if lag_one < 0.2:
        verdict = (
            "the recent past says almost nothing about the next frame, so no "
            "forecaster can separate from any other and a tie between arms is "
            "not evidence about structure"
        )
    elif lag_one < 0.5:
        verdict = "weak temporal structure; differences between arms will be small"
    else:
        verdict = "strong temporal structure; the arms have something to compete over"
    return {
        "frames": int(activity.shape[0]),
        "cells": int(activity.shape[1]),
        "autocorrelation": autocorrelation,
        "between_condition_variance_share": round(share, 4),
        "median_active_cells_per_frame": active,
        "verdict": verdict,
    }


def _stratified_rows(
    available: Sequence[int],
    config: BenchmarkConfig,
) -> tuple[list[int], int, int]:
    """Deal contiguous blocks of frames between train, validation and test.

    Frames keep their original order inside a block, so a context window never
    spans a seam that was not in the recording. Ten blocks is what the
    seventy-ten-twenty deal needs to land on the right proportions; a recording
    too short to have both ten blocks and blocks long enough to hold a window
    falls back to the plain time cut rather than producing an empty test side.
    """
    frames = len(available)
    minimum = max(config.contexts) + config.horizon + 1
    block = max(minimum, min(config.stratify_blocks, max(1, frames // 10)))
    if frames // block < 4:
        return (
            list(available),
            int(frames * config.split[0]),
            int(frames * config.split[1]),
        )
    train_rows: list[int] = []
    val_rows: list[int] = []
    test_rows: list[int] = []
    for index, start in enumerate(range(0, frames, block)):
        rows = list(available[start : start + block])
        slot = index % 10
        if slot < 7:
            train_rows.extend(rows)
        elif slot < 8:
            val_rows.extend(rows)
        else:
            test_rows.extend(rows)
    if not test_rows:
        return (
            list(available),
            int(frames * config.split[0]),
            int(frames * config.split[1]),
        )
    return (
        train_rows + val_rows + test_rows,
        len(train_rows),
        len(train_rows) + len(val_rows),
    )


@dataclass
class ArmResult:
    """One model at one context length."""

    arm: str
    context: int
    mae: float
    mae_by_step: list[float]
    examples: int
    per_cell_mae: Any = None

    def as_json(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "context": self.context,
            "mae": round(self.mae, 6),
            "mae_step_1": round(self.mae_by_step[0], 6) if self.mae_by_step else 0.0,
            "mae_step_8": round(self.mae_by_step[7], 6) if len(self.mae_by_step) > 7 else 0.0,
            "mae_step_32": round(self.mae_by_step[-1], 6) if self.mae_by_step else 0.0,
            "examples": self.examples,
        }


@dataclass
class BenchmarkReport:
    """Every arm, plus the paired comparison the benchmark exists to make."""

    dataset: dict[str, Any]
    arms: list[ArmResult]
    structure_test: dict[str, Any]
    predictability: dict[str, Any] = field(default_factory=dict)
    held_out: dict[str, Any] = field(default_factory=dict)
    adjacency: dict[str, Any] = field(default_factory=dict)

    def best(self, context: int) -> ArmResult | None:
        candidates = [arm for arm in self.arms if arm.context == context]
        return min(candidates, key=lambda a: a.mae) if candidates else None

    def as_json(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "adjacency": self.adjacency,
            "predictability": self.predictability,
            "arms": [arm.as_json() for arm in self.arms],
            "structure_test": self.structure_test,
            "held_out": self.held_out,
        }


def _paired_bootstrap(
    left: Any,
    right: Any,
    *,
    draws: int,
    seed: int,
    subset: Any = None,
) -> dict[str, Any]:
    """Bootstrap the per-cell difference between two arms, on mean and median.

    Resampling cells rather than examples is the right unit: examples from one
    cell are not independent of each other.

    Both statistics are reported because on a real recording they disagree and
    the disagreement is the finding. A mean over twelve thousand cells is moved
    by a few hundred with enormous errors; the median and the sign test are not.
    An arm that wins on 93% of cells and loses on the mean has won on almost
    every cell and lost badly on a few, and calling that a loss throws the
    result away.
    """
    import numpy as np

    if left.size == 0 or right.size == 0 or left.size != right.size:
        return {"mean_difference": 0.0, "median_difference": 0.0, "draws": 0}
    raw = left - right
    # A cell with no neighbours in the graph has no cell-specific connectome
    # feature, so the two arms differ for it only through a shared weight — the
    # same small number for every such cell. In a real recording most cells are
    # that cell, and leaving them in gives thousands of identical differences,
    # a median that lands on the same value in every resample, and an interval
    # that collapses to a point and reports as significant. The comparison is
    # restricted to the cells the connectome actually says something about.
    if subset is not None:
        informative = np.asarray(subset, dtype=bool)
    else:
        informative = raw != 0.0
    difference = raw[informative]
    excluded = int((~informative).sum())
    if difference.size < 30:
        return {
            "mean_difference": float(raw.mean()),
            "median_difference": float(np.median(raw)),
            "cells_compared": int(difference.size),
            "cells_identical": excluded,
            "mean_significant": False,
            "median_significant": False,
            "draws": 0,
            "reason": "too few cells are treated differently by the two arms",
        }
    rng = np.random.default_rng(seed)
    n = difference.size
    means = np.empty(draws, dtype=np.float64)
    medians = np.empty(draws, dtype=np.float64)
    for i in range(draws):
        picks = rng.integers(0, n, size=n)
        sample = difference[picks]
        means[i] = sample.mean()
        medians[i] = np.median(sample)
    mean_low, mean_high = np.percentile(means, [2.5, 97.5])
    median_low, median_high = np.percentile(medians, [2.5, 97.5])
    # A confidence interval that is a single point is not a tight interval, it
    # is a distribution with a spike in it, and saying so is the difference
    # between a result and an artefact.
    _, tallies = np.unique(np.round(difference, 12), return_counts=True)
    degenerate = float(tallies.max() / difference.size) if tallies.size else 0.0
    better = int((difference < 0).sum())
    worse = int((difference > 0).sum())
    return {
        "cells_compared": int(difference.size),
        "cells_identical": excluded,
        "mean_difference": float(difference.mean()),
        "mean_ci_low": float(mean_low),
        "mean_ci_high": float(mean_high),
        "mean_significant": bool(mean_low > 0 or mean_high < 0),
        "median_difference": float(np.median(difference)),
        "median_ci_low": float(median_low),
        "median_ci_high": float(median_high),
        "median_significant": bool(
            (median_low > 0 or median_high < 0) and degenerate < 0.2
        ),
        "largest_shared_value_share": round(degenerate, 4),
        "cells_better": better,
        "cells_worse": worse,
        "share_better": round(better / max(1, better + worse), 4),
        "draws": draws,
    }


def run_benchmark(
    trace: ActivityTrace,
    snapshot: ConnectomeSnapshot,
    config: BenchmarkConfig | None = None,
) -> BenchmarkReport:
    """Run every arm at every context and settle the structure question."""

    config = config or BenchmarkConfig()
    activity = trace.matrix() if config.signal == "spikes" else trace.calcium()
    uids = list(trace.uids)
    conditions = list(trace.conditions)
    if activity.size == 0:
        return BenchmarkReport(dataset={"frames": 0, "cells": 0}, arms=[], structure_test={})

    if config.drop_silent_cells:
        alive = activity.std(axis=0) > 0
        dropped = int((~alive).sum())
        activity = activity[:, alive]
        uids = [uid for uid, keep in zip(uids, alive, strict=False) if keep]
    else:
        dropped = 0

    frames, cells = activity.shape
    held_out_rows: list[int] = []
    if config.held_out_condition:
        held_out_rows = [i for i, c in enumerate(conditions) if c == config.held_out_condition]
    available = [i for i in range(frames) if i not in set(held_out_rows)]

    # A single time cut puts whole conditions on one side, which makes the
    # stimulus-conditioned mean identical to the plain mean and hides the two
    # thirds of the variance the workload explains. Blocks are dealt round robin
    # instead, so every condition appears on both sides and each block stays
    # long enough to hold a context and a horizon.
    #
    # The split returns row numbers into the original recording rather than a
    # reordered array. Everything downstream — the neighbour channels, the
    # global series, the conditions — is indexed by those same rows, which is
    # the only way the arms stay aligned with the frames they are predicting.
    if config.stratify_blocks > 0:
        ordered_rows, cut_train, cut_val = _stratified_rows(available, config)
    else:
        ordered_rows = list(available)
        cut_train = int(len(ordered_rows) * config.split[0])
        cut_val = int(len(ordered_rows) * config.split[1])

    train_rows = ordered_rows[:cut_train]
    test_rows = ordered_rows[cut_val:]

    if config.standardise and train_rows:
        reference = activity[train_rows]
        centre = reference.mean(axis=0)
        scale = reference.std(axis=0)
        scale[scale <= 0] = 1.0
        activity = (activity - centre) / scale

    kept = activity[ordered_rows]
    kept_conditions = [conditions[i] for i in ordered_rows]
    train = activity[train_rows]
    test = activity[test_rows]
    train_conditions = [conditions[i] for i in train_rows]
    test_conditions = [conditions[i] for i in test_rows]

    in_adj, out_adj, in_stats = build_adjacency(snapshot, uids, rewire=False, seed=config.seed)
    rin_adj, rout_adj, rewired_stats = build_adjacency(
        snapshot, uids, rewire=True, seed=config.seed + 7
    )

    channels = {
        False: (
            neighbour_means(activity, in_adj),
            neighbour_means(activity, out_adj),
        ),
        True: (
            neighbour_means(activity, rin_adj),
            neighbour_means(activity, rout_adj),
        ),
    }
    global_series = activity.mean(axis=1)

    def _slice(source: Any, rows: Sequence[int]) -> Any:
        return source[rows]

    arms: list[ArmResult] = []
    per_cell: dict[tuple[str, int], Any] = {}

    for context in config.contexts:
        window = Window(context=context, horizon=config.horizon)
        if len(train) < context + config.horizon + config.min_frames_per_context:
            logger.info("context %d skipped: the recording is too short", context)
            continue
        if len(test) < context + config.horizon:
            logger.info("context %d skipped: the test split is too short", context)
            continue

        for name, block in (
            ("mean", mean_baseline(train, test, window)),
            (
                "condition_mean",
                condition_mean_baseline(
                    train, train_conditions, test, test_conditions, window
                ),
            ),
            ("persistence", persistence_baseline(test, window)),
        ):
            step_error, _cells, examples = baseline_score(block, test, window)
            arms.append(
                ArmResult(
                    arm=name,
                    context=context,
                    mae=float(step_error.mean()) if examples else 0.0,
                    mae_by_step=[float(v) for v in step_error],
                    examples=examples,
                )
            )

        for arm in ARM_NAMES:
            spec = SPECS[arm]
            rewired = arm == "rewired"
            in_means, out_means = channels[rewired]
            model = RidgeForecaster(spec=spec, window=window, ridge=config.ridge)
            model.fit(
                train,
                global_series=_slice(global_series, train_rows),
                in_means=_slice(in_means, train_rows),
                out_means=_slice(out_means, train_rows),
            )
            step_error, cell_error, examples = model.score(
                test,
                global_series=_slice(global_series, test_rows),
                in_means=_slice(in_means, test_rows),
                out_means=_slice(out_means, test_rows),
            )
            arms.append(
                ArmResult(
                    arm=arm,
                    context=context,
                    mae=float(step_error.mean()) if examples else 0.0,
                    mae_by_step=[float(v) for v in step_error],
                    examples=examples,
                )
            )
            per_cell[(arm, context)] = cell_error

    # The cells the connectome has anything to say about: those with at least
    # one neighbour among the recorded cells. Everything else is compared on a
    # feature that is zero in every arm.
    import numpy as np

    connected = np.asarray(
        (np.asarray(in_adj.sum(axis=1)).ravel() > 0)
        | (np.asarray(out_adj.sum(axis=1)).ravel() > 0)
    )

    structure: dict[str, Any] = {}
    for context in config.contexts:
        left = per_cell.get(("connectome", context))
        right = per_cell.get(("rewired", context))
        blind = per_cell.get(("blind", context))
        if left is None or right is None:
            continue
        against_null = _paired_bootstrap(
            left, right, draws=config.bootstrap, seed=config.seed + 11, subset=connected
        )
        against_blind = (
            _paired_bootstrap(
                left, blind, draws=config.bootstrap, seed=config.seed + 13, subset=connected
            )
            if blind is not None
            else {}
        )
        median = against_null.get("median_difference", 0.0)
        share = against_null.get("share_better", 0.0)
        if against_null.get("median_significant") and median < 0:
            verdict = (
                f"wiring predicts activity: the connectome beats its own rewiring on "
                f"{share:.1%} of cells and on the median"
            )
        elif against_null.get("median_significant"):
            verdict = "the rewiring beats the connectome, which needs explaining before use"
        else:
            verdict = (
                "no detectable effect of wiring on activity under a linear model "
                "over neighbour means"
            )
        if (
            against_null.get("median_significant")
            and against_null.get("mean_significant")
            and median * against_null.get("mean_difference", 0.0) < 0
        ):
            verdict += (
                "; the mean disagrees with the median, so a small number of cells "
                "carry most of the error"
            )
        structure[f"context_{context}"] = {
            "connectome_vs_rewired": against_null,
            "connectome_vs_blind": against_blind,
            "verdict": verdict,
        }

    held: dict[str, Any] = {}
    if held_out_rows and len(held_out_rows) >= config.horizon + min(config.contexts) + 1:
        held_activity = activity[held_out_rows]
        for context in config.contexts:
            window = Window(context=context, horizon=config.horizon)
            if len(held_activity) < context + config.horizon:
                continue
            in_means, out_means = channels[False]
            model = RidgeForecaster(spec=SPECS["connectome"], window=window, ridge=config.ridge)
            model.fit(
                train,
                in_means=_slice(in_means, train_rows),
                out_means=_slice(out_means, train_rows),
            )
            step_error, _cells, examples = model.score(
                held_activity,
                in_means=_slice(in_means, held_out_rows),
                out_means=_slice(out_means, held_out_rows),
            )
            if examples:
                import numpy as np

                cell_error = np.asarray(_cells)
                held[f"context_{context}"] = {
                    "condition": config.held_out_condition,
                    "mae": float(step_error.mean()),
                    "median_cell_mae": float(np.median(cell_error)),
                    "cells_over_100x_median": int(
                        (cell_error > 100.0 * max(1e-9, float(np.median(cell_error)))).sum()
                    ),
                    "frames": len(held_out_rows),
                    "note": (
                        "the mean is carried by cells the training split never saw fire, "
                        "whose standardised values are unbounded; the median is the "
                        "number to read"
                    ),
                }

    return BenchmarkReport(
        predictability=predictability(kept, kept_conditions),
        dataset={
            "frames": frames,
            "cells": cells,
            "silent_cells_dropped": dropped,
            "train_frames": len(train),
            "test_frames": len(test),
            "conditions": sorted(set(conditions)),
            "test_conditions": sorted(set(test_conditions)),
            "test_conditions_unseen_in_training": sorted(
                set(test_conditions) - set(train_conditions)
            ),
            "held_out_condition": config.held_out_condition,
            "held_out_frames": len(held_out_rows),
            "frame_seconds": trace.frame_seconds,
            "horizon": config.horizon,
            "signal": config.signal,
        },
        arms=arms,
        structure_test=structure,
        held_out=held,
        adjacency={"connectome": in_stats, "rewired": rewired_stats},
    )
