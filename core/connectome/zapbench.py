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
    held_out: dict[str, Any] = field(default_factory=dict)
    adjacency: dict[str, Any] = field(default_factory=dict)

    def best(self, context: int) -> ArmResult | None:
        candidates = [arm for arm in self.arms if arm.context == context]
        return min(candidates, key=lambda a: a.mae) if candidates else None

    def as_json(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "adjacency": self.adjacency,
            "arms": [arm.as_json() for arm in self.arms],
            "structure_test": self.structure_test,
            "held_out": self.held_out,
        }


def _per_cell_absolute_error(prediction: Any, target: Any, cells: int) -> Any:
    """Mean absolute error per cell, with examples stacked cell-major."""
    import numpy as np

    if prediction.shape[0] == 0:
        return np.zeros(cells)
    errors = np.abs(prediction - target).mean(axis=1)
    blocks = errors.reshape(-1, cells)
    return blocks.mean(axis=0)


def _paired_bootstrap(
    left: Any,
    right: Any,
    *,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    """Bootstrap the per-cell difference between two arms.

    Resampling cells rather than examples is the right unit here: examples from
    one cell are not independent of each other, and treating them as though
    they were is how a difference in the fourth decimal acquires a tight
    interval it has not earned.
    """
    import numpy as np

    if left.size == 0 or right.size == 0 or left.size != right.size:
        return {"difference": 0.0, "ci_low": 0.0, "ci_high": 0.0, "draws": 0}
    difference = left - right
    rng = np.random.default_rng(seed)
    n = difference.size
    samples = np.empty(draws, dtype=np.float64)
    for i in range(draws):
        picks = rng.integers(0, n, size=n)
        samples[i] = difference[picks].mean()
    low, high = np.percentile(samples, [2.5, 97.5])
    observed = float(difference.mean())
    return {
        "difference": observed,
        "ci_low": float(low),
        "ci_high": float(high),
        "draws": draws,
        "cells_better": int((difference < 0).sum()),
        "cells_worse": int((difference > 0).sum()),
        "significant": bool(low > 0 or high < 0),
    }


def run_benchmark(
    trace: ActivityTrace,
    snapshot: ConnectomeSnapshot,
    config: BenchmarkConfig | None = None,
) -> BenchmarkReport:
    """Run every arm at every context and settle the structure question."""
    import numpy as np

    config = config or BenchmarkConfig()
    activity = trace.calcium()
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
    keep_rows = [i for i in range(frames) if i not in set(held_out_rows)]
    kept = activity[keep_rows]
    kept_conditions = [conditions[i] for i in keep_rows]

    cut_train = int(len(kept) * config.split[0])
    cut_val = int(len(kept) * config.split[1])
    train = kept[:cut_train]
    test = kept[cut_val:]
    train_conditions = kept_conditions[:cut_train]
    test_conditions = kept_conditions[cut_val:]

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

        prediction, target = mean_baseline(train, test, window)
        arms.append(
            ArmResult(
                arm="mean",
                context=context,
                mae=float(np.abs(prediction - target).mean()),
                mae_by_step=list(np.abs(prediction - target).mean(axis=0)),
                examples=int(prediction.shape[0]),
            )
        )
        prediction, target = condition_mean_baseline(
            train, train_conditions, test, test_conditions, window
        )
        arms.append(
            ArmResult(
                arm="condition_mean",
                context=context,
                mae=float(np.abs(prediction - target).mean()),
                mae_by_step=list(np.abs(prediction - target).mean(axis=0)),
                examples=int(prediction.shape[0]),
            )
        )
        prediction, target = persistence_baseline(test, window)
        arms.append(
            ArmResult(
                arm="persistence",
                context=context,
                mae=float(np.abs(prediction - target).mean()),
                mae_by_step=list(np.abs(prediction - target).mean(axis=0)),
                examples=int(prediction.shape[0]),
            )
        )

        for arm in ARM_NAMES:
            spec = SPECS[arm]
            rewired = arm == "rewired"
            in_means, out_means = channels[rewired]
            model = RidgeForecaster(spec=spec, window=window, ridge=config.ridge)
            model.fit(
                train,
                global_series=global_series[:cut_train],
                in_means=_slice(in_means, keep_rows)[:cut_train],
                out_means=_slice(out_means, keep_rows)[:cut_train],
            )
            prediction, target = model.predict(
                test,
                global_series=global_series[cut_val + len(held_out_rows) :][: len(test)],
                in_means=_slice(in_means, keep_rows)[cut_val:],
                out_means=_slice(out_means, keep_rows)[cut_val:],
            )
            errors = np.abs(prediction - target)
            arms.append(
                ArmResult(
                    arm=arm,
                    context=context,
                    mae=float(errors.mean()),
                    mae_by_step=list(errors.mean(axis=0)),
                    examples=int(prediction.shape[0]),
                )
            )
            per_cell[(arm, context)] = _per_cell_absolute_error(prediction, target, cells)

    structure: dict[str, Any] = {}
    for context in config.contexts:
        left = per_cell.get(("connectome", context))
        right = per_cell.get(("rewired", context))
        blind = per_cell.get(("blind", context))
        if left is None or right is None:
            continue
        against_null = _paired_bootstrap(
            left, right, draws=config.bootstrap, seed=config.seed + 11
        )
        against_blind = (
            _paired_bootstrap(left, blind, draws=config.bootstrap, seed=config.seed + 13)
            if blind is not None
            else {}
        )
        if against_null.get("significant") and against_null["difference"] < 0:
            verdict = "wiring predicts activity: the connectome beats its own rewiring"
        elif against_null.get("significant"):
            verdict = "the rewiring beats the connectome, which needs explaining before use"
        else:
            verdict = (
                "no detectable effect of wiring on activity under a linear model "
                "over neighbour means"
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
                in_means=_slice(in_means, keep_rows)[:cut_train],
                out_means=_slice(out_means, keep_rows)[:cut_train],
            )
            prediction, target = model.predict(
                held_activity,
                in_means=_slice(in_means, held_out_rows),
                out_means=_slice(out_means, held_out_rows),
            )
            if prediction.shape[0]:
                held[f"context_{context}"] = {
                    "condition": config.held_out_condition,
                    "mae": float(np.abs(prediction - target).mean()),
                    "frames": len(held_out_rows),
                }

    return BenchmarkReport(
        dataset={
            "frames": frames,
            "cells": cells,
            "silent_cells_dropped": dropped,
            "train_frames": len(train),
            "test_frames": len(test),
            "conditions": sorted(set(conditions)),
            "held_out_condition": config.held_out_condition,
            "held_out_frames": len(held_out_rows),
            "frame_seconds": trace.frame_seconds,
            "horizon": config.horizon,
        },
        arms=arms,
        structure_test=structure,
        held_out=held,
        adjacency={"connectome": in_stats, "rewired": rewired_stats},
    )
