"""Runtime-facing collectors for Subject Core v25.

Designed against Aura commit 171cb15... and the existing offline SubjectRuntime.
The collectors deliberately reuse SubjectRuntime.snapshot/restore, the current
domain clamp, and clamp.compose instead of editing cognitive phases.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from core.subject.clamp import clamped, compose
from core.subject.state import DOMAINS


@dataclass(frozen=True)
class Anchor:
    snapshot: Any
    current: np.ndarray
    history: np.ndarray
    source_condition: str


@dataclass(frozen=True)
class PartitionTrajectories:
    intact_a: list[Any]
    intact_b: list[Any]
    cut: list[Any]
    left_free: list[Any]
    right_free: list[Any]


def bipartitions(domains: Sequence[str] = DOMAINS) -> tuple[tuple[tuple[str, ...], tuple[str, ...]], ...]:
    """Every nontrivial bipartition exactly once.

    The first domain is fixed on the left to quotient A|B and B|A.
    Ten domains -> 511 cuts.
    """
    items = tuple(domains)
    if len(items) < 2:
        return ()
    first = items[0]
    tail = items[1:]
    out: list[tuple[tuple[str, ...], tuple[str, ...]]] = []
    # Choose any subset of tail to join `first`, except all of tail.
    for mask in range(1 << len(tail)):
        left = (first,) + tuple(tail[i] for i in range(len(tail)) if mask & (1 << i))
        if len(left) == len(items):
            continue
        left_set = set(left)
        right = tuple(x for x in items if x not in left_set)
        out.append((left, right))
    return tuple(out)


async def _run_turns(runtime: Any, condition: Any, turns: int) -> list[Any]:
    rows: list[Any] = []
    for _ in range(turns):
        rows.extend(await runtime.turn_once(condition))
    return rows


async def paired_partition_trajectories(
    runtime: Any,
    snapshot: Any,
    condition: Any,
    *,
    left: Sequence[str],
    right: Sequence[str],
    turns: int,
) -> PartitionTrajectories:
    """Intact/sham and true complementary-clamp cut from one common fork."""
    runtime.restore(snapshot)
    intact_a = await _run_turns(runtime, condition, turns)

    runtime.restore(snapshot)
    intact_b = await _run_turns(runtime, condition, turns)

    runtime.restore(snapshot)
    with clamped(runtime, right):
        left_free = await _run_turns(runtime, condition, turns)

    runtime.restore(snapshot)
    with clamped(runtime, left):
        right_free = await _run_turns(runtime, condition, turns)

    cut = compose(left_free, right_free, left)
    return PartitionTrajectories(
        intact_a=intact_a,
        intact_b=intact_b,
        cut=cut,
        left_free=left_free,
        right_free=right_free,
    )


def lag_vector(rows: Sequence[Any], lag: int) -> np.ndarray:
    """Vector after `lag` recorded frames from the common fork."""
    if lag < 1:
        raise ValueError("lag is one-based and must be >= 1")
    if not rows:
        raise ValueError("trajectory is empty")
    index = min(lag - 1, len(rows) - 1)
    return np.asarray(rows[index].vector(), dtype=np.float64)


#: What the two untouched forks from one anchor recorded, kept as the vectors
#: each horizon reads and the length of each fork. A frame is about 284 KB and a
#: sweep holds one pair per anchor for its whole length, so the frames are not.
Untouched = dict[tuple[int, str, int], tuple[dict[int, np.ndarray], dict[int, np.ndarray], int]]


async def collect_partition_samples(
    runtime: Any,
    anchors: Sequence[Anchor],
    conditions: Sequence[Any],
    *,
    left: Sequence[str],
    right: Sequence[str],
    turns: int,
    lags: Sequence[int],
    offset: int = 0,
    untouched: Untouched | None = None,
) -> dict[int, dict[str, np.ndarray]]:
    """Collect matched context/intact/cut/sham arrays for one physical cut.

    `offset` is where `anchors` starts in the bank, so a later round that
    collects only the anchors it has not seen pairs each one with the
    condition it had when the whole bank was collected at once.

    `untouched` holds the two untouched forks from each anchor across cuts.
    Neither of them depends on the cut: two of every cut's four arms were the
    same two runs from the same snapshot, repeated for each of 511 cuts.
    """
    buckets: dict[int, dict[str, list[np.ndarray]]] = {
        int(lag): {
            "context": [], "intact": [], "cut": [], "sham_a": [], "sham_b": [],
            # Whether every arm of this rollout actually recorded this many
            # frames. `lag_vector` clamps a lag past the end to the last frame,
            # so a lag longer than the rollout would otherwise be read as that
            # frame and scored as though it were a different horizon.
            "reached": [],
        }
        for lag in lags
    }
    condition_names = [getattr(c, "name", str(i)) for i, c in enumerate(conditions)]
    name_to_index = {name: i for i, name in enumerate(condition_names)}

    # Anchor/condition PAIRS, not their cross product. Every cut needs matched
    # contexts and it needs them across conditions, and taking the product
    # multiplies the cost of one cut by eight before the sweep has looked at
    # the other five hundred and ten. Pairing keeps the same spread of
    # conditions over the bank at a cost that is linear in the anchors, which
    # is what makes an exhaustive sweep affordable at all.
    pairs = [
        (offset + index, anchor, conditions[(offset + index) % len(conditions)])
        for index, anchor in enumerate(anchors)
    ]
    for position, anchor, condition in pairs:
        condition_name = getattr(condition, "name", "")
        key = (position, condition_name, int(turns))
        if untouched is not None and key in untouched and all(int(lag) in untouched[key][0] for lag in lags):
            intact_a, intact_b, untouched_length = untouched[key]
        else:
            runtime.restore(anchor.snapshot)
            arm_a = await _run_turns(runtime, condition, turns)
            runtime.restore(anchor.snapshot)
            arm_b = await _run_turns(runtime, condition, turns)
            intact_a = {int(lag): lag_vector(arm_a, int(lag)) for lag in lags}
            intact_b = {int(lag): lag_vector(arm_b, int(lag)) for lag in lags}
            untouched_length = min(len(arm_a), len(arm_b))
            if untouched is not None:
                untouched[key] = (intact_a, intact_b, untouched_length)
        runtime.restore(anchor.snapshot)
        with clamped(runtime, right):
            left_free = await _run_turns(runtime, condition, turns)
        runtime.restore(anchor.snapshot)
        with clamped(runtime, left):
            right_free = await _run_turns(runtime, condition, turns)
        cut = compose(left_free, right_free, left)
        one_hot = np.zeros(len(conditions), dtype=np.float64)
        one_hot[name_to_index[condition_name]] = 1.0
        context = np.concatenate([anchor.current, one_hot])
        shortest = min(untouched_length, len(cut))
        for lag in lags:
            slot = buckets[int(lag)]
            slot["context"].append(context)
            # `intact` and `sham_a` are the same arm on purpose. The signal is
            # that arm against the cut; the floor is it against a second
            # untouched fork from the same snapshot. Two untouched forks are
            # not numerically identical in practice, and the third arm is what
            # measures how far apart they are.
            slot["intact"].append(intact_a[int(lag)])
            slot["sham_a"].append(intact_a[int(lag)])
            slot["sham_b"].append(intact_b[int(lag)])
            slot["cut"].append(lag_vector(cut, int(lag)))
            slot["reached"].append(np.asarray([1.0 if int(lag) <= shortest else 0.0]))

    return {
        lag: {key: np.vstack(values) for key, values in slot.items()}
        for lag, slot in buckets.items()
    }


async def collect_anchor_bank(
    runtime: Any,
    conditions: Sequence[Any],
    *,
    rounds: int,
    history_turns: int = 8,
    every: int = 1,
) -> tuple[Anchor, ...]:
    """Collect forkable ordinary-life states and their recent turn histories."""
    history: list[np.ndarray] = []
    anchors: list[Anchor] = []
    turn_index = 0
    for _ in range(rounds):
        for condition in conditions:
            frames = await runtime.turn_once(condition)
            if not frames:
                continue
            end = np.asarray(frames[-1].vector(), dtype=np.float64)
            history.append(end)
            turn_index += 1
            if turn_index % max(1, every):
                continue
            window = history[-history_turns:]
            # Left-pad with the earliest available state so history width is fixed.
            if len(window) < history_turns:
                window = [window[0]] * (history_turns - len(window)) + window
            anchors.append(
                Anchor(
                    snapshot=runtime.snapshot(),
                    current=end.copy(),
                    history=np.concatenate(window),
                    source_condition=getattr(condition, "name", ""),
                )
            )
    return tuple(anchors)
