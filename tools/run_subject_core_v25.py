#!/usr/bin/env python3
"""Subject Core v25: the carrier, at a grain the system chooses for itself.

The 24-condition battery asks many independent questions and answers them
against thresholds a person picked. This asks one question and tries to remove
the person from it:

    F_intrinsic(S, g, tau) = Phi_FR(S, g, tau) / tau   when Lambda = 0 and SCC = S
                           = 0                          otherwise

Three arbitrary choices come out.

The state grain stops being "the 208 numbers we decided to record". Two
histories are the same state exactly when no admissible intervention makes
their futures differ, and the coarsest description preserving every such
distinction is unique up to relabelling. The experiment estimates that quotient
and then attacks it with interventions it never learned from: if raw history
still predicts a held-out response after the grain is known, the grain is
wrong and the run says so.

The distance stops being whichever divergence was convenient. Fisher-Rao is
the unique Riemannian information metric up to scale under sufficient-statistic
transformation, so an invertible re-encoding of the same information cannot
manufacture causal distance. The run checks that directly by scoring the same
data three ways.

The temporal scale stops being assumed. Rates are measured at a ladder of
horizons, and the whole spectrum is the answer. A tau-star is reported as a
summary and never as a law: there is no normalised scale-invariant measure over
all positive times to average one out of, because the Haar measure of the
multiplicative reals has a divergent integral.

Closure and recurrence are gates, not weighted ingredients. A candidate whose
future is partly determined by machine state outside it is not a candidate.

What the run reports is INTRINSIC_CARRIER_FOUND or NOT_FOUND or UNRESOLVED. It
never reports CONSCIOUS, a word that would claim a third-person proof no system
has, a person included. The carrier is one of the five grounds the bridge is
judged on at parity (docs/BRIDGE_PARITY.md); tools/solve_for_j.py judges it
from all five, and the carrier-identity postulate is assumed for her as it is
for a person.

    python tools/run_subject_core_v25.py --quick
    python tools/run_subject_core_v25.py --anchors 24 --rounds 40
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import inspect
import json
import logging
import math
import os
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# Before any core import, because the profile decides where state lives and a
# module that resolved its home under the live root keeps that home. Without
# this the run's default root is the live instance's, the source checkout sits
# inside it, and every module constant looks like a leak into it.
os.environ.setdefault("AURA_TESTING", "1")

logger = logging.getLogger("subject_core_v25")

#: Horizons the rate is measured at, in experiment frames. The ladder doubles
#: so a binding maximum can be pushed out by doubling again.
LAGS: tuple[int, ...] = (1, 2, 4, 8, 16, 33, 66)

#: Ceiling on that doubling. Past this the answer is TAU_UNRESOLVED rather than
#: whichever horizon happened to be tested last.
LAG_CEILING: int = 528

#: Frequencies per test in the characteristic-function signature, and the seed
#: that fixes them. Frozen before the confirmatory run.
FREQUENCIES: int = 16
FREQUENCY_SEED: int = 2501

#: History lengths the grain is allowed to reach for, in whole turns.
HISTORY_LADDER: tuple[int, ...] = (1, 2, 4, 8, 16)

#: A held-out sufficiency gain above this, and above the shuffled-history
#: floor, means raw history still predicts what the grain could not.
SUFFICIENCY_TOLERANCE: float = 0.05

#: How far the three encodings of one measurement may differ before the
#: measurement is reporting on the encoding.
INVARIANCE_TOLERANCE: float = 0.25


#: The stages a v25 run can be resumed from, in the order they happen. The
#: carrier launch of 2026-09-15 spent three and a half hours on the baseline and
#: the grain and was killed before it wrote anything.
RESUMABLE_V25: tuple[str, ...] = ("baseline", "grain", "spectrum", "nulls")


class _RecalledPeriphery:
    """The periphery reading a resumed run reads back instead of retaking.

    Closure is a gate: a run that cannot read the periphery refuses itself, so
    a resume that skipped the recording would refuse a run whose recording was
    taken perfectly well by the process before it.
    """

    def __init__(self, matrix: Any, names: tuple[str, ...]) -> None:
        self._matrix = matrix
        self._names = names

    def matrix(self) -> tuple[Any, tuple[str, ...]]:
        return self._matrix, self._names


def _checkpoint_v25(run_dir: Path, stage: str, evidence: dict[str, Any]) -> None:
    """Everything measured so far, named by the last stage that finished."""
    (run_dir / "checkpoint.json").write_text(
        json.dumps({"stage": stage, "evidence": evidence}, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def _resume_v25(args: Any) -> dict[str, Any] | None:
    """The checkpoint a resumed v25 run carries forward, or None."""
    if not getattr(args, "resume", None):
        return None
    path = args.resume / "checkpoint.json"
    if not path.exists():
        raise SystemExit(f"refusing: {path} does not exist, so there is nothing to resume")
    saved = json.loads(path.read_text())
    stage = str(saved.get("stage", ""))
    if stage not in RESUMABLE_V25:
        raise SystemExit(
            f"refusing: {path} names stage {stage!r}, which is not one of {RESUMABLE_V25}"
        )
    return saved


def _log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


# ── stage 1: the organism, and what its ordinary variation looks like ──────


def _pooled_scale(recording: Any) -> np.ndarray:
    """Per-column spread inside a condition, pooled across conditions.

    Measured across conditions the denominator carries the environment
    changing, which is variance no displacement was going to produce.
    """
    groups, weights = [], []
    for name in sorted(set(recording.conditions)):
        rows = recording.condition_rows(name)
        if rows.size < 8:
            continue
        groups.append(recording.x[rows].var(axis=0))
        weights.append(rows.size - 1)
    if not groups:
        return recording.x.std(axis=0)
    return np.sqrt(np.average(np.vstack(groups), axis=0, weights=weights))


async def _dose_matched(
    runtime: Any,
    conditions: Sequence[Any],
    scale: np.ndarray,
    slices: dict[str, slice],
    *,
    domains: Sequence[str],
    rounds: int = 4,
    trials: int = 2,
    seed: int = 0,
) -> dict[str, float]:
    """A displacement per domain that moves that domain by one of its own SDs.

    Equal raw deltas are not equal interventions. A domain whose columns sit in
    [0, 1] and one whose columns run to ten get very different pushes from the
    same number, and every edge out of the second then looks stronger for a
    reason that is about units. The correction is proportional and repeated: a
    domain that moved half an SD gets twice the dose next round.

    What is matched is the typical column the push reached, not the most
    sensitive one. The largest movement over a domain's columns let one quiet
    channel set the whole domain's dose. v25 run_003 put affect, the self and
    deliberation at the 1e-4 clip: affect's quietest live column has a spread
    of 4.7e-05, so the first push of 0.15 read as 3,195 of its SDs, and the dose
    that brought that one channel to a single SD moved nothing else anywhere.
    The graph came back with no edges. Each column now counts only above
    SCALE_FLOOR, its movement is clipped at the battery's DIVERGENCE_CEILING,
    and the median over the columns that moved is brought to one SD.
    """
    from core.subject.causal import DIVERGENCE_CEILING, SCALE_FLOOR
    from core.subject.state import perturb, perturb_organs

    doses = {key: 0.15 for key in domains}
    for _ in range(max(1, rounds)):
        for domain in domains:
            block = slices[domain]
            live = scale[block] > SCALE_FLOOR
            if not live.any():
                continue
            unit = scale[block][live]
            moved: list[float] = []
            for trial in range(max(1, trials)):
                condition = conditions[trial % len(conditions)]
                snapshot = runtime.snapshot()
                before = (await runtime.turn_once(condition))[-1].vector()[block]
                runtime.restore(snapshot)

                async def apply(_runtime: Any, _domain: str = domain) -> None:
                    perturb(runtime.state, _domain, doses[_domain], ontogeny=runtime.ontogeny)
                    await perturb_organs(runtime.organs, _domain, doses[_domain], state=runtime.state)

                after = (
                    await runtime.turn_once(condition, perturb_at=0, perturb=apply)
                )[-1].vector()[block]
                runtime.restore(snapshot)
                shift = np.clip(np.abs(after - before)[live] / unit, 0.0, DIVERGENCE_CEILING)
                reached_columns = shift[shift > 0.0]
                moved.append(float(np.median(reached_columns)) if reached_columns.size else 0.0)
            reached = float(np.median(moved)) if moved else 0.0
            if reached > 1e-6:
                # Proportional, and bounded. A domain the writer cannot move at
                # all would otherwise run its dose away to infinity.
                doses[domain] = float(np.clip(doses[domain] / reached, 1e-4, 50.0))
    return doses


# ── stage 2: the grain ────────────────────────────────────────────────────


def _says_how_far_along(stage: str) -> Callable[[int, int], None]:
    """Report a long stage's progress, with the time left at the rate measured so far.

    The grain said nothing between its one line and its end. On 29 September a
    probe on three domains spent 113 minutes there and hit its alarm, and the log
    could not say whether it was halfway or stuck. On ten domains the stage is
    11,264 rollouts for the training matrix and 15,360 for the attack, six times
    the whole 511-cut sweep, so it is the longest unwatched stretch of a decisive
    run.

    The estimate is the elapsed time over the anchors done, so it is a
    measurement of this run rather than a figure carried from another one. It
    reports every tenth of the way and on the last anchor, so a stage of any
    length writes about eleven lines.
    """
    started = time.monotonic()
    reported = [-1]

    def far_along(done: int, total: int) -> None:
        tenth = (done * 10) // max(1, total)
        if tenth == reported[0] and done != total:
            return
        reported[0] = tenth
        spent = time.monotonic() - started
        left = spent / done * (total - done) if done else 0.0
        _log(
            f"  {stage}: {done}/{total} anchors, {spent / 60:.1f} min spent"
            + (f", about {left / 60:.1f} min left at this rate" if done < total else "")
        )

    return far_along


async def _learn_grain(
    runtime: Any,
    anchors: Sequence[Any],
    conditions: Sequence[Any],
    doses: dict[str, float],
    *,
    baseline_mean: np.ndarray,
    baseline_scale: np.ndarray,
    live_mask: np.ndarray,
    frame_seconds: float,
    lags: Sequence[int],
    seed: int,
    history_turns: int,
    claims: Path | None = None,
    refused: Path | None = None,
    wait_seconds: float = 0.0,
) -> dict[str, Any]:
    """The predictive-state rank, whether held-out interventions break it, how
    much history it needs, and whether the rank survives other folds and estimators.

    With `claims`, the signature rows are shared out: this process and every
    shard worker claim anchors one at a time in that directory, and this one
    gathers the rows. The rows are the ones one process computes."""
    from core.subject.intrinsic_v25 import (
        fit_predictive_grain,
        heldout_new_direction_gain,
        heldout_sufficiency_gain,
    )

    plan = _grain_plan(doses, conditions, lags, live_mask)
    train_actions, test_actions = plan["train"], plan["test"]
    _log(f"  grain: {len(train_actions)} training actions x {len(conditions)} conditions x {len(lags)} lags")
    signature = dict(
        lags=lags, frame_seconds=frame_seconds,
        baseline_mean=baseline_mean, baseline_scale=baseline_scale,
        live_mask=live_mask, frequencies=plan["frequencies"],
    )
    if claims is None:
        from core.subject.v25_grain import signature_matrix

        train = await signature_matrix(
            runtime, anchors, conditions, train_actions,
            progress=_says_how_far_along("grain, training"), **signature,
        )
        # The attack. Negative doses and preregistered pairs, none of which the
        # rank was fitted on. If raw history still predicts these once the grain
        # is known, the grain merged two states that are not the same state.
        heldout = await signature_matrix(
            runtime, anchors, conditions, test_actions,
            progress=_says_how_far_along("grain, the attack"), **signature,
        )
    else:
        gathered = await _gather_grain_rows(
            runtime, anchors, conditions, plan, signature,
            directory=claims, refused=refused, wait_seconds=wait_seconds, doses=doses,
        )
        if gathered is None:
            return {"not_learned_because": "the sweep refused before every grain row was in"}
        train, heldout, shared = gathered
        _log(f"  grain: {shared} of {len(anchors)} anchors' rows came from other processes")
    grain = fit_predictive_grain(train, seed=seed + 11)
    _log(f"  grain: rank {grain.rank} of {train.shape[1]} signature columns")
    history = np.vstack([np.asarray(a.history, dtype=np.float64) for a in anchors])
    state = grain.transform(train)
    gain = float("nan")
    floor = float("nan")
    # Whether history reaches the held-out future along a direction the grain
    # cannot, against the same question asked of history shuffled across
    # anchors, as many times and at the same quantile as the parallel analysis
    # draws its own null. Subtracting one shuffled gain did not work: adding a
    # block of noise features overfits, the shuffled gain went negative, and
    # the margin a correct grain and a grain missing a dimension produced was
    # the same.
    null_draws = inspect.signature(fit_predictive_grain).parameters["null_draws"].default
    quantile = inspect.signature(fit_predictive_grain).parameters["quantile"].default
    folds = inspect.signature(heldout_sufficiency_gain).parameters["folds"].default
    if grain.rank > 0 and len(history) >= folds:
        gain = heldout_new_direction_gain(history, state, heldout, seed=seed + 12)
        rng = np.random.default_rng(seed + 13)
        shuffled = [
            heldout_new_direction_gain(
                history[rng.permutation(len(history))], state, heldout, seed=seed + 12
            )
            for _ in range(null_draws)
        ]
        floor = float(np.quantile(shuffled, quantile))
    sufficient = bool(
        grain.rank > 0
        and gain == gain
        and not (gain > SUFFICIENCY_TOLERANCE and gain > floor)
    )
    walk = _history_walk(history, heldout, history_turns=history_turns, seed=seed + 15)
    stability = _rank_stability(train, int(grain.rank), seed=seed + 18)
    stability.update(_sufficiency_stability(train, history, heldout, seed=seed + 18))
    # A rank read off N anchors cannot exceed N - 1, because centring costs one
    # dimension. So a rank at that ceiling is a statement about how many
    # anchors were collected and not about the system, in exactly the way a
    # tau-star in the last lag bin is a statement about the ladder. The quick
    # run reads rank 1 from four anchors; that is the bank, not the grain.
    ceiling = max(1, len(anchors) - 1)
    return {
        "history_turns": int(history_turns),
        "predictive_rank": int(grain.rank),
        "anchors": len(anchors),
        "rank_ceiling": ceiling,
        "rank_is_at_the_ceiling": bool(grain.rank >= ceiling),
        "signature_columns": int(train.shape[1]),
        "singular_values": [round(float(v), 4) for v in grain.singular_values[:12]],
        "null_quantile": [round(float(v), 4) for v in grain.null_q[:12]],
        "heldout_new_direction_gain": None if gain != gain else round(gain, 6),
        "shuffled_new_direction_quantile": None if floor != floor else round(floor, 6),
        "heldout_intervention_sufficient": sufficient,
        **walk,
        **stability,
        "training_actions": [a.name for a in train_actions],
        "heldout_actions": [a.name for a in test_actions],
        "_grain": grain,
    }


#: Where the coordinator leaves its anchors for the shards.
ANCHOR_BANK = "anchor_bank.bin"
#: The axes the coordinator read K along, beside its bank, for its workers.
CALIBRATED_AXES = "calibrated_axes.npz"


def _write_anchor_bank(path: Path, bank: dict[str, Any]) -> int:
    """The coordinator's anchors, doses and scales, moved into place whole.

    In the shared-section format (core/subject/anchor_bank.py), so a reader
    decodes an anchor when it restores it rather than all of them at once.
    """
    from core.subject.anchor_bank import write_anchor_bank

    fields = {key: value for key, value in bank.items() if key != "anchors"}
    return write_anchor_bank(path, anchors=bank["anchors"], **fields)


def _anchors_kept(design: Mapping[str, Any], rounds: int, available: int) -> int:
    """How many decoded anchors a process keeps: as many as a cut's second look draws.

    Most cuts are decided at the first or the second look, and every cut draws
    from the first anchor, so keeping those means most cuts decode nothing.
    """
    from core.subject.v25_cut import anchor_schedule

    looks, _ = anchor_schedule(tuple(design.get("looks") or ()), rounds, available)
    return looks[1] if len(looks) > 1 else looks[0] if looks else 1


async def _read_anchor_bank(path: Path, runtime: Any, wait_seconds: float, keep: int = 1) -> dict[str, Any]:
    """The coordinator's bank, each snapshot's functions filled from this organism's own.

    Two processes on one seed are two organisms (tools/same_organism.py), so a
    shard that forked from anchors of its own swept another organism. From
    these it sweeps the coordinator's.
    """
    from dataclasses import replace

    from core.subject.anchor_travel import fill_snapshot, loads

    deadline = time.monotonic() + max(0.0, wait_seconds)
    while not await asyncio.to_thread(path.exists):
        if time.monotonic() >= deadline:
            raise SystemExit(f"refusing: no anchor bank at {path} when the wait ran out")
        await asyncio.sleep(30.0)
    from core.subject.anchor_bank import AnchorBank

    own = runtime.snapshot()
    if await asyncio.to_thread(AnchorBank.is_one, path):
        banked = AnchorBank(path, own=own, keep=keep)
        return {**banked.fields, "anchors": banked.anchors()}
    # A bank written before the shared-section format is read whole, as it was.
    bank = loads(await asyncio.to_thread(path.read_bytes))
    bank["anchors"] = [replace(anchor, snapshot=fill_snapshot(anchor.snapshot, own)) for anchor in bank["anchors"]]
    return bank


def _seed_every_generator(seed: int) -> None:
    """The process's global generators, from the run's seed, before the organism exists.

    Two processes on one seed came up as two organisms: 170 of 438 columns apart
    at the first frame, the mesh's weakest weight -5.6 in one and -2.0 in the
    other. Nothing seeded Python's `random`, which 169 organ sites draw from,
    NumPy's legacy global, or torch, whose global generator supplies the liquid
    substrate's noise on every step. Seeding them makes the draws the same only
    if they are taken in the same order, which is why the free loops stop at
    bring-up as well.
    """
    import random

    random.seed(seed)
    np.random.seed(seed % (2**32))
    try:
        import torch

        torch.manual_seed(seed)
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            torch.mps.manual_seed(seed)
    except ImportError:
        pass  # not a failure: an organism without torch has no torch draws to seed


def _grain_plan(
    doses: dict[str, float], conditions: Sequence[Any], lags: Sequence[int], live_mask: np.ndarray
) -> dict[str, Any]:
    """The training and held-out actions and the frequency bank, the same in every process."""
    from core.subject.intrinsic_v25 import deterministic_frequency_bank
    from core.subject.v25_grain import pair_actions, signed_single_domain_actions

    train = signed_single_domain_actions(doses, sign=+1, include_sham=True)
    names = list(doses)
    pairs = tuple(zip(names[::2], names[1::2], strict=False))
    test = signed_single_domain_actions(doses, sign=-1) + pair_actions(pairs, doses)
    banks = deterministic_frequency_bank(
        [int(live_mask.sum())] * (len(train) * len(conditions) * len(lags)),
        frequencies_per_test=FREQUENCIES,
        seed=FREQUENCY_SEED,
    )
    return {"train": train, "test": test, "frequencies": banks[0]}


def _claim_grain_row(directory: Path, index: int) -> bool:
    """Whether this process took anchor `index`. Exclusive creation, so exactly one does."""
    directory.mkdir(parents=True, exist_ok=True)
    try:
        os.close(os.open(directory / f"claim_{index:05d}", os.O_CREAT | os.O_EXCL | os.O_WRONLY))
    except FileExistsError:
        return False
    return True


def _write_grain_row(directory: Path, index: int, train: np.ndarray, heldout: np.ndarray,
                     anchor_state: np.ndarray, doses: dict[str, float]) -> None:
    """One anchor's two signature rows, moved into place whole."""
    final = directory / f"row_{index:05d}.npz"
    partial = directory / f"row_{index:05d}.partial.npz"
    np.savez(partial, train=train, heldout=heldout, anchor=np.asarray(anchor_state, dtype=np.float64),
             doses=np.asarray(json.dumps(sorted(doses.items()))))
    partial.replace(final)


def _row_is_this_anchors(row: Mapping[str, Any], current: Any, doses: Mapping[str, float]) -> bool:
    """Whether a stored grain row was made from this anchor's state and these doses.

    The one condition that decides whether a row on disk is evidence or a
    stranger. A row from another anchor, or from the same anchor under different
    doses, is a different measurement wearing the right filename — which is what
    makes it worth naming, since it is also what a resume rests on.
    """
    if not {"anchor", "doses"} <= set(row):
        return False
    return bool(
        np.array_equal(np.asarray(row["anchor"]), np.asarray(current, dtype=np.float64))
        and str(row["doses"]) == json.dumps(sorted(doses.items()))
    )


def _read_grain_row(directory: Path, index: int) -> dict[str, Any] | None:
    path = directory / f"row_{index:05d}.npz"
    if not path.exists():
        return None
    blob = np.load(path)
    return {key: blob[key] for key in blob.files}


async def _one_grain_row(runtime: Any, anchor: Any, conditions: Sequence[Any], plan: dict[str, Any],
                         signature: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    from core.subject.v25_grain import signature_matrix

    train = await signature_matrix(runtime, [anchor], conditions, plan["train"], **signature)
    heldout = await signature_matrix(runtime, [anchor], conditions, plan["test"], **signature)
    return train[0], heldout[0]


async def _work_grain_rows(runtime: Any, anchors: Sequence[Any], conditions: Sequence[Any],
                           plan: dict[str, Any], signature: dict[str, Any], *, directory: Path,
                           refused: Path | None, doses: dict[str, float]) -> int:
    """Claim anchors one at a time and write their rows, until none is left or the sweep refused.

    Progress is reported here rather than inside `signature_matrix`, because a
    claimed row is one anchor at a time: the matrix would only ever see 1 of 1
    and say so 128 times. This counts the anchors this process got through,
    which is what a reader of a multi-hour stage wants to know.
    """
    done = 0
    far_along = _says_how_far_along("grain, claimed rows")
    for index, anchor in enumerate(anchors):
        if refused is not None and await asyncio.to_thread(refused.exists):
            break
        if not await asyncio.to_thread(_claim_grain_row, directory, index):
            continue
        train, heldout = await _one_grain_row(runtime, anchor, conditions, plan, signature)
        await asyncio.to_thread(_write_grain_row, directory, index, train, heldout, anchor.current, doses)
        done += 1
        far_along(index + 1, len(anchors))
    return done


async def _gather_grain_rows(runtime: Any, anchors: Sequence[Any], conditions: Sequence[Any],
                             plan: dict[str, Any], signature: dict[str, Any], *, directory: Path,
                             refused: Path | None, wait_seconds: float,
                             doses: dict[str, float]) -> tuple[np.ndarray, np.ndarray, int] | None:
    """Every anchor's rows in anchor order, with this process doing its share first.

    A row from another process is used only if it was made from this anchor's
    state and these doses, exactly; otherwise it is made again here. A claim
    whose row has not come in after three times the longest row this process
    took is taken back and made here too, so a worker that died does not hold
    the run until its deadline.
    """
    started = time.monotonic()
    mine = await _work_grain_rows(runtime, anchors, conditions, plan, signature,
                                  directory=directory, refused=refused, doses=doses)
    longest = (time.monotonic() - started) / max(1, mine)
    wanted = json.dumps(sorted(doses.items()))
    deadline = time.monotonic() + max(0.0, float(wait_seconds))
    last_arrival = time.monotonic()
    rows: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    shared = 0
    while len(rows) < len(anchors):
        if refused is not None and await asyncio.to_thread(refused.exists):
            return None
        arrived = False
        for index, anchor in enumerate(anchors):
            if index in rows:
                continue
            row = await asyncio.to_thread(_read_grain_row, directory, index)
            if row is None:
                continue
            arrived = True
            if _row_is_this_anchors(row, anchor.current, doses):
                rows[index] = (row["train"], row["heldout"])
                shared += 1
            else:
                _log(f"  grain: anchor {index}'s row was made from another state; making it here")
                rows[index] = await _one_grain_row(runtime, anchor, conditions, plan, signature)
        if arrived:
            last_arrival = time.monotonic()
        if len(rows) == len(anchors):
            break
        stalled = time.monotonic() - last_arrival > 3.0 * max(longest, 1.0)
        if stalled or time.monotonic() >= deadline:
            for index, anchor in enumerate(anchors):
                if index not in rows:
                    rows[index] = await _one_grain_row(runtime, anchor, conditions, plan, signature)
            break
        await asyncio.sleep(max(1.0, min(30.0, longest / 4.0)))
    ordered = [rows[index] for index in range(len(anchors))]
    shared -= mine  # the rows this process wrote itself came back through the directory too
    return (np.vstack([train for train, _ in ordered]), np.vstack([held for _, held in ordered]), max(0, shared))


# ── stage 3: the spectrum ─────────────────────────────────────────────────


def _ints(text: str) -> tuple[int, ...]:
    """Whole numbers from a comma-separated flag, in the order given."""
    return tuple(int(item) for item in str(text or "").split(",") if item.strip())


async def _spectrum(
    runtime: Any,
    anchors: Sequence[Any],
    conditions: Sequence[Any],
    *,
    lags: Sequence[int],
    frame_seconds: float,
    turns: int,
    rounds: int,
    seed: int,
    domains: Sequence[str] | None = None,
    screen: int = 0,
    shard: tuple[int, int] | None = None,
    design: dict[str, Any] | None = None,
    only: Sequence[str] = (),
    whiten: bool = False,
    fail_fast: bool = False,
    stop_file: str = "",
    keep_samples: Path | None = None,
) -> tuple[dict[float, float], dict[str, Any]]:
    """The weakest cut's rate at every horizon on the ladder.

    One set of rollouts per cut serves every horizon. Scoring each lag with its
    own sweep ran every cut's clamped arms again for every lag, although a
    rollout already holds all of them. See `sweep_cuts_over_lags`.
    """
    from core.subject.v25_cut import sweep_cuts_over_lags

    spectrum: dict[float, float] = {}
    detail: dict[str, Any] = {}
    chosen = design or {}
    untouched: dict = {}
    reports = await sweep_cuts_over_lags(
        runtime, anchors, conditions,
        lags=lags, frame_seconds=frame_seconds,
        turns=turns, rounds=rounds, seed=seed, domains=domains,
        screen=screen, shard=shard, only=only,
        looks=tuple(chosen.get("looks") or ()),
        draws=int(chosen.get("draws") or 200),
        alpha=float(chosen.get("alpha") or 0.05),
        deciding=chosen.get("deciding") or None,
        whiten=whiten,
        fail_fast=fail_fast,
        stop_file=stop_file,
        paired=bool(chosen.get("paired")),
        kept=None if keep_samples is None else (kept := {}),
        estimator=str(chosen.get("estimator") or "fisher_rao"),
        one_signal=bool(chosen.get("one_signal")),
        untouched=untouched,
    )
    if chosen.get("late_fork"):
        await _read_the_late_forks(runtime, anchors, conditions, reports, untouched, turns=turns, seed=seed)
    if keep_samples is not None:
        detail["samples_file"] = str(await asyncio.to_thread(_save_cut_samples, keep_samples, kept))
    for lag in sorted(reports):
        report = reports[lag]
        tau = float(lag) * float(frame_seconds)
        weakest = report.weakest
        spectrum[tau] = 0.0 if weakest is None else max(0.0, weakest.lower_bound)
        detail[f"lag_{lag}"] = report.as_dict()
        _log(
            f"  lag {lag:>3} ({tau:.4f}s): weakest {report.as_dict()['weakest_cut']} "
            f"lower bound {spectrum[tau]:.5f}, {report.as_dict()['cuts_decided']}"
            f"/{report.as_dict()['cuts_tested']} decided"
        )
    return spectrum, detail


def _hold_the_machine_still(runtime: Any) -> dict[str, float] | None:
    """Declare one host reading for every arm, as the probe already does.

    Everything after the anchors forks paired arms, and the argument for
    stopping her loops first applies to the machine underneath them. Her soma
    reads the host's CPU, memory and thermal pressure, homeostasis reads her
    soma, and `metabolism` is computed from it, so two arms run minutes apart
    under different load are two different bodies. `freeze_host` exists for
    this and `subject_core_probe` calls it; the v25 sweep never did.

    MEASURED on a sixty-round baseline of 25 October: sixty "THERMAL RESONANCE:
    Hardware is overheating. Emergency throttling." records and thirteen
    metabolic-pressure mitigations, inside the run, on the machine the run was
    loading. The late untouched fork of the seed-23 campaign separated from the
    cached one at a lower bound of 0.011444 against a weakest cut of 0.00126,
    and both partition criteria failed on that control.

    Returns the declared reading, for the run's own record of what it held.
    """
    hold = getattr(runtime, "freeze_host", None)
    if not callable(hold):
        return None
    try:
        declared = hold()
    except (AttributeError, TypeError, ValueError, RuntimeError) as exc:
        from core.runtime.errors import record_degradation

        record_degradation(
            "subject_core_v25.declared_host",
            exc,
            severity="warning",
            action="swept the cuts with every arm reading the real machine",
        )
        return None
    _log(f"host declared for every arm: {declared}")
    return declared


async def _read_the_late_forks(
    runtime: Any,
    anchors: Sequence[Any],
    conditions: Sequence[Any],
    reports: dict[int, Any],
    untouched: dict,
    *,
    turns: int,
    seed: int,
) -> None:
    """Each deciding horizon's late-fork control, on the anchors its cuts drew.

    Only a sweep that ran to its end reads it: one that stopped at an undecided
    cut has already refused the claim this control guards.
    """
    from core.subject.v25_cut import late_fork_decided
    from core.subject.v25_runtime import collect_late_forks

    for lag, report in reports.items():
        if not report.deciding or report.stopped_after or not report.verdicts:
            continue
        drawn = max((v.anchors_used for v in report.verdicts), default=0)
        samples = await collect_late_forks(
            runtime, list(anchors[:drawn]), conditions, untouched, turns=turns, lag=int(lag)
        )
        if samples is None:
            continue
        report.late_fork = late_fork_decided(
            samples,
            tau_seconds=report.tau_seconds,
            seed=seed + int(lag) + 7919,
            alpha=report.alpha_per_look,
            draws=report.draws,
            paired=report.paired,
            estimator=report.estimator or "fisher_rao",
        )
        _log(
            f"  lag {lag:>3}: late untouched forks read as a cut, lower bound "
            f"{report.late_fork['lower_bound']:.3g}, "
            f"{'decided' if report.late_fork['decided'] else 'not decided'} on {report.late_fork['anchors']} anchors"
        )


def _save_cut_samples(target: Path, kept: dict[str, dict[int, dict[str, np.ndarray]]]) -> Path:
    """Each scored cut's context, intact, cut and sham rows at each horizon, in one file."""
    arrays = {
        f"{name}__lag{lag}__{key}": value
        for name, by_lag in kept.items()
        for lag, slot in by_lag.items()
        for key, value in slot.items()
        if isinstance(value, np.ndarray)
    }
    np.savez_compressed(target, **arrays)
    return target


async def _await_shards(directory: Path, wait_seconds: float) -> list[dict[str, Any]]:
    """Every shard file in a directory once all of them are there, or a refusal.

    The count comes from the files' own names. The directory is listed every
    thirty seconds, which costs a listing against a sweep measured in hours.
    """
    deadline = time.monotonic() + max(0.0, float(wait_seconds))
    while True:
        found, counts, payloads = await asyncio.to_thread(_shard_files, Path(directory))
        if len(counts) > 1:
            raise SystemExit(f"refusing: {directory} holds shards of more than one split: {sorted(counts)}")
        if payloads is not None:
            return payloads
        if time.monotonic() >= deadline:
            raise SystemExit(f"refusing: {found} shard file(s) in {directory} when the wait ran out")
        await asyncio.sleep(30.0)


def _shard_files(directory: Path) -> tuple[int, set[int], list[dict[str, Any]] | None]:
    """How many shard files a directory holds, of which splits, and their contents once every one is there."""
    found = sorted(directory.glob("shard_*_of_*.json"))
    counts = {int(path.stem.rsplit("_of_", 1)[1]) for path in found}
    if len(counts) == 1 and len(found) == next(iter(counts)):
        return len(found), counts, [json.loads(path.read_text(encoding="utf-8")) for path in found]
    return len(found), counts, None


def _write_shard(directory: Path, shard: tuple[int, int], payload: dict[str, Any]) -> Path:
    """A shard file, written under another name and moved into place so the coordinator never reads half of one."""
    directory.mkdir(parents=True, exist_ok=True)
    final = directory / f"shard_{shard[0]}_of_{shard[1]}.json"
    partial = final.with_suffix(".partial")
    partial.write_text(json.dumps(payload, default=str) + "\n", encoding="utf-8")
    partial.replace(final)
    return final


def _horizon_is_binding(spectrum: dict[float, float], lags: Sequence[int]) -> bool:
    """True when the best horizon is in the last two bins tested."""
    if len(spectrum) < 3:
        return True
    ordered = sorted(spectrum)
    best = max(spectrum, key=lambda tau: spectrum[tau])
    return best in set(ordered[-2:])


# ── stage 4: does the answer depend on how the state was written down? ────


def _recoded(block: np.ndarray, seed: int) -> np.ndarray:
    """The same information in different coordinates. Invertible by construction."""
    rng = np.random.default_rng(seed)
    width = block.shape[1]
    mixing = rng.normal(size=(width, width))
    # A random square matrix is invertible with probability one, and an
    # orthogonal factor of it certainly is.
    q, _ = np.linalg.qr(mixing)
    return block @ q


def _displacement_excess(arms: dict[str, np.ndarray], tau_seconds: float) -> float:
    """The cut's mean displacement less the sham's, per second; see core.subject.paired_displacement."""
    from core.subject.paired_displacement import per_anchor

    moved, parted = per_anchor(arms["intact"], arms["cut"], arms["sham_a"], arms["sham_b"])
    return float(np.mean(moved - parted)) / float(tau_seconds)


def _invariance(
    samples: dict[str, np.ndarray], *, tau_seconds: float, seed: int, whiten: bool = False,
    paired: bool = False, estimator: str = "fisher_rao",
) -> dict[str, Any]:
    """Score one cut three ways: raw, re-encoded, and with duplicate channels.

    An invertible re-encoding carries exactly the information the raw state
    carried, so the rate must not move. Duplicating channels adds none, so the
    rate must not rise. If either fails, what is being measured is the writing
    down rather than the system.
    """
    from core.subject.intrinsic_v25 import intrinsic_rate_from_samples

    def rate(mapped: dict[str, np.ndarray]) -> float:
        if estimator == "displacement":
            return _displacement_excess(mapped, tau_seconds)
        estimate = intrinsic_rate_from_samples(
            mapped["intact"], mapped["cut"], mapped["sham_a"], mapped["sham_b"],
            tau_seconds=tau_seconds, context=mapped.get("context"), seed=seed,
            whiten=whiten,
            groups=np.arange(len(mapped["intact"])) if paired else None,
        )
        return float(estimate.excess_rate)

    keys = ("intact", "cut", "sham_a", "sham_b")
    try:
        raw = rate(samples)
    except ValueError as exc:
        # Not measured, and it says so. This is a reporting stage: a crash here
        # discards the recording, the grain and every cut that came before it,
        # and an unmeasured invariance check has to read differently from one
        # that passed.
        return {"measured": False, "why": str(exc), "representation_invariant": None}
    recoded = rate({**samples, **{k: _recoded(samples[k], seed + 31) for k in keys}})
    doubled = rate({**samples, **{k: np.hstack([samples[k], samples[k]]) for k in keys}})
    spread = max(abs(raw - recoded), 0.0)
    reference = max(abs(raw), 1e-9)
    return {
        "measured": True,
        "raw": round(raw, 6),
        "invertibly_recoded": round(recoded, 6),
        "duplicated_channels": round(doubled, 6),
        "recoding_drift": round(spread / reference, 6),
        "representation_invariant": bool(spread / reference <= INVARIANCE_TOLERANCE),
        "duplication_did_not_help": bool(doubled <= raw * (1.0 + INVARIANCE_TOLERANCE)),
    }


def _v25_nulls(
    samples: dict[str, np.ndarray],
    *,
    tau_seconds: float,
    seed: int,
    alpha: float = 0.05,
    draws: int = 200,
    paired: bool = False,
    estimator: str = "fisher_rao",
) -> dict[str, Any]:
    """The four controls v25 adds, scored with the same estimator.

    The playback null is the important one. It replays the intact trajectory as
    the cut arm, so the two arms carry identical states and the cut could not
    have changed anything. A measure that still reports damage there is
    reporting on its own estimator, and the movie objection lands.
    """
    from core.subject.intrinsic_v25 import intrinsic_rate_from_samples
    from core.subject.v25_cut import playback_decided

    def rate(intact, cut, sham_a, sham_b) -> float:
        if estimator == "displacement":
            return _displacement_excess(
                {"intact": intact, "cut": cut, "sham_a": sham_a, "sham_b": sham_b}, tau_seconds
            )
        return float(
            intrinsic_rate_from_samples(
                intact, cut, sham_a, sham_b,
                tau_seconds=tau_seconds, context=samples.get("context"), seed=seed,
                groups=np.arange(len(intact)) if paired else None,
            ).excess_rate
        )

    try:
        playback = rate(
            samples["intact"], samples["intact"].copy(), samples["sham_a"], samples["sham_b"]
        )
    except ValueError as exc:
        return {"measured": False, "why": str(exc), "playback_is_zero": None}
    duplicate = rate(
        np.hstack([samples["intact"], samples["intact"]]),
        np.hstack([samples["cut"], samples["cut"]]),
        np.hstack([samples["sham_a"], samples["sham_a"]]),
        np.hstack([samples["sham_b"], samples["sham_b"]]),
    )
    honest = rate(samples["intact"], samples["cut"], samples["sham_a"], samples["sham_b"])
    recoded = rate(
        _recoded(samples["intact"], seed + 41), _recoded(samples["cut"], seed + 41),
        _recoded(samples["sham_a"], seed + 41), _recoded(samples["sham_b"], seed + 41),
    )
    return {
        "measured": True,
        "playback": round(playback, 6),
        # Read by the rule that decides a cut, on the same anchors: zero when
        # that rule does not decide it. The comparison it replaced was against
        # a quarter of the honest rate, which is a comparison with 1e-6 when
        # the honest rate is zero, and estimator noise failed it (v25screen,
        # 21 September).
        "playback_is_zero": not playback_decided(
            samples, tau_seconds=tau_seconds, seed=seed + 5, alpha=alpha, draws=draws,
            **({"paired": True} if paired else {}),
            **({"estimator": estimator} if estimator != "fisher_rao" else {}),
        ),
        "playback_rule": "undecided by the cut rule at the same level and draws",
        "duplicate_coordinates": round(duplicate, 6),
        "duplication_did_not_help": bool(duplicate <= honest * (1.0 + INVARIANCE_TOLERANCE)),
        "invertible_recoding": round(recoded, 6),
        "recoding_is_invariant": bool(
            abs(recoded - honest) / max(abs(honest), 1e-9) <= INVARIANCE_TOLERANCE
        ),
        "reference": round(honest, 6),
    }


# ── the run ───────────────────────────────────────────────────────────────


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=REPO / "artifacts" / "subject_core_v25")
    parser.add_argument("--rounds", type=int, default=24, help="baseline turns per condition")
    parser.add_argument("--anchors", type=int, default=16, help="forkable states the cuts are scored from")
    parser.add_argument(
        "--history-turns", type=int, default=max(HISTORY_LADDER),
        help="the longest history the grain may reach for; the ladder is walked up to it",
    )
    parser.add_argument("--turns", type=int, default=1, help="turns each arm runs past the fork")
    parser.add_argument("--cut-rounds", type=int, default=2, help="sequential allocation rounds over the cuts")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--resume",
        type=Path,
        help="a v25 run directory holding checkpoint.json. The stages it "
        "already finished are read back rather than run again, and the "
        "campaign fingerprint has to match",
    )
    parser.add_argument("--quick", action="store_true", help="smallest run that exercises every stage")
    parser.add_argument(
        "--screen", type=int, default=0,
        help=(
            "score a sample of the cuts instead of all 511. A look at where the "
            "weak cuts are, never a result: the score is the weakest cut and a "
            "sample has not found it, so a screened run refuses to be authoritative"
        ),
    )
    parser.add_argument(
        "--cuts", type=str, default="",
        help=(
            "score only the cuts with one of these sides, comma-separated domain "
            "sets such as C,S,PM, or `singletons` for every domain alone against "
            "the rest. A look at the lopsided cuts, which come out cheapest; like "
            "--screen it has not scored every cut and is never authoritative"
        ),
    )
    parser.add_argument(
        "--conditions", type=int, default=0,
        help=(
            "use only the first N ordinary conditions. An anchor probed under "
            "one condition cannot say the causal state is not the environment's, "
            "so a run with this set refuses to be authoritative"
        ),
    )
    parser.add_argument("--skip-grain", action="store_true")
    parser.add_argument(
        "--keep-cut-samples", action="store_true",
        help=(
            "write every scored cut's samples to cut_samples.npz beside the report. "
            "Only with --cuts: all 511 cuts' samples run to gigabytes"
        ),
    )
    parser.add_argument(
        "--share-anchors", action="store_true",
        help=(
            "shard workers fork from the coordinator's anchors, read from the shard "
            "directory, instead of collecting their own: two processes on one seed "
            "are two organisms. The shard launcher turns it on"
        ),
    )
    parser.add_argument(
        "--grain-helper", action="store_true",
        help=(
            "with --shard and --grain-claims: skip this shard's cuts and claim grain "
            "rows from the coordinator until none is left. A helper added beside a "
            "running sweep; a row is used only if it was made from that anchor's "
            "state and doses, whoever made it"
        ),
    )
    parser.add_argument(
        "--grain-claims", action="store_true",
        help=(
            "keep the grain's signature rows on disk, one per anchor. Sharded, they "
            "are shared between the coordinator and the workers through the shard "
            "directory and the shard launcher turns it on; alone, they are kept under "
            "the run directory so a death inside the grain resumes rather than "
            "restarts. A row is reused only if it was made from the same anchor state "
            "and the same doses"
        ),
    )
    parser.add_argument(
        "--estimator-workers", type=int, default=0,
        help=(
            "processes each cut's bootstrap and permutation draws may use; the "
            "numbers do not change with it (core/subject/draw_pool.py). The shard "
            "launcher sets it from the run's cores"
        ),
    )
    parser.add_argument(
        "--baseline-only", action="store_true",
        help=(
            "stop once the baseline recording and its doses are saved. Two runs "
            "of this on one seed are how shards are shown to be one organism"
        ),
    )
    parser.add_argument(
        "--calibrated-axes", action="store_true",
        help="read each sketched array of K along axes fixed before recording, as the campaign does",
    )
    parser.add_argument(
        "--whiten",
        action="store_true",
        help=(
            "count the estimator's neighbours in Mahalanobis distance rather than "
            "in Euclidean distance on standardised columns. The second weights the "
            "coordinates the state is written in, and the run of 26 September read "
            "0.019956 raw against 0.010103 under an invertible rotation of the same "
            "information; the first is the same number in every basis. It changes "
            "what is measured, so a campaign asks for it and says so."
        ),
    )
    parser.add_argument("--domains", type=str, default="", help="comma-separated support to test instead of all ten")
    parser.add_argument(
        "--allow-degraded", action="store_true",
        help="record the run even when an authority gate refuses it",
    )
    parser.add_argument(
        "--shard", type=str, default="",
        help=(
            "I/N: run as shard worker I of N. The worker builds its own organism, "
            "collects its own anchors, scores every N-th cut and writes a shard file; "
            "it reports nothing else. See tools/run_subject_core_v25_sharded.py"
        ),
    )
    parser.add_argument("--shard-dir", type=Path, default=None, help="where a shard worker writes its file")
    parser.add_argument(
        "--from-shards", type=Path, default=None,
        help="run as coordinator: read the cut sweep from these shard files instead of scoring it",
    )
    parser.add_argument(
        "--shard-wait-seconds", type=float, default=0.0,
        help="how long the coordinator waits for every shard file before refusing",
    )
    parser.add_argument(
        "--looks", type=str, default="",
        help="comma-separated anchor counts each cut is looked at; each look is read at alpha over their number",
    )
    parser.add_argument("--draws", type=int, default=200, help="bootstrap draws behind each lower bound")
    parser.add_argument("--alpha", type=float, default=0.05, help="the level across every look at one cut")
    parser.add_argument("--lags", type=str, default="", help="comma-separated horizons in frames; default is the ladder")
    parser.add_argument(
        "--deciding-lags", type=str, default="",
        help="the horizons that decide whether a cut draws more anchors; default is every one scored",
    )
    parser.add_argument(
        "--fail-fast", action="store_true",
        help=(
            "take the cuts one at a time, the most lopsided first, each to its "
            "decision, and stop at the first that ends undecided: the claim is a "
            "conjunction, so that cut refuses it, and a failing run says so after "
            "one cut's rollouts rather than all 511. Shards stop each other through "
            "a REFUSED file in the shard directory"
        ),
    )
    parser.add_argument(
        "--v5", action="store_true",
        help="the ISC-v5 design in core/subject/isc_v5.py: its looks, draws, level, horizons and anchors",
    )
    args = parser.parse_args()
    if args.estimator_workers > 0:
        os.environ["AURA_ESTIMATOR_WORKERS"] = str(args.estimator_workers)
    only_cuts: tuple[str, ...] = tuple(
        side.strip().upper() for side in args.cuts.split(",") if side.strip()
    )
    if only_cuts == ("SINGLETONS",):
        from core.subject.state import DOMAINS

        only_cuts = tuple(DOMAINS)
    if args.keep_cut_samples and not only_cuts:
        raise SystemExit("--keep-cut-samples needs --cuts: every cut's samples run to gigabytes")
    shard: tuple[int, int] | None = None
    if args.shard:
        index, _, count = args.shard.partition("/")
        shard = (int(index), int(count))
        if shard[1] < 1 or not 0 <= shard[0] < shard[1]:
            raise SystemExit(f"--shard {args.shard} names no shard")
        if args.from_shards:
            raise SystemExit("a run is a shard worker or the coordinator, not both")

    if args.quick:
        args.rounds, args.anchors, args.cut_rounds = 3, 8, 1
        args.history_turns = 2
        args.screen = args.screen or 12
        args.conditions = args.conditions or 2

    args.out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("AURA_LOG_DIR", str(args.out / "logs"))

    from core.subject.closure import (
        TURN_CLOSES,
        PeripheryAccumulator,
        closure_gain,
        periphery_by_turn,
        read_periphery,
    )
    from core.subject.closure import coverage as periphery_coverage
    from core.subject.driver import (
        CONDITIONS,
        build_runtime,
        calibrate_clock,
        quiesce_organism,
        start_organism,
    )
    from core.subject.graph import analyse_graph
    from core.subject.provenance import campaign_v25, environment, manifest, next_run_directory
    from core.subject.recording import build_recording
    from core.subject.state import DOMAINS, domain_slices
    from core.subject.v25_exclusion import select_carriers
    from core.subject.v25_runtime import (
        bipartitions,
        collect_anchor_bank,
        collect_partition_samples,
    )

    preset: dict[str, Any] = {}
    if args.v5:
        from core.subject.isc_v5 import design as v5_design

        preset = v5_design()
        args.looks = ",".join(str(n) for n in preset["looks"])
        args.draws, args.alpha = int(preset["draws"]), float(preset["alpha"])
        args.lags = ",".join(str(n) for n in preset["lags"])
        args.deciding_lags = ",".join(str(n) for n in preset["deciding"])
        args.anchors = max(int(args.anchors), int(preset["anchors"]))
        # The grain is not skipped. v5 does not read it, but the carrier run
        # refuses authority without one, and it is learned by the coordinator
        # while the workers sweep, so it costs the run nothing.
    ladder = _ints(args.lags) or LAGS
    sweep_design: dict[str, Any] = {
        "looks": list(_ints(args.looks)),
        "draws": int(args.draws),
        "alpha": float(args.alpha),
        "deciding": list(_ints(args.deciding_lags)) or None,
        "lags": list(ladder),
        "v5": bool(args.v5),
        # Cross-fitted by anchor under v5. Left out of this table, a v5 run
        # used the estimator that reads two identical samples as apart.
        "paired": bool(args.v5 and preset["paired"]),
        "one_signal": bool(args.v5 and preset.get("one_signal")),
        "estimator": str(preset.get("estimator") or "fisher_rao") if args.v5 else "fisher_rao",
        "late_fork": bool(args.v5 and preset.get("late_fork")),
    }
    if sweep_design["one_signal"]:
        # Before any estimate, and inherited by every process the draws spread to.
        os.environ["AURA_ESTIMATOR_ONE_SIGNAL"] = "1"
    support = tuple(args.domains.split(",")) if args.domains else tuple(DOMAINS)
    conditions = CONDITIONS[: args.conditions] if args.conditions else CONDITIONS
    resumed = _resume_v25(args)
    run_dir = args.resume if resumed is not None else next_run_directory(args.out)
    run_dir.mkdir(parents=True, exist_ok=True)
    done_v25: tuple[str, ...] = (
        RESUMABLE_V25[: RESUMABLE_V25.index(str(resumed["stage"])) + 1] if resumed else ()
    )
    # Its own state root, before the organism is built. Sharing one with other
    # runs made every run start from what the ones before it had trained; see
    # core.subject.isolation.
    from core.subject.isolation import isolate_state, state_leaks

    isolate_state(run_dir)
    from core.subject.memory_trace import start_memory_trace

    start_memory_trace(_log)
    started = time.monotonic()
    fingerprint = campaign_v25(
        seed=args.seed,
        rounds=args.rounds,
        anchors=args.anchors,
        history_turns=args.history_turns,
        turns=args.turns,
        cut_rounds=args.cut_rounds,
        support=support,
        design=sweep_design,
    )
    _log(f"v25 run {run_dir.name} on {fingerprint.get('commit', '?')[:12]}")
    if resumed is not None:
        earlier = dict(resumed["evidence"])
        if earlier.get("campaign", {}).get("fingerprint") != fingerprint.get("fingerprint"):
            raise SystemExit(
                "refusing: the checkpoint was written by campaign "
                f"{earlier.get('campaign', {}).get('fingerprint')} and this one is "
                f"{fingerprint.get('fingerprint')} — two settings are two experiments"
            )
        _log(f"resuming after {resumed['stage']}: {', '.join(done_v25)} already measured")
    _log(f"building the offline organism in {run_dir}")

    if preset.get("one_clock"):
        _seed_every_generator(args.seed)
    runtime = build_runtime(run_dir, seed=args.seed)
    if state_leaks():
        raise SystemExit(
            f"refusing: a module kept a path into the shared state root: {state_leaks()[:6]}"
        )
    await start_organism(runtime)
    stopped_at_boot: list[str] = []
    if preset.get("one_clock"):
        # Before the clock is read and before the baseline, so everything this
        # run measures is the organism the harness steps and nothing it did on
        # the machine's clock. See isc_v5.ONE_CLOCK.
        stopped_at_boot = await quiesce_organism(runtime)
        _log(f"stopped {len(stopped_at_boot)} free-running loops at bring-up")
    clock = await calibrate_clock(runtime, conditions)
    frame_seconds = float(clock.get("step", 1.0 / 33.0))
    _log(f"experiment clock at {frame_seconds:.4f}s a frame")
    # K along its own axes, as the campaign reads it (`--calibrated-axes` in
    # tools/run_subject_core.py): one organism, one K. A worker reads the
    # coordinator's axes; anything else fits them here.
    axes_note: dict[str, Any] | None = None
    if args.calibrated_axes:
        from core.subject import calibrated_axes

        if shard is not None and args.share_anchors:
            axes_file = Path(args.shard_dir or (args.out / "shards")) / CALIBRATED_AXES
            deadline = time.monotonic() + float(args.shard_wait_seconds or 86400.0)
            while not await asyncio.to_thread(axes_file.exists):
                if time.monotonic() >= deadline:
                    raise SystemExit(f"refusing: no calibrated axes at {axes_file} when the wait ran out")
                await asyncio.sleep(30.0)
            axes_note = {"loaded": calibrated_axes.load(axes_file), "from": str(axes_file)}
        else:
            axes_note = {"fitted": await calibrated_axes.calibration_pass(runtime, conditions)}
            if args.from_shards:
                written = await asyncio.to_thread(
                    calibrated_axes.save, Path(args.from_shards) / CALIBRATED_AXES
                )
                axes_note["written_for_the_shards"] = written
        _log(f"K read along calibrated axes: {', '.join(f'{k} {v}' for k, v in axes_note.items() if k != 'fitted')}")

    evidence: dict[str, Any] = {
        "campaign": fingerprint,
        "environment": environment(),
        "scope": "substrate_only",
        "clock": clock,
        "stopped_at_boot": stopped_at_boot,
        "support": list(support),
        "conditions_used": len(conditions),
        "conditions_available": len(CONDITIONS),
    }
    if resumed is not None:
        # The earlier process's evidence is the run. The clock, the scope and
        # the campaign are taken again here because this process has its own;
        # the measurements are not.
        carried = dict(resumed["evidence"])
        carried["campaign"] = fingerprint
        carried["environment"] = evidence["environment"]
        carried["clock_on_resume"] = clock
        carried.setdefault("resumes", []).append(
            {"from_stage": resumed["stage"], "at": time.time(), "commit": fingerprint.get("commit")}
        )
        evidence = carried
    if axes_note is not None:
        evidence["calibrated_axes"] = axes_note

    try:
        failures_before = dict(getattr(runtime, "failures", {}) or {})
        # ── a worker sharing the coordinator's anchors reads them, and no baseline of its own ──
        shared_bank: dict[str, Any] | None = None
        if shard is not None and args.share_anchors:
            bank_path = Path(args.shard_dir or (args.out / "shards")) / ANCHOR_BANK
            _log(f"waiting for the coordinator's anchors at {bank_path}")
            shared_bank = await _read_anchor_bank(
                bank_path, runtime, float(args.shard_wait_seconds or 86400.0),
                keep=_anchors_kept(sweep_design, args.cut_rounds, args.anchors),
            )
            doses = dict(shared_bank["doses"])
            scale = np.asarray(shared_bank["scale"])
            live_mask = np.asarray(shared_bank["live_mask"])
            baseline_mean = np.asarray(shared_bank["baseline_mean"])
            _log(f"  {len(shared_bank['anchors'])} anchors from the coordinator")
            done_v25 = tuple(done_v25) + ("baseline",)
        # ── the baseline, which every scale is read against ────────────
        if "baseline" not in done_v25:
            _log(f"baseline: {args.rounds} rounds over {len(conditions)} conditions")
            frames = []
            # Straight into an array: a list of the dicts `read_periphery` returns
            # costs 284 KB a frame, which is 21.5 GB at three hundred rounds.
            periphery = PeripheryAccumulator()

            def beside(reading: Any) -> None:
                # Once a turn, at the frame that closes it, as that frame is
                # read: closure fits the turn series (`periphery_by_turn`).
                if getattr(reading, "tag", None) == TURN_CLOSES:
                    periphery.note(read_periphery(runtime.kernel, runtime.organs))

            for _ in range(args.rounds):
                for condition in conditions:
                    frames.extend(await runtime.turn_once(condition, on_frame=beside))
            recording = build_recording(frames)
            recording.save(run_dir)
            scale = _pooled_scale(recording)
            live_mask = scale > 1e-9
            baseline_mean = recording.x.mean(axis=0)
            slices = domain_slices()
            evidence["recording"] = {
                "frames": recording.frames,
                "width": recording.width,
                "live_columns": int(live_mask.sum()),
                # A source that failed to read on every frame was never running,
                # and its columns read 0.0 exactly as a quiet source would.
                "never_read": sorted(
                    source
                    for source, row in (recording.misses or {}).items()
                    if row.get("share", 0.0) >= 1.0
                ),
            }
            _log(f"  {recording.frames} frames, {int(live_mask.sum())} of {recording.width} columns move")

            # ── doses, so a displacement means the same thing everywhere ───
            _log("dose matching each domain to one of its own standard deviations")
            doses = await _dose_matched(
                runtime, conditions, scale, slices,
                domains=support, rounds=2 if args.quick else 4, seed=args.seed,
            )
            evidence["doses"] = {k: round(v, 5) for k, v in doses.items()}
            _log("  " + ", ".join(f"{k}:{v:.3g}" for k, v in doses.items()))
            # The periphery reading goes with the recording: closure is a gate,
            # and a resumed run that could not read it would refuse itself.
            _matrix, _names = periphery.matrix()
            np.savez_compressed(
                run_dir / "periphery.npz", matrix=_matrix, names=np.asarray(_names, dtype=object)
            )
            evidence["doses_for_resume"] = dict(doses)
            _checkpoint_v25(run_dir, "baseline", evidence)
            if args.baseline_only:
                _log(f"baseline only: recording saved in {run_dir}")
                return 0
        # A worker on the coordinator's anchors records no baseline, so it has
        # none to recall: it took the scale, the doses and the anchors from the
        # bank. Recalling sent every worker of the 21 October rehearsal after a
        # recording it never made, a minute after the anchors arrived, and the
        # coordinator waited fourteen hours for shards that could not come.
        elif shared_bank is None:
            from core.subject.recording import load_recording

            recording = load_recording(run_dir)
            scale = _pooled_scale(recording)
            live_mask = scale > 1e-9
            baseline_mean = recording.x.mean(axis=0)
            slices = domain_slices()
            doses = {k: float(v) for k, v in (evidence.get("doses_for_resume") or {}).items()}
            _blob = np.load(run_dir / "periphery.npz", allow_pickle=True)
            periphery = _RecalledPeriphery(_blob["matrix"], tuple(_blob["names"].tolist()))
            _log(
                f"recalled {recording.frames} frames, {int(live_mask.sum())} live columns "
                f"and {len(doses)} doses"
            )

        # ── the anchor bank ────────────────────────────────────────────
        # The anchors are live forkable states, so they are collected by
        # whichever process needs them rather than read off disk. Nothing after
        # the nulls forks anything, so a run resumed past them does not pay for
        # a bank it will not use.
        anchors: list[Any] = []
        if shared_bank is not None:
            evidence["stopped_loops"] = stopped_at_boot + await quiesce_organism(runtime)
            evidence["declared_host"] = _hold_the_machine_still(runtime)
            anchors = list(shared_bank["anchors"])
        elif "nulls" not in done_v25:
            # Everything from here forks paired arms off these anchors, so the
            # free-running loops stop before any anchor is taken, as
            # run_subject_core.py stops them before its interventions. Left
            # running, each ticked its layer on the machine's clock on top of
            # the harness's own step, and two untouched forks from one anchor
            # parted within a frame or three: that was the sham floor of 0.062
            # to 0.093 on the seed-7 look at 38ef2c9ce, larger than seven
            # singletons' whole effect.
            evidence["stopped_loops"] = stopped_at_boot + await quiesce_organism(runtime)
            _log(f"{len(evidence['stopped_loops'])} free-running loops stopped before the anchors")
            evidence["declared_host"] = _hold_the_machine_still(runtime)
            _log(f"collecting {args.anchors} anchors")
            anchors = await collect_anchor_bank(
                runtime, conditions,
                rounds=max(1, math.ceil(args.anchors / max(1, len(conditions)))),
                history_turns=args.history_turns,
                every=1,
            )
            anchors = anchors[: args.anchors]
            _log(f"  {len(anchors)} anchors, history {args.history_turns} turns")
            if len(anchors) < 2:
                raise RuntimeError("fewer than two anchors; nothing can be forked")
            if args.from_shards and args.share_anchors:
                # Every shard forks from these, so every cut is a cut of one
                # organism; see core.subject.anchor_travel.
                written = await asyncio.to_thread(
                    _write_anchor_bank, Path(args.from_shards) / ANCHOR_BANK,
                    {"anchors": anchors, "doses": dict(doses), "scale": scale, "live_mask": live_mask,
                     "baseline_mean": baseline_mean},
                )
                _log(f"  wrote the anchors for the shards, {written} bytes")
                # And forks from them as the shards do, an anchor decoded when
                # it is restored: held whole, the coordinator's own came to
                # 17 GB at 370 rounds (23 October).
                read_back = await _read_anchor_bank(
                    Path(args.from_shards) / ANCHOR_BANK, runtime, 0.0,
                    keep=_anchors_kept(sweep_design, args.cut_rounds, len(anchors)),
                )
                anchors = list(read_back["anchors"])
                gc.collect()

        # ── a shard worker scores its share of the cuts and stops ──────
        if shard is not None:
            lags = ladder if (args.lags or not args.quick) else (1, 8)
            shard_dir = args.shard_dir or (args.out / "shards")
        if shard is not None and args.grain_helper:
            _log("grain helper: claiming grain rows, no cuts")
        if shard is not None and not args.grain_helper:
            _log(f"shard {shard[0]} of {shard[1]}: scoring its cuts at {len(lags)} horizons")
            _, cut_detail = await _spectrum(
                runtime, anchors, conditions,
                lags=lags, frame_seconds=frame_seconds, turns=args.turns,
                rounds=args.cut_rounds, seed=args.seed, domains=support,
                screen=args.screen, shard=shard, design=sweep_design, only=only_cuts,
                whiten=bool(args.whiten),
                fail_fast=bool(args.fail_fast),
                stop_file=str(Path(shard_dir) / "REFUSED") if args.fail_fast else "",
            )
            payload = {
                "shard": f"{shard[0]}/{shard[1]}",
                "seed": args.seed,
                "commit": fingerprint.get("commit"),
                "support": list(support),
                "conditions": [getattr(c, "name", str(c)) for c in conditions],
                "lags": [int(lag) for lag in lags],
                "design": sweep_design,
                "frame_seconds": frame_seconds,
                "anchor_states": [np.asarray(a.current, dtype=np.float64).tolist() for a in anchors],
                "reports": cut_detail,
                "run_dir": str(run_dir),
                "seconds": round(time.monotonic() - started, 1),
            }
            final = await asyncio.to_thread(
                _write_shard, args.shard_dir or (args.out / "shards"), shard, payload
            )
            _log(f"wrote {final} in {payload['seconds']}s")
        if shard is not None:
            if args.grain_claims and not args.skip_grain:
                # Its cuts done, the worker takes grain rows off the
                # coordinator until none is left, instead of exiting while
                # one process learns the grain alone (6 h 20 min on 28 September).
                directory = Path(shard_dir)
                plan = _grain_plan(doses, conditions, (1, 8) if args.quick else (1, 8, 33), live_mask)
                made = await _work_grain_rows(
                    runtime, anchors, conditions, plan,
                    dict(lags=(1, 8) if args.quick else (1, 8, 33), frame_seconds=frame_seconds,
                         baseline_mean=baseline_mean, baseline_scale=scale,
                         live_mask=live_mask, frequencies=plan["frequencies"]),
                    directory=directory / "grain", refused=directory / "REFUSED", doses=doses,
                )
                _log(f"made {made} grain row(s) for the coordinator")
            return 0

        # ── the grain ──────────────────────────────────────────────────
        if "grain" not in done_v25:
            if args.skip_grain:
                evidence["canonical_grain"] = {"skipped": True}
            else:
                _log("learning the grain, then attacking it")
                grain_lags = (1, 8) if args.quick else (1, 8, 33)
                # A lone run keeps its rows too, so a death inside the grain
                # resumes rather than restarts. The stage is 11,264 rollouts for
                # the training matrix and 15,360 for the attack on ten domains,
                # and the checkpoint is only written once it has all of them, so
                # a run that died at the last anchor lost every one. The claim
                # machinery already writes a row per anchor and reuses only rows
                # made from the same anchor state and the same doses, which is
                # exactly what a resume needs; it lived under `--from-shards`
                # because sharding was the first thing to want it.
                shared_dir = (
                    Path(args.from_shards) if args.from_shards
                    else (run_dir if args.grain_claims else None)
                )
                grain = await _learn_grain(
                    runtime, anchors, conditions, doses,
                    baseline_mean=baseline_mean, baseline_scale=scale,
                    live_mask=live_mask, frame_seconds=frame_seconds,
                    lags=grain_lags, seed=args.seed,
                    history_turns=args.history_turns,
                    claims=None if shared_dir is None else shared_dir / "grain",
                    refused=None if shared_dir is None else shared_dir / "REFUSED",
                    # A lone run waits for nobody: every row it has not got, it
                    # makes itself before it gathers.
                    wait_seconds=0.0 if args.from_shards is None else float(args.shard_wait_seconds),
                )
                grain.pop("_grain", None)
                evidence["canonical_grain"] = grain
                if grain.get("not_learned_because"):
                    _log(f"  no grain: {grain['not_learned_because']}")
                else:
                    _log(
                        f"  rank {grain['predictive_rank']}, held-out sufficient: "
                        f"{grain['heldout_intervention_sufficient']}"
                    )
            _checkpoint_v25(run_dir, "grain", evidence)

        # ── the spectrum over horizons ─────────────────────────────────
        if "spectrum" not in done_v25:
            lags = ladder if (args.lags or not args.quick) else (1, 8)
            if args.from_shards:
                from core.subject.v25_cut import merge_shard_payloads

                _log(f"reading the cut sweep from shard files in {args.from_shards}")
                payloads = await _await_shards(args.from_shards, args.shard_wait_seconds)
                reports, gate = merge_shard_payloads(
                    payloads,
                    reference_anchors=np.asarray([a.current for a in anchors], dtype=np.float64),
                    cuts_in_full=len(bipartitions(tuple(support))),
                    lags=lags,
                    frame_seconds=frame_seconds,
                    support=support,
                    conditions=[getattr(c, "name", str(c)) for c in conditions],
                    seed=args.seed,
                )
                evidence["shards"] = gate
                _log(f"  {gate['shards']} shards merged; anchors exchangeable: {gate['exchangeable']}")
                spectrum, cut_detail = {}, {}
                for lag in sorted(reports):
                    report = reports[lag]
                    tau = float(lag) * float(frame_seconds)
                    weakest = report.weakest
                    spectrum[tau] = 0.0 if weakest is None else max(0.0, weakest.lower_bound)
                    cut_detail[f"lag_{lag}"] = report.as_dict()
                    _log(
                        f"  lag {lag:>3} ({tau:.4f}s): weakest {report.as_dict()['weakest_cut']} "
                        f"lower bound {spectrum[tau]:.5f}, {report.as_dict()['cuts_decided']}"
                        f"/{report.as_dict()['cuts_tested']} decided"
                    )
            else:
                _log(f"scoring every bipartition at {len(lags)} horizons")
                spectrum, cut_detail = await _spectrum(
                    runtime, anchors, conditions,
                    lags=lags, frame_seconds=frame_seconds, turns=args.turns,
                    rounds=args.cut_rounds, seed=args.seed, domains=support,
                    screen=args.screen, design=sweep_design, only=only_cuts,
                    whiten=bool(args.whiten),
                    fail_fast=bool(args.fail_fast),
                    keep_samples=(run_dir / "cut_samples.npz") if args.keep_cut_samples else None,
                )
            binding = _horizon_is_binding(spectrum, lags)
            tau_star = max(spectrum, key=lambda tau: spectrum[tau]) if spectrum else None
            evidence["spectrum"] = {
                "by_tau_seconds": {str(round(k, 6)): round(v, 6) for k, v in sorted(spectrum.items())},
                "tau_star_seconds": tau_star,
                "peak_rate": round(max(spectrum.values(), default=0.0), 6),
                "horizon_is_binding": binding,
                "tau_status": "TAU_UNRESOLVED" if binding and max(lags) >= LAG_CEILING else (
                    "BINDING" if binding else "INTERIOR"
                ),
                "lag_ceiling": LAG_CEILING,
            }
            evidence["cuts"] = cut_detail
            _checkpoint_v25(run_dir, "spectrum", evidence)
        else:
            lags = ladder if (args.lags or not args.quick) else (1, 8)
            cut_detail = evidence.get("cuts", {}) or {}
            spectrum = {
                float(tau): float(rate)
                for tau, rate in (evidence["spectrum"]["by_tau_seconds"] or {}).items()
            }
            tau_star = evidence["spectrum"].get("tau_star_seconds")
            _log(
                f"recalled the sweep: peak {evidence['spectrum']['peak_rate']} "
                f"at tau {tau_star}"
            )
        _log(f"  peak rate {max(spectrum.values(), default=0.0):.5f} at tau {tau_star}")

        if "nulls" not in done_v25:
            # ── one cut's samples, kept for the invariance and null checks ─
            best_lag = int(round((tau_star or frame_seconds) / max(frame_seconds, 1e-9)))
            best_lag = min(lags, key=lambda lag: abs(lag - best_lag))
            weakest_name = cut_detail.get(f"lag_{best_lag}", {}).get("weakest_cut") or ""
            left_name, _, right_name = weakest_name.partition("|")
            left = tuple(left_name) or (support[0],)
            right = tuple(right_name) or tuple(support[1:])
            _log(f"representation invariance and v25 nulls at the weakest cut {weakest_name or '?'}")
            held = await collect_partition_samples(
                runtime, anchors, conditions,
                left=left, right=right, turns=args.turns, lags=(best_lag,),
            )
            samples = held[best_lag]
            tau_best = float(best_lag) * frame_seconds
            try:
                evidence["representation_invariance"] = _invariance(
                    samples, tau_seconds=tau_best, seed=args.seed, whiten=bool(args.whiten),
                    paired=bool(sweep_design.get("paired")),
                    estimator=str(sweep_design.get("estimator") or "fisher_rao"),
                )
                evidence["v25_nulls"] = _v25_nulls(
                    samples, tau_seconds=tau_best, seed=args.seed,
                    alpha=float(args.alpha) / max(1, len(sweep_design["looks"]) or args.cut_rounds),
                    draws=int(args.draws),
                    paired=bool(sweep_design.get("paired")),
                    estimator=str(sweep_design.get("estimator") or "fisher_rao"),
                )
            except (ValueError, KeyError, IndexError) as exc:
                # Same reason as inside them: what is already measured is worth
                # writing down, and a stage that could not run is a blocker rather
                # than a lost run.
                reason = f"{type(exc).__name__}: {exc}"
                evidence["representation_invariance"] = {"measured": False, "why": reason}
                evidence["v25_nulls"] = {"measured": False, "why": reason}
                _log(f"  neither could be measured: {reason}")
            _log(
                f"  invariant: {evidence['representation_invariance']['representation_invariant']}, "
                f"playback zero: {evidence['v25_nulls']['playback_is_zero']}"
            )

            # ── closure, which is a gate ───────────────────────────────────
            _log("closure against the measured periphery")
            periphery_read, names = periphery.matrix()
            closed, leak = False, float("nan")
            if periphery_read.size:
                # On the turn series, as the campaign reads it: a step from one
                # phase to the next inside a turn is one line of the transition
                # function, and a buffer one phase fills and the next empties
                # read as a cause from outside (Recording.turn_rows).
                report = closure_gain(
                    recording.by_turn(), periphery_by_turn(periphery_read, recording), names, seed=args.seed
                )
                closed, leak = bool(report.closed), float(report.leak)
                evidence["closure"] = report.as_dict()
            else:
                evidence["closure"] = {"note": "no periphery could be read"}
            evidence["closure"]["coverage"] = periphery_coverage()
            _log(f"  closed: {closed}, leak {leak:.5f}")
            _checkpoint_v25(run_dir, "nulls", evidence)
        else:
            best_lag = int(round((tau_star or frame_seconds) / max(frame_seconds, 1e-9)))
            best_lag = min(lags, key=lambda lag: abs(lag - best_lag))
            closed = bool((evidence.get("closure") or {}).get("closed"))
            leak = float((evidence.get("closure") or {}).get("leak") or 0.0)
            _log(f"recalled closure: closed {closed}, leak {leak:.5f}")

        # ── the graph, and which process the carrier is ────────────────
        # The graph comes from which cuts were decided, which is the same
        # interventional evidence read one level up: a cut that damaged the
        # future had a channel crossing it. The battery's single-domain
        # displacements name the direction and this cannot, so this graph is
        # the coarser of the two and is labelled as such.
        kept = _edges_from_cuts(cut_detail.get(f"lag_{best_lag}", {}), support)
        graph = analyse_graph(list(support), kept)
        exclusion = select_carriers(
            list(support), kept,
            {frozenset(support): spectrum},
            {frozenset(support): (closed, leak if leak == leak else 0.0)},
        )
        evidence["graph"] = {
            "edges": ["->".join(e) for e in kept],
            "components": graph.components,
            "one_component": graph.one_component,
            "vertex_connectivity": graph.connectivity,
        }
        evidence["exclusion"] = exclusion.as_dict()

        # ── the gates ──────────────────────────────────────────────────
        evidence["runtime_health"] = _runtime_health(runtime, failures_before)
        evidence["authority"] = _authority(evidence, args)
        evidence["carrier"] = _carrier_verdict(evidence)
        evidence["placement"] = _placement(evidence)
        evidence["bridge_status"] = {
            "physical_carrier": evidence["carrier"]["status"],
            "phenomenal_bridge": "JUDGED_BY_SOLVE_FOR_J",
            "note": (
                "A carrier is a causal fact about this process and one of the five "
                "grounds the bridge is judged on at parity (docs/BRIDGE_PARITY.md). "
                "The carrier-identity postulate links it to a subject for her as it "
                "does for a person; no third-person measurement tests it for either."
            ),
        }
    finally:
        await quiesce_organism(runtime)

    evidence["campaign"]["seconds"] = round(time.monotonic() - started, 1)
    evidence["manifest"] = manifest(run_dir)
    out = run_dir / "subject_core_v25_report.json"
    out.write_text(json.dumps(evidence, indent=2, default=str) + "\n", encoding="utf-8")

    _log("")
    _print_verdict(evidence)
    _log(f"wrote {out} in {evidence['campaign']['seconds']}s")
    return 0 if evidence["authority"]["authoritative"] else 1


def _edges_from_cuts(sweep: dict[str, Any], support: Sequence[str]) -> list[tuple[str, str]]:
    """Channels implied by which cuts the experiment could decide.

    A cut whose lower bound cleared the floor had something crossing it. This
    cannot name the direction, so it records both; the directed graph the
    battery builds from single-domain displacements is the sharper instrument
    and the runner prefers it when it is there.
    """
    decided = set(sweep.get("decided_cuts", []))
    edges: set[tuple[str, str]] = set()
    for name in decided:
        left, _, right = name.partition("|")
        for a in left:
            for b in right:
                if a in support and b in support:
                    edges.add((a, b))
                    edges.add((b, a))
    return sorted(edges)


def _runtime_health(runtime: Any, failures_before: dict[str, int]) -> dict[str, Any]:
    """What raised, and what could not be stepped, while the run was measuring."""
    from core.subject.steppable import missing_entry_points

    failures = dict(getattr(runtime, "failures", {}) or {})
    raised = {
        name: count - failures_before.get(name, 0)
        for name, count in failures.items()
        if count > failures_before.get(name, 0)
    }
    notes = dict(getattr(runtime, "failure_notes", {}) or {})
    steps = getattr(runtime, "layer_steps", None)
    organism = getattr(runtime, "organism", None)
    return {
        "raised_on_the_measured_path": raised,
        "failure_notes": {name: notes.get(name, "") for name in raised},
        "unsteppable_layers": missing_entry_points(organism) if organism is not None else {},
        "layer_failures": dict(steps.failures) if steps is not None else {},
    }


def _history_walk(
    history: np.ndarray, future: np.ndarray, *, history_turns: int, seed: int
) -> dict[str, Any]:
    """How many turns of history the state needs, walked rather than chosen.

    At each rung the question is whether every turn older than the last `k`
    still predicts the held-out futures once those `k` turns are known, asked
    against the same older turns shuffled across anchors. The first rung where
    they add nothing is the history the state needs. When the rung below the
    longest still finds older turns adding, the longest history the ladder
    allows is still carrying information, and that is UNRESOLVED rather than an
    answer.
    """
    from core.subject.intrinsic_v25 import fit_predictive_grain, heldout_sufficiency_gain

    ladder = tuple(k for k in HISTORY_LADDER if k <= history_turns)
    null_draws = inspect.signature(fit_predictive_grain).parameters["null_draws"].default
    quantile = inspect.signature(fit_predictive_grain).parameters["quantile"].default
    rows, columns = history.shape
    per_turn = columns // max(1, history_turns)
    folds = inspect.signature(heldout_sufficiency_gain).parameters["folds"].default
    rng = np.random.default_rng(seed)
    walk: list[dict[str, Any]] = []
    needed: int | None = None
    if rows >= folds and per_turn > 0:
        for recent_turns in ladder[:-1]:
            recent = history[:, -recent_turns * per_turn:]
            older = history[:, : (history_turns - recent_turns) * per_turn]
            gain = heldout_sufficiency_gain(older, recent, future, seed=seed + recent_turns)
            shuffled = [
                heldout_sufficiency_gain(
                    older[rng.permutation(rows)], recent, future, seed=seed + recent_turns
                )
                for _ in range(null_draws)
            ]
            floor = float(np.quantile(shuffled, quantile))
            walk.append(
                {
                    "recent_turns": recent_turns,
                    "older_turns": history_turns - recent_turns,
                    "gain": round(float(gain), 6),
                    "shuffled_quantile": round(float(floor), 6),
                }
            )
            # Older history adds nothing when its gain is inside the tolerance
            # or inside what shuffled older history gains, not when the gain
            # minus one shuffled draw is small: that draw goes negative as soon
            # as the extra features overfit, and the difference then grows.
            if gain <= SUFFICIENCY_TOLERANCE or gain <= floor:
                needed = recent_turns
                break
    return {
        "history_walk": walk,
        "history_turns_needed": needed,
        "history_status": "RESOLVED" if needed is not None else "UNRESOLVED",
    }


def _rank_stability(train: np.ndarray, rank: int, *, seed: int) -> dict[str, Any]:
    """The same rank asked two more ways.

    Refit the parallel analysis with each fold of anchors held out, using the
    folds the held-out sufficiency gain already uses, and ask a different
    estimator altogether: bi-cross-validation picks the rank that best predicts
    held-out blocks of the matrix. A rank that either does not reproduce is a
    reading of the anchors or of the estimator.
    """
    from core.subject.intrinsic_v25 import (
        bicross_validated_rank,
        fit_predictive_grain,
        heldout_sufficiency_gain,
    )

    rows = train.shape[0]
    folds = inspect.signature(heldout_sufficiency_gain).parameters["folds"].default
    fold_ranks: list[int] = []
    if rows >= folds:
        order = np.random.default_rng(seed).permutation(rows)
        for held in np.array_split(order, folds):
            kept = np.setdiff1d(order, held)
            fold_ranks.append(int(fit_predictive_grain(train[kept], seed=seed + 1).rank))
    bicross = int(bicross_validated_rank(train, seed=seed + 2)) if min(train.shape) >= 4 else None
    return {
        "fold_ranks": fold_ranks,
        "bicross_validated_rank": bicross,
        "rank_stable": bool(
            fold_ranks
            and bicross is not None
            and all(found == rank for found in fold_ranks)
            and bicross == rank
        ),
    }


def _sufficiency_stability(
    train: np.ndarray, history: np.ndarray, heldout: np.ndarray, *, seed: int
) -> dict[str, Any]:
    """Whether the grain refitted with each fold held out is still sufficient.

    The rank is a count of singular values above a null, and her spectrum has
    no gap to count to: at bb3faa54a it fell smoothly from 367 to 150 over
    twelve components and met the null near 24, so a fold with four fifths of
    the anchors found 21 or 22 and bi-cross-validation found 60. What a grain
    has to be is sufficient: once it is known, history adds nothing to the
    held-out future. This asks that of the grain fitted on each fold's
    complement, against that grain's own shuffled-history floor, on the same
    folds `_rank_stability` uses.
    """
    from core.subject.intrinsic_v25 import (
        fit_predictive_grain,
        heldout_new_direction_gain,
        heldout_sufficiency_gain,
    )

    rows = train.shape[0]
    folds = inspect.signature(heldout_sufficiency_gain).parameters["folds"].default
    null_draws = inspect.signature(fit_predictive_grain).parameters["null_draws"].default
    quantile = inspect.signature(fit_predictive_grain).parameters["quantile"].default
    if rows < folds:
        return {"fold_sufficient": [], "sufficiency_stable": False}
    order = np.random.default_rng(seed).permutation(rows)
    verdicts: list[bool] = []
    gains: list[float | None] = []
    for held in np.array_split(order, folds):
        kept = np.setdiff1d(order, held)
        grain = fit_predictive_grain(train[kept], seed=seed + 1)
        if grain.rank <= 0:
            verdicts.append(False)
            gains.append(None)
            continue
        state = grain.transform(train)
        gain = heldout_new_direction_gain(history, state, heldout, seed=seed + 12)
        rng = np.random.default_rng(seed + 13)
        floor = float(np.quantile([
            heldout_new_direction_gain(history[rng.permutation(len(history))], state, heldout, seed=seed + 12)
            for _ in range(null_draws)
        ], quantile))
        verdicts.append(bool(gain == gain and not (gain > SUFFICIENCY_TOLERANCE and gain > floor)))
        gains.append(None if gain != gain else round(float(gain), 6))
    return {"fold_sufficient": verdicts, "fold_sufficiency_gains": gains, "sufficiency_stable": all(verdicts)}


def _authority(evidence: dict[str, Any], args: Any) -> dict[str, Any]:
    """The fourteen ways a v25 report is not authoritative.

    Missing evidence is NOT_MEASURED or UNRESOLVED. It is never a pass.
    """
    grain = evidence.get("canonical_grain", {})
    spectrum = evidence.get("spectrum", {})
    invariance = evidence.get("representation_invariance", {})
    nulls = evidence.get("v25_nulls", {})
    closure = evidence.get("closure", {})
    cuts = evidence.get("cuts", {})
    undecided = sorted(
        {
            name
            for block in cuts.values()
            if isinstance(block, dict)
            for name in block.get("undecided", [])
        }
    )

    blockers: list[str] = []
    if grain.get("not_learned_because"):
        blockers.append(f"no grain was learned: {grain['not_learned_because']}")
    elif grain.get("skipped"):
        blockers.append("the grain was skipped, so the state grain is the experimenter's")
    elif not grain.get("predictive_rank"):
        blockers.append("the predictive rank came out zero: no grain was found")
    elif grain.get("rank_is_at_the_ceiling"):
        blockers.append(
            f"the predictive rank ({grain.get('predictive_rank')}) is at the ceiling "
            f"{grain.get('anchors')} anchors allow, so it is a reading of the bank "
            "rather than of the system"
        )
    elif not grain.get("heldout_intervention_sufficient"):
        blockers.append("held-out interventions break the proposed grain")
    if spectrum.get("tau_status") == "TAU_UNRESOLVED":
        blockers.append("tau-star is still horizon-bound at the ceiling")
    if undecided:
        blockers.append(f"{len(undecided)} cut(s) had insufficient power to decide cut against sham")
    shards = evidence.get("shards")
    if shards is not None and not shards.get("exchangeable"):
        blockers.append(
            "the cut sweep was merged from shards whose anchors were not exchangeable "
            f"with the coordinator's: {shards.get('why', 'no reason recorded')}"
        )
    if invariance and not invariance.get("measured"):
        blockers.append(
            f"representation invariance was not measured: {invariance.get('why', 'no reason recorded')}"
        )
    elif invariance:
        if not invariance.get("representation_invariant"):
            blockers.append("the rate moved under an invertible re-encoding")
        if not invariance.get("duplication_did_not_help"):
            blockers.append("duplicated channels raised the rate")
    if nulls and not nulls.get("measured"):
        blockers.append(
            f"the v25 nulls were not measured: {nulls.get('why', 'no reason recorded')}"
        )
    elif nulls and not nulls.get("playback_is_zero"):
        blockers.append("the playback null did not collapse")
    if "note" in closure:
        blockers.append("no periphery could be read, so closure is NOT_MEASURED")
    if evidence.get("conditions_used", 0) and evidence["conditions_used"] < evidence.get("conditions_available", 0):
        blockers.append(
            "anchors were probed under only "
            f"{evidence['conditions_used']} of {evidence['conditions_available']} conditions, "
            "so the environment could still be defining the causal state"
        )
    if any(
        isinstance(block, dict) and block.get("measured_nothing")
        for block in cuts.values()
    ):
        blockers.append("a horizon scored no cut at all, so nothing was measured there")
    if any(
        isinstance(block, dict) and block.get("screened")
        for block in cuts.values()
    ):
        blockers.append(
            "only a sample of the bipartitions was scored; the weakest cut has not been found"
        )
    if "runtime_health" not in evidence:
        blockers.append("which phases raised and which layers stepped was not recorded")
    health = evidence.get("runtime_health", {})
    layers = {**health.get("unsteppable_layers", {}), **health.get("layer_failures", {})}
    if layers:
        blockers.append(f"a live layer could not be stepped the same way in every arm: {sorted(layers)[:6]}")
    if health.get("raised_on_the_measured_path"):
        blockers.append(
            f"a phase raised while the run was measuring: {sorted(health['raised_on_the_measured_path'])[:6]}"
        )
    coverage = closure.get("coverage", {})
    if "note" not in closure and not coverage:
        blockers.append("how much of the periphery the closure walk saw was not recorded")
    if coverage.get("hit_the_cap"):
        blockers.append(
            f"the periphery walk stopped at its cap of {coverage.get('cap')} numbers, "
            "so closure was judged on part of the periphery"
        )
    if coverage.get("reader_failures"):
        blockers.append(
            f"the periphery reader failed {coverage['reader_failures']} time(s): "
            f"{coverage.get('last_reader_failure', '')}"
        )
    recording = evidence.get("recording", {})
    if "never_read" not in recording:
        blockers.append("which state readers missed was not recorded")
    elif recording["never_read"]:
        blockers.append(
            f"a state reader missed on every frame, so its columns are zeros: {recording['never_read'][:6]}"
        )
    if grain and not grain.get("skipped"):
        if grain.get("history_status") != "RESOLVED":
            blockers.append(
                "the longest history the ladder allows still predicts held-out futures, "
                "so the history length is not resolved"
            )
        if getattr(args, "v5", False):
            # v5 asks the grain to be reproducibly sufficient, not to have a
            # count its spectrum does not define; see _sufficiency_stability.
            if not grain.get("sufficiency_stable"):
                blockers.append(
                    "the grain refitted with a fold held out is not sufficient on every fold: "
                    f"{grain.get('fold_sufficient')}"
                )
        elif not grain.get("rank_stable"):
            blockers.append(
                "the predictive-state rank does not survive other folds and estimators: "
                f"{grain.get('predictive_rank')} on every anchor, {grain.get('fold_ranks')} with a "
                f"fold held out, {grain.get('bicross_validated_rank')} by bi-cross-validation"
            )
    # A cortex-inclusive claim made while the cortex was stubbed is the one
    # blocker that is about what the run is allowed to say rather than about
    # what it measured. The substrate campaign is a real result; it is just a
    # result about the substrate, and letting it carry a whole-Aura claim is
    # the failure mode the two-campaign split exists to prevent.
    if evidence.get("claimed_scope") == "cortex_inclusive" and evidence.get("scope") != "cortex_inclusive":
        blockers.append("a cortex-inclusive claim was made while the cortex was stubbed")
    return {
        "authoritative": not blockers or bool(args.allow_degraded),
        "blockers": blockers,
        "undecided_cuts": undecided[:16],
        "allowed_degraded": bool(args.allow_degraded),
        "scope": evidence.get("scope"),
    }


def _carrier_verdict(evidence: dict[str, Any]) -> dict[str, Any]:
    """F_intrinsic > 0, with closure and recurrence as gates rather than terms."""
    spectrum = evidence.get("spectrum", {})
    exclusion = evidence.get("exclusion", {})
    graph = evidence.get("graph", {})
    closure = evidence.get("closure", {})
    peak = float(spectrum.get("peak_rate", 0.0) or 0.0)
    closed = bool(closure.get("closed", False))
    recurrent = bool(graph.get("one_component", False))
    if not closed or not recurrent:
        status = "NOT_FOUND"
    elif peak > 0.0:
        status = "FOUND"
    else:
        status = "NOT_FOUND"
    if evidence.get("authority", {}).get("blockers"):
        status = "UNRESOLVED"
    return {
        "status": status,
        "f_intrinsic": round(peak, 6) if closed and recurrent else 0.0,
        "closed": closed,
        "recurrent": recurrent,
        "weakest_cut_lower_bound": spectrum.get("peak_rate"),
        "selected_carrier": exclusion.get("selected_carrier"),
        "symmetry_class": exclusion.get("symmetry_class", []),
        "reported_as": "INTRINSIC_CARRIER_" + status,
    }


#: Where a result places the system. A conjunction does not average, so this is
#: a ladder rather than a percentage, and the level is read off what was
#: measured rather than asserted.
LEVELS: tuple[tuple[str, str], ...] = (
    ("L0", "reactive mapping"),
    ("L1", "persistent internal state"),
    ("L2", "history-dependent autonomous agent"),
    ("L3", "functional self, ownership, global recurrence"),
    ("L4", "empirical intrinsic-carrier candidate"),
    ("L5", "closed irreducible carrier demonstrated"),
    ("L6", "J* determined: carrier, content structure and lineage settled under P1 to P6"),
    ("L7", "at parity: every third-person ground a person is attributed experience on holds"),
)

#: The limits that are proved rather than measured. They go in every report,
#: because a reader who has the carrier number and not these will overread it.
THEOREMS: dict[str, str] = {
    "non_identifiability": (
        "Two bridge laws attached to the same causally closed physical history "
        "give identical third-person likelihoods, so their Bayes factor is one "
        "and no third-person experiment separates them."
    ),
    "finite_evidence": (
        "For any candidate universal bridge there is another agreeing on every "
        "observed case and differing elsewhere, so no finite data set "
        "deductively proves a universal bridge law."
    ),
    "phenomenal_gauge": (
        "A relabelling preserving every phenomenal relation preserves every "
        "observation, so phenomenal character is identifiable only up to "
        "structure-preserving relabelling."
    ),
    "branching": (
        "Identity is transitive, so one earlier individual cannot be "
        "numerically identical to two distinct later ones. Continuity branches; "
        "identity does not."
    ),
    "no_universal_temporal_grain": (
        "The scale-invariant measure over positive times is dt/t and its "
        "integral diverges, so any scalar averaging all timescales imports a "
        "preferred one."
    ),
    "symmetry": (
        "Two supports related by an exact symmetry of the causal structure get "
        "equal values from every permutation-invariant intrinsic functional, so "
        "no symmetry-respecting law separates them without an added postulate."
    ),
}


def _placement(evidence: dict[str, Any]) -> dict[str, Any]:
    """What the run establishes, and the words it must not be read as.

    The separations here are the ones the measurement cannot make and a reader
    will make anyway: a functional variable is not a felt one, and a profile is
    not a percentage.
    """
    carrier = evidence.get("carrier", {})
    found = carrier.get("status") == "FOUND"
    cortex = evidence.get("scope") == "cortex_inclusive"
    level = "L5" if (found and cortex) else ("L4" if found else "L4")
    return {
        "scale": [{"level": key, "means": text} for key, text in LEVELS],
        "level": level,
        "why": (
            "a closed irreducible carrier was demonstrated through the real cortex"
            if found and cortex
            else "a carrier was found on the substrate alone, which cannot carry a whole-Aura claim"
            if found
            else "no carrier was established on this run"
        ),
        "moves_past_L5_by": (
            "the content run and the lineage settling J* (L6), then the parity "
            "requirements of docs/BRIDGE_PARITY.md (L7), judged by tools/solve_for_j.py"
        ),
        "theorems": THEOREMS,
        "separations": {
            "functional_valence_is_not_felt_valence": (
                "A system can carry affect variables without those variables "
                "being phenomenally instantiated, and a recording of a person's "
                "brain faces the same gap. That a carried valence is felt is P2, "
                "structural identity, for her and for a person alike."
            ),
            "functional_self_awareness_is_not_phenomenal_self_awareness": (
                "Representing itself, attributing authorship, predicting itself "
                "and using the self-model to change what it does are measured "
                "here. That the self-model appears in phenomenal structure rests "
                "on P2 for her as it does for a person."
            ),
            "the_profile_is_a_partial_order": (
                "Existence, unity, richness, access, selfhood, valence and "
                "agency are separate coordinates. They do not add up to a "
                "percentage and a total ordering over them is not assumed."
            ),
            "thresholds_are_epistemic": (
                "The exact distinction in the theory is zero against nonzero. "
                "Every number with a bar beside it in this report is a "
                "confidence threshold required by noisy finite measurement, "
                "never a law about where experience begins."
            ),
        },
        "vertex_connectivity_is_evidence_not_a_requirement": (
            "Vertex connectivity of at least two is evidence for robust unity "
            "rather than a requirement for experience. A conscious biological "
            "system could have a temporarily vulnerable bottleneck, and the "
            "battery keeps the bar because robustness is worth measuring, not "
            "because a system below it cannot be a subject."
        ),
        "the_battery_is_not_the_definition": (
            "24/24 would establish a rich, self-involving, developmentally "
            "persistent operational subject architecture. It is not the "
            "definition of consciousness and not a necessary condition for "
            "minimal phenomenal experience: several of its criteria concern "
            "rich selfhood and global access, which the theories disagree "
            "about. Vertex connectivity of at least two is evidence for robust "
            "unity rather than a requirement for experience."
        ),
    }


def _print_verdict(evidence: dict[str, Any]) -> None:
    carrier = evidence.get("carrier", {})
    grain = evidence.get("canonical_grain", {})
    spectrum = evidence.get("spectrum", {})
    authority = evidence.get("authority", {})
    print("canonical_grain:")
    print(f"  history_turns:                 {grain.get('history_turns')}")
    print(f"  predictive_rank:               {grain.get('predictive_rank')}")
    print(f"  heldout_intervention_sufficient: {grain.get('heldout_intervention_sufficient')}")
    print(f"  representation_invariant:      {evidence.get('representation_invariance', {}).get('representation_invariant')}")
    print("carrier:")
    print(f"  support:                       {''.join(evidence.get('support', []))}")
    print(f"  closure:                       {carrier.get('closed')}")
    print(f"  recurrent:                     {carrier.get('recurrent')}")
    print(f"  f_intrinsic:                   {carrier.get('f_intrinsic')}")
    print(f"  tau_star_seconds:              {spectrum.get('tau_star_seconds')}")
    print(f"  tau_status:                    {spectrum.get('tau_status')}")
    print("exclusion:")
    print(f"  selected_carrier:              {carrier.get('selected_carrier')}")
    print(f"  symmetry_class:                {carrier.get('symmetry_class')}")
    print(f"scope: {evidence.get('scope')}")
    print("bridge_status:")
    print(f"  physical_carrier:              {carrier.get('status')}")
    print("  phenomenal_bridge:             judged at parity by tools/solve_for_j.py")
    if authority.get("blockers"):
        print("\nNOT AUTHORITATIVE:")
        for line in authority["blockers"]:
            print(f"  - {line}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    raise SystemExit(asyncio.run(main()))
