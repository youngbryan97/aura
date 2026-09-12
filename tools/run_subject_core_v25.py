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
never reports CONSCIOUS. The step from a carrier to a phenomenal subject is a
bridge postulate that no measurement here tests, and the report says so in the
same breath every time.

    python tools/run_subject_core_v25.py --quick
    python tools/run_subject_core_v25.py --anchors 24 --rounds 40
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

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
    """
    from core.subject.state import perturb, perturb_organs

    doses = {key: 0.15 for key in domains}
    for _ in range(max(1, rounds)):
        for domain in domains:
            block = slices[domain]
            unit = np.where(scale[block] > 1e-9, scale[block], 1.0)
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
                moved.append(float(np.max(np.abs(after - before) / unit)))
            reached = float(np.median(moved)) if moved else 0.0
            if reached > 1e-6:
                # Proportional, and bounded. A domain the writer cannot move at
                # all would otherwise run its dose away to infinity.
                doses[domain] = float(np.clip(doses[domain] / reached, 1e-4, 50.0))
    return doses


# ── stage 2: the grain ────────────────────────────────────────────────────


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
) -> dict[str, Any]:
    """The predictive-state rank, and whether held-out interventions break it."""
    from core.subject.intrinsic_v25 import (
        deterministic_frequency_bank,
        fit_predictive_grain,
        heldout_sufficiency_gain,
    )
    from core.subject.v25_grain import (
        pair_actions,
        signature_matrix,
        signed_single_domain_actions,
    )

    width = int(live_mask.sum())
    train_actions = signed_single_domain_actions(doses, sign=+1, include_sham=True)
    banks = deterministic_frequency_bank(
        [width] * (len(train_actions) * len(conditions) * len(lags)),
        frequencies_per_test=FREQUENCIES,
        seed=FREQUENCY_SEED,
    )
    _log(f"  grain: {len(train_actions)} training actions x {len(conditions)} conditions x {len(lags)} lags")
    train = await signature_matrix(
        runtime, anchors, conditions, train_actions,
        lags=lags, frame_seconds=frame_seconds,
        baseline_mean=baseline_mean, baseline_scale=baseline_scale,
        live_mask=live_mask, frequencies=banks[0],
    )
    grain = fit_predictive_grain(train, seed=seed + 11)
    _log(f"  grain: rank {grain.rank} of {train.shape[1]} signature columns")

    # The attack. Negative doses and preregistered pairs, none of which the
    # rank was fitted on. If raw history still predicts these once the grain is
    # known, the grain merged two states that are not the same state.
    names = list(doses)
    pairs = tuple(zip(names[::2], names[1::2], strict=False))
    test_actions = signed_single_domain_actions(doses, sign=-1) + pair_actions(pairs, doses)
    heldout = await signature_matrix(
        runtime, anchors, conditions, test_actions,
        lags=lags, frame_seconds=frame_seconds,
        baseline_mean=baseline_mean, baseline_scale=baseline_scale,
        live_mask=live_mask, frequencies=banks[0],
    )
    history = np.vstack([np.asarray(a.history, dtype=np.float64) for a in anchors])
    state = grain.transform(train)
    gain = float("nan")
    floor = float("nan")
    if grain.rank > 0 and len(history) >= 5:
        gain = heldout_sufficiency_gain(history, state, heldout, seed=seed + 12)
        rng = np.random.default_rng(seed + 13)
        floor = heldout_sufficiency_gain(
            history[rng.permutation(len(history))], state, heldout, seed=seed + 14
        )
    sufficient = bool(
        grain.rank > 0
        and gain == gain
        and (gain - (floor if floor == floor else 0.0)) <= SUFFICIENCY_TOLERANCE
    )
    # A rank read off N anchors cannot exceed N - 1, because centring costs one
    # dimension. So a rank at that ceiling is a statement about how many
    # anchors were collected and not about the system, in exactly the way a
    # tau-star in the last lag bin is a statement about the ladder. The quick
    # run reads rank 1 from four anchors; that is the bank, not the grain.
    ceiling = max(1, len(anchors) - 1)
    return {
        "history_turns": int(getattr(anchors[0], "history", np.zeros(0)).size // max(1, width)) if anchors else 0,
        "predictive_rank": int(grain.rank),
        "anchors": len(anchors),
        "rank_ceiling": ceiling,
        "rank_is_at_the_ceiling": bool(grain.rank >= ceiling),
        "signature_columns": int(train.shape[1]),
        "singular_values": [round(float(v), 4) for v in grain.singular_values[:12]],
        "null_quantile": [round(float(v), 4) for v in grain.null_q[:12]],
        "heldout_sufficiency_gain": None if gain != gain else round(gain, 6),
        "shuffled_history_floor": None if floor != floor else round(floor, 6),
        "heldout_intervention_sufficient": sufficient,
        "training_actions": [a.name for a in train_actions],
        "heldout_actions": [a.name for a in test_actions],
        "_grain": grain,
    }


# ── stage 3: the spectrum ─────────────────────────────────────────────────


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
) -> tuple[dict[float, float], dict[str, Any]]:
    """The weakest cut's rate at every horizon on the ladder."""
    from core.subject.v25_cut import sweep_cuts

    spectrum: dict[float, float] = {}
    detail: dict[str, Any] = {}
    for lag in lags:
        tau = float(lag) * float(frame_seconds)
        report = await sweep_cuts(
            runtime, anchors, conditions,
            tau_frames=int(lag), tau_seconds=tau,
            turns=turns, rounds=rounds, seed=seed + lag, domains=domains,
            screen=screen,
        )
        weakest = report.weakest
        spectrum[tau] = 0.0 if weakest is None else max(0.0, weakest.lower_bound)
        detail[f"lag_{lag}"] = report.as_dict()
        _log(
            f"  lag {lag:>3} ({tau:.4f}s): weakest {report.as_dict()['weakest_cut']} "
            f"lower bound {spectrum[tau]:.5f}, {report.as_dict()['cuts_decided']}"
            f"/{report.as_dict()['cuts_tested']} decided"
        )
    return spectrum, detail


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


def _invariance(
    samples: dict[str, np.ndarray], *, tau_seconds: float, seed: int
) -> dict[str, Any]:
    """Score one cut three ways: raw, re-encoded, and with duplicate channels.

    An invertible re-encoding carries exactly the information the raw state
    carried, so the rate must not move. Duplicating channels adds none, so the
    rate must not rise. If either fails, what is being measured is the writing
    down rather than the system.
    """
    from core.subject.intrinsic_v25 import intrinsic_rate_from_samples

    def rate(mapped: dict[str, np.ndarray]) -> float:
        estimate = intrinsic_rate_from_samples(
            mapped["intact"], mapped["cut"], mapped["sham_a"], mapped["sham_b"],
            tau_seconds=tau_seconds, context=mapped.get("context"), seed=seed,
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
    samples: dict[str, np.ndarray], *, tau_seconds: float, seed: int
) -> dict[str, Any]:
    """The four controls v25 adds, scored with the same estimator.

    The playback null is the important one. It replays the intact trajectory as
    the cut arm, so the two arms carry identical states and the cut could not
    have changed anything. A measure that still reports damage there is
    reporting on its own estimator, and the movie objection lands.
    """
    from core.subject.intrinsic_v25 import intrinsic_rate_from_samples

    def rate(intact, cut, sham_a, sham_b) -> float:
        return float(
            intrinsic_rate_from_samples(
                intact, cut, sham_a, sham_b,
                tau_seconds=tau_seconds, context=samples.get("context"), seed=seed,
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
        "playback_is_zero": bool(playback <= max(1e-6, honest * 0.25)),
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
    parser.add_argument("--history-turns", type=int, default=8)
    parser.add_argument("--turns", type=int, default=1, help="turns each arm runs past the fork")
    parser.add_argument("--cut-rounds", type=int, default=2, help="sequential allocation rounds over the cuts")
    parser.add_argument("--seed", type=int, default=7)
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
        "--conditions", type=int, default=0,
        help=(
            "use only the first N ordinary conditions. An anchor probed under "
            "one condition cannot say the causal state is not the environment's, "
            "so a run with this set refuses to be authoritative"
        ),
    )
    parser.add_argument("--skip-grain", action="store_true")
    parser.add_argument("--domains", type=str, default="", help="comma-separated support to test instead of all ten")
    parser.add_argument(
        "--allow-degraded", action="store_true",
        help="record the run even when an authority gate refuses it",
    )
    args = parser.parse_args()

    if args.quick:
        args.rounds, args.anchors, args.cut_rounds = 3, 8, 1
        args.history_turns = 2
        args.screen = args.screen or 12
        args.conditions = args.conditions or 2

    args.out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("AURA_LOG_DIR", str(args.out / "logs"))

    from core.subject.closure import closure_gain, read_periphery
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
    from core.subject.v25_runtime import collect_anchor_bank, collect_partition_samples

    support = tuple(args.domains.split(",")) if args.domains else tuple(DOMAINS)
    conditions = CONDITIONS[: args.conditions] if args.conditions else CONDITIONS
    run_dir = next_run_directory(args.out)
    run_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    fingerprint = campaign_v25(
        seed=args.seed,
        rounds=args.rounds,
        anchors=args.anchors,
        history_turns=args.history_turns,
        turns=args.turns,
        cut_rounds=args.cut_rounds,
        support=support,
    )
    _log(f"v25 run {run_dir.name} on {fingerprint.get('commit', '?')[:12]}")
    _log(f"building the offline organism in {run_dir}")

    runtime = build_runtime(run_dir, seed=args.seed)
    await start_organism(runtime)
    clock = await calibrate_clock(runtime, conditions)
    frame_seconds = float(clock.get("step", 1.0 / 33.0))
    _log(f"experiment clock at {frame_seconds:.4f}s a frame")

    evidence: dict[str, Any] = {
        "campaign": fingerprint,
        "environment": environment(),
        "scope": "substrate_only",
        "clock": clock,
        "support": list(support),
        "conditions_used": len(conditions),
        "conditions_available": len(CONDITIONS),
    }

    try:
        # ── the baseline, which every scale is read against ────────────
        _log(f"baseline: {args.rounds} rounds over {len(conditions)} conditions")
        frames = []
        for _ in range(args.rounds):
            for condition in conditions:
                frames.extend(await runtime.turn_once(condition))
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

        # ── the anchor bank ────────────────────────────────────────────
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

        # ── the grain ──────────────────────────────────────────────────
        if args.skip_grain:
            evidence["canonical_grain"] = {"skipped": True}
        else:
            _log("learning the grain, then attacking it")
            grain_lags = (1, 8) if args.quick else (1, 8, 33)
            grain = await _learn_grain(
                runtime, anchors, conditions, doses,
                baseline_mean=baseline_mean, baseline_scale=scale,
                live_mask=live_mask, frame_seconds=frame_seconds,
                lags=grain_lags, seed=args.seed,
            )
            grain.pop("_grain", None)
            evidence["canonical_grain"] = grain
            _log(
                f"  rank {grain['predictive_rank']}, held-out sufficient: "
                f"{grain['heldout_intervention_sufficient']}"
            )

        # ── the spectrum over horizons ─────────────────────────────────
        lags = (1, 8) if args.quick else LAGS
        _log(f"scoring every bipartition at {len(lags)} horizons")
        spectrum, cut_detail = await _spectrum(
            runtime, anchors, conditions,
            lags=lags, frame_seconds=frame_seconds, turns=args.turns,
            rounds=args.cut_rounds, seed=args.seed, domains=support,
            screen=args.screen,
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
        _log(f"  peak rate {max(spectrum.values(), default=0.0):.5f} at tau {tau_star}")

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
                samples, tau_seconds=tau_best, seed=args.seed
            )
            evidence["v25_nulls"] = _v25_nulls(
                samples, tau_seconds=tau_best, seed=args.seed
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
        periphery, names = read_periphery(runtime)
        closed, leak = False, float("nan")
        if periphery.size:
            report = closure_gain(recording, periphery, names, seed=args.seed)
            closed, leak = bool(report.closed), float(report.leak)
            evidence["closure"] = report.as_dict()
        else:
            evidence["closure"] = {"note": "no periphery could be read"}
        _log(f"  closed: {closed}, leak {leak:.5f}")

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
        evidence["authority"] = _authority(evidence, args)
        evidence["carrier"] = _carrier_verdict(evidence)
        evidence["placement"] = _placement(evidence)
        evidence["bridge_status"] = {
            "physical_carrier": evidence["carrier"]["status"],
            "phenomenal_bridge": "UNVALIDATED",
            "note": (
                "A carrier is a causal fact about this process. Whether a carrier is a "
                "phenomenal subject is the carrier-identity postulate, which nothing in "
                "this run tests and no measurement of this kind can."
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
    if grain.get("skipped"):
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
    ("L6", "psychophysical carrier identity independently validated"),
    ("L7", "content map and continuity law independently validated"),
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
        "moves_to_L6_only_by": (
            "independent validation of the carrier-identity postulate against "
            "human and animal consciousness under novel perturbations"
        ),
        "theorems": THEOREMS,
        "separations": {
            "functional_valence_is_not_felt_valence": (
                "A system can carry affect variables without those variables "
                "being phenomenally instantiated. Sentience needs a carrier and "
                "a valence inside its phenomenal structure, and the second is "
                "not something this run measures."
            ),
            "functional_self_awareness_is_not_phenomenal_self_awareness": (
                "Representing itself, attributing authorship, predicting itself "
                "and using the self-model to change what it does are measured "
                "here. Whether the self-model appears in phenomenal structure "
                "is not."
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
    print("  phenomenal_bridge:             UNVALIDATED")
    if authority.get("blockers"):
        print("\nNOT AUTHORITATIVE:")
        for line in authority["blockers"]:
            print(f"  - {line}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    raise SystemExit(asyncio.run(main()))
