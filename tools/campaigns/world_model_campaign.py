#!/usr/bin/env python3
"""Two questions a world model only answers by being run.

  140  a latent predictor beats reconstruction on downstream planning at
       equal compute
  198  a recovered causal graph predicts unseen interventions better than a
       correlational model

Both are the same shape: a model that scores well on the thing it was trained
on can be the worse model, and only a second measurement says so. A
reconstruction objective spends capacity on pixels the controller never
reads; a correlational fit predicts observations and falls over the moment
somebody reaches into the world.

The world here is a small linear structural causal model with hidden
confounding, which is the minimum that can tell the two apart: the
correlational fit and the causal one agree on every observation and disagree
under intervention. Nothing about it is a claim about Aura's live world
model — it is a claim about whether the comparison machinery reports the
right winner when the answer is known in advance.

    python tools/campaigns/world_model_campaign.py --trials 40
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.world_model.prediction_quality import (  # noqa: E402
    InterventionLedger,
    ObjectiveComparison,
)


# ── the world ─────────────────────────────────────────────────────────────
#
#   H  →  X  →  Y      and      H  →  Y
#
# H is hidden. Observationally X predicts Y well, because both move with H.
# Under do(X = x) the H → X arrow is cut and X's real effect on Y is all
# that is left, which is smaller. A fit that never saw an intervention
# cannot know the difference, and that is the whole point.

X_TO_Y = 0.4
H_TO_X = 1.0
H_TO_Y = 1.6


def _observational_sample(rng: random.Random) -> tuple[float, float]:
    h = rng.gauss(0.0, 1.0)
    x = H_TO_X * h + rng.gauss(0.0, 0.3)
    y = X_TO_Y * x + H_TO_Y * h + rng.gauss(0.0, 0.3)
    return x, y


def _interventional_sample(rng: random.Random, x: float) -> float:
    """do(X = x): the arrow into X is cut, H still reaches Y."""
    h = rng.gauss(0.0, 1.0)
    return X_TO_Y * x + H_TO_Y * h + rng.gauss(0.0, 0.3)


def _fit_slope(rows: list[tuple[float, float]]) -> float:
    n = len(rows)
    mean_x = sum(x for x, _ in rows) / n
    mean_y = sum(y for _, y in rows) / n
    cov = sum((x - mean_x) * (y - mean_y) for x, y in rows)
    var = sum((x - mean_x) ** 2 for x, _ in rows)
    return cov / var if var else 0.0


def causal_arm(
    *, observations: int, interventions_seen: int, tested: int, seed: int
) -> dict[str, object]:
    """Two models, the same budget of rows, scored on unseen interventions.

    The correlational arm spends its whole budget on observations. The causal
    arm spends the same total, some of it on interventions it is allowed to
    see. Equal rows is the compute match: the causal arm is not given more
    data, it is given differently sourced data.
    """
    rng = random.Random(seed)
    total = observations + interventions_seen

    correlational_rows = [_observational_sample(rng) for _ in range(total)]
    correlational_slope = _fit_slope(correlational_rows)

    causal_observations = [_observational_sample(rng) for _ in range(observations)]
    del causal_observations  # spent from the same budget, used by neither slope
    seen = [
        (x, _interventional_sample(rng, x))
        for x in (rng.gauss(0.0, 1.0) for _ in range(interventions_seen))
    ]
    causal_slope = _fit_slope(seen)

    ledger = InterventionLedger()
    correlational_ledger = InterventionLedger()
    for _ in range(tested):
        x = rng.uniform(-3.0, 3.0)
        observed = _interventional_sample(rng, x)
        ledger.record_intervention("y", causal_slope * x, observed)
        correlational_ledger.record_intervention("y", correlational_slope * x, observed)
        obs_x, obs_y = _observational_sample(rng)
        ledger.record_observation("y", causal_slope * obs_x, obs_y)
        correlational_ledger.record_observation(
            "y", correlational_slope * obs_x, obs_y
        )

    causal = ledger.verdict()
    correlational = correlational_ledger.verdict()
    return {
        "rows_each": total,
        "true_effect": X_TO_Y,
        "causal_slope": round(causal_slope, 4),
        "correlational_slope": round(correlational_slope, 4),
        "causal_interventional_rmse": round(causal["interventional_rmse"], 4),
        "correlational_interventional_rmse": round(
            correlational["interventional_rmse"], 4
        ),
        "causal_wins_under_intervention": (
            causal["interventional_rmse"] < correlational["interventional_rmse"]
        ),
        # The trap: on observations alone the correlational fit is the better
        # model, which is why an observational score cannot settle this.
        "correlational_wins_on_observations": (
            correlational["observational_rmse"] < causal["observational_rmse"]
        ),
        "causal_reading": causal["reading"],
        "correlational_reading": correlational["reading"],
    }


def objective_arm(
    *, capacity: int, steps: int, observed_dims: int, noise_scale: float, seed: int
) -> dict[str, object]:
    """Reconstruction against latent prediction, both picking their own basis.

    The world is a two-dimensional latent that moves smoothly, projected into
    ``observed_dims`` dimensions, plus loud directions that carry no signal and
    do not persist. Both objectives are given the same budget: ``capacity``
    directions of the observation to keep. They differ only in how they choose.

    Reconstruction keeps the directions with the most variance, because that is
    what minimises reconstruction error. The loud noise is high-variance, so it
    buys those directions with its budget.

    Latent prediction keeps the directions that are PREDICTABLE — highest
    lag-one autocorrelation — because that is what minimises next-step error in
    the code. Noise is not predictable, so it does not buy those.

    Both are then scored on the same control task: recover the latent from the
    kept directions, which is what a controller actually needs. Neither the
    ranking nor the score is authored here; both fall out of the same data.
    """
    import numpy as np

    rng = np.random.default_rng(seed)

    # Two slow latents.
    latent = np.zeros((steps, 2))
    for t in range(1, steps):
        latent[t] = 0.95 * latent[t - 1] + rng.normal(0.0, 0.1, size=2)

    # Half the channels carry the latent quietly; the other half are loud and
    # carry nothing. Adding noise to every channel instead would make every
    # channel signal-bearing, and then the two objectives are choosing between
    # near-identical options and the comparison measures rounding.
    signal_dims = observed_dims // 2
    mixing = rng.normal(0.0, 1.0, size=(2, signal_dims))
    quiet = latent @ mixing + rng.normal(0.0, 0.2, size=(steps, signal_dims))
    loud = rng.normal(0.0, noise_scale, size=(steps, observed_dims - signal_dims))
    observation = np.c_[quiet, loud]

    variance = observation.var(axis=0)
    centred = observation - observation.mean(axis=0)
    denom = (centred[:-1] ** 2).sum(axis=0)
    autocorrelation = np.divide(
        (centred[:-1] * centred[1:]).sum(axis=0),
        denom,
        out=np.zeros(observed_dims),
        where=denom > 0,
    )

    by_variance = np.argsort(-variance)[:capacity]
    by_predictability = np.argsort(-autocorrelation)[:capacity]

    def _control_success(kept: "np.ndarray") -> float:
        """How much of the latent a controller can recover from these dims.

        Least squares from the kept directions to the true latent, scored as
        the fraction of latent variance explained on a held-out half.
        """
        half = steps // 2
        design = np.c_[observation[:, kept], np.ones(steps)]
        coefficients, *_ = np.linalg.lstsq(
            design[:half], latent[:half], rcond=None
        )
        predicted = design[half:] @ coefficients
        residual = ((latent[half:] - predicted) ** 2).sum()
        total = ((latent[half:] - latent[half:].mean(axis=0)) ** 2).sum()
        return float(max(0.0, 1.0 - residual / total)) if total else 0.0

    def _next_step_loss(kept: "np.ndarray") -> float:
        code = observation[:, kept]
        design = np.c_[code[:-1], np.ones(steps - 1)]
        coefficients, *_ = np.linalg.lstsq(design, code[1:], rcond=None)
        return float(((code[1:] - design @ coefficients) ** 2).mean())

    def _reconstruction_loss(kept: "np.ndarray") -> float:
        dropped = np.setdiff1d(np.arange(observed_dims), kept)
        return float((observation[:, dropped] ** 2).mean()) if dropped.size else 0.0

    comparison = ObjectiveComparison(
        reconstruction_loss=_reconstruction_loss(by_variance),
        latent_loss=_reconstruction_loss(by_predictability),
        reconstruction_control_success=_control_success(by_variance),
        latent_control_success=_control_success(by_predictability),
        compute_matched=True,
    )
    return comparison.to_dict() | {
        "capacity": capacity,
        "observed_dims": observed_dims,
        "signal_dims": signal_dims,
        "reconstruction_kept_signal_dims": int((by_variance < signal_dims).sum()),
        "latent_kept_signal_dims": int((by_predictability < signal_dims).sum()),
        "steps": steps,
        "noise_scale": noise_scale,
        "dims_both_kept": int(np.intersect1d(by_variance, by_predictability).size),
        "reconstruction_next_step_loss": round(_next_step_loss(by_variance), 5),
        "latent_next_step_loss": round(_next_step_loss(by_predictability), 5),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=40)
    parser.add_argument("--observations", type=int, default=400)
    parser.add_argument("--interventions-seen", type=int, default=100)
    parser.add_argument("--tested", type=int, default=200)
    parser.add_argument("--capacity", type=int, default=8)
    parser.add_argument("--observed-dims", type=int, default=48)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--noise-scale", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=1_618_033)
    parser.add_argument("--out", default="docs/evidence/world_model_campaign.json")
    args = parser.parse_args()

    causal_rows = [
        causal_arm(
            observations=args.observations,
            interventions_seen=args.interventions_seen,
            tested=args.tested,
            seed=args.seed + i,
        )
        for i in range(args.trials)
    ]
    # A sweep, not a point: the loud channels only outrank the quiet ones
    # above some noise scale, and reporting one value invites the reading
    # that it was chosen to give this answer.
    objective_sweep = [
        objective_arm(
            capacity=args.capacity,
            steps=args.steps,
            observed_dims=args.observed_dims,
            noise_scale=scale,
            seed=args.seed,
        )
        for scale in (0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 6.0)
    ]
    objective = objective_arm(
        capacity=args.capacity,
        steps=args.steps,
        observed_dims=args.observed_dims,
        noise_scale=args.noise_scale,
        seed=args.seed,
    )

    causal_wins = sum(1 for r in causal_rows if r["causal_wins_under_intervention"])
    observational_trap = sum(
        1 for r in causal_rows if r["correlational_wins_on_observations"]
    )
    payload = {
        "schema": "aura.world_model_campaign.v1",
        "cards": ["140", "198"],
        "claim_boundary": (
            "a linear structural causal model with hidden confounding, and two "
            "training objectives choosing a basis from the same synthetic "
            "observations at the same capacity; measures "
            "whether the comparison machinery names the right winner when the "
            "answer is known, not Aura's world model on any real environment"
        ),
        "config": {
            "trials": args.trials,
            "observations": args.observations,
            "interventions_seen": args.interventions_seen,
            "tested": args.tested,
            "capacity": args.capacity,
            "observed_dims": args.observed_dims,
            "steps": args.steps,
            "noise_scale": args.noise_scale,
            "seed": args.seed,
        },
        "causal": {
            "trials": len(causal_rows),
            "causal_wins_under_intervention": causal_wins,
            "correlational_wins_on_observations": observational_trap,
            "median_causal_slope": round(
                statistics.median(r["causal_slope"] for r in causal_rows), 4
            ),
            "median_correlational_slope": round(
                statistics.median(r["correlational_slope"] for r in causal_rows), 4
            ),
            "true_effect": X_TO_Y,
            "median_causal_rmse": round(
                statistics.median(
                    r["causal_interventional_rmse"] for r in causal_rows
                ),
                4,
            ),
            "median_correlational_rmse": round(
                statistics.median(
                    r["correlational_interventional_rmse"] for r in causal_rows
                ),
                4,
            ),
        },
        "objective": objective,
        "objective_sweep": [
            {
                "noise_scale": row["noise_scale"],
                "reconstruction_kept_signal_dims": row["reconstruction_kept_signal_dims"],
                "latent_kept_signal_dims": row["latent_kept_signal_dims"],
                "reconstruction_control_success": round(
                    row["reconstruction_control_success"], 4
                ),
                "latent_control_success": round(row["latent_control_success"], 4),
                "verdict": row["verdict"],
            }
            for row in objective_sweep
        ],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))

    from core.governance_context import local_internal_governed_scope
    from core.runtime.file_write_gateway import get_file_write_gateway

    out = ROOT / args.out
    with local_internal_governed_scope("world_model_campaign"):
        get_file_write_gateway().write_text(
            out, json.dumps(payload, indent=2, sort_keys=True) + "\n"
        )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
