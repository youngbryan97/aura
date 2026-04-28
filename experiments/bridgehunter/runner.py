"""experiments/bridgehunter/runner.py

BridgeHunter — Optuna-driven consciousness metric hyperparameter search.

Searches the substrate configuration space for parameters that maximize
consciousness-correlated metrics (Φ, ignition rate, causal emergence, etc.).

Usage:
    python -m experiments.bridgehunter.runner --n-trials 50
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

logger = logging.getLogger("BridgeHunter.Runner")


def create_substrate_and_simulate(params: dict, config) -> dict:
    """Create a substrate with given params, run simulation, return metrics.

    This runs a lightweight simulation without the full AURA stack,
    using just the ODE dynamics and metric computations.
    """
    from experiments.bridgehunter.metrics import (
        compute_causal_emergence,
        compute_ignition_rate,
        compute_phi_surrogate,
        compute_self_reference,
        compute_spectral_entropy,
    )

    N = params["neuron_count"]
    dt = config.dt
    warmup = config.warmup_ticks
    measure = config.measurement_ticks
    total = warmup + measure

    # Initialize substrate state
    x = np.zeros(N)
    W = np.random.randn(N, N) * 0.1
    alpha = params["recurrence_alpha"]
    prior = None

    # Storage
    trajectory = np.zeros((measure, N))
    priorities = []

    for t in range(total):
        # ODE step: dx/dt = -decay*x + tanh(Wx) + noise
        noise = np.random.randn(N) * params["noise_level"]
        dx = (-params["decay_rate"] * x + np.tanh(W @ x) + noise) * dt
        x = np.clip(x + dx, -1.0, 1.0)

        # Recurrent self-model blend
        if prior is None:
            prior = x.copy()
        blended = alpha * prior + (1.0 - alpha) * x
        x = np.clip(blended, -1.0, 1.0)
        prior = x.copy()

        # Hebbian update (sparse)
        if t % 50 == 0:
            outer = np.outer(x, x)
            W += params["hebbian_rate"] * outer
            norm = np.linalg.norm(W)
            if norm > 10.0:
                W *= 10.0 / norm

        # Store after warmup
        if t >= warmup:
            trajectory[t - warmup] = x
            # Simulate priority (energy-based)
            energy = np.mean(np.abs(x))
            priorities.append(energy)

    # Compute metrics
    phi = compute_phi_surrogate(trajectory)
    ignition = compute_ignition_rate(priorities, threshold=0.6)
    causal = compute_causal_emergence(trajectory)
    entropy = compute_spectral_entropy(trajectory[-1])
    self_ref = compute_self_reference(trajectory)

    return {
        "phi": phi,
        "ignition_rate": ignition,
        "causal_emergence": causal,
        "spectral_entropy": entropy,
        "self_reference": self_ref,
    }


def objective(trial, config):
    """Optuna objective function."""
    params = {
        "neuron_count": trial.suggest_int("neuron_count", *config.neuron_count_range, step=8),
        "decay_rate": trial.suggest_float("decay_rate", *config.decay_rate_range),
        "noise_level": trial.suggest_float("noise_level", *config.noise_level_range, log=True),
        "hebbian_rate": trial.suggest_float("hebbian_rate", *config.hebbian_rate_range, log=True),
        "recurrence_alpha": trial.suggest_float("recurrence_alpha", *config.recurrence_alpha_range),
    }

    metrics = create_substrate_and_simulate(params, config)

    # Weighted composite score
    score = 0.0
    for metric_name, weight in config.metric_weights.items():
        val = metrics.get(metric_name, 0.0)
        score += weight * val

    # Log for visibility
    logger.info(
        "Trial %d: score=%.4f (phi=%.4f, ignition=%.4f, ce=%.4f)",
        trial.number, score,
        metrics["phi"], metrics["ignition_rate"], metrics["causal_emergence"],
    )

    return score


def main():
    parser = argparse.ArgumentParser(description="BridgeHunter — Consciousness Metric Search")
    parser.add_argument("--n-trials", type=int, default=50, help="Number of Optuna trials")
    parser.add_argument("--study-name", type=str, default="aura_consciousness", help="Study name")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")

    try:
        import optuna
    except ImportError:
        logger.error("Optuna not installed. Run: pip install optuna")
        sys.exit(1)

    from experiments.bridgehunter.config import BridgeHunterConfig

    config = BridgeHunterConfig(n_trials=args.n_trials, study_name=args.study_name)

    study = optuna.create_study(
        study_name=config.study_name,
        direction="maximize",
    )

    study.optimize(
        lambda trial: objective(trial, config),
        n_trials=config.n_trials,
        timeout=config.timeout_seconds,
    )

    # Report
    best = study.best_trial
    logger.info("=" * 60)
    logger.info("BEST TRIAL: #%d (score=%.4f)", best.number, best.value)
    for key, val in best.params.items():
        logger.info("  %s: %s", key, val)
    logger.info("=" * 60)

    # Save results
    results_dir = Path(__file__).parent / "results"
    get_task_tracker().create_task(get_storage_gateway().create_dir(results_dir, cause='main'))
    results_path = results_dir / f"{config.study_name}_best.txt"
    with open(results_path, "w") as f:
        f.write(f"Best Trial #{best.number}\n")
        f.write(f"Score: {best.value:.6f}\n\n")
        for key, val in best.params.items():
            f.write(f"{key}: {val}\n")

    logger.info("Results saved to %s", results_path)


if __name__ == "__main__":
    main()
