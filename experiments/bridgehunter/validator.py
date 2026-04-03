"""experiments/bridgehunter/validator.py

BridgeHunter Validator — Visualization and validation of consciousness metrics.

Generates plots of metric trajectories for visual inspection.

Usage:
    python -m experiments.bridgehunter.validator --ticks 500
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

logger = logging.getLogger("BridgeHunter.Validator")


def run_and_collect(params: dict, n_ticks: int = 500, dt: float = 0.1):
    """Run a full simulation and collect per-tick metrics."""
    from experiments.bridgehunter.metrics import (
        compute_phi_surrogate,
        compute_spectral_entropy,
    )

    N = params.get("neuron_count", 64)
    x = np.zeros(N)
    W = np.random.randn(N, N) * 0.1
    alpha = params.get("recurrence_alpha", 0.3)
    prior = None

    # Per-tick collection
    phi_values = []
    energy_values = []
    entropy_values = []
    buffer = []

    for t in range(n_ticks):
        noise = np.random.randn(N) * params.get("noise_level", 0.01)
        dx = (-params.get("decay_rate", 0.05) * x + np.tanh(W @ x) + noise) * dt
        x = np.clip(x + dx, -1.0, 1.0)

        if prior is None:
            prior = x.copy()
        blended = alpha * prior + (1.0 - alpha) * x
        x = np.clip(blended, -1.0, 1.0)
        prior = x.copy()

        if t % 50 == 0:
            outer = np.outer(x, x)
            W += params.get("hebbian_rate", 0.001) * outer
            norm = np.linalg.norm(W)
            if norm > 10.0:
                W *= 10.0 / norm

        buffer.append(x.copy())
        if len(buffer) > 64:
            buffer.pop(0)

        # Metrics
        energy_values.append(float(np.mean(np.abs(x))))
        entropy_values.append(compute_spectral_entropy(x))

        if len(buffer) >= 16:
            traj = np.array(buffer)
            phi_values.append(compute_phi_surrogate(traj))
        else:
            phi_values.append(0.0)

    return {
        "phi": phi_values,
        "energy": energy_values,
        "entropy": entropy_values,
    }


def plot_metrics(metrics: dict, output_path: str = None):
    """Generate matplotlib plots for the metric trajectories."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.error("matplotlib not installed. Run: pip install matplotlib")
        return

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    axes[0].plot(metrics["phi"], color="#8B5CF6", linewidth=0.8)
    axes[0].set_ylabel("Φ Surrogate")
    axes[0].set_title("BridgeHunter — Consciousness Metric Trajectories")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(metrics["energy"], color="#10B981", linewidth=0.8)
    axes[1].set_ylabel("Substrate Energy")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(metrics["entropy"], color="#F59E0B", linewidth=0.8)
    axes[2].set_ylabel("Spectral Entropy")
    axes[2].set_xlabel("Tick")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info("Plot saved to %s", output_path)
    else:
        results_dir = Path(__file__).parent / "results"
        results_dir.mkdir(exist_ok=True)
        out = str(results_dir / "metric_trajectories.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        logger.info("Plot saved to %s", out)

    plt.close()


def main():
    parser = argparse.ArgumentParser(description="BridgeHunter Validator")
    parser.add_argument("--ticks", type=int, default=500, help="Simulation ticks")
    parser.add_argument("--output", type=str, default=None, help="Output path for plot")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")

    params = {
        "neuron_count": 64,
        "decay_rate": 0.05,
        "noise_level": 0.01,
        "hebbian_rate": 0.001,
        "recurrence_alpha": 0.3,
    }

    logger.info("Running %d-tick simulation...", args.ticks)
    metrics = run_and_collect(params, n_ticks=args.ticks)

    logger.info(
        "Final metrics — Φ: %.4f, Energy: %.4f, Entropy: %.4f",
        metrics["phi"][-1], metrics["energy"][-1], metrics["entropy"][-1],
    )

    plot_metrics(metrics, output_path=args.output)


if __name__ == "__main__":
    main()
