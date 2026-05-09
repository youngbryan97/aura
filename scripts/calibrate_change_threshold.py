#!/usr/bin/env python3
"""scripts/calibrate_change_threshold.py -- Empirical Threshold Calibration
============================================================================
Calibrates the CHANGE_DETECTION_THRESHOLD used by the substrate token
generator and other prediction-error-gated systems.

Instead of using a hardcoded threshold (the old 0.5, then 0.34), this
script empirically derives the optimal threshold by:
  1. Generating a set of reference prompts (known-simple, known-complex)
  2. Running them through the substrate token generator
  3. Computing the prediction error distribution
  4. Setting the threshold at a percentile that separates
     "substrate-answerable" from "needs-full-model" prompts

Usage:
    python scripts/calibrate_change_threshold.py [--percentile 65]

Output:
    Writes the calibrated threshold to ~/.aura/data/calibration/threshold.json
    Sets AURA_SUBSTRATE_PREDICTION_THRESHOLD env suggestion
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

logger = logging.getLogger("Aura.Calibration")

_OUTPUT_DIR = Path.home() / ".aura" / "data" / "calibration"
_OUTPUT_PATH = _OUTPUT_DIR / "threshold.json"


# Reference prompt sets: simple prompts the substrate should handle,
# complex prompts that need the full model.
SIMPLE_PROMPTS = [
    "continue",
    "acknowledge",
    "hold that thought",
    "I understand",
    "ok",
    "yes",
    "go on",
    "noted",
    "ready",
    "steady state",
    "checking in",
    "status report",
    "how are you feeling",
    "what's your current state",
    "maintain focus",
    "stay grounded",
    "keep going",
    "I'm here",
    "listening",
    "proceed",
]

COMPLEX_PROMPTS = [
    "Explain the difference between phenomenal consciousness and access consciousness in the context of IIT.",
    "Write me a Python script that implements a red-black tree with deletion support.",
    "What are the ethical implications of autonomous AI systems making medical decisions?",
    "Debug this error: TypeError: unsupported operand type(s) for +: 'NoneType' and 'int' in line 42",
    "Compare and contrast the architectural patterns of microservices vs monolithic applications.",
    "Can you help me write a formal proof that the halting problem is undecidable?",
    "I need to design a distributed consensus algorithm for a Byzantine fault-tolerant system.",
    "Analyze the socioeconomic factors that contributed to the 2008 financial crisis.",
    "Create a comprehensive test suite for a REST API with authentication, pagination, and error handling.",
    "How would you implement a custom garbage collector for a language runtime?",
    "Write a shader program that renders volumetric clouds with realistic lighting.",
    "Help me understand the relationship between entropy and information in Shannon's theory.",
    "Design a neural architecture search algorithm that optimizes for both accuracy and latency.",
    "What's the optimal strategy for the multi-armed bandit problem with non-stationary rewards?",
    "Implement a lock-free concurrent hash map in Rust.",
    "Explain how quantum error correction works in surface codes.",
    "Design a real-time collaborative editing system like Google Docs.",
    "What are the privacy implications of federated learning, and how can differential privacy help?",
    "Implement a parser for a context-free grammar using the Earley algorithm.",
    "How does the attention mechanism in transformers relate to memory systems in cognitive neuroscience?",
]


def calibrate(
    percentile: float = 65.0,
    seed: int = 42,
) -> Dict[str, Any]:
    """Run the calibration procedure.

    Args:
        percentile: The percentile of simple-prompt errors to use as threshold.
                   Higher = more permissive (substrate handles more).
                   Lower = more conservative (full model handles more).
        seed: Random seed for reproducibility.

    Returns:
        Dict with calibration results.
    """
    # Import substrate token generator
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    from core.brain.llm.continuous_substrate import ContinuousSubstrate
    from core.brain.llm.substrate_token_generator import SubstrateTokenGenerator

    # Create a substrate
    substrate = ContinuousSubstrate()
    # Run a few steps to warm up the state
    for _ in range(100):
        substrate._step_once()

    generator = SubstrateTokenGenerator(substrate, seed=seed)

    # Measure prediction errors for both sets
    simple_errors: List[float] = []
    complex_errors: List[float] = []

    for prompt in SIMPLE_PROMPTS:
        error = generator.estimate_prediction_error(prompt)
        simple_errors.append(error)

    for prompt in COMPLEX_PROMPTS:
        error = generator.estimate_prediction_error(prompt)
        complex_errors.append(error)

    simple_arr = np.array(simple_errors)
    complex_arr = np.array(complex_errors)

    # Compute threshold at the given percentile of simple errors
    threshold = float(np.percentile(simple_arr, percentile))

    # Compute separation metrics
    simple_below = np.sum(simple_arr <= threshold)
    complex_above = np.sum(complex_arr > threshold)
    accuracy = (simple_below + complex_above) / (len(simple_errors) + len(complex_errors))

    results = {
        "threshold": round(threshold, 6),
        "percentile": percentile,
        "seed": seed,
        "calibrated_at": time.time(),
        "simple_prompts": len(SIMPLE_PROMPTS),
        "complex_prompts": len(COMPLEX_PROMPTS),
        "simple_stats": {
            "mean": round(float(simple_arr.mean()), 6),
            "std": round(float(simple_arr.std()), 6),
            "min": round(float(simple_arr.min()), 6),
            "max": round(float(simple_arr.max()), 6),
            "median": round(float(np.median(simple_arr)), 6),
        },
        "complex_stats": {
            "mean": round(float(complex_arr.mean()), 6),
            "std": round(float(complex_arr.std()), 6),
            "min": round(float(complex_arr.min()), 6),
            "max": round(float(complex_arr.max()), 6),
            "median": round(float(np.median(complex_arr)), 6),
        },
        "separation": {
            "accuracy": round(accuracy, 4),
            "simple_correctly_classified": int(simple_below),
            "complex_correctly_classified": int(complex_above),
        },
    }

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate substrate prediction error threshold"
    )
    parser.add_argument(
        "--percentile", type=float, default=65.0,
        help="Percentile of simple-prompt errors to use as threshold (default: 65)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")

    print("🔬 Running calibration procedure...")
    print(f"   Percentile: {args.percentile}")
    print(f"   Seed: {args.seed}")
    print()

    results = calibrate(percentile=args.percentile, seed=args.seed)

    # Save results
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _OUTPUT_PATH.write_text(
        json.dumps(results, indent=2, default=str),
        encoding="utf-8",
    )

    # Display results
    print(f"📊 Calibration Results:")
    print(f"   Calibrated threshold: {results['threshold']}")
    print(f"   Simple prompts: mean={results['simple_stats']['mean']:.4f}, "
          f"median={results['simple_stats']['median']:.4f}")
    print(f"   Complex prompts: mean={results['complex_stats']['mean']:.4f}, "
          f"median={results['complex_stats']['median']:.4f}")
    print(f"   Separation accuracy: {results['separation']['accuracy']:.1%}")
    print(f"\n💾 Results saved to: {_OUTPUT_PATH}")
    print(f"\n💡 To apply: export AURA_SUBSTRATE_PREDICTION_THRESHOLD={results['threshold']}")


if __name__ == "__main__":
    main()
