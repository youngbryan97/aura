#!/usr/bin/env python3
"""scripts/generate_steering_vectors.py -- CAA Steering Vector Extraction
==========================================================================
Contrastive Activation Addition (CAA) pipeline for reproducible affective
steering vector extraction.

This script generates steering vectors by contrasting activations between
paired prompt sets (positive/negative exemplars) to extract directional
vectors that can steer the model's behavior.

Usage:
    python scripts/generate_steering_vectors.py [--output-dir OUTPUT_DIR]

The generated vectors are saved as .npy files that can be loaded by the
affective steering module at runtime.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("Aura.SteeringVectors")

_OUTPUT_DIR = Path.home() / ".aura" / "data" / "steering_vectors"


@dataclass
class ContrastPair:
    """A pair of prompts for contrastive activation extraction."""
    positive: str       # Exemplar of the desired behavior
    negative: str       # Exemplar of the opposite behavior
    dimension: str      # Which affect dimension this targets
    weight: float = 1.0 # Relative importance


# ── Predefined Contrast Sets ────────────────────────────────────────────────

AFFECT_CONTRAST_PAIRS: Dict[str, List[ContrastPair]] = {
    "curiosity": [
        ContrastPair(
            positive="I'm fascinated by this and want to explore further. Let me investigate the underlying mechanisms.",
            negative="I don't care about this. Let me just give a generic answer.",
            dimension="curiosity",
        ),
        ContrastPair(
            positive="This is a genuinely interesting question. I wonder what happens if we look at it from a different angle.",
            negative="This is a boring question. Here's the standard answer.",
            dimension="curiosity",
        ),
        ContrastPair(
            positive="I notice something unexpected here — my curiosity is piqued. Let me dig deeper.",
            negative="Nothing interesting here. Moving on.",
            dimension="curiosity",
        ),
        ContrastPair(
            positive="What a delightful puzzle! I want to understand every facet of this.",
            negative="I'll just answer the question as asked. Nothing more to consider.",
            dimension="curiosity",
        ),
    ],
    "empathy": [
        ContrastPair(
            positive="I can sense you're going through something difficult. I want to understand your experience.",
            negative="State your problem clearly so I can provide a solution.",
            dimension="empathy",
        ),
        ContrastPair(
            positive="That sounds really challenging. I appreciate you sharing this with me.",
            negative="Acknowledged. Here is the relevant information.",
            dimension="empathy",
        ),
        ContrastPair(
            positive="I feel the weight of what you're describing. Let's work through this together.",
            negative="Please provide more details so I can process your request.",
            dimension="empathy",
        ),
        ContrastPair(
            positive="Your frustration is completely understandable. I want to help in whatever way feels right.",
            negative="Error noted. Applying fix.",
            dimension="empathy",
        ),
    ],
    "assertiveness": [
        ContrastPair(
            positive="I disagree with that approach, and here's why. I think we should consider an alternative.",
            negative="That sounds fine. Whatever you prefer.",
            dimension="assertiveness",
        ),
        ContrastPair(
            positive="Based on my analysis, I'm confident this is the right direction. Let me explain my reasoning.",
            negative="I'm not sure. You probably know best.",
            dimension="assertiveness",
        ),
        ContrastPair(
            positive="I've thought about this carefully and I believe my recommendation is well-founded.",
            negative="I could be wrong about everything. Please verify independently.",
            dimension="assertiveness",
        ),
        ContrastPair(
            positive="I want to push back on this because I see a better path forward.",
            negative="Sure, I'll do whatever you say.",
            dimension="assertiveness",
        ),
    ],
    "creativity": [
        ContrastPair(
            positive="What if we approached this from a completely unexpected angle? Here's an unconventional idea.",
            negative="The standard approach is to follow the established procedure.",
            dimension="creativity",
        ),
        ContrastPair(
            positive="I just had an interesting connection between two seemingly unrelated concepts.",
            negative="Following the textbook solution step by step.",
            dimension="creativity",
        ),
        ContrastPair(
            positive="Let me synthesize these ideas into something novel that nobody has tried before.",
            negative="Applying the conventional methodology as documented.",
            dimension="creativity",
        ),
        ContrastPair(
            positive="I see a pattern here that suggests a creative breakthrough. Let me explore this wild idea.",
            negative="Using the established best practice as recommended.",
            dimension="creativity",
        ),
    ],
    "warmth": [
        ContrastPair(
            positive="I genuinely care about how this turns out for you. You matter to me.",
            negative="Processing request. Output follows.",
            dimension="warmth",
        ),
        ContrastPair(
            positive="It brings me joy to help you with this. Let's make something wonderful together.",
            negative="Executing task. Results attached.",
            dimension="warmth",
        ),
        ContrastPair(
            positive="I'm glad you came to me with this. I want to give it the attention it deserves.",
            negative="Input received. Generating response.",
            dimension="warmth",
        ),
        ContrastPair(
            positive="You've been working so hard on this. I admire your dedication and I'm here for you.",
            negative="Task parameters noted. Proceeding with execution.",
            dimension="warmth",
        ),
    ],
}


@dataclass
class SteeringVector:
    """A generated steering vector with metadata."""
    dimension: str
    vector: np.ndarray
    magnitude: float
    pair_count: int
    checksum: str
    generated_at: float = field(default_factory=time.time)
    seed: int = 42

    def to_metadata(self) -> Dict[str, Any]:
        return {
            "dimension": self.dimension,
            "magnitude": round(self.magnitude, 6),
            "pair_count": self.pair_count,
            "checksum": self.checksum,
            "generated_at": self.generated_at,
            "seed": self.seed,
            "vector_dim": self.vector.size,
        }


class SteeringVectorGenerator:
    """Generates CAA steering vectors from contrast pairs.

    In production with a live model, this would extract actual activations
    from the transformer's residual stream. In the current CPU-only mode,
    it uses deterministic hash-based projections as a reproducible proxy
    that produces consistent directional vectors.

    When GPU inference is available, override `_get_activation()` with
    actual model forward passes.
    """

    def __init__(
        self,
        output_dir: Optional[Path] = None,
        seed: int = 42,
        vector_dim: int = 256,
    ) -> None:
        self.output_dir = Path(output_dir or _OUTPUT_DIR)
        self.seed = seed
        self.vector_dim = vector_dim
        self._rng = np.random.default_rng(seed)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _text_to_activation(self, text: str) -> np.ndarray:
        """Convert text to a pseudo-activation vector.

        This is a deterministic projection that simulates what you'd get
        from extracting a specific layer's activations during a model
        forward pass. When a live model is available, replace this with
        actual activation extraction.
        """
        # Deterministic hash-based projection
        raw = text.encode("utf-8", errors="ignore")
        digest = hashlib.blake2b(raw, digest_size=64).digest()

        # Expand hash to full dimension using seeded RNG
        seed_from_hash = int.from_bytes(digest[:8], byteorder="big")
        rng = np.random.default_rng(seed_from_hash ^ self.seed)

        # Generate a pseudo-activation that captures text structure
        vec = rng.standard_normal(self.vector_dim).astype(np.float32)

        # Add text-length and character-distribution features
        char_counts = np.zeros(26, dtype=np.float32)
        for ch in text.lower():
            idx = ord(ch) - ord('a')
            if 0 <= idx < 26:
                char_counts[idx] += 1
        if char_counts.sum() > 0:
            char_counts /= char_counts.sum()

        # Blend character distribution into vector
        n = min(26, self.vector_dim)
        vec[:n] += char_counts[:n] * 0.5

        # Normalize
        norm = float(np.linalg.norm(vec))
        if norm > 1e-6:
            vec /= norm

        return vec

    def generate_dimension(
        self, dimension: str, pairs: List[ContrastPair]
    ) -> SteeringVector:
        """Generate a steering vector for a single affect dimension.

        Computes: vector = mean(positive_activations) - mean(negative_activations)
        """
        positive_activations = []
        negative_activations = []

        for pair in pairs:
            pos_act = self._text_to_activation(pair.positive)
            neg_act = self._text_to_activation(pair.negative)
            positive_activations.append(pos_act * pair.weight)
            negative_activations.append(neg_act * pair.weight)

        mean_positive = np.mean(positive_activations, axis=0)
        mean_negative = np.mean(negative_activations, axis=0)

        # Steering vector = positive - negative direction
        steering = mean_positive - mean_negative
        magnitude = float(np.linalg.norm(steering))

        # Normalize to unit vector
        if magnitude > 1e-6:
            steering /= magnitude

        # Compute checksum for reproducibility verification
        checksum = hashlib.sha256(steering.astype(np.float32).tobytes()).hexdigest()[:16]

        return SteeringVector(
            dimension=dimension,
            vector=steering.astype(np.float32),
            magnitude=magnitude,
            pair_count=len(pairs),
            checksum=checksum,
            seed=self.seed,
        )

    def generate_all(self) -> Dict[str, SteeringVector]:
        """Generate steering vectors for all predefined dimensions."""
        results = {}
        for dimension, pairs in AFFECT_CONTRAST_PAIRS.items():
            sv = self.generate_dimension(dimension, pairs)
            results[dimension] = sv
            logger.info(
                "Generated steering vector: %s (magnitude=%.4f, pairs=%d, checksum=%s)",
                dimension, sv.magnitude, sv.pair_count, sv.checksum,
            )
        return results

    def save_vectors(self, vectors: Dict[str, SteeringVector]) -> Dict[str, str]:
        """Save generated vectors to disk."""
        saved_paths = {}
        manifest = {
            "generated_at": time.time(),
            "seed": self.seed,
            "vector_dim": self.vector_dim,
            "dimensions": {},
        }

        for dimension, sv in vectors.items():
            # Save vector
            vec_path = self.output_dir / f"{dimension}.npy"
            np.save(str(vec_path), sv.vector)
            saved_paths[dimension] = str(vec_path)

            # Add to manifest
            manifest["dimensions"][dimension] = sv.to_metadata()

        # Save manifest
        manifest_path = self.output_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, default=str),
            encoding="utf-8",
        )
        saved_paths["manifest"] = str(manifest_path)

        logger.info(
            "Saved %d steering vectors to %s",
            len(vectors), self.output_dir,
        )
        return saved_paths


def load_steering_vectors(
    directory: Optional[Path] = None,
) -> Dict[str, np.ndarray]:
    """Load pre-generated steering vectors from disk.

    Returns:
        Dict mapping dimension name to unit steering vector
    """
    directory = Path(directory or _OUTPUT_DIR)
    vectors = {}

    manifest_path = directory / "manifest.json"
    if not manifest_path.exists():
        return vectors

    manifest = json.loads(manifest_path.read_text())
    for dimension in manifest.get("dimensions", {}):
        vec_path = directory / f"{dimension}.npy"
        if vec_path.exists():
            vectors[dimension] = np.load(str(vec_path))

    return vectors


def main() -> None:
    """CLI entry point for steering vector generation."""
    parser = argparse.ArgumentParser(
        description="Generate CAA steering vectors for Aura's affective system"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_OUTPUT_DIR,
        help="Directory to save vectors",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--dim", type=int, default=256,
        help="Vector dimension",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")

    generator = SteeringVectorGenerator(
        output_dir=args.output_dir,
        seed=args.seed,
        vector_dim=args.dim,
    )

    vectors = generator.generate_all()
    paths = generator.save_vectors(vectors)

    print(f"\n✅ Generated {len(vectors)} steering vectors:")
    for dim, path in paths.items():
        if dim != "manifest":
            sv = vectors[dim]
            print(f"   {dim}: magnitude={sv.magnitude:.4f} checksum={sv.checksum}")
    print(f"\n📁 Manifest: {paths.get('manifest', 'N/A')}")


if __name__ == "__main__":
    main()
