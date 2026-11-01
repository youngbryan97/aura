#!/usr/bin/env python3
"""A steering basis of random directions, each the length of the one it stands for.

The random-direction control of the reports ground. Her run steers her cortex
along the directions fitted to her (her felt push at layer 43 among them), and
her rating followed her state. Steering-off and an untuned model said the
steering is what carries it. They cannot say whether her directions matter or
whether any push of that size would do: a random push moves a distribution
about as far as a fitted one of the same norm (fusion_certificate.beats_noise).
This basis is her basis with every direction replaced and every length kept,
so a run under it differs from hers only in where the push points.

Everything else is hers and stays as it was: the files, the layers, the slot
her felt valence is read from, the reference, spread and push each vector is
weighed by, and the checkpoint each file is bound to. So it attaches to her
cortex exactly as her basis does, and it steers only under a fusion
certificate earned by its own digest, like any other basis.

    python tools/make_random_steering_basis.py <her basis> <out dir> --seed 1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

#: The arrays a basis file carries as directions. Every other key is copied.
DIRECTIONS: tuple[str, ...] = ("v", "everywhere")
STATEMENT: str = "basis.json"
SOURCE: str = "random_directions"


def random_like(vector: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """A direction drawn uniformly on the sphere, at the length of `vector`.

    A zero vector stays zero: it pushes nowhere, and a random one would push
    where her basis does not.
    """
    vector = np.asarray(vector, dtype=np.float64)
    length = float(np.linalg.norm(vector))
    if length == 0.0:
        return np.zeros_like(vector, dtype=np.float32)
    drawn = rng.standard_normal(vector.shape)
    return (drawn * (length / float(np.linalg.norm(drawn)))).astype(np.float32)


def make_random_basis(source: Path, out: Path, seed: int) -> dict[str, Any]:
    """Write the random basis for `source` into `out`, which must not exist, and return its statement."""
    statement = json.loads((source / STATEMENT).read_text())
    if statement.get("source") == SOURCE:
        raise ValueError(f"refusing: {source} is already a random basis")
    files = sorted(source.glob("*.npz"))
    if not files:
        raise ValueError(f"refusing: {source} holds no vectors")
    out.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(seed)
    replaced: dict[str, list[str]] = {}
    for path in files:
        with np.load(path, allow_pickle=False) as held:
            arrays = {key: held[key] for key in held.files}
        moved = [key for key in DIRECTIONS if key in arrays]
        for key in moved:
            arrays[key] = random_like(arrays[key], rng)
        np.savez(out / path.name, **arrays)
        replaced[path.name] = moved
    random = {
        "source": SOURCE,
        "seed": int(seed),
        "rule": "each direction drawn uniformly on the sphere at its own stored length; every other field kept",
        "stands_for": str(source.resolve()),
        "stands_for_statement": statement,
        "vectors": len(files),
        "everywhere": sum("everywhere" in moved for moved in replaced.values()),
        "model_descriptor_sha256": statement.get("model_descriptor_sha256", ""),
    }
    (out / STATEMENT).write_text(json.dumps(random, indent=1) + "\n")
    return random


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source", type=Path, help="the basis whose directions are replaced")
    parser.add_argument("out", type=Path, help="a directory that does not exist yet")
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args(argv)
    made = make_random_basis(args.source, args.out, args.seed)
    print(json.dumps({key: value for key, value in made.items() if key != "stands_for_statement"}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
