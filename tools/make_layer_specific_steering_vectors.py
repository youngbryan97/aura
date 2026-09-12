#!/usr/bin/env python3
"""Keep what each layer's steering vector says that the other layers do not.

The specificity control permutes the vectors among the layers they were
derived at. On the 27B it scored 0.44 against a steered 0.75 and a baseline of
0.08 — well over half the movement survived being put at the wrong layer, so
which layer held which vector was not load-bearing, and the control was right
to say so.

The reason is measurable rather than mysterious. Across the four full-attention
layers the same trait's vectors sit at cosine 0.61 to 0.89 of each other, mean
0.76. Three quarters of each vector is a direction all four share, so a
permutation mostly moves a vector to a layer that was already being pushed
nearly the same way.

This writes the other three quarters' complement: for each trait, the direction
common to its layers is estimated and removed, leaving at each layer only the
part that layer contributes. The shared part is not discarded quietly — it is
written beside them, so an experiment can put it back at one layer and ask
whether the movement was ever anything else.

What this cannot do is decide whether the result still steers. That is what the
campaign is for, and both answers are worth having: if the movement survives,
it is now carried by something a permutation destroys; if it does not, the
effect was never layer-specific and the number says so.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
DEFAULT_IN = REPO / "training/vectors/cortex-52d313c2"


def _load(path: Path) -> dict[str, object]:
    with np.load(path, allow_pickle=False) as handle:
        return {name: handle[name] for name in handle.files}


def _shared_direction(vectors: list[np.ndarray]) -> np.ndarray:
    """The one direction these layers agree on, as a unit vector.

    The first principal direction rather than the mean: the mean of vectors of
    different lengths is pulled by the longest, and what is wanted here is the
    axis they lie near, not their centre of mass.
    """
    stacked = np.stack([v / (np.linalg.norm(v) + 1e-12) for v in vectors])
    _, _, right = np.linalg.svd(stacked, full_matrices=False)
    axis = right[0]
    # Sign is arbitrary out of an SVD. Point it the way the layers do, so the
    # shared part that gets written is the one they actually carry.
    if float(np.dot(stacked.mean(axis=0), axis)) < 0.0:
        axis = -axis
    return axis.astype(np.float64)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vectors", type=Path, default=DEFAULT_IN)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--layers", type=int, nargs="*", default=[27, 31, 35, 39])
    parser.add_argument(
        "--keep-norm",
        action="store_true",
        default=True,
        help="rescale each residual to the length of the vector it came from, so "
        "the campaign's alpha means the same thing it did before",
    )
    arguments = parser.parse_args(argv)

    source = arguments.vectors
    target = arguments.out or source.parent / f"{source.name}-layer-specific"
    if not source.is_dir():
        print(f"no vectors at {source}", file=sys.stderr)
        return 1
    target.mkdir(parents=True, exist_ok=True)

    traits = sorted({p.name.split("_layer")[0] for p in source.glob("*.npz")})
    report: dict[str, object] = {
        "schema": "aura.steering.layer_specific.v1",
        "source": str(source.relative_to(REPO)),
        "written_to": str(target.relative_to(REPO)),
        "layers": list(arguments.layers),
        "written_at": time.time(),
        "traits": {},
    }

    for trait in traits:
        held: dict[int, dict[str, object]] = {}
        for layer in arguments.layers:
            path = source / f"{trait}_layer{layer}.npz"
            if path.exists():
                held[layer] = _load(path)
        if len(held) < 2:
            continue
        layers = sorted(held)
        raw = [np.asarray(held[layer]["v"], dtype=np.float64) for layer in layers]
        axis = _shared_direction(raw)

        before = _pairwise(raw)
        residuals = []
        for vector in raw:
            along = float(np.dot(vector, axis))
            residual = vector - along * axis
            length = float(np.linalg.norm(residual))
            if length <= 1e-9:
                residuals.append(residual)
                continue
            if arguments.keep_norm:
                residual = residual * (float(np.linalg.norm(vector)) / length)
            residuals.append(residual)
        after = _pairwise(residuals)

        for layer, residual, original in zip(layers, residuals, raw, strict=True):
            payload = dict(held[layer])
            payload["v"] = residual.astype(np.float32)
            payload["source"] = np.array("layer_specific_residual")
            payload["shared_fraction"] = np.array(
                float(abs(np.dot(original, axis)) / (np.linalg.norm(original) + 1e-12))
            )
            np.savez(target / f"{trait}_layer{layer}.npz", **payload)

        np.savez(
            target / f"{trait}_shared.npz",
            v=(axis * float(np.mean([np.linalg.norm(v) for v in raw]))).astype(np.float32),
            dimension=np.array(trait),
            source=np.array("shared_across_layers"),
            layers=np.array(layers),
        )
        report["traits"][trait] = {
            "layers": layers,
            "pairwise_cosine_before": before,
            "pairwise_cosine_after": after,
            "shared_fraction": [
                round(float(abs(np.dot(v, axis)) / (np.linalg.norm(v) + 1e-12)), 4)
                for v in raw
            ],
        }
        print(
            f"{trait:17s} cos {before['mean']:+.3f} -> {after['mean']:+.3f}   "
            f"shared {np.mean(report['traits'][trait]['shared_fraction']):.2f}"
        )

    out = target / "layer_specificity.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nwritten to {target}")
    return 0


def _pairwise(vectors: list[np.ndarray]) -> dict[str, float]:
    unit = np.stack([v / (np.linalg.norm(v) + 1e-12) for v in vectors])
    grid = unit @ unit.T
    off = grid[np.triu_indices(len(vectors), 1)]
    return {
        "min": round(float(off.min()), 4),
        "mean": round(float(off.mean()), 4),
        "max": round(float(off.max()), 4),
    }


if __name__ == "__main__":
    raise SystemExit(main())
