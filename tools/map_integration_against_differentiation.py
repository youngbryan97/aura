#!/usr/bin/env python3
"""Whether one system can clear both bars, measured rather than argued.

The battery asks for irreducibility above 0.05 and, in its original one-sided
reading, an effective dimension of at least four tenths of the width. Those two
pull opposite ways by construction: effective dimension is the participation
ratio of the state correlation spectrum, so coupling lowers it, because that is
what coupling is. A system with no coupling has the most dimensions and the
least integration; a system that is one broadcast has the most integration and
almost no repertoire.

The question is whether there is a middle where both hold. This sweeps coupling
strength through the recurrent reference and measures both at each step, so the
answer is a curve rather than an intuition. If no point on it clears both, the
one-sided reading is internally impossible for the architecture the test was
designed around, and that is a fact about the test.

    python tools/map_integration_against_differentiation.py
    python tools/map_integration_against_differentiation.py --json out.json

The sweep runs on the synthetic architectures, not on Aura. A toy of forty
columns is the right place to ask whether a bar is satisfiable at all, because
its coupling is a number that can be turned.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

#: Coupling strengths swept through the recurrent reference. Wide enough at
#: both ends that the uncoupled and the collapsed cases are both on the curve.
STRENGTHS: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.45, 0.6, 0.8, 1.0, 1.4, 2.0)

#: Draws per point, so a curve is not ten single instantiations of a random
#: weight matrix.
DRAWS: int = 3
STEPS: int = 900


def _point(strength: float, *, seed: int, steps: int) -> dict[str, float]:
    from core.subject.differentiation import effective_dimension
    from core.subject.irreducibility import phi_do
    from core.subject.nulls import architecture, toy_recording

    phis: list[float] = []
    ratios: list[float] = []
    dims: list[float] = []
    tops: list[float] = []
    for draw in range(DRAWS):
        system = architecture("recurrent", seed=seed + draw, strength=strength)
        recording = toy_recording(system, steps=steps, seed=seed + draw)
        phis.append(float(phi_do(recording).phi))
        spectrum = effective_dimension(recording)
        ratios.append(float(spectrum.normalised))
        dims.append(float(spectrum.d_eff))
        tops.append(float(spectrum.top_share))
    return {
        "strength": strength,
        "phi_do": round(float(np.median(phis)), 5),
        "phi_min": round(min(phis), 5),
        "phi_max": round(max(phis), 5),
        "d_eff": round(float(np.median(dims)), 4),
        "d_eff_normalised": round(float(np.median(ratios)), 4),
        "largest_component_share": round(float(np.median(tops)), 4),
    }


def sweep(*, seed: int = 7, steps: int = STEPS) -> dict[str, Any]:
    from core.subject.battery import THRESHOLDS

    rows = [_point(value, seed=seed, steps=steps) for value in STRENGTHS]
    phi_bar = float(THRESHOLDS["phi_do"])
    ratio_bar = float(THRESHOLDS["d_eff_normalised"])
    share_bar = float(THRESHOLDS["component_share"])

    for row in rows:
        row["clears_phi"] = row["phi_do"] > phi_bar
        # The original one-sided reading: a large share of the width live.
        row["clears_the_one_sided_reading"] = row["d_eff_normalised"] >= ratio_bar
        # And the two-sided one the battery applies instead: the ratio under
        # the bar, with a dominance limit beneath it.
        row["clears_the_two_sided_reading"] = (
            row["d_eff_normalised"] < ratio_bar
            and row["largest_component_share"] < share_bar
        )
        row["clears_both_one_sided"] = row["clears_phi"] and row["clears_the_one_sided_reading"]
        row["clears_both_two_sided"] = row["clears_phi"] and row["clears_the_two_sided_reading"]

    both_one_sided = [r for r in rows if r["clears_both_one_sided"]]
    both_two_sided = [r for r in rows if r["clears_both_two_sided"]]
    return {
        "bars": {
            "phi_do": phi_bar,
            "d_eff_normalised": ratio_bar,
            "component_share": share_bar,
        },
        "draws_per_point": DRAWS,
        "steps": steps,
        "curve": rows,
        "a_point_clears_both_one_sided": bool(both_one_sided),
        "a_point_clears_both_two_sided": bool(both_two_sided),
        "strengths_clearing_both_one_sided": [r["strength"] for r in both_one_sided],
        "strengths_clearing_both_two_sided": [r["strength"] for r in both_two_sided],
        "verdict": _verdict(bool(both_one_sided), bool(both_two_sided)),
    }


def _verdict(one_sided: bool, two_sided: bool) -> str:
    if one_sided:
        return (
            "some coupling strength clears irreducibility and the one-sided "
            "differentiation reading at once, so the original conjunction is "
            "satisfiable for this architecture"
        )
    if two_sided:
        return (
            "no coupling strength clears irreducibility and the one-sided "
            "differentiation reading at once, so that conjunction is internally "
            "impossible for the architecture the test was designed around. The "
            "two-sided reading — the ratio under the bar with a dominance limit "
            "beneath it — is satisfiable, and it is what the battery applies."
        )
    return (
        "no coupling strength clears irreducibility together with either "
        "differentiation reading, which is a finding about the estimator rather "
        "than about the architectures"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=Path)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--steps", type=int, default=STEPS)
    args = parser.parse_args()

    out = sweep(seed=args.seed, steps=args.steps)
    bars = out["bars"]
    print(
        f"{'strength':>9} {'phi':>8} {'d_eff':>8} {'ratio':>7} {'top':>7}  "
        f"{'phi>' + str(bars['phi_do']):>9} {'ratio>=' + str(bars['d_eff_normalised']):>11}  both"
    )
    for row in out["curve"]:
        print(
            f"{row['strength']:9.2f} {row['phi_do']:8.4f} {row['d_eff']:8.3f} "
            f"{row['d_eff_normalised']:7.4f} {row['largest_component_share']:7.4f}  "
            f"{str(row['clears_phi']):>9} {str(row['clears_the_one_sided_reading']):>11}  "
            f"{str(row['clears_both_one_sided'])}"
        )
    print()
    print(out["verdict"])

    if args.json:
        args.json.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
