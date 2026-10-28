#!/usr/bin/env python3
"""Exhaust a finite intervention blind spot and measure the subset-plan repair."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error("choose a new output path")

    from core.runtime.atomic_writer import atomic_write_text
    from core.subject.coalitions import plan_coalitions

    units = [(str(i), None) for i in range(4)]
    plan = plan_coalitions(units, budget=10, seed=97)
    traditional = {0, 15, *(1 << i for i in range(4)), *(15 ^ (1 << i) for i in range(4))}
    hidden = sorted(set(range(16)) - traditional)
    extended = traditional | {sum(1 << i for i in indices) for indices in plan.subsets}
    repaired = missed = 0
    for mask in range(1, 1 << len(hidden)):
        nonzero = {value for bit, value in enumerate(hidden) if mask & (1 << bit)}
        missed += int(not (nonzero & traditional))
        repaired += int(bool(nonzero & extended))
    complete = extended == set(range(16)) and missed == repaired == 63

    wide = plan_coalitions([(str(i), None) for i in range(96)], budget=128, seed=97)
    hits = [i for i, subset in enumerate(wide.subsets)
            if 0 in subset and 1 in subset and 2 not in subset and 3 not in subset]
    draws = math.ceil(math.log(.01) / math.log1p(-2 ** -4))
    output = {
        "constructed_worlds": True, "actual_aura_transition_witness": False,
        "verified": complete and bool(hits),
        "four_bit_truth_tables": {
            "baseline_anchors": ["0000", "1111"],
            "single_and_whole_states": sorted(traditional),
            "missed_states": hidden,
            "undetected_nonconstant_functions": missed,
            "detected_with_complete_subsets": repaired,
            "null_function_stays_zero": True,
            "plan": plan.description(),
        },
        "ninety_six_unit_masked_cause": {
            "effect": "a*b*(1-c)*(1-d); the other 92 units are irrelevant",
            "seed": 97, "draws": 128, "detecting_masks": hits,
        },
        "conditional_sampling_math": {
            "assumptions": "A deterministic effect for at least one assignment of r relevant unit masks, every other mask irrelevant, and every single and whole swap already inert. Donor changes must implement those assignments and the chosen horizon and observations must reveal the effect.",
            "per_draw_probability_lower_bound": "2**(-r)",
            "miss_probability_upper_bound": "(1 - 2**(-r))**draws",
            "r": 4, "draws_for_miss_below_one_percent": draws,
            "miss_bound_at_that_count": (1 - 2 ** -4) ** draws,
            "sampling": "Uniform proper nonsingle subsets without replacement; removal of already inert masks cannot reduce the effect fraction, and sampling without replacement cannot increase this miss bound.",
        },
        "limits": "This finite proof is about specified Boolean mechanisms and donors. It is not universal closure, proof of an Aura defect, or power calibration for arbitrary interactions. Complete subset enumeration is exponential; 96 units are sampled, not exhausted.",
        "source_sha256": {name: hashlib.sha256((REPO / name).read_bytes()).hexdigest()
                          for name in ("core/subject/coalitions.py", "tools/run_interventional_closure.py",
                                       "tools/prove_intervention_subset_coverage.py")},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(args.out, json.dumps(output, indent=2) + "\n")
    print(json.dumps({"verified": output["verified"], "hidden_functions": missed,
                      "repaired_functions": repaired, "wide_detecting_masks": len(hits),
                      "draws_for_one_percent": draws}, indent=2))
    return 0 if output["verified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
