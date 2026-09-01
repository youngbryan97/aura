#!/usr/bin/env python3
"""tools/evidence_report.py — everything Aura currently claims, and what it rests on.

The scientific machinery — the claim ladder, the experiment registry, the
parameter registry, organ accounting, the neuroscience register, the capability
board, replication packs — each answers one question. This asks all of them at
once and prints the answer as one picture, because the interesting reading is
across them: a claim at CAUSAL with no compute-matched arm, a parameter marked
FITTED with n=1 holding up a calibration claim, an organ classified
load-bearing whose effect vanishes when the compute is given back.

It exits non-zero on a contradiction rather than on a low number. A young
system with few claims is fine; a system asserting more than it has established
is the thing this is for.

    python tools/evidence_report.py           # print the picture
    python tools/evidence_report.py --json    # machine-readable
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def campaign_surface() -> dict:
    """Which campaigns have been run, and what each would establish if it were.

    Every entry here is a measurement that only exists after somebody runs it.
    Reporting them as available rather than as done is the point: the reviews'
    central finding is that the architecture is ahead of its evidence, and a
    surface that hides which measurements are outstanding would hide exactly
    that.
    """
    from core.science.baseline_portfolio import REQUIRED, BaselineKind
    from core.science.capability_board import Capability
    from core.science.continual_metrics import DEFAULT_FORGETTING_BUDGET
    from core.science.developmental_campaign import MIN_BLOCKS
    from core.science.environment_bench import RETENTION_PER_DECADE
    from core.science.redteam_ledger import RedTeamLedger
    from core.science.replication_pack import ReplicationRegistry
    from core.science.retrieval_latency import MIN_OBSERVATIONS
    from core.world_model.prediction_quality import CalibrationCurve

    return {
        "developmental_campaign": {
            "establishes": "whether living through the earlier tasks helped",
            "needs": f"{MIN_BLOCKS} blocks, a reset arm and a lesion arm",
        },
        "capability_board": {
            "establishes": "whether the architecture beats the model it is built on",
            "needs": f"a cortex-only arm on each of {len(list(Capability))} capabilities",
        },
        "baseline_portfolio": {
            "establishes": "value over the strongest simple alternative",
            "needs": ", ".join(k.value for k in REQUIRED),
        },
        "environment_bench": {
            "establishes": "whether the same code works in an unseen world",
            "needs": f"held-out families and retention per decade above {RETENTION_PER_DECADE}",
        },
        "continual_metrics": {
            "establishes": "what learning the new thing cost the old ones",
            "needs": f"old tasks re-run after each block, budget {DEFAULT_FORGETTING_BUDGET}",
        },
        "retrieval_latency": {
            "establishes": "an Aura-native latency law",
            "needs": f"{MIN_OBSERVATIONS} recalls with their work counted",
        },
        "world_model_calibration": {
            "establishes": "whether the world model should be believed",
            "needs": f"{CalibrationCurve().__class__.__name__} over predicted against observed",
        },
        "replication": {
            "establishes": "that a result survives leaving this machine",
            "needs": "a sealed pack and one run on other hardware",
            "registry": ReplicationRegistry().report(),
        },
        "red_team": {
            "establishes": "that findings fall and do not come back",
            "needs": "findings recorded per release with pinned regression tests",
            "ledger": RedTeamLedger().trend(),
        },
    }


def gather() -> dict:
    from core.cognition.library_compression import LibraryCompressor
    from core.cognition.operator_invention import OperatorKernel
    from core.learning.shadow_archive import ShadowArchive
    from core.science.claim_ladder import Rung, get_ladder
    from core.science.experiment_registry import get_experiment_registry
    from core.science.neuro_reference import get_neuro_reference
    from core.science.parameter_registry import get_parameter_registry
    from core.science.organ_accounting import get_organ_accounting

    ladder = get_ladder()
    parameters = get_parameter_registry()
    return {
        "claims": ladder.audit(),
        "claim_detail": [c.to_dict() for c in ladder.claims()],
        "experiments": get_experiment_registry().report(),
        "parameters": parameters.report(),
        "organs": get_organ_accounting().report(),
        "biological_mappings": get_neuro_reference().audit(),
        "campaigns": campaign_surface(),
        "shadow_search": {"available": ShadowArchive.__name__, "archives": 0},
        "library": {"compressor": LibraryCompressor.__name__,
                    "operator_kernel": OperatorKernel.__name__},
        "highest_rung_claimed": max(
            (int(c.rung) for c in ladder.claims() if c.rung), default=0
        ),
        "rungs": {rung.name.lower(): rung.question for rung in Rung},
    }


def contradictions(picture: dict) -> list[str]:
    """Where the registries disagree with each other. This is the useful output."""
    problems: list[str] = []
    if picture["claims"]["degraded"]:
        problems.append(
            f"{len(picture['claims']['degraded'])} claim(s) rest on artifacts that "
            "have been deleted"
        )
    unidentifiable = picture["parameters"]["unidentifiable"]
    if unidentifiable:
        problems.append(
            "fitted parameters the data could not have chosen between: "
            + ", ".join(unidentifiable)
        )
    mislabelled = [
        organ for organ, names in picture["organs"]["by_classification"].items()
        if organ in ("compute_not_computation", "presence_not_content") and names
    ]
    if mislabelled:
        problems.append(
            "organs kept for the wrong reason: "
            + ", ".join(
                f"{k}={picture['organs']['by_classification'][k]}" for k in mislabelled
            )
        )
    metaphors = picture["biological_mappings"]["metaphor_only"]
    if metaphors:
        problems.append("biological names licensing nothing: " + ", ".join(metaphors))
    if picture["experiments"]["refused"] and not picture["experiments"]["experiments"]:
        problems.append("every experiment offered was refused as malformed")
    return problems


def render(picture: dict) -> str:
    lines = ["Aura evidence report", "=" * 60, ""]
    claims = picture["claims"]
    lines.append(f"Claims: {claims['claims']}  " + ", ".join(
        f"{k}={v}" for k, v in claims["by_rung"].items()
    ))
    lines.append(f"  at or above CAUSAL: {claims['at_or_above_causal']}")
    lines.append(f"  at or above USEFUL: {claims['at_or_above_useful']}")
    lines.append("")
    for claim in picture["claim_detail"]:
        lines.append(f"  [{claim['rung']}] {claim['statement']}")
        lines.append(f"      boundary: {claim['boundary']}")
    lines.append("")
    parameters = picture["parameters"]
    lines.append(f"Parameters: {parameters['parameters']}  " + ", ".join(
        f"{k}={v}" for k, v in parameters["by_kind"].items()
    ))
    lines.append(f"  able to support a calibration claim: {parameters['can_support_calibration']}")
    lines.append("")
    mappings = picture["biological_mappings"]
    lines.append(f"Biological mappings: {mappings['mappings']}  " + ", ".join(
        f"{k}={v}" for k, v in mappings["by_grade"].items()
    ))
    lines.append("")
    experiments = picture["experiments"]
    lines.append(
        f"Experiments: {experiments['experiments']} recorded, "
        f"{experiments['refused']} refused as malformed"
    )
    organs = picture["organs"]
    lines.append(f"Organs accounted for: {organs['organs']}")
    for classification, names in organs["by_classification"].items():
        lines.append(f"  {classification}: {len(names)}")
    lines.append("")
    lines.append("Campaigns outstanding")
    for name, entry in sorted(picture["campaigns"].items()):
        lines.append(f"  {name}: establishes {entry['establishes']}")
        lines.append(f"      needs {entry['needs']}")
    lines.append("")
    problems = contradictions(picture)
    if problems:
        lines.append("CONTRADICTIONS")
        lines.extend(f"  - {p}" for p in problems)
    else:
        lines.append("No contradiction between the registries.")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    picture = gather()
    problems = contradictions(picture)
    print(json.dumps(picture, indent=2) if args.json else render(picture))
    if problems:
        print(f"\nevidence-report: {len(problems)} contradiction(s)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
