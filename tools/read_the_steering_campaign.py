#!/usr/bin/env python3
"""Turn a campaign result into a verdict, so the verdict is re-derivable.

``campaign_verdict.json`` was written by hand beside the result it describes.
Every number in it was real, and none of it could be recomputed — which is the
one property a record of a negative result has to have, because a negative is
exactly what somebody will want to re-check.

This reads the result, replays the scoring through the same independent path
the campaign uses, and writes the verdict. It decides nothing: the pass
predicates live in ``core/evaluation/steering_ab.py`` and the replay in
``core/evaluation/caa_causal_evaluation.py``. What it adds is that the verdict
file and the samples underneath it cannot drift apart.

    python tools/read_the_steering_campaign.py --result <path> --out <path>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from core.evaluation.caa_causal_evaluation import replay_campaign  # noqa: E402


def _means(result: dict) -> dict[str, float]:
    scores = result.get("target_scores") or {}
    return {
        name: round(sum(values) / len(values), 4)
        for name, values in sorted(scores.items())
        if values
    }


def read(result: dict) -> dict:
    """The verdict this result supports, and nothing beyond it."""
    replay = replay_campaign(result)
    analysis = replay.get("analysis") or {}
    trials = int(result.get("n_trials_per_task") or 0)
    tasks = len(result.get("held_out_tasks") or ())
    return {
        "schema": "aura.caa.campaign_verdict.v2",
        "alpha": result.get("alpha"),
        "model_descriptor_sha256": result.get("model_descriptor_sha256"),
        "vectors": result.get("vectors") or "",
        "samples_per_condition": trials * tasks,
        "condition_means": _means(result),
        "treatment_successes": replay["treatment_successes"],
        "matched_control_successes": replay["matched_control_successes"],
        "lesion_successes": replay["lesion_successes"],
        "no_regression": replay["no_regression"],
        "causal_effect_positive": replay["causal_effect_positive"],
        "passes_adversarial_control": replay["passes_adversarial_control"],
        "unmet_requirements": replay["unmet_requirements"],
        # A different question from whether steering beats the words, and
        # deliberately not part of the pass predicate.
        "adds_to_text": analysis.get("adds_to_text"),
        "task_target_deltas": replay["task_target_deltas"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result",
        type=Path,
        default=REPO / "artifacts/migration/27b/recovery/campaign_result.json",
    )
    parser.add_argument("--out", type=Path, default=None)
    arguments = parser.parse_args()

    result = json.loads(arguments.result.read_text(encoding="utf-8"))
    verdict = read(result)
    out = arguments.out or arguments.result.with_name(
        arguments.result.stem.replace("campaign_result", "campaign_verdict") + ".json"
    )
    out.write_text(json.dumps(verdict, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"alpha {verdict['alpha']}  n={verdict['samples_per_condition']} per condition")
    for name, mean in verdict["condition_means"].items():
        print(f"  {name:24s} {mean:.4f}")
    print(f"treatment wins       {verdict['treatment_successes']}")
    print(f"matched no-op wins   {verdict['matched_control_successes']}")
    print(f"lesion wins          {verdict['lesion_successes']}")
    print(f"no regression        {verdict['no_regression']}")
    print(f"adds to text         {verdict['adds_to_text']}")
    print(f"causal effect        {verdict['causal_effect_positive']}")
    print(f"unmet                {verdict['unmet_requirements'] or 'none'}")
    print(f"\nwritten to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
