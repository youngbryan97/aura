"""tools/agi_gauntlet/bundle.py — what somebody else needs to run this.

Independent reproducibility has two halves and only one of them is code. The
half that is code: the freeze, the environments, the receipts, and enough of
the protocol that a run elsewhere is the same run. The half that is not: a
team that builds its own task families after this freeze, never sees the ones
here, and reports what it found.

The bundle carries the first and states the second, including the part this
harness cannot supply — human baselines. A gate whose pass condition is
"roughly competent-human" and whose harness has never seen a human is a gate
comparing a number to an assumption, so the slots are declared and left
empty rather than filled with a guess.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tools.agi_gauntlet.protocol import Freeze, take_the_freeze

__all__ = ["WHAT_A_HUMAN_WOULD_SCORE", "the_bundle", "write_the_bundle"]


#: The gates whose pass condition mentions a person, and what baseline each
#: needs. Empty on purpose: a number here that nobody measured would turn
#: every one of these into a comparison against an assumption.
WHAT_A_HUMAN_WOULD_SCORE: dict[str, dict[str, Any]] = {
    "fluid intelligence": {
        "needs": "the share of these sealed rules a competent adult gets, "
                 "with the same three examples and no time limit",
        "measured": None,
    },
    "interactive novel-world learning": {
        "needs": "completion and moves-to-goal for a person given the same "
                 "world, the same acts, the same lives and no instructions",
        "measured": None,
    },
    "learning from experience": {
        "needs": "a person's curve over the same twelve episodes",
        "measured": None,
    },
    "new-skill acquisition": {
        "needs": "a person's apprenticeship curve in the same sealed "
                 "environments under the same access — the half of this gate "
                 "the offline run cannot supply",
        "measured": None,
    },
    "cross-domain transfer": {
        "needs": "how often a person carries the structure across, on the "
                 "same pairs and the same negative controls",
        "measured": None,
    },
    "planning under novelty": {
        "needs": "a person's recovery rate after the same silent change",
        "measured": None,
    },
    "robustness": {
        "needs": "what a person does with one example instead of three: the "
                 "gate asks for refusal rather than a guess, and whether that "
                 "is the human answer is a measurement",
        "measured": None,
    },
    "multimodal integration": {
        "needs": "human completion on the same mixed-modality tasks",
        "measured": None,
    },
    "broad everyday competence": {
        "needs": "GAIA's own measured human baseline was 92%; a private set "
                 "needs its own",
        "measured": None,
    },
    "computer-world competence": {
        "needs": "OSWorld-Human action counts on the same tasks",
        "measured": None,
    },
    "long-horizon autonomy": {
        "needs": "how long a competent person takes on each task, which is "
                 "what the gate's thresholds are stated in",
        "measured": None,
    },
    "social and instructional intelligence": {
        "needs": "humans onboarding into the same fictional organisation",
        "measured": None,
    },
}


def the_bundle(freeze: Freeze | None = None) -> dict[str, Any]:
    """Everything an outside evaluator needs, and what they still have to do."""

    from tools.agi_gauntlet.ablations import THE_LESIONS
    from tools.agi_gauntlet.gates import THE_GATES

    held = freeze if freeze is not None else take_the_freeze()
    return {
        "freeze": held.to_dict(),
        "how_to_run": [
            "git checkout <commit>",
            "confirm the working tree is clean; a dirty tree is not a freeze",
            "python tools/run_agi_gauntlet.py",
            "compare the receipts under artifacts/agi_gauntlet/",
        ],
        "what_reproduces_exactly": [
            "every sealed environment, because each is a function of the seed",
            "every answer on them, because the search is deterministic and "
            "anything that draws is seeded",
        ],
        "what_does_not": [
            "wall-clock timings, and the one search bounded by a clock",
        ],
        "gates": [one.to_dict() for one in THE_GATES],
        "lesions": [
            {
                "name": one.name,
                "removes": one.what_it_removes,
                "runs_here": one.can_be_applied,
                "needs": one.needs,
            }
            for one in THE_LESIONS
        ],
        "human_baselines": WHAT_A_HUMAN_WOULD_SCORE,
        "still_needed": [
            "task families built after this freeze by somebody who has not "
            "seen the ones here",
            "the same weights in a plain read-decide-act scaffold, with the "
            "same tools, token budget and compute budget",
            "human baselines for every gate whose pass condition mentions a "
            "person",
            "a second group repeating the first group's run",
        ],
    }


def write_the_bundle(into: Path, freeze: Freeze | None = None) -> Path:
    into.mkdir(parents=True, exist_ok=True)
    where = into / "reproduction_bundle.json"
    where.write_text(json.dumps(the_bundle(freeze), indent=2), encoding="utf-8")
    return where
