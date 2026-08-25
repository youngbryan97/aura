#!/usr/bin/env python3
"""Does moving one named dimension of z_Aura move anything downstream?

    python tools/endogenous_causal_battery.py

Runs offline by default: it measures what happens between the state and the
decode-time bias, which needs no model at all. Every effect is reported beside
the matched nulls that make it a measurement — the same-sized intervention on
peer dimensions, and the same ablation on peer channels.

The text arm, where an intervention is checked against what she actually says,
is deliberately not here. It needs the resident model, and this machine has one
model serving conversation; loading a second is not a thing a battery script
gets to do. ``core.brain.llm.endogenous_intervention.causal_text_experiment``
runs that arm against a client the runtime already holds.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402

from core.brain.llm.cognitive_code import read_code  # noqa: E402
from core.brain.llm.endogenous_absorption import (  # noqa: E402
    Proposal,
    arbitrate,
)
from core.brain.llm.endogenous_intervention import (  # noqa: E402
    channel_influence_map,
    measure_ablation,
    measure_contrast,
    measure_intervention,
    sweep_dimension,
)
from core.brain.llm.endogenous_state import (  # noqa: E402
    CHANNELS,
    FEATURES,
    STATE_DIM,
    EndogenousState,
    assemble_state,
    describe_layout,
    layout_digest,
)
from core.brain.llm.endogenous_vocab_head import (  # noqa: E402
    EndogenousVocabHead,
    HeadUnusableError,
    head_directory,
)

#: The dimension the architecture argument keeps coming back to: change how
#: sure she is, and see whether the readout and the words follow.
PRIMARY_FEATURE = "uncertainty.confidence"


def _synthetic_state() -> EndogenousState:
    """A fully-answered state, for when no runtime is up to read one from.

    Marked as constructed in the receipt. A battery run against a synthetic
    state measures the pathway; it does not measure Aura.
    """
    rng = np.random.default_rng(11)
    values = {f.name: float(rng.uniform(f.low, f.high)) for f in FEATURES}
    return assemble_state(overrides=values)


def _load_head() -> tuple[EndogenousVocabHead | None, str]:
    try:
        return EndogenousVocabHead.load(head_directory()), "loaded"
    except HeadUnusableError as exc:
        return None, str(exc)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="artifacts/endogenous_language")
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="use a fully-answered constructed state when no runtime is up",
    )
    args = parser.parse_args()

    live = assemble_state()
    constructed = args.synthetic or live.coverage <= 0.0
    state = _synthetic_state() if constructed else live
    head, head_reason = _load_head()

    report: dict[str, Any] = {
        "ran_at": time.time(),
        "layout": layout_digest(),
        "state_dim": STATE_DIM,
        "state_is_constructed": constructed,
        "state_coverage": round(state.coverage, 4),
        "live_channels": list(state.live_channels),
        "head": {"present": head is not None, "reason": head_reason},
        "layout_features": describe_layout(),
    }

    report["code_before"] = read_code(state, include_organ_lines=False).render()
    report["primary_intervention"] = measure_intervention(
        state, PRIMARY_FEATURE, 0.95, head=head
    ).as_dict()
    report["primary_contrast"] = measure_contrast(
        state, PRIMARY_FEATURE, 0.05, 0.95, head=head
    ).as_dict()
    report["primary_sweep"] = sweep_dimension(
        state, PRIMARY_FEATURE, [0.05, 0.25, 0.5, 0.75, 0.95], head=head
    )
    report["channel_ablations"] = [
        measure_ablation(state, channel, head=head).as_dict() for channel in CHANNELS
    ]
    if head is not None:
        report["channel_influence"] = channel_influence_map(state, head)

    # Arbitration has to be causal too: the same proposal, two states, and the
    # verdict has to follow the state rather than the proposal.
    proposal = Proposal(
        summary="commit to the first explanation and drop the current goal",
        asserted_confidence=0.95,
        abandons_active_goal=True,
        requires_action=True,
    )
    unsure = state.do(
        **{
            "uncertainty.confidence": 0.1,
            "uncertainty.evidence_support": 0.1,
            "goal.active": 1.0,
            "goal.priority": 0.95,
        }
    )
    settled = state.do(
        **{
            "uncertainty.confidence": 0.95,
            "uncertainty.evidence_support": 0.9,
            "goal.active": 0.0,
            "goal.priority": 0.0,
        }
    )
    report["arbitration"] = {
        "proposal": proposal.as_dict(),
        "under_unsure_state": arbitrate(proposal, unsure).as_dict(),
        "under_settled_state": arbitrate(proposal, settled).as_dict(),
        "verdict_follows_state": (
            arbitrate(proposal, unsure).decision != arbitrate(proposal, settled).decision
        ),
    }

    from core.runtime.file_write_gateway import get_file_write_gateway

    target = Path(args.out) / "causal_battery.json"
    get_file_write_gateway().write_text(
        target,
        json.dumps(report, indent=2, sort_keys=True, default=str) + "\n",
        source="endogenous_causal_battery",
    )
    print(json.dumps({k: report[k] for k in (
        "state_is_constructed", "state_coverage", "head", "arbitration",
    )}, indent=2, default=str))
    print(f"\nFull receipt: {target}")
    if head is None:
        print(
            "No head on disk, so the bias half of every effect reads zero. "
            "That is the pathway waiting for a fit, not a failure."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
