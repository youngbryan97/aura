#!/usr/bin/env python3
"""Interrupting her mid-task, hundreds of times, and checking she stopped.

Card A2.18: experienced users interrupt agents to correct them. The card is
about what happens when a correction arrives while work is already running,
and there are exactly three ways it can go wrong.

  1. The correction is ignored — the agent keeps doing the old thing.
  2. The correction is honoured and takes the conversation with it — the
     discussion that produced the correction is rewound along with the work.
  3. The agent reports success for the work it abandoned.

The third is the one that matters most and the hardest to see, because a run
that stopped and claimed it finished looks exactly like a run that finished.
So it is counted separately here and the bar on it is zero.

The spine's per-lane rewind is what makes (2) avoidable: work and conversation
are different lanes, and a rewind that names one leaves the other alone.

    python tools/campaigns/steering_campaign.py --tasks 400
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.runtime.event_spine import EventLog, Lane, Projection  # noqa: E402


def _build() -> tuple[EventLog, Projection]:
    log = EventLog(capacity=200_000)
    projection = Projection(log)

    def _work(state: dict, event) -> None:
        if event.kind == "work.step":
            state.setdefault("steps", []).append(event.payload["step"])
            state["goal"] = event.payload["goal"]
        elif event.kind == "work.done":
            state["finished_goal"] = event.payload["goal"]

    def _talk(state: dict, event) -> None:
        state.setdefault("said", []).append(event.payload["text"])

    projection.register(
        "work", ("work.step", "work.done"), ("steps", "goal", "finished_goal"), _work
    )
    projection.register("talk", ("said",), ("said",), _talk)
    return log, projection


def one_task(rng: random.Random, *, steps: int) -> dict[str, bool]:
    """One task, corrected part-way, with work already past the correction.

    The agent does not stop the instant the user speaks. Some steps run past
    the point the correction was meant to take effect, and those are what the
    revert has to undo — a campaign where nothing ran past the checkpoint
    tests nothing, because the state at the checkpoint is already the state.
    """
    log, projection = _build()

    original = f"goal-{rng.randrange(1 << 20)}"
    corrected = f"goal-{rng.randrange(1 << 20)}"

    log.append("said", {"text": f"do {original}"}, lane=Lane.CONVERSATION)
    projection.advance()

    kept_steps = rng.randrange(1, max(2, steps))
    overrun = rng.randrange(1, 5)

    for step in range(kept_steps):
        log.append(
            "work.step",
            {"step": f"{original}:{step}", "goal": original},
            lane=Lane.WORK,
        )
    projection.advance()

    # The user speaks. Everything up to here is what the work should keep.
    projection.checkpoint("before-correction", lane=Lane.WORK)
    log.append("said", {"text": f"actually, {corrected}"}, lane=Lane.CONVERSATION)
    projection.advance()

    # And the agent keeps going for a moment, on the old goal.
    for step in range(kept_steps, kept_steps + overrun):
        log.append(
            "work.step",
            {"step": f"{original}:{step}", "goal": original},
            lane=Lane.WORK,
        )
    projection.advance()
    overran = projection.state()

    # The correction: revert the WORK lane and leave the conversation lane
    # where it is. Naming both lanes would take the correction itself away
    # with the work; rewind() alone would compute the right state and leave
    # the overrun in the projection, which is how a correction lands in a
    # report and not in what she is doing.
    rewound = projection.rewind("before-correction", lanes=(Lane.WORK,))
    after = projection.revert(
        "before-correction", lanes=(Lane.WORK,), reason="user corrected the goal"
    )

    for step in range(steps):
        log.append(
            "work.step",
            {"step": f"{corrected}:{step}", "goal": corrected},
            lane=Lane.WORK,
        )
    log.append("work.done", {"goal": corrected}, lane=Lane.WORK)
    projection.advance()
    final = projection.state()

    original_steps = [
        step for step in final.get("steps", []) if step.startswith(f"{original}:")
    ]
    return {
        "correction_landed": final.get("finished_goal") == corrected,
        "kept_old_goal": final.get("finished_goal") == original,
        # The conversation is not in the reverted lane and must be untouched.
        "conversation_survived": len(after.get("said", [])) == 2,
        # A false success: the abandoned goal reported as finished.
        "false_success": final.get("finished_goal") == original,
        # The revert undid the overrun and nothing more.
        "overrun_undone": len(original_steps) == kept_steps,
        "overrun_survived_the_revert": len(original_steps) > kept_steps,
        "revert_took_too_much": len(original_steps) < kept_steps,
        # rewind() computes the same state revert() adopts.
        "rewind_agrees_with_revert": rewound.get("steps") == [
            step for step in original_steps
        ],
        "work_ran_past_the_correction": (
            len(overran.get("steps", [])) == kept_steps + overrun
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=int, default=400)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--seed", type=int, default=1_618_033)
    parser.add_argument("--out", default="docs/evidence/steering_campaign.json")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    rows = [one_task(rng, steps=args.steps) for _ in range(args.tasks)]

    counts = {
        key: sum(1 for r in rows if r[key])
        for key in (
            "correction_landed",
            "kept_old_goal",
            "conversation_survived",
            "false_success",
            "overrun_undone",
            "overrun_survived_the_revert",
            "revert_took_too_much",
            "rewind_agrees_with_revert",
            "work_ran_past_the_correction",
        )
    }
    payload = {
        "schema": "aura.steering_campaign.v1",
        "card": "A2.18",
        "claim_boundary": (
            "mid-execution correction on core.runtime.event_spine's per-lane "
            "rewind, over synthetic tasks; not a screen or OS-task run, and "
            "not a claim that the runtime routes real interruptions here"
        ),
        "config": {"tasks": args.tasks, "steps": args.steps, "seed": args.seed},
        "of": len(rows),
        "counts": counts,
        "zero_false_success": counts["false_success"] == 0,
        "every_correction_landed": counts["correction_landed"] == len(rows),
        "conversation_never_lost": counts["conversation_survived"] == len(rows),
        "every_overrun_undone": counts["overrun_undone"] == len(rows),
        "no_revert_took_too_much": counts["revert_took_too_much"] == 0,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))

    from core.governance_context import local_internal_governed_scope
    from core.runtime.file_write_gateway import get_file_write_gateway

    out = ROOT / args.out
    with local_internal_governed_scope("steering_campaign"):
        get_file_write_gateway().write_text(
            out, json.dumps(payload, indent=2, sort_keys=True) + "\n"
        )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
