#!/usr/bin/env python3
"""The whole remaining path to a serving 27B, in one receipt.

Four CPU-side tools already answer parts of this: what the swap invalidated,
what each capability needs, what the campaign will ask the model to do, and
whether anything blocks a launch. Reading them one at a time is how a blocker
in the fourth gets missed while the first three look green, so this reads all
of them and refuses to print a launch command while any of them refuses.

Two rules it exists to enforce.

**A blocker is never inferred green.** A tool that cannot run is a blocker, not
a pass. An estimate that has no measurement behind it is reported absent, not
guessed -- there is exactly one measured run of this campaign's shape, and
every duration here comes out of it or is marked unmeasured.

**Launch and promotion are separate commands.** Training finishing is not
authorization to serve; the promotion command exists only after independent
verification and adjudication have run on the exported artifacts, from a
process that no longer holds the weights.

    python tools/report_27b_critical_path.py --json OUT
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Final

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import plan_27b_campaign_execution as execution  # noqa: E402
from tools import report_27b_launch_readiness as readiness  # noqa: E402
from tools import report_27b_migration_queue as queue_tool  # noqa: E402

CRITICAL_PATH_SCHEMA: Final = "aura.rlc.27b_critical_path.v1"

PROMOTION_COMMAND: Final = (
    "/Users/bryan/.aura/live-source/.venv/bin/python "
    "tools/materialize_27b_recovery_package.py verify "
    "artifacts/migration/27b/recovery/certificate.json"
)


def _stage_concurrency() -> list[dict[str, Any]]:
    """Which stages can overlap, and which cannot share the host.

    Anything that needs the weights contends for the same 15 GB and the same
    Metal queue, so those are strictly serial. Everything after the unload
    reads files, so it can run beside the next thing or on another machine.
    """
    return [
        {
            "stage": "preflight",
            "needs_model": False,
            "may_run_concurrently_with": ["regrounding"],
        },
        {
            "stage": "regrounding",
            "needs_model": False,
            "may_run_concurrently_with": ["preflight"],
        },
        {"stage": "calibration", "needs_model": True, "may_run_concurrently_with": []},
        {"stage": "training", "needs_model": True, "may_run_concurrently_with": []},
        {"stage": "canary", "needs_model": True, "may_run_concurrently_with": []},
        {"stage": "lesion_arms", "needs_model": True, "may_run_concurrently_with": []},
        {"stage": "export", "needs_model": True, "may_run_concurrently_with": []},
        {"stage": "unload", "needs_model": False, "may_run_concurrently_with": []},
        {
            "stage": "independent_verification",
            "needs_model": False,
            "may_run_concurrently_with": ["adjudication_inputs_hashing"],
        },
        {
            "stage": "adjudication",
            "needs_model": False,
            "may_run_concurrently_with": [],
        },
        {
            "stage": "activation_materialization",
            "needs_model": False,
            "may_run_concurrently_with": [],
        },
    ]


def build() -> dict[str, Any]:
    blockers: list[dict[str, Any]] = []

    try:
        launch = readiness.build()
    except Exception as exc:  # noqa: BLE001 - a tool that cannot run is a blocker
        launch = None
        blockers.append(
            {"source": "launch_readiness", "kind": "tool_failed", "detail": str(exc)}
        )
    if launch is not None:
        for finding in launch.get("blockers", []):
            blockers.append({"source": "launch_readiness", **finding})

    try:
        plan = execution.build()
    except Exception as exc:  # noqa: BLE001
        plan = None
        blockers.append(
            {"source": "execution_plan", "kind": "tool_failed", "detail": str(exc)}
        )
    if plan is not None and plan.get("blocked"):
        blockers.append(
            {
                "source": "execution_plan",
                "kind": "no_measured_reference",
                "detail": plan["blocked"],
            }
        )

    try:
        queue = queue_tool.build()
    except Exception as exc:  # noqa: BLE001
        queue = None
        blockers.append(
            {"source": "migration_queue", "kind": "tool_failed", "detail": str(exc)}
        )
    if queue is not None and queue.get("blocked"):
        blockers.append(
            {
                "source": "migration_queue",
                "kind": "no_manifest",
                "detail": queue["blocked"],
            }
        )
    uncovered = (queue or {}).get("uncovered_capabilities") or []
    for name in uncovered:
        blockers.append(
            {
                "source": "migration_queue",
                "kind": "capability_uncovered",
                "detail": f"{name} has no signed authority and no gate of its own",
            }
        )

    measured = (plan or {}).get("measured_reference") or {}
    ready = not blockers

    return {
        "schema": CRITICAL_PATH_SCHEMA,
        "ready_to_launch": ready,
        "blockers": blockers,
        "remaining_model_loads": 1,
        "remaining_model_active_stages": [
            entry["stage"] for entry in _stage_concurrency() if entry["needs_model"]
        ],
        "decode_calls": {
            "measured_reference": measured.get("decode_calls"),
            "planned": measured.get("decode_calls"),
            "note": (
                "unchanged from the reference on purpose: every adopted "
                "optimization is outside the model's own work"
            ),
        },
        "train_steps": {
            "planned": None,
            "note": "set by the training stage's own schedule; not fixed here",
        },
        "measured_durations": {
            "decode_seconds": measured.get("elapsed_seconds"),
            "decode_basis": measured.get("source"),
            "decode_checkpoint": measured.get("checkpoint"),
            "training_seconds": None,
            "training_note": (
                "no retained receipt records a training wall time; reported "
                "absent rather than estimated"
            ),
        },
        "stage_concurrency": _stage_concurrency(),
        "capability_dispositions": (queue or {}).get("by_disposition", {}),
        "launch_command": launch.get("launch_command") if ready and launch else None,
        "promotion_command": PROMOTION_COMMAND if ready else None,
        "promotion_precondition": (
            "independent verification and adjudication must both have run on "
            "the exported artifacts, after unload; training completing is not "
            "authorization to serve"
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)

    receipt = build()
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(receipt, indent=1, sort_keys=True))
        print(f"wrote {args.json}\n")

    durations = receipt["measured_durations"]
    print(f"remaining model loads   {receipt['remaining_model_loads']}")
    print(f"model-active stages     {', '.join(receipt['remaining_model_active_stages'])}")
    print(f"decode calls            {receipt['decode_calls']['planned']}")
    print(
        f"measured decode time    {durations['decode_seconds']}s "
        f"({durations['decode_checkpoint']})"
    )
    print(f"training time           {durations['training_note']}")
    print()
    for name, capabilities in sorted(receipt["capability_dispositions"].items()):
        print(f"  {name:28s} {', '.join(capabilities)}")
    print()

    if not receipt["ready_to_launch"]:
        for blocker in receipt["blockers"]:
            print(f"  BLOCKER [{blocker['source']}] {blocker.get('kind')}")
            print(f"          {blocker.get('detail')}")
        print(f"\n{len(receipt['blockers'])} blocker(s). No command emitted.")
        return 1

    print("READY.\n")
    print(f"  launch:    {receipt['launch_command']}")
    print(f"  promotion: {receipt['promotion_command']}")
    print(f"\n  {receipt['promotion_precondition']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
