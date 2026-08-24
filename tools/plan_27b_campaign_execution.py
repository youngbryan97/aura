#!/usr/bin/env python3
"""What the recovery campaign will actually ask the model to do, counted.

Wall-clock is an engineering objective here, and the way to pursue it without
damaging the science is to count the work rather than guess at it. Everything
below is either read from a retained receipt or derived from the code that will
run. Nothing is estimated, and an optimization whose benefit cannot be counted
is listed as unmeasured rather than claimed.

Three things this exists to prevent.

**Claiming an optimization that is already there.** Prompt-prefix caching is
implemented: in CP566 the ordinary arm prefilled zero tokens on every task
because its prefix was already resident. Reporting that as a saving would be
inventing a speedup out of existing behaviour.

**Optimizing away load-bearing work.** The decode loop retries when a response
fails serialization, and CP566's retries concentrate in the two arms designed
to fail -- 32 extra decodes on the coefficient lesion, 29 on the wrong-state
control, 3 on treatment. Those retries are the lesion arms behaving correctly.
Removing them would change what the controls measure.

**Buying speed with arm inequality.** ``_arm_order`` counterbalances the order
of arms per task, which is why one arm's cache state cannot systematically
advantage another. Batching decodes by arm would destroy exactly the
interleaving that counterbalancing exists to create, so it is rejected here
rather than implemented behind a flag.

    python tools/plan_27b_campaign_execution.py --json OUT
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

from tools.export_active_descriptor import installation_root  # noqa: E402

EXECUTION_PLAN_SCHEMA: Final = "aura.rlc.27b_campaign_execution_plan.v1"

#: The only run of this shape that has ever been measured. Every count below
#: that says "measured" comes out of this file.
REFERENCE_RESULT: Final = (
    "artifacts/closeout/latent_cortex/"
    "cp566_resident_mixed_multidomain_replication/result.json"
)

#: Stages that need the weights resident, in the order they run.
MODEL_ACTIVE_STAGES: Final = (
    "calibration",
    "training",
    "canary",
    "lesion_arms",
    "export",
)


def _reference() -> dict[str, Any] | None:
    path = installation_root() / REFERENCE_RESULT
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def measured_workload(result: dict[str, Any]) -> dict[str, Any]:
    """Decode calls, retries, and per-arm cost, read from the receipt."""
    rows = result.get("raw_outputs") or []
    decode_calls = 0
    retries_by_arm: dict[str, int] = {}
    rows_by_arm: dict[str, int] = {}
    zero_prefill_rows = 0
    for row in rows:
        arm = str(row.get("arm") or "")
        attempts = row.get("attempts") or []
        decode_calls += len(attempts)
        rows_by_arm[arm] = rows_by_arm.get(arm, 0) + 1
        retries_by_arm[arm] = retries_by_arm.get(arm, 0) + max(0, len(attempts) - 1)
        if attempts and not attempts[0].get("prefill_tokens"):
            zero_prefill_rows += 1

    arms = result.get("arms") or {}
    per_arm_seconds = {
        name: round(
            float(entry.get("mean_latency_ms", 0.0)) * int(entry.get("examples", 0)) / 1000.0,
            1,
        )
        for name, entry in arms.items()
    }
    return {
        "source": REFERENCE_RESULT,
        "checkpoint": "Qwen2.5-32B fuse (the only measured run of this shape)",
        "task_count": result.get("task_count"),
        "arm_rows": len(rows),
        "decode_calls": decode_calls,
        "retry_decodes": decode_calls - len(rows),
        "retries_by_arm": dict(sorted(retries_by_arm.items())),
        "rows_by_arm": dict(sorted(rows_by_arm.items())),
        "rows_whose_prefix_was_already_cached": zero_prefill_rows,
        "elapsed_seconds": result.get("elapsed_seconds"),
        "arm_seconds": dict(sorted(per_arm_seconds.items())),
        "dominant_arm": max(per_arm_seconds, key=per_arm_seconds.get)
        if per_arm_seconds
        else None,
    }


def _tokenizations(result: dict[str, Any]) -> dict[str, Any]:
    """How often the task prompt is tokenized, and how often it must be.

    ``_prompt_tokens`` runs per decode attempt. The task prompt inside it is
    immutable for the campaign; only the per-arm context changes. So the task
    half is retokenized once per attempt and needs to be done once per task.
    """
    rows = result.get("raw_outputs") or []
    attempts = sum(len(row.get("attempts") or []) for row in rows)
    tasks = len({str(row.get("task_id") or "") for row in rows})
    return {
        "task_prompt_tokenizations_today": attempts,
        "task_prompt_tokenizations_required": tasks,
        "eliminated": max(0, attempts - tasks),
        "note": (
            "only the immutable task half is shared; the per-arm context and "
            "correction text differ per attempt and are tokenized either way"
        ),
    }


def optimizations(measured: dict[str, Any], tokenization: dict[str, Any]) -> list[dict[str, Any]]:
    """Each entry says what it changes, and what it is allowed to claim."""
    return [
        {
            "name": "single_resident_worker_across_stages",
            "status": "adopt",
            "changes": "process lifecycle",
            "before": f"{len(MODEL_ACTIVE_STAGES)} model loads (one per stage command)",
            "after": "1 model load, held across the five model-active stages",
            "counted_saving": f"{len(MODEL_ACTIVE_STAGES) - 1} model loads",
            "wall_clock": "unmeasured -- no retained receipt records a load duration",
            "affects_measured_compute": False,
        },
        {
            "name": "pretokenize_the_frozen_task_set",
            "status": "adopt",
            "changes": "CPU work before the first decode",
            "before": f"{tokenization['task_prompt_tokenizations_today']} task tokenizations",
            "after": f"{tokenization['task_prompt_tokenizations_required']} task tokenizations, digest-bound to the campaign",
            "counted_saving": f"{tokenization['eliminated']} tokenizations",
            "wall_clock": "unmeasured -- tokenization is not separately timed in the receipt",
            "affects_measured_compute": False,
        },
        {
            "name": "cpu_stages_outside_the_model_process",
            "status": "adopt",
            "changes": "where verification, hashing, rendering and adjudication run",
            "before": "in-process with the weights resident",
            "after": "after unload, from immutable artifacts, in a separate process",
            "counted_saving": "0 decode calls; frees the model lane earlier",
            "wall_clock": "unmeasured",
            "affects_measured_compute": False,
        },
        {
            "name": "prompt_prefix_reuse",
            "status": "already_implemented",
            "changes": "nothing -- this is existing behaviour",
            "evidence": (
                f"{measured['rows_whose_prefix_was_already_cached']} of "
                f"{measured['arm_rows']} rows prefilled zero tokens in CP566"
            ),
            "counted_saving": "0 -- claiming it would invent a speedup from existing behaviour",
            "affects_measured_compute": False,
        },
        {
            "name": "batch_greedy_decodes_across_arms",
            "status": "rejected",
            "reason": (
                "_arm_order counterbalances arm order per task so no arm's cache "
                "state can systematically advantage another. Batching by arm "
                "destroys the interleaving counterbalancing exists to create."
            ),
            "affects_measured_compute": True,
        },
        {
            "name": "drop_serialization_retries",
            "status": "rejected",
            "reason": (
                "retries concentrate in the arms designed to fail -- "
                f"{measured['retries_by_arm']} -- so they are the lesion and "
                "wrong-state controls behaving correctly, not waste"
            ),
            "affects_measured_compute": True,
        },
        {
            "name": "share_post_treatment_state_between_arms",
            "status": "rejected",
            "reason": "an arm reading another arm's post-treatment state is not that arm",
            "affects_measured_compute": True,
        },
    ]


def build() -> dict[str, Any]:
    result = _reference()
    if result is None:
        return {
            "schema": EXECUTION_PLAN_SCHEMA,
            "measured": None,
            "blocked": "the reference result is not installed; nothing can be counted",
        }
    measured = measured_workload(result)
    tokenization = _tokenizations(result)
    entries = optimizations(measured, tokenization)
    adopted = [entry for entry in entries if entry["status"] == "adopt"]
    rejected = [entry for entry in entries if entry["status"] == "rejected"]
    return {
        "schema": EXECUTION_PLAN_SCHEMA,
        "measured_reference": measured,
        "tokenization": tokenization,
        "optimizations": entries,
        "counted_changes": {
            "model_loads_before": len(MODEL_ACTIVE_STAGES),
            "model_loads_after": 1,
            "decode_calls_before": measured["decode_calls"],
            "decode_calls_after": measured["decode_calls"],
            "decode_calls_note": (
                "unchanged on purpose: every adopted optimization is outside "
                "the model's own work, so the scientific workload is identical"
            ),
            "task_tokenizations_before": tokenization["task_prompt_tokenizations_today"],
            "task_tokenizations_after": tokenization["task_prompt_tokenizations_required"],
        },
        "wall_clock_claim": (
            "none. Every adopted optimization removes process and CPU overhead "
            "that no retained receipt times separately, so the saving is counted "
            "in loads and tokenizations and not in seconds."
        ),
        "adopted": [entry["name"] for entry in adopted],
        "rejected": [entry["name"] for entry in rejected],
        "arm_equality_rule": (
            "an optimization that changes any arm's measured compute is "
            "rejected, including one that would make a control cheaper: "
            "equal-compute and lesion validity outrank wall-clock"
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)

    plan = build()
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(plan, indent=1, sort_keys=True))
        print(f"wrote {args.json}\n")

    if plan.get("blocked"):
        print(f"blocked: {plan['blocked']}")
        return 1

    measured = plan["measured_reference"]
    print(f"measured reference   {measured['source']}")
    print(
        f"  {measured['task_count']} tasks, {measured['arm_rows']} arm rows, "
        f"{measured['decode_calls']} decode calls "
        f"({measured['retry_decodes']} of them retries)"
    )
    print(f"  elapsed {measured['elapsed_seconds']}s, dominated by {measured['dominant_arm']}")
    for arm, seconds in measured["arm_seconds"].items():
        print(f"      {arm:22s} {seconds:>8.1f}s")
    print()
    counted = plan["counted_changes"]
    print("counted changes")
    print(f"  model loads        {counted['model_loads_before']} -> {counted['model_loads_after']}")
    print(
        f"  task tokenizations {counted['task_tokenizations_before']} -> "
        f"{counted['task_tokenizations_after']}"
    )
    print(
        f"  decode calls       {counted['decode_calls_before']} -> "
        f"{counted['decode_calls_after']} (unchanged, on purpose)"
    )
    print()
    for entry in plan["optimizations"]:
        mark = {"adopt": "adopt   ", "rejected": "REJECT  ", "already_implemented": "existing"}[
            entry["status"]
        ]
        print(f"  {mark} {entry['name']}")
        if entry["status"] == "rejected":
            print(f"           {entry['reason']}")
    print(f"\n{plan['wall_clock_claim']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
