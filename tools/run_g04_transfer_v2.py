#!/usr/bin/env python3
"""Run G04's second preregistered transfer protocol, exactly as the plan froze it.

As the first runner (tools/run_g04_transfer.py), whose decode, grading and
no-execution guard it uses unchanged: every frozen identity is rechecked
before the first decode, every committed task is decoded once per arm with
program execution unavailable, each row is written once, and an interrupted
run resumes on the same tasks.

Arms: the incumbent, candidate v13, v12, and v13 with its phrase readouts or
its recency features taken out (g04_transfer_protocol_v2.lesions). Paired
tasks follow the plan's counterbalanced order; construction tasks, which the
plan holds as an absolute claim, take the arms in the declared order, which
does not change a deterministic decode.

Usage:
    run_g04_transfer_v2.py --protocol DIR --plan FILE --features DIR --output DIR
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def exact_one_sided(wins: int, losses: int) -> float:
    from scipy.stats import binomtest

    discordant = wins + losses
    return float(binomtest(wins, discordant, 0.5, alternative="greater").pvalue) if discordant else 1.0


def analyse(spec: dict, plan: dict, rows: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    """The plan's analysis from the rows alone: primary, construction, secondary, descriptive."""
    alpha = plan["parameters"]["per_comparison_alpha"]
    report: dict[str, Any] = {"per_comparison_alpha": alpha, "primary": {}, "secondary": {}, "arms": {}}
    for stratum in spec["domains"]:
        tasks = rows[stratum]
        wins = sum(r["candidate"]["equivalent"] and not r["incumbent"]["equivalent"] for r in tasks.values())
        losses = sum(r["incumbent"]["equivalent"] and not r["candidate"]["equivalent"] for r in tasks.values())
        p_value = exact_one_sided(wins, losses)
        report["primary"][stratum] = {
            "tasks": len(tasks), "candidate_only": wins, "incumbent_only": losses,
            "exact_one_sided_p": p_value, "rejects": bool(p_value <= alpha),
        }
    construction = rows["construction"]
    report["construction"] = {
        "tasks": len(construction),
        "candidate_equivalent": sum(bool(r["candidate"]["equivalent"]) for r in construction.values()),
    }
    report["construction"]["holds"] = report["construction"]["candidate_equivalent"] == len(construction)
    secondary = spec["secondary"]
    for comparison in secondary["comparisons"]:
        tasks = rows[comparison["stratum"]]
        treat, control = comparison["treatment"], comparison["control"]
        wins = sum(r[treat]["equivalent"] and not r[control]["equivalent"] for r in tasks.values())
        losses = sum(r[control]["equivalent"] and not r[treat]["equivalent"] for r in tasks.values())
        p_value = exact_one_sided(wins, losses)
        report["secondary"][f"{treat}_vs_{control}:{comparison['stratum']}"] = {
            "treatment_only": wins, "control_only": losses, "exact_one_sided_p": p_value,
            "rejects": bool(p_value <= secondary["per_test_alpha"]),
        }
    for stratum, tasks in rows.items():
        report["arms"][stratum] = {
            arm: {
                "equivalent": sum(bool(r[arm]["equivalent"]) for r in tasks.values()),
                "answers_correct": sum(bool(r[arm]["answer_correct"]) for r in tasks.values()),
                "decoded_without_execution": sum(bool(r[arm]["decoded_without_execution"]) for r in tasks.values()),
            }
            for arm in spec["arms"]
        }
    report["all_primary_reject"] = all(entry["rejects"] for entry in report["primary"].values())
    report["g04_closure_rule_holds"] = report["all_primary_reject"] and report["construction"]["holds"]
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from core.learning.semantic_operation_peaks import peak_recognition_transducer_from_dict
    from core.learning.semantic_program_campaign import training_examples_from_feature_bundle
    from core.learning.semantic_program_compositional_transducer import (
        compositional_semantic_program_transducer_from_dict as restore,
    )
    from core.learning.semantic_program_feature_materialization import (
        load_standard_semantic_feature_bundle,
    )
    from tools import g04_transfer_protocol_v2 as protocol
    from tools.run_g04_transfer import decode, grade
    from tools.run_semantic_peak_recognition import _write_once

    spec = json.loads((args.protocol / "spec.json").read_text(encoding="utf-8"))
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    summary = json.loads((args.protocol / "plan_summary.json").read_text(encoding="utf-8"))
    if Path(summary["plan_path"]).name != args.plan.name:
        raise SystemExit("the plan given is not the one the protocol published")
    observed = {
        "candidate": protocol._files_sha256([protocol.CANDIDATE]),
        "incumbent": protocol._files_sha256([protocol.INCUMBENT]),
        "v12": protocol._files_sha256([protocol.PREVIOUS]),
        "resident_model": protocol._files_sha256(sorted(p for p in protocol.MODEL.iterdir() if p.is_file())),
        "task_generator": protocol._files_sha256([ROOT / p for p in protocol.GENERATOR_FILES]),
        "scorer": protocol._files_sha256([ROOT / p for p in protocol.SCORER_FILES]),
        "runtime": protocol._files_sha256([ROOT / p for p in protocol.RUNTIME_FILES]),
    }
    drift = sorted(name for name, value in spec["artifacts"].items() if observed.get(name) != value)
    if drift:
        raise SystemExit(f"frozen artifacts changed since the plan: {drift}")

    incumbent = restore(json.loads(protocol.INCUMBENT.read_text(encoding="utf-8")))
    candidate = peak_recognition_transducer_from_dict(
        json.loads(protocol.CANDIDATE.read_text(encoding="utf-8")), restore_base=restore)
    previous = peak_recognition_transducer_from_dict(
        json.loads(protocol.PREVIOUS.read_text(encoding="utf-8")), restore_base=restore)
    arms = {"incumbent": incumbent, "candidate": candidate, "v12": previous, **protocol.lesions(candidate)}
    if {name: model.receipt_sha256 for name, model in arms.items()} != spec["arm_receipts"]:
        raise SystemExit("an arm is not the one the plan committed")

    bundle = load_standard_semantic_feature_bundle(args.features.expanduser())
    items = training_examples_from_feature_bundle(bundle, required_splits=frozenset({"test"}))
    by_source = {item.ir.source_text_sha256: replace(
        item, ir=replace(item.ir, model_basis_receipt_sha256=incumbent.model_basis_sha256)) for item in items}
    committed = {task["source_sha256"]: (stratum, task["id"])
                 for stratum, tasks in spec["domains"].items() for task in tasks}
    committed.update({task["source_sha256"]: ("construction", task["id"])
                      for task in spec["absolute_domains"]["construction"]["tasks"]})
    if set(by_source) != set(committed):
        raise SystemExit("the feature bundle does not hold exactly the committed tasks")
    order = plan["parameters"]["arm_order_by_task"]

    output = args.output.expanduser()
    rows: dict[str, dict[str, dict[str, Any]]] = {}
    started = time.time()
    for count, (source, (stratum, task_id)) in enumerate(sorted(committed.items(), key=lambda kv: kv[1]), 1):
        item = by_source[source]
        row: dict[str, Any] = {}
        for arm in order.get(task_id, spec["arms"]):
            path = output / "rows" / stratum / arm / f"{source}.json"
            if path.exists():
                row[arm] = json.loads(path.read_text(encoding="utf-8"))
                continue
            began = time.monotonic()
            outcome, failure = decode(arms[arm], item)
            row[arm] = _write_once(path, {
                "task_id": task_id, "stratum": stratum, "arm": arm, "source_sha256": source,
                "construction_id": item.construction_id,
                "refusal": getattr(outcome, "refusal", "") or "",
                "failure": failure, "decoded_without_execution": not failure.startswith("execution"),
                "seconds": round(time.monotonic() - began, 3), **grade(item, outcome),
            })
        rows.setdefault(stratum, {})[task_id] = row
        if count % 25 == 0:
            print(f"  {count} of {len(committed)} tasks, {time.time() - started:.0f}s", flush=True)
    report = {"plan_hash": plan.get("plan_hash") or summary["plan_hash"], **analyse(spec, plan, rows)}
    _write_once(output / "report.json", report)
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
