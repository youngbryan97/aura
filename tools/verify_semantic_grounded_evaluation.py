#!/usr/bin/env python3
"""Verify a paired development receipt without loading the resident backbone."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def document(value, key):
    from tools.evaluate_semantic_grounded_native import digest
    if not isinstance(value, dict) or value.get(key) != digest({k: v for k, v in value.items() if k != key}):
        raise ValueError("joint development document checksum differs")
    return value


def verify_report(report, plan, fit):
    from tools.evaluate_semantic_grounded_native import digest

    document(report, "receipt_sha256")
    document(plan, "plan_sha256")
    document(fit, "receipt_sha256")
    comparison = document(report["comparison"], "report_sha256")
    names = ["source_parent", "global_chart", "joint_native"]
    identities = plan["held_ids"]
    if (report.get("schema") != "aura.grounded_native_development.v1"
            or plan.get("schema") != "aura.grounded_native_development_plan.v1"
            or report["plan"] != plan or plan["fit_verification"] != fit
            or fit.get("learned_checkpoint_selected") is not True or fit["selected_step"] <= 0
            or not identities or len(set(identities)) != len(identities)
            or plan.get("arms") != names or set(comparison["candidates"]) != set(names)
            or set(report["decodes"]) != set(names)
            or comparison.get("validation_example_ids") != identities
            or comparison.get("validation_examples") != len(identities)
            or comparison.get("scoring") != "source_anchors_v2"
            or comparison.get("incumbent") != "source_parent" or comparison.get("test_examples_used") != 0
            or comparison.get("equivalence_used_for_selection") is not False
            or plan.get("cohort") != "previously_exposed_source_bank_development"
            or any(report.get(key) is not False for key in ("g03_complete", "fresh_transfer_proven",
                "held_used_for_fit_or_checkpoint_selection", "qualification_evidence", "serving_authority"))
            or plan.get("held_controls_fit_or_checkpoint_selection") is not False
            or plan.get("serving_authority") is not False
            or any(comparison.get(key) is not False for key in (
                "expected_answers_available", "gold_program_available_to_decode", "serving_authority"))):
        raise ValueError("joint development identities, proof boundaries or selected fit differ")
    metrics = comparison["candidates"]
    for name in names:
        row = metrics[name]
        for field, count in (("program_correct", "program_exact"),
                             ("program_equivalent_correct", "program_equivalent")):
            values = row[field]
            if (len(values) != len(identities) or any(type(value) is not bool for value in values)
                    or row[count] != sum(values)):
                raise ValueError("joint development measured counts differ")
        if any(exact and not equivalent for exact, equivalent in zip(
                row["program_correct"], row["program_equivalent_correct"], strict=True)):
            raise ValueError("joint development exact results contradict equivalence")
        for prefix, field in (("", "program_correct"), ("equivalent_", "program_equivalent_correct")):
            values, baseline = row[field], metrics["source_parent"][field]
            if (row[prefix + "gains"] != sum(new and not old for new, old in zip(values, baseline, strict=True))
                    or row[prefix + "regressions"] != sum(old and not new for new, old in zip(values, baseline, strict=True))):
                raise ValueError("joint development paired gains differ")
        if row["transducer_receipt_sha256"] != digest({"arm": name, "plan": plan["plan_sha256"]}):
            raise ValueError("joint development arm receipt differs")
        decodes = report["decodes"][name]
        if set(decodes) != set(identities):
            raise ValueError("joint development request inventory differs")
        for index, identity in enumerate(identities):
            decoded = decodes[identity]
            if plan.get("profile_only") is True:
                document(decoded["profile"], "receipt_sha256")
            elapsed = decoded.get("elapsed_seconds")
            if (not isinstance(elapsed, (int, float)) or not math.isfinite(elapsed) or elapsed < 0
                    or (decoded["program_sha256"] is None) != bool(decoded["refusal"])
                    or row["program_equivalent_correct"][index] and decoded["program_sha256"] is None):
                raise ValueError("joint development outcome or timing differs")
            if name == "joint_native":
                receipt = decoded["joint_receipt"]
                if (not isinstance(receipt, dict) or receipt.get("schema") != "aura.grounded_native_chart_decode.v1"
                        or receipt.get("source_id") != identity or receipt.get("source_only_native_capture") is not True
                        or receipt.get("target_available_to_decoder") is not False
                        or receipt.get("serving_authority") is not False
                        or receipt.get("fit_receipt_sha256") != fit["fit_receipt_sha256"]
                        or receipt.get("weights_sha256") != fit["weights_sha256"]
                        or receipt.get("selected_step") != fit["selected_step"]
                        or receipt.get("learned_checkpoint_selected") is not True
                        or receipt.get("refusal") != decoded["refusal"]
                        or (receipt.get("selected_chart") is None) != (decoded["program_sha256"] is None)
                        or receipt.get("selected_chart") is not None and receipt["selected_chart"].get("status") != "bound"):
                    raise ValueError("joint development measured native selection differs")
    correct = metrics["joint_native"]["program_equivalent_correct"]
    losses = {name: sum(old and not new for old, new in zip(
        metrics[name]["program_equivalent_correct"], correct, strict=True)) for name in names[:2]}
    advance = all(correct) and metrics["joint_native"]["equivalent_gains"] > 0 and not any(losses.values())
    if plan.get("profile_only") is True:
        advance = False
    if report["paired_regressions"] != losses or report["advance_development"] is not advance:
        raise ValueError("joint development advancement verdict differs")
    body = {"schema": "aura.grounded_native_development_verification.v1",
        "report_receipt_sha256": report["receipt_sha256"], "plan_sha256": plan["plan_sha256"],
        "fit_verification_receipt_sha256": fit["receipt_sha256"], "measured_requests": len(identities),
        "advance_development": advance, "artifacts_verified": True,
        "semantic_interpretations_independently_replayed": False,
        "g03_complete": False, "qualification_evidence": False, "serving_authority": False}
    return {**body, "receipt_sha256": digest(body)}


def verify(path, directory):
    from core.learning.semantic_grounded_binding_engine import implementation_receipt
    from core.runtime.file_read_gateway import read_stable_bytes
    from tools.verify_semantic_grounded_fit import verify as verify_fit

    path = Path(path)
    report = json.loads(read_stable_bytes(path, max_bytes=64 * 1024 ** 2))
    plan = json.loads(read_stable_bytes(path.with_suffix(".plan.json"), max_bytes=64 * 1024 ** 2))
    if (plan["implementation"] != implementation_receipt()
            or plan["evaluator_sha256"] != hashlib.sha256((ROOT / "tools/evaluate_semantic_grounded_native.py").read_bytes()).hexdigest()):
        raise ValueError("joint development implementation changed")
    return verify_report(report, plan, verify_fit(directory))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.report, args.directory)
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0 if result["advance_development"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
