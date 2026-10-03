#!/usr/bin/env python3
"""Advance one supervised grounded preparation into its exact fit and verification."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode()).hexdigest()


def verified_document(path, field="receipt_sha256"):
    from core.runtime.file_read_gateway import read_stable_bytes

    value = json.loads(read_stable_bytes(path, max_bytes=64 * 1024 ** 2))
    if not isinstance(value, dict) or value.get(field) != digest(
            {key: item for key, item in value.items() if key != field}):
        raise ValueError("grounded preparation artifact checksum differs")
    return value


def preparation_paths(supervised):
    command = supervised.get("command")
    cwd = Path(supervised.get("cwd", ""))
    if (not isinstance(command, list) or len(command) < 4
            or any(not isinstance(value, str) or not value for value in command)
            or not cwd.is_absolute() or not Path(command[0]).is_absolute()
            or command[1] != "-u" or Path(command[2]) != cwd / "tools/fit_semantic_grounded_binding.py"
            or command.count("--prepare-only") != 1 or command.count("--native") != 1
            or any(flag in command for flag in ("--resume", "--resume-if-available", "--no-native"))):
        raise ValueError("grounded handoff requires one unchanged fresh native preparation command")

    def path(flag):
        indices = [index for index, value in enumerate(command) if value == flag]
        if (len(indices) != 1 or indices[0] + 1 == len(command)
                or command[indices[0] + 1].startswith("--")
                or not Path(command[indices[0] + 1]).is_absolute()):
            raise ValueError("grounded preparation path custody is incomplete or repeated")
        return Path(command[indices[0] + 1])

    paths = {"native": path("--directory"), "parent": path("--parent"),
        "source": path("--source-report"), "folds": path("--folds"), "bank": path("--bank"),
        "authority": path("--authority-key-file")}
    names = set()
    for index, value in enumerate(command):
        if value != "--bundle":
            continue
        if index + 1 == len(command):
            raise ValueError("grounded preparation bundle is incomplete")
        name, separator, raw = command[index + 1].partition("=")
        if not name or separator != "=" or not Path(raw).is_absolute() or name in names:
            raise ValueError("grounded preparation bundle custody differs")
        names.add(name)
    if not names:
        raise ValueError("grounded preparation has no acquired source bundles")
    if "--complete-program-objective" in command:
        if "--native-operation-field" not in command:
            raise ValueError("complete-program preparation has no native operation field")
        paths["pools"] = path("--program-pool-directory")
    return {**paths, "cwd": cwd, "command": [value for value in command if value != "--prepare-only"]}


def fit_jobs(paths, directory):
    directory = Path(directory)
    verify = [paths["command"][0], "-u", str(paths["cwd"] / "tools/verify_semantic_grounded_fit.py"),
        "--directory", str(paths["native"])]
    return [{"name": name, "command": command, "cwd": str(paths["cwd"]),
        "stdout_path": str(directory / "logs" / (name + ".log")),
        "timeout_s_max": timeout, "max_invocations": 1}
        for name, command, timeout in (("grounded-fit", paths["command"], 18000.),
            ("grounded-fit-verify", verify, 1800.))]


def verify_preparation(paths):
    from core.learning.semantic_grounded_program_objective import (
        GroundedProgramSupervision,
        program_objective_contract,
    )
    from core.learning.semantic_native_operation_field import NativeOperationField
    from tools.train_nested_semantic_ranker import _verified_pair

    native = paths["native"]
    custody = native.parent / (native.name + "-native-custody")
    if native.exists() or (custody / "completion.json").exists() or (custody / "prefixes").exists():
        raise ValueError("grounded fit already has evidence; handoff cannot repeat acquisition or training")
    plan = verified_document(custody / "plan.json", "plan_sha256")
    implementation = plan.get("implementation")
    if not isinstance(implementation, dict) or not implementation:
        raise ValueError("grounded preparation has no implementation custody")
    for name, sha in implementation.items():
        relative = Path(name)
        if (relative.is_absolute() or ".." in relative.parts
                or hashlib.sha256((paths["cwd"] / relative).read_bytes()).hexdigest() != sha):
            raise ValueError("grounded fit changed the prepared implementation")
    outer, _report = _verified_pair(paths["bank"])
    basis = {"parent_sha256": hashlib.sha256(paths["parent"].read_bytes()).hexdigest(),
        "source_report_sha256": hashlib.sha256(paths["source"].read_bytes()).hexdigest(),
        "folds_sha256": hashlib.sha256(paths["folds"].read_bytes()).hexdigest(),
        "bank_plan_sha256": outer["plan_sha256"]}
    if plan.get("source_basis") != basis or plan.get("installed_arithmetic", {}).get("MLX_ENABLE_TF32") != "0":
        raise ValueError("grounded preparation source basis or arithmetic differs")
    fit_ids, calibration_ids = plan.get("fit_ids"), plan.get("calibration_ids")
    if (not isinstance(fit_ids, list) or not fit_ids or not isinstance(calibration_ids, list)
            or not calibration_ids or any(not isinstance(value, str) or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
                for value in fit_ids + calibration_ids)
            or len(set(fit_ids + calibration_ids)) != len(fit_ids + calibration_ids)
            or set(fit_ids) != set(outer["fit_ids"])
            or not set(calibration_ids) <= set(outer["calibration_ids"])
            or set(fit_ids + calibration_ids) & set(outer["held_ids"])):
        raise ValueError("grounded preparation lacks disjoint populated source partitions")
    checked = {}
    contract = plan.get("program_objective_contract")
    if "pools" in paths:
        if not isinstance(contract, dict) or contract.get("source_ids") != sorted(fit_ids + calibration_ids):
            raise ValueError("grounded program pool population differs from prepared source custody")
        field = NativeOperationField.from_contract(plan["operation_field_contract"])
        programs = {}
        for identity in contract["source_ids"]:
            pool = verified_document(paths["pools"] / (identity + ".json"))
            pool_basis = pool["basis"]
            if (pool.get("basis_sha256") != digest(pool_basis) or pool_basis.get("source_id") != identity
                    or pool_basis.get("implementation") != implementation
                    or pool_basis.get("source_annotation", {}).get("source_text_sha256") != identity):
                raise ValueError("grounded retained program source or implementation custody differs")
            program = GroundedProgramSupervision.from_receipt(pool["programs"])
            if program.source_id != identity:
                raise ValueError("grounded retained program changed source identity")
            program.validate(field, len(pool_basis["source_annotation"]["source_token_ids"]))
            programs[identity] = program
            checked[identity] = pool["receipt_sha256"]
        if program_objective_contract(programs, contract["weight"]) != contract:
            raise ValueError("grounded retained program supervision differs from the prepared objective")
    elif contract is not None:
        raise ValueError("grounded prepared program objective has no bound pool directory")
    return {"plan_sha256": plan["plan_sha256"], "source_fit_count": len(fit_ids),
        "source_calibration_count": len(calibration_ids), "source_pool_receipts": checked,
        "partial_search_pools": sum(not row.get("requested_searches_completed", True)
            for row in (contract or {}).get("mining", [])),
        "model_weights_loaded": False, "semantic_success": None,
        "qualification_evidence": False, "serving_authority": False}


def verify_fit_log(path, prepared):
    from core.runtime.file_read_gateway import read_stable_bytes

    documents = []
    for line in read_stable_bytes(path, max_bytes=16 * 1024 ** 2).decode().splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and value.get("schema") == "aura.grounded_binding_verification.v1":
            documents.append(value)
    if len(documents) != 1:
        raise ValueError("grounded fit verifier must report exactly one artifact verdict")
    verification = documents[0]
    if (verification.get("receipt_sha256") != digest({key: value for key, value in verification.items()
            if key != "receipt_sha256"}) or verification.get("learned_checkpoint_selected") is not True
            or verification.get("artifacts_verified") is not True
            or type(verification.get("selected_step")) is not int or verification["selected_step"] <= 0
            or type(verification.get("completed_updates")) is not int
            or verification["completed_updates"] < verification["selected_step"]
            or verification.get("current_implementation_drift") != []
            or verification.get("source_fit_count") != prepared["source_fit_count"]
            or verification.get("source_calibration_count") != prepared["source_calibration_count"]
            or verification.get("model_weights_loaded") is not False
            or verification.get("held_sources_scored") is not False
            or verification.get("semantic_success") is not None
            or verification.get("qualification_evidence") is not False
            or verification.get("serving_authority") is not False):
        raise ValueError("grounded fit has no independently verified positive-step checkpoint")
    return verification


def wait_for_model_lane(*, timeout, observe=None, clock=time.monotonic, sleep=time.sleep):
    if not math.isfinite(timeout) or not 0 < timeout <= 18000:
        raise ValueError("grounded model-lane wait requires a bounded positive allowance")
    if observe is None:
        from core.runtime.model_lane_control import get_model_lane_controller
        observe = get_model_lane_controller().owner_observations
    deadline, previous = clock() + timeout, None
    while True:
        owners = observe()
        if not owners:
            return
        identities = sorted(owner.owner_id for owner in owners)
        if identities != previous:
            print(json.dumps({"stage": "grounded_model_lane_wait", "owner_ids": identities,
                "owner_eviction_allowed": False}), flush=True)
            previous = identities
        remaining = deadline - clock()
        if remaining <= 0:
            raise TimeoutError("grounded model lane stayed occupied; no fit was launched")
        sleep(min(30., remaining))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preparation-supervisor", type=Path, required=True)
    parser.add_argument("--preparation-plan-sha256", required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--wait-seconds", type=float, default=18000.)
    parser.add_argument("--lane-wait-seconds", type=float, default=18000.)
    parser.add_argument("--policy-output", type=Path)
    args = parser.parse_args()
    if (not math.isfinite(args.wait_seconds) or not 0 < args.wait_seconds <= 18000
            or not math.isfinite(args.lane_wait_seconds) or not 0 < args.lane_wait_seconds <= 18000):
        parser.error("grounded handoff waits must be finite and bounded")
    from core.runtime.detached_subprocess_broker import broker_available, run_brokered_process
    from tools.refit_semantic_argument_proposals import configure_refit_environment
    from tools.run_detached_step import PLAN_FILE, _verify_plan
    from tools.run_semantic_grounded_handoff import check_terminal, publish
    from tools.run_semantic_native_micro_stages import wait_for_fit

    supervisor, directory = args.preparation_supervisor.resolve(), args.directory.resolve()
    supervised = json.loads((supervisor / PLAN_FILE).read_bytes())
    _verify_plan(supervised, supervisor / PLAN_FILE)
    if supervised["plan_sha256"] != args.preparation_plan_sha256:
        raise ValueError("grounded handoff belongs to another preparation supervisor")
    paths = preparation_paths(supervised)
    jobs = fit_jobs(paths, directory)
    scripts = {name: hashlib.sha256((paths["cwd"] / "tools" / name).read_bytes()).hexdigest()
        for name in ("fit_semantic_grounded_binding.py", "verify_semantic_grounded_fit.py")}
    if args.policy_output is not None:
        from core.governance_context import local_internal_governed_scope
        from core.runtime.file_write_gateway import get_file_write_gateway
        (directory / "logs").mkdir(parents=True, exist_ok=True)
        with local_internal_governed_scope("grounded_fit_handoff_policy", domain="file_write"):
            if not get_file_write_gateway().write_bytes_if_absent(args.policy_output,
                    json.dumps([{key: value for key, value in row.items() if key != "name"}
                        for row in jobs], indent=2).encode(), source="grounded_fit_handoff_policy", mode=0o400):
                raise FileExistsError(args.policy_output)
        print(json.dumps({"stage": "grounded_fit_handoff_policy", "commands": len(jobs)}), flush=True)
        return 0
    if not broker_available():
        raise ValueError("grounded fit handoff requires detached exact-command supervision")
    configure_refit_environment(directory / "handoff.json")
    plan = publish(directory / "handoff.json", {"schema": "aura.grounded_fit_handoff_plan.v1",
        "preparation_supervisor": str(supervisor), "preparation_plan_sha256": supervised["plan_sha256"],
        "preparation_command_sha256": supervised["command_sha256"], "jobs": jobs, "scripts": scripts,
        "wait_seconds": args.wait_seconds, "lane_wait_seconds": args.lane_wait_seconds,
        "duplicate_training_allowed": False, "owner_eviction_allowed": False,
        "qualification_evidence": False, "serving_authority": False})
    terminal = wait_for_fit(supervisor, timeout=args.wait_seconds)
    check_terminal(terminal, supervised)
    verified = verify_preparation(paths)
    publish(directory / "preparation.json", {"schema": "aura.grounded_fit_handoff_prerequisite.v1",
        "handoff_plan_receipt_sha256": plan["receipt_sha256"],
        "preparation_terminal_receipt_sha256": terminal["receipt_sha256"], **verified})
    wait_for_model_lane(timeout=args.lane_wait_seconds)
    if verify_preparation(paths) != verified or any(hashlib.sha256(
            (paths["cwd"] / "tools" / name).read_bytes()).hexdigest() != sha for name, sha in scripts.items()):
        raise ValueError("grounded preparation or executable changed before fit launch")
    outcomes = []
    for row in jobs:
        print(json.dumps({"stage": row["name"], "status": "started"}), flush=True)
        outcome = run_brokered_process(row["command"], cwd=Path(row["cwd"]),
            stdout_path=Path(row["stdout_path"]), timeout_s=row["timeout_s_max"])
        if (outcome.returncode != 0 or outcome.status != "passed" or outcome.timed_out
                or not outcome.containment_verified):
            raise ValueError(f"grounded fit handoff failed: {row['name']}:{outcome.status}")
        outcomes.append(outcome.receipt_sha256)
    verification = verify_fit_log(Path(jobs[1]["stdout_path"]), verified)
    custody = paths["native"].parent / (paths["native"].name + "-native-custody")
    if verified_document(custody / "plan.json", "plan_sha256")["plan_sha256"] != verified["plan_sha256"]:
        raise ValueError("grounded fit changed its independently checked preparation plan")
    complete = publish(directory / "completion.json", {"schema": "aura.grounded_fit_handoff_complete.v1",
        "handoff_plan_receipt_sha256": plan["receipt_sha256"],
        "preparation_terminal_receipt_sha256": terminal["receipt_sha256"],
        "training_plan_sha256": verified["plan_sha256"], "broker_receipts": outcomes,
        "fit_verification": verification, "semantic_success": None,
        "g03_complete": False, "qualification_evidence": False, "serving_authority": False})
    print(json.dumps({"stage": "grounded_fit_handoff_complete", **complete}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
