#!/usr/bin/env python3
"""Wait for one bound joint trainer, then run and verify its development screen."""

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


def evaluation_arguments(supervised, *, output, search_seconds):
    command = supervised["command"]
    if (len(command) < 4 or command[1] != "-u"
            or Path(command[2]).name != "fit_semantic_grounded_binding.py"
            or "--native" not in command or "--no-native" in command
            or "--prepare-only" in command
            or not math.isfinite(search_seconds) or not 0 < search_seconds <= 300):
        raise ValueError("joint handoff needs the exact native fit command and bounded screen")
    arguments = []
    for flag in ("--parent", "--source-report", "--folds", "--bank", "--directory", "--authority-key-file"):
        indices = [index for index, value in enumerate(command) if value == flag]
        if (len(indices) != 1 or indices[0] + 1 >= len(command)
                or command[indices[0] + 1].startswith("--")
                or not Path(command[indices[0] + 1]).is_absolute()):
            raise ValueError("joint handoff fit input custody is incomplete or repeated")
        arguments += [flag, command[indices[0] + 1]]
    bundles = []
    for index, value in enumerate(command):
        if value == "--bundle":
            if index + 1 == len(command):
                raise ValueError("joint handoff source bundle is incomplete")
            name, separator, path = command[index + 1].partition("=")
            if not name or separator != "=" or not Path(path).is_absolute() or name in bundles:
                raise ValueError("joint handoff source bundle custody differs")
            bundles.append(name)
            arguments += ["--bundle", command[index + 1]]
    if not bundles:
        raise ValueError("joint handoff has no source bundles")
    return [*arguments, "--output", str(output), "--search-seconds", str(search_seconds)]


def check_terminal(receipt, supervised):
    if (receipt.get("plan_sha256") != supervised["plan_sha256"]
            or receipt.get("command_sha256") != supervised["command_sha256"]
            or receipt.get("passed") is not True or receipt.get("containment_verified") is not True
            or receipt.get("returncode") != 0 or receipt.get("timed_out") is not False):
        raise ValueError("joint handoff terminal belongs to a different or failed fit")


def publish(path, body):
    from core.governance_context import local_internal_governed_scope
    from core.runtime.file_write_gateway import get_file_write_gateway
    from tools.evaluate_semantic_grounded_native import digest

    value = {**body, "receipt_sha256": digest(body)}
    with local_internal_governed_scope("grounded_handoff", domain="file_write"):
        if not get_file_write_gateway().write_bytes_if_absent(path, json.dumps(value, indent=2).encode(),
                source="grounded_handoff", mode=0o400):
            raise FileExistsError(path)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-supervisor", type=Path, required=True)
    parser.add_argument("--fit-supervisor-plan-sha256", required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--wait-seconds", type=float, default=15000.)
    parser.add_argument("--search-seconds", type=float, default=30.)
    args = parser.parse_args()
    if (not math.isfinite(args.wait_seconds) or not 0 < args.wait_seconds <= 15000
            or args.directory.exists()):
        parser.error("joint handoff needs a fresh directory and bounded prerequisite wait")
    from core.learning.semantic_grounded_binding_engine import implementation_receipt
    from core.runtime.file_read_gateway import read_stable_bytes
    from tools.evaluate_semantic_grounded_native import main as evaluate
    from tools.refit_semantic_argument_proposals import configure_refit_environment
    from tools.run_detached_step import PLAN_FILE, _verify_plan
    from tools.run_semantic_native_micro_stages import wait_for_fit
    from tools.verify_semantic_grounded_evaluation import verify as verify_evaluation
    from tools.verify_semantic_grounded_fit import verify as verify_fit

    supervisor, directory = args.fit_supervisor.resolve(), args.directory.resolve()
    supervised = json.loads(read_stable_bytes(supervisor / PLAN_FILE, max_bytes=64 * 1024 ** 2))
    _verify_plan(supervised, supervisor / PLAN_FILE)
    if supervised["plan_sha256"] != args.fit_supervisor_plan_sha256:
        raise ValueError("joint handoff prerequisite differs from the declared supervisor")
    output = directory / "development.json"
    arguments = evaluation_arguments(supervised, output=output, search_seconds=args.search_seconds)
    native = Path(arguments[arguments.index("--directory") + 1])
    implementation = implementation_receipt()
    scripts = {name: hashlib.sha256((ROOT / "tools" / name).read_bytes()).hexdigest() for name in (
        "run_semantic_grounded_handoff.py", "evaluate_semantic_grounded_native.py",
        "verify_semantic_grounded_evaluation.py")}
    configure_refit_environment(directory / "handoff.json")
    directory.mkdir(parents=True, exist_ok=False)
    plan = publish(directory / "handoff.json", {"schema": "aura.grounded_development_handoff_plan.v1",
        "fit_supervisor": str(supervisor), "fit_supervisor_plan_sha256": supervised["plan_sha256"],
        "fit_command_sha256": supervised["command_sha256"], "evaluation_arguments": arguments,
        "implementation": implementation, "scripts": scripts, "wait_seconds": args.wait_seconds,
        "duplicate_training_allowed": False, "qualification_evidence": False, "serving_authority": False})
    terminal = wait_for_fit(supervisor, timeout=args.wait_seconds)
    check_terminal(terminal, supervised)
    verified = verify_fit(native)
    if verified["learned_checkpoint_selected"] is not True:
        raise ValueError("joint handoff cannot advance an initial-only checkpoint")
    if (implementation_receipt() != implementation or any(
            hashlib.sha256((ROOT / "tools" / name).read_bytes()).hexdigest() != sha for name, sha in scripts.items())):
        raise ValueError("joint handoff implementation changed while waiting")
    publish(directory / "prerequisite.json", {"schema": "aura.grounded_development_prerequisite.v1",
        "handoff_plan_receipt_sha256": plan["receipt_sha256"], "fit_terminal_receipt_sha256": terminal["receipt_sha256"],
        "fit_verification": verified, "qualification_evidence": False, "serving_authority": False})
    print(json.dumps({"stage": "grounded_handoff_fit_verified", "selected_step": verified["selected_step"]}), flush=True)
    result = evaluate(arguments)
    measured = verify_evaluation(output, native)
    if result != (0 if measured["advance_development"] else 2):
        raise ValueError("joint development exit code disagrees with independent report verification")
    publish(directory / "verification.json", {"schema": "aura.grounded_development_handoff.v1",
        "handoff_plan_receipt_sha256": plan["receipt_sha256"], "evaluation_verification": measured,
        "g03_complete": False, "qualification_evidence": False, "serving_authority": False})
    print(json.dumps({"stage": "grounded_handoff_complete", **measured}), flush=True)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
