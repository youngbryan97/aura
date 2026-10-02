#!/usr/bin/env python3
"""Verify one completed development run and adjudicate its existing public rows."""

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


def adjudication_arguments(supervised, *, report, native_directory, output):
    command = supervised["command"]
    if (len(command) < 4 or command[1] != "-u"
            or Path(command[2]).name != "evaluate_semantic_grounded_native.py"
            or command.count("--archive-decodes") != 1 or "--profile-source-id" in command):
        raise ValueError("adjudication requires the original complete archived evaluation command")
    def argument(flag):
        if command.count(flag) != 1:
            raise ValueError("adjudication command input is missing or repeated")
        index = command.index(flag)
        if index + 1 >= len(command) or command[index + 1].startswith("--") or not Path(command[index + 1]).is_absolute():
            raise ValueError("adjudication command input needs an absolute path")
        return command[index + 1]
    if Path(argument("--output")) != report or Path(argument("--directory")) != native_directory:
        raise ValueError("adjudication cannot replace the evaluation report or fitted artifact")
    arguments = ["--report", str(report), "--directory", str(native_directory), "--output", str(output)]
    for flag in ("--parent", "--source-report"):
        arguments += [flag, argument(flag)]
    names = set()
    for index, value in enumerate(command):
        if value == "--bundle":
            if index + 1 == len(command):
                raise ValueError("adjudication source bundle is incomplete")
            name, separator, path = command[index + 1].partition("=")
            if not name or not separator or not Path(path).is_absolute() or name in names:
                raise ValueError("adjudication source bundle is invalid or repeated")
            names.add(name)
            arguments += ["--bundle", command[index + 1]]
    if not names:
        raise ValueError("adjudication needs the original source bundles")
    return arguments


def wait_for_evaluation(directory, supervised, *, timeout, inspect=None, clock=time.monotonic, sleep=time.sleep):
    if not math.isfinite(timeout) or not 0 < timeout <= 8000.:
        raise ValueError("adjudication wait must be positive, finite and bounded")
    if inspect is None:
        from tools.run_detached_step import _status
        inspect = _status
    deadline = clock() + timeout
    while True:
        status = inspect(directory)
        if status.get("terminal") is True:
            receipt = status.get("receipt") or {}
            if (receipt.get("plan_sha256") != supervised["plan_sha256"]
                    or receipt.get("command_sha256") != supervised["command_sha256"]
                    or receipt.get("containment_verified") is not True or receipt.get("timed_out") is not False
                    or type(receipt.get("returncode")) is not int or receipt["returncode"] not in (0, 2)
                    or status.get("child_state") != "dead"):
                raise ValueError("evaluation terminal is incomplete, escaped or belongs to a different run")
            return receipt
        if status.get("supervisor_alive") is not True or status.get("completion_indeterminate") is True:
            raise ValueError("evaluation completion is indeterminate; no duplicate decode is authorized")
        remaining = deadline - clock()
        if remaining <= 0:
            raise TimeoutError("bounded evaluation prerequisite wait expired")
        sleep(min(60., remaining))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("evaluation-supervisor", "report", "native-directory", "directory"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--evaluation-supervisor-plan-sha256", required=True)
    parser.add_argument("--evaluation-plan-sha256", required=True)
    parser.add_argument("--wait-seconds", type=float, default=7200.)
    args = parser.parse_args(argv)
    if not math.isfinite(args.wait_seconds) or not 0 < args.wait_seconds <= 8000. or args.directory.exists():
        parser.error("adjudication handoff needs a fresh directory and bounded wait")
    from core.runtime.file_read_gateway import read_stable_bytes
    from tools.adjudicate_semantic_grounded_development import adjudication_contract, main as adjudicate
    from tools.refit_semantic_argument_proposals import configure_refit_environment
    from tools.run_detached_step import PLAN_FILE, _verify_plan
    from tools.run_semantic_grounded_handoff import publish
    from tools.semantic_grounded_development_archive import MAX_ROW_BYTES, digest
    from tools.verify_semantic_grounded_evaluation import document, verify

    supervisor, report, native, directory = (path.resolve() for path in (
        args.evaluation_supervisor, args.report, args.native_directory, args.directory))
    supervised = json.loads(read_stable_bytes(supervisor / PLAN_FILE, max_bytes=MAX_ROW_BYTES))
    _verify_plan(supervised, supervisor / PLAN_FILE)
    evaluation_plan = document(json.loads(read_stable_bytes(report.with_suffix(".plan.json"),
        max_bytes=MAX_ROW_BYTES)), "plan_sha256")
    if (supervised["plan_sha256"] != args.evaluation_supervisor_plan_sha256
            or evaluation_plan["plan_sha256"] != args.evaluation_plan_sha256):
        raise ValueError("adjudication prerequisite differs from the declared exact plans")
    output = directory / "adjudication.json"
    arguments = adjudication_arguments(supervised, report=report, native_directory=native, output=output)
    scripts = {name: hashlib.sha256((ROOT / "tools" / name).read_bytes()).hexdigest() for name in (
        "run_semantic_grounded_adjudication_handoff.py", "adjudicate_semantic_grounded_development.py",
        "verify_semantic_grounded_evaluation.py", "semantic_grounded_development_archive.py")}
    grading = adjudication_contract()
    configure_refit_environment(directory / "handoff.json")
    directory.mkdir(parents=True, exist_ok=False)
    handoff = publish(directory / "handoff.json", {
        "schema": "aura.grounded_adjudication_handoff_plan.v1", "evaluation_supervisor": str(supervisor),
        "evaluation_supervisor_plan_sha256": supervised["plan_sha256"],
        "evaluation_command_sha256": supervised["command_sha256"],
        "evaluation_plan_sha256": evaluation_plan["plan_sha256"], "scripts": scripts,
        "adjudication_contract": grading, "arguments": arguments, "wait_seconds": args.wait_seconds,
        "duplicate_decode_allowed": False, "backbone_loaded": False,
        "qualification_evidence": False, "serving_authority": False})
    print(json.dumps({"stage": "adjudication_wait", "handoff_receipt_sha256": handoff["receipt_sha256"]}), flush=True)
    terminal = wait_for_evaluation(supervisor, supervised, timeout=args.wait_seconds)
    if (adjudication_contract() != grading or any(hashlib.sha256((ROOT / "tools" / name).read_bytes()).hexdigest() != sha
            for name, sha in scripts.items())):
        raise ValueError("adjudication implementation changed during its prerequisite wait")
    verified = verify(report, native)
    if terminal["returncode"] != (0 if verified["advance_development"] else 2):
        raise ValueError("evaluation process exit disagrees with independently verified verdict")
    publish(directory / "verification.json", {"schema": "aura.grounded_adjudication_prerequisite.v1",
        "handoff_receipt_sha256": handoff["receipt_sha256"],
        "evaluation_terminal_receipt_sha256": terminal["receipt_sha256"], "verification": verified,
        "qualification_evidence": False, "serving_authority": False})
    print(json.dumps({"stage": "adjudication_evaluation_verified", "advance_development": verified["advance_development"]}), flush=True)
    if adjudicate(arguments) != 0:
        raise ValueError("independent archived-program adjudication failed")
    measured = document(json.loads(read_stable_bytes(output, max_bytes=MAX_ROW_BYTES)), "receipt_sha256")
    body = {"schema": "aura.grounded_adjudication_handoff_completion.v1",
        "handoff_receipt_sha256": handoff["receipt_sha256"],
        "evaluation_terminal_receipt_sha256": terminal["receipt_sha256"],
        "verification_receipt_sha256": verified["receipt_sha256"],
        "adjudication_receipt_sha256": measured["receipt_sha256"], "metrics": measured["metrics"],
        "advance_development": verified["advance_development"], "g03_complete": False,
        "backbone_loaded": False, "qualification_evidence": False, "serving_authority": False}
    completed = publish(directory / "completion.json", body)
    print(json.dumps({"stage": "adjudication_complete", **completed}), flush=True)
    if completed["receipt_sha256"] != digest(body):
        raise ValueError("adjudication handoff publication differs from completion")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
