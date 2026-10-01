#!/usr/bin/env python3
"""Check an unfinished native optimizer generation without loading a backbone."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode()).hexdigest()


def verify(directory):
    import mlx.core as mx

    from core.learning.semantic_grounded_binding_engine import implementation_receipt
    from core.runtime.file_read_gateway import read_stable_bytes
    from tools.probe_semantic_native_prefix_branches import installed_arithmetic_basis

    directory = Path(directory)
    def read(name, limit):
        return read_stable_bytes(directory / name, max_bytes=limit)

    custody = directory.parent / (directory.name + "-native-custody")
    native = json.loads(read_stable_bytes(custody / "plan.json", max_bytes=16 * 1024 ** 2))
    plan = {key: value for key, value in native.items() if key != "plan_sha256"}
    owner = json.loads(read("fit-owner.json", 1024 ** 2))
    pointer = json.loads(read("resume.json", 16384))
    if (digest(plan) != native.get("plan_sha256")
            or plan.get("implementation") != implementation_receipt()
            or plan.get("installed_arithmetic") != installed_arithmetic_basis()
            or owner.get("schema") != "aura.grounded_binding_owner.v2"
            or re.fullmatch(r"[a-f0-9]{64}", owner.get("identity", "")) is None
            or sorted(owner.get("fit_ids", ())) != plan.get("fit_ids")
            or sorted(owner.get("calibration_ids", ())) != plan.get("calibration_ids")
            or any(not plan[name] or len(plan[name]) != len(set(plan[name])) for name in ("fit_ids", "calibration_ids"))
            or set(plan["fit_ids"]) & set(plan["calibration_ids"])
            or set(pointer) != {"schema", "file", "sha256", "identity", "step"}
            or pointer.get("schema") != "aura.grounded_restart_pointer.v1"
            or pointer.get("identity") != owner["identity"]
            or re.fullmatch(r"[a-f0-9]{64}", pointer.get("sha256", "")) is None
            or pointer.get("file") != f"resume-{pointer['sha256']}.safetensors"):
        raise ValueError("grounded restart plan, owner or pointer differs")
    options, step = plan["fit_options"], pointer["step"]
    if type(step) is not int or not 0 <= step <= options["steps"] or step % options["save_every"]:
        raise ValueError("grounded restart update is outside its declared schedule")
    payload = read(pointer["file"], 4 * 1024 ** 3)
    if hashlib.sha256(payload).hexdigest() != pointer["sha256"]:
        raise ValueError("grounded restart generation checksum differs")
    tensors, metadata = mx.load(io.BytesIO(payload), format="safetensors", return_metadata=True)
    state = json.loads(metadata["state"])
    if (state.get("schema") != "aura.grounded_optimizer_restart.v1"
            or state.get("identity") != owner["identity"] or state.get("step") != step
            or set(tensors) != set(state["inventory"])
            or any([list(value.shape), str(value.dtype)] != state["inventory"][key]
                or not mx.all(mx.isfinite(value)).item() for key, value in tensors.items())
            or any(key.split("/", 1)[0] not in {"model", "optimizer", "selected", "rng"}
                for key in tensors)
            or "optimizer/step" not in tensors or tensors["optimizer/step"].size != 1
            or tensors["optimizer/step"].item() != step
            or [row["step"] for row in state["history"]] != list(range(options["save_every"], step + 1, options["save_every"]))
            or any(not math.isfinite(row["calibration_loss"]) for row in state["history"])
            or len(state["best"]) != 2 or not math.isfinite(state["best"][0])
            or type(state["best"][1]) is not int or not 0 <= state["best"][1] <= step
            or not state["log_weights"] or any(not math.isfinite(value) for value in state["log_weights"].values())):
        raise ValueError("grounded restart state, inventory or progress differs")
    partitions = {name: {key[len(name) + 1:]: value for key, value in tensors.items()
        if key.startswith(name + "/")} for name in ("model", "optimizer", "selected", "rng")}
    if (any(not values for values in partitions.values())
            or set(partitions["rng"]) != {str(index) for index in range(len(mx.random.state))}
            or any(value.shape != mx.random.state[int(key)].shape or value.dtype != mx.random.state[int(key)].dtype
                for key, value in partitions["rng"].items())
            or set(partitions["selected"]) != {key for key in partitions["model"] if not key.startswith("nuisance.")}
            or any(value.shape != partitions["model"][key].shape or value.dtype != partitions["model"][key].dtype
                for key, value in partitions["selected"].items())
            or not any(key.startswith("native_suffix.") for key in partitions["selected"])):
        raise ValueError("grounded restart tensor ownership differs")
    optimizer = partitions["optimizer"]
    expected_optimizer = {"step", "learning_rate", *(key + suffix for key in partitions["model"] for suffix in (".m", ".v"))}
    if (set(optimizer) != expected_optimizer
            or any(optimizer[key + suffix].shape != value.shape for key, value in partitions["model"].items()
                for suffix in (".m", ".v"))
            or optimizer["learning_rate"].size != 1
            or not mx.allclose(optimizer["learning_rate"], mx.array(options["learning_rate"]), atol=0., rtol=1e-6).item()
            or json.loads(read("resume.json", 16384)) != pointer):
        raise ValueError("grounded restart Adam state or stable pointer differs")
    if (custody / "completion.json").is_file():
        from tools.verify_semantic_grounded_fit import verify as verify_complete
        verify_complete(directory)
    body = {"schema": "aura.grounded_restart_verification.v1", "plan_sha256": native["plan_sha256"],
        "fit_identity": owner["identity"], "generation_sha256": pointer["sha256"], "step": step,
        "generation_integrity_verified": True, "native_geometry_verification": "required_by_resume_loader",
        "report_exists": (directory / "report.json").is_file(),
        "completion_exists": (custody / "completion.json").is_file(),
        "model_weights_loaded": False, "held_sources_scored": False, "semantic_success": None,
        "qualification_evidence": False, "serving_authority": False}
    return {**body, "receipt_sha256": digest(body)}


def detached_verdict(result, context):
    if (context.get("transport") != "stdout-v3" or type(context.get("prior_attempt")) is not int
            or context["prior_attempt"] < 1
            or any(re.fullmatch(r"[a-f0-9]{64}", context.get(name, "")) is None
                for name in ("plan_sha256", "command_sha256", "prior_journal_head_sha256"))):
        raise ValueError("grounded restart needs a bound detached attempt")
    bindings = {key: value for key, value in context.items() if key != "transport"}
    evidence = {"schema": "aura.detached_step.resume_evidence.v2", **bindings,
        "checkpoint_sequence": result["step"], "grounded_restart": result}
    evidence_sha = digest(evidence)
    identity = digest({"prior_attempt": context["prior_attempt"],
        "prior_journal_head_sha256": context["prior_journal_head_sha256"],
        "checkpoint_sequence": result["step"], "evidence_sha256": evidence_sha})
    return {"schema": "aura.detached_step.resume_verdict.v3", **bindings, "evidence": evidence,
        "evidence_sha256": evidence_sha, "checkpoint_sequence": result["step"], "checkpoint_identity": identity,
        "verdict": "already_completed" if result["completion_exists"] else "safe_to_resume"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--detached", action="store_true")
    args = parser.parse_args()
    result = verify(args.directory)
    if args.detached:
        result = detached_verdict(result, {
            "plan_sha256": os.environ.get("AURA_DETACHED_PLAN_SHA256", ""),
            "command_sha256": os.environ.get("AURA_DETACHED_COMMAND_SHA256", ""),
            "prior_attempt": int(os.environ.get("AURA_DETACHED_PRIOR_ATTEMPT", "0")),
            "prior_journal_head_sha256": os.environ.get("AURA_DETACHED_PRIOR_JOURNAL_HEAD_SHA256", ""),
            "transport": os.environ.get("AURA_DETACHED_RESUME_EVIDENCE_TRANSPORT", "")})
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
