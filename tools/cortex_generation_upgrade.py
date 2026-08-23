#!/usr/bin/env python
"""Cortex generation upgrade — operator CLI for the governed base-model swap.

The pipeline evaluates a candidate generation (breadth + reasoning +
identity batteries), plans identity migration, stages the activation
pointer with a byte-exact rollback, and activates ONLY with an explicit
operator authorization plus a PASS verdict. Effect is at next boot; the
running mind is never hot-swapped.

    .venv/bin/python tools/cortex_generation_upgrade.py evaluate \
        --candidate ~/models/Qwen3.8-27B-4bit \
        --critical-gates artifacts/current/cortex_upgrade/critical_gates.json \
        --out artifacts/current/cortex_upgrade
    .venv/bin/python tools/cortex_generation_upgrade.py compare \
        --current-battery artifacts/current/cortex_upgrade/battery_current.json \
        --candidate-battery artifacts/current/cortex_upgrade/battery_candidate.json \
        --descriptor artifacts/current/cortex_upgrade/artifact_descriptor.json \
        --out artifacts/current/cortex_upgrade
    .venv/bin/python tools/cortex_generation_upgrade.py contracts \
        --candidate ~/models/Qwen3.8-27B-4bit \
        --repository mlx-community/Qwen3.8-27B-4bit --revision <sha> \
        --lane-limits artifacts/current/cortex_upgrade/lane_limits.json \
        --serving-qualification artifacts/current/cortex_upgrade/serving_qualification.json \
        --migration-components artifacts/current/cortex_upgrade/migration_components.json
    .venv/bin/python tools/cortex_generation_upgrade.py plan
    .venv/bin/python tools/cortex_generation_upgrade.py stage \
        --candidate ~/models/Qwen3.8-27B-4bit --base Qwen3.8-27B --tag qwen3.8 \
        --descriptor artifacts/current/cortex_upgrade/artifact_descriptor.json \
        --evaluation artifacts/current/cortex_upgrade/comparison.json \
        --serving-profile artifacts/current/cortex_upgrade/serving_profile.json \
        --migration-contract artifacts/current/cortex_upgrade/migration_contract.json
    .venv/bin/python tools/cortex_generation_upgrade.py activate \
        --authorized-by "bryan" --evaluation artifacts/current/cortex_upgrade/comparison.json
    .venv/bin/python tools/cortex_generation_upgrade.py rollback

MEMORY SAFETY: `evaluate` loads models and is guarded — it refuses when the
host cannot afford the candidate beside resident processes (live app or a
training run). Never force it past a refusal.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("AURA_LOG_DIR", str(Path.home() / ".aura" / "lab-logs"))


def _write(out_dir: Path, name: str, payload: dict) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    path.write_text(json.dumps(payload, indent=1, sort_keys=True), encoding="utf-8")
    print(f"📄 {path}")
    return path


def _evaluation_source_sha256() -> str:
    """Bind resumable cells to every source file that defines their meaning."""
    from core.brain.llm import chat_format
    from core.brain.llm.latent_cortex import experiment_tasks
    from core.learning import cortex_generation_upgrade, interference_battery

    digest = hashlib.sha256()
    for module in (
        cortex_generation_upgrade,
        interference_battery,
        experiment_tasks,
        chat_format,
    ):
        path = Path(module.__file__).resolve()
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _progress_recorder(
    out_dir: Path,
    *,
    label: str,
    model_path: str,
    descriptor_sha256: str,
):
    from core.runtime.atomic_writer import atomic_write_text

    path = out_dir / f"progress_{label}.json"
    binding = {
        "label": label,
        "model_path": str(Path(model_path).expanduser().resolve()),
        "model_descriptor_sha256": descriptor_sha256,
        "evaluation_source_sha256": _evaluation_source_sha256(),
    }
    journal = {
        "schema": "aura.cortex_upgrade.evaluation_journal.v1",
        "binding": binding,
        "events": [],
        "complete": False,
        "updated_at": time.time(),
    }
    if path.is_file():
        observed = json.loads(path.read_text(encoding="utf-8"))
        if observed.get("schema") != journal["schema"] or observed.get("binding") != binding:
            raise RuntimeError(f"evaluation_progress_binding_mismatch:{path}")
        if not isinstance(observed.get("events"), list):
            raise RuntimeError(f"evaluation_progress_events_invalid:{path}")
        journal = observed
        print(f"↻ resuming {label}: {len(journal['events'])} durable cell(s)", flush=True)

    def persist() -> None:
        journal["updated_at"] = time.time()
        atomic_write_text(
            path,
            json.dumps(journal, indent=1, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def record(event: dict) -> None:
        events_by_cell = {
            existing.get("cell_id"): index
            for index, existing in enumerate(journal["events"])
            if isinstance(existing, dict)
        }
        cell_id = event.get("cell_id")
        if cell_id in events_by_cell:
            journal["events"][events_by_cell[cell_id]] = event
        else:
            journal["events"].append(event)
        persist()
        print(
            f"  {label} {event.get('phase')} "
            f"{event.get('completed')}/{event.get('total')}",
            flush=True,
        )

    return journal, record, persist


@contextmanager
def _model_session(path: str):
    from mlx_lm import load

    from core.runtime.model_lane_control import standalone_model_lane

    with standalone_model_lane(
        owner_id=f"cortex-generation-upgrade:{Path(path).name}",
        model_path=path,
        purpose="evaluation",
        preemptible=False,
        metadata={"tool": "cortex_generation_upgrade", "operator_launched": True},
    ):
        yield load(path)


def cmd_evaluate(args) -> int:
    from core.brain.llm.model_artifact_profile import build_model_artifact_descriptor
    from core.learning.cortex_generation_upgrade import (
        MemoryGuard,
        capability_battery,
        compare_batteries,
    )

    out_dir = Path(args.out)
    guard = MemoryGuard()
    admission = guard.admit(args.candidate)
    _write(out_dir, "admission.json", admission)
    if not admission["admitted"]:
        print(f"🚫 refused: {admission.get('refusal_reason')}")
        return 2

    receipts = {}
    descriptors = {}
    for label, model_path in (("current", args.current), ("candidate", args.candidate)):
        if not model_path:
            continue
        descriptors[label] = build_model_artifact_descriptor(
            model_path,
            repository_id=args.repository if label == "candidate" else "",
            revision=args.revision if label == "candidate" else "",
        )
        _write(out_dir, f"artifact_descriptor_{label}.json", descriptors[label])
        journal, record_progress, persist_progress = _progress_recorder(
            out_dir,
            label=label,
            model_path=model_path,
            descriptor_sha256=str(descriptors[label]["descriptor_sha256"]),
        )
        print(f"▶ loading {label}: {model_path}", flush=True)
        with _model_session(model_path) as (model, tokenizer):
            receipts[label] = capability_battery(
                model,
                tokenizer,
                label=label,
                progress_callback=record_progress,
                resume_events=journal["events"],
            )
            del model, tokenizer
        _write(out_dir, f"battery_{label}.json", receipts[label])
        journal["complete"] = True
        journal["receipt_sha256"] = hashlib.sha256(
            json.dumps(
                receipts[label],
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        persist_progress()
        try:
            import mlx.core as mx

            mx.clear_cache()
        except (ImportError, AttributeError):
            pass
    if "current" in receipts and "candidate" in receipts:
        descriptor = descriptors["candidate"]
        _write(out_dir, "artifact_descriptor.json", descriptor)
        critical_gates = {}
        if args.critical_gates:
            critical_gates = json.loads(Path(args.critical_gates).read_text())
        comparison = compare_batteries(
            receipts["current"],
            receipts["candidate"],
            candidate_descriptor=descriptor,
            critical_gates=critical_gates,
        )
        _write(out_dir, "comparison.json", comparison)
        print(f"VERDICT: {comparison['verdict']} "
              f"(breadth {comparison['breadth_delta']:+.3f}, "
              f"reasoning {comparison['reasoning_delta']:+.3f})")
        return 0 if comparison["promotion_eligible"] else 1
    return 0


def cmd_compare(args) -> int:
    """Re-adjudicate immutable battery receipts without loading either model."""
    from core.learning.cortex_generation_upgrade import (
        EVALUATION_SCHEMA,
        compare_batteries,
    )

    def read_object(path: str, *, label: str) -> dict:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"{label}_must_be_json_object")
        return value

    current = read_object(args.current_battery, label="current_battery")
    candidate = read_object(args.candidate_battery, label="candidate_battery")
    descriptor = read_object(args.descriptor, label="descriptor")
    for label, receipt in (("current", current), ("candidate", candidate)):
        if receipt.get("schema") != EVALUATION_SCHEMA:
            raise ValueError(f"{label}_battery_schema_mismatch")
    critical_gates = (
        read_object(args.critical_gates, label="critical_gates")
        if args.critical_gates
        else {}
    )
    comparison = compare_batteries(
        current,
        candidate,
        candidate_descriptor=descriptor,
        critical_gates=critical_gates,
    )
    _write(Path(args.out), "comparison.json", comparison)
    print(
        f"VERDICT: {comparison['verdict']} "
        f"(breadth {comparison['breadth_delta']:+.3f}, "
        f"reasoning {comparison['reasoning_delta']:+.3f}; "
        f"promotion_eligible={comparison['promotion_eligible']})"
    )
    return 0 if comparison["verdict"] == "PASS" else 1


def cmd_contracts(args) -> int:
    from core.brain.llm.model_artifact_profile import (
        build_model_artifact_descriptor,
        build_model_serving_profile,
    )
    from core.learning.cortex_generation_upgrade import build_migration_contract

    out_dir = Path(args.out)
    descriptor = build_model_artifact_descriptor(
        args.candidate,
        repository_id=args.repository,
        revision=args.revision,
    )
    lane_limits = json.loads(Path(args.lane_limits).read_text())
    qualification = json.loads(Path(args.serving_qualification).read_text())
    components = json.loads(Path(args.migration_components).read_text())
    serving = build_model_serving_profile(
        descriptor,
        served_context_tokens=args.served_context,
        prefill_chunk_tokens=args.prefill_chunk,
        lane_limits=lane_limits,
        qualification=qualification,
    )
    migration = build_migration_contract(descriptor, components=components)
    _write(out_dir, "artifact_descriptor.json", descriptor)
    _write(out_dir, "serving_profile.json", serving)
    _write(out_dir, "migration_contract.json", migration)
    return 0


def cmd_plan(args) -> int:
    from core.learning.cortex_generation_upgrade import build_migration_plan

    plan = build_migration_plan()
    _write(Path(args.out), "migration_plan.json", plan)
    for step in plan["steps"]:
        marker = "•" if step["lane"] == "automatic" else "◦"
        print(f" {marker} {step['name']} [{step['lane']}] exists={step['exists']}")
    return 0


def cmd_stage(args) -> int:
    from core.learning.cortex_generation_upgrade import stage_upgrade

    evaluation = json.loads(Path(args.evaluation).read_text())
    descriptor = json.loads(Path(args.descriptor).read_text())
    serving_profile = json.loads(Path(args.serving_profile).read_text())
    migration_contract = json.loads(Path(args.migration_contract).read_text())
    receipt = stage_upgrade(
        candidate_model_path=args.candidate,
        base_model_path=args.base,
        tag=args.tag,
        evaluation=evaluation,
        artifact_descriptor=descriptor,
        serving_profile=serving_profile,
        migration_contract=migration_contract,
    )
    _write(Path(args.out), "staging.json", receipt)
    return 0


def cmd_activate(args) -> int:
    from core.learning.cortex_generation_upgrade import activate_upgrade

    evaluation = json.loads(Path(args.evaluation).read_text())
    receipt = activate_upgrade(
        authorized_by=args.authorized_by, evaluation=evaluation
    )
    _write(Path(args.out), "activation.json", receipt)
    print("⚠️  effective at NEXT BOOT — restart Aura to think on the new cortex")
    return 0


def cmd_rollback(args) -> int:
    from core.learning.cortex_generation_upgrade import rollback_upgrade

    receipt = rollback_upgrade()
    _write(Path(args.out), "rollback.json", receipt)
    return 0 if receipt["byte_exact"] else 1


def cmd_status(args) -> int:
    from core.brain.llm.model_registry import BASE_DIR
    from core.learning.cortex_generation_upgrade import (
        ROLLBACK_POINTER_NAME,
        STAGED_POINTER_NAME,
    )

    fused = Path(BASE_DIR) / "training" / "fused-model"
    status = {
        "active": json.loads((fused / "active.json").read_text())
        if (fused / "active.json").is_file()
        else None,
        "staged": json.loads((fused / STAGED_POINTER_NAME).read_text())
        if (fused / STAGED_POINTER_NAME).is_file()
        else None,
        "rollback_available": (fused / ROLLBACK_POINTER_NAME).is_file(),
        "checked_at": time.time(),
    }
    print(json.dumps(status, indent=2, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--candidate", required=True)
    evaluate.add_argument("--current", default="")
    evaluate.add_argument("--repository", default="")
    evaluate.add_argument("--revision", default="")
    evaluate.add_argument("--critical-gates", default="")
    evaluate.add_argument("--out", default="artifacts/current/cortex_upgrade")
    evaluate.set_defaults(func=cmd_evaluate)

    compare = sub.add_parser("compare")
    compare.add_argument("--current-battery", required=True)
    compare.add_argument("--candidate-battery", required=True)
    compare.add_argument("--descriptor", required=True)
    compare.add_argument("--critical-gates", default="")
    compare.add_argument("--out", default="artifacts/current/cortex_upgrade")
    compare.set_defaults(func=cmd_compare)

    contracts = sub.add_parser("contracts")
    contracts.add_argument("--candidate", required=True)
    contracts.add_argument("--repository", required=True)
    contracts.add_argument("--revision", required=True)
    contracts.add_argument("--served-context", type=int, required=True)
    contracts.add_argument("--prefill-chunk", type=int, required=True)
    contracts.add_argument("--lane-limits", required=True)
    contracts.add_argument("--serving-qualification", required=True)
    contracts.add_argument("--migration-components", required=True)
    contracts.add_argument("--out", default="artifacts/current/cortex_upgrade")
    contracts.set_defaults(func=cmd_contracts)

    plan = sub.add_parser("plan")
    plan.add_argument("--out", default="artifacts/current/cortex_upgrade")
    plan.set_defaults(func=cmd_plan)

    stage = sub.add_parser("stage")
    stage.add_argument("--candidate", required=True)
    stage.add_argument("--base", required=True)
    stage.add_argument("--tag", required=True)
    stage.add_argument("--descriptor", required=True)
    stage.add_argument("--evaluation", required=True)
    stage.add_argument("--serving-profile", required=True)
    stage.add_argument("--migration-contract", required=True)
    stage.add_argument("--out", default="artifacts/current/cortex_upgrade")
    stage.set_defaults(func=cmd_stage)

    activate = sub.add_parser("activate")
    activate.add_argument("--authorized-by", required=True)
    activate.add_argument("--evaluation", required=True)
    activate.add_argument("--out", default="artifacts/current/cortex_upgrade")
    activate.set_defaults(func=cmd_activate)

    rollback = sub.add_parser("rollback")
    rollback.add_argument("--out", default="artifacts/current/cortex_upgrade")
    rollback.set_defaults(func=cmd_rollback)

    status = sub.add_parser("status")
    status.set_defaults(func=cmd_status)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
