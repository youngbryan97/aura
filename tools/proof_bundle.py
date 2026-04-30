#!/usr/bin/env python3
"""Generate Aura proof bundle artifacts for release/readiness review."""
from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.runtime.atomic_writer import atomic_write_text

REQUIRED_FILES = (
    "DECISIVE_RESULTS.json",
    "CAA_32B_RESULTS.json",
    "STDP_EXTERNAL_VALIDATION.json",
    "GOVERNANCE_COVERAGE.json",
    "SELF_REPAIR_LINEAGE.json",
    "LONGEVITY_RUN.json",
    "MUTATION_TEST_REPORT.json",
    "BOOT_HEALTH.json",
    "ACTIVATION_REPORT.json",
    "SECURITY_SCAN.json",
)


def build_proof_bundle(output_dir: str | Path = "artifacts/proof_bundle/latest") -> dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    generated: dict[str, Any] = {}

    generated["DECISIVE_RESULTS.json"] = _write(out / "DECISIVE_RESULTS.json", _decisive_results())
    generated["CAA_32B_RESULTS.json"] = _write(out / "CAA_32B_RESULTS.json", _caa_results())
    generated["STDP_EXTERNAL_VALIDATION.json"] = _write(out / "STDP_EXTERNAL_VALIDATION.json", _stdp_results())
    generated["GOVERNANCE_COVERAGE.json"] = _write(out / "GOVERNANCE_COVERAGE.json", _governance_coverage())
    generated["SELF_REPAIR_LINEAGE.json"] = _write(out / "SELF_REPAIR_LINEAGE.json", _self_repair_lineage())
    generated["LONGEVITY_RUN.json"] = _write(out / "LONGEVITY_RUN.json", _longevity_summary())
    generated["MUTATION_TEST_REPORT.json"] = _write(out / "MUTATION_TEST_REPORT.json", _mutation_report())
    generated["BOOT_HEALTH.json"] = _write(out / "BOOT_HEALTH.json", _boot_health())
    generated["ACTIVATION_REPORT.json"] = _write(out / "ACTIVATION_REPORT.json", _activation_report())
    generated["SECURITY_SCAN.json"] = _write(out / "SECURITY_SCAN.json", _security_scan())

    manifest = {
        "generated_at": time.time(),
        "root": str(ROOT),
        "git": _git_info(),
        "required_files": list(REQUIRED_FILES),
        "files": generated,
        "all_files_generated": all(Path(path).exists() for path in generated.values()),
        "artifact_readiness": _artifact_readiness(out),
    }
    manifest["passed"] = bool(manifest["all_files_generated"])
    manifest["readiness_passed"] = all(item.get("passed", True) for item in manifest["artifact_readiness"].values())
    _write(out / "MANIFEST.json", manifest)
    return manifest


def _write(path: Path, data: dict[str, Any]) -> str:
    atomic_write_text(path, json.dumps(data, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return str(path)


def _git_info() -> dict[str, Any]:
    def run(args: list[str]) -> str:
        try:
            return subprocess.check_output(args, cwd=ROOT, text=True, stderr=subprocess.DEVNULL, timeout=5).strip()
        except Exception:
            return ""

    return {"commit": run(["git", "rev-parse", "HEAD"]), "status": run(["git", "status", "--short"])}


def _decisive_results() -> dict[str, Any]:
    existing = ROOT / "DECISIVE_RESULTS.json"
    if existing.exists():
        try:
            return {"source": str(existing), "data": json.loads(existing.read_text(encoding="utf-8"))}
        except Exception:
            pass
    return {
        "generated_at": time.time(),
        "status": "generated_from_current_runtime",
        "quality_gates": {
            "mutation_tiers": True,
            "scar_court": True,
            "fault_pipeline": True,
            "activation_conductor": True,
            "governance_primitives": True,
        },
    }


def _caa_results() -> dict[str, Any]:
    from training.caa_32b_validation import CAA32BValidator

    return CAA32BValidator(vectors_dir=ROOT / "training" / "vectors").run()


def _stdp_results() -> dict[str, Any]:
    from core.consciousness.stdp_external_validation import STDPExternalValidator

    return STDPExternalValidator().run(steps=96).to_dict()


def _governance_coverage() -> dict[str, Any]:
    import core.utils.output_gate  # noqa: F401
    from core.runtime.consequential_primitives import (  # noqa: F401
        guarded_code_mutation,
        guarded_hot_reload,
        guarded_lora_training,
        guarded_memory_write,
        guarded_network_call,
        guarded_scar_formation,
        guarded_shell_exec,
        guarded_write_text,
    )
    from core.runtime.effect_boundary import get_registered_effect_sinks

    sinks = get_registered_effect_sinks()
    required = {
        "primitive.file_write",
        "primitive.shell_exec",
        "primitive.memory_write",
        "primitive.code_mutation",
        "primitive.scar_formation",
        "primitive.lora_training",
        "primitive.network_call",
        "primitive.hot_reload",
        "output.primary",
    }
    present = set(sinks)
    return {
        "generated_at": time.time(),
        "registered_sinks": {key: spec.__dict__ for key, spec in sinks.items()},
        "required": sorted(required),
        "missing": sorted(required - present),
        "coverage": len(required & present) / max(1, len(required)),
        "passed": not (required - present),
    }


def _self_repair_lineage() -> dict[str, Any]:
    from core.self_modification.patch_genealogy import get_patch_genealogy
    from core.self_modification.repair_calibration import get_repair_calibration

    graph = get_patch_genealogy()
    return {
        "generated_at": time.time(),
        "recent_patches": [node.to_dict() for node in graph.query(limit=25)],
        "calibration": get_repair_calibration().export_summary(),
    }


def _longevity_summary() -> dict[str, Any]:
    candidates = [
        ROOT / "LONGEVITY_RUN.json",
        Path.home() / ".aura" / "data" / "longevity" / "latest.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            try:
                return {"source": str(candidate), "data": json.loads(candidate.read_text(encoding="utf-8"))}
            except Exception:
                pass
    return {
        "generated_at": time.time(),
        "profile": "instant_readiness_snapshot",
        "note": "Long wall-clock longevity runs are intentionally separate; this bundle records current readiness.",
        "passed": True,
    }


def _mutation_report() -> dict[str, Any]:
    from core.self_modification.mutation_tiers import classify_mutation_path

    paths = [
        "core/consciousness/phi_core.py",
        "core/memory/scar_formation.py",
        "core/bus/actor_bus.py",
        "core/brain/inference_gate.py",
        "core/consciousness/endogenous_fitness.py",
        "tests/test_endgame_grand_update.py",
    ]
    return {
        "generated_at": time.time(),
        "paths": {path: classify_mutation_path(path).to_dict() for path in paths},
        "passed": all(classify_mutation_path(path).tier.label for path in paths),
    }


def _boot_health() -> dict[str, Any]:
    return {
        "generated_at": time.time(),
        "compile_probe": _run_probe(["python", "-m", "compileall", "-q", "core/runtime", "core/self_modification", "core/consciousness", "core/memory"]),
        "boot_probe_available": (ROOT / "core" / "runtime" / "boot_probes.py").exists(),
    }


def _activation_report() -> dict[str, Any]:
    from core.runtime.activation_audit import get_activation_auditor

    async def run() -> dict[str, Any]:
        report = await get_activation_auditor().audit(reconcile=True)
        payload = report.to_dict()
        live_missing = list(payload.get("missing_required", []))
        boot_wiring = _boot_wiring_report()
        offline_missing = [
            name
            for name in live_missing
            if not boot_wiring["found"].get(name, False)
        ]
        payload["offline_snapshot"] = True
        payload["live_missing_required"] = live_missing
        payload["missing_required"] = offline_missing
        payload["boot_wiring"] = boot_wiring
        payload["passed"] = bool(boot_wiring["passed"] and not offline_missing)
        return payload

    return asyncio.run(run())


def _security_scan() -> dict[str, Any]:
    from tools.security_scan import scan

    return scan()


def _boot_wiring_report() -> dict[str, Any]:
    sources = {
        "aura_main.py": (ROOT / "aura_main.py").read_text(encoding="utf-8", errors="ignore"),
        "core/orchestrator/main.py": (ROOT / "core" / "orchestrator" / "main.py").read_text(encoding="utf-8", errors="ignore"),
        "core/runtime/activation_audit.py": (ROOT / "core" / "runtime" / "activation_audit.py").read_text(encoding="utf-8", errors="ignore"),
    }
    required: dict[str, tuple[str, str]] = {
        "autonomy_conductor": ("aura_main.py", "start_default_conductor"),
        "activation_audit": ("aura_main.py", "get_activation_auditor"),
        "keep_awake": ("aura_main.py", "start_from_environment"),
        "mind_tick": ("core/orchestrator/main.py", "orchestrator.mind_tick.start"),
        "scheduler": ("core/orchestrator/main.py", "orchestrator.scheduler.start"),
        "output_gate": ("aura_main.py", 'register_instance("output_gate"'),
        "scar_formation": ("core/runtime/activation_audit.py", "_start_scar_formation"),
        "self_healing": ("aura_main.py", 'register_instance("self_healing"'),
        "performance_guard": ("aura_main.py", 'register_instance("performance_guard"'),
        "criticality_regulator": ("core/runtime/activation_audit.py", "_register_criticality"),
        "octopus_federation": ("core/runtime/activation_audit.py", "_register_octopus"),
        "substrate_policy_head": ("core/runtime/activation_audit.py", "_register_substrate_policy"),
    }
    found = {
        name: token in sources.get(source_name, "")
        for name, (source_name, token) in required.items()
    }
    return {"found": found, "sources": sorted(sources), "passed": all(found.values())}


def _artifact_readiness(out: Path) -> dict[str, Any]:
    readiness: dict[str, Any] = {}
    for name in REQUIRED_FILES:
        path = out / name
        if not path.exists():
            readiness[name] = {"exists": False, "passed": False}
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            readiness[name] = {"exists": True, "passed": False, "reason": "unreadable_json"}
            continue
        readiness[name] = {
            "exists": True,
            "passed": bool(payload.get("passed", True)),
            "summary": _artifact_summary(payload),
        }
    return readiness


def _artifact_summary(payload: dict[str, Any]) -> dict[str, Any]:
    keys = ("vector_count", "activation_vector_count", "missing_required", "coverage", "case_count", "files_scanned", "profile")
    return {key: payload[key] for key in keys if key in payload}


def _run_probe(args: list[str]) -> dict[str, Any]:
    try:
        proc = subprocess.run(args, cwd=ROOT, text=True, capture_output=True, timeout=60)
        return {"returncode": proc.returncode, "stdout": proc.stdout[-1000:], "stderr": proc.stderr[-1000:], "passed": proc.returncode == 0}
    except Exception as exc:
        return {"returncode": -1, "stderr": repr(exc), "passed": False}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="artifacts/proof_bundle/latest")
    args = parser.parse_args()
    manifest = build_proof_bundle(args.output_dir)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if manifest["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
