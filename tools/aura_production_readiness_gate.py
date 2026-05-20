#!/usr/bin/env python3
"""Executable production-readiness contract for Aura.

This gate is intentionally lightweight and stdlib-only.  It does not replace
the long 24h/72h longevity runs; it verifies that every non-longevity
production control has a live code, test, workflow, or runbook anchor.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@dataclass
class Check:
    name: str
    passed: bool
    detail: str


def _read(rel: str) -> str:
    path = ROOT / rel
    return path.read_text(encoding="utf-8", errors="ignore") if path.exists() else ""


def _exists(rel: str) -> bool:
    return (ROOT / rel).exists()


def _contains(rel: str, *needles: str) -> bool:
    text = _read(rel).lower()
    return all(needle.lower() in text for needle in needles)


def _any_file(pattern: str) -> bool:
    return any(ROOT.glob(pattern))


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".aura_gate_", dir=str(path.parent), text=True)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except OSError:
        tmp.unlink(missing_ok=True)
        raise


def _architecture_map_smoke() -> tuple[bool, str]:
    try:
        from tools.arch_map import ARCH_MAP_SCHEMA, build_architecture_report

        report = build_architecture_report()
        required_surfaces = {
            "will_decision",
            "memory_write",
            "state_mutation",
            "tool_execution",
            "patching",
            "llm_call",
            "external_io",
        }
        surfaces = set(report.get("operational_surfaces", {}))
        missing = sorted(required_surfaces - surfaces)
        passed = (
            report.get("schema") == ARCH_MAP_SCHEMA
            and int(report.get("totals", {}).get("subsystems", 0)) > 0
            and not missing
        )
        detail = (
            f"{report.get('totals', {}).get('subsystems', 0)} subsystems, "
            f"{len(report.get('dependency_edges', []))} dependency edges, "
            f"{len(surfaces)} operational surfaces"
        )
        if missing:
            detail += f"; missing surfaces: {', '.join(missing)}"
        return passed, detail
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def run_checks() -> list[Check]:
    checks: list[Check] = []

    def add(name: str, passed: bool, detail: str) -> None:
        checks.append(Check(name, bool(passed), detail))

    makefile = _read("Makefile")
    release = _read(".github/workflows/release.yml")
    pyproject = _read("pyproject.toml")
    operator = _read("docs/OPERATOR_GUIDE.md")

    add("clean_clone_install_documented", "make setup" in operator and "make quality" in operator, "Operator guide includes setup and quality path")
    add("compile_gate", "compile:" in makefile and "compileall" in makefile, "Makefile compile target uses compileall")
    add("pytest_collection_gate", "enterprise-collect:" in makefile, "Fast pytest collection gate is present")
    add("full_test_gate", re.search(r"^test:", makefile, re.MULTILINE) is not None, "Makefile test target is present")
    add("source_hygiene_gate", "source-hygiene:" in makefile and "__pycache__" in makefile and "*.py[cod]" in _read(".gitignore"), "Tracked cache artifacts are blocked from release snapshots")
    add("ruff_format_type_security_gates", all(item in makefile for item in ("lint:", "typecheck:", "security:", "governance-lint:")), "Quality gates include lint/type/security/governance")
    add("whole_surface_lint_bug_gate", all(token in makefile for token in ("RUFF_SURFACE_TARGETS", "RUFF_CRITICAL_TARGETS", "F821,F822,F823,F601", "--select E9")), "Ruff covers whole-surface syntax and production undefined-name/repeated-key bugs")
    add("production_anchor_typecheck", all(token in makefile for token in ("core/runtime/atomic_writer.py", "core/consciousness/continuous_experience.py", "tools/build_provenance.py")), "Mypy covers runtime/proof anchors beyond the legacy curated slice")
    add("quality_runs_enterprise_collect", re.search(r"^quality:.*enterprise-collect", makefile, re.MULTILINE) is not None, "Quality target includes pytest collection")
    add("enterprise_static_gate", _exists("tools/aura_enterprise_gate.py") and _exists("config/aura_enterprise_gate_baseline.json"), "Enterprise ratchet gate and baseline exist")
    add("governance_bypass_sweep", _contains("tools/lint_governance.py", "CONSEQUENTIAL_CALLS", "ALLOW_LIST", "governance lint"), "Governance lint scans direct consequential calls")
    add("proof_bundle_regeneration", _exists("tools/proof_bundle.py") and "proof-bundle:" in makefile and "readiness_passed" in _read("tools/proof_bundle.py"), "Proof bundle tool fails closed on artifact readiness")
    add("runtime_authentication", _contains("interface/auth.py", "compare_digest", "Authentication not configured", "validate_runtime_security_request"), "Runtime requests fail closed without auth")
    add("authorization_gate", _contains("core/executive/authority_gateway.py", "authorize_tool_execution", "authorize_memory_write", "authorize_state_mutation"), "Authority gateway covers tool/memory/state effects")
    add("secret_management", _contains("core/zenith_secrets.py", "Keychain", "get_secret", "store_credential"), "Secrets resolve through environment/Keychain helpers")
    add("secret_scan", _exists("tools/security_scan.py") and "secret_like_literal" in _read("tools/security_scan.py"), "Secret-like literal scan is present")
    add("dependency_sbom_provenance", _exists("tools/build_provenance.py") and "provenance:" in makefile and "sbom" in release.lower(), "Provenance/SBOM generation is wired")
    add("signed_release_required", all(token in release for token in ("codesign", "notarytool", "stapler", "Require signing credentials")), "Release workflow requires signing/notarization")
    add("sandbox_escape_tests", _exists("tests/test_sandbox_hardening.py") and _exists("tests/test_local_sandbox_hardening.py"), "Sandbox hardening suites exist")
    add("red_team_tests", _exists("tests/unity/test_unity_adversarial_prompting.py") and _exists("tools/behavioral_proof_smoke.py"), "Adversarial prompting and behavioral proof smoke exist")
    add("privacy_controls", _exists("interface/routes/privacy.py") and _exists("docs/DATA_RETENTION_DELETION_POLICY.md"), "Privacy routes and deletion policy exist")
    add("deployment_observability", _exists("docs/SLO.md") and _exists(".github/workflows/slo-gate.yml") and "dashboard" in operator.lower(), "SLO and dashboard observability documented")
    add("incident_response", _exists("docs/OPERATOR_GUIDE.md") and _exists("docs/runbooks/dirty-shutdown-recovery.md"), "Operator guide and incident runbooks exist")
    add("reproducible_builds", _exists("tools/build_provenance.py") and "git_commit" in _read("tools/build_provenance.py"), "Build provenance captures commit and materials")
    add("load_performance_tests", _exists("tests/test_load_stress.py") and _exists("tests/performance/locustfile.py"), "Load/stress suites exist")
    add("secure_update_rollback", _contains("core/runtime/release_channels.py", "rollback_pass", "stable") and _contains("docs/OPERATOR_GUIDE.md", "rollback"), "Release policy and runbooks require rollback")
    add("model_provider_failure_policy", _exists("docs/MODEL_PROVIDER_FAILURE_POLICY.md") and _exists("docs/runbooks/model-fails-to-load.md"), "Model/provider failure policy and runbook exist")
    add("memory_state_atomic_replayable", _exists("core/runtime/atomic_writer.py") and _exists("core/consciousness/continuous_experience.py"), "Atomic writer and replayable experience stream exist")
    arch_map_ok, arch_map_detail = _architecture_map_smoke()
    add("operational_architecture_dependency_map", arch_map_ok, arch_map_detail)
    add("crash_restart_recovery", _exists("docs/runbooks/dirty-shutdown-recovery.md") and _exists("docs/runbooks/checkpoint-restore-failed.md"), "Crash/restart recovery runbooks exist")
    add("failure_degrades_honestly", _contains("core/runtime/errors.py", "record_degradation") and _contains("core/unity/runtime.py", "record_degradation"), "Runtime records degraded paths instead of hiding them")
    add("continuous_experience_learning", _exists("core/consciousness/continuous_experience.py") and _exists("tests/test_continuous_experience_stream.py"), "Movie-like stream is implemented and tested")
    add("quality_tooling_declared", all(item in pyproject for item in ("[tool.ruff]", "[tool.mypy]", "[tool.bandit]")), "Ruff, mypy, and bandit configs exist")
    add("production_workflow", _exists(".github/workflows/production-readiness.yml"), "CI runs the production readiness gate")
    add("policy_docs", all(_exists(path) for path in ("docs/PRODUCTION_READINESS_STANDARD.md", "docs/DATA_RETENTION_DELETION_POLICY.md", "docs/MODEL_PROVIDER_FAILURE_POLICY.md")), "Production policy documents exist")
    add("focused_contract_tests", _exists("tests/test_production_readiness_contracts.py"), "Production readiness contract tests exist")
    add("requirements_declared", _any_file("requirements/*.txt"), "Requirements files exist for install review")
    return checks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)
    checks = run_checks()
    report = {
        "generated_at": time.time(),
        "passed": all(check.passed for check in checks),
        "checks": [asdict(check) for check in checks],
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.out:
        _atomic_write_text(Path(args.out), text)
    print(text)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
