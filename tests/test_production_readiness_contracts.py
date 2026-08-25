from __future__ import annotations

import json

from tools.aura_production_readiness_gate import run_checks
from tools.build_provenance import build


def test_production_readiness_gate_contract_is_complete():
    checks = run_checks()
    failed = [check.name for check in checks if not check.passed]
    assert not failed
    assert len(checks) >= 35


def test_build_provenance_generates_sbom_and_materials(tmp_path):
    report = build(tmp_path)

    sbom = json.loads((tmp_path / "sbom.json").read_text(encoding="utf-8"))
    provenance = json.loads((tmp_path / "provenance.json").read_text(encoding="utf-8"))

    assert sbom["dependency_count"] == len(report["sbom"]["dependencies"])
    assert provenance["materials"]
    assert any(item["path"] == "pyproject.toml" for item in provenance["materials"])


def test_makefile_gates_whole_surface_lint_and_source_hygiene():
    makefile = open("Makefile", encoding="utf-8").read()
    strict_files = open("config/mypy_strict_files.txt", encoding="utf-8").read()

    assert "source-hygiene:" in makefile
    # Prerequisites of `quality`, parsed rather than matched as a fixed
    # substring. The substring pinned those three to the FRONT of the list, so
    # adding a gate before them failed a test about gates being present.
    quality = next(
        line for line in makefile.splitlines() if line.startswith("quality:")
    )
    prerequisites = set(quality.split(":", 1)[1].split())
    for gate in ("source-hygiene", "enterprise-gate", "enterprise-collect"):
        assert gate in prerequisites, gate
    assert "RUFF_SURFACE_TARGETS" in makefile
    assert "RUFF_CRITICAL_TARGETS" in makefile
    assert "F821,F822,F823,F601" in makefile
    assert "config/mypy_strict_files.txt" in makefile
    assert "$(MYPY_TARGETS)" in makefile
    assert "core/runtime/atomic_writer.py" in strict_files
    assert "core/consciousness/continuous_experience.py" in strict_files
    assert "tools/build_provenance.py" in strict_files


def test_final_proof_requires_live_desktop_runtime_evidence():
    makefile = open("Makefile", encoding="utf-8").read()

    assert "final-proof:" in makefile
    assert "--name live_desktop_runtime" in makefile
    assert "tools/live_boot_proof.py" in makefile
    assert "--mode desktop" in makefile
    assert "--restart-continuity" in makefile
    assert "--out-dir artifacts/current/live_desktop_runtime" in makefile
