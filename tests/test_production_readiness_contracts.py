from __future__ import annotations

import json

from tools.aura_production_readiness_gate import run_checks
from tools.build_provenance import build


def test_production_readiness_gate_contract_is_complete():
    checks = run_checks()
    failed = [check.name for check in checks if not check.passed]
    assert not failed
    assert len(checks) >= 25


def test_build_provenance_generates_sbom_and_materials(tmp_path):
    report = build(tmp_path)

    sbom = json.loads((tmp_path / "sbom.json").read_text(encoding="utf-8"))
    provenance = json.loads((tmp_path / "provenance.json").read_text(encoding="utf-8"))

    assert sbom["dependency_count"] == len(report["sbom"]["dependencies"])
    assert provenance["materials"]
    assert any(item["path"] == "pyproject.toml" for item in provenance["materials"])
