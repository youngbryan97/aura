"""Repair receipts cannot claim an omitted, skipped or stale regression."""
from __future__ import annotations

import hashlib
import json

import pytest

from core.organism import audit_repair_validation as repairs
from core.organism.model_validation import Evidence, ValidationSuite
from core.organism.nothing_measured import NothingMeasured


def _receipt(root, monkeypatch, cases='<testcase classname="tests.test_sqlite_memory_readiness" name="forced_interleaving"/>'):
    monkeypatch.setattr(repairs, "ROOT", root)
    _, test, owners = repairs.CONTRACTS["sqlite_readiness"]
    hashes = {}
    for name in (*owners, test, "core/organism/audit_repair_validation.py"):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("source under test\n")
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    xml = f"<testsuites><testsuite>{cases}</testsuite></testsuites>".encode()
    path = root / repairs.JUNIT
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(xml)
    record = {"schema": "aura.audit_repair_regressions.v1", "execution": {"returncode": 0, "command": ["pytest", test]},
              "source_sha256": hashes, "junit_sha256": hashlib.sha256(xml).hexdigest()}
    (root / repairs.RECEIPT).write_text(json.dumps(record))


def test_current_recorded_regression_is_recognized(tmp_path, monkeypatch):
    _receipt(tmp_path, monkeypatch)
    assert repairs.repair_contract_holds("sqlite_readiness")


def test_changed_source_retracts_the_archived_result(tmp_path, monkeypatch):
    _receipt(tmp_path, monkeypatch)
    (tmp_path / "core/memory/sqlite_storage.py").write_text("changed")
    with pytest.raises(NothingMeasured, match="source changed"):
        repairs.repair_contract_holds("sqlite_readiness")


def test_a_pass_in_another_file_cannot_fill_an_omitted_test(tmp_path, monkeypatch):
    _receipt(tmp_path, monkeypatch, '<testcase classname="tests.other" name="unrelated"/>')
    with pytest.raises(NothingMeasured, match="no cases"):
        repairs.repair_contract_holds("sqlite_readiness")


def test_a_skip_is_not_repair_evidence(tmp_path, monkeypatch):
    _receipt(tmp_path, monkeypatch, '<testcase classname="tests.test_sqlite_memory_readiness" name="unmeasured"><skipped/></testcase>')
    assert not repairs.repair_contract_holds("sqlite_readiness")


def test_repair_claims_are_registered_as_offline_evidence():
    suite = ValidationSuite()
    repairs.install_audit_repair_claims(suite)
    assert len(suite._claims) == len(repairs.CONTRACTS)
    assert all(claim.evidence is Evidence.MEASURED_SYNTHETIC for claim in suite._claims.values())
