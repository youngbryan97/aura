"""Bind offline repair regressions to their recorded tests and exact source."""
from __future__ import annotations

import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

from core.organism.nothing_measured import NothingMeasured

ROOT = Path(__file__).resolve().parents[2]
RECEIPT = "docs/evidence/audit-repairs-2026-10-23.json"
JUNIT = "docs/evidence/audit-repairs-2026-10-23.xml"
CONTRACTS = {
    "sqlite_readiness": (
        "SQLite regression checks retain concurrent first-use writes and reject synchronous entry on an event loop.",
        "tests/test_sqlite_memory_readiness.py", ("core/memory/sqlite_storage.py",)),
    "state_snapshots": (
        "Registry regression checks preserve coherent copies and a bounded pending notification queue.",
        "tests/test_state_registry_owned_snapshots.py", ("core/state/state_registry.py",)),
    "report_ratings": (
        "Report regression checks leave echoed question scales and invalid numeric samples unmeasured.",
        "tests/test_report_ratings_are_public_measurements.py", ("core/subject/report_grounding.py",)),
    "claim_liveness": (
        "A failed liveness reader leaves its bound live claim unmeasured in regression checks.",
        "tests/test_claim_liveness_failure_is_unmeasured.py", ("core/organism/model_validation.py",)),
    "patch_interface": (
        "Patch regression checks reject changes to declared public call interfaces.",
        "tests/test_patch_checks_preserve_the_public_interface.py",
        ("core/self_modification/public_contract.py", "core/self_modification/formal_verifier.py", "core/self_modification/safe_pipeline.py")),
    "trainer_ownership": (
        "A timed-out trainer stop retains the running thread's ownership in regression checks.",
        "tests/test_world_model_stop_keeps_the_running_trainer_owned.py", ("core/world_model/learned_world_model.py",)),
    "causal_response": (
        "Two-bit intervention checks separate common input from bidirectional coupling and refuse oversized kernels.",
        "tests/test_interventions_do_not_condition_on_the_assignment.py",
        ("core/consciousness/causal_transition_response.py", "core/consciousness/phi_core.py")),
    "paired_controls": (
        "Paired steering checks distinguish raw-count wins from family-corrected control significance.",
        "tests/test_fitted_controls_are_compared_as_pairs.py",
        ("core/evaluation/fitted_steering_evaluation.py", "core/learning/cortex_migration_authority.py",
         "tools/qualify_fitted_steering.py")),
}


def _bounded_bytes(path: Path) -> bytes:
    with path.open("rb") as stream:
        data = stream.read(2 * 1024 * 1024 + 1)
    if len(data) > 2 * 1024 * 1024:
        raise ValueError("repair evidence exceeds its read budget")
    return data


def repair_contract_holds(name: str) -> bool:
    """An archived offline result holds only while its source and tests match."""
    _, test, owners = CONTRACTS[name]
    try:
        receipt = json.loads(_bounded_bytes(ROOT / RECEIPT))
        if receipt.get("schema") != "aura.audit_repair_regressions.v1":
            raise ValueError("unknown repair evidence schema")
        if receipt["execution"]["returncode"] != 0:
            return False
        if test not in receipt["execution"]["command"]:
            raise ValueError(f"repair run did not select the full test file: {test}")
        for path in (*owners, test, "core/organism/audit_repair_validation.py"):
            expected = receipt["source_sha256"][path]
            actual = hashlib.sha256(_bounded_bytes(ROOT / path)).hexdigest()
            if actual != expected:
                raise ValueError(f"repair evidence source changed: {path}")
        xml = _bounded_bytes(ROOT / JUNIT)
        if hashlib.sha256(xml).hexdigest() != receipt["junit_sha256"]:
            raise ValueError("repair test report changed")
        report = ET.fromstring(xml)
        classname = test.removesuffix(".py").replace("/", ".")
        cases = [case for case in report.iter("testcase") if case.get("classname") == classname]
        if not cases:
            raise ValueError(f"repair report contains no cases for {test}")
        return all(not any(case.find(tag) is not None for tag in ("failure", "error", "skipped")) for case in cases)
    except (OSError, ValueError, TypeError, KeyError, AttributeError, ET.ParseError) as exc:
        raise NothingMeasured(f"offline repair evidence unavailable: {exc}") from exc


def install_audit_repair_claims(suite) -> None:
    from core.organism.model_validation import (
        Claim,
        Evidence,
        Observation,
        ValidationTest,
        boolean_score,
    )

    for key, (statement, test, owners) in CONTRACTS.items():
        name = f"audit_repair_{key}"
        suite.add_test(ValidationTest(
            name=name, description=statement, required_capability="",
            observation=Observation(name=name, value=True, source=test,
                                    note="Recorded offline regression; source and test hashes are checked."),
            predict=lambda _model, contract=key: repair_contract_holds(contract),
            score=lambda predicted, observed: boolean_score(predicted, expected=observed.value, subject="offline repair contract"),
            owner=owners[0],
        ))
        suite.add_claim(Claim(
            statement=statement, test=name, owner=owners[0], asserted_in=RECEIPT,
            evidence=Evidence.MEASURED_SYNTHETIC,
            evidence_note="Constructed regression inputs at recorded source hashes; no broad live capability claim.",
        ))
