"""Archived G11 proof and counterexamples to accidental false closure."""

import copy
import hashlib
import json
from pathlib import Path

import pytest

from core.organism.qualified_desktop_evidence import verify_archive

ARCHIVE = Path(__file__).resolve().parents[1] / (
    "artifacts/migration/27b/recovery/desktop-serving-20260914/delivery.json"
)


def _archive():
    return json.loads(ARCHIVE.read_text())


def test_observed_desktop_deliveries_replay_without_claiming_broad_gain():
    result = verify_archive(_archive())
    assert result["passed"]
    assert len(result["deliveries"]) == 4
    assert not result["broad_transfer_proven"]
    assert not result["runtime_health_proven"]


@pytest.mark.parametrize("field,value", [
    ("observed_desktop_text", "not the displayed answer"),
    ("prompt", "another task"),
    ("state", "running"),
    ("turn_id", "another turn"),
    ("expected_answer", {}),
    ("model_path", "retired checkpoint"),
    ("response_hash", "0" * 64),
])
def test_archive_rejects_drift(field, value):
    archive = _archive()
    archive["deliveries"][0][field] = value
    assert not verify_archive(archive)["passed"]


@pytest.mark.parametrize("field,value", [
    ("qualified_recurrent_path_proven", False),
    ("qualified_recurrent_delivery_errors", ["inactive"]),
    ("request_surface", "offline"),
    ("foreground_model_generation_consumed", True),
    ("response_path", "ordinary_decode"),
])
def test_resealed_journal_cannot_supply_missing_runtime_proof(field, value):
    archive = _archive()
    row = archive["deliveries"][0]
    payload = json.loads(row["response_json"])
    payload["live_turn_contract"][field] = value
    row["response_json"] = json.dumps(payload)
    row["response_hash"] = hashlib.sha256(row["response_json"].encode()).hexdigest()
    assert not verify_archive(archive)["passed"]


def test_no_duplicate_turn_or_omitted_family_can_close_serving():
    archive = _archive()
    archive["deliveries"].append(copy.deepcopy(archive["deliveries"][0]))
    assert not verify_archive(archive)["passed"]
    archive = _archive()
    archive["deliveries"].pop()
    assert not verify_archive(archive)["passed"]


def test_stale_runtime_source_cannot_close_serving():
    archive = _archive()
    archive["boot_observation"]["runtime_revision"]["source_current"] = False
    assert not verify_archive(archive)["passed"]
