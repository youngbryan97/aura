"""Historical termination evidence remains immutable and scope-limited."""

import json
from pathlib import Path

import pytest

from tools.audit_semantic_decode_termination import audit

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "artifacts/migration/27b/recovery/cp1003-semantic-canary/result.json"
JOURNAL = REPORT.with_name("journal.jsonl")


def test_audit_reads_all_historical_rows_without_inventing_native_channel_state():
    before = REPORT.read_bytes(), JOURNAL.read_bytes()
    result = audit(REPORT, JOURNAL)
    ordinary = result["arms"]["ordinary_base"]
    assert ordinary["count"] == 60
    assert ordinary["not_stopped"] == 18
    assert ordinary["not_stopped_token_counts"] == {"384": 18}
    assert ordinary["stopped_unparsed"] == 42
    assert ordinary["contract_reasons"] == {"no_marker": 18, "payload_not_json_object": 42}
    assert len(result["rows"]) == 300
    assert result["native_channel_state_recoverable"] is False
    assert result["serving_authority"] is False
    assert before == (REPORT.read_bytes(), JOURNAL.read_bytes())


def test_audit_rejects_a_modified_report(tmp_path):
    payload = json.loads(REPORT.read_bytes())
    payload["admitted"] = False
    report = tmp_path / "result.json"
    report.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="seal differs"):
        audit(report, JOURNAL)


def test_audit_rejects_a_modified_journal(tmp_path):
    lines = JOURNAL.read_text().splitlines()
    row = json.loads(lines[1])
    row["row"]["stopped"] = not row["row"]["stopped"]
    lines[1] = json.dumps(row)
    journal = tmp_path / "journal.jsonl"
    journal.write_text("\n".join(lines) + "\n")
    with pytest.raises(RuntimeError, match="receipt chain broke"):
        audit(REPORT, journal)
