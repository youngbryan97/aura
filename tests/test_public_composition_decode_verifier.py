"""Synthetic channel receipts exercise the verifier, not model qualification."""

import hashlib
import json
from pathlib import Path

import pytest

from core.brain.llm.public_channel_decode import PUBLIC_CHANNEL_DECODE_POLICY, PublicChannelDecode
from tools import run_semantic_neural_composition_decode_canary as runner
from tools import verify_semantic_neural_composition_decode_canary as verifier

ROOT = Path(__file__).resolve().parents[1]
HISTORICAL = ROOT / "artifacts/closeout/latent_cortex/typed_composition_decode_canary_20260831/result.json"


def fixture(tmp_path, monkeypatch):
    payload = json.loads(HISTORICAL.read_bytes())
    # This unit test isolates channel/resource verification from live identity.
    monkeypatch.setattr(verifier, "_manifest", lambda identity, model: identity)
    monkeypatch.setattr(verifier, "_git_blob_sha", lambda *_: "a" * 64)
    payload.update(schema=runner.PUBLIC_SCHEMA, decode_policy=PUBLIC_CHANNEL_DECODE_POLICY,
                   source_sha256s={p: "a" * 64 for p in runner.PUBLIC_SOURCE_PATHS},
                   max_tokens=4096, censored_decodes=0)
    for row in payload["rows"]:
        row.update(generated_tokens=32, latency_ms=10, stopped=True)
        row["decode"] = PublicChannelDecode(
            row["response"], 32, row["prefill_tokens"], 0, 10, "eos", False, True, 0,
            hashlib.sha256(b"").hexdigest(), "a" * 64,
        ).receipt()
    payload["arms"] = {arm: runner._summary(payload["rows"], arm) for arm in runner.ARMS}
    return payload


def journal(payload, path):
    payload["journal_path"] = str(path)
    previous = runner._append_journal_event(path, {
        "event": "campaign_started", "source_commit": payload["source_commit"],
        "seed": payload["seed"], "task_count": payload["task_count"], "arm_count": len(runner.ARMS),
        "max_tokens": payload["max_tokens"], "model_identity": payload["model_identity"],
        "decode_policy": payload["decode_policy"],
    }, previous_receipt_sha256="0" * 64)
    for index, row in enumerate(payload["rows"], 1):
        previous = runner._append_journal_event(path, {
            "event": "decode_committed", "completed": index, "total": len(payload["rows"]), "row": row,
        }, previous_receipt_sha256=previous)
    payload["journal_last_decode_receipt_sha256"] = previous
    payload["receipt_sha256"] = runner._sha({k: v for k, v in payload.items() if k != "receipt_sha256"})
    runner._append_journal_event(path, {
        "event": "campaign_completed", "admitted": payload["admitted"],
        "report_receipt_sha256": payload["receipt_sha256"],
    }, previous_receipt_sha256=previous)


def test_public_composition_verifier_reads_channel_receipts(tmp_path, monkeypatch):
    payload = fixture(tmp_path, monkeypatch)
    path = tmp_path / "journal.jsonl"
    journal(payload, path)
    result = verifier.verify(payload, journal_path=path)
    assert result["verified"]
    assert result["censored_decodes"] == 0


@pytest.mark.parametrize("mutation", ["private", "censored", "count", "policy", "downgrade"])
def test_public_composition_rejects_resealed_mismeasurement(tmp_path, monkeypatch, mutation):
    payload = fixture(tmp_path, monkeypatch)
    row = payload["rows"][0]
    if mutation == "private":
        row["decode"].update(boundary_closed=False, stop_reason="public_contract")
    elif mutation == "censored":
        row["decode"].update(stop_reason="token_limit", generated_tokens=4096)
        row.update(generated_tokens=4096, stopped=False)
        payload["censored_decodes"] = 1
        payload["arms"] = {arm: runner._summary(payload["rows"], arm) for arm in runner.ARMS}
    elif mutation == "count":
        row["decode"]["generated_tokens"] += 1
    elif mutation == "policy":
        payload["decode_policy"] = "unknown"
    else:
        payload["schema"] = runner.SCHEMA
    path = tmp_path / "journal.jsonl"
    journal(payload, path)
    with pytest.raises((RuntimeError, ValueError)):
        verifier.verify(payload, journal_path=path)
