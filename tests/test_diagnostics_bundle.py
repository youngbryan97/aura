"""Tests for `aura doctor --bundle`."""
from __future__ import annotations

import json
import tarfile
from pathlib import Path

import pytest

from core.runtime.diagnostics_bundle import (
    REDACTED,
    SENSITIVE_KEY_PATTERNS,
    SENSITIVE_VALUE_PATTERNS,
    build_bundle,
    redact_value,
)


# ---------------------------------------------------------------------------
# redaction unit tests
# ---------------------------------------------------------------------------
def test_redact_strips_known_secret_keys():
    payload = {
        "openai_api_key": "secret-value",
        "API_TOKEN": "another-secret",
        "private_key": "rsa-blob",
        "password": "p@ss",
        "credentials": {"slack_secret": "xx"},
        "auth_header": "Bearer 1234",
        "safe_field": "keep-me",
    }
    out = redact_value(payload)
    assert out["openai_api_key"] == REDACTED
    assert out["API_TOKEN"] == REDACTED
    assert out["private_key"] == REDACTED
    assert out["password"] == REDACTED
    assert out["credentials"] == REDACTED
    assert out["auth_header"] == REDACTED
    assert out["safe_field"] == "keep-me"


def test_redact_recognizes_high_entropy_tokens_in_values():
    payload = {
        "note": "leaking sk-ABCDEFGHIJKLMNOPQRSTUVWX in a casual log",
        "ok": "perfectly innocent",
    }
    out = redact_value(payload)
    assert out["note"] == REDACTED
    assert out["ok"] == "perfectly innocent"


def test_redact_walks_lists():
    payload = {"items": [{"token": "abc"}, {"name": "alice"}]}
    out = redact_value(payload)
    assert out["items"][0]["token"] == REDACTED
    assert out["items"][1]["name"] == "alice"


def test_redact_handles_pem_block():
    payload = {"cert_blob": "-----BEGIN PRIVATE KEY-----\nABCDEF\n-----END PRIVATE KEY-----"}
    out = redact_value(payload)
    assert out["cert_blob"] == REDACTED


def test_redact_handles_jwt_in_value():
    payload = {"hint": "auth=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxIn0.x"}
    out = redact_value(payload)
    assert out["hint"] == REDACTED


# ---------------------------------------------------------------------------
# bundle smoke tests
# ---------------------------------------------------------------------------
def test_bundle_produces_tarball_with_required_files(tmp_path: Path):
    out = tmp_path / "bundle.tar.gz"
    info = build_bundle(output_path=out, workspace=tmp_path / "ws")
    assert info["ok"] is True
    assert Path(info["path"]).exists()
    assert info["bytes"] > 0
    assert len(info["sha256"]) == 64

    with tarfile.open(info["path"], "r:gz") as tar:
        names = tar.getnames()

    # The bundle must always include a manifest.
    assert any(n.endswith("bundle_manifest.json") for n in names)
    # And at least the core collectors (whether they succeeded or not, an
    # error stub is written, so the names list will contain them).
    expected = {
        "config",
        "health",
        "metrics",
        "tasks",
        "models",
        "memory",
        "gateway",
        "receipts",
    }
    found = set()
    for n in names:
        for label in expected:
            if n.endswith(f"/{label}.json") or n.endswith(f"/{label}.error.txt"):
                found.add(label)
    assert expected.issubset(found), f"missing collectors: {expected - found}"


def test_bundle_manifest_lists_files_consistently(tmp_path: Path):
    out = tmp_path / "bundle.tar.gz"
    info = build_bundle(output_path=out, workspace=tmp_path / "ws")
    with tarfile.open(info["path"], "r:gz") as tar:
        manifest_member = next(
            m for m in tar.getmembers() if m.name.endswith("bundle_manifest.json")
        )
        f = tar.extractfile(manifest_member)
        assert f is not None
        manifest = json.loads(f.read().decode("utf-8"))

    assert "schema_version" in manifest
    assert "files" in manifest
    assert "bundle_manifest.json" in manifest["files"]


def test_bundle_redacts_config_values(tmp_path: Path, monkeypatch):
    # Inject a fake config object with a fake secret so we can assert the
    # collector applies redaction at the source.
    import sys
    import types

    fake_config_module = types.ModuleType("core.config")
    fake_config = types.SimpleNamespace(
        api_token="sk-this-should-not-leak",
        public_endpoint="https://example.com",
    )
    # Provide a model_dump method so the collector picks the structured path.
    fake_config.model_dump = lambda: {
        "api_token": fake_config.api_token,
        "public_endpoint": fake_config.public_endpoint,
    }
    fake_config_module.config = fake_config
    monkeypatch.setitem(sys.modules, "core.config", fake_config_module)

    out = tmp_path / "bundle.tar.gz"
    info = build_bundle(output_path=out, workspace=tmp_path / "ws")
    with tarfile.open(info["path"], "r:gz") as tar:
        member = next(m for m in tar.getmembers() if m.name.endswith("config.json"))
        f = tar.extractfile(member)
        assert f is not None
        cfg = json.loads(f.read().decode("utf-8"))

    assert cfg["api_token"] == REDACTED
    assert cfg["public_endpoint"] == "https://example.com"


def test_bundle_records_collector_errors_without_aborting(tmp_path: Path, monkeypatch):
    """A failing collector should produce an _error file and not raise."""
    import core.runtime.diagnostics_bundle as db

    def _boom():
        raise RuntimeError("simulated outage")

    monkeypatch.setattr(db, "collect_health", _boom)
    out = tmp_path / "bundle.tar.gz"
    info = build_bundle(output_path=out, workspace=tmp_path / "ws")
    assert info["ok"] is True
    assert "health" in info["errors"]
    assert "simulated outage" in info["errors"]["health"]
    with tarfile.open(info["path"], "r:gz") as tar:
        names = tar.getnames()
    assert any(n.endswith("health.error.txt") for n in names)


def test_bundle_includes_audit_chain_subdirectory(tmp_path: Path):
    out = tmp_path / "bundle.tar.gz"
    info = build_bundle(output_path=out, workspace=tmp_path / "ws")
    with tarfile.open(info["path"], "r:gz") as tar:
        names = tar.getnames()
    assert any("/audit_chain/" in n for n in names)


def test_sha256_matches_actual_file(tmp_path: Path):
    import hashlib

    out = tmp_path / "bundle.tar.gz"
    info = build_bundle(output_path=out, workspace=tmp_path / "ws")
    digest = hashlib.sha256(Path(info["path"]).read_bytes()).hexdigest()
    assert info["sha256"] == digest


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------
def test_cli_doctor_bundle_flag_invokes_bundle(tmp_path: Path, monkeypatch):
    from core.runtime import operator_cli

    captured = {}

    def fake_build_bundle(*, output_path=None, workspace=None):
        captured["output_path"] = output_path
        return {
            "ok": True,
            "path": str(output_path) if output_path else "",
            "bytes": 0,
            "sha256": "0" * 64,
            "errors": {},
            "file_count": 0,
            "included": [],
        }

    import core.runtime.diagnostics_bundle as db_module

    monkeypatch.setattr(db_module, "build_bundle", fake_build_bundle)

    out = tmp_path / "explicit.tar.gz"
    result = operator_cli.run_command(["doctor", "--bundle", "--bundle-path", str(out)])
    assert result["command"] == "doctor"
    assert result["ok"] is True
    assert captured["output_path"] == out
