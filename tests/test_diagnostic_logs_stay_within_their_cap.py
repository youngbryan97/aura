"""A log can grow or rotate after its directory entry was inspected."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from core.runtime import diagnostics_bundle as bundle


@pytest.fixture
def logs(tmp_path, monkeypatch):
    source = tmp_path / "logs"
    source.mkdir()
    monkeypatch.setenv("AURA_LOG_DIR", str(source))
    return source, tmp_path / "new" / "bundle" / "logs"


def test_logs_share_one_byte_cap_and_newest_logs_go_first(logs):
    source, target = logs
    (source / "old.log").write_bytes(b"0123456789")
    (source / "new.log").write_bytes(b"new")
    os.utime(source / "old.log", (1, 1))
    os.utime(source / "new.log", (2, 2))

    info = bundle.collect_logs(target, max_total_bytes=8)

    assert info["bytes"] == 8
    assert (target / "new.log").read_bytes() == b"new"
    assert (target / "old.log").read_bytes() == b"56789"
    assert [entry["size"] for entry in info["copied"]] == [3, 5]


@pytest.mark.parametrize("replacement", [b"longer than the budget", b"x"])
def test_the_open_file_decides_the_tail_and_actual_byte_count(logs, monkeypatch, replacement):
    source, target = logs
    log = source / "changing.log"
    log.write_bytes(b"start")
    original_open = Path.open
    changed = []

    def change_before_read(path, mode="r", *args, **kwargs):
        if path == log and mode == "rb" and not changed:
            changed.append(True)
            with original_open(path, "wb") as handle:
                handle.write(replacement)
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", change_before_read)
    info = bundle.collect_logs(target, max_total_bytes=8)

    actual = (target / log.name).read_bytes()
    assert actual == replacement[-8:]
    assert info["bytes"] == len(actual) <= 8
    assert info["copied"][0]["size"] == len(actual)
    assert info["copied"][0]["truncated_tail"] is (len(replacement) > 8)


def test_a_zero_cap_reads_nothing_and_creates_no_log_directory(logs):
    source, target = logs
    (source / "not-read.log").write_bytes(b"still here")
    info = bundle.collect_logs(target, max_total_bytes=0)
    assert info["bytes"] == 0 and info["copied"] == []
    assert not target.exists()


def test_the_byte_bound_has_a_registered_invariant_and_validation():
    from core.organism.model_validation import ValidationSuite, _install_repair_executor_claims

    assert bundle._diagnostic_log_read_invariant() == ()
    suite = ValidationSuite()
    _install_repair_executor_claims(suite)
    check = next(test for test in suite.tests() if test.name == "diagnostic_log_read_bound")
    assert check.predict(None) is True
    assert any(claim.test == check.name for claim in suite.claims())


def test_the_configured_log_root_wins_over_other_existing_logs(logs, tmp_path, monkeypatch):
    old, target = logs
    (old / "other-instance.log").write_bytes(b"not this directory")
    configured = tmp_path / "configured"
    configured.mkdir()
    (configured / "chosen.log").write_bytes(b"the configured directory")
    monkeypatch.setenv("AURA_LOG_DIR", str(configured))

    info = bundle.collect_logs(target)

    assert info["source"] == str(configured)
    assert (target / "chosen.log").read_bytes() == b"the configured directory"
    assert not (target / "other-instance.log").exists()


def test_a_missing_configured_root_cannot_fall_back_to_another_instance(logs, tmp_path, monkeypatch):
    old, target = logs
    (old / "other-instance.log").write_bytes(b"not this directory")
    monkeypatch.setenv("AURA_LOG_DIR", str(tmp_path / "missing"))

    info = bundle.collect_logs(target)

    assert info["available"] is False
    assert not target.exists()


def test_the_default_root_comes_from_the_logging_owner(tmp_path, monkeypatch):
    from core.observability import logging_config

    root = tmp_path / "logger-state"
    source = root / "logs"
    source.mkdir(parents=True)
    (source / "default.log").write_bytes(b"logger default")
    monkeypatch.delenv("AURA_LOG_DIR", raising=False)
    monkeypatch.setattr(logging_config, "state_root", lambda: root)
    target = tmp_path / "copied"

    info = bundle.collect_logs(target)

    assert info["source"] == str(source)
    assert (target / "default.log").read_bytes() == b"logger default"
