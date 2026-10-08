"""Switching a reader package on and back leaves the operational activation as it was."""

from __future__ import annotations

import pytest

from tools import build_semantic_reader_package as packaging


@pytest.fixture
def active(tmp_path, monkeypatch):
    path = tmp_path / "fused-model" / "compositional-semantic-active.json"
    monkeypatch.setattr(packaging, "ACTIVE", path)
    return path


def _package(tmp_path, body: bytes):
    package = tmp_path / "package"
    package.mkdir()
    (package / "activation.json").write_bytes(body)
    return package


def test_rollback_removes_an_activation_that_did_not_exist_before(tmp_path, active) -> None:
    package = _package(tmp_path, b'{"package_id":"new"}\n')
    packaging.activate(package)
    assert active.read_bytes() == b'{"package_id":"new"}\n'
    packaging.rollback(package)
    assert not active.exists()
    assert not (package / "previous_active.json").exists()


def test_rollback_restores_the_previous_activation_byte_for_byte(tmp_path, active) -> None:
    active.parent.mkdir(parents=True)
    active.write_bytes(b'{"package_id":"old"}\n')
    package = _package(tmp_path, b'{"package_id":"new"}\n')
    packaging.activate(package)
    assert active.read_bytes() == b'{"package_id":"new"}\n'
    packaging.rollback(package)
    assert active.read_bytes() == b'{"package_id":"old"}\n'


def test_a_package_cannot_be_activated_twice_without_rolling_back(tmp_path, active) -> None:
    package = _package(tmp_path, b"{}\n")
    packaging.activate(package)
    with pytest.raises(SystemExit):
        packaging.activate(package)


def test_a_changed_copy_of_the_previous_activation_refuses_rollback(tmp_path, active) -> None:
    active.parent.mkdir(parents=True)
    active.write_bytes(b'{"package_id":"old"}\n')
    package = _package(tmp_path, b"{}\n")
    packaging.activate(package)
    (package / "previous_active.bytes").write_bytes(b"tampered")
    with pytest.raises(SystemExit):
        packaging.rollback(package)
