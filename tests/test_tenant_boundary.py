"""Tests for the single-tenant install boundary."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.runtime.tenant_boundary import (
    DEFAULT_TENANT_ID,
    TENANT_FILE,
    TenantBoundary,
    TenantMismatchError,
    TenantStamp,
    configured_tenant_id,
)


# ---------------------------------------------------------------------------
# env-driven tenant id
# ---------------------------------------------------------------------------
def test_default_tenant_id_when_env_unset(monkeypatch):
    monkeypatch.delenv("AURA_TENANT_ID", raising=False)
    assert configured_tenant_id() == DEFAULT_TENANT_ID


def test_env_overrides_default(monkeypatch):
    monkeypatch.setenv("AURA_TENANT_ID", "youngbryan")
    assert configured_tenant_id() == "youngbryan"


def test_blank_env_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("AURA_TENANT_ID", "   ")
    assert configured_tenant_id() == DEFAULT_TENANT_ID


# ---------------------------------------------------------------------------
# stamp lifecycle
# ---------------------------------------------------------------------------
def test_stamp_writes_tenant_json(tmp_path: Path):
    boundary = TenantBoundary(tmp_path, tenant_id="alpha")
    stamp = boundary.stamp()
    assert (tmp_path / TENANT_FILE).exists()
    assert stamp.tenant_id == "alpha"
    assert stamp.install_id.startswith("install-")
    assert stamp.created_at > 0


def test_stamp_is_idempotent_for_same_tenant(tmp_path: Path):
    boundary = TenantBoundary(tmp_path, tenant_id="alpha")
    a = boundary.stamp()
    b = boundary.stamp()
    assert a.install_id == b.install_id  # not re-stamped


def test_stamp_force_overwrites(tmp_path: Path):
    a = TenantBoundary(tmp_path, tenant_id="alpha").stamp()
    b = TenantBoundary(tmp_path, tenant_id="alpha").stamp(force=True)
    assert a.install_id != b.install_id


# ---------------------------------------------------------------------------
# assert_owned
# ---------------------------------------------------------------------------
def test_assert_owned_first_touch_writes_stamp(tmp_path: Path):
    boundary = TenantBoundary(tmp_path, tenant_id="alpha")
    stamp = boundary.assert_owned()
    assert stamp.tenant_id == "alpha"
    assert (tmp_path / TENANT_FILE).exists()


def test_assert_owned_passes_for_matching_tenant(tmp_path: Path):
    TenantBoundary(tmp_path, tenant_id="alpha").stamp()
    boundary = TenantBoundary(tmp_path, tenant_id="alpha")
    stamp = boundary.assert_owned()
    assert stamp.tenant_id == "alpha"


def test_assert_owned_refuses_mismatched_tenant(tmp_path: Path):
    TenantBoundary(tmp_path, tenant_id="alpha").stamp()
    boundary = TenantBoundary(tmp_path, tenant_id="beta")
    with pytest.raises(TenantMismatchError) as exc:
        boundary.assert_owned()
    assert exc.value.expected == "beta"
    assert exc.value.found == "alpha"


def test_two_data_dirs_can_have_different_tenants(tmp_path: Path):
    a_dir = tmp_path / "a"
    b_dir = tmp_path / "b"
    TenantBoundary(a_dir, tenant_id="alpha").stamp()
    TenantBoundary(b_dir, tenant_id="beta").stamp()
    assert TenantBoundary(a_dir, tenant_id="alpha").assert_owned().tenant_id == "alpha"
    assert TenantBoundary(b_dir, tenant_id="beta").assert_owned().tenant_id == "beta"


# ---------------------------------------------------------------------------
# malformed / missing stamps
# ---------------------------------------------------------------------------
def test_corrupted_stamp_is_treated_as_missing(tmp_path: Path):
    (tmp_path / TENANT_FILE).write_text("{ this is not json", encoding="utf-8")
    boundary = TenantBoundary(tmp_path, tenant_id="alpha")
    # First touch with a corrupted stamp should overwrite cleanly.
    stamp = boundary.assert_owned()
    assert stamp.tenant_id == "alpha"


def test_current_stamp_returns_none_when_no_stamp(tmp_path: Path):
    boundary = TenantBoundary(tmp_path, tenant_id="alpha")
    assert boundary.current_stamp() is None


def test_current_stamp_after_write(tmp_path: Path):
    TenantBoundary(tmp_path, tenant_id="alpha").stamp()
    boundary = TenantBoundary(tmp_path, tenant_id="alpha")
    s = boundary.current_stamp()
    assert isinstance(s, TenantStamp)
    assert s.tenant_id == "alpha"


# ---------------------------------------------------------------------------
# integration: env-driven boundary refuses mismatched stamp
# ---------------------------------------------------------------------------
def test_env_tenant_refuses_foreign_data_dir(tmp_path: Path, monkeypatch):
    foreign = tmp_path / "foreign"
    TenantBoundary(foreign, tenant_id="org-a").stamp()
    # Now run as org-b against the same dir.
    monkeypatch.setenv("AURA_TENANT_ID", "org-b")
    boundary = TenantBoundary(foreign)  # picks up env tenant
    with pytest.raises(TenantMismatchError):
        boundary.assert_owned()
