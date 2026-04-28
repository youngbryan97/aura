"""Tests for the plugin SHA-256 allowlist."""
from __future__ import annotations

from pathlib import Path

import pytest

from core.security.plugin_allowlist import (
    AllowlistEntry,
    PluginAllowlist,
    PluginPolicyError,
    compute_sha256,
)


def _make_plugin(tmp_path: Path, name: str, body: str) -> Path:
    p = tmp_path / "plugins"
    p.mkdir(exist_ok=True)
    f = p / f"{name}.py"
    f.write_text(body, encoding="utf-8")
    return f


# ---------------------------------------------------------------------------
# hash function
# ---------------------------------------------------------------------------
def test_compute_sha256_is_stable(tmp_path):
    f = _make_plugin(tmp_path, "p", "print('hi')\n")
    assert compute_sha256(f) == compute_sha256(f)
    assert compute_sha256(f).startswith("sha256:")


def test_compute_sha256_changes_on_edit(tmp_path):
    f = _make_plugin(tmp_path, "p", "x = 1\n")
    h1 = compute_sha256(f)
    f.write_text("x = 2\n", encoding="utf-8")
    h2 = compute_sha256(f)
    assert h1 != h2


# ---------------------------------------------------------------------------
# unlisted refused / listed loads
# ---------------------------------------------------------------------------
def test_unlisted_plugin_is_refused(tmp_path):
    al = PluginAllowlist(tmp_path / "allow.json")
    plugin = _make_plugin(tmp_path, "p", "x = 1\n")
    with pytest.raises(PluginPolicyError) as exc:
        al.is_allowed(plugin)
    assert exc.value.reason == "hash_not_in_allowlist"


def test_listed_plugin_is_accepted(tmp_path):
    al = PluginAllowlist(tmp_path / "allow.json", plugin_root=tmp_path / "plugins")
    plugin = _make_plugin(tmp_path, "p", "x = 1\n")
    al.record(plugin, approved_by="alice", reason="initial approval")
    entry = al.is_allowed(plugin)
    assert isinstance(entry, AllowlistEntry)
    assert entry.approved_by == "alice"
    assert entry.is_active()


# ---------------------------------------------------------------------------
# hash drift caught
# ---------------------------------------------------------------------------
def test_hash_drift_after_edit_refuses(tmp_path):
    al = PluginAllowlist(tmp_path / "allow.json")
    plugin = _make_plugin(tmp_path, "p", "x = 1\n")
    al.record(plugin, approved_by="alice")
    plugin.write_text("x = 999  # malicious\n", encoding="utf-8")
    with pytest.raises(PluginPolicyError) as exc:
        al.is_allowed(plugin)
    # Drift surfaces as a fresh hash that's not in the allowlist.
    assert exc.value.reason == "hash_not_in_allowlist"


# ---------------------------------------------------------------------------
# revoke
# ---------------------------------------------------------------------------
def test_revoke_inactivates_an_approval(tmp_path):
    al = PluginAllowlist(tmp_path / "allow.json")
    plugin = _make_plugin(tmp_path, "p", "x = 1\n")
    al.record(plugin, approved_by="alice")
    assert al.revoke(plugin) is True
    with pytest.raises(PluginPolicyError) as exc:
        al.is_allowed(plugin)
    assert exc.value.reason == "hash_revoked"


def test_revoking_unknown_plugin_returns_false(tmp_path):
    al = PluginAllowlist(tmp_path / "allow.json")
    plugin = _make_plugin(tmp_path, "p", "x = 1\n")
    assert al.revoke(plugin) is False


# ---------------------------------------------------------------------------
# persistence
# ---------------------------------------------------------------------------
def test_allowlist_persists_across_reopen(tmp_path):
    plugin = _make_plugin(tmp_path, "p", "x = 1\n")

    al_a = PluginAllowlist(tmp_path / "allow.json")
    al_a.record(plugin, approved_by="alice", reason="initial")

    al_b = PluginAllowlist(tmp_path / "allow.json")
    entry = al_b.is_allowed(plugin)
    assert entry.approved_by == "alice"
    assert entry.reason == "initial"


def test_malformed_allowlist_starts_empty(tmp_path):
    bad = tmp_path / "allow.json"
    bad.write_text("{ this is not json", encoding="utf-8")
    al = PluginAllowlist(bad)
    plugin = _make_plugin(tmp_path, "p", "x = 1\n")
    with pytest.raises(PluginPolicyError):
        al.is_allowed(plugin)


# ---------------------------------------------------------------------------
# listing + stats
# ---------------------------------------------------------------------------
def test_list_entries_filters_revoked_by_default(tmp_path):
    al = PluginAllowlist(tmp_path / "allow.json")
    p1 = _make_plugin(tmp_path, "a", "x = 1\n")
    p2 = _make_plugin(tmp_path, "b", "x = 2\n")
    al.record(p1, approved_by="alice")
    al.record(p2, approved_by="bob")
    al.revoke(p1)

    active = al.list_entries()
    assert len(active) == 1
    assert active[0].approved_by == "bob"

    everything = al.list_entries(include_revoked=True)
    assert len(everything) == 2


def test_stats_reports_active_and_revoked(tmp_path):
    al = PluginAllowlist(tmp_path / "allow.json")
    p1 = _make_plugin(tmp_path, "a", "x = 1\n")
    p2 = _make_plugin(tmp_path, "b", "x = 2\n")
    al.record(p1, approved_by="alice")
    al.record(p2, approved_by="bob")
    al.revoke(p1)
    s = al.stats()
    assert s == {"total": 2, "active": 1, "revoked": 1}


def test_missing_plugin_path_raises_descriptively(tmp_path):
    al = PluginAllowlist(tmp_path / "allow.json")
    with pytest.raises(PluginPolicyError) as exc:
        al.is_allowed(tmp_path / "does_not_exist.py")
    assert exc.value.reason == "file_not_found"
