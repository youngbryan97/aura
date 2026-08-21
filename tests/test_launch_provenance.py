from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.runtime import launch_provenance


def test_runtime_shell_request_path_is_canonical_and_rejects_traversal():
    assert (
        launch_provenance.runtime_shell_request_path("interface/static/aura.js")
        == "/static/aura.js"
    )
    for invalid in (
        "static/aura.js",
        "/interface/static/aura.js",
        "interface/../aura.js",
        "interface/static\\aura.js",
    ):
        with pytest.raises(ValueError):
            launch_provenance.runtime_shell_request_path(invalid)


def test_runtime_shell_digest_is_path_bound_and_content_sensitive(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    first = tmp_path / "interface" / "static" / "aura.js"
    second = tmp_path / "interface" / "static" / "aura.css"
    first.parent.mkdir(parents=True)
    first.write_text("one", encoding="utf-8")
    second.write_text("two", encoding="utf-8")
    monkeypatch.setattr(
        launch_provenance,
        "RUNTIME_SHELL_ASSETS",
        ("interface/static/aura.js", "interface/static/aura.css"),
    )

    before = launch_provenance.runtime_shell_assets_sha256(tmp_path)
    second.write_text("changed", encoding="utf-8")
    after = launch_provenance.runtime_shell_assets_sha256(tmp_path)

    assert before != after


def test_runtime_shell_digest_rejects_symlink_assets(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    target = tmp_path / "target.js"
    target.write_text("payload", encoding="utf-8")
    link = tmp_path / "interface" / "static" / "aura.js"
    link.parent.mkdir(parents=True)
    link.symlink_to(target)
    monkeypatch.setattr(
        launch_provenance,
        "RUNTIME_SHELL_ASSETS",
        ("interface/static/aura.js",),
    )

    with pytest.raises(RuntimeError, match="symlink"):
        launch_provenance.runtime_shell_assets_sha256(tmp_path)


def test_build_manifest_signs_stable_runtime_shell(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    launcher = tmp_path / "scripts" / "AuraLauncher.swift"
    launcher.parent.mkdir()
    launcher.write_text("launcher", encoding="utf-8")
    identity = {
        "source_root": str(tmp_path.resolve()),
        "commit_sha": "a" * 40,
        "branch": "main",
    }
    workspace = {
        "workspace_state_sha256": "b" * 64,
        "source_dirty": False,
        "source_change_count": 0,
        "source_changed_paths": [],
        "source_changed_paths_truncated": False,
    }
    monkeypatch.setattr(launch_provenance, "_git_identity", lambda _root: dict(identity))
    monkeypatch.setattr(
        launch_provenance,
        "_workspace_state_uncached",
        lambda _root: dict(workspace),
    )
    monkeypatch.setattr(
        launch_provenance,
        "runtime_shell_assets_sha256",
        lambda _root: "c" * 64,
    )

    manifest = launch_provenance.build_launch_manifest(
        tmp_path,
        version="Aura test",
        launcher_source=launcher,
    )

    assert manifest["shell_assets_sha256"] == "c" * 64
    assert manifest["workspace_state_sha256"] == "b" * 64


def test_build_manifest_rejects_workspace_change_during_capture(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    launcher = tmp_path / "scripts" / "AuraLauncher.swift"
    launcher.parent.mkdir()
    launcher.write_text("launcher", encoding="utf-8")
    identity = {
        "source_root": str(tmp_path.resolve()),
        "commit_sha": "a" * 40,
        "branch": "main",
    }
    workspace_digests = iter(("b" * 64, "d" * 64))
    monkeypatch.setattr(launch_provenance, "_git_identity", lambda _root: dict(identity))
    monkeypatch.setattr(
        launch_provenance,
        "_workspace_state_uncached",
        lambda _root: {"workspace_state_sha256": next(workspace_digests)},
    )
    monkeypatch.setattr(
        launch_provenance,
        "runtime_shell_assets_sha256",
        lambda _root: "c" * 64,
    )

    with pytest.raises(RuntimeError, match="changed while"):
        launch_provenance.build_launch_manifest(
            tmp_path,
            version="Aura test",
            launcher_source=launcher,
        )


def _app_contract(tmp_path: Path) -> tuple[Path, Path, dict[str, str], dict[str, object]]:
    app = tmp_path / "Aura.app"
    executable = app / "Contents" / "MacOS" / "aura-launcher"
    manifest_path = app / "Contents" / "Resources" / "aura-launch-provenance.json"
    executable.parent.mkdir(parents=True)
    manifest_path.parent.mkdir(parents=True)
    executable.write_text("launcher", encoding="utf-8")
    manifest: dict[str, object] = {
        "schema": launch_provenance.LAUNCH_PROVENANCE_SCHEMA,
        "source_root": str(tmp_path.resolve()),
        "commit_sha": "a" * 40,
        "branch": "main",
        "workspace_state_sha256": "b" * 64,
        "bundle_identifier": launch_provenance.EXPECTED_BUNDLE_ID,
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    env = {
        "AURA_LAUNCHED_FROM_APP": "1",
        "AURA_LAUNCH_MANIFEST_PATH": str(manifest_path),
        "AURA_LAUNCH_APP_EXECUTABLE": str(executable),
        "AURA_LAUNCH_EXPECTED_ROOT": str(tmp_path.resolve()),
        "AURA_LAUNCH_EXPECTED_COMMIT": "a" * 40,
        "AURA_LAUNCH_EXPECTED_BRANCH": "main",
        "AURA_LAUNCH_EXPECTED_WORKSPACE_SHA256": "b" * 64,
        "AURA_LAUNCH_BUNDLE_ID": launch_provenance.EXPECTED_BUNDLE_ID,
    }
    return executable, manifest_path, env, manifest


def _stub_source(monkeypatch, root: Path) -> None:
    monkeypatch.setattr(
        launch_provenance,
        "_git_identity",
        lambda _root: {
            "source_root": str(root.resolve()),
            "commit_sha": "a" * 40,
            "branch": "main",
        },
    )
    monkeypatch.setattr(
        launch_provenance,
        "_workspace_state",
        lambda _root, *, commit_sha: {
            "workspace_state_sha256": "b" * 64,
            "source_dirty": False,
            "source_change_count": 0,
            "source_changed_paths": [],
            "source_changed_paths_truncated": False,
        },
    )


def test_signed_app_source_preflight_accepts_exact_manifest(monkeypatch, tmp_path):
    _executable, _manifest_path, env, _manifest = _app_contract(tmp_path)
    _stub_source(monkeypatch, tmp_path)

    result = launch_provenance.validate_launch_source(tmp_path, env=env)

    assert result["required"] is True
    assert result["source_verified"] is True
    assert result["issues"] == []
    assert result["actual"]["commit_sha"] == "a" * 40


def test_commit_drift_is_reported_not_failed(monkeypatch, tmp_path):
    """A moved HEAD is Aura's steady state, not a fault.

    This asserted source_verified is False, which made the launch path refuse
    to start after any commit and a human rebuild the only way back. Identity
    (root, bundle) still verifies; how far the workspace has moved is measured
    and published instead of being a verdict.
    """
    _executable, _manifest_path, env, _manifest = _app_contract(tmp_path)
    _stub_source(monkeypatch, tmp_path)
    env["AURA_LAUNCH_EXPECTED_COMMIT"] = "c" * 40

    result = launch_provenance.validate_launch_source(tmp_path, env=env)

    assert result["source_verified"] is True
    assert result["issues"] == []
    assert result["source_current"] is False
    assert "commit_sha" in result["source_drift"]
    # The measured value is the running one, not the manifest's.
    assert result["actual"]["commit_sha"] == "a" * 40


def test_dirty_workspace_drift_is_reported_not_failed(monkeypatch, tmp_path):
    _executable, _manifest_path, env, _manifest = _app_contract(tmp_path)
    _stub_source(monkeypatch, tmp_path)
    env["AURA_LAUNCH_EXPECTED_WORKSPACE_SHA256"] = "d" * 64

    result = launch_provenance.validate_launch_source(tmp_path, env=env)

    assert result["source_verified"] is True
    assert result["issues"] == []
    assert result["source_current"] is False
    assert "workspace_state_sha256" in result["source_drift"]


def test_a_bundle_from_another_checkout_is_still_rejected(monkeypatch, tmp_path):
    """The safety property that actually matters must not have been loosened:
    an app belonging to a DIFFERENT workspace may never run cleanup here."""
    _executable, _manifest_path, env, _manifest = _app_contract(tmp_path)
    _stub_source(monkeypatch, tmp_path)
    env["AURA_LAUNCH_EXPECTED_ROOT"] = str(tmp_path / "somewhere-else")

    result = launch_provenance.validate_launch_source(tmp_path, env=env)

    assert result["source_verified"] is False
    assert "source_root_mismatch" in result["issues"]


def test_a_foreign_bundle_identifier_is_still_rejected(monkeypatch, tmp_path):
    _executable, _manifest_path, env, _manifest = _app_contract(tmp_path)
    _stub_source(monkeypatch, tmp_path)
    env["AURA_LAUNCH_BUNDLE_ID"] = "com.someone.else"

    result = launch_provenance.validate_launch_source(tmp_path, env=env)

    assert result["source_verified"] is False
    assert "bundle_identifier_mismatch" in result["issues"]


def test_an_unmoved_workspace_reports_current(monkeypatch, tmp_path):
    _executable, _manifest_path, env, _manifest = _app_contract(tmp_path)
    _stub_source(monkeypatch, tmp_path)

    result = launch_provenance.validate_launch_source(tmp_path, env=env)

    assert result["source_verified"] is True
    assert result["source_current"] is True
    assert result["source_drift"] == []


def test_signed_app_source_preflight_rejects_manifest_outside_bundle(monkeypatch, tmp_path):
    _executable, _manifest_path, env, manifest = _app_contract(tmp_path)
    _stub_source(monkeypatch, tmp_path)
    detached_manifest = tmp_path / "detached.json"
    detached_manifest.write_text(json.dumps(manifest), encoding="utf-8")
    env["AURA_LAUNCH_MANIFEST_PATH"] = str(detached_manifest)

    result = launch_provenance.validate_launch_source(tmp_path, env=env)

    assert result["source_verified"] is False
    assert "manifest_outside_app_bundle" in result["issues"]


def test_runtime_provenance_requires_resident_stably_signed_strict_bundle(
    monkeypatch,
    tmp_path,
):
    executable, _manifest_path, env, _manifest = _app_contract(tmp_path)
    _stub_source(monkeypatch, tmp_path)
    from core.security import native_desktop_bridge

    monkeypatch.setattr(
        native_desktop_bridge,
        "native_desktop_bridge_identity",
        lambda *, executable: {
            "bridge_executable": str(executable),
            "resident_running": True,
            "code_signature": {
                "available": True,
                "stable_tcc_identity": True,
                "identifier": launch_provenance.EXPECTED_BUNDLE_ID,
            },
        },
    )
    monkeypatch.setattr(
        launch_provenance,
        "_strict_bundle_verification",
        lambda _executable: {"ok": True, "bundle_path": str(executable.parents[2])},
    )

    result = launch_provenance.collect_runtime_launch_provenance(tmp_path, env=env)

    assert result["app_executable"] == str(executable)
    assert result["source_verified"] is True
    assert result["verified"] is True
    assert result["issues"] == []


def test_runtime_provenance_rejects_orphaned_app(monkeypatch, tmp_path):
    _executable, _manifest_path, env, _manifest = _app_contract(tmp_path)
    _stub_source(monkeypatch, tmp_path)
    from core.security import native_desktop_bridge

    monkeypatch.setattr(
        native_desktop_bridge,
        "native_desktop_bridge_identity",
        lambda *, executable: {
            "bridge_executable": str(executable),
            "resident_running": False,
            "code_signature": {
                "available": True,
                "stable_tcc_identity": True,
                "identifier": launch_provenance.EXPECTED_BUNDLE_ID,
            },
        },
    )
    monkeypatch.setattr(
        launch_provenance,
        "_strict_bundle_verification",
        lambda _executable: {"ok": True},
    )

    result = launch_provenance.collect_runtime_launch_provenance(tmp_path, env=env)

    assert result["verified"] is False
    assert "resident_app_not_running" in result["issues"]


def test_boot_health_fails_closed_on_required_launch_provenance(monkeypatch):
    from interface.routes import system as system_routes

    monkeypatch.setattr(
        launch_provenance,
        "collect_runtime_launch_provenance",
        lambda _root: {
            "required": True,
            "verified": False,
            "issues": ["commit_sha_mismatch"],
        },
    )

    payload, status = system_routes._attach_launch_provenance_contract(
        {
            "ready": True,
            "launcher_ready": True,
            "system_ready": True,
            "checks": {},
            "blockers": [],
        },
        200,
        runtime_revision=system_routes._runtime_revision_unavailable(
            "", required=False
        ),
    )

    assert status == 503
    assert payload["ready"] is False
    assert payload["checks"]["launch_provenance"] is False
    assert payload["blockers"] == ["launch_provenance"]


def test_boot_health_keeps_direct_runtime_semantics(monkeypatch):
    from interface.routes import system as system_routes

    monkeypatch.setattr(
        launch_provenance,
        "collect_runtime_launch_provenance",
        lambda _root: {"required": False, "verified": False, "launch_mode": "direct"},
    )

    payload, status = system_routes._attach_launch_provenance_contract(
        {"ready": True, "checks": {}, "blockers": []},
        200,
        runtime_revision=system_routes._runtime_revision_unavailable(
            "", required=False
        ),
    )

    assert status == 200
    assert payload["ready"] is True
    assert payload["checks"]["launch_provenance"] is True


def test_boot_health_fallback_never_runs_blocking_provenance_probe(monkeypatch):
    from interface.routes import system as system_routes

    monkeypatch.setenv("AURA_LAUNCHED_FROM_APP", "1")
    monkeypatch.setattr(
        launch_provenance,
        "collect_runtime_launch_provenance",
        lambda _root: (_ for _ in ()).throw(AssertionError("blocking probe called")),
    )

    fallback = system_routes._fallback_launch_provenance(
        {"required": True, "verified": True, "issues": []}
    )
    payload, status = system_routes._attach_launch_provenance_contract(
        {"ready": True, "checks": {}, "blockers": []},
        200,
        provenance=fallback,
    )

    assert status == 503
    assert payload["ready"] is False
    assert payload["launch_provenance"]["verified"] is False
    assert "launch_provenance_live_refresh_unavailable" in payload["launch_provenance"]["issues"]
