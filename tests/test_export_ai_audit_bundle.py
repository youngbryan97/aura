from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from scripts.export_ai_audit_bundle import (
    AuditExportError,
    _run_git,
    export_bundle,
    require_clean_source_snapshot,
    select_files,
    verify_bundle,
)


def test_git_export_probe_declares_no_accelerator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[list[str], dict[str, object]]] = []

    class _Completed:
        returncode = 0
        stdout = "tracked.py\n"
        stderr = ""

    class _Gateway:
        def run(self, command: list[str], **kwargs: object) -> _Completed:
            calls.append((command, kwargs))
            return _Completed()

    monkeypatch.setattr(
        "scripts.export_ai_audit_bundle.get_subprocess_gateway",
        lambda: _Gateway(),
    )

    assert _run_git(tmp_path, "ls-files") == "tracked.py\n"
    assert calls == [
        (
            ["git", "ls-files"],
            {
                "cwd": tmp_path,
                "timeout": 30.0,
                "read_only": True,
                "accelerator_capability": "none",
                "source": "maintenance_tooling:ai_audit_export",
                "check": False,
                "capture_output": True,
            },
        )
    ]


def _write(root: Path, relative_path: str, text: str) -> None:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_select_files_keeps_source_and_excludes_runtime_data(tmp_path: Path) -> None:
    _write(tmp_path, "core/runtime.py", "VALUE = 1\n")
    _write(tmp_path, "tests/test_runtime.py", "def test_value(): pass\n")
    _write(tmp_path, ".github/workflows/ci.yml", "name: ci\n")
    _write(tmp_path, "requirements/runtime.txt", "pytest\n")
    _write(tmp_path, "docs/design.py", "SHOULD_NOT_EXPORT = True\n")
    _write(tmp_path, "artifacts/run.json", "{}\n")
    _write(tmp_path, "training/data/train.jsonl", "{}\n")
    _write(tmp_path, "interface/static/app.min.js", "minified()\n")

    selected, excluded = select_files(
        tmp_path,
        [
            "training/data/train.jsonl",
            "tests/test_runtime.py",
            "requirements/runtime.txt",
            "interface/static/app.min.js",
            "docs/design.py",
            "core/runtime.py",
            "artifacts/run.json",
            ".github/workflows/ci.yml",
        ],
    )

    assert [item.relative_path for item in selected] == [
        "core/runtime.py",
        ".github/workflows/ci.yml",
        "requirements/runtime.txt",
        "tests/test_runtime.py",
    ]
    assert excluded == {
        "generated_frontend": 1,
        "prose_data_or_archive": 2,
        "runtime_or_binary_tree": 1,
    }


def test_export_bundle_is_deterministic_and_self_identifying(tmp_path: Path) -> None:
    _write(tmp_path, "core/a.py", "A = 1\nFILE: source text is not framing\n")
    _write(tmp_path, "tests/test_a.py", "def test_a():\n    assert True\n")
    selected, excluded = select_files(
        tmp_path,
        ["tests/test_a.py", "core/a.py"],
    )
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"

    first_manifest = export_bundle(
        root=tmp_path,
        output_path=first,
        files=selected,
        excluded_counts=excluded,
        commit_sha="a" * 40,
    )
    second_manifest = export_bundle(
        root=tmp_path,
        output_path=second,
        files=selected,
        excluded_counts=excluded,
        commit_sha="a" * 40,
    )

    assert first.read_bytes() == second.read_bytes()
    assert first_manifest.output_sha256 == second_manifest.output_sha256
    assert hashlib.sha256(first.read_bytes()).hexdigest() == first_manifest.output_sha256
    text = first.read_text(encoding="utf-8")
    assert "# Commit SHA: " + ("a" * 40) in text
    assert text.index("FILE: core/a.py") < text.index("FILE: tests/test_a.py")
    verify_bundle(first, first_manifest)


def test_verify_bundle_rejects_tampering(tmp_path: Path) -> None:
    _write(tmp_path, "core/a.py", "A = 1\n")
    selected, excluded = select_files(tmp_path, ["core/a.py"])
    output = tmp_path / "audit.txt"
    manifest = export_bundle(
        root=tmp_path,
        output_path=output,
        files=selected,
        excluded_counts=excluded,
        commit_sha="c" * 40,
    )
    output.write_bytes(output.read_bytes() + b"tampered")

    with pytest.raises(AuditExportError, match="size mismatch"):
        verify_bundle(output, manifest)


def test_export_bundle_fails_closed_when_complete_scope_exceeds_cap(
    tmp_path: Path,
) -> None:
    _write(tmp_path, "core/large.py", "x = 1\n" * 100)
    selected, excluded = select_files(tmp_path, ["core/large.py"])
    output = tmp_path / "audit.txt"

    with pytest.raises(AuditExportError, match="exceed the audit cap"):
        export_bundle(
            root=tmp_path,
            output_path=output,
            files=selected,
            excluded_counts=excluded,
            commit_sha="b" * 40,
            max_output_bytes=64,
        )

    assert not output.exists()


def test_require_clean_source_snapshot_ignores_excluded_artifact_changes(
    tmp_path: Path,
) -> None:
    _write(tmp_path, "core/a.py", "A = 1\n")
    selected, _excluded = select_files(tmp_path, ["core/a.py"])

    require_clean_source_snapshot(selected, {"artifacts/current/result.json"})

    with pytest.raises(AuditExportError, match="commit it before export"):
        require_clean_source_snapshot(selected, {"core/a.py"})
