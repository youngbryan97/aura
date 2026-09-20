"""Identity contracts for a source-bound resident recurrence campaign."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from tools.unified_intrinsic_resident_identity import (
    UnifiedResidentIdentityError,
    build_model_manifest,
    build_source_git_identity,
    build_source_manifest,
    canonical_sha256,
    trainer_model_identity_from_manifest,
    verify_model_manifest,
    verify_source_git_identity,
    verify_source_manifest,
)


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-c", "core.fsmonitor=false", "-C", str(root), *args],
        capture_output=True,
        check=True,
        text=True,
    )
    return result.stdout.strip()


def _source_capsule(tmp_path: Path) -> tuple[Path, str]:
    root = tmp_path / "source"
    for directory in ("core", "tools", "config"):
        (root / directory).mkdir(parents=True, exist_ok=True)
        (root / directory / "owned.py").write_text(
            f"ROLE = {directory!r}\n",
            encoding="ascii",
        )
    for name in ("pyproject.toml", "requirements.txt", "requirements_lock.txt"):
        (root / name).write_text(f"# {name}\n", encoding="ascii")
    _git(root, "init", "-q")
    _git(root, "config", "user.name", "Test")
    _git(root, "config", "user.email", "test@example.com")
    _git(root, "add", ".")
    _git(root, "commit", "-qm", "fixture")
    commit = _git(root, "rev-parse", "HEAD")
    _git(root, "checkout", "--detach", "-q", commit)
    return root, commit


def test_source_capsule_identity_requires_clean_detached_exact_commit(
    tmp_path: Path,
) -> None:
    root, commit = _source_capsule(tmp_path)
    git_identity = build_source_git_identity(root, source_commit=commit)
    manifest = build_source_manifest(root, source_commit=commit)

    verify_source_git_identity(root, git_identity)
    verify_source_manifest(root, manifest)
    assert git_identity["commit"] == commit
    assert manifest["file_count"] == 6
    assert len(manifest["manifest_sha256"]) == 64

    (root / "core" / "owned.py").write_text("ROLE = 'drift'\n", encoding="ascii")
    with pytest.raises(UnifiedResidentIdentityError, match="source_capsule_dirty"):
        verify_source_git_identity(root, git_identity)
    with pytest.raises(UnifiedResidentIdentityError, match="file_drift"):
        verify_source_manifest(root, manifest)


def test_source_capsule_rejects_attached_branch_and_symlink(tmp_path: Path) -> None:
    root, commit = _source_capsule(tmp_path)
    _git(root, "switch", "-q", "-c", "attached")
    with pytest.raises(UnifiedResidentIdentityError, match="not_detached"):
        build_source_git_identity(root, source_commit=commit)

    _git(root, "checkout", "--detach", "-q", commit)
    (root / "core" / "linked.py").symlink_to(root / "core" / "owned.py")
    with pytest.raises(UnifiedResidentIdentityError, match="symlink_rejected"):
        build_source_git_identity(root, source_commit=commit)


def test_model_manifest_binds_behavior_and_weight_bytes(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen2",
                "num_hidden_layers": 64,
                "hidden_size": 5120,
                "vocab_size": 152064,
                "quantization_config": {"bits": 4, "group_size": 64},
            }
        )
        + "\n",
        encoding="ascii",
    )
    (model / "tokenizer.json").write_text('{"version":1}\n', encoding="ascii")
    (model / "tokenizer_config.json").write_text('{"eos":2}\n', encoding="ascii")
    (model / "model.safetensors").write_bytes(b"weights")
    (model / "README.md").write_text("human documentation\n", encoding="ascii")

    manifest = build_model_manifest(model)
    verify_model_manifest(manifest)
    trainer_identity = trainer_model_identity_from_manifest(manifest)
    assert manifest["file_count"] == 5
    assert len(manifest["manifest_sha256"]) == 64
    assert trainer_identity["canonical_path"] == str(model)
    assert trainer_identity["weights"][0]["name"] == "model.safetensors"
    assert all(
        row["name"] != "README.md" for row in trainer_identity["behavior_files"]
    )

    (model / "tokenizer.json").write_text('{"version":2}\n', encoding="ascii")
    with pytest.raises(UnifiedResidentIdentityError, match="model_manifest_drift"):
        verify_model_manifest(manifest)


def test_model_manifest_requires_complete_checkpoint(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text("{}\n", encoding="ascii")
    with pytest.raises(UnifiedResidentIdentityError, match="incomplete"):
        build_model_manifest(tmp_path)


def test_canonical_sha_is_order_independent() -> None:
    assert canonical_sha256({"a": 1, "b": 2}) == canonical_sha256({"b": 2, "a": 1})
