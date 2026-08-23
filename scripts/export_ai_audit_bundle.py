#!/usr/bin/env python3
"""Export Aura's tracked implementation surface for external AI audit.

The export intentionally excludes runtime state, evidence artifacts, model
weights, datasets, prose-only documentation, generated bundles, and binaries.
It includes tracked implementation, tests, build/release infrastructure,
workflows, and source configuration in one deterministic text file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Iterable, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.runtime.subprocess_gateway import get_subprocess_gateway


DEFAULT_MAX_OUTPUT_BYTES = 95_000_000
DEFAULT_MAX_FILE_BYTES = 2_000_000
DEFAULT_OUTPUT_NAME = "aura_codebase_ai_audit.txt"

SOURCE_SUFFIXES = {
    ".bash",
    ".c",
    ".cc",
    ".cfg",
    ".conf",
    ".cpp",
    ".css",
    ".entitlements",
    ".h",
    ".hpp",
    ".html",
    ".ini",
    ".js",
    ".json",
    ".json5",
    ".jsx",
    ".proto",
    ".py",
    ".pyi",
    ".rs",
    ".service",
    ".sh",
    ".sql",
    ".toml",
    ".ts",
    ".tsx",
    ".xml",
    ".yaml",
    ".yml",
    ".zsh",
}

SOURCE_NAMES = {
    ".gitignore",
    ".python-version",
    "Dockerfile",
    "Makefile",
}

EXCLUDED_PARTS = {
    ".aura",
    ".aura_runtime",
    ".git",
    ".mypy_cache",
    ".next",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "artifacts",
    "backups",
    "build",
    "data",
    "dist",
    "logs",
    "memory_store",
    "models",
    "models_gguf",
    "node_modules",
    "recordings",
    "screenshots",
    "storage",
    "test_brain",
    "test_data",
    "test_vdb",
    "venv",
}

EXCLUDED_PREFIXES = (
    "archive/",
    "dev_archive/",
    "docs/",
    "scoping/",
    "scratch/",
    "training/adapters/",
    "training/data/",
    "training/fused-model/",
    "training/raw_data/",
)

EXCLUDED_NAMES = {
    "aura_codebase_ai_audit.txt",
    "aura_full_codebase_audit.txt",
    "package-lock.json",
    "yarn.lock",
}

EXCLUDED_NAME_SUFFIXES = (
    ".min.css",
    ".min.js",
)


@dataclass(frozen=True)
class SelectedFile:
    relative_path: str
    size_bytes: int
    priority: int


@dataclass(frozen=True)
class AuditExportManifest:
    source_root: str
    output_path: str
    generated_at: str
    commit_sha: str
    max_output_bytes: int
    max_file_bytes: int
    files_exported: int
    source_bytes: int
    output_bytes: int
    output_sha256: str
    excluded_counts: dict[str, int]
    files: list[dict[str, object]]


class AuditExportError(RuntimeError):
    """Raised when a complete, trustworthy audit export cannot be produced."""


def _run_git(root: Path, *args: str) -> str:
    completed = get_subprocess_gateway().run(
        ["git", *args],
        cwd=root,
        timeout=30.0,
        read_only=True,
        accelerator_capability="none",
        source="maintenance_tooling:ai_audit_export",
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip()
        raise AuditExportError(f"git {' '.join(args)} failed: {detail}")
    return completed.stdout


def tracked_paths(root: Path) -> list[str]:
    output = _run_git(root, "ls-files", "-z")
    return sorted(
        path
        for path in output.split("\0")
        if path
    )


def current_commit(root: Path) -> str:
    return _run_git(root, "rev-parse", "HEAD").strip()


def dirty_tracked_paths(root: Path) -> set[str]:
    output = _run_git(root, "diff", "--name-only", "-z", "HEAD", "--")
    return {path for path in output.split("\0") if path}


def _is_requirements_file(path: PurePosixPath) -> bool:
    if path.suffix != ".txt":
        return False
    return path.name.startswith("requirements") or path.parts[0] == "requirements"


def _source_priority(path: PurePosixPath) -> int:
    first = path.parts[0]
    if len(path.parts) == 1:
        return 0
    if first == "core":
        return 1
    if first in {"interface", "llm", "senses", "security", "executors"}:
        return 2
    if first in {
        ".github",
        "config",
        "docker",
        "infrastructure",
        "native",
        "proof_kernel",
        "requirements",
        "rust_extensions",
        "scripts",
        "systemd",
        "tools",
    }:
        return 3
    if first in {"aura_bench", "benchmarks", "evals", "tests"}:
        return 4
    return 5


def _exclusion_reason(path_text: str, root: Path, max_file_bytes: int) -> str | None:
    path = PurePosixPath(path_text)
    if not path.parts:
        return "invalid_path"
    if path.name in EXCLUDED_NAMES or path.name.startswith("aura_source_part_"):
        return "generated_export"
    if path.name.endswith(EXCLUDED_NAME_SUFFIXES):
        return "generated_frontend"
    if path_text.startswith(EXCLUDED_PREFIXES):
        return "prose_data_or_archive"
    if any(part in EXCLUDED_PARTS for part in path.parts):
        return "runtime_or_binary_tree"

    include_by_name = (
        path.name in SOURCE_NAMES
        or path.name.startswith(("Dockerfile.", "Makefile.", "docker-compose"))
        or _is_requirements_file(path)
    )
    if not include_by_name and path.suffix.lower() not in SOURCE_SUFFIXES:
        return "non_source_type"

    absolute = root / path_text
    if not absolute.is_file():
        return "missing_worktree_file"
    size = absolute.stat().st_size
    if size > max_file_bytes:
        return "file_too_large"
    return None


def select_files(
    root: Path,
    paths: Iterable[str],
    *,
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
) -> tuple[list[SelectedFile], dict[str, int]]:
    root = root.resolve()
    excluded: Counter[str] = Counter()
    selected: list[SelectedFile] = []
    for path_text in paths:
        reason = _exclusion_reason(path_text, root, max_file_bytes)
        if reason is not None:
            excluded[reason] += 1
            continue
        path = PurePosixPath(path_text)
        selected.append(
            SelectedFile(
                relative_path=path_text,
                size_bytes=(root / path_text).stat().st_size,
                priority=_source_priority(path),
            )
        )
    selected.sort(key=lambda item: (item.priority, item.relative_path))
    return selected, dict(sorted(excluded.items()))


def require_clean_source_snapshot(
    files: Sequence[SelectedFile],
    dirty_paths: Iterable[str],
) -> None:
    selected_paths = {item.relative_path for item in files}
    dirty_selected = sorted(selected_paths.intersection(dirty_paths))
    if dirty_selected:
        sample = ", ".join(dirty_selected[:10])
        remainder = len(dirty_selected) - min(len(dirty_selected), 10)
        suffix = f" (+{remainder} more)" if remainder else ""
        raise AuditExportError(
            "Selected audit source differs from HEAD; commit it before export: "
            f"{sample}{suffix}"
        )


def _header(
    *,
    root: Path,
    commit_sha: str,
    files: Sequence[SelectedFile],
    excluded_counts: dict[str, int],
) -> bytes:
    exclusion_summary = json.dumps(excluded_counts, sort_keys=True)
    text = (
        "# AURA CODEBASE FOR AI AUDIT\n"
        "# Scope: tracked implementation, tests, interfaces, workflows, "
        "build/release infrastructure, and source configuration.\n"
        "# Excluded: prose-only docs, runtime state, evidence artifacts, "
        "weights, datasets, archives, generated bundles, and binaries.\n"
        f"# Source root: {root}\n"
        f"# Commit SHA: {commit_sha}\n"
        f"# Files exported: {len(files)}\n"
        f"# Excluded tracked files by reason: {exclusion_summary}\n"
        "# Each source file begins with an explicit repository-relative path.\n\n"
    )
    return text.encode("utf-8")


def _file_header(relative_path: str, size_bytes: int) -> bytes:
    return (
        "\n"
        + "=" * 88
        + f"\nFILE: {relative_path}\nSOURCE_BYTES: {size_bytes}\n"
        + "=" * 88
        + "\n"
    ).encode("utf-8")


def export_bundle(
    *,
    root: Path,
    output_path: Path,
    files: Sequence[SelectedFile],
    excluded_counts: dict[str, int],
    commit_sha: str,
    max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
) -> AuditExportManifest:
    root = root.resolve()
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    header = _header(
        root=root,
        commit_sha=commit_sha,
        files=files,
        excluded_counts=excluded_counts,
    )
    projected_size = len(header)
    for item in files:
        projected_size += len(_file_header(item.relative_path, item.size_bytes))
        projected_size += item.size_bytes + 1
    if projected_size > max_output_bytes:
        raise AuditExportError(
            "Complete selected source would exceed the audit cap: "
            f"{projected_size:,} > {max_output_bytes:,} bytes"
        )

    digest = hashlib.sha256()
    source_bytes = 0
    output_bytes = 0
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)

            def write(payload: bytes) -> None:
                nonlocal output_bytes
                handle.write(payload)
                digest.update(payload)
                output_bytes += len(payload)

            write(header)
            for item in files:
                write(_file_header(item.relative_path, item.size_bytes))
                payload = (root / item.relative_path).read_bytes()
                if len(payload) != item.size_bytes:
                    raise AuditExportError(
                        f"Source changed during export: {item.relative_path}"
                    )
                write(payload)
                source_bytes += len(payload)
                if not payload.endswith(b"\n"):
                    write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, output_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    return AuditExportManifest(
        source_root=str(root),
        output_path=str(output_path),
        generated_at=datetime.now(timezone.utc).isoformat(),
        commit_sha=commit_sha,
        max_output_bytes=max_output_bytes,
        max_file_bytes=max_file_bytes,
        files_exported=len(files),
        source_bytes=source_bytes,
        output_bytes=output_bytes,
        output_sha256=digest.hexdigest(),
        excluded_counts=excluded_counts,
        files=[asdict(item) for item in files],
    )


def verify_bundle(path: Path, manifest: AuditExportManifest) -> None:
    """Verify output framing, ordering, byte counts, and complete-file hash."""
    path = path.expanduser().resolve()
    payload = path.read_bytes()
    if len(payload) != manifest.output_bytes:
        raise AuditExportError(
            f"Audit export size mismatch: {len(payload):,} != {manifest.output_bytes:,}"
        )
    actual_sha256 = hashlib.sha256(payload).hexdigest()
    if actual_sha256 != manifest.output_sha256:
        raise AuditExportError(
            "Audit export SHA-256 mismatch: "
            f"{actual_sha256} != {manifest.output_sha256}"
        )

    files = [SelectedFile(**item) for item in manifest.files]
    expected_header = _header(
        root=Path(manifest.source_root),
        commit_sha=manifest.commit_sha,
        files=files,
        excluded_counts=manifest.excluded_counts,
    )
    if not payload.startswith(expected_header):
        raise AuditExportError("Audit export header does not match its manifest")

    cursor = len(expected_header)
    verified_source_bytes = 0
    for item in files:
        file_header = _file_header(item.relative_path, item.size_bytes)
        if payload[cursor : cursor + len(file_header)] != file_header:
            raise AuditExportError(
                f"Audit export framing mismatch before {item.relative_path}"
            )
        cursor += len(file_header)
        file_end = cursor + item.size_bytes
        if file_end > len(payload):
            raise AuditExportError(
                f"Audit export truncated inside {item.relative_path}"
            )
        file_payload = payload[cursor:file_end]
        cursor = file_end
        verified_source_bytes += len(file_payload)
        if not file_payload.endswith(b"\n"):
            if payload[cursor : cursor + 1] != b"\n":
                raise AuditExportError(
                    f"Audit export separator missing after {item.relative_path}"
                )
            cursor += 1

    if cursor != len(payload):
        raise AuditExportError(
            f"Audit export has {len(payload) - cursor:,} trailing unframed bytes"
        )
    if verified_source_bytes != manifest.source_bytes:
        raise AuditExportError(
            "Audit export source byte count mismatch: "
            f"{verified_source_bytes:,} != {manifest.source_bytes:,}"
        )


def write_manifest(path: Path, manifest: AuditExportManifest) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(asdict(manifest), indent=2, sort_keys=True) + "\n"
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path.home() / "Downloads" / DEFAULT_OUTPUT_NAME,
    )
    parser.add_argument("--max-output-bytes", type=int, default=DEFAULT_MAX_OUTPUT_BYTES)
    parser.add_argument("--max-file-bytes", type=int, default=DEFAULT_MAX_FILE_BYTES)
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    files, excluded_counts = select_files(
        root,
        tracked_paths(root),
        max_file_bytes=args.max_file_bytes,
    )
    if not files:
        raise AuditExportError("No tracked source files matched the audit scope")
    require_clean_source_snapshot(files, dirty_tracked_paths(root))

    manifest = export_bundle(
        root=root,
        output_path=args.output,
        files=files,
        excluded_counts=excluded_counts,
        commit_sha=current_commit(root),
        max_output_bytes=args.max_output_bytes,
        max_file_bytes=args.max_file_bytes,
    )
    verify_bundle(args.output, manifest)
    manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
    write_manifest(manifest_path, manifest)
    print(
        json.dumps(
            {
                "output": manifest.output_path,
                "manifest": str(manifest_path.expanduser().resolve()),
                "commit_sha": manifest.commit_sha,
                "files_exported": manifest.files_exported,
                "output_bytes": manifest.output_bytes,
                "sha256": manifest.output_sha256,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
