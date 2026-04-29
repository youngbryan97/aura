#!/usr/bin/env python3
"""Export Aura architecture source to Downloads.

Outputs:
  * aura_source_part_0001.txt ... with at most 4,000,000 characters each.
  * aura_source_copy/ with the top 1000 architecture files by priority.
  * aura_source_manifest.json with counts, paths, and limits.
"""
from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List


DEFAULT_CHAR_LIMIT = 4_000_000
DEFAULT_COPY_LIMIT = 1000

ARCHITECTURE_SUFFIXES = {
    ".cfg",
    ".conf",
    ".css",
    ".html",
    ".ini",
    ".js",
    ".jsx",
    ".json",
    ".md",
    ".proto",
    ".py",
    ".rs",
    ".sh",
    ".toml",
    ".ts",
    ".tsx",
    ".yaml",
    ".yml",
}

EXCLUDED_DIR_PARTS = {
    ".aura",
    ".aura_runtime",
    ".claude",
    ".git",
    ".mypy_cache",
    ".next",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "artifacts",
    "assets",
    "backups",
    "brain",
    "build",
    "data",
    "dist",
    "logs",
    "memory_store",
    "models",
    "models_gguf",
    "node_modules",
    "qa_reports",
    "scratch",
    "storage",
    "test_brain",
    "test_data",
    "test_vdb",
    "venv",
}

EXCLUDED_REL_PREFIXES = {
    "interface/static/dist/",
    "training/adapters/",
    "training/fused-model/",
    "training/data/",
}

EXCLUDED_SUFFIXES = {
    ".bin",
    ".db",
    ".dylib",
    ".exe",
    ".gguf",
    ".h5",
    ".ico",
    ".jpg",
    ".jpeg",
    ".log",
    ".map",
    ".mp3",
    ".npy",
    ".npz",
    ".onnx",
    ".pdf",
    ".pkl",
    ".png",
    ".pt",
    ".pth",
    ".pyc",
    ".pyd",
    ".pyo",
    ".safetensors",
    ".so",
    ".sqlite",
    ".tar",
    ".wav",
    ".whl",
    ".zip",
}

EXCLUDED_NAMES = {
    ".DS_Store",
    "aura_source_manifest.json",
    "aura_source_part_0001.txt",
}


@dataclass(frozen=True)
class ExportedPart:
    path: str
    chars: int


@dataclass(frozen=True)
class ExportManifest:
    source_root: str
    output_dir: str
    copied_dir: str
    generated_at: str
    char_limit_per_part: int
    copy_file_limit: int
    files_exported: int
    files_copied: int
    total_chars: int
    parts: List[ExportedPart]

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["parts"] = [asdict(part) for part in self.parts]
        return payload


def is_architecture_file(path: Path, root: Path) -> bool:
    rel = path.relative_to(root).as_posix()
    if path.name in EXCLUDED_NAMES:
        return False
    if any(part in EXCLUDED_DIR_PARTS for part in path.relative_to(root).parts):
        return False
    if any(rel.startswith(prefix) for prefix in EXCLUDED_REL_PREFIXES):
        return False
    if path.suffix.lower() in EXCLUDED_SUFFIXES:
        return False
    if path.name in {"Dockerfile", "Makefile"}:
        return True
    if path.name.startswith(("Dockerfile.", "Makefile.", "docker-compose")):
        return True
    return path.suffix.lower() in ARCHITECTURE_SUFFIXES


def priority(path: Path, root: Path) -> tuple[int, str]:
    rel = path.relative_to(root).as_posix()
    if rel.startswith("core/") and path.suffix == ".py":
        return (1, rel)
    if rel in {"aura_main.py", "ARCHITECTURE.md", "HOW_IT_WORKS.md", "ROADMAP.md", "TESTING.md"}:
        return (2, rel)
    if rel.startswith(("llm/", "senses/", "security/", "interface/")) and path.suffix in {".py", ".js", ".jsx", ".css", ".html"}:
        return (3, rel)
    if rel.startswith(("tools/", "scripts/", "docker/", "rust_extensions/", "proof_kernel/")):
        return (4, rel)
    if rel.startswith(("tests/", "aura_bench/")):
        return (5, rel)
    if rel.startswith(("training/", "docs/", "specs/", "demos/", "scoping/")):
        return (6, rel)
    if path.suffix in {".toml", ".yaml", ".yml", ".json", ".sh"}:
        return (7, rel)
    return (8, rel)


def discover_files(root: Path) -> List[Path]:
    files = [path for path in root.rglob("*") if path.is_file() and is_architecture_file(path, root)]
    return sorted(files, key=lambda path: priority(path, root))


def cleanup_previous(output_dir: Path, copy_dir: Path) -> None:
    for part in output_dir.glob("aura_source_part_*.txt"):
        part.unlink()
    if copy_dir.exists():
        shutil.rmtree(copy_dir)
    copy_dir.mkdir(parents=True, exist_ok=True)


def write_text_parts(files: List[Path], root: Path, output_dir: Path, char_limit: int) -> tuple[List[ExportedPart], int]:
    if char_limit < 10_000:
        raise ValueError("char limit is too small")
    parts: List[ExportedPart] = []
    part_number = 1
    current_chars = 0
    total_chars = 0
    current_path = output_dir / f"aura_source_part_{part_number:04d}.txt"
    handle = current_path.open("w", encoding="utf-8")

    def close_part() -> None:
        nonlocal handle, current_chars
        handle.flush()
        handle.close()
        parts.append(ExportedPart(path=str(current_path), chars=current_chars))

    def open_next_part() -> None:
        nonlocal part_number, current_path, current_chars, handle
        close_part()
        part_number += 1
        current_path = output_dir / f"aura_source_part_{part_number:04d}.txt"
        current_chars = 0
        handle = current_path.open("w", encoding="utf-8")

    def write_chunk(text: str) -> None:
        nonlocal current_chars, total_chars
        start = 0
        while start < len(text):
            remaining = char_limit - current_chars
            if remaining <= 0:
                open_next_part()
                remaining = char_limit
            chunk = text[start:start + remaining]
            handle.write(chunk)
            current_chars += len(chunk)
            total_chars += len(chunk)
            start += len(chunk)

    banner = (
        "AURA FULL ARCHITECTURE SOURCE EXPORT\n"
        f"Generated: {datetime.now().isoformat(timespec='seconds')}\n"
        f"Source root: {root}\n"
        f"Character limit per part: {char_limit}\n\n"
    )
    write_chunk(banner)
    for path in files:
        rel = path.relative_to(root).as_posix()
        header = "\n\n" + ("=" * 78) + f"\nFILE: {rel}\n" + ("=" * 78) + "\n"
        write_chunk(header)
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            text = f"[UNREADABLE: {type(exc).__name__}: {exc}]\n"
        write_chunk(text)
        if not text.endswith("\n"):
            write_chunk("\n")
    close_part()
    return parts, total_chars


def copy_top_files(files: List[Path], root: Path, copy_dir: Path, limit: int) -> int:
    copied = 0
    for path in files[:limit]:
        rel = path.relative_to(root)
        dest = copy_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
        copied += 1
    return copied


def export(root: Path, output_dir: Path, *, char_limit: int, copy_limit: int) -> ExportManifest:
    root = root.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    copy_dir = output_dir / "aura_source_copy"
    cleanup_previous(output_dir, copy_dir)

    files = discover_files(root)
    parts, total_chars = write_text_parts(files, root, output_dir, char_limit)
    copied = copy_top_files(files, root, copy_dir, copy_limit)
    manifest = ExportManifest(
        source_root=str(root),
        output_dir=str(output_dir),
        copied_dir=str(copy_dir),
        generated_at=datetime.now().isoformat(timespec="seconds"),
        char_limit_per_part=char_limit,
        copy_file_limit=copy_limit,
        files_exported=len(files),
        files_copied=copied,
        total_chars=total_chars,
        parts=parts,
    )
    manifest_path = output_dir / "aura_source_manifest.json"
    manifest_path.write_text(json.dumps(manifest.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--output-dir", default=str(Path.home() / "Downloads"))
    parser.add_argument("--char-limit", type=int, default=DEFAULT_CHAR_LIMIT)
    parser.add_argument("--copy-limit", type=int, default=DEFAULT_COPY_LIMIT)
    args = parser.parse_args()

    manifest = export(
        Path(args.root),
        Path(args.output_dir),
        char_limit=args.char_limit,
        copy_limit=args.copy_limit,
    )
    print(json.dumps(manifest.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
