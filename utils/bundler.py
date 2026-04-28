"""
utils/bundler.py
────────────────
Single, authoritative source bundler. Replaces:
  bundle_source.py, export_hardened.py, export_monolith.py,
  export_source.py, export_source_lite.py, make_bundle.py,
  bundle_aura.py, bundle_code.py, bundle_for_desktop.py,
  compact_bundler.py, and the scripts/ duplicates.

Usage:
    python -m utils.bundler                      # full bundle → ~/Downloads/
    python -m utils.bundler --lite               # source only, no binary data
    python -m utils.bundler --out /tmp/aura.txt  # custom path
    python -m utils.bundler --check              # dry-run: list files only
"""
from __future__ import annotations


import argparse
import hashlib
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

# ── Configuration ────────────────────────────────────────────

# Try to pull version from canonical source; graceful fallback
try:
    from core.version import VERSION, CODENAME
    _VERSION_STR = f"{CODENAME} v{VERSION}"
except ImportError:
    _VERSION_STR = "Unknown (core.version not found)"

#: Source extensions always included
SOURCE_EXTS: frozenset[str] = frozenset({
    ".py", ".html", ".css", ".js", ".md", ".txt",
    ".toml", ".yaml", ".yml", ".sh", ".json",
})

#: Extensions included only in full (non-lite) bundles
BINARY_EXTS: frozenset[str] = frozenset({".db", ".sqlite", ".sqlite3"})

#: Directories never included, regardless of mode
EXCLUDE_DIRS: frozenset[str] = frozenset({
    "__pycache__", ".git", ".venv", "venv", "env",
    "node_modules", "dist", "build", ".mypy_cache",
    ".pytest_cache", "data", "logs", "backups",
    "browser_profile", ".sys_temp_integrity",
})

#: Files never included
EXCLUDE_FILES: frozenset[str] = frozenset({
    "bundle_source.py", "export_hardened.py", "export_monolith.py",
    "export_source.py", "export_source_lite.py", "make_bundle.py",
    "bundle_aura.py", "bundle_code.py", "bundle_for_desktop.py",
    "aura_source.txt", "aura_source_code.txt",
    ".DS_Store", "server.pid",
})

# ── Path Discovery ───────────────────────────────────────────

def iter_source_files(root: Path, lite: bool = False) -> Iterator[Path]:
    """
    Yield candidate source files under *root*, applying exclusion rules.

    Args:
        root: Project root directory.
        lite: If True, skip binary files (DB, etc.).
    """
    include_exts = SOURCE_EXTS if lite else SOURCE_EXTS | BINARY_EXTS

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue

        rel = path.relative_to(root)

        # Skip excluded directories
        if any(part in EXCLUDE_DIRS for part in rel.parts):
            continue

        # Skip excluded filenames
        if path.name in EXCLUDE_FILES:
            continue

        # Skip hidden files
        if any(part.startswith(".") for part in rel.parts):
            continue

        if path.suffix.lower() in include_exts:
            yield path


# ── Bundle Writing ───────────────────────────────────────────

def write_bundle(
    root: Path,
    output: Path,
    lite: bool = False,
    check_only: bool = False,
) -> int:
    """
    Write a source bundle to *output*.

    Args:
        root:       Project root.
        output:     Destination file path.
        lite:       Exclude binary files.
        check_only: Dry-run — only list files, do not write.

    Returns:
        Number of files included.
    """
    files  = list(iter_source_files(root, lite=lite))
    mode   = "LITE" if lite else "FULL"
    ts     = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    if check_only:
        print(f"[DRY RUN] {mode} bundle would include {len(files)} files:")
        for f in files:
            print(f"  {f.relative_to(root)}")
        return len(files)

    get_task_tracker().create_task(get_storage_gateway().create_dir(output.parent, cause='write_bundle'))
    sep = "=" * 80

    with output.open("w", encoding="utf-8", errors="replace") as fh:
        fh.write(f"# AURA SOURCE BUNDLE — {mode}\n")
        fh.write(f"# Version: {_VERSION_STR}\n")
        fh.write(f"# Generated: {ts}\n")
        fh.write(f"# Files: {len(files)}\n")
        fh.write(f"# Root: {root}\n")
        fh.write("#\n# " + sep + "\n\n")

        for path in files:
            rel = path.relative_to(root)
            fh.write(f"\n{sep}\n")
            fh.write(f"FILE: {rel}\n")
            fh.write(f"{sep}\n")

            try:
                text = path.read_text(encoding="utf-8", errors="replace")
                fh.write(text)
            except Exception as exc:
                fh.write(f"[ERROR reading {rel}: {exc}]\n")

    # Checksum the output
    sha = hashlib.sha256(output.read_bytes()).hexdigest()[:12]
    size_kb = output.stat().st_size / 1024

    print(f"✅  Bundle written: {output}")
    print(f"    Files:    {len(files)}")
    print(f"    Size:     {size_kb:.1f} KB")
    print(f"    SHA256:   {sha}…")

    return len(files)


# ── CLI ──────────────────────────────────────────────────────

def _default_output(lite: bool) -> Path:
    label  = "lite" if lite else "full"
    stamp  = datetime.now().strftime("%Y%m%d_%H%M")
    fname  = f"aura_source_{label}_{stamp}.txt"
    return Path.home() / "Downloads" / fname


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Aura source bundler — single authoritative script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--lite",  action="store_true", help="Exclude binary/DB files")
    parser.add_argument("--check", action="store_true", help="Dry-run: list files, don't write")
    parser.add_argument("--root",  type=Path, default=None,
                        help="Project root (default: parent of this file)")
    parser.add_argument("--out",   type=Path, default=None,
                        help="Output file path (default: ~/Downloads/aura_source_*.txt)")
    args = parser.parse_args(argv)

    root   = args.root   or Path(__file__).resolve().parent.parent
    output = args.out    or _default_output(args.lite)

    n = write_bundle(root, output, lite=args.lite, check_only=args.check)
    return 0 if n > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
