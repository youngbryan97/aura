#!/usr/bin/env python3
"""Generate Aura release provenance and a lightweight dependency SBOM."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_OUTPUT_DIR = ROOT / "artifacts" / "provenance"


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".aura_provenance_", dir=str(path.parent), text=True)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except OSError:
        tmp.unlink(missing_ok=True)
        raise


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_head() -> str:
    git_dir = ROOT / ".git"
    if git_dir.is_file():
        content = git_dir.read_text(encoding="utf-8", errors="ignore").strip()
        if content.startswith("gitdir:"):
            git_dir = (ROOT / content.split(":", 1)[1].strip()).resolve()
    head = git_dir / "HEAD"
    if not head.exists():
        return ""
    value = head.read_text(encoding="utf-8", errors="ignore").strip()
    if not value.startswith("ref:"):
        return value
    ref = git_dir / value.split(" ", 1)[1].strip()
    return ref.read_text(encoding="utf-8", errors="ignore").strip() if ref.exists() else ""


def _parse_requirement(line: str) -> dict[str, str] | None:
    value = line.strip()
    if not value or value.startswith("#") or value.startswith("-"):
        return None
    for marker in ("==", ">=", "<=", "~=", ">", "<"):
        if marker in value:
            name, version = value.split(marker, 1)
            return {"name": name.strip(), "specifier": marker + version.strip(), "source": "requirements"}
    return {"name": value, "specifier": "", "source": "requirements"}


def _dependencies() -> list[dict[str, str]]:
    deps: list[dict[str, str]] = []
    pyproject = ROOT / "pyproject.toml"
    if pyproject.exists():
        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        for item in data.get("project", {}).get("dependencies", []) or []:
            parsed = _parse_requirement(str(item))
            if parsed:
                parsed["source"] = "pyproject"
                deps.append(parsed)
    for req in sorted((ROOT / "requirements").glob("*.txt")):
        for line in req.read_text(encoding="utf-8", errors="ignore").splitlines():
            parsed = _parse_requirement(line)
            if parsed:
                parsed["source"] = req.relative_to(ROOT).as_posix()
                deps.append(parsed)
    seen: set[tuple[str, str, str]] = set()
    unique: list[dict[str, str]] = []
    for dep in deps:
        key = (dep["name"].lower(), dep.get("specifier", ""), dep.get("source", ""))
        if key not in seen:
            unique.append(dep)
            seen.add(key)
    return unique


def _important_files() -> list[dict[str, str]]:
    paths = [
        "pyproject.toml",
        "requirements/core.txt",
        "requirements/dev.txt",
        "Makefile",
        ".github/workflows/enterprise-gate.yml",
        ".github/workflows/production-readiness.yml",
        ".github/workflows/release.yml",
        "tools/aura_enterprise_gate.py",
        "tools/aura_production_readiness_gate.py",
        "tools/security_scan.py",
        "tools/proof_bundle.py",
        "docs/PRODUCTION_READINESS_STANDARD.md",
        "docs/DATA_RETENTION_DELETION_POLICY.md",
        "docs/MODEL_PROVIDER_FAILURE_POLICY.md",
        "docs/OPERATOR_GUIDE.md",
    ]
    out: list[dict[str, str]] = []
    for rel in paths:
        path = ROOT / rel
        if path.exists():
            out.append({"path": rel, "sha256": _sha256(path)})
    return out


def build(output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    commit = os.environ.get("GITHUB_SHA") or _git_head()
    status = os.environ.get("AURA_GIT_STATUS_SHORT")
    sbom = {
        "schema": "aura-lightweight-sbom/v1",
        "generated_at": time.time(),
        "dependency_count": len(_dependencies()),
        "dependencies": _dependencies(),
    }
    provenance = {
        "schema": "aura-release-provenance/v1",
        "generated_at": time.time(),
        "git_commit": commit,
        "git_dirty": None if status is None else bool(status),
        "git_status_short": status if status is not None else "unavailable_without_git_subprocess",
        "builder": "tools/build_provenance.py",
        "python_version": sys.version.split()[0],
        "materials": _important_files(),
        "sbom_path": "sbom.json",
    }
    _atomic_write_text(
        output_dir / "sbom.json",
        json.dumps(sbom, indent=2, sort_keys=True),
    )
    _atomic_write_text(
        output_dir / "provenance.json",
        json.dumps(provenance, indent=2, sort_keys=True),
    )
    return {"sbom": sbom, "provenance": provenance}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args(argv)
    report = build(Path(args.output_dir))
    print(json.dumps({"ok": True, "output_dir": args.output_dir, "dependency_count": report["sbom"]["dependency_count"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
