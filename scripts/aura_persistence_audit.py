#!/usr/bin/env python3
"""Audit direct persistence writes in Aura.

This is report-first. It identifies places that may bypass AtomicWriter /
persistence_ownership. It does not blindly rewrite because some write_text calls
are harmless generated assets, docs, or tests.
"""

from __future__ import annotations

import argparse
import ast
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


EXCLUDE_DIRS = {".git", ".venv", ".venv_aura", "__pycache__", ".pytest_cache", "node_modules", "dist", "build"}
ALLOWED = {
    "core/runtime/atomic_writer.py",
    "core/runtime/persistence_ownership.py",
}


@dataclass
class PersistenceFinding:
    path: str
    line: int
    kind: str
    text: str
    severity: str = "warning"
    suggestion: str = "Use core.runtime.persistence_ownership or core.runtime.atomic_writer for durable state."

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def iter_py(root: Path) -> Iterable[Path]:
    for p in root.rglob("*.py"):
        rel = p.relative_to(root).as_posix()
        if any(part in EXCLUDE_DIRS for part in p.parts):
            continue
        if rel.startswith("tests/") or "/tests/" in rel:
            continue
        if rel in ALLOWED:
            continue
        yield p


def call_name(node: ast.Call) -> str:
    func = node.func
    parts: list[str] = []
    while isinstance(func, ast.Attribute):
        parts.append(func.attr)
        func = func.value
    if isinstance(func, ast.Name):
        parts.append(func.id)
    return ".".join(reversed(parts))


def is_open_write_call(node: ast.Call) -> bool:
    name = call_name(node)
    if name != "open":
        return False
    if len(node.args) < 2:
        return False
    mode = node.args[1]
    if isinstance(mode, ast.Constant) and isinstance(mode.value, str):
        return any(flag in mode.value for flag in ("w", "a", "+"))
    return False


def scan_file(root: Path, path: Path) -> list[PersistenceFinding]:
    rel = path.relative_to(root).as_posix()
    src = path.read_text(encoding="utf-8", errors="ignore")
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return [PersistenceFinding(rel, 0, "syntax_error", "file does not parse", "error", "Fix syntax before release.")]

    lines = src.splitlines()
    findings: list[PersistenceFinding] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = call_name(node)
        line = getattr(node, "lineno", 0)
        text = lines[line - 1].strip() if 0 < line <= len(lines) else name

        if name.endswith(".write_text") or name.endswith(".write_bytes"):
            findings.append(PersistenceFinding(rel, line, name, text))
        elif name in {"json.dump", "yaml.dump", "toml.dump"}:
            findings.append(PersistenceFinding(rel, line, name, text))
        elif is_open_write_call(node):
            findings.append(PersistenceFinding(rel, line, "open_write", text))

    return findings


def scan(root: Path) -> list[PersistenceFinding]:
    out: list[PersistenceFinding] = []
    for path in iter_py(root):
        out.extend(scan_file(root, path))
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", default=".")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--fail-on-error", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    findings = scan(root)

    if args.json:
        print(json.dumps({"root": str(root), "finding_count": len(findings), "findings": [f.to_dict() for f in findings]}, indent=2))
    else:
        print(f"Persistence findings: {len(findings)}")
        for f in findings:
            print(f"[{f.severity.upper()}] {f.path}:{f.line} {f.kind} — {f.text}")
            print(f"  -> {f.suggestion}")

    if args.fail_on_error and any(f.severity == "error" for f in findings):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
