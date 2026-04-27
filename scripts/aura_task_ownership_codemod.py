#!/usr/bin/env python3
# Conservative task ownership scanner/codemod.
#
# Default mode is report-only. It finds raw asyncio.create_task /
# asyncio.ensure_future calls in production files. With --apply, it only
# rewrites the very narrow and safe pattern:
#   asyncio.ensure_future(record_organ_formation_episode(...))
# into fire_and_forget(...), because this exact pattern is known to be
# non-critical episodic telemetry.

from __future__ import annotations

import argparse
import ast
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


EXCLUDE_DIRS = {".git", ".venv", ".venv_aura", "__pycache__", ".pytest_cache", "node_modules", "dist", "build"}
ALLOWED = {"core/utils/task_tracker.py", "core/runtime/task_ownership.py", "core/utils/asyncio_patch.py"}


@dataclass
class Finding:
    path: str
    line: int
    kind: str
    text: str
    safe_autofix: bool = False

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


def scan_file(root: Path, path: Path) -> list[Finding]:
    rel = path.relative_to(root).as_posix()
    src = path.read_text(encoding="utf-8", errors="ignore")
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return [Finding(rel, 0, "syntax_error", "file does not parse", False)]

    findings: list[Finding] = []
    lines = src.splitlines()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = call_name(node)
            if name in {"asyncio.create_task", "asyncio.ensure_future"}:
                line_no = getattr(node, "lineno", 0)
                text = lines[line_no - 1].strip() if 0 < line_no <= len(lines) else name
                safe = "record_organ_formation_episode" in text and name == "asyncio.ensure_future"
                findings.append(Finding(rel, line_no, name, text, safe))
    return findings


def scan(root: Path) -> list[Finding]:
    out: list[Finding] = []
    for path in iter_py(root):
        out.extend(scan_file(root, path))
    return out


def apply_known_fixes(root: Path, findings: list[Finding]) -> int:
    changed = 0
    for f in findings:
        if not f.safe_autofix:
            continue
        path = root / f.path
        src = path.read_text(encoding="utf-8")
        old = "asyncio.ensure_future(record_organ_formation_episode(organ.to_dict()))"
        new = (
            "from core.runtime.task_ownership import fire_and_forget\n"
            "                        fire_and_forget(\n"
            "                            record_organ_formation_episode(organ.to_dict()),\n"
            "                            name=\"morphogenesis.organ_episode\",\n"
            "                            bounded=True,\n"
            "                        )"
        )
        if old in src:
            backup = path.with_suffix(path.suffix + ".bak_taskcodemod")
            if not backup.exists():
                backup.write_text(src, encoding="utf-8")
            path.write_text(src.replace(old, new, 1), encoding="utf-8")
            changed += 1
    return changed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", default=".")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    findings = scan(root)
    changed = apply_known_fixes(root, findings) if args.apply else 0

    if args.json:
        print(json.dumps({"root": str(root), "changed": changed, "findings": [f.to_dict() for f in findings]}, indent=2))
    else:
        print(f"Task ownership findings: {len(findings)}")
        if changed:
            print(f"Applied safe fixes: {changed}")
        for f in findings:
            mark = "AUTO" if f.safe_autofix else "MANUAL"
            print(f"[{mark}] {f.path}:{f.line} {f.kind} — {f.text}")

    return 1 if findings and not args.apply else 0


if __name__ == "__main__":
    raise SystemExit(main())
