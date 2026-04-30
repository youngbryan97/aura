#!/usr/bin/env python3
"""Lightweight local security scan for Aura release gates."""
from __future__ import annotations

import json
import re
import ast
import math
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCAN_ROOTS = ("core", "interface", "skills", "tools", "training")
SKIP_PARTS = {"__pycache__", ".git", ".venv", ".venv_aura", "node_modules", "artifacts"}

SECRET_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
    re.compile(r"AIza[0-9A-Za-z_-]{20,}"),
)


def scan() -> dict:
    findings: list[dict] = []
    files_scanned = 0
    for root in SCAN_ROOTS:
        base = ROOT / root
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if not path.is_file() or path.suffix not in {".py", ".js", ".ts", ".tsx", ".json", ".yaml", ".yml", ".toml", ".md"}:
                continue
            if any(part in SKIP_PARTS for part in path.parts):
                continue
            files_scanned += 1
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            rel = path.relative_to(ROOT).as_posix()
            findings.extend(_scan_text_literals(text, rel))
            if path.suffix == ".py":
                findings.extend(_scan_python_ast(text, rel))
    return {
        "generated_at": time.time(),
        "files_scanned": files_scanned,
        "findings": findings,
        "passed": not findings,
    }


def _line(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def _scan_text_literals(text: str, rel: str) -> list[dict]:
    findings: list[dict] = []
    for pattern in SECRET_PATTERNS:
        for match in pattern.finditer(text):
            findings.append({"kind": "secret_like_literal", "file": rel, "line": _line(text, match.start())})
    return findings


def _scan_python_ast(text: str, rel: str) -> list[dict]:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    findings: list[dict] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            findings.extend(_scan_assignment(node, rel))
        elif isinstance(node, ast.AnnAssign):
            findings.extend(_scan_assignment(node, rel))
        elif isinstance(node, ast.Call):
            finding = _scan_call(node, rel)
            if finding:
                findings.append(finding)
    return findings


def _scan_assignment(node: ast.Assign | ast.AnnAssign, rel: str) -> list[dict]:
    names: list[str] = []
    if isinstance(node, ast.Assign):
        for target in node.targets:
            names.extend(_target_names(target))
        value = node.value
    else:
        names.extend(_target_names(node.target))
        value = node.value
    if not names or not isinstance(value, ast.Constant) or not isinstance(value.value, str):
        return []
    lowered = " ".join(names).lower()
    if not any(key in lowered for key in ("api_key", "apikey", "secret", "password", "token")):
        return []
    if lowered.isupper() or "enum" in lowered:
        return []
    literal = value.value.strip()
    if len(literal) < 20 or literal.startswith("$") or literal.startswith("<") or _looks_like_identifier(literal):
        return []
    if _entropy(literal) < 3.0:
        return []
    return [{"kind": "secret_like_literal", "file": rel, "line": getattr(node, "lineno", 1)}]


def _scan_call(node: ast.Call, rel: str) -> dict | None:
    name = _call_name(node.func)
    if name == "subprocess.run":
        for keyword in node.keywords:
            if keyword.arg == "shell" and isinstance(keyword.value, ast.Constant) and keyword.value.value is True:
                return {"kind": "dangerous_runtime_pattern", "file": rel, "line": node.lineno}
    if name == "os.system":
        return {"kind": "dangerous_runtime_pattern", "file": rel, "line": node.lineno}
    if name in {"pickle.load", "pickle.loads"}:
        return {"kind": "dangerous_runtime_pattern", "file": rel, "line": node.lineno}
    return None


def _target_names(node: ast.AST) -> list[str]:
    if isinstance(node, ast.Name):
        return [node.id]
    if isinstance(node, ast.Attribute):
        return [node.attr]
    if isinstance(node, (ast.Tuple, ast.List)):
        out: list[str] = []
        for item in node.elts:
            out.extend(_target_names(item))
        return out
    return []


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _call_name(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    return ""


def _looks_like_identifier(value: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z0-9_.:-]+", value)) and any(ch.isalpha() for ch in value)


def _entropy(value: str) -> float:
    if not value:
        return 0.0
    total = len(value)
    counts = {ch: value.count(ch) for ch in set(value)}
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


def main() -> int:
    report = scan()
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
