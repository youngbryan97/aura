"""Repo-evidence truth engine — answers that cite the codebase must be grounded.

The single most common local hallucination is a confident claim about a file or
symbol that does not exist ("it's handled in ``core/foo/bar.py`` by ``do_thing()``").
This engine extracts ``path/like.py`` references and ``symbol()`` claims from a
candidate and checks them against the actual repository: missing files are a hard
fail; claimed-but-undefined symbols lower the score. Pure filesystem + AST, no
execution, no subprocess.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any

from core.runtime.errors import record_degradation
from core.utils.paths import PROJECT_ROOT

from .base import VerificationResult

# core/foo/bar.py  — a path with at least one slash and a .py/.md/.json/etc suffix.
_PATH_RE = re.compile(r"(?<![\w/.])([A-Za-z_][\w./-]*\.(?:py|md|json|toml|yaml|yml|txt|cfg|ini|sh))(?![\w])")
# `symbol` or symbol() referenced as a definition.
_SYMBOL_RE = re.compile(r"`([A-Za-z_]\w+)`|(?<![\w.])([A-Za-z_]\w+)\(\)")
_IGNORE_SYMBOLS = frozenset({
    "self", "cls", "print", "len", "range", "int", "str", "list", "dict", "set",
    "tuple", "float", "bool", "type", "super", "open", "format", "isinstance",
})


def _iter_referenced_paths(text: str) -> list[str]:
    seen: list[str] = []
    from core.intent.opaque_spans import without_web_addresses

    for m in _PATH_RE.finditer(without_web_addresses(text)):
        ref = m.group(1)
        if ref not in seen and not ref.startswith("http"):
            seen.append(ref)
    return seen[:20]


def _iter_referenced_symbols(text: str) -> list[str]:
    out: list[str] = []
    for m in _SYMBOL_RE.finditer(text or ""):
        sym = m.group(1) or m.group(2)
        if sym and sym not in _IGNORE_SYMBOLS and not sym[0].isupper() and sym not in out:
            out.append(sym)
    return out[:20]


class RepoEvidenceEngine:
    name = "repo_evidence"
    domains = ("repo", "repo_audit", "architecture", "code_audit", "self_claim")

    def __init__(self, root: Path | None = None) -> None:
        self._root = Path(root or PROJECT_ROOT)
        self._symbol_index: set[str] | None = None

    def handles(self, task_type: str) -> bool:
        return task_type in self.domains

    async def verify(self, candidate: str, *, context: dict[str, Any] | None = None) -> VerificationResult:
        text = str(candidate or "")
        paths = _iter_referenced_paths(text)
        if not paths:
            return VerificationResult(domain="repo_evidence", ok=True, checked=False, engine=self.name)

        missing: list[str] = []
        present: list[str] = []
        for ref in paths:
            candidate_path = (self._root / ref)
            if candidate_path.exists() or list(self._root.rglob(Path(ref).name))[:1]:
                present.append(ref)
            else:
                missing.append(ref)

        issues = [f"referenced path not found in repo: {m}" for m in missing]
        evidence = [f"verified path exists: {p}" for p in present]

        # Symbol grounding is advisory (lowers score, not a hard fail) — building a full
        # index is only worth it when the answer makes definition-shaped claims.
        symbols = _iter_referenced_symbols(text)
        undefined: list[str] = []
        if symbols and len(present) >= 1:
            index = self._ensure_symbol_index()
            undefined = [s for s in symbols if s not in index][:8]
            if undefined:
                issues.extend(f"symbol not defined anywhere in repo: {s}()" for s in undefined)

        ok = not missing
        ground_ratio = len(present) / max(1, len(paths))
        score = ground_ratio * (0.95 if not undefined else 0.7)
        return VerificationResult(
            domain="repo_evidence",
            ok=ok,
            checked=True,
            score=round(min(0.98, max(0.05, score)), 4),
            engine=self.name,
            issues=issues,
            evidence=evidence,
            detail={"paths": len(paths), "present": len(present), "missing": len(missing)},
        )

    def _ensure_symbol_index(self) -> set[str]:
        if self._symbol_index is not None:
            return self._symbol_index
        index: set[str] = set()
        try:
            for py in self._root.rglob("*.py"):
                # Skip vendored / cache dirs to keep this bounded.
                parts = set(py.parts)
                if parts & {".venv", "__pycache__", "node_modules", ".git", "archive", "dev_archive"}:
                    continue
                try:
                    tree = ast.parse(py.read_text(encoding="utf-8", errors="ignore"))
                except (SyntaxError, ValueError, OSError):
                    continue
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        index.add(node.name)
        except (OSError, RuntimeError) as exc:
            record_degradation("repo_evidence_index", exc)
        self._symbol_index = index
        return index
