"""Read-only source self-model for Aura.

The model parses architecture files into hashes, imports, and public symbols so
RSI validation can reason about the codebase without granting write or execute
authority. It is deliberately introspective only: edits still go through the
existing self-modification safety pipeline.
"""
from __future__ import annotations

import ast
import hashlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


ARCHITECTURE_SUFFIXES = {
    ".py",
    ".toml",
    ".yaml",
    ".yml",
    ".json",
    ".cfg",
    ".ini",
    ".conf",
    ".sh",
    ".rs",
}

EXCLUDED_PARTS = {
    ".git",
    ".aura_runtime",
    ".claude",
    ".mypy_cache",
    ".venv",
    "venv",
    "__pycache__",
    "archive",
    "node_modules",
    "dist",
    "build",
    "models",
    "training/adapters",
    "training/fused-model",
    "data",
    "storage",
    "memory_store",
    "scratch",
}

PROTECTED_SYMBOLS = {
    "AuthorityGateway",
    "ConstitutionalGuard",
    "ResourceGovernor",
    "SafeSelfModification",
    "UnifiedWill",
}

PROTECTED_SOURCE_PATHS = {
    "core/executive/authority_gateway.py",
    "core/learning/recursive_self_improvement.py",
    "core/resilience/resource_governor.py",
    "core/security/constitutional_guard.py",
    "core/self_modification/safe_modification.py",
    "core/will.py",
}


@dataclass(frozen=True)
class SourceSymbol:
    name: str
    kind: str
    line: int
    end_line: int
    qualname: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SourceFile:
    path: str
    sha256: str
    size_bytes: int
    language: str
    imports: List[str] = field(default_factory=list)
    symbols: List[SourceSymbol] = field(default_factory=list)
    syntax_error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["symbols"] = [symbol.to_dict() for symbol in self.symbols]
        return payload


@dataclass(frozen=True)
class SourceModel:
    root: str
    files: List[SourceFile]

    @property
    def file_count(self) -> int:
        return len(self.files)

    @property
    def symbol_count(self) -> int:
        return sum(len(file.symbols) for file in self.files)

    @property
    def protected_symbols_present(self) -> Dict[str, bool]:
        names = {symbol.name for file in self.files for symbol in file.symbols}
        return {name: name in names for name in sorted(PROTECTED_SYMBOLS)}

    def find_symbol(self, name: str) -> List[SourceSymbol]:
        return [
            symbol
            for file in self.files
            for symbol in file.symbols
            if symbol.name == name or symbol.qualname.endswith(f".{name}")
        ]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "root": self.root,
            "file_count": self.file_count,
            "symbol_count": self.symbol_count,
            "protected_symbols_present": self.protected_symbols_present,
            "files": [file.to_dict() for file in self.files],
        }


class SourceIntrospector:
    """Build a read-only AST/text index over the architecture code."""

    def __init__(self, root: Path | str):
        self.root = Path(root).resolve()

    def iter_architecture_files(self, *, max_files: int = 5000) -> Iterable[Path]:
        candidates: List[Path] = []
        for path in self.root.rglob("*"):
            if not path.is_file():
                continue
            rel = path.relative_to(self.root)
            rel_parts = rel.parts
            rel_string = rel.as_posix()
            if any(part in EXCLUDED_PARTS for part in rel_parts):
                continue
            if any(excluded in rel_string for excluded in EXCLUDED_PARTS if "/" in excluded):
                continue
            if path.name in {"Dockerfile", "Makefile"} or path.suffix.lower() in ARCHITECTURE_SUFFIXES:
                candidates.append(path)

        def priority(path: Path) -> tuple[int, str]:
            rel = path.relative_to(self.root).as_posix()
            if rel in PROTECTED_SOURCE_PATHS:
                return (0, rel)
            if rel.startswith("core/"):
                return (1, rel)
            if rel.startswith(("llm/", "senses/", "interface/", "security/", "tools/")):
                return (2, rel)
            if rel.startswith(("tests/", "scripts/", "training/")):
                return (4, rel)
            return (3, rel)

        for path in sorted(candidates, key=priority)[:max_files]:
            yield path

    def build(self, *, max_files: int = 5000) -> SourceModel:
        files = [self._index_file(path) for path in self.iter_architecture_files(max_files=max_files)]
        return SourceModel(root=str(self.root), files=files)

    def get_file_text(self, rel_path: str, *, max_bytes: int = 2_000_000) -> str:
        target = (self.root / rel_path).resolve()
        target.relative_to(self.root)
        if not target.is_file():
            raise FileNotFoundError(rel_path)
        if target.stat().st_size > max_bytes:
            raise ValueError(f"source file too large for direct recall: {rel_path}")
        return target.read_text(encoding="utf-8", errors="replace")

    def _index_file(self, path: Path) -> SourceFile:
        rel = path.relative_to(self.root).as_posix()
        data = path.read_bytes()
        sha = "sha256:" + hashlib.sha256(data).hexdigest()
        language = path.suffix.lower().lstrip(".") or path.name
        if path.suffix.lower() != ".py":
            return SourceFile(path=rel, sha256=sha, size_bytes=len(data), language=language)

        source = data.decode("utf-8", errors="replace")
        try:
            tree = ast.parse(source, filename=rel)
        except SyntaxError as exc:
            return SourceFile(
                path=rel,
                sha256=sha,
                size_bytes=len(data),
                language=language,
                syntax_error=f"{exc.msg}:line{exc.lineno or 0}",
            )

        imports: List[str] = []
        symbols: List[SourceSymbol] = []
        module = rel[:-3].replace("/", ".")
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                imports.extend(f"{mod}.{alias.name}".strip(".") for alias in node.names)
            elif isinstance(node, ast.ClassDef):
                symbols.append(
                    SourceSymbol(
                        name=node.name,
                        kind="class",
                        line=node.lineno,
                        end_line=node.end_lineno or node.lineno,
                        qualname=f"{module}.{node.name}",
                    )
                )
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                symbols.append(
                    SourceSymbol(
                        name=node.name,
                        kind="async_function" if isinstance(node, ast.AsyncFunctionDef) else "function",
                        line=node.lineno,
                        end_line=node.end_lineno or node.lineno,
                        qualname=f"{module}.{node.name}",
                    )
                )
        return SourceFile(
            path=rel,
            sha256=sha,
            size_bytes=len(data),
            language=language,
            imports=sorted(set(imports)),
            symbols=sorted(symbols, key=lambda item: (item.line, item.name)),
        )


__all__ = [
    "PROTECTED_SYMBOLS",
    "SourceFile",
    "SourceIntrospector",
    "SourceModel",
    "SourceSymbol",
]
