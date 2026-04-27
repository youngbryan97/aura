from __future__ import annotations
"""Machine-readable flagship readiness checks for Aura."""


import ast
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional


@dataclass
class FlagshipIssue:
    code: str
    severity: str
    path: str
    line: int
    message: str
    suggestion: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FlagshipReport:
    root: str
    ok: bool
    issues: list[FlagshipIssue] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "root": self.root,
            "ok": self.ok,
            "issue_count": len(self.issues),
            "issues": [issue.to_dict() for issue in self.issues],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


_EXCLUDE_DIRS = {".git", ".venv", ".venv_aura", "__pycache__", ".pytest_cache", "node_modules", "dist", "build"}
_ALLOWED_CREATE_TASK_FILES = {"core/utils/task_tracker.py", "core/runtime/task_ownership.py", "core/utils/asyncio_patch.py"}
_ALLOWED_DIRECT_WRITE_FILES = {"core/runtime/atomic_writer.py"}


def _iter_py(root: Path) -> Iterable[Path]:
    for path in root.rglob("*.py"):
        rel = path.relative_to(root).as_posix()
        if any(part in _EXCLUDE_DIRS for part in path.parts):
            continue
        if "/.claude/" in f"/{rel}/":
            continue
        yield path


def _rel(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def _is_test(rel: str) -> bool:
    return rel.startswith("tests/") or "/tests/" in rel or rel.startswith("scripts/verify_")


def _safe_read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(errors="ignore")


class _Visitor(ast.NodeVisitor):
    def __init__(self, rel: str):
        self.rel = rel
        self.issues: list[FlagshipIssue] = []
        self._async_depth = 0
        self._class_depth = 0

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> Any:
        self._async_depth += 1
        self.generic_visit(node)
        self._async_depth -= 1

    def visit_ClassDef(self, node: ast.ClassDef) -> Any:
        self._class_depth += 1
        self.generic_visit(node)
        self._class_depth -= 1

    def visit_Call(self, node: ast.Call) -> Any:
        name = self._call_name(node)
        if name in {"asyncio.create_task", "asyncio.ensure_future"}:
            if not _is_test(self.rel) and self.rel not in _ALLOWED_CREATE_TASK_FILES:
                self.issues.append(
                    FlagshipIssue(
                        code="RAW_ASYNCIO_TASK",
                        severity="warning",
                        path=self.rel,
                        line=getattr(node, "lineno", 0),
                        message=f"Raw {name} in production code.",
                        suggestion="Use core.runtime.task_ownership.create_tracked_task/fire_and_forget or TaskTracker.",
                    )
                )
        if name.endswith(".write_text") or name in {"Path.write_text", "pathlib.Path.write_text"}:
            if not _is_test(self.rel) and self.rel not in _ALLOWED_DIRECT_WRITE_FILES:
                self.issues.append(
                    FlagshipIssue(
                        code="DIRECT_WRITE_TEXT",
                        severity="warning",
                        path=self.rel,
                        line=getattr(node, "lineno", 0),
                        message="Direct write_text call may bypass AtomicWriter durability/receipt policy.",
                        suggestion="Use core.runtime.atomic_writer for durable state.",
                    )
                )
        if name == "sys.exit" and self._async_depth > 0:
            self.issues.append(
                FlagshipIssue(
                    code="ASYNC_SYS_EXIT",
                    severity="error",
                    path=self.rel,
                    line=getattr(node, "lineno", 0),
                    message="sys.exit inside async function can bypass cleanup.",
                    suggestion="Return status or delegate process exit to the launcher.",
                )
            )
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> Any:
        self._check_import_time_async_primitive(node.value, getattr(node, "lineno", 0))
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> Any:
        self._check_import_time_async_primitive(node.value, getattr(node, "lineno", 0))
        self.generic_visit(node)

    def _check_import_time_async_primitive(self, value: Any, line: int) -> None:
        if self._class_depth or self._async_depth:
            return
        if isinstance(value, ast.Call):
            name = self._call_name(value)
            if name in {"asyncio.Lock", "asyncio.Event", "asyncio.Semaphore", "asyncio.Queue"}:
                self.issues.append(
                    FlagshipIssue(
                        code="IMPORT_TIME_ASYNC_PRIMITIVE",
                        severity="error",
                        path=self.rel,
                        line=line,
                        message=f"Import-time {name} can bind to the wrong/stale event loop.",
                        suggestion="Lazy-create the primitive inside the running loop.",
                    )
                )

    @staticmethod
    def _call_name(node: ast.Call) -> str:
        func = node.func
        parts: list[str] = []
        while isinstance(func, ast.Attribute):
            parts.append(func.attr)
            func = func.value
        if isinstance(func, ast.Name):
            parts.append(func.id)
        return ".".join(reversed(parts))


def scan_codebase(root: str | Path) -> FlagshipReport:
    root = Path(root).resolve()
    issues: list[FlagshipIssue] = []

    if sys.version_info < (3, 12):
        issues.append(
            FlagshipIssue(
                code="PYTHON_VERSION",
                severity="error",
                path="<runtime>",
                line=0,
                message=f"Running Python {sys.version_info.major}.{sys.version_info.minor}; Aura flagship gate expects 3.12+.",
                suggestion="Run with Python 3.12+.",
            )
        )

    for path in _iter_py(root):
        rel = _rel(root, path)
        source = _safe_read(path)
        try:
            tree = ast.parse(source, filename=rel)
        except SyntaxError as exc:
            issues.append(FlagshipIssue("SYNTAX_ERROR", "error", rel, exc.lineno or 0, str(exc), "Fix syntax before release."))
            continue
        visitor = _Visitor(rel)
        visitor.visit(tree)
        issues.extend(visitor.issues)

    aura_main = root / "aura_main.py"
    if aura_main.exists():
        src = _safe_read(aura_main)
        if "start_morphogenesis_runtime" not in src:
            issues.append(FlagshipIssue("MORPHOGENESIS_NOT_BOOTED", "warning", "aura_main.py", 0, "Morphogenetic runtime boot hook not found.", "Start morphogenesis after core services are registered."))
        if "core.utils.asyncio_patch" not in src:
            issues.append(FlagshipIssue("ASYNCIO_PATCH_NOT_IMPORTED", "warning", "aura_main.py", 0, "Global asyncio task supervision patch not imported.", "Import core.utils.asyncio_patch near the top of aura_main.py."))

    registry = root / "core" / "morphogenesis" / "registry.py"
    if registry.exists():
        src = _safe_read(registry)
        if '"quarantined"' not in src or '"dead"' not in src:
            issues.append(FlagshipIssue("MORPHOGENESIS_STATUS_COUNTERS", "warning", "core/morphogenesis/registry.py", 0, "Morphogenesis registry status lacks direct lifecycle counters.", "Expose direct quarantined/dead/active/dormant counters in status()."))

    ok = not any(issue.severity == "error" for issue in issues)
    return FlagshipReport(root=str(root), ok=ok, issues=issues)


def main(argv: Optional[list[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Aura flagship readiness gate")
    parser.add_argument("root", nargs="?", default=".", help="Aura repository root")
    parser.add_argument("--json", action="store_true", help="Print JSON report")
    parser.add_argument("--strict", action="store_true", help="Treat warnings as failures")
    args = parser.parse_args(argv)

    report = scan_codebase(args.root)
    if args.json:
        print(report.to_json())
    else:
        print(f"Aura flagship gate: {'PASS' if report.ok else 'FAIL'}")
        for issue in report.issues:
            print(f"[{issue.severity.upper()}] {issue.code} {issue.path}:{issue.line} — {issue.message}")
            if issue.suggestion:
                print(f"  -> {issue.suggestion}")

    if args.strict and report.issues:
        return 1
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
