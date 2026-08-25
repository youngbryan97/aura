#!/usr/bin/env python3
"""Authoritative Production Surface Linter for Aura.

Scans production paths for architectural bypasses, raw task creations,
direct writes, hardcoded paths, and swallowed exceptions.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

EXCLUDED_DIRS = {
    ".git",
    ".venv",
    ".venv_aura",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    "archive",
    "dev_archive",
    "artifacts",
    "build",
    "dist",
    "node_modules",
    "tests",
    "tools",
    "scratch",
    "demos",
    "experiments",
    "benchmarks",
    "cloud",
    "training",
    "integration",
    ".claude",
    ".aura_architect",
    ".aura_runtime",
    ".aura_snapshots",
    "scripts",
    "aura_bench",
}
ALWAYS_EXCLUDED_DIRS = {
    ".git",
    ".venv",
    ".venv_aura",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    "build",
    "dist",
    "node_modules",
}
ROOT_EXCLUDED_DIRS = EXCLUDED_DIRS - ALWAYS_EXCLUDED_DIRS

# Production files with audited exceptions. The target for closure is zero; keep
# this structure only so the gate can report regressions if a future exception is
# deliberately introduced with compensating evidence.
EXEMPT_FILES: dict[str, dict[str, str]] = {}

_HARDCODED_LOCAL_PATH = re.compile(r"/(Users|home|tmp)/[a-zA-Z0-9_-]+")


@dataclass
class LintFinding:
    severity: str
    kind: str
    file: str
    line: int
    message: str


def _docstring_nodes(tree: ast.AST) -> set[int]:
    """Return identities of AST constants that are structural docstrings."""

    identities: set[int] = set()
    owners = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
    for owner in ast.walk(tree):
        if not isinstance(owner, owners) or not owner.body:
            continue
        first = owner.body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            identities.add(id(first.value))
    return identities


def hardcoded_local_path_findings(tree: ast.AST, rel: str) -> list[LintFinding]:
    """Find machine-specific paths in executable string literals only.

    Comments and docstrings often document hostile examples or expected error
    messages. Treating them as runtime configuration creates false closure work
    while allowing the actual risky form, a string consumed by code, to hide in
    the same noise.
    """

    docstrings = _docstring_nodes(tree)
    findings: list[LintFinding] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in docstrings
            and _HARDCODED_LOCAL_PATH.search(node.value)
        ):
            findings.append(
                LintFinding(
                    "high",
                    "hardcoded_local_path",
                    rel,
                    int(getattr(node, "lineno", 0) or 0),
                    "Hardcoded local path detected in executable string literal.",
                )
            )
    return findings


def iter_files(scope: str) -> Iterable[Path]:
    root_prune_dirs = set(ROOT_EXCLUDED_DIRS)
    if scope == "repo":
        root_prune_dirs.discard("tests")
        root_prune_dirs.discard("tools")
    for root, dirs, files in os.walk(ROOT):
        rel_root = Path(root).resolve().relative_to(ROOT)
        kept_dirs: list[str] = []
        for dirname in dirs:
            if dirname in ALWAYS_EXCLUDED_DIRS:
                continue
            if rel_root == Path(".") and dirname in root_prune_dirs:
                continue
            kept_dirs.append(dirname)
        dirs[:] = kept_dirs
        for f in files:
            if f.endswith(".py"):
                yield Path(root) / f



APPROVED_SUBPROCESS_SINKS = {
    "core/runtime/action_executor.py",
    "core/runtime/desktop_action_gateway.py",
    "core/runtime/subprocess_gateway.py",
    # Documented emergency exemption: the external memory sentinel must
    # be spawned at canonical boot BEFORE governance services exist —
    # it is the killer that survives when the governed process cannot.
    "aura_main.py",
}
APPROVED_NETWORK_SINKS = {
    "core/runtime/action_executor.py",
    "core/runtime/network_gateway.py",
}
APPROVED_FILE_WRITE_SINKS = {
    "core/runtime/archive_gateway.py",
    "core/runtime/action_executor.py",
    "core/runtime/atomic_writer.py",
    "core/runtime/file_write_gateway.py",
    "core/runtime/post_action_receipt.py",
    # Documented emergency exemptions (crash forensics): faulthandler
    # requires a raw OS file descriptor and must work precisely when the
    # governed gateway stack cannot be trusted or scheduled — these
    # writes are append-only diagnostic dumps under data/error_logs/.
    "aura_main.py",                       # fault forensics + sentinel log
    "core/resilience/memory_watchdog.py", # spike-stack faulthandler dumps
    "core/resilience/stall_watchdog.py",  # loop-wedge faulthandler dump before hard-exit
}
APPROVED_DYNAMIC_CODE_SINKS = {
    # Ephemeral, one-request native-sandbox worker. The governed parent owns
    # admission, deadline, framing, and lifecycle; this is its audited sink.
    "core/agency/repl_daemon.py",
    "core/runtime/dynamic_execution_gateway.py",
}


def is_approved_direct_surface(rel_path: str, kind: str) -> bool:
    if kind == "unapproved_direct_subprocess":
        return rel_path in APPROVED_SUBPROCESS_SINKS
    if kind == "unapproved_direct_network":
        return rel_path in APPROVED_NETWORK_SINKS
    if kind == "unapproved_direct_file_write":
        return rel_path in APPROVED_FILE_WRITE_SINKS
    if kind == "raw_dynamic_code":
        return rel_path in APPROVED_DYNAMIC_CODE_SINKS
    return False


class AstLinter(ast.NodeVisitor):
    def __init__(self, rel: str):
        self.rel = rel
        self.findings: list[LintFinding] = []
        self.async_depth = 0
        #: Innermost-first: whether the function we are inside is a plain def.
        self._in_sync_function: list[bool] = []
        self.func_depth = 0
        self.file_gateway_vars: set[str] = set()
        self.in_memory_binary_vars: set[str] = set()
        self.import_aliases: dict[str, str] = {}

    def add(self, severity: str, kind: str, node: ast.AST, message: str) -> None:
        if self.rel in EXEMPT_FILES:
            return  # Audited and exempted from strict lints
        if kind in {"unapproved_direct_subprocess", "unapproved_direct_network", "unapproved_direct_file_write", "raw_dynamic_code"}:
            if is_approved_direct_surface(self.rel, kind):
                return
        self.findings.append(
            LintFinding(severity, kind, self.rel, getattr(node, "lineno", 0), message)
        )

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.async_depth += 1
        self.func_depth += 1
        self._in_sync_function.append(False)
        self.generic_visit(node)
        self._in_sync_function.pop()
        self.async_depth -= 1
        self.func_depth -= 1

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.func_depth += 1
        # A plain def nested inside a coroutine is a THREAD BODY, not the
        # coroutine. core/brain/llm/nucleus_manager.py defines _thread_worker
        # inside an async generator and runs it on a worker thread, where
        # time.sleep is the correct call — and was reported as a blocking
        # sleep on the event loop because any enclosing async def counted.
        self._in_sync_function.append(True)
        self.generic_visit(node)
        self._in_sync_function.pop()
        self.func_depth -= 1

    @property
    def _on_the_event_loop(self) -> bool:
        return self.async_depth > 0 and not (
            self._in_sync_function and self._in_sync_function[-1]
        )

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if alias.name in {"asyncio", "httpx", "os", "requests", "subprocess", "time", "urllib.request"}:
                if alias.asname:
                    self.import_aliases[alias.asname] = alias.name
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module in {"asyncio", "httpx", "os", "requests", "subprocess", "time", "urllib.request"}:
            for alias in node.names:
                if alias.name == "*":
                    continue
                self.import_aliases[alias.asname or alias.name] = f"{node.module}.{alias.name}"
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        if isinstance(node.value, ast.Call) and self._call_name(node.value) == "get_file_write_gateway":
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self.file_gateway_vars.add(target.id)
        if isinstance(node.value, ast.Call) and self._call_name(node.value) == "io.BytesIO":
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self.in_memory_binary_vars.add(target.id)
        self.generic_visit(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        broad = node.type is None or (
            isinstance(node.type, ast.Name) and node.type.id in {"Exception", "BaseException"}
        )
        if broad:
            exception_name = node.name if isinstance(node.name, str) else ""

            def return_reports_exception(stmt: ast.stmt) -> bool:
                if not isinstance(stmt, ast.Return) or stmt.value is None or not exception_name:
                    return False
                return any(
                    isinstance(child, ast.Name) and child.id == exception_name
                    for child in ast.walk(stmt.value)
                )

            has_pass = any(isinstance(stmt, ast.Pass) for stmt in node.body) or all(
                isinstance(stmt, (ast.Pass, ast.Break, ast.Continue))
                or (isinstance(stmt, ast.Return) and not return_reports_exception(stmt))
                for stmt in node.body
            )
            if has_pass:
                self.add(
                    "high",
                    "swallowed_broad_exception",
                    node,
                    "Broad except blocks must not silently swallow exceptions.",
                )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        name = self._call_name(node)
        task_owner = name.rsplit(".", 2)[-2] if name.endswith(".create_task") else ""
        raw_loop_task = task_owner in {
            "loop",
            "event_loop",
            "running_loop",
            "target_loop",
            "main_loop",
        } or task_owner.endswith("_loop")
        if name in {"asyncio.create_task", "asyncio.ensure_future"} or raw_loop_task:
            if self.rel == "core/runtime/task_ownership.py" and name == "asyncio.create_task":
                self.generic_visit(node)
                return
            self.add(
                "high",
                "raw_async_task",
                node,
                f"Raw task creation {name} is blocked in production code.",
            )
        elif name == "time.sleep" and self._on_the_event_loop:
            self.add(
                "high",
                "blocking_sleep_in_async",
                node,
                "Blocking sleep in async function is prohibited.",
            )
        elif self._is_builtin_dynamic_code_call(node):
            self.add(
                "critical",
                "raw_dynamic_code",
                node,
                f"Dynamic code execution call {name} outside sandbox is prohibited.",
            )

        # Check direct subprocess/command calls, including subprocess callables
        # passed through wrappers such as asyncio.to_thread(subprocess.run, ...).
        if name in {
            "subprocess.run",
            "subprocess.Popen",
            "subprocess.check_output",
            "subprocess.check_call",
            "os.system",
            "os.popen",
            "os.posix_spawn",
            "os.posix_spawnp",
            "os.spawnl",
            "os.spawnle",
            "os.spawnlp",
            "os.spawnlpe",
            "os.spawnv",
            "os.spawnve",
            "os.spawnvp",
            "os.spawnvpe",
        } or name.endswith((
            ".create_subprocess_exec",
            ".create_subprocess_shell",
        )) or name in {
            "create_subprocess_exec",
            "create_subprocess_shell",
        } or self._has_subprocess_callable_arg(node):
            self.add(
                "high",
                "unapproved_direct_subprocess",
                node,
                f"Direct command execution via {name} is prohibited outside approved gateways.",
            )

        # Check direct network calls
        elif name in {
            "requests.get", "requests.post", "requests.put", "requests.delete", "requests.patch", "requests.request",
            "urllib.request.urlopen", "urllib.request.Request", "urllib.request.urlretrieve",
            "httpx.get", "httpx.post", "httpx.request", "httpx.Client", "httpx.AsyncClient"
        } or self._has_network_callable_arg(node):
            self.add(
                "high",
                "unapproved_direct_network",
                node,
                f"Direct network call via {name} is prohibited outside approved gateways.",
            )

        # Check direct file writes
        elif name == "open" or name.endswith(".open"):
            if self._is_in_memory_binary_write_call(node, name):
                self.generic_visit(node)
                return
            mode = "r"
            if len(node.args) > 1:
                if isinstance(node.args[1], ast.Constant) and isinstance(node.args[1].value, str):
                    mode = node.args[1].value
            for kw in node.keywords:
                if kw.arg == "mode" and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                    mode = kw.value.value

            if any(char in mode for char in "wax+"):
                self.add(
                    "high",
                    "unapproved_direct_file_write",
                    node,
                    "Direct write open() is prohibited outside approved gateways.",
                )
        elif (
            name.endswith(".write_text")
            or name.endswith(".write_bytes")
            or name in {"write_text", "write_bytes"}
        ):
            if self._is_file_gateway_write_call(node):
                self.generic_visit(node)
                return
            self.add(
                "high",
                "unapproved_direct_file_write",
                node,
                f"Direct file write via {name} is prohibited outside approved gateways.",
            )
        self.generic_visit(node)

    def _call_name(self, node: ast.Call) -> str:
        return self._canonical_call_name(self._call_name_from_func(node.func))

    def _canonical_call_name(self, name: str) -> str:
        if not name:
            return name
        parts = name.split(".")
        mapped_root = self.import_aliases.get(parts[0])
        if mapped_root is None:
            return name
        return ".".join([mapped_root, *parts[1:]])

    @staticmethod
    def _call_name_from_func(func: ast.AST) -> str:
        parts: list[str] = []
        while isinstance(func, ast.Attribute):
            parts.append(func.attr)
            func = func.value
        if isinstance(func, ast.Name):
            parts.append(func.id)
        return ".".join(reversed(parts))

    def _is_builtin_dynamic_code_call(self, node: ast.Call) -> bool:
        if isinstance(node.func, ast.Name):
            name = self._canonical_call_name(node.func.id)
            return name in {"compile", "eval", "exec", "builtins.compile", "builtins.eval", "builtins.exec"}
        if isinstance(node.func, ast.Attribute):
            name = self._canonical_call_name(self._call_name_from_func(node.func))
            return name in {"builtins.compile", "builtins.eval", "builtins.exec"}
        return False

    def _has_subprocess_callable_arg(self, node: ast.Call) -> bool:
        forbidden = {
            "subprocess.run",
            "subprocess.Popen",
            "subprocess.call",
            "subprocess.check_call",
            "subprocess.check_output",
            "asyncio.create_subprocess_exec",
            "asyncio.create_subprocess_shell",
            "os.posix_spawn",
            "os.posix_spawnp",
            "os.spawnl",
            "os.spawnle",
            "os.spawnlp",
            "os.spawnlpe",
            "os.spawnv",
            "os.spawnve",
            "os.spawnvp",
            "os.spawnvpe",
        }
        for arg in node.args:
            if self._canonical_call_name(AstLinter._call_name_from_func(arg)) in forbidden:
                return True
        return False

    def _has_network_callable_arg(self, node: ast.Call) -> bool:
        forbidden = {
            "requests.get",
            "requests.post",
            "requests.put",
            "requests.delete",
            "requests.patch",
            "requests.request",
            "httpx.get",
            "httpx.post",
            "httpx.request",
            "urllib.request.urlopen",
            "urllib.request.Request",
            "urllib.request.urlretrieve",
        }
        for arg in node.args:
            if self._canonical_call_name(AstLinter._call_name_from_func(arg)) in forbidden:
                return True
        return False

    @staticmethod
    def _attribute_receiver_name(node: ast.Call) -> str:
        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
            return node.func.value.id
        return ""

    def _is_file_gateway_write_call(self, node: ast.Call) -> bool:
        if not isinstance(node.func, ast.Attribute):
            return False
        receiver = node.func.value
        if isinstance(receiver, ast.Name):
            return receiver.id in self.file_gateway_vars
        if isinstance(receiver, ast.Call):
            return self._call_name(receiver) == "get_file_write_gateway"
        return False

    def _is_in_memory_binary_write_call(self, node: ast.Call, name: str) -> bool:
        if name == "wave.open" and node.args:
            target = node.args[0]
            return isinstance(target, ast.Name) and target.id in self.in_memory_binary_vars
        if name == "tarfile.open":
            for kw in node.keywords:
                if (
                    kw.arg == "fileobj"
                    and isinstance(kw.value, ast.Name)
                    and kw.value.id in self.in_memory_binary_vars
                ):
                    return True
        return False


def scan_file(path: Path) -> list[LintFinding]:
    rel = path.relative_to(ROOT).as_posix()
    findings: list[LintFinding] = []
    try:
        source = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        source = path.read_text(encoding="utf-8", errors="ignore")
    except OSError as exc:
        return [
            LintFinding(
                file=rel,
                line=1,
                kind="unreadable_production_file",
                severity="high",
                message=str(exc),
            )
        ]

    try:
        tree = ast.parse(source, filename=rel)
        if rel not in EXEMPT_FILES:
            findings.extend(hardcoded_local_path_findings(tree, rel))
        visitor = AstLinter(rel)
        visitor.visit(tree)
        findings.extend(visitor.findings)
    except SyntaxError as exc:
        findings.append(
            LintFinding("critical", "syntax_error", rel, exc.lineno or 0, str(exc))
        )

    return findings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=["production", "repo"], default="production")
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)

    # 1. Enforce per-file justification check for exemptions
    for fname, details in EXEMPT_FILES.items():
        if not isinstance(details, dict) or not details.get("justification") or not details.get("compensating_tests"):
            print(f"Error: Exempt file '{fname}' is missing a valid justification or compensating_tests entry.", file=sys.stderr)
            return 1

    findings: list[LintFinding] = []
    for path in iter_files(args.scope):
        findings.extend(scan_file(path))

    # Exclude warnings for non-production scope unless repo-scope is strictly specified
    high_or_critical = [f for f in findings if f.severity in {"high", "critical"}]

    report = {
        "generated_at": time.time(),
        "scope": args.scope,
        "passed": len(high_or_critical) == 0,
        "findings": [asdict(f) for f in findings],
        "findings_count": len(findings),
        "high_or_critical_count": len(high_or_critical),
        "audited_exemptions_count": len(EXEMPT_FILES)
    }

    output = json.dumps(report, indent=2, sort_keys=True)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output, encoding="utf-8")
    else:
        print(output)

    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
