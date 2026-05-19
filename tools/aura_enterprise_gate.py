#!/usr/bin/env python3
"""Dependency-light enterprise quality gate for Aura.

The gate is deliberately stdlib-only so it can run before the full development
environment is installed. It catches obvious enterprise-runtime regressions and
can compare the current inventory against a checked-in baseline while the repo
continues retiring older debt.
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import io
import json
import os
import py_compile
import re
import subprocess
import sys
import tempfile
import time
import tokenize
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path

EXCLUDED_DIRS = {
    ".git",
    ".agents",
    ".claude",
    ".aura_architect",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    ".venv_aura",
    "__pycache__",
    "archive",
    "artifacts",
    "build",
    "data",
    "dist",
    "htmlcov",
    "logs",
    "node_modules",
    "scratch",
    "test_vdb",
    "venv",
}

DEFAULT_PRODUCTION_DIRS = {
    "core",
    "executors",
    "infrastructure",
    "interface",
    "llm",
    "security",
    "senses",
    "skills",
}
DEFAULT_PRODUCTION_FILES = {"aura_main.py"}

ALLOW_DYNAMIC_CODE = {
    "core/agency/repl_daemon.py",
    "core/brain/react_loop.py",
    "core/capability_engine.py",
    "core/kernel/shadow_kernel.py",
    "core/runtime/self_repair_ladder.py",
    "core/sandbox/bash_daemon.py",
    "core/sandbox/runner.py",
    "core/self_modification/mutation_safety.py",
    "core/self_modification/shadow_runtime.py",
    "security/code_sandbox.py",
    "security/sandbox.py",
}

ALLOW_SUBPROCESS = {
    "aura_main.py",
    "core/agency/agency_orchestrator.py",
    "core/brain/llm/mlx_client.py",
    "core/runtime/consequential_primitives.py",
    "core/sandbox/bash_daemon.py",
    "core/security/integrity_guardian.py",
    "core/skills/sovereign_terminal.py",
    "security/sandbox.py",
    "skills/shell.py",
    "tools/aura_enterprise_gate.py",
}

ALLOW_BLOCKING_SLEEP_IN_ASYNC = {
    # This chaos fault deliberately stalls the loop to verify lag detection
    # and recovery alarms. It is not production request handling.
    "tools/chaos/injector.py",
}

SELF_DESCRIPTIVE_PATTERN_FILES = {
    "tools/aura_enterprise_gate.py",
}

_TMP_PATH_PREFIX = "/" + "tmp" + "/"
_USERS_PATH_PREFIX = "/" + "Users" + "/"
_HOME_PATH_PREFIX = "/" + "home" + "/"
_WINDOWS_USERS_PREFIX = "C:" + "\\\\" + "Users" + "\\\\"

TEXT_PATTERNS = {
    "hardcoded_local_path": re.compile(
        rf"({re.escape(_USERS_PATH_PREFIX)}|"
        rf"{re.escape(_HOME_PATH_PREFIX)}[^/\s]+/|"
        rf"{re.escape(_WINDOWS_USERS_PREFIX)}|"
        rf"{re.escape(_TMP_PATH_PREFIX)}[^\s]+)"
    ),
    "placeholder_stub_mock": re.compile(
        r"\b(placeholder|stub|mock|dummy|not implemented|notimplemented)\b",
        re.IGNORECASE,
    ),
    "potential_secret": re.compile(
        r"(sk-[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9_]{20,}|xox[baprs]-[A-Za-z0-9-]{10,})"
    ),
    "pytest_skip_xfail": re.compile(r"pytest\.mark\.skip|pytest\.skip|xfail", re.IGNORECASE),
}
TODO_MARKER_PATTERN = re.compile(
    r"^(TODO|FIXME|XXX|HACK)\b(?:\([^)]*\))?\s*(?::|-|\s|$)",
    re.IGNORECASE,
)

FAILURE_KINDS = {
    "baseline_regression",
    "compile_failure",
    "pytest_collect_failure",
    "pytest_collect_timeout",
    "syntax_error",
}


@dataclass(frozen=True)
class Finding:
    severity: str
    kind: str
    file: str
    line: int = 0
    detail: str = ""


@dataclass
class GateReport:
    root: str
    generated_at_unix: float
    python_files: int = 0
    compile_ok: bool | None = None
    pytest_collect_ok: bool | None = None
    pytest_collect_seconds: float | None = None
    pytest_collect_output_tail: str = ""
    findings: list[Finding] = field(default_factory=list)

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for finding in self.findings:
            out[finding.kind] = out.get(finding.kind, 0) + 1
        return dict(sorted(out.items(), key=lambda kv: (-kv[1], kv[0])))

    def severity_counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for finding in self.findings:
            out[finding.severity] = out.get(finding.severity, 0) + 1
        return dict(sorted(out.items(), key=lambda kv: kv[0]))

    def high_or_critical_count(self) -> int:
        return sum(1 for finding in self.findings if finding.severity in {"high", "critical"})

    def to_json_dict(self) -> dict:
        payload = asdict(self)
        payload["findings"] = [asdict(finding) for finding in self.findings]
        payload["counts"] = self.counts()
        payload["severity_counts"] = self.severity_counts()
        payload["high_or_critical_count"] = self.high_or_critical_count()
        return payload

    def to_json(self) -> str:
        return json.dumps(self.to_json_dict(), indent=2, sort_keys=True)


def rel_path(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def iter_py(root: Path) -> Iterable[Path]:
    for path in root.rglob("*.py"):
        rel_parts = path.relative_to(root).parts
        if any(part in EXCLUDED_DIRS for part in rel_parts):
            continue
        yield path


def is_production(rel: str) -> bool:
    first = rel.split("/", 1)[0]
    return first in DEFAULT_PRODUCTION_DIRS or rel in DEFAULT_PRODUCTION_FILES


def dotted_call_name(node: ast.Call) -> str:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    parts: list[str] = []
    while isinstance(func, ast.Attribute):
        parts.append(func.attr)
        func = func.value
    if isinstance(func, ast.Name):
        parts.append(func.id)
        return ".".join(reversed(parts))
    return ""


def body_without_docstring(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[ast.stmt]:
    body = list(node.body)
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(getattr(body[0], "value", None), ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        return body[1:]
    return body


def decorator_name(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parts = [node.attr]
        value = node.value
        while isinstance(value, ast.Attribute):
            parts.append(value.attr)
            value = value.value
        if isinstance(value, ast.Name):
            parts.append(value.id)
        return ".".join(reversed(parts))
    if isinstance(node, ast.Call):
        return decorator_name(node.func)
    return ""


def is_abstract_function(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    names = {decorator_name(item) for item in node.decorator_list}
    return bool(
        names
        & {"abstractmethod", "abc.abstractmethod", "abstractclassmethod", "abstractstaticmethod"}
    )


def is_not_implemented_only(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    body = body_without_docstring(node)
    if len(body) != 1 or not isinstance(body[0], ast.Raise):
        return False
    exc = body[0].exc
    if isinstance(exc, ast.Call):
        exc = exc.func
    return isinstance(exc, ast.Name) and exc.id == "NotImplementedError"


class AstGate(ast.NodeVisitor):
    def __init__(self, rel: str, report: GateReport):
        self.rel = rel
        self.report = report
        self.async_depth = 0

    def add(self, severity: str, kind: str, node: ast.AST, detail: str = "") -> None:
        self.report.findings.append(
            Finding(severity, kind, self.rel, getattr(node, "lineno", 0), detail)
        )

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if any(alias.name == "*" for alias in node.names):
            self.add("medium", "wildcard_import", node, node.module or "")
        self.generic_visit(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        broad = node.type is None or (
            isinstance(node.type, ast.Name) and node.type.id in {"BaseException", "Exception"}
        )
        if broad:
            severity = "high" if is_production(self.rel) else "medium"
            if node.type is None:
                self.add(severity, "bare_except", node)
            elif any(isinstance(item, ast.Pass) for item in node.body) or all(
                isinstance(item, (ast.Break, ast.Continue, ast.Pass, ast.Return))
                for item in node.body
            ):
                self.add(severity, "swallowed_broad_exception", node)
            else:
                self.add(
                    "medium" if is_production(self.rel) else "low",
                    "broad_exception_review",
                    node,
                )
        self.generic_visit(node)

    def visit_While(self, node: ast.While) -> None:
        if isinstance(node.test, ast.Constant) and node.test.value is True:
            self.add("medium" if is_production(self.rel) else "low", "unbounded_loop_review", node)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        body = body_without_docstring(node)
        if len(body) == 1 and isinstance(body[0], ast.Pass) and not is_abstract_function(node):
            self.add(
                "high" if is_production(self.rel) else "medium",
                "pass_only_function",
                node,
                node.name,
            )
        if (
            len(body) == 1
            and isinstance(body[0], ast.Raise)
            and not (is_abstract_function(node) and is_not_implemented_only(node))
        ):
            self.add(
                "high" if is_production(self.rel) else "medium",
                "raise_only_function",
                node,
                node.name,
            )
        self.async_depth += 1
        self.generic_visit(node)
        self.async_depth -= 1

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        body = body_without_docstring(node)
        if len(body) == 1 and isinstance(body[0], ast.Pass) and not is_abstract_function(node):
            self.add(
                "high" if is_production(self.rel) else "medium",
                "pass_only_function",
                node,
                node.name,
            )
        if (
            len(body) == 1
            and isinstance(body[0], ast.Raise)
            and not (is_abstract_function(node) and is_not_implemented_only(node))
        ):
            self.add(
                "high" if is_production(self.rel) else "medium",
                "raise_only_function",
                node,
                node.name,
            )
        previous_async_depth = self.async_depth
        self.async_depth = 0
        try:
            self.generic_visit(node)
        finally:
            self.async_depth = previous_async_depth

    def visit_Call(self, node: ast.Call) -> None:
        name = dotted_call_name(node)
        if name in {"compile", "eval", "exec"} and self.rel not in ALLOW_DYNAMIC_CODE:
            self.add(
                "critical" if is_production(self.rel) else "medium",
                "dynamic_code_execution",
                node,
                name,
            )
        if name in {
            "os.system",
            "subprocess.Popen",
            "subprocess.call",
            "subprocess.check_call",
            "subprocess.check_output",
            "subprocess.run",
        }:
            shell_true = any(
                keyword.arg == "shell"
                and isinstance(keyword.value, ast.Constant)
                and keyword.value.value is True
                for keyword in node.keywords
            )
            if shell_true:
                self.add("critical", "subprocess_shell_true", node, name)
            elif self.rel not in ALLOW_SUBPROCESS:
                self.add(
                    "high" if is_production(self.rel) else "medium",
                    "subprocess_usage_review",
                    node,
                    name,
                )
        if name in {"dill.load", "dill.loads", "pickle.load", "pickle.loads"}:
            self.add(
                "critical" if is_production(self.rel) else "high",
                "unsafe_deserialization",
                node,
                name,
            )
        if (
            name == "time.sleep"
            and self.async_depth
            and self.rel not in ALLOW_BLOCKING_SLEEP_IN_ASYNC
        ):
            self.add("high", "blocking_sleep_in_async", node)
        self.generic_visit(node)


def compile_gate(root: Path, report: GateReport, timeout_s: int) -> None:
    started = time.monotonic()
    failures = 0
    with tempfile.TemporaryDirectory(prefix="aura_compile_gate_") as tmp_dir:
        tmp_root = Path(tmp_dir)
        for index, path in enumerate(iter_py(root)):
            if time.monotonic() - started > timeout_s:
                report.findings.append(
                    Finding("critical", "compile_failure", ".", 0, f"Timed out after {timeout_s}s")
                )
                failures += 1
                break
            rel = rel_path(path, root)
            try:
                py_compile.compile(str(path), cfile=str(tmp_root / f"{index}.pyc"), doraise=True)
            except py_compile.PyCompileError as exc:
                failures += 1
                report.findings.append(
                    Finding("critical", "compile_failure", rel, 0, str(exc)[-4000:])
                )
    report.compile_ok = failures == 0


def pytest_collect_gate(root: Path, report: GateReport, timeout_s: int) -> None:
    start = time.time()
    env = os.environ.copy()
    env.setdefault("AURA_TEST_MODE", "1")
    env.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    cmd = [sys.executable, "-m", "pytest"]
    if importlib.util.find_spec("pytest_asyncio") is not None:
        cmd.extend(["-p", "pytest_asyncio.plugin"])
    cmd.extend(["--collect-only", "-q"])
    try:
        proc = subprocess.run(
            cmd,
            cwd=root,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout_s,
        )
        report.pytest_collect_ok = proc.returncode == 0
        report.pytest_collect_output_tail = proc.stdout[-4000:]
        if proc.returncode != 0:
            report.findings.append(
                Finding("critical", "pytest_collect_failure", ".", 0, proc.stdout[-4000:])
            )
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout or ""
        if isinstance(output, bytes):
            output = output.decode(errors="replace")
        report.pytest_collect_ok = False
        report.pytest_collect_output_tail = output[-4000:]
        report.findings.append(
            Finding("critical", "pytest_collect_timeout", ".", 0, f"Timed out after {timeout_s}s")
        )
    finally:
        report.pytest_collect_seconds = round(time.time() - start, 3)


def scan_file(path: Path, root: Path, report: GateReport) -> None:
    rel = rel_path(path, root)
    try:
        source = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        source = path.read_text(encoding="utf-8", errors="replace")

    report.python_files += 1

    for line_no, line in enumerate(source.splitlines(), start=1):
        for kind, pattern in TEXT_PATTERNS.items():
            if not pattern.search(line):
                continue
            if rel in SELF_DESCRIPTIVE_PATTERN_FILES and kind in {
                "placeholder_stub_mock",
                "pytest_skip_xfail",
            }:
                continue
            if kind == "potential_secret":
                severity = "critical"
            elif kind in {"hardcoded_local_path", "placeholder_stub_mock"} and is_production(rel):
                severity = "high"
            elif kind == "pytest_skip_xfail":
                severity = "medium"
            else:
                severity = "low"
            report.findings.append(Finding(severity, kind, rel, line_no, line.strip()[:240]))

    try:
        tokens = tokenize.generate_tokens(io.StringIO(source).readline)
        for token in tokens:
            if token.type != tokenize.COMMENT:
                continue
            comment = token.string.lstrip("#").strip()
            if TODO_MARKER_PATTERN.search(comment):
                report.findings.append(
                    Finding(
                        "low",
                        "todo_fixme_hack",
                        rel,
                        token.start[0],
                        token.string.strip()[:240],
                    )
                )
    except tokenize.TokenError as exc:
        report.findings.append(Finding("critical", "syntax_error", rel, exc.args[1][0], str(exc)))

    try:
        tree = ast.parse(source, filename=rel)
    except SyntaxError as exc:
        report.findings.append(Finding("critical", "syntax_error", rel, exc.lineno or 0, exc.msg))
        return

    AstGate(rel, report).visit(tree)


def run_gate(
    root: Path,
    *,
    include_compile: bool,
    include_pytest_collect: bool,
    compile_timeout: int,
    pytest_timeout: int,
) -> GateReport:
    report = GateReport(root=str(root), generated_at_unix=time.time())

    if include_compile:
        compile_gate(root, report, compile_timeout)

    for path in iter_py(root):
        scan_file(path, root, report)

    if include_pytest_collect:
        pytest_collect_gate(root, report, pytest_timeout)

    return report


def make_baseline(report: GateReport) -> dict:
    payload = report.to_json_dict()
    return {
        "description": "Aura enterprise gate debt baseline. Reduce counts over time; do not raise them.",
        "generated_at_unix": payload["generated_at_unix"],
        "python_files": payload["python_files"],
        "max_counts": payload["counts"],
        "max_high_or_critical_count": payload["high_or_critical_count"],
    }


def load_baseline(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def compare_to_baseline(report: GateReport, baseline: dict) -> None:
    current_counts = report.counts()
    max_counts = baseline.get("max_counts", {})
    for kind, count in sorted(current_counts.items()):
        allowed = int(max_counts.get(kind, 0))
        if count > allowed:
            report.findings.append(
                Finding(
                    "critical",
                    "baseline_regression",
                    ".",
                    0,
                    f"{kind} count {count} exceeds baseline {allowed}",
                )
            )

    current_high_critical = report.high_or_critical_count()
    max_high_critical = int(baseline.get("max_high_or_critical_count", 0))
    if current_high_critical > max_high_critical:
        report.findings.append(
            Finding(
                "critical",
                "baseline_regression",
                ".",
                0,
                "high_or_critical_count "
                f"{current_high_critical} exceeds baseline {max_high_critical}",
            )
        )


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".", help="Repository root to scan.")
    parser.add_argument("--out", default="", help="Optional JSON report output path.")
    parser.add_argument("--baseline", default="", help="Optional debt baseline JSON.")
    parser.add_argument(
        "--write-baseline", default="", help="Write a new baseline JSON from this run."
    )
    parser.add_argument("--strict", action="store_true", help="Fail on any high/critical finding.")
    parser.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="Fail when current findings exceed --baseline.",
    )
    parser.add_argument("--skip-compile", action="store_true", help="Skip compileall gate.")
    parser.add_argument(
        "--skip-pytest-collect",
        action="store_true",
        help="Skip pytest --collect-only gate.",
    )
    parser.add_argument("--compile-timeout", type=int, default=120)
    parser.add_argument("--pytest-timeout", type=int, default=90)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    root = Path(args.root).resolve()

    report = run_gate(
        root,
        include_compile=not args.skip_compile,
        include_pytest_collect=not args.skip_pytest_collect,
        compile_timeout=args.compile_timeout,
        pytest_timeout=args.pytest_timeout,
    )

    if args.baseline:
        compare_to_baseline(report, load_baseline(Path(args.baseline)))

    if args.write_baseline:
        write_text(
            Path(args.write_baseline), json.dumps(make_baseline(report), indent=2, sort_keys=True)
        )

    output = report.to_json()
    if args.out:
        write_text(Path(args.out), output)
    else:
        print(output)

    failed_gate = any(finding.kind in FAILURE_KINDS for finding in report.findings)
    if args.fail_on_regression and any(
        finding.kind == "baseline_regression" for finding in report.findings
    ):
        return 1
    if args.strict and (failed_gate or report.high_or_critical_count() > 0):
        return 1
    if failed_gate:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
