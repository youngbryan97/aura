#!/usr/bin/env python3
"""Dependency-light enterprise quality gate for Aura.

The CLI remains a compatibility facade. Shared policy/data contracts, AST
semantics, and source-text semantics are isolated in dependency-light modules
so each rule family can be reviewed and tested independently.
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
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

try:
    from tools.aura_enterprise_ast_scan import (
        AstGate,
        body_without_docstring,
        decorator_name,
        dotted_call_name,
        handler_always_reraises,
        handler_answers_with_a_value,
        handler_records_a_degradation,
        is_abstract_function,
        is_deliberate_constructor_override,
        is_deliberate_refusal,
        is_not_implemented_only,
        is_serialization_guard,
        loop_can_end,
        raised_exception_name,
    )
    from tools.aura_enterprise_contracts import (
        ALLOW_BLOCKING_SLEEP_IN_ASYNC,
        ALLOW_DYNAMIC_CODE,
        DEFAULT_PRODUCTION_DIRS,
        DEFAULT_PRODUCTION_FILES,
        EXCLUDED_DIRS,
        FAILURE_KINDS,
        GATEWAY_OWNED_ROOTS,
        SUBPROCESS_GATEWAY_MODULE,
        TEXT_PATTERNS,
        TODO_MARKER_PATTERN,
        Finding,
        GateReport,
        _is_non_secret_literal,
        is_production,
        iter_py,
        rel_path,
        subprocess_must_use_gateway,
    )
    from tools.aura_enterprise_text_scan import (
        _PROSE_SENSITIVE_KINDS,
        FileTextContext,
        _call_name,
        _local_path_is_inert,
        _marker_is_not_a_claim,
        _marker_string_lines,
        _marker_text,
        _multiline_string_lines,
        _path_shaped_constants,
        _quoted_skip_lines,
        _skip_is_not_parked_debt,
        _unconditional_skip_lines,
        _vocabulary_string_lines,
        docstring_line_numbers,
        file_text_context,
    )
except ModuleNotFoundError as exc:
    if exc.name not in {
        "tools",
        "tools.aura_enterprise_ast_scan",
        "tools.aura_enterprise_contracts",
        "tools.aura_enterprise_text_scan",
    }:
        raise
    from aura_enterprise_ast_scan import (
        AstGate,
        body_without_docstring,
        decorator_name,
        dotted_call_name,
        handler_always_reraises,
        handler_answers_with_a_value,
        handler_records_a_degradation,
        is_abstract_function,
        is_deliberate_constructor_override,
        is_deliberate_refusal,
        is_not_implemented_only,
        is_serialization_guard,
        loop_can_end,
        raised_exception_name,
    )
    from aura_enterprise_contracts import (
        ALLOW_BLOCKING_SLEEP_IN_ASYNC,
        ALLOW_DYNAMIC_CODE,
        DEFAULT_PRODUCTION_DIRS,
        DEFAULT_PRODUCTION_FILES,
        EXCLUDED_DIRS,
        FAILURE_KINDS,
        GATEWAY_OWNED_ROOTS,
        SUBPROCESS_GATEWAY_MODULE,
        TEXT_PATTERNS,
        TODO_MARKER_PATTERN,
        Finding,
        GateReport,
        _is_non_secret_literal,
        is_production,
        iter_py,
        rel_path,
        subprocess_must_use_gateway,
    )
    from aura_enterprise_text_scan import (
        _PROSE_SENSITIVE_KINDS,
        FileTextContext,
        _call_name,
        _local_path_is_inert,
        _marker_is_not_a_claim,
        _marker_string_lines,
        _marker_text,
        _multiline_string_lines,
        _path_shaped_constants,
        _quoted_skip_lines,
        _skip_is_not_parked_debt,
        _unconditional_skip_lines,
        _vocabulary_string_lines,
        docstring_line_numbers,
        file_text_context,
    )

__all__ = [
    "ALLOW_BLOCKING_SLEEP_IN_ASYNC",
    "ALLOW_DYNAMIC_CODE",
    "DEFAULT_PRODUCTION_DIRS",
    "DEFAULT_PRODUCTION_FILES",
    "EXCLUDED_DIRS",
    "FAILURE_KINDS",
    "Finding",
    "GATEWAY_OWNED_ROOTS",
    "GateReport",
    "SUBPROCESS_GATEWAY_MODULE",
    "TEXT_PATTERNS",
    "TODO_MARKER_PATTERN",
    "_is_non_secret_literal",
    "is_production",
    "iter_py",
    "rel_path",
    "subprocess_must_use_gateway",
    "AstGate",
    "body_without_docstring",
    "decorator_name",
    "dotted_call_name",
    "handler_always_reraises",
    "handler_answers_with_a_value",
    "handler_records_a_degradation",
    "is_abstract_function",
    "is_deliberate_constructor_override",
    "is_deliberate_refusal",
    "is_not_implemented_only",
    "is_serialization_guard",
    "loop_can_end",
    "raised_exception_name",
    "FileTextContext",
    "_PROSE_SENSITIVE_KINDS",
    "_call_name",
    "_local_path_is_inert",
    "_marker_is_not_a_claim",
    "_marker_string_lines",
    "_marker_text",
    "_multiline_string_lines",
    "_path_shaped_constants",
    "_quoted_skip_lines",
    "_skip_is_not_parked_debt",
    "_unconditional_skip_lines",
    "_vocabulary_string_lines",
    "docstring_line_numbers",
    "file_text_context",
    "compare_to_baseline",
    "compile_gate",
    "load_baseline",
    "main",
    "make_baseline",
    "parse_args",
    "pytest_collect_gate",
    "run_gate",
    "scan_file",
    "write_text",
]


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

    # One parse for the whole file. The path rules, the comment sweep and
    # AstGate all used to parse it separately; on ~5,000 files that cost real
    # seconds on the clock the pre-commit gate runs against.
    try:
        tree: ast.AST | None = ast.parse(source, filename=rel)
        parse_error: SyntaxError | None = None
    except SyntaxError as exc:
        tree, parse_error = None, exc

    # Both are built on the first line that needs them, and never at all for
    # the large majority of files where no text rule matches. They are kept
    # apart because the prose sweep is cheap and often needed, while the path
    # analysis is a full expression walk that almost no file asks for.
    cached_prose: list[set[int]] = []
    cached_context: list[FileTextContext] = []

    def prose_lines() -> set[int]:
        if not cached_prose:
            cached_prose.append(docstring_line_numbers(tree))
        return cached_prose[0]

    def context() -> FileTextContext:
        if not cached_context:
            cached_context.append(file_text_context(tree))
        return cached_context[0]

    for line_no, line in enumerate(source.splitlines(), start=1):
        for kind, pattern in TEXT_PATTERNS.items():
            match = pattern.search(line)
            if match is None:
                continue
            if kind in _PROSE_SENSITIVE_KINDS and (
                line.lstrip().startswith("#") or line_no in prose_lines()
            ):
                continue
            if kind == "hardcoded_local_path" and _local_path_is_inert(
                match.group(0), line_no, context()
            ):
                continue
            if kind == "placeholder_stub_mock" and re.search(
                r"""(?:id|class)\s*=\s*["'][^"']*(?:placeholder|stub|mock)"""
                r"""|[.#][\w-]*(?:placeholder|stub|mock)[\w-]*\b"""
                r"""|["'][\w-]*-(?:placeholder|stub|mock)[\w-]*["']""",
                line,
                re.IGNORECASE,
            ):
                # A UI element NAMED "…-placeholder" is a name, not unfinished
                # work: the lane-status element is called lane-placeholder, and
                # its tests matched this rule fourteen times for saying so. The
                # rule is looking for incomplete product code, and an HTML id,
                # a CSS selector or a hyphenated token is neither. A bare
                # "returns a placeholder" is still a finding.
                continue
            if kind == "placeholder_stub_mock" and re.search(
                r"\b(?:from\s+unittest(?:\.mock)?\s+import|"
                r"mock\.(?:patch|AsyncMock|MagicMock)|MagicMock)\b",
                line,
            ):
                # Concrete test-double syntax is not incomplete product code.
                # Descriptive uses of stub/mock/placeholder remain findings.
                continue
            if kind == "placeholder_stub_mock" and re.search(
                r"\b(?:audit|detect|detected|prevent|contaminat|scanner|forbid|refus|quarantin)\w*\b",
                line,
                re.IGNORECASE,
            ):
                # Anti-mock tooling talks ABOUT mocks (audits, detectors,
                # contamination guards). Flagging the auditor for naming its
                # target is a false positive; passive mock USAGE still flags.
                continue
            if kind == "placeholder_stub_mock" and re.search(
                r"\b(?:not|never|no)\b[^.\n]{0,40}?\b(?:placeholder|stub|mock|dummy)s?\b",
                line,
                re.IGNORECASE,
            ):
                # Negated usage asserts the OPPOSITE of incomplete code
                # ("...a running app, not a stub"). Flagging the denial of a
                # stub as a stub is a false positive.
                continue
            if kind == "placeholder_stub_mock" and re.search(
                r"(?:\b(?:empty|blank)\s+placeholder\b|\bplaceholder\s*(?:=|attribute|value|text)\b|AXPlaceholder)",
                line,
                re.IGNORECASE,
            ):
                # DOM/AX "placeholder" is a UI attribute (input hint text),
                # not unfinished code.
                continue
            if kind == "pytest_skip_xfail" and _skip_is_not_parked_debt(
                line, line_no, context()
            ):
                continue
            if kind == "placeholder_stub_mock" and _marker_is_not_a_claim(
                line_no, rel, context()
            ):
                continue
            if kind == "potential_secret":
                if _is_non_secret_literal(line):
                    continue
                # A secret matters because something authenticates with it.
                # Bound to API_KEY, passed as token=, handed to authenticate()
                # — reported wherever it sits, tests included. A
                # credential-shaped literal in a parametrize list is not
                # that, and flagging it made the gate report the fixtures of
                # the test that proves the gate works. Same distinction the
                # /tmp/ rule already draws with disk_lines: judged by USE,
                # never by which file it sits in.
                used = context().credential_use_lines
                if used is not None and line_no not in used:
                    continue
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
    except (tokenize.TokenError, IndentationError, SyntaxError) as exc:
        line_no = getattr(exc, "lineno", 0) or (
            exc.args[1][0] if len(exc.args) > 1 and isinstance(exc.args[1], tuple) else 0
        )
        report.findings.append(Finding("critical", "syntax_error", rel, line_no, str(exc)))

    if parse_error is not None or tree is None:
        line_no = getattr(parse_error, "lineno", 0) or 0
        message = getattr(parse_error, "msg", "unparseable")
        report.findings.append(Finding("critical", "syntax_error", rel, line_no, message))
        return

    AstGate(rel, report, source_lines=source.splitlines()).visit(tree)


try:
    from tools.cpu_budget import cores_available
except ModuleNotFoundError:  # run as `tools/<name>.py`; only tools/ is on the path
    from cpu_budget import cores_available


def _scan_one(job: tuple[str, str]) -> tuple[int, list[Finding]]:
    """Scan one file into a report of its own, for a worker to hand back."""
    path_str, root_str = job
    one = GateReport(root=root_str, generated_at_unix=0.0)
    scan_file(Path(path_str), Path(root_str), one)
    return one.python_files, one.findings


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

    # One core read the whole tree and the gate took 70s against a 60s
    # timeout — so under any other load on the host it failed on the clock
    # rather than on a finding. Every file is scanned independently and
    # `iter_py` is ordered, so mapping them over a pool and merging in input
    # order gives the same report from the same tree.
    jobs = [(str(path), str(root)) for path in iter_py(root)]
    workers = cores_available()
    if workers > 1 and len(jobs) > 64:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            scanned = list(pool.map(_scan_one, jobs, chunksize=32))
    else:
        scanned = [_scan_one(job) for job in jobs]
    for counted, findings in scanned:
        report.python_files += counted
        report.findings.extend(findings)

    if include_pytest_collect:
        pytest_collect_gate(root, report, pytest_timeout)

    return report


def make_baseline(report: GateReport) -> dict:
    inventory_findings = [
        finding for finding in report.findings if finding.kind != "baseline_regression"
    ]
    counts: dict[str, int] = {}
    for finding in inventory_findings:
        counts[finding.kind] = counts.get(finding.kind, 0) + 1
    high_or_critical_count = sum(
        1 for finding in inventory_findings if finding.severity in {"high", "critical"}
    )
    return {
        "description": "Aura enterprise gate debt baseline. Reduce counts over time; do not raise them.",
        "generated_at_unix": report.generated_at_unix,
        "python_files": report.python_files,
        "max_counts": dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))),
        "max_high_or_critical_count": high_or_critical_count,
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

    current_high_critical = sum(
        1
        for finding in report.findings
        if finding.kind != "baseline_regression" and finding.severity in {"high", "critical"}
    )
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
    regressions = [
        finding for finding in report.findings if finding.kind == "baseline_regression"
    ]

    def explain(reason: str, shown: list[Finding]) -> None:
        """Say why the gate failed.

        With --out the report goes to a file and this used to print nothing at
        all, so `aura_enterprise_gate.py ... --out x.json` exited 1 in silence
        and test_enterprise_static_contracts asserted with an empty message.
        A gate whose failure carries no reason does not get acted on.
        """
        print(f"enterprise gate FAILED: {reason}", file=sys.stderr)
        for finding in shown:
            location = (
                f"{finding.file}:{finding.line}" if finding.file not in {"", "."} else "repo"
            )
            print(f"  [{finding.severity}] {location} {finding.detail}", file=sys.stderr)
        if args.out:
            print(f"  full report: {args.out}", file=sys.stderr)

    if args.fail_on_regression and regressions:
        explain(f"{len(regressions)} count(s) above the debt baseline", regressions)
        return 1
    if args.strict and (failed_gate or report.high_or_critical_count() > 0):
        explain(
            f"strict mode: {report.high_or_critical_count()} high/critical finding(s)",
            [f for f in report.findings if f.kind in FAILURE_KINDS][:40],
        )
        return 1
    if failed_gate:
        explain(
            "blocking finding kind(s) present",
            [f for f in report.findings if f.kind in FAILURE_KINDS][:40],
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
