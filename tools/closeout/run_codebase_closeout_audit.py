#!/usr/bin/env python3
"""Run Aura's enterprise closeout source audit checkpoint.

The closeout prompt asks for every line to be inspected and for checkpoints to
be explicit. This runner makes that operational: it enumerates every tracked
file, hashes every text line into a ledger, records binary files separately,
and writes a machine-readable checkpoint bundle. It is not a substitute for a
human semantic review of every line or for the 24-72h live runtime soaks; the
verdict keeps those claims separate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from subprocess import TimeoutExpired
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.runtime.subprocess_gateway import get_subprocess_gateway  # noqa: E402

DEFAULT_OUT = ROOT / "artifacts" / "current" / "closeout_audit"
_SUBPROCESS_GATEWAY = get_subprocess_gateway()
_GATE_RECOVERABLE_ERRORS = (
    ImportError,
    AttributeError,
    RuntimeError,
    TypeError,
    ValueError,
    OSError,
    TimeoutError,
)

TEXT_EXTENSIONS = {
    ".cfg",
    ".css",
    ".csv",
    ".env",
    ".gitignore",
    ".html",
    ".ini",
    ".js",
    ".json",
    ".jsonl",
    ".jsx",
    ".lock",
    ".md",
    ".mjs",
    ".py",
    ".rs",
    ".sh",
    ".sql",
    ".toml",
    ".ts",
    ".tsx",
    ".txt",
    ".yaml",
    ".yml",
}
CODE_EXTENSIONS = {
    ".css",
    ".html",
    ".js",
    ".jsx",
    ".mjs",
    ".py",
    ".rs",
    ".sh",
    ".sql",
    ".ts",
    ".tsx",
}
TODO_RE = re.compile(r"\b(TODO|FIXME|XXX|HACK)\b", re.IGNORECASE)
SCAFFOLD_RE = re.compile(r"\b(stub|placeholder|dummy|not implemented|notimplemented)\b", re.IGNORECASE)
BROAD_EXCEPT_RE = re.compile(r"^\s*except\s+(Exception|BaseException)\b")


@dataclass(frozen=True)
class FileAudit:
    path: str
    size_bytes: int
    sha256: str
    text: bool
    code: bool
    line_count: int
    blank_lines: int
    comment_lines: int
    todo_markers: int
    scaffold_markers: int
    broad_exception_handlers: int


@dataclass(frozen=True)
class GateResult:
    name: str
    passed: bool
    command: list[str] | None = None
    detail: str = ""
    duration_s: float = 0.0


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, default=str) + "\n")


def _run_git(args: list[str]):
    return _SUBPROCESS_GATEWAY.run(
        ["git", *args],
        cwd=ROOT,
        timeout=60,
        read_only=True,
        source="closeout_audit_git",
        accelerator_capability="none",
    )


def tracked_files() -> list[Path]:
    proc = _run_git(["ls-files", "-z"])
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "git ls-files failed")
    files: list[Path] = []
    for raw in proc.stdout.split("\0"):
        if not raw:
            continue
        path = ROOT / raw
        if path.is_file():
            files.append(path)
    return sorted(files)


def git_status() -> dict[str, Any]:
    branch = _run_git(["branch", "--show-current"]).stdout.strip()
    head = _run_git(["rev-parse", "HEAD"]).stdout.strip()
    porcelain = _run_git(["status", "--porcelain"]).stdout.splitlines()
    return {
        "branch": branch,
        "head": head,
        "dirty": bool(porcelain),
        "porcelain": porcelain,
    }


def _is_probably_text(path: Path, data: bytes) -> bool:
    if b"\0" in data[:8192]:
        return False
    if path.suffix.lower() in TEXT_EXTENSIONS or path.name in {".gitignore", "Makefile"}:
        return True
    try:
        data[:8192].decode("utf-8")
        return True
    except UnicodeDecodeError:
        return False


def _line_kind(line: str) -> str:
    stripped = line.strip()
    if not stripped:
        return "blank"
    if stripped.startswith(("#", "//", "/*", "*", "--")):
        return "comment"
    return "content"


def audit_file(path: Path, *, line_ledger_path: Path | None = None) -> FileAudit:
    try:
        rel = path.relative_to(ROOT).as_posix()
    except ValueError:
        rel = path.as_posix()
    data = path.read_bytes()
    is_text = _is_probably_text(path, data)
    digest = _sha256_bytes(data)
    code = path.suffix.lower() in CODE_EXTENSIONS or path.name == "Makefile"
    if not is_text:
        return FileAudit(
            path=rel,
            size_bytes=len(data),
            sha256=digest,
            text=False,
            code=code,
            line_count=0,
            blank_lines=0,
            comment_lines=0,
            todo_markers=0,
            scaffold_markers=0,
            broad_exception_handlers=0,
        )

    text = data.decode("utf-8", errors="replace")
    lines = text.splitlines()
    blank = 0
    comments = 0
    todo = 0
    scaffold = 0
    broad_except = 0
    for index, line in enumerate(lines, start=1):
        kind = _line_kind(line)
        blank += int(kind == "blank")
        comments += int(kind == "comment")
        todo += len(TODO_RE.findall(line))
        scaffold += len(SCAFFOLD_RE.findall(line))
        broad_except += int(bool(BROAD_EXCEPT_RE.search(line)))
        if line_ledger_path is not None:
            _append_jsonl(
                line_ledger_path,
                {
                    "file": rel,
                    "line": index,
                    "kind": kind,
                    "sha256": _sha256_bytes(line.encode("utf-8", errors="replace")),
                },
            )

    return FileAudit(
        path=rel,
        size_bytes=len(data),
        sha256=digest,
        text=True,
        code=code,
        line_count=len(lines),
        blank_lines=blank,
        comment_lines=comments,
        todo_markers=todo,
        scaffold_markers=scaffold,
        broad_exception_handlers=broad_except,
    )


def run_command_gate(name: str, command: list[str], *, timeout_s: int = 240) -> GateResult:
    started = time.time()
    try:
        proc = _SUBPROCESS_GATEWAY.run(
            command,
            cwd=ROOT,
            timeout=timeout_s,
            read_only=True,
            source=f"closeout_audit_gate:{name}",
            accelerator_capability="auto",
        )
        detail = (proc.stdout + "\n" + proc.stderr).strip()
        return GateResult(
            name=name,
            passed=proc.returncode == 0,
            command=command,
            detail=detail[-4000:],
            duration_s=round(time.time() - started, 4),
        )
    except TimeoutExpired as exc:
        return GateResult(
            name=name,
            passed=False,
            command=command,
            detail=f"timeout_after_{timeout_s}s:{exc}",
            duration_s=round(time.time() - started, 4),
        )


def _make_gate_command(target: str) -> list[str]:
    """Bind Make gates to the interpreter that admitted the audit process."""
    return ["make", target, f"PYTHON={sys.executable}"]


def production_readiness_gate() -> GateResult:
    started = time.time()
    try:
        from tools.aura_production_readiness_gate import run_checks

        checks = run_checks()
        failed = [check.name for check in checks if not check.passed]
        return GateResult(
            name="production_readiness_contract",
            passed=not failed,
            detail=json.dumps(
                {
                    "check_count": len(checks),
                    "failed": failed,
                },
                sort_keys=True,
            ),
            duration_s=round(time.time() - started, 4),
        )
    except _GATE_RECOVERABLE_ERRORS as exc:
        return GateResult(
            name="production_readiness_contract",
            passed=False,
            detail=f"{type(exc).__name__}: {exc}",
            duration_s=round(time.time() - started, 4),
        )


def architecture_map_gate(out_dir: Path) -> GateResult:
    started = time.time()
    try:
        from tools.arch_map import ARCH_MAP_SCHEMA, build_architecture_report

        report = build_architecture_report()
        _write_json(out_dir / "ARCHITECTURE_MAP.json", report)
        surfaces = report.get("operational_surfaces", {})
        required = {
            "will_decision",
            "memory_write",
            "state_mutation",
            "tool_execution",
            "patching",
            "llm_call",
            "external_io",
        }
        missing = sorted(required - set(surfaces))
        passed = report.get("schema") == ARCH_MAP_SCHEMA and not missing
        return GateResult(
            name="architecture_dependency_map",
            passed=passed,
            detail=json.dumps(
                {
                    "subsystems": report.get("totals", {}).get("subsystems"),
                    "dependency_edges": len(report.get("dependency_edges", [])),
                    "missing_surfaces": missing,
                },
                sort_keys=True,
            ),
            duration_s=round(time.time() - started, 4),
        )
    except _GATE_RECOVERABLE_ERRORS as exc:
        return GateResult(
            name="architecture_dependency_map",
            passed=False,
            detail=f"{type(exc).__name__}: {exc}",
            duration_s=round(time.time() - started, 4),
        )


def model_load_ownership_gate() -> GateResult:
    started = time.time()
    try:
        from tools.closeout.audit_model_load_ownership import run_audit

        report = run_audit()
        return GateResult(
            name="model_load_ownership",
            passed=bool(report.get("passed", False)),
            detail=json.dumps(
                {
                    "owned_paths": report.get("owned_paths"),
                    "load_references": report.get("load_references"),
                    "findings": report.get("findings"),
                },
                sort_keys=True,
            ),
            duration_s=round(time.time() - started, 4),
        )
    except _GATE_RECOVERABLE_ERRORS as exc:
        return GateResult(
            name="model_load_ownership",
            passed=False,
            detail=f"{type(exc).__name__}: {exc}",
            duration_s=round(time.time() - started, 4),
        )


def resource_observation_ownership_gate() -> GateResult:
    started = time.time()
    try:
        from tools.closeout.audit_resource_observation_ownership import run_audit

        report = run_audit()
        return GateResult(
            name="resource_observation_ownership",
            passed=bool(report.get("passed", False)),
            detail=json.dumps(
                {
                    "scanned_python_files": report.get("scanned_python_files"),
                    "finding_count": report.get("finding_count"),
                    "parse_errors": report.get("parse_errors"),
                },
                sort_keys=True,
            ),
            duration_s=round(time.time() - started, 4),
        )
    except _GATE_RECOVERABLE_ERRORS as exc:
        return GateResult(
            name="resource_observation_ownership",
            passed=False,
            detail=f"{type(exc).__name__}: {exc}",
            duration_s=round(time.time() - started, 4),
        )


def write_manifest(out_dir: Path) -> dict[str, Any]:
    files: dict[str, dict[str, Any]] = {}
    for path in sorted(out_dir.rglob("*")):
        if not path.is_file() or path.name == "MANIFEST.json":
            continue
        rel = path.relative_to(out_dir).as_posix()
        files[rel] = {"sha256": _sha256_file(path), "size_bytes": path.stat().st_size}
    manifest = {
        "schema": "aura.closeout.manifest.v1",
        "generated_at_unix": time.time(),
        "artifact_dir": str(out_dir),
        "files": files,
    }
    _write_json(out_dir / "MANIFEST.json", manifest)
    return manifest


def _sample_files(files: list[Path], limit: int) -> list[Path]:
    """Take an even stride across the tree rather than the first N.

    Truncating a sorted list makes ``--max-files`` an "audit the top of the
    alphabet" flag: the first twelve tracked paths are ``.aura/memfs`` and
    ``.github``, none of them code, so a twelve-file run reported
    ``code_line_count: 0`` and a FAIL verdict on a healthy tree. A sample of a
    codebase has to contain some.

    A stride keeps the sample deterministic — the same tree gives the same
    sample — while spreading it across every directory the sort passes through.
    """
    limit = max(0, limit)
    if limit == 0 or not files:
        return []
    if limit >= len(files):
        return list(files)
    stride = len(files) / limit
    return [files[int(i * stride)] for i in range(limit)]


def build_closeout_audit(
    *,
    out_dir: Path,
    allow_dirty: bool,
    run_gates: bool,
    max_files: int | None = None,
) -> dict[str, Any]:
    out_dir = out_dir.resolve()
    if out_dir.exists():
        for child in sorted(out_dir.rglob("*"), reverse=True):
            if child.is_file():
                child.unlink()
            elif child.is_dir():
                child.rmdir()
    out_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    files = tracked_files()
    if max_files is not None:
        files = _sample_files(files, int(max_files))

    line_ledger = out_dir / "SOURCE_LINE_LEDGER.jsonl"
    file_ledger = out_dir / "SOURCE_FILE_LEDGER.jsonl"
    line_ledger.write_text("", encoding="utf-8")
    file_ledger.write_text("", encoding="utf-8")

    file_audits: list[FileAudit] = []
    for path in files:
        audit = audit_file(path, line_ledger_path=line_ledger)
        file_audits.append(audit)
        _append_jsonl(file_ledger, asdict(audit))

    text_files = [item for item in file_audits if item.text]
    code_files = [item for item in file_audits if item.code]
    from tools.closeout.semantic_review_ledger import DEFAULT_LEDGER, summarize_semantic_reviews

    semantic_review = summarize_semantic_reviews(ledger_path=DEFAULT_LEDGER, tracked_paths=files)
    _write_json(out_dir / "SEMANTIC_REVIEW_STATUS.json", semantic_review)
    findings = {
        "schema": "aura.closeout.findings.v1",
        "todo_marker_total": sum(item.todo_markers for item in file_audits),
        "scaffold_marker_total": sum(item.scaffold_markers for item in file_audits),
        "broad_exception_handler_total": sum(item.broad_exception_handlers for item in file_audits),
        "top_todo_files": [
            asdict(item)
            for item in sorted(file_audits, key=lambda item: item.todo_markers, reverse=True)
            if item.todo_markers
        ][:50],
        "top_scaffold_files": [
            asdict(item)
            for item in sorted(file_audits, key=lambda item: item.scaffold_markers, reverse=True)
            if item.scaffold_markers
        ][:50],
        "top_broad_exception_files": [
            asdict(item)
            for item in sorted(file_audits, key=lambda item: item.broad_exception_handlers, reverse=True)
            if item.broad_exception_handlers
        ][:50],
        "finding_boundary": (
            "Keyword findings are triage candidates, not automatic defects; "
            "production lints and semantic review decide remediation."
        ),
    }
    _write_json(out_dir / "FINDINGS.json", findings)

    gates: list[GateResult] = [
        GateResult(
            name="git_worktree_clean_or_explicitly_allowed",
            passed=allow_dirty or not git_status()["dirty"],
            detail="dirty allowed for in-progress checkpoint" if allow_dirty else "worktree must be clean",
        ),
        production_readiness_gate(),
        architecture_map_gate(out_dir),
        model_load_ownership_gate(),
        resource_observation_ownership_gate(),
    ]
    if run_gates:
        gates.extend(
            [
                run_command_gate("git_diff_check", ["git", "diff", "--check"]),
                run_command_gate(
                    "make_lint",
                    _make_gate_command("lint"),
                    timeout_s=300,
                ),
                run_command_gate("governance_lint", [sys.executable, "tools/lint_governance.py"]),
            ]
        )
    _write_json(
        out_dir / "GATE_RESULTS.json",
        {
            "schema": "aura.closeout.gate_results.v1",
            "run_gates": run_gates,
            "gates": [asdict(gate) for gate in gates],
        },
    )

    status = git_status()
    summary = {
        "schema": "aura.closeout.checkpoint.v1",
        "generated_at_unix": time.time(),
        "elapsed_s": round(time.time() - started, 4),
        "git": status,
        "tracked_file_count": len(file_audits),
        "text_file_count": len(text_files),
        "binary_file_count": len(file_audits) - len(text_files),
        "code_file_count": len(code_files),
        "text_line_count": sum(item.line_count for item in text_files),
        "code_line_count": sum(item.line_count for item in code_files),
        "line_ledger_sha256": _sha256_file(line_ledger),
        "file_ledger_sha256": _sha256_file(file_ledger),
        "line_ledger_entries": sum(item.line_count for item in text_files),
        "semantic_review": semantic_review,
        "gate_passed": all(gate.passed for gate in gates),
        "gates": [asdict(gate) for gate in gates],
        "claim_supported": "closeout_mechanical_source_audit_checkpoint",
        "claim_not_supported": [
            "human_semantic_review_of_every_line_complete",
            "all_issues_fixed",
            "full_end_state_aura_complete",
            "24_72_hour_live_survival",
            "literal_personhood",
            "phenomenal_consciousness",
            "unbounded_agi_or_asi",
        ],
        "full_closeout_complete": False,
        "boundary": (
            "This checkpoint proves that every tracked text line was mechanically "
            "enumerated and hashed, and that configured gates passed. It does not "
            "prove every line was semantically reviewed or that no issues remain."
        ),
    }
    summary["verdict"] = "PASS" if summary["gate_passed"] and summary["line_ledger_entries"] > 0 else "FAIL"
    _write_json(out_dir / "CLOSEOUT_CHECKPOINT.json", summary)
    (out_dir / "FINAL_VERDICT.txt").write_text(
        json.dumps(
            {
                "schema": "aura.closeout.final_verdict.v1",
                "verdict": summary["verdict"],
                "claim_supported": summary["claim_supported"] if summary["verdict"] == "PASS" else "none",
                "claim_not_supported": summary["claim_not_supported"],
                "full_closeout_complete": False,
                "tracked_file_count": summary["tracked_file_count"],
                "text_line_count": summary["text_line_count"],
                "code_line_count": summary["code_line_count"],
                "semantic_review_coverage_ratio": semantic_review["semantic_review_coverage_ratio"],
                "full_semantic_review_current": semantic_review["full_semantic_review_current"],
                "gate_passed": summary["gate_passed"],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    write_manifest(out_dir)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=os.environ.get("AURA_CLOSEOUT_OUT", str(DEFAULT_OUT)))
    parser.add_argument("--allow-dirty", action="store_true", default=os.environ.get("AURA_CLOSEOUT_ALLOW_DIRTY") == "1")
    parser.add_argument("--run-gates", action="store_true", default=os.environ.get("AURA_CLOSEOUT_RUN_GATES", "1") == "1")
    parser.add_argument("--max-files", type=int, default=None, help="Test-only limit; do not use for real closeout checkpoints.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    summary = build_closeout_audit(
        out_dir=Path(args.out),
        allow_dirty=bool(args.allow_dirty),
        run_gates=bool(args.run_gates),
        max_files=args.max_files,
    )
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))
    return 0 if summary.get("verdict") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
