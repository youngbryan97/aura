"""tools/release_preflight.py — the one release checklist, machine-enforced.

(Distinct from `make preflight` / tools/runtime_preflight.py, which checks
the runtime ENVIRONMENT — disk, RAM, port, models. This is the CODE gate.)

Aerospace doesn't ask each engineer to remember the walkaround; the checklist
IS the procedure. Before this tool, "the gates" were folklore spread across
make targets (compile, lint, smoke, governance-lint, security,
enterprise-gate, production-gate, triage) that a session could — and
sometimes did — run partially. Preflight runs them ALL, in dependency order,
fail-fast, and writes one receipt (artifacts/reliability/preflight.json)
that says exactly what was checked, what passed, and how long each took.

The checklist is PINNED by tests/test_preflight_gate.py: removing or
reordering a gate fails the suite, so the checklist can only grow
deliberately.

Deliberately NOT here (deferred gates, named in the receipt so their absence
is a statement, not an oversight):
  * the full 6-chunk offline suite (~7,400 tests; run at certification)
  * the C4 startup-budget probe (needs a live boot; release-train step)
  * endurance soaks (need a live instance and hours)

Usage:
  python tools/release_preflight.py              # fail-fast, table + receipt
  python tools/release_preflight.py --keep-going # run all, report all fails
  make release-preflight
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

RECEIPT_SCHEMA = "aura.preflight.v1"
DEFAULT_RECEIPT = REPO_ROOT / "artifacts" / "reliability" / "preflight.json"

# Gates the checklist intentionally defers, with the reason on record.
DEFERRED_GATES: dict[str, str] = {
    "full_suite_6_chunks": "certification step — ~7,400 tests, run via tools/run_test_chunks.py",
    "startup_budget_probe": "needs a live boot — release-train step (tools/live_surface_probe.py)",
    "endurance_soak": "needs a live instance and hours — always the LAST gate before sign-off",
}


@dataclass(frozen=True)
class PreflightCheck:
    """One checklist item: a named command with a purpose."""

    name: str
    purpose: str
    command: tuple[str, ...]
    timeout_s: float = 600.0


def _python() -> str:
    return sys.executable


def default_checks() -> tuple[PreflightCheck, ...]:
    """The pinned checklist, in dependency order: syntax before style before
    behavior before scrutiny before forensics."""
    py = _python()
    return (
        PreflightCheck(
            name="compile",
            purpose="every Python file parses (core + tests)",
            command=(py, "-m", "compileall", "-q", "core", "tests"),
        ),
        PreflightCheck(
            name="lint",
            purpose="ruff three-pass (surface E9, critical F-codes, curated files)",
            command=("make", "lint", f"PYTHON={py}"),
        ),
        PreflightCheck(
            name="smoke",
            purpose="~100 contract tests, the fast behavioral floor",
            command=("make", "smoke", f"PYTHON={py}"),
        ),
        PreflightCheck(
            name="governance_lint",
            purpose="no ungoverned consequential calls; effect-ownership baseline holds",
            command=(py, "tools/lint_governance.py"),
        ),
        PreflightCheck(
            name="security_scan",
            purpose="local security scan",
            command=(py, "tools/security_scan.py"),
        ),
        PreflightCheck(
            name="enterprise_gate",
            purpose="static ratchets (placeholders, silent excepts, subprocess allowlist…)",
            command=("make", "enterprise-gate", f"PYTHON={py}"),
            timeout_s=900.0,
        ),
        PreflightCheck(
            name="production_gate",
            purpose="production readiness contract",
            command=("make", "production-gate", f"PYTHON={py}"),
        ),
        PreflightCheck(
            name="fresh_hard_deaths",
            purpose="crash triage: no hard deaths in the last 24h (exit honored, no '|| true')",
            command=(
                py, "tools/crash_triage.py", "--window-days", "1",
                "--out", "artifacts/reliability/preflight_triage.json",
            ),
        ),
        PreflightCheck(
            name="reqproof_structural",
            purpose=(
                "requirement registry matches the tracker, closure graph sound, "
                "corpus coverage zero-unmapped, defect ratchet holds"
            ),
            command=(py, "tools/reqproof/gate.py", "--mode", "structural"),
        ),
        PreflightCheck(
            name="reqproof_progress",
            purpose=(
                "recompute acceptance-granular certified completion and the "
                "total pushed-checkpoint forecast without tracker-prose credit"
            ),
            command=(
                py,
                "tools/reqproof/progress.py",
                "--markdown",
                "artifacts/reqproof/PROGRESS_REPORT.md",
            ),
        ),
    )


@dataclass
class CheckResult:
    name: str
    purpose: str
    status: str  # "pass" | "fail" | "skipped"
    duration_s: float = 0.0
    exit_code: int | None = None
    tail: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "purpose": self.purpose,
            "status": self.status,
            "duration_s": round(self.duration_s, 2),
            "exit_code": self.exit_code,
            "tail": self.tail,
        }


@dataclass
class PreflightReport:
    verdict: str
    results: list[CheckResult] = field(default_factory=list)
    started_at: float = 0.0
    finished_at: float = 0.0

    def to_dict(self) -> dict:
        return {
            "schema": RECEIPT_SCHEMA,
            "verdict": self.verdict,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_s": round(self.finished_at - self.started_at, 2),
            "checks": [result.to_dict() for result in self.results],
            "deferred_gates": DEFERRED_GATES,
        }


def _run_check(check: PreflightCheck, *, env: dict[str, str]) -> CheckResult:
    started = time.monotonic()
    try:
        completed = subprocess.run(
            list(check.command),
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=check.timeout_s,
        )
        duration = time.monotonic() - started
        tail_source = (completed.stdout or "") + (completed.stderr or "")
        tail = "\n".join(tail_source.strip().splitlines()[-6:])
        return CheckResult(
            name=check.name,
            purpose=check.purpose,
            status="pass" if completed.returncode == 0 else "fail",
            duration_s=duration,
            exit_code=completed.returncode,
            tail=tail if completed.returncode != 0 else "",
        )
    except subprocess.TimeoutExpired:
        return CheckResult(
            name=check.name,
            purpose=check.purpose,
            status="fail",
            duration_s=time.monotonic() - started,
            exit_code=None,
            tail=f"timed out after {check.timeout_s:.0f}s",
        )


def run_preflight(
    checks: tuple[PreflightCheck, ...] | None = None,
    *,
    keep_going: bool = False,
    env: dict[str, str] | None = None,
) -> PreflightReport:
    checks = checks if checks is not None else default_checks()
    run_env = dict(os.environ)
    if env:
        run_env.update(env)
    report = PreflightReport(verdict="PASS", started_at=time.time())
    failed = False
    for check in checks:
        if failed and not keep_going:
            report.results.append(
                CheckResult(
                    name=check.name,
                    purpose=check.purpose,
                    status="skipped",
                    tail="skipped: earlier check failed (fail-fast)",
                )
            )
            continue
        result = _run_check(check, env=run_env)
        report.results.append(result)
        if result.status == "fail":
            failed = True
    report.verdict = "FAIL" if failed else "PASS"
    report.finished_at = time.time()
    return report


def _render(report: PreflightReport) -> str:
    icon = {"pass": "ok", "fail": "failed", "skipped": "skipped "}
    lines = [f"preflight — {report.verdict}"]
    for result in report.results:
        duration = f"{result.duration_s:6.1f}s" if result.status != "skipped" else "      -"
        lines.append(
            f"  {icon.get(result.status, '?')} {result.name:<18} {duration}  {result.purpose}"
        )
        if result.status == "fail" and result.tail:
            for tail_line in result.tail.splitlines()[-3:]:
                lines.append(f"       │ {tail_line}")
    lines.append(f"deferred (deliberate): {', '.join(DEFERRED_GATES)}")
    return "\n".join(lines)


def _write_receipt(report: PreflightReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--keep-going", action="store_true",
                        help="run every check even after a failure")
    parser.add_argument("--out", type=Path, default=DEFAULT_RECEIPT,
                        help="receipt path (JSON)")
    args = parser.parse_args(argv)

    report = run_preflight(keep_going=args.keep_going)
    _write_receipt(report, args.out)
    print(_render(report))
    print(f"receipt: {args.out}")
    return 0 if report.verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
