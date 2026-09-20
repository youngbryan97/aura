#!/usr/bin/env python3
"""Assemble the final closeout evidence bundle.

The closeout contract treats ``artifacts/current/final_closeout`` as the
last live artifact. This tool does not invent proof. It verifies the current
evidence set, runs the lightweight final validators, records hashes and git
state, and writes a compact final bundle that can be audited independently.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.runtime.subprocess_gateway import get_subprocess_gateway  # noqa: E402
DEFAULT_ARTIFACTS_DIR = ROOT / "artifacts" / "current"
DEFAULT_OUT_DIR = DEFAULT_ARTIFACTS_DIR / "final_closeout"


@dataclass(frozen=True)
class EvidenceSpec:
    key: str
    path: str
    required_pass_key: str | None = None


REQUIRED_EVIDENCE: tuple[EvidenceSpec, ...] = (
    EvidenceSpec("live_desktop_runtime", "live_desktop_runtime/LATEST_VERDICT.json", "passed"),
    EvidenceSpec("background_autonomy", "background_autonomy/MANIFEST.json", "passed"),
    EvidenceSpec("background_autonomy_report", "background_autonomy/BACKGROUND_AUTONOMY_REPORT.json", "passed"),
    EvidenceSpec("dnu_run_status", "agi_live/RUN_STATUS.json"),
    EvidenceSpec("dnu_scorecard", "agi_live/SCORECARD.json"),
    EvidenceSpec("dnu_proof", "agi_live/DNU_AGI_PROOF.json"),
    EvidenceSpec("aletheia_tier5", "aletheia_tier5_validation.json", "passed"),
    EvidenceSpec("receipt_coverage", "receipt_coverage.json", "passed"),
    EvidenceSpec("artifact_consistency", "artifact_consistency.json", "passed"),
    EvidenceSpec("final_claim_validation", "final_claim_validation.json", "passed"),
)

REQUIRED_CLOSEOUT_EVIDENCE: tuple[EvidenceSpec, ...] = (
    EvidenceSpec(
        "operational_label_battery",
        "../closeout/operational_label_battery_latest.json",
        "passed",
    ),
    EvidenceSpec(
        "frontier_standards",
        "../closeout/frontier_standards_latest.json",
        "passed",
    ),
    EvidenceSpec(
        "remaining_checkpoint_contract",
        "../closeout/remaining_checkpoint_contract_latest.json",
    ),
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return data


def _run(command: list[str], *, timeout: float = 300.0) -> dict[str, Any]:
    started = time.time()
    completed = get_subprocess_gateway().run(
        command,
        cwd=ROOT,
        timeout=timeout,
        offline_tooling=True,
        source="certification_tooling:final_closeout_assembler.run_step",
        accelerator_capability="auto",
    )
    finished = time.time()
    return {
        "command": command,
        "returncode": completed.returncode,
        "passed": completed.returncode == 0,
        "duration_s": round(finished - started, 3),
        "stdout_tail": (completed.stdout or "")[-4000:],
        "stderr_tail": (completed.stderr or "")[-4000:],
    }


def _git_state() -> dict[str, Any]:
    def _git(args: list[str]) -> str:
        completed = get_subprocess_gateway().run(
            ["git", "-c", "core.fsmonitor=false", *args],
            cwd=ROOT,
            timeout=30,
            read_only=True,
            source="certification_tooling:final_closeout_assembler.git_state",
            accelerator_capability="none",
        )
        return (completed.stdout or completed.stderr or "").strip()

    status = _git(["status", "--short"])
    return {
        "commit": _git(["rev-parse", "HEAD"]),
        "branch": _git(["rev-parse", "--abbrev-ref", "HEAD"]),
        "status_short": status,
        "clean": status == "",
    }


def _evidence_record(base: Path, spec: EvidenceSpec) -> dict[str, Any]:
    path = (base / spec.path).resolve()
    if not path.exists():
        return {
            "key": spec.key,
            "path": str(path),
            "exists": False,
            "passed": False,
            "reason": "missing",
        }
    record: dict[str, Any] = {
        "key": spec.key,
        "path": str(path),
        "exists": True,
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
        "passed": True,
    }
    if path.suffix.lower() == ".json":
        try:
            data = _load_json(path)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            record.update({"passed": False, "reason": f"json_error:{type(exc).__name__}: {exc}"})
            return record
        if spec.required_pass_key:
            record["reported_pass"] = data.get(spec.required_pass_key)
            record["passed"] = data.get(spec.required_pass_key) is True
            if not record["passed"]:
                record["reason"] = f"{spec.required_pass_key} is not true"
        if spec.key == "dnu_run_status":
            record["reported_status"] = data.get("status")
            record["tasks_completed"] = data.get("tasks_completed")
            record["total_tasks"] = data.get("total_tasks")
            record["passed"] = (
                data.get("status") == "complete"
                and data.get("runner_completed") is True
                and int(data.get("tasks_completed") or 0) == int(data.get("total_tasks") or -1)
                and int(data.get("total_tasks") or 0) >= 100
            )
            if not record["passed"]:
                record["reason"] = "dnu run status is incomplete"
        if spec.key == "background_autonomy_report":
            record["components_running"] = data.get("components_running")
            record["components_total"] = data.get("components_total")
            record["desktop_access"] = {
                key: data.get("desktop_access", {}).get(key)
                for key in (
                    "overall_status",
                    "permission_confidence",
                    "screen_capture_ready",
                    "desktop_control_ready",
                    "screen_text_ready",
                    "blocking_permissions",
                )
            }
        if spec.key == "remaining_checkpoint_contract":
            record["gaps"] = data.get("summary", {}).get("gaps")
            record["remaining_checkpoints"] = data.get("summary", {}).get("remaining_checkpoints")
    return record


def assemble(
    *,
    artifacts_dir: Path,
    out_dir: Path,
    skip_validators: bool = False,
) -> tuple[int, dict[str, Any]]:
    artifacts_dir = artifacts_dir.resolve()
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    validator_results: list[dict[str, Any]] = []
    if not skip_validators:
        validator_results.extend(
            [
                _run([sys.executable, "tools/artifact_consistency_validator.py", "--artifacts", str(artifacts_dir)]),
                _run(
                    [
                        sys.executable,
                        "tools/final_claim_validator.py",
                        "--claims",
                        "CLAIMS_MATRIX.md",
                        "--artifacts",
                        str(artifacts_dir),
                    ]
                ),
                _run(
                    [
                        sys.executable,
                        "tools/closeout/remaining_checkpoint_contract.py",
                        "--json",
                        "--require-live",
                    ]
                ),
            ]
        )

    evidence = [
        _evidence_record(artifacts_dir, spec)
        for spec in REQUIRED_EVIDENCE
    ]
    closeout_base = artifacts_dir / ".." / "closeout"
    evidence.extend(_evidence_record(closeout_base, spec) for spec in REQUIRED_CLOSEOUT_EVIDENCE)

    failed_evidence = [item for item in evidence if not item.get("passed")]
    failed_validators = [item for item in validator_results if not item.get("passed")]
    git_state = _git_state()
    report = {
        "schema": "aura.final_closeout.v1",
        "generated_at_unix": time.time(),
        "artifacts_dir": str(artifacts_dir),
        "out_dir": str(out_dir),
        "git": git_state,
        "validators": validator_results,
        "evidence": evidence,
        "failed_evidence": failed_evidence,
        "failed_validators": failed_validators,
        "passed": not failed_evidence and not failed_validators,
        "claim_boundary": (
            "Final closeout verifies the configured local evidence profile and "
            "daily-runtime artifacts. It does not prove metaphysical consciousness, "
            "legal personhood, ASI, or unrestricted autonomy."
        ),
    }

    final_json = out_dir / "FINAL_CLOSEOUT.json"
    final_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    sums = [
        f"{item.get('sha256')}  {_display_path(Path(str(item.get('path'))))}"
        for item in evidence
        if item.get("exists") and item.get("sha256")
    ]
    (out_dir / "SHA256SUMS.txt").write_text("\n".join(sums) + "\n", encoding="utf-8")
    (out_dir / "FINAL_CLOSEOUT.md").write_text(_markdown_summary(report), encoding="utf-8")
    return (0 if report["passed"] else 1), report


def _markdown_summary(report: dict[str, Any]) -> str:
    status = "PASS" if report.get("passed") else "FAIL"
    failed_evidence = report.get("failed_evidence") or []
    failed_validators = report.get("failed_validators") or []
    lines = [
        "# Aura Final Closeout",
        "",
        f"Status: **{status}**",
        "",
        f"Git commit: `{report.get('git', {}).get('commit', '')}`",
        f"Git clean at generation: `{report.get('git', {}).get('clean')}`",
        "",
        "## Evidence",
        "",
    ]
    for item in report.get("evidence", []):
        mark = "PASS" if item.get("passed") else "FAIL"
        rel = item.get("path", "")
        rel = _display_path(Path(str(rel)))
        lines.append(f"- {mark}: `{item.get('key')}` -> `{rel}`")
    lines.extend(["", "## Validators", ""])
    for item in report.get("validators", []):
        mark = "PASS" if item.get("passed") else "FAIL"
        command = " ".join(str(part) for part in item.get("command", []))
        lines.append(f"- {mark}: `{command}` ({item.get('duration_s')}s)")
    if failed_evidence or failed_validators:
        lines.extend(["", "## Failures", ""])
        for item in failed_evidence:
            lines.append(f"- Evidence `{item.get('key')}`: {item.get('reason', 'failed')}")
        for item in failed_validators:
            lines.append(f"- Validator `{item.get('command')}`: rc={item.get('returncode')}")
    lines.extend(["", "## Boundary", "", str(report.get("claim_boundary", "")), ""])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--skip-validators", action="store_true")
    args = parser.parse_args(argv)
    rc, report = assemble(
        artifacts_dir=args.artifacts_dir,
        out_dir=args.out_dir,
        skip_validators=args.skip_validators,
    )
    print(json.dumps({
        "passed": report.get("passed"),
        "out_dir": report.get("out_dir"),
        "failed_evidence": [item.get("key") for item in report.get("failed_evidence", [])],
        "failed_validators": [item.get("command") for item in report.get("failed_validators", [])],
    }, indent=2, sort_keys=True))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
