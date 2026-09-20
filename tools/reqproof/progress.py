#!/usr/bin/env python3
"""Evidence-weighted Aura completion and total-checkpoint forecast engine.

This engine never infers completion from tracker prose or a requirement's
status label. Credit exists only for a verified evidence artifact covering one
explicit acceptance/evidence-class cell. Forecasts are estimates, kept
separate from certified completion and accompanied by their calibration state.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from decimal import ROUND_CEILING, ROUND_DOWN, Decimal
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.reqproof.evidence import (  # noqa: E402
    DEFAULT_EVIDENCE_LEDGER_PATH,
    EvidenceLedger,
    EvidenceLedgerError,
    load_evidence_ledger,
    verify_ledger_binding,
)
from tools.reqproof.migrate import DEFAULT_REGISTRY_PATH  # noqa: E402
from tools.reqproof.schema import EVIDENCE_CLASSES, Registry, load_registry  # noqa: E402
from tools.reqproof.tracker_parse import TRACKER_RELPATH  # noqa: E402
from tools.reqproof.validate import (  # noqa: E402
    default_commit_exists,
    verified_acceptance_coverage,
)

PROGRESS_SCHEMA_VERSION = 1
POLICY_SCHEMA_VERSION = 1
DEFAULT_POLICY_PATH = ROOT / "config" / "reqproof_progress_policy.json"
DEFAULT_SCOPE_BASELINE_PATH = ROOT / "config" / "reqproof_scope_baseline.json"
DEFAULT_SCOPE_MIGRATION_PATH = (
    ROOT / "artifacts" / "reqproof" / "SCOPE_DENOMINATOR_MIGRATION.json"
)
DEFAULT_REPORT_PATH = ROOT / "artifacts" / "reqproof" / "PROGRESS_REPORT.json"
DEFAULT_MARKDOWN_PATH = ROOT / "docs" / "AURA_PROGRESS.md"
CHECKPOINT_RE = re.compile(
    r"^## Checkpoint (?P<record_id>\d{4}-\d{2}-\d{2}-\d+): (?P<title>.+)$"
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ProgressError(ValueError):
    """Progress inputs or source-control evidence violated the contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ProgressError(message)


def _content_sha256(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    try:
        with tmp_path.open("w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


@dataclass(frozen=True)
class ScopeBaseline:
    fingerprints: tuple[str, ...]
    content_sha256: str = field(default="", compare=False)

    ALLOWED_KEYS = frozenset({"schema_version", "fingerprints", "content_sha256"})

    @classmethod
    def from_registry(cls, registry: Registry) -> ScopeBaseline:
        return cls(fingerprints=scope_fingerprints(registry))

    @classmethod
    def from_dict(cls, data: Any, *, verify_hash: bool = True) -> ScopeBaseline:
        _require(isinstance(data, dict), "scope baseline must be an object")
        unknown = set(data) - cls.ALLOWED_KEYS
        _require(not unknown, f"scope baseline has unknown fields: {sorted(unknown)}")
        _require(data.get("schema_version") == 1, "bad scope baseline schema")
        fingerprints = data.get("fingerprints")
        _require(isinstance(fingerprints, list), "scope fingerprints must be a list")
        _require(
            all(isinstance(value, str) and bool(value) for value in fingerprints),
            "scope fingerprints must be non-empty strings",
        )
        _require(
            fingerprints == sorted(set(fingerprints)),
            "scope fingerprints must be sorted and unique",
        )
        baseline = cls(
            fingerprints=tuple(fingerprints),
            content_sha256=str(data.get("content_sha256", "")),
        )
        if verify_hash:
            recorded = data.get("content_sha256")
            _require(
                isinstance(recorded, str) and bool(SHA256_RE.match(recorded)),
                "scope baseline content_sha256 missing or malformed",
            )
            _require(
                recorded == baseline.compute_content_sha256(),
                "scope baseline content hash mismatch",
            )
        return baseline

    def body_dict(self) -> dict[str, Any]:
        return {"schema_version": 1, "fingerprints": list(self.fingerprints)}

    def compute_content_sha256(self) -> str:
        return _content_sha256(self.body_dict())

    def to_dict(self) -> dict[str, Any]:
        body = self.body_dict()
        body["content_sha256"] = self.compute_content_sha256()
        return body


def scope_fingerprints(registry: Registry) -> tuple[str, ...]:
    fingerprints = []
    for requirement in registry.requirements:
        if not requirement.mandatory:
            continue
        for acceptance_id, class_name in requirement.required_evidence_cells():
            fingerprints.append(f"{requirement.id}::{acceptance_id}::{class_name}")
    return tuple(sorted(fingerprints))


def acceptance_fingerprints_from_cells(
    fingerprints: tuple[str, ...],
) -> tuple[str, ...]:
    acceptance_units: set[str] = set()
    for fingerprint in fingerprints:
        parts = fingerprint.split("::")
        _require(
            len(parts) == 3 and re.fullmatch(r"A[1-9][0-9]*", parts[1]) is not None,
            f"malformed scope fingerprint {fingerprint!r}",
        )
        acceptance_units.add("::".join(parts[:2]))
    return tuple(sorted(acceptance_units))


def build_scope_migration_receipt(
    previous: ScopeBaseline,
    registry: Registry,
    *,
    reason: str,
) -> tuple[ScopeBaseline, dict[str, Any]]:
    """Authorize a modality correction without permitting scope-unit loss."""
    _require(bool(reason.strip()), "scope migration reason must be non-empty")
    current = ScopeBaseline.from_registry(registry)
    previous_acceptance = set(
        acceptance_fingerprints_from_cells(previous.fingerprints)
    )
    current_acceptance = set(
        acceptance_fingerprints_from_cells(current.fingerprints)
    )
    removed_acceptance = sorted(previous_acceptance - current_acceptance)
    _require(
        not removed_acceptance,
        "scope migration refuses acceptance-obligation shrink: "
        f"{removed_acceptance[:10]}",
    )
    removed = sorted(set(previous.fingerprints) - set(current.fingerprints))
    added = sorted(set(current.fingerprints) - set(previous.fingerprints))
    body: dict[str, Any] = {
        "schema": "aura.reqproof.scope_denominator_migration.v1",
        "previous_baseline_sha256": previous.compute_content_sha256(),
        "next_baseline_sha256": current.compute_content_sha256(),
        "registry_sha256": registry.compute_content_sha256(),
        "reason": reason.strip(),
        "invariants": {
            "acceptance_units_before": len(previous_acceptance),
            "acceptance_units_after": len(current_acceptance),
            "acceptance_units_removed": removed_acceptance,
            "acceptance_obligation_shrink": False,
        },
        "removed_cells": removed,
        "added_cells": added,
    }
    body["receipt_sha256"] = _content_sha256(body)
    return current, body


def verify_scope_baseline(baseline: ScopeBaseline, registry: Registry) -> None:
    current = set(scope_fingerprints(registry))
    recorded = set(baseline.fingerprints)
    removed = sorted(recorded - current)
    added = sorted(current - recorded)
    _require(
        not removed,
        "scope denominator shrank; recorded cells disappeared: "
        f"{removed[:10]}",
    )
    _require(
        not added,
        "scope baseline is stale because new cells were added; run "
        f"progress.py --refresh-scope-baseline: {added[:10]}",
    )


def load_scope_baseline(path: Path) -> ScopeBaseline:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ProgressError(f"scope baseline not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ProgressError(f"scope baseline is not valid JSON: {path}: {exc}") from exc
    return ScopeBaseline.from_dict(data)


def write_scope_baseline_atomic(baseline: ScopeBaseline, path: Path) -> None:
    _atomic_write(path, json.dumps(baseline.to_dict(), indent=2, sort_keys=True) + "\n")


@dataclass(frozen=True)
class ProgressPolicy:
    class_weights: dict[str, int]
    optimistic_points_per_checkpoint: int
    conservative_points_per_checkpoint: int
    medium_confidence_verified_basis_points: int
    high_confidence_verified_basis_points: int
    legacy_engineering_estimate_basis_points: int
    content_sha256: str = field(default="", compare=False)

    ALLOWED_KEYS = frozenset(
        {
            "schema_version",
            "class_weights",
            "forecast",
            "confidence",
            "legacy_engineering_estimate_basis_points",
            "content_sha256",
        }
    )

    @classmethod
    def from_dict(cls, data: Any, *, verify_hash: bool = True) -> ProgressPolicy:
        _require(isinstance(data, dict), "progress policy must be an object")
        unknown = set(data) - cls.ALLOWED_KEYS
        _require(not unknown, f"progress policy has unknown fields: {sorted(unknown)}")
        _require(data.get("schema_version") == POLICY_SCHEMA_VERSION, "bad policy schema")

        weights = data.get("class_weights")
        _require(isinstance(weights, dict), "class_weights must be an object")
        _require(set(weights) == set(EVIDENCE_CLASSES), "class_weights must be exhaustive")
        _require(
            all(isinstance(value, int) and not isinstance(value, bool) and value > 0 for value in weights.values()),
            "class_weights must be positive integers",
        )
        forecast = data.get("forecast")
        _require(isinstance(forecast, dict), "forecast must be an object")
        _require(
            set(forecast)
            == {"optimistic_points_per_checkpoint", "conservative_points_per_checkpoint"},
            "forecast fields are invalid",
        )
        optimistic = forecast["optimistic_points_per_checkpoint"]
        conservative = forecast["conservative_points_per_checkpoint"]
        _require(
            all(isinstance(value, int) and not isinstance(value, bool) and value > 0 for value in (optimistic, conservative)),
            "forecast throughput values must be positive integers",
        )
        _require(
            optimistic > conservative,
            "optimistic throughput must exceed conservative throughput",
        )
        confidence = data.get("confidence")
        _require(isinstance(confidence, dict), "confidence must be an object")
        _require(
            set(confidence)
            == {"medium_verified_basis_points", "high_verified_basis_points"},
            "confidence fields are invalid",
        )
        medium = confidence["medium_verified_basis_points"]
        high = confidence["high_verified_basis_points"]
        _require(
            all(isinstance(value, int) and not isinstance(value, bool) for value in (medium, high)),
            "confidence thresholds must be integers",
        )
        _require(0 <= medium < high <= 10_000, "confidence thresholds are out of range")
        legacy_estimate = data.get("legacy_engineering_estimate_basis_points")
        _require(
            isinstance(legacy_estimate, int)
            and not isinstance(legacy_estimate, bool)
            and 0 <= legacy_estimate <= 10_000,
            "legacy engineering estimate must be integer basis points",
        )

        policy = cls(
            class_weights=dict(sorted(weights.items())),
            optimistic_points_per_checkpoint=optimistic,
            conservative_points_per_checkpoint=conservative,
            medium_confidence_verified_basis_points=medium,
            high_confidence_verified_basis_points=high,
            legacy_engineering_estimate_basis_points=legacy_estimate,
            content_sha256=str(data.get("content_sha256", "")),
        )
        if verify_hash:
            recorded = data.get("content_sha256")
            _require(
                isinstance(recorded, str) and bool(SHA256_RE.match(recorded)),
                "progress policy content_sha256 missing or malformed",
            )
            actual = policy.compute_content_sha256()
            _require(recorded == actual, "progress policy content hash mismatch")
        return policy

    def body_dict(self) -> dict[str, Any]:
        return {
            "schema_version": POLICY_SCHEMA_VERSION,
            "class_weights": dict(sorted(self.class_weights.items())),
            "forecast": {
                "optimistic_points_per_checkpoint": self.optimistic_points_per_checkpoint,
                "conservative_points_per_checkpoint": self.conservative_points_per_checkpoint,
            },
            "confidence": {
                "medium_verified_basis_points": self.medium_confidence_verified_basis_points,
                "high_verified_basis_points": self.high_confidence_verified_basis_points,
            },
            "legacy_engineering_estimate_basis_points": self.legacy_engineering_estimate_basis_points,
        }

    def compute_content_sha256(self) -> str:
        return _content_sha256(self.body_dict())

    def to_dict(self) -> dict[str, Any]:
        body = self.body_dict()
        body["content_sha256"] = self.compute_content_sha256()
        return body


def load_policy(path: Path) -> ProgressPolicy:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ProgressError(f"progress policy not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ProgressError(f"progress policy is not valid JSON: {path}: {exc}") from exc
    return ProgressPolicy.from_dict(data)


@dataclass(frozen=True)
class CheckpointRecord:
    record_id: str
    title: str
    commit: str
    pushed: bool


def parse_checkpoint_blame(blame_text: str, remote_commits: set[str]) -> tuple[CheckpointRecord, ...]:
    current_commit = ""
    records: list[CheckpointRecord] = []
    for line in blame_text.splitlines():
        header = re.match(r"^([0-9a-f]{40}) \d+ \d+(?: \d+)?$", line)
        if header:
            current_commit = header.group(1)
            continue
        if not line.startswith("\t"):
            continue
        match = CHECKPOINT_RE.match(line[1:])
        if match:
            _require(bool(current_commit), "checkpoint blame line has no commit")
            records.append(
                CheckpointRecord(
                    record_id=match.group("record_id"),
                    title=match.group("title"),
                    commit=current_commit,
                    pushed=current_commit in remote_commits,
                )
            )
    ids = [record.record_id for record in records]
    _require(len(ids) == len(set(ids)), "duplicate checkpoint record IDs exist")
    return tuple(records)


def load_checkpoint_records(root: Path, tracker_path: Path) -> tuple[CheckpointRecord, ...]:
    from core.runtime.subprocess_gateway import get_subprocess_gateway

    gateway = get_subprocess_gateway()
    tracker_ref = str(tracker_path.resolve().relative_to(root.resolve()))
    blame = gateway.run(
        ["git", "blame", "--line-porcelain", "HEAD", "--", tracker_ref],
        cwd=root,
        timeout=90,
        read_only=True,
        source="reqproof_progress_checkpoint_blame",
        accelerator_capability="none",
    )
    _require(blame.returncode == 0, f"checkpoint blame failed: {blame.stderr.strip()}")
    remote = gateway.run(
        ["git", "-c", "core.fsmonitor=false", "rev-list", "origin/main"],
        cwd=root,
        timeout=90,
        read_only=True,
        source="reqproof_progress_remote_commits",
        accelerator_capability="none",
    )
    _require(remote.returncode == 0, f"origin/main inventory failed: {remote.stderr.strip()}")
    return parse_checkpoint_blame(blame.stdout, set(remote.stdout.splitlines()))


def _decimal_text(value: Decimal) -> str:
    return format(value.quantize(Decimal("0.01"), rounding=ROUND_DOWN), "f")


def build_progress_report(
    *,
    root: Path,
    registry: Registry,
    ledger: EvidenceLedger,
    policy: ProgressPolicy,
    scope_baseline: ScopeBaseline,
    checkpoint_records: tuple[CheckpointRecord, ...],
    commit_exists: Callable[[str], bool] | None = None,
) -> dict[str, Any]:
    try:
        verify_ledger_binding(ledger, registry)
    except EvidenceLedgerError as exc:
        raise ProgressError(str(exc)) from exc
    verify_scope_baseline(scope_baseline, registry)
    if commit_exists is None:
        commit_exists = default_commit_exists(root)

    entries_by_requirement = ledger.entries_by_requirement()
    total_points = Decimal(0)
    verified_points = Decimal(0)
    total_cells = 0
    verified_cells = 0
    class_totals: Counter[str] = Counter()
    class_verified: Counter[str] = Counter()
    requirement_rows: list[dict[str, Any]] = []
    assigned_weights = 0

    for requirement in registry.requirements:
        if not requirement.mandatory:
            continue
        if requirement.weight_provenance == "assigned":
            assigned_weights += 1
        acceptance_ids = requirement.acceptance_ids()
        coverage = verified_acceptance_coverage(
            requirement,
            requirement.evidence,
            entries_by_requirement.get(requirement.id, ()),
            root,
            commit_exists,
        )

        row_total = Decimal(0)
        row_verified = Decimal(0)
        row_cells = 0
        row_verified_cells = 0
        weight_base = Decimal(str(requirement.risk_weight)) * Decimal(
            str(requirement.proof_weight)
        )
        for acceptance_id, class_name in requirement.required_evidence_cells():
            point_weight = weight_base * Decimal(policy.class_weights[class_name])
            total_cells += 1
            row_cells += 1
            class_totals[class_name] += 1
            total_points += point_weight
            row_total += point_weight
            if acceptance_id in coverage.get(class_name, set()):
                verified_cells += 1
                row_verified_cells += 1
                class_verified[class_name] += 1
                verified_points += point_weight
                row_verified += point_weight
        requirement_rows.append(
            {
                "id": requirement.id,
                "kind": requirement.kind,
                "state_claim": requirement.state,
                "weight_provenance": requirement.weight_provenance,
                "acceptance_units": len(acceptance_ids),
                "required_evidence_classes": list(requirement.evidence_required),
                "acceptance_evidence_required": [
                    list(classes)
                    for classes in requirement.acceptance_evidence_required
                ],
                "cells_total": row_cells,
                "cells_verified": row_verified_cells,
                "weighted_points_total": _decimal_text(row_total),
                "weighted_points_verified": _decimal_text(row_verified),
            }
        )

    _require(total_points > 0, "progress denominator is empty")
    completion = verified_points * Decimal(100) / total_points
    verified_basis_points = int(
        (verified_points * Decimal(10_000) / total_points).to_integral_value(
            rounding=ROUND_DOWN
        )
    )
    remaining_points = total_points - verified_points
    remaining_low = int(
        (remaining_points / Decimal(policy.optimistic_points_per_checkpoint)).to_integral_value(
            rounding=ROUND_CEILING
        )
    )
    remaining_high = int(
        (remaining_points / Decimal(policy.conservative_points_per_checkpoint)).to_integral_value(
            rounding=ROUND_CEILING
        )
    )

    pushed_records = [record for record in checkpoint_records if record.pushed]
    unpushed_records = [record for record in checkpoint_records if not record.pushed]
    pushed_commits = {record.commit for record in pushed_records}
    records_per_commit = Counter(record.commit for record in pushed_records)
    shared_commit_records = sum(
        count for count in records_per_commit.values() if count > 1
    )
    if verified_basis_points >= policy.high_confidence_verified_basis_points:
        confidence = "high"
    elif verified_basis_points >= policy.medium_confidence_verified_basis_points:
        confidence = "medium"
    else:
        confidence = "low"

    class_rows = []
    for class_name in EVIDENCE_CLASSES:
        class_rows.append(
            {
                "evidence_class": class_name,
                "weight": policy.class_weights[class_name],
                "cells_total": class_totals[class_name],
                "cells_verified": class_verified[class_name],
            }
        )

    body: dict[str, Any] = {
        "schema_version": PROGRESS_SCHEMA_VERSION,
        "inputs": {
            "registry_sha256": registry.compute_content_sha256(),
            "evidence_ledger_sha256": ledger.compute_content_sha256(),
            "progress_policy_sha256": policy.compute_content_sha256(),
            "scope_baseline_sha256": scope_baseline.compute_content_sha256(),
        },
        "scope": {
            "requirements_total": len(registry.requirements),
            "mandatory_requirements": len(requirement_rows),
            "assigned_weight_requirements": assigned_weights,
            "acceptance_evidence_cells_total": total_cells,
            "scope_baseline_cells": len(scope_baseline.fingerprints),
            "acceptance_evidence_cells_verified": verified_cells,
            "weighted_points_total": _decimal_text(total_points),
            "weighted_points_verified": _decimal_text(verified_points),
            "weighted_points_remaining": _decimal_text(remaining_points),
        },
        "completion": {
            "machine_certified_percent": _decimal_text(completion),
            "verified_basis_points": verified_basis_points,
            "status": (
                "certified_complete"
                if verified_basis_points == 10_000
                else "provisional_evidence_backfill_incomplete"
            ),
            "engineering_estimate_percent": _decimal_text(
                Decimal(policy.legacy_engineering_estimate_basis_points) / Decimal(100)
            ),
            "engineering_estimate_status": "legacy_uncertified_not_used_for_release",
        },
        "checkpoint_inventory": {
            "records_in_tracker": len(checkpoint_records),
            "pushed_checkpoint_records": len(pushed_records),
            "unpushed_checkpoint_records": len(unpushed_records),
            "distinct_pushed_commits": len(pushed_commits),
            "records_on_shared_commits": shared_commit_records,
            "unpushed_record_ids": [record.record_id for record in unpushed_records],
        },
        "forecast": {
            "confidence": confidence,
            "remaining_checkpoint_records_low": remaining_low,
            "remaining_checkpoint_records_high": remaining_high,
            "total_checkpoint_records_low": len(pushed_records) + remaining_low,
            "total_checkpoint_records_high": len(pushed_records) + remaining_high,
            "optimistic_weighted_points_per_checkpoint": policy.optimistic_points_per_checkpoint,
            "conservative_weighted_points_per_checkpoint": policy.conservative_points_per_checkpoint,
            "calibration": (
                "policy_prior_only"
                if assigned_weights == 0 or verified_cells == 0
                else "policy_prior_with_partial_evidence"
            ),
        },
        "evidence_classes": class_rows,
        "requirements": requirement_rows,
        "non_claims": [
            "The 27 percent engineering estimate is not machine-certified and cannot release Aura.",
            "A forecast is an explicit throughput estimate, not evidence that any remaining work is complete.",
            "Checkpoint records sharing one commit are counted as records and exposed separately from distinct commits.",
            "Parent and child requirements are both counted when each declares its own integration or proof burden.",
        ],
    }
    body["report_sha256"] = _content_sha256(body)
    return body


def render_markdown(report: dict[str, Any]) -> str:
    scope = report["scope"]
    completion = report["completion"]
    checkpoints = report["checkpoint_inventory"]
    forecast = report["forecast"]
    lines = [
        "# Aura Progress Control",
        "",
        "> Generated by `tools/reqproof/progress.py`. Do not edit by hand.",
        "",
        "## Current Truth",
        "",
        f"- Machine-certified completion: **{completion['machine_certified_percent']}%**",
        f"- Legacy engineering estimate: **{completion['engineering_estimate_percent']}%** ({completion['engineering_estimate_status']})",
        f"- Verified acceptance/evidence cells: **{scope['acceptance_evidence_cells_verified']} / {scope['acceptance_evidence_cells_total']}**",
        f"- Mandatory requirements: **{scope['mandatory_requirements']}**",
        f"- Weight calibration: **{scope['assigned_weight_requirements']} assigned / {scope['mandatory_requirements']}**",
        "",
        "## Checkpoints",
        "",
        f"- Pushed checkpoint records: **{checkpoints['pushed_checkpoint_records']}**",
        f"- Distinct pushed commits: **{checkpoints['distinct_pushed_commits']}**",
        f"- Records on shared commits: **{checkpoints['records_on_shared_commits']}**",
        f"- Unpushed checkpoint records: **{checkpoints['unpushed_checkpoint_records']}**",
        f"- Forecast total: **{forecast['total_checkpoint_records_low']}-{forecast['total_checkpoint_records_high']} records** ({forecast['confidence']} confidence; {forecast['calibration']})",
        f"- Forecast remaining: **{forecast['remaining_checkpoint_records_low']}-{forecast['remaining_checkpoint_records_high']} records**",
        "",
        "## Evidence Burden",
        "",
        "| Class | Weight | Verified | Total |",
        "|---|---:|---:|---:|",
    ]
    for row in report["evidence_classes"]:
        lines.append(
            f"| `{row['evidence_class']}` | {row['weight']} | "
            f"{row['cells_verified']} | {row['cells_total']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The certified percentage starts at zero until reviewed historical artifacts are entered into the acceptance-granular ledger. This does not assert that no engineering exists; it prevents unverified history from being promoted into release credit.",
            "",
            "The forecast is conservative while evidence and weight calibration are incomplete. It will narrow from observed verified points per pushed checkpoint without changing the denominator or dropping open scope.",
            "",
            f"Report SHA-256: `{report['report_sha256']}`",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY_PATH))
    parser.add_argument("--evidence-ledger", default=str(DEFAULT_EVIDENCE_LEDGER_PATH))
    parser.add_argument("--policy", default=str(DEFAULT_POLICY_PATH))
    parser.add_argument("--scope-baseline", default=str(DEFAULT_SCOPE_BASELINE_PATH))
    parser.add_argument("--tracker", default=str(ROOT / TRACKER_RELPATH))
    parser.add_argument("--report", default=str(DEFAULT_REPORT_PATH))
    parser.add_argument("--markdown", default=str(DEFAULT_MARKDOWN_PATH))
    parser.add_argument(
        "--refresh-scope-baseline",
        action="store_true",
        help="add new scope cells; refuses any denominator shrink",
    )
    parser.add_argument(
        "--migrate-scope-baseline",
        action="store_true",
        help="replace an invalid Cartesian denominator without dropping acceptance units",
    )
    parser.add_argument(
        "--scope-migration-report",
        default=str(DEFAULT_SCOPE_MIGRATION_PATH),
    )
    parser.add_argument(
        "--scope-migration-reason",
        default="",
    )
    args = parser.parse_args()

    registry = load_registry(Path(args.registry))
    ledger = load_evidence_ledger(Path(args.evidence_ledger))
    policy = load_policy(Path(args.policy))
    scope_baseline_path = Path(args.scope_baseline)
    _require(
        not (args.refresh_scope_baseline and args.migrate_scope_baseline),
        "scope baseline refresh and migration are mutually exclusive",
    )
    if args.migrate_scope_baseline:
        previous = load_scope_baseline(scope_baseline_path)
        scope_baseline, receipt = build_scope_migration_receipt(
            previous,
            registry,
            reason=args.scope_migration_reason,
        )
        _atomic_write(
            Path(args.scope_migration_report),
            json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        )
        write_scope_baseline_atomic(scope_baseline, scope_baseline_path)
    elif args.refresh_scope_baseline:
        previous = load_scope_baseline(scope_baseline_path)
        current_fingerprints = scope_fingerprints(registry)
        removed = sorted(set(previous.fingerprints) - set(current_fingerprints))
        _require(
            not removed,
            f"scope baseline refresh refuses denominator shrink: {removed[:10]}",
        )
        scope_baseline = ScopeBaseline(fingerprints=current_fingerprints)
        write_scope_baseline_atomic(scope_baseline, scope_baseline_path)
    else:
        scope_baseline = load_scope_baseline(scope_baseline_path)
    records = load_checkpoint_records(ROOT, Path(args.tracker))
    report = build_progress_report(
        root=ROOT,
        registry=registry,
        ledger=ledger,
        policy=policy,
        scope_baseline=scope_baseline,
        checkpoint_records=records,
    )
    _atomic_write(
        Path(args.report), json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    _atomic_write(Path(args.markdown), render_markdown(report))
    print(
        json.dumps(
            {
                "machine_certified_percent": report["completion"]["machine_certified_percent"],
                "pushed_checkpoint_records": report["checkpoint_inventory"]["pushed_checkpoint_records"],
                "forecast": report["forecast"],
                "report_sha256": report["report_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
