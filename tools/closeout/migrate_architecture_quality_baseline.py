#!/usr/bin/env python3
"""Reconstruct, attribute, and attest an architecture baseline migration."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.architecture_quality.attestation import (  # noqa: E402
    attest_payload,
    verify_attested_payload,
)
from core.architecture_quality.scorer import ArchitectureQualityReport, score_codebase  # noqa: E402

LEGACY_COMMIT = "a6428d22d564b5a25754d1fd27443859c673b632"
OBSERVATION_COMMIT = "096f79defb9e71e4e9fff6250822403f25049d3c"
MIGRATION_SCHEMA = "aura.architecture_quality_baseline_migration.v1"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument(
        "--legacy-baseline",
        default=str(ROOT / "config" / "aura_architecture_quality_baseline.json"),
    )
    parser.add_argument("--legacy-commit", default=LEGACY_COMMIT)
    parser.add_argument("--observation-commit", default=OBSERVATION_COMMIT)
    parser.add_argument("--target-commit", default="HEAD")
    parser.add_argument("--output")
    parser.add_argument(
        "--signing-key",
        default="~/.aura/trust/architecture_quality_ed25519_private.pem",
    )
    parser.add_argument(
        "--trust-root",
        default=str(ROOT / "config" / "trust" / "architecture_quality_ed25519_public.pem"),
    )
    parser.add_argument("--verify", help="Verify an existing migration receipt")
    args = parser.parse_args()

    trust_root = Path(args.trust_root).read_bytes()
    if args.verify:
        data = json.loads(Path(args.verify).read_text(encoding="utf-8"))
        verify_migration_receipt(data, trusted_public_key_pem=trust_root)
        print(f"verified {data['migration_sha256']}")
        return 0
    if not args.output:
        parser.error("--output is required unless --verify is used")

    root = Path(args.root).resolve(strict=True)
    legacy_data = _load_legacy_baseline(
        root,
        legacy_commit=args.legacy_commit,
        baseline_path=Path(args.legacy_baseline),
    )
    receipt = build_migration_receipt(
        root,
        legacy_data=legacy_data,
        legacy_commit=args.legacy_commit,
        observation_commit=args.observation_commit,
        target_commit=args.target_commit,
        signing_key_path=Path(args.signing_key).expanduser(),
    )
    verify_migration_receipt(receipt, trusted_public_key_pem=trust_root)
    _atomic_write_json(Path(args.output), receipt)
    print(
        json.dumps(
            {
                "migration_sha256": receipt["migration_sha256"],
                "claim_reproduced": receipt["historical_claim_reproduction"]["all_checks_passed"],
                "legacy_to_observation": receipt["phases"][0]["summary"],
                "observation_to_target": receipt["phases"][1]["summary"],
            },
            sort_keys=True,
        )
    )
    return 0


def build_migration_receipt(
    root: Path,
    *,
    legacy_data: dict[str, Any],
    legacy_commit: str,
    observation_commit: str,
    target_commit: str,
    signing_key_path: Path,
) -> dict[str, Any]:
    legacy_commit = _commit_identity(root, legacy_commit)
    observation_commit = _commit_identity(root, observation_commit)
    target_commit = _commit_identity(root, target_commit)
    _require_ancestor(root, legacy_commit, observation_commit)
    _require_ancestor(root, observation_commit, target_commit)
    include_roots = tuple(legacy_data.get("include_roots") or ())
    if not include_roots:
        raise ValueError("legacy architecture baseline has no include_roots")

    with tempfile.TemporaryDirectory(prefix="aura-architecture-migration-") as temp:
        temp_path = Path(temp)
        legacy_root = _extract_commit(root, legacy_commit, temp_path / "legacy")
        observation_root = _extract_commit(root, observation_commit, temp_path / "observation")
        target_root = _extract_commit(root, target_commit, temp_path / "target")
        legacy_report = score_codebase(legacy_root, include_roots=include_roots)
        observation_report = score_codebase(observation_root, include_roots=include_roots)
        target_report = score_codebase(target_root, include_roots=include_roots)
        observed_legacy_report = _run_snapshot_scorer(observation_root, include_roots)

    path_changes_to_observation = _path_last_changes(
        root,
        observation_commit,
        include_roots,
    )
    path_changes_to_target = _path_last_changes(
        root,
        target_commit,
        include_roots,
    )
    phases = [
        _phase_delta(
            "legacy_to_observation",
            legacy_report,
            observation_report,
            path_changes_to_observation,
        ),
        _phase_delta(
            "observation_to_target",
            observation_report,
            target_report,
            path_changes_to_target,
        ),
    ]
    claim = _historical_claim_reproduction(legacy_data, observed_legacy_report)
    if not claim["all_checks_passed"]:
        raise ValueError(f"historical architecture claim did not reproduce: {claim['checks']}")
    failed_integrity = {
        phase["name"]: {
            key: value
            for key, value in phase["integrity"].items()
            if key != "all_checks_passed" and not value
        }
        for phase in phases
        if not phase["integrity"]["all_checks_passed"]
    }
    if failed_integrity:
        raise ValueError(f"architecture migration delta integrity failed: {failed_integrity}")

    payload = {
        "schema": MIGRATION_SCHEMA,
        "legacy_baseline_sha256": hashlib.sha256(
            json.dumps(legacy_data, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "snapshots": {
            "legacy": _snapshot(root, legacy_commit, legacy_report),
            "observation": _snapshot(root, observation_commit, observation_report),
            "target": _snapshot(root, target_commit, target_report),
        },
        "historical_reported_baseline": {
            "score": legacy_data["score"],
            "metrics": legacy_data["metrics"],
        },
        "historical_claim_reproduction": claim,
        "phases": phases,
        "migration_decision": {
            "status": "explicitly_migrated_not_erased",
            "legacy_debt_preserved": True,
            "future_ratchet_origin": target_commit,
            "future_regressions_waived": False,
            "rationale": (
                "The complete historical source deltas are retained below under one analyzer "
                "schema. The target baseline starts a strict non-regression ratchet; it does "
                "not relabel accumulated architecture debt as resolved."
            ),
        },
    }
    return attest_payload(
        payload,
        digest_field="migration_sha256",
        signing_key_path=signing_key_path,
    )


def verify_migration_receipt(
    data: dict[str, Any],
    *,
    trusted_public_key_pem: bytes,
) -> None:
    if data.get("schema") != MIGRATION_SCHEMA:
        raise ValueError("unsupported architecture migration schema")
    verify_attested_payload(
        data,
        digest_field="migration_sha256",
        trusted_public_key_pem=trusted_public_key_pem,
    )
    if not data.get("historical_claim_reproduction", {}).get("all_checks_passed"):
        raise ValueError("migration receipt did not reproduce the historical claim")
    phases = data.get("phases")
    if not isinstance(phases, list) or len(phases) != 2:
        raise ValueError("migration receipt must contain exactly two source phases")
    if not all(phase.get("integrity", {}).get("all_checks_passed") for phase in phases):
        raise ValueError("migration receipt contains an unverified source delta")
    decision = data.get("migration_decision", {})
    if not decision.get("legacy_debt_preserved") or decision.get("future_regressions_waived"):
        raise ValueError("migration receipt weakens the architecture ratchet")


def _phase_delta(
    name: str,
    before: ArchitectureQualityReport,
    after: ArchitectureQualityReport,
    path_changes: dict[str, str],
) -> dict[str, Any]:
    before_edges = _edges(before)
    after_edges = _edges(after)
    added_edges = sorted(after_edges - before_edges)
    removed_edges = sorted(before_edges - after_edges)
    before_modules = set(before.module_to_path)
    after_modules = set(after.module_to_path)
    before_oversized = _finding_paths(before, "structurally_oversized_module")
    after_oversized = _finding_paths(after, "structurally_oversized_module")
    before_owners = _finding_modules(before, "ambiguous_module_owner")
    after_owners = _finding_modules(after, "ambiguous_module_owner")
    before_cycles = {tuple(sorted(cycle)) for cycle in before.cycles}
    after_cycles = {tuple(sorted(cycle)) for cycle in after.cycles}
    before_executable = {tuple(sorted(cycle)) for cycle in before.executable_cycles}
    after_executable = {tuple(sorted(cycle)) for cycle in after.executable_cycles}
    metric_delta = {
        key: after.metrics.to_dict()[key] - before.metrics.to_dict()[key]
        for key in before.metrics.to_dict()
        if isinstance(before.metrics.to_dict()[key], (int, float))
    }
    changed_source_paths = sorted(
        {
            after.module_to_path.get(source) or before.module_to_path.get(source) or ""
            for source, _target in (*added_edges, *removed_edges)
        }
        | (after_oversized ^ before_oversized)
        | {
            after.module_to_path.get(module) or before.module_to_path.get(module) or ""
            for module in (after_modules ^ before_modules) | (after_owners ^ before_owners)
        }
    )
    changed_source_paths = [path for path in changed_source_paths if path]
    source_attribution = {
        path: path_changes.get(path, "unattributed") for path in changed_source_paths
    }
    integrity_checks = {
        "edge_delta_matches_metrics": (
            len(added_edges) - len(removed_edges)
            == after.metrics.dependency_edges - before.metrics.dependency_edges
        ),
        "module_delta_matches_metrics": (
            len(after_modules - before_modules) - len(before_modules - after_modules)
            == after.metrics.module_count - before.metrics.module_count
        ),
        "cycle_population_complete": len(after_cycles) == after.metrics.cycle_count,
        "oversized_population_complete": len(after_oversized) == after.metrics.god_file_count,
        "all_changed_sources_attributed": all(
            commit != "unattributed" for commit in source_attribution.values()
        ),
    }
    return {
        "name": name,
        "summary": {
            "score_delta": round(after.score - before.score, 6),
            "module_delta": metric_delta["module_count"],
            "dependency_edge_delta": metric_delta["dependency_edges"],
            "cycle_delta": metric_delta["cycle_count"],
            "oversized_module_delta": metric_delta["god_file_count"],
        },
        "metric_delta": metric_delta,
        "modules_added": sorted(after_modules - before_modules),
        "modules_removed": sorted(before_modules - after_modules),
        "dependency_edges_added": [list(edge) for edge in added_edges],
        "dependency_edges_removed": [list(edge) for edge in removed_edges],
        "runtime_cycles_added": [list(cycle) for cycle in sorted(after_cycles - before_cycles)],
        "runtime_cycles_removed": [list(cycle) for cycle in sorted(before_cycles - after_cycles)],
        "executable_cycles_added": [
            list(cycle) for cycle in sorted(after_executable - before_executable)
        ],
        "executable_cycles_removed": [
            list(cycle) for cycle in sorted(before_executable - after_executable)
        ],
        "oversized_modules_added": sorted(after_oversized - before_oversized),
        "oversized_modules_removed": sorted(before_oversized - after_oversized),
        "ambiguous_owners_added": sorted(after_owners - before_owners),
        "ambiguous_owners_removed": sorted(before_owners - after_owners),
        "source_path_last_change_commit": source_attribution,
        "integrity": {
            **integrity_checks,
            "all_checks_passed": all(integrity_checks.values()),
        },
    }


def _historical_claim_reproduction(
    baseline: dict[str, Any],
    observation: dict[str, Any],
) -> dict[str, Any]:
    baseline_metrics = baseline["metrics"]
    observed_metrics = observation["metrics"]
    raw_edge_growth = observed_metrics["dependency_edges"] - baseline_metrics["dependency_edges"]
    checks = {
        "baseline_score_46_38": float(baseline["score"]) == 46.38,
        "observation_score_42_49": float(observation["score"]) == 42.49,
        "raw_edge_growth_945": raw_edge_growth == 945,
        "edge_growth_beyond_one_edge_allowance_944": raw_edge_growth - 1 == 944,
        "oversized_module_growth_12": (
            observed_metrics["god_file_count"] - baseline_metrics["god_file_count"] == 12
        ),
        "largest_scc_expanded": (
            observed_metrics["largest_cycle_size"] > baseline_metrics["largest_cycle_size"]
        ),
    }
    return {
        "observed_legacy_scorer": observation,
        "raw_edge_growth": raw_edge_growth,
        "edge_growth_beyond_allowance": raw_edge_growth - 1,
        "oversized_module_growth": (
            observed_metrics["god_file_count"] - baseline_metrics["god_file_count"]
        ),
        "checks": checks,
        "all_checks_passed": all(checks.values()),
    }


def _snapshot(root: Path, commit: str, report: ArchitectureQualityReport) -> dict[str, Any]:
    metadata = subprocess.run(
        ["git", "-c", "core.fsmonitor=false", "show", "-s", "--format=%aI%x00%s", commit],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.rstrip("\n").split("\x00", 1)
    return {
        "commit": commit,
        "authored_at": metadata[0],
        "subject": metadata[1],
        "report_attestation_sha256": report.attestation_sha256,
        "analysis_scope_sha256": report.analysis_scope_sha256,
        "score": round(report.score, 6),
        "metrics": report.metrics.to_dict(),
    }


def _edges(report: ArchitectureQualityReport) -> set[tuple[str, str]]:
    return {
        (source, target)
        for source, targets in report.graph.items()
        for target in targets
    }


def _finding_paths(report: ArchitectureQualityReport, code: str) -> set[str]:
    return {
        finding.path
        for finding in report.findings
        if finding.code == code and finding.path is not None
    }


def _finding_modules(report: ArchitectureQualityReport, code: str) -> set[str]:
    return {
        finding.modules[0]
        for finding in report.findings
        if finding.code == code and finding.modules
    }


def _extract_commit(root: Path, commit: str, destination: Path) -> Path:
    destination.mkdir(parents=True)
    with tempfile.TemporaryFile() as archive:
        subprocess.run(
            ["git", "archive", "--format=tar", commit],
            cwd=root,
            check=True,
            stdout=archive,
        )
        archive.seek(0)
        with tarfile.open(fileobj=archive, mode="r:") as bundle:
            bundle.extractall(destination, filter="data")
    return destination


def _run_snapshot_scorer(root: Path, include_roots: tuple[str, ...]) -> dict[str, Any]:
    script = (
        "import json; from core.architecture_quality.scorer import score_codebase; "
        f"r=score_codebase('.', include_roots={include_roots!r}); "
        "print(json.dumps({'score':r.score,'metrics':r.metrics.to_dict()}))"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def _path_last_changes(
    root: Path,
    end: str,
    include_roots: tuple[str, ...],
) -> dict[str, str]:
    output = subprocess.run(
        [
            "git", "-c", "core.fsmonitor=false",
            "log",
            "--format=@@%H",
            "--name-only",
            end,
            "--",
            *include_roots,
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    current = ""
    changes: dict[str, str] = {}
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if line.startswith("@@"):
            current = line[2:]
        elif line and current:
            changes.setdefault(line, current)
    return changes


def _commit_identity(root: Path, revision: str) -> str:
    return subprocess.run(
        ["git", "-c", "core.fsmonitor=false", "rev-parse", f"{revision}^{{commit}}"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _load_legacy_baseline(
    root: Path,
    *,
    legacy_commit: str,
    baseline_path: Path,
) -> dict[str, Any]:
    commit = _commit_identity(root, legacy_commit)
    try:
        relative = baseline_path.resolve().relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError("legacy baseline path must be inside the repository") from exc
    try:
        encoded = subprocess.run(
            ["git", "-c", "core.fsmonitor=false", "show", f"{commit}:{relative}"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except subprocess.CalledProcessError as exc:
        raise ValueError(
            f"legacy baseline {relative} is absent from commit {commit}"
        ) from exc
    data = json.loads(encoded)
    if int(data.get("schema_version", 1)) != 1:
        raise ValueError("historical baseline source is not the expected schema-1 evidence")
    return data


def _require_ancestor(root: Path, ancestor: str, descendant: str) -> None:
    result = subprocess.run(
        ["git", "-c", "core.fsmonitor=false", "merge-base", "--is-ancestor", ancestor, descendant],
        cwd=root,
        check=False,
    )
    if result.returncode != 0:
        raise ValueError(f"{ancestor} is not an ancestor of {descendant}")


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


if __name__ == "__main__":
    raise SystemExit(main())
