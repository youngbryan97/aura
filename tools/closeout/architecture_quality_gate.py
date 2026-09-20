#!/usr/bin/env python3
"""Run Aura's deterministic architecture-quality gate."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.architecture_quality.attestation import (  # noqa: E402
    attest_payload,
    verify_attested_payload,
)
from core.architecture_quality.gate import ArchitectureQualityGate  # noqa: E402
from core.architecture_quality.scorer import score_codebase  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(ROOT), help="Repository root to score")
    parser.add_argument(
        "--include",
        action="append",
        default=None,
        help="Top-level package/root to include. Repeatable.",
    )
    parser.add_argument("--baseline", help="Baseline report JSON to compare against")
    parser.add_argument("--write-baseline", help="Write current report JSON to this path")
    parser.add_argument(
        "--migration-receipt",
        help="Signed migration receipt required when replacing an existing baseline",
    )
    parser.add_argument(
        "--signing-key",
        default="~/.aura/trust/architecture_quality_ed25519_private.pem",
        help="External Ed25519 private key used only when writing a baseline",
    )
    parser.add_argument(
        "--trust-root",
        default=str(ROOT / "config" / "trust" / "architecture_quality_ed25519_public.pem"),
        help="Pinned Ed25519 public key used to verify schema-2 baselines",
    )
    parser.add_argument("--min-score", type=float, default=None)
    parser.add_argument("--json", action="store_true", help="Emit full JSON")
    args = parser.parse_args()

    include_roots = tuple(args.include or ("core", "interface", "infrastructure", "slo", "tools"))
    root = Path(args.root).resolve()
    current = score_codebase(root, include_roots=include_roots)

    payload: dict[str, Any] = {"current": current.to_dict(), "passed": True, "reasons": []}

    if args.min_score is not None and current.score < args.min_score:
        payload["passed"] = False
        payload["reasons"].append(
            f"score {current.score:.1f} below required minimum {args.min_score:.1f}"
        )

    if args.baseline:
        try:
            baseline_data = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
            baseline = _report_from_baseline(
                baseline_data,
                trusted_public_key_pem=Path(args.trust_root).read_bytes(),
            )
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            payload["passed"] = False
            payload["reasons"].append(f"invalid architecture baseline: {exc}")
        else:
            result = ArchitectureQualityGate(root, include_roots=include_roots).evaluate_reports(
                baseline,
                current,
            )
            payload["baseline"] = baseline.to_dict()
            payload["passed"] = bool(payload["passed"] and result.passed)
            payload["reasons"].extend(result.reasons)

    if args.write_baseline:
        output_path = Path(args.write_baseline)
        try:
            migration_sha256 = _validated_migration_receipt(
                args.migration_receipt,
                current=current,
                root=root,
                trusted_public_key_pem=Path(args.trust_root).read_bytes(),
                required=output_path.exists(),
            )
            baseline_payload = _baseline_payload(
                current,
                signing_key_path=Path(args.signing_key).expanduser(),
                migration_receipt_sha256=migration_sha256,
            )
            _atomic_write_json(output_path, baseline_payload)
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            payload["passed"] = False
            payload["reasons"].append(f"architecture baseline write refused: {exc}")

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(current.summary())
        for reason in payload["reasons"]:
            print(f"- {reason}")

    return 0 if payload["passed"] else 1


def _report_from_baseline(
    data: dict[str, Any],
    *,
    trusted_public_key_pem: bytes,
):
    """Rehydrate the report subset needed for comparisons."""
    from core.architecture_quality.models import (
        SCHEMA_VERSION,
        ArchitectureQualityFinding,
        ArchitectureQualityMetrics,
        ArchitectureQualityReport,
        ModuleStructure,
    )

    schema_version = int(data.get("schema_version", 1))
    if schema_version >= 2:
        verify_attested_payload(
            data,
            digest_field="baseline_sha256",
            trusted_public_key_pem=trusted_public_key_pem,
        )

    metrics_data = data["metrics"]
    metrics = ArchitectureQualityMetrics(
        module_count=int(metrics_data["module_count"]),
        dependency_edges=int(metrics_data["dependency_edges"]),
        cycle_count=int(metrics_data["cycle_count"]),
        largest_cycle_size=int(metrics_data["largest_cycle_size"]),
        god_file_count=int(metrics_data["god_file_count"]),
        max_file_lines=int(metrics_data["max_file_lines"]),
        max_out_degree=int(metrics_data["max_out_degree"]),
        max_in_degree=int(metrics_data["max_in_degree"]),
        dependency_concentration_pct=float(metrics_data["dependency_concentration_pct"]),
        parse_error_count=int(metrics_data.get("parse_error_count", 0)),
        type_only_dependency_edges=int(metrics_data.get("type_only_dependency_edges", 0)),
        optional_dependency_edges=int(metrics_data.get("optional_dependency_edges", 0)),
        conditional_dependency_edges=int(metrics_data.get("conditional_dependency_edges", 0)),
        deferred_dependency_edges=int(metrics_data.get("deferred_dependency_edges", 0)),
        dynamic_dependency_edges=int(metrics_data.get("dynamic_dependency_edges", 0)),
        unresolved_dynamic_imports=int(metrics_data.get("unresolved_dynamic_imports", 0)),
        unresolved_local_imports=int(metrics_data.get("unresolved_local_imports", 0)),
        invalid_relative_imports=int(metrics_data.get("invalid_relative_imports", 0)),
        executable_dependency_edges=int(metrics_data.get("executable_dependency_edges", 0)),
        executable_cycle_count=int(metrics_data.get("executable_cycle_count", 0)),
        largest_executable_cycle_size=int(
            metrics_data.get("largest_executable_cycle_size", 0)
        ),
        max_code_lines=int(metrics_data.get("max_code_lines", 0)),
        max_complexity=int(metrics_data.get("max_complexity", 0)),
        max_symbol_count=int(metrics_data.get("max_symbol_count", 0)),
        architecture_debt=float(metrics_data.get("architecture_debt", 0.0)),
        cyclic_module_count=int(metrics_data.get("cyclic_module_count", 0)),
        executable_cyclic_module_count=int(
            metrics_data.get("executable_cyclic_module_count", 0)
        ),
    )
    findings = tuple(
        ArchitectureQualityFinding(
            severity=str(item["severity"]),
            code=str(item["code"]),
            message=str(item["message"]),
            path=item.get("path"),
            modules=tuple(item.get("modules") or ()),
            value=item.get("value"),
        )
        for item in data.get("findings", ())
    )
    structures = {
        str(path): ModuleStructure(
            source_lines=int(item["source_lines"]),
            code_lines=int(item["code_lines"]),
            comment_lines=int(item["comment_lines"]),
            statement_count=int(item["statement_count"]),
            symbol_count=int(item["symbol_count"]),
            branch_points=int(item["branch_points"]),
            max_nesting=int(item["max_nesting"]),
        )
        for path, item in data.get("module_structures", {}).items()
    }
    report = ArchitectureQualityReport(
        root=str(data.get("root", "")),
        include_roots=tuple(data.get("include_roots") or ()),
        exclude_parts=tuple(data.get("exclude_parts") or ()),
        god_file_threshold=int(data.get("god_file_threshold", 1500)),
        metrics=metrics,
        score=float(data["score"]),
        line_counts={str(key): int(value) for key, value in data.get("line_counts", {}).items()},
        module_to_path={str(key): str(value) for key, value in data.get("module_to_path", {}).items()},
        graph={
            str(key): tuple(str(item) for item in value)
            for key, value in data.get("graph", {}).items()
        },
        reverse_graph=_graph_from_json(data.get("reverse_graph", {})),
        cycles=tuple(tuple(str(item) for item in cycle) for cycle in data.get("cycles", ())),
        findings=findings,
        module_structures=structures,
        type_checking_graph=_graph_from_json(data.get("type_checking_graph", {})),
        optional_graph=_graph_from_json(data.get("optional_graph", {})),
        conditional_graph=_graph_from_json(data.get("conditional_graph", {})),
        deferred_graph=_graph_from_json(data.get("deferred_graph", {})),
        dynamic_graph=_graph_from_json(data.get("dynamic_graph", {})),
        executable_graph=_graph_from_json(data.get("executable_graph", {})),
        executable_cycles=tuple(
            tuple(str(item) for item in cycle)
            for cycle in data.get("executable_cycles", ())
        ),
        findings_complete=bool(data.get("findings_complete", False)),
        findings_omitted=int(data.get("findings_omitted", 0)),
        schema_version=int(data.get("schema_version", 1)),
    )
    if report.schema_version > SCHEMA_VERSION:
        raise ValueError(
            f"architecture baseline schema {report.schema_version} is newer than {SCHEMA_VERSION}"
        )
    return report


def _graph_from_json(data: dict[str, Any]) -> dict[str, tuple[str, ...]]:
    return {
        str(key): tuple(str(item) for item in value)
        for key, value in data.items()
    }


def _baseline_payload(
    report,
    *,
    signing_key_path: Path,
    migration_receipt_sha256: str | None = None,
) -> dict[str, Any]:
    """Persist and attest the compact evidence needed for comparison."""
    data = report.to_dict()
    payload = {
        "schema_version": data["schema_version"],
        "root": data["root"],
        "include_roots": data["include_roots"],
        "exclude_parts": data["exclude_parts"],
        "god_file_threshold": data["god_file_threshold"],
        "metrics": data["metrics"],
        "score": data["score"],
        "line_counts": data["line_counts"],
        "module_to_path": data["module_to_path"],
        "cycles": data["cycles"],
        "executable_cycles": data["executable_cycles"],
        "findings": data["findings"],
        "findings_complete": data["findings_complete"],
        "findings_omitted": data["findings_omitted"],
        "source_report_attestation_sha256": data["attestation_sha256"],
    }
    if migration_receipt_sha256 is not None:
        payload["migration_receipt_sha256"] = migration_receipt_sha256
    return attest_payload(
        payload,
        digest_field="baseline_sha256",
        signing_key_path=signing_key_path,
    )


def _validated_migration_receipt(
    receipt_path: str | None,
    *,
    current,
    root: Path,
    trusted_public_key_pem: bytes,
    required: bool,
) -> str | None:
    if not receipt_path:
        if required:
            raise ValueError("replacing an existing baseline requires --migration-receipt")
        return None
    receipt = json.loads(Path(receipt_path).read_text(encoding="utf-8"))
    if receipt.get("schema") != "aura.architecture_quality_baseline_migration.v1":
        raise ValueError("unsupported architecture baseline migration receipt")
    verify_attested_payload(
        receipt,
        digest_field="migration_sha256",
        trusted_public_key_pem=trusted_public_key_pem,
    )
    target = receipt.get("snapshots", {}).get("target", {})
    target_commit = target.get("commit")
    try:
        current_commit = subprocess.run(
            ["git", "-c", "core.fsmonitor=false", "rev-parse", "HEAD^{commit}"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError("migration-bound baseline root is not a Git checkout") from exc
    if target_commit != current_commit:
        raise ValueError(
            f"migration target commit differs from current source ({target_commit}->{current_commit})"
        )
    if target.get("report_attestation_sha256") != current.attestation_sha256:
        raise ValueError("migration target report differs from current architecture evidence")
    return str(receipt["migration_sha256"])


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    import os
    import tempfile

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
