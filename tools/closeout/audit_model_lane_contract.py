#!/usr/bin/env python3
"""Fail when Aura's atomic model-lane ownership contract drifts."""

from __future__ import annotations

import argparse
import ast
import json
import sys
from collections.abc import Callable
from dataclasses import fields
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.runtime.model_lane_control import (  # noqa: E402
    LaneTransactionDecision,
    LaneTransactionState,
)
from tools.closeout.audit_model_load_ownership import run_audit as run_model_load_audit  # noqa: E402

_EXPECTED_STATES = {
    "reserved", "evicting", "ready", "committed", "refused", "cancelled", "expired",
}
_REQUIRED_DECISION_FIELDS = {
    "request_id", "transaction_id", "fencing_token", "admitted", "ready_to_spawn",
    "state", "reason", "owner_id", "model_path", "lane", "qos", "request_gb",
    "committed_gb", "reserved_gb", "budget_gb", "observation_source",
    "observation_scenario_id", "resource_observation_available", "evict_owner_ids",
    "evicted_owner_ids", "receipt_id", "replayed",
}
_REQUIRED_CALLS: dict[str, dict[str, set[str]]] = {
    "core/runtime/control_plane.py": {
        "ResourceAdmissionController.acquire": {
            "_pressure_block_reason", "_blocking_leases_locked",
            "_older_waiter_blocks_locked", "_emit_receipt_async", "on_preempt",
        },
        "ResourceAdmissionController.release": {
            "_emit_receipt_async", "_append_history_locked",
        },
        "ResourceAdmissionController.status": {
            "pressure_snapshot", "_expire_leases_locked",
        },
    },
    "core/runtime/model_lane_control.py": {
        "ModelLaneController.reserve": {"reconcile_expired_compensations", "to_thread"},
        "ModelLaneController.reserve_sync": {
            "_refresh_external_owners", "_capacity_totals", "interprocess_file_lock",
            "_persist_terminal_receipt",
        },
        "ModelLaneController.prepare": {
            "_assert_fence", "wait_for", "cancel", "_owner_process_tree_liveness",
            "_capacity_totals", "lane_budget_gb",
        },
        "ModelLaneController.commit_sync": {
            "_assert_fence", "_process_liveness", "interprocess_file_lock",
            "_persist_terminal_receipt",
        },
        "ModelLaneController.cancel": {
            "_mark_cancelled_sync", "reconcile_expired_compensations",
            "_persist_terminal_receipt",
        },
        "ModelLaneController.reconcile_expired_compensations": {
            "_claim_expired_compensation_sync", "_finish_expired_compensation_sync",
            "wait_for",
        },
        "ModelLaneController.heartbeat_owner_sync": {"_append_event", "_save_locked"},
        "ModelLaneController.owner_observations": {"_refresh_external_owners", "_prune_locked"},
        "ModelLaneController.snapshot": {
            "_refresh_external_owners", "_capacity_totals", "_persist_missing_terminal_receipts",
        },
        "discover_external_model_processes": {
            "estimate_model_job_footprint_gb", "lane_budget_gb", "getpgid", "getsid",
        },
    },
    "core/runtime/subprocess_gateway.py": {
        "_reserve_model_lane_process": {"prepare_model_lane_claim"},
        "_inferred_model_lane_claim": {"infer_model_process_claim"},
        "_declared_model_lane_claim": {"declared_model_process_claim"},
    },
    "core/brain/llm/mlx_client.py": {
        "_model_load_admission_context": {
            "get_runtime_control_plane", "get_model_lane_controller", "acquire",
            "reserve", "prepare", "commit", "cancel", "release",
            "clear_admission_backoff",
        },
        "MLXLocalClient._ensure_worker_alive": {
            "_model_load_admission_backoff_active",
            "_release_durable_model_lane_owner",
            "_model_load_admission_context",
            "_note_model_load_admission_denial",
        },
        "MLXLocalClient._note_model_load_admission_denial": {
            "_model_load_admission_backoff_seconds", "monotonic",
        },
    },
    "tools/live_resource_pressure_proof.py": {
        "_run_physical_lane_sequence": {
            "reserve_sync", "prepare", "commit_sync", "release_owner_sync",
        },
        "_run_reservation_race": {"reserve_sync", "cancel_sync"},
    },
}
_REQUIRED_SNAPSHOT_FIELDS = {
    "schema", "generation", "budget_gb", "observation_source",
    "observation_scenario_id", "committed_gb", "reserved_gb", "owners",
    "reservations", "events",
}
_REQUIRED_EXTERNAL_METADATA = {
    "externally_discovered", "model_identity_status", "command_sha256",
    "process_group_id", "process_session_id", "process_tree_escape",
    "registered_parent_owner_id",
}
_REQUIRED_LIVE_CHECKS = {
    "physical_lane_sequence",
    "required_eviction_before_candidate_load",
    "physical_lane_no_overcommit",
    "physical_lane_cold_gap_bounded",
    "concurrent_reservation_single_winner",
    "race_no_capacity_double_spend",
}


def _call_name(node: ast.Call) -> str:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return ""


def _referenced_symbols(node: ast.AST) -> set[str]:
    symbols = {
        item.id for item in ast.walk(node) if isinstance(item, ast.Name)
    }
    symbols.update(
        item.attr for item in ast.walk(node) if isinstance(item, ast.Attribute)
    )
    symbols.update(
        _call_name(item) for item in ast.walk(node) if isinstance(item, ast.Call)
    )
    symbols.discard("")
    return symbols


def _inherited_methods(node: ast.ClassDef, home: Path) -> dict[str, ast.AST]:
    """A class's methods that live in a mixin module beside its own file.

    `MLXLocalClient._ensure_worker_alive` is what this audit means, and it was
    read as "a def of that name in mlx_client.py". When the worker lifecycle
    was lifted into `mlx_client_worker_lifecycle.py` the method did not move
    away from the class — the class inherits it — but the audit reported the
    lane contract as missing. A contract on a class is a contract on what the
    class DOES, so the bases count, and a lift names its module after the one
    it came from.
    """
    bases = {base.id for base in node.bases if isinstance(base, ast.Name)}
    if not bases or home.suffix != ".py":
        return {}
    found: dict[str, ast.AST] = {}
    for sibling in sorted(home.parent.glob(f"{home.stem}_*.py")):
        try:
            tree = ast.parse(sibling.read_text(encoding="utf-8"), filename=str(sibling))
        except (OSError, SyntaxError, UnicodeError):  # pragma: no cover - unreadable sibling
            continue
        for item in tree.body:
            if not isinstance(item, ast.ClassDef) or item.name not in bases:
                continue
            for child in item.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    found.setdefault(child.name, child)
    return found


def _qualified_functions(tree: ast.Module, home: Path | None = None) -> dict[str, ast.AST]:
    functions: dict[str, ast.AST] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions[node.name] = node
        elif isinstance(node, ast.ClassDef):
            own = {
                child.name: child
                for child in node.body
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
            inherited = _inherited_methods(node, home) if home is not None else {}
            for name, child in {**inherited, **own}.items():
                functions[f"{node.name}.{name}"] = child
    return functions


def _literal_strings(node: ast.AST) -> set[str]:
    return {
        item.value
        for item in ast.walk(node)
        if isinstance(item, ast.Constant) and isinstance(item.value, str)
    }


def _returned_dict_keys(node: ast.AST) -> set[str]:
    keys: set[str] = set()
    for item in ast.walk(node):
        if not isinstance(item, ast.Return) or not isinstance(item.value, ast.Dict):
            continue
        for key in item.value.keys:
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                keys.add(key.value)
    return keys


def audit(
    root: Path,
    *,
    ownership_runner: Callable[..., dict[str, Any]] = run_model_load_audit,
) -> dict[str, Any]:
    issues: list[str] = []
    checked_functions = 0
    parsed: dict[str, dict[str, ast.AST]] = {}

    states = {state.value for state in LaneTransactionState}
    if states != _EXPECTED_STATES:
        issues.append(
            f"transaction states drifted: expected={sorted(_EXPECTED_STATES)!r} "
            f"actual={sorted(states)!r}"
        )
    decision_fields = {item.name for item in fields(LaneTransactionDecision)}
    missing_decision_fields = sorted(_REQUIRED_DECISION_FIELDS - decision_fields)
    if missing_decision_fields:
        issues.append(f"transaction decision lost fields {missing_decision_fields!r}")

    for relative, contracts in _REQUIRED_CALLS.items():
        path = root / relative
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, SyntaxError, UnicodeError) as exc:
            issues.append(f"cannot parse {relative}: {type(exc).__name__}: {exc}")
            continue
        functions = _qualified_functions(tree, path)
        parsed[relative] = functions
        for qualified_name, required_calls in contracts.items():
            function = functions.get(qualified_name)
            if function is None:
                issues.append(f"missing lane contract function: {relative}:{qualified_name}")
                continue
            checked_functions += 1
            missing_calls = sorted(required_calls - _referenced_symbols(function))
            if missing_calls:
                issues.append(f"{relative}:{qualified_name} lost calls {missing_calls!r}")

    lane_functions = parsed.get("core/runtime/model_lane_control.py", {})
    snapshot = lane_functions.get("ModelLaneController.snapshot")
    if snapshot is not None:
        missing = sorted(_REQUIRED_SNAPSHOT_FIELDS - _returned_dict_keys(snapshot))
        if missing:
            issues.append(f"lane snapshot lost fields {missing!r}")
    discovery = lane_functions.get("discover_external_model_processes")
    if discovery is not None:
        missing = sorted(_REQUIRED_EXTERNAL_METADATA - _literal_strings(discovery))
        if missing:
            issues.append(f"external owner accounting lost metadata {missing!r}")
    live_run = parsed.get("tools/live_resource_pressure_proof.py", {}).get("run_proof")
    if live_run is None:
        issues.append("bounded live pressure proof entrypoint is missing")
    else:
        missing = sorted(_REQUIRED_LIVE_CHECKS - _literal_strings(live_run))
        if missing:
            issues.append(f"bounded live pressure proof lost checks {missing!r}")

    try:
        ownership = ownership_runner(
            root=root,
            inventory_path=root / "config" / "model_load_ownership.json",
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        ownership = {}
        issues.append(f"model load ownership audit failed: {type(exc).__name__}: {exc}")
    if ownership.get("passed") is not True:
        issues.append("model load ownership audit did not pass")

    return {
        "schema": "aura.closeout.model_lane_contract_audit.v1",
        "passed": not issues,
        "checked_functions": checked_functions,
        "transaction_states": sorted(states),
        "decision_fields": sorted(decision_fields),
        "snapshot_fields": sorted(_REQUIRED_SNAPSHOT_FIELDS),
        "external_owner_metadata": sorted(_REQUIRED_EXTERNAL_METADATA),
        "bounded_live_checks": sorted(_REQUIRED_LIVE_CHECKS),
        "unified_admission_contract": {
            "policy_owner": "ResourceAdmissionController",
            "durable_capacity_owner": "ModelLaneController",
            "production_transaction": "_model_load_admission_context",
            "anti_thrash_owner": "MLXLocalClient._ensure_worker_alive",
        },
        "model_load_ownership": {
            "passed": ownership.get("passed") is True,
            "inventory_entries": ownership.get("inventory_entries"),
            "owned_paths": ownership.get("owned_paths"),
            "load_references": ownership.get("load_references"),
            "findings": ownership.get("findings", []),
        },
        "issues": issues,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = audit(args.root.resolve())
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        status = "passed" if report["passed"] else "failed"
        print(
            f"Model lane contract audit {status}: {report['checked_functions']} functions, "
            f"{report['model_load_ownership']['owned_paths']} owned load paths"
        )
        for issue in report["issues"]:
            print(f"- {issue}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
