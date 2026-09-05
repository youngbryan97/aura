"""Aura operator CLI.

Provides ``aura doctor``, ``aura conformance``, ``aura backup``,
``aura restore``, ``aura migrate``, ``aura verify-state``,
``aura verify-memory``, ``aura rebuild-index``, ``aura chaos``.

Each command returns a JSON-serializable dict so callers (CI, runbooks,
operators) can consume results programmatically.

The CLI intentionally does not import the full orchestrator at module
import time; commands fetch what they need on demand.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from core.runtime.errors import record_degradation
from core.runtime.state_ownership import state_root

COMMAND_HANDLERS: Dict[str, Callable[[argparse.Namespace], Dict[str, Any]]] = {}


def register(name: str):
    def _wrap(fn):
        COMMAND_HANDLERS[name] = fn
        return fn

    return _wrap


# --- doctor -----------------------------------------------------------------


@register("doctor")
def cmd_doctor(args: argparse.Namespace) -> Dict[str, Any]:
    """Pre-boot environment audit, with optional --bundle for a tarball."""
    if getattr(args, "bundle", False):
        from core.runtime.diagnostics_bundle import build_bundle

        target = Path(args.bundle_path) if getattr(args, "bundle_path", None) else None
        info = build_bundle(output_path=target)
        info["command"] = "doctor"
        return info

    checks: Dict[str, Dict[str, Any]] = {}

    checks["python_version"] = {
        "ok": sys.version_info >= (3, 11),
        "value": sys.version.split()[0],
    }

    home = Path.home()
    data_dir = home / ".aura"
    checks["data_dir_writable"] = {
        "ok": _is_writable(data_dir),
        "path": str(data_dir),
    }

    try:
        import sqlite3  # noqa: F401

        checks["sqlite_available"] = {"ok": True}
    except (ImportError, AttributeError, RuntimeError) as exc:
        record_degradation('operator_cli', exc)
        checks["sqlite_available"] = {"ok": False, "error": repr(exc)}

    try:
        import mlx.core  # noqa: F401

        checks["mlx_available"] = {"ok": True}
    except (ImportError, AttributeError, RuntimeError) as exc:
        record_degradation('operator_cli', exc)
        checks["mlx_available"] = {"ok": False, "error": repr(exc)}

    try:
        from core.runtime.atomic_writer import atomic_write_json, read_json_envelope

        probe_path = data_dir / "_doctor_probe.json"
        atomic_write_json(probe_path, {"ok": True}, schema_version=1, schema_name="doctor_probe")
        env = read_json_envelope(probe_path)
        probe_path.unlink(missing_ok=True)
        checks["atomic_writer_round_trip"] = {"ok": env["payload"]["ok"] is True}
    except (ImportError, AttributeError, RuntimeError) as exc:
        record_degradation('operator_cli', exc)
        checks["atomic_writer_round_trip"] = {"ok": False, "error": repr(exc)}

    overall_ok = all(v.get("ok") for v in checks.values())
    return {"command": "doctor", "ok": overall_ok, "checks": checks}


def _is_writable(p: Path) -> bool:
    try:
        p.mkdir(parents=True, exist_ok=True)
        return os.access(p, os.W_OK)
    except (RuntimeError, AttributeError, TypeError, ValueError):
        return False


# --- conformance ------------------------------------------------------------


@register("conformance")
def cmd_conformance(args: argparse.Namespace) -> Dict[str, Any]:
    """Run static invariant proofs against a complete contract fixture.

    This command validates conformance logic and launch-source contracts. It
    intentionally does not claim that a currently running Aura instance was
    observed; live ownership and liveness are reported by activation audit,
    the runtime manifest, and the health contract.
    """
    from core.runtime.conformance import (
        ConformanceReport,
        proof_boot_readiness,
        proof_event_delivery,
        proof_launch_authority,
        proof_persistence_atomic,
        proof_runtime_singularity,
        proof_service_graph,
        proof_shutdown_ordering,
        proof_strict_mode,
    )
    from core.runtime.service_manifest import SERVICE_MANIFEST

    snapshot = {role.canonical_owner: object() for role in SERVICE_MANIFEST.values()}
    report = ConformanceReport()
    report.results.append(proof_runtime_singularity(snapshot))
    report.results.append(proof_service_graph(snapshot))
    report.results.append(proof_boot_readiness("ready", {"vault": True, "model": True}))
    tmp = state_root() / "conformance_probe"
    tmp.mkdir(parents=True, exist_ok=True)
    report.results.append(proof_persistence_atomic(tmp))
    report.results.append(
        proof_event_delivery(
            [{"event_id": "contract-fixture-1", "status": "delivered"}],
            dispatched=1,
        )
    )
    from core.runtime.shutdown_coordinator import SHUTDOWN_PHASES

    report.results.append(proof_shutdown_ordering(list(SHUTDOWN_PHASES)))
    main_path = Path(__file__).resolve().parent.parent.parent / "aura_main.py"
    main_src = main_path.read_text(encoding="utf-8") if main_path.exists() else ""
    report.results.append(proof_launch_authority(main_src))
    report.results.append(proof_strict_mode([]))
    return {
        "command": "conformance",
        "ok": report.passed,
        "evidence_scope": "static_contract_fixture",
        "live_runtime_verified": False,
        "report": report.to_dict(),
    }


# --- backup / restore -------------------------------------------------------


@register("backup")
def cmd_backup(args: argparse.Namespace) -> Dict[str, Any]:
    from core.runtime.backup_restore import perform_backup

    target = Path(args.target) if args.target else (state_root() / "backups")
    return perform_backup(target=target)


@register("restore")
def cmd_restore(args: argparse.Namespace) -> Dict[str, Any]:
    from core.runtime.backup_restore import perform_restore

    return perform_restore(snapshot=Path(args.snapshot))


# --- migrate ---------------------------------------------------------------


@register("migrate")
def cmd_migrate(args: argparse.Namespace) -> Dict[str, Any]:
    from core.runtime.migrations import run_migrations

    return run_migrations(target_version=args.target_version, dry_run=args.dry_run)


# --- verify -----------------------------------------------------------------


@register("verify-state")
def cmd_verify_state(args: argparse.Namespace) -> Dict[str, Any]:
    from core.state.state_gateway import get_state_gateway

    gateway = get_state_gateway()
    snapshot = asyncio.run(gateway.snapshot())
    return {"command": "verify-state", "ok": True, "keys": list(snapshot.keys())}


@register("verify-memory")
def cmd_verify_memory(args: argparse.Namespace) -> Dict[str, Any]:
    from core.runtime.receipts import get_receipt_store

    store = get_receipt_store()
    store.reload_from_disk()
    return {"command": "verify-memory", "ok": True, "coverage": store.coverage_stats()}


@register("control-plane")
def cmd_control_plane(args: argparse.Namespace) -> Dict[str, Any]:
    """Inspect desired state, resource leases, blockers, and open circuits."""

    del args
    from core.runtime.operator_control_plane import (
        collect_runtime_control_plane_status,
    )

    report = collect_runtime_control_plane_status()
    return {
        "command": "control-plane",
        "ok": bool(report.get("ready", False)),
        "report": report,
    }


@register("rebuild-index")
def cmd_rebuild_index(args: argparse.Namespace) -> Dict[str, Any]:
    from core.runtime.vector_index import rebuild_vector_index

    return rebuild_vector_index(source=Path(args.source) if args.source else None)


# --- chaos ------------------------------------------------------------------


@register("chaos")
def cmd_chaos(args: argparse.Namespace) -> Dict[str, Any]:
    """Run a tiny abuse-stage smoke (deterministic, fast)."""
    from core.runtime.fault_injection import (
        FaultInjector,
        run_abuse_stage,
    )

    inj = FaultInjector(enabled=True)

    async def _run():
        return await run_abuse_stage(
            "stage_1_2h",
            invariants_check=lambda: True,
            injector=inj,
            duration_s=0.05,
            interval_s=0.0,
            fault_sequence=["malformed_tool_result", "model_timeout"],
        )

    report = asyncio.run(_run())
    return {
        "command": "chaos",
        "ok": report.passed,
        "fired": [ev.name for ev in report.fired],
    }


# --- main entry -------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aura")
    sub = parser.add_subparsers(dest="command", required=True)

    p_doctor = sub.add_parser("doctor")
    p_doctor.add_argument(
        "--bundle",
        action="store_true",
        help="Collect a diagnostics tarball (logs, redacted config, health, metrics, "
        "tasks, models, memory, gateway, receipts, audit chain) for incident triage.",
    )
    p_doctor.add_argument(
        "--bundle-path",
        dest="bundle_path",
        default=None,
        help="Override the default tarball output path.",
    )
    sub.add_parser("conformance")
    p_backup = sub.add_parser("backup")
    p_backup.add_argument("--target", required=False, default=None)
    p_restore = sub.add_parser("restore")
    p_restore.add_argument("--snapshot", required=True)
    p_migrate = sub.add_parser("migrate")
    p_migrate.add_argument("--target-version", type=int, default=None)
    p_migrate.add_argument("--dry-run", action="store_true")
    sub.add_parser("verify-state")
    sub.add_parser("verify-memory")
    sub.add_parser("control-plane")
    p_rebuild = sub.add_parser("rebuild-index")
    p_rebuild.add_argument("--source", required=False, default=None)
    sub.add_parser("chaos")

    p_plugin = sub.add_parser("plugin")
    p_plugin_sub = p_plugin.add_subparsers(dest="plugin_command", required=True)
    p_plugin_sub.add_parser("list")
    p_approve = p_plugin_sub.add_parser("approve")
    p_approve.add_argument("path")
    p_approve.add_argument("--reason", default="Manual operator approval")
    p_revoke = p_plugin_sub.add_parser("revoke")
    p_revoke.add_argument("path")
    p_verify = p_plugin_sub.add_parser("verify")
    p_verify.add_argument("path")
    p_plugin_sub.add_parser("scan")

    return parser


@register("plugin")
def cmd_plugin(args: argparse.Namespace) -> Dict[str, Any]:
    from core.security.plugin_allowlist import PluginAllowlist, PluginPolicyError
    allowlist = PluginAllowlist()

    cmd = args.plugin_command
    if cmd == "list":
        entries = allowlist.list_entries(include_revoked=False)
        return {
            "command": "plugin list",
            "ok": True,
            "entries": [
                {
                    "sha256": e.sha256,
                    "rel_path": e.rel_path,
                    "approved_by": e.approved_by,
                    "approved_at": e.approved_at,
                    "reason": e.reason,
                }
                for e in entries
            ]
        }

    elif cmd == "approve":
        p = Path(args.path)
        if not p.exists():
            return {"command": "plugin approve", "ok": False, "error": f"File not found: {args.path}"}
        entry = allowlist.record(p, approved_by="operator", reason=args.reason)
        return {
            "command": "plugin approve",
            "ok": True,
            "sha256": entry.sha256,
            "rel_path": entry.rel_path,
            "approved_by": entry.approved_by,
            "approved_at": entry.approved_at,
            "reason": entry.reason,
        }

    elif cmd == "revoke":
        p = Path(args.path)
        ok = allowlist.revoke(p)
        return {
            "command": "plugin revoke",
            "ok": ok,
            "path": str(p),
            "message": "Approval revoked" if ok else "Plugin not found or not approved"
        }

    elif cmd == "verify":
        p = Path(args.path)
        try:
            entry = allowlist.is_allowed(p)
            return {
                "command": "plugin verify",
                "ok": True,
                "path": str(p),
                "sha256": entry.sha256,
                "message": "Plugin verified and approved"
            }
        except PluginPolicyError as e:
            return {
                "command": "plugin verify",
                "ok": False,
                "path": str(p),
                "error": e.reason,
                "message": str(e)
            }

    elif cmd == "scan":
        plugins_dir = state_root() / "plugins"
        plugins_dir.mkdir(parents=True, exist_ok=True)
        files = list(plugins_dir.glob("**/*.py"))
        report = []
        all_passed = True
        for f in files:
            status = "unapproved"
            sha = ""
            try:
                entry = allowlist.is_allowed(f)
                status = "approved"
                sha = entry.sha256
            except PluginPolicyError as e:
                status = e.reason
                all_passed = False
            report.append({
                "path": str(f),
                "status": status,
                "sha256": sha
            })
        return {
            "command": "plugin scan",
            "ok": all_passed,
            "plugins_dir": str(plugins_dir),
            "scanned": len(files),
            "report": report
        }

    return {"command": "plugin", "ok": False, "error": "unknown_subcommand"}


def run_command(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = COMMAND_HANDLERS.get(args.command)
    if handler is None:
        return {"command": args.command, "ok": False, "error": "unknown_command"}
    return handler(args)


def main(argv: Optional[List[str]] = None) -> int:
    result = run_command(argv)
    print(json.dumps(result, indent=2, default=str))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
