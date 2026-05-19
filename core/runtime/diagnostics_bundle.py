
"""Diagnostics bundle for `aura doctor --bundle`.

Aura's runbooks reference ``aura doctor`` and ``aura status`` as the
first response to incidents.  Until now those commands either did not
exist or only checked Python version + writable data dir, so the
verification sections of every runbook fall back to "(when CLI ships)".

This module ships the missing CLI piece: a single command that
collects, in one tarball, everything an oncall needs to triage Aura
without standing next to her.  The bundle contains:

    health.json          health aggregator snapshot (subsystems + system metrics)
    config.json          redacted config (env values, secrets, PII stripped)
    metrics.json         metrics snapshot from registered services
    tasks.json           active task tracker snapshot
    models.json          model loader status
    memory.json          memory facade status
    gateway.json         service gateway readiness
    receipts.json        recent receipts (last N per kind)
    audit_chain/         exported tamper-evident chain (chain.jsonl + MANIFEST.txt)
    logs/                last few MB of recent log files
    bundle_manifest.json  high-level manifest with checksum and counts

Each collector is fail-safe: on error it writes an ``_error.txt`` next
to its file rather than aborting the whole bundle.  That way the
operator always gets *something* even when half the system is down,
which is precisely the situation in which they need the bundle.

Redaction is applied at the source: any field whose key matches a
sensitive pattern (token / secret / password / key / credential / auth)
is replaced with ``"[REDACTED]"`` before being written.
"""
from __future__ import annotations

import logging
logger = logging.getLogger("core.runtime.diagnostics_bundle")
import datetime
import heapq
import hashlib
import json
import os
import re
import shutil
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


SENSITIVE_KEY_PATTERNS = [
    re.compile(r"token", re.IGNORECASE),
    re.compile(r"secret", re.IGNORECASE),
    re.compile(r"password", re.IGNORECASE),
    re.compile(r"passwd", re.IGNORECASE),
    re.compile(r"api[_-]?key", re.IGNORECASE),
    re.compile(r"^key$", re.IGNORECASE),
    re.compile(r"credential", re.IGNORECASE),
    re.compile(r"auth", re.IGNORECASE),
    re.compile(r"private[_-]?key", re.IGNORECASE),
    re.compile(r"bearer", re.IGNORECASE),
]

# Patterns that look like high-entropy secrets within string values.
SENSITIVE_VALUE_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9]{20,}"),                   # OpenAI-style
    re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}"),          # Slack
    re.compile(r"ghp_[A-Za-z0-9]{20,}"),                  # GitHub PAT
    re.compile(r"AKIA[0-9A-Z]{16}"),                      # AWS access key id
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),    # PEM
    re.compile(r"eyJ[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+"),  # JWT
]

REDACTED = "[REDACTED]"


def redact_value(value: Any) -> Any:
    """Recursively scrub sensitive fields and high-entropy values."""
    if isinstance(value, dict):
        out: Dict[str, Any] = {}
        for k, v in value.items():
            if any(p.search(str(k)) for p in SENSITIVE_KEY_PATTERNS):
                out[k] = REDACTED
            else:
                out[k] = redact_value(v)
        return out
    if isinstance(value, list):
        return [redact_value(v) for v in value]
    if isinstance(value, str):
        for pat in SENSITIVE_VALUE_PATTERNS:
            if pat.search(value):
                return REDACTED
        return value
    return value


# ---------------------------------------------------------------------------
# collectors
# ---------------------------------------------------------------------------
def _safe_call(label: str, fn: Callable[[], Any]) -> Tuple[Any, Optional[str]]:
    try:
        return fn(), None
    except Exception as e:  # noqa: BLE001 - last-resort safety net
        logger.debug("Safe call failed on %s: %s", label, e)
        return None, f"{type(e).__name__}: {e}"


def collect_health() -> Dict[str, Any]:
    try:
        from core.health_endpoint import HealthAggregator
        agg = HealthAggregator()
        import asyncio

        return asyncio.run(agg.get_report())
    except Exception as e:  # noqa: BLE001
        logger.debug("Failed to collect health report: %s", e)
        # Fall back to a minimal snapshot so the bundle still has *something*.
        return {
            "_collector_error": f"{type(e).__name__}: {e}",
            "system": _basic_system_metrics(),
        }


def _basic_system_metrics() -> Dict[str, Any]:
    try:
        import psutil  # type: ignore

        proc = psutil.Process()
        return {
            "cpu_percent": psutil.cpu_percent(interval=0.0),
            "memory_percent": psutil.virtual_memory().percent,
            "process_rss_mb": proc.memory_info().rss / (1024 * 1024),
            "threads": proc.num_threads(),
            "pid": proc.pid,
        }
    except Exception as e:
        logger.debug("Failed to collect basic system metrics: %s", e)
        return {"available": False}


def collect_config_redacted() -> Dict[str, Any]:
    try:
        from core.config import config

        # config is typically a pydantic-style settings object; walk it.
        if hasattr(config, "model_dump"):
            raw = config.model_dump()
        elif hasattr(config, "dict"):
            raw = config.dict()
        else:
            raw = {k: getattr(config, k) for k in dir(config) if not k.startswith("_")}
    except Exception as e:  # noqa: BLE001
        raw = {"_collector_error": f"{type(e).__name__}: {e}"}
    return redact_value(raw)


def collect_metrics() -> Dict[str, Any]:
    try:
        from core.container import ServiceContainer

        out: Dict[str, Any] = {"system": _basic_system_metrics()}
        for name in ("metrics_collector", "telemetry", "metrics"):
            svc = ServiceContainer.get(name, default=None)
            if svc is not None and hasattr(svc, "snapshot"):
                try:
                    out[name] = redact_value(svc.snapshot())
                except Exception as e:  # noqa: BLE001
                    out[name] = {"_collector_error": str(e)}
        return out
    except Exception as e:  # noqa: BLE001
        logger.debug("Failed to collect metrics: %s", e)
        return {"_collector_error": f"{type(e).__name__}: {e}"}


def collect_tasks() -> Dict[str, Any]:
    try:
        from core.utils.task_tracker import get_task_tracker

        tracker = get_task_tracker()
        snapshot = []
        for t in getattr(tracker, "_tasks", []) or []:
            snapshot.append(
                {
                    "name": getattr(t, "get_name", lambda: "")(),
                    "done": getattr(t, "done", lambda: True)(),
                    "cancelled": getattr(t, "cancelled", lambda: False)(),
                }
            )
        return {"count": len(snapshot), "tasks": snapshot[:200]}
    except Exception as e:  # noqa: BLE001
        logger.debug("Failed to collect tasks: %s", e)
        return {"_collector_error": f"{type(e).__name__}: {e}"}


def collect_models() -> Dict[str, Any]:
    try:
        from core.container import ServiceContainer

        out: Dict[str, Any] = {}
        for name in ("model_loader", "model_runtime", "llm_router", "model_index"):
            svc = ServiceContainer.get(name, default=None)
            if svc is None:
                continue
            for method in ("status", "get_status", "snapshot", "to_dict"):
                if hasattr(svc, method):
                    try:
                        out[name] = redact_value(getattr(svc, method)())
                        break
                    except Exception as e:  # noqa: BLE001
                        out[name] = {"_collector_error": str(e)}
        return out
    except Exception as e:  # noqa: BLE001
        logger.debug("Failed to collect models: %s", e)
        return {"_collector_error": f"{type(e).__name__}: {e}"}


def collect_memory() -> Dict[str, Any]:
    try:
        from core.container import ServiceContainer

        out: Dict[str, Any] = {}
        for name in ("memory", "memory_facade"):
            svc = ServiceContainer.get(name, default=None)
            if svc is None:
                continue
            for method in ("status", "get_status", "stats", "snapshot"):
                if hasattr(svc, method):
                    try:
                        out[name] = redact_value(getattr(svc, method)())
                        break
                    except Exception as e:  # noqa: BLE001
                        out[name] = {"_collector_error": str(e)}
        return out
    except Exception as e:  # noqa: BLE001
        logger.debug("Failed to collect memory: %s", e)
        return {"_collector_error": f"{type(e).__name__}: {e}"}


def collect_gateway() -> Dict[str, Any]:
    try:
        from core.container import ServiceContainer

        out: Dict[str, Any] = {"registered": []}
        services = getattr(ServiceContainer, "_services", {}) or {}
        for name, desc in services.items():
            instance = getattr(desc, "instance", None)
            ready = instance is not None
            out["registered"].append({"name": name, "ready": ready})
        return out
    except Exception as e:  # noqa: BLE001
        logger.debug("Failed to collect gateway services: %s", e)
        return {"_collector_error": f"{type(e).__name__}: {e}"}


def collect_research_core() -> Dict[str, Any]:
    """Snapshot the SelfImprovingResearchCore via its dedicated collector."""
    try:
        from core.research_core.doctor import collect_research_core_status

        return collect_research_core_status()
    except Exception as e:  # noqa: BLE001
        logger.debug("Failed to collect research core status: %s", e)
        return {"available": False, "_collector_error": f"{type(e).__name__}: {e}"}


def collect_recent_receipts(per_kind_limit: int = 20) -> Dict[str, Any]:
    try:
        from core.runtime.receipts import _RECEIPT_CLASSES, get_receipt_store

        store = get_receipt_store()
        kinds: Dict[str, List[Dict[str, Any]]] = {}
        counts: Dict[str, int] = {kind: 0 for kind in _RECEIPT_CLASSES}
        for kind in _RECEIPT_CLASSES:
            kind_dir = store.root / kind
            if not kind_dir.exists():
                kinds[kind] = []
                continue
            files = []
            for path in kind_dir.glob("*.json"):
                try:
                    stat = path.stat()
                except OSError:
                    continue
                counts[kind] += 1
                files.append((stat.st_mtime, path))
            recent_files = heapq.nlargest(max(0, int(per_kind_limit)), files, key=lambda item: item[0])
            recent_payloads: List[Dict[str, Any]] = []
            for _mtime, path in sorted(recent_files, key=lambda item: item[0]):
                try:
                    from core.runtime.atomic_writer import read_json_envelope

                    env = read_json_envelope(path)
                    payload = env.get("payload") if isinstance(env, dict) else None
                    if isinstance(payload, dict):
                        payload = dict(payload)
                        payload.setdefault("kind", kind)
                        recent_payloads.append(redact_value(payload))
                except Exception as e:
                    logger.debug("Failed to read receipt at %s: %s", path, e)
                    continue
            kinds[kind] = recent_payloads
        return {"counts": counts, "recent": kinds}
    except Exception as e:  # noqa: BLE001
        logger.debug("Failed to collect recent receipts: %s", e)
        return {"_collector_error": f"{type(e).__name__}: {e}"}


def collect_audit_chain(dest_dir: Path) -> Dict[str, Any]:
    try:
        from core.runtime.receipts import get_receipt_store

        store = get_receipt_store()
        full_verify = os.environ.get("AURA_DOCTOR_FULL_AUDIT", "").strip().lower() in {"1", "true", "yes", "on"}
        chain = getattr(store, "_chain", None)
        if full_verify:
            info = store.export_chain(dest_dir)
            verify = store.verify_chain()
        else:
            dest_dir.mkdir(parents=True, exist_ok=True)
            if chain is not None:
                try:
                    if hasattr(chain, "flush"):
                        chain.flush()
                except Exception as _exc:
                    logger.debug("Suppressed %s in core.runtime.diagnostics_bundle: %s", type(_exc).__name__, _exc)
                tail_entries = max(1, int(os.environ.get("AURA_DOCTOR_AUDIT_TAIL_ENTRIES", "200")))
                chain_src = Path(getattr(chain, "path", ""))
                chain_tail = dest_dir / "chain_tail.jsonl"
                copied_entries = 0
                if chain_src.exists():
                    from collections import deque

                    tail = deque(maxlen=tail_entries)
                    with open(chain_src, "r", encoding="utf-8") as fh:
                        for line in fh:
                            if line.strip():
                                tail.append(line)
                    chain_tail.write_text("".join(tail), encoding="utf-8")
                    copied_entries = len(tail)
                manifest = (
                    "audit_chain_export\n"
                    "mode=tail\n"
                    f"length={chain.length()}\n"
                    f"head_hash={chain.head_hash()}\n"
                    f"tail_entries={copied_entries}\n"
                    "note=Set AURA_DOCTOR_FULL_AUDIT=1 for full chain export and body verification.\n"
                )
                manifest_path = dest_dir / "MANIFEST.txt"
                manifest_path.write_text(manifest, encoding="utf-8")
                info = {
                    "chain_path": str(chain_tail),
                    "manifest_path": str(manifest_path),
                    "length": chain.length(),
                    "head_hash": chain.head_hash(),
                    "mode": "tail",
                    "tail_entries": copied_entries,
                }
                verify = {
                    "ok": None,
                    "length": chain.length(),
                    "head_hash": chain.head_hash(),
                    "problems": [],
                    "link_verification": "skipped",
                    "body_verification": "skipped",
                    "note": "Set AURA_DOCTOR_FULL_AUDIT=1 to export and verify the full audit chain.",
                }
            else:
                info = {
                    "chain_path": "",
                    "manifest_path": "",
                    "length": 0,
                    "head_hash": "",
                    "mode": "unavailable",
                    "tail_entries": 0,
                }
                verify = {
                    "ok": False,
                    "length": 0,
                    "head_hash": "",
                    "problems": [{"reason": "chain not initialised"}],
                    "link_verification": "skipped",
                    "body_verification": "skipped",
                }
        return {"export": info, "verify": verify}
    except Exception as e:  # noqa: BLE001
        logger.debug("Failed to collect audit chain: %s", e)
        return {"_collector_error": f"{type(e).__name__}: {e}"}


def collect_logs(dest_dir: Path, max_total_bytes: int = 512 * 1024) -> Dict[str, Any]:
    """Copy the most recent log files up to a total byte cap.

    Logs commonly live under ``~/.aura_runtime/logs/`` or ``logs/``.  We
    pick whichever exists and copy newest files first until the cap.
    """
    candidates = [
        Path.home() / ".aura_runtime" / "logs",
        Path.cwd() / "logs",
        Path.home() / ".aura" / "logs",
    ]
    src = next((p for p in candidates if p.exists() and p.is_dir()), None)
    if src is None:
        return {"available": False, "reason": "no logs directory found"}
    dest_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(
        (p for p in src.rglob("*") if p.is_file()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    copied: List[Dict[str, Any]] = []
    total = 0
    for f in files:
        size = f.stat().st_size
        remaining = max_total_bytes - total
        if remaining <= 0:
            break
        rel = f.relative_to(src)
        dst = dest_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if size > remaining:
            try:
                with open(f, "rb") as in_fh, open(dst, "wb") as out_fh:
                    in_fh.seek(max(0, size - remaining))
                    out_fh.write(in_fh.read(remaining))
                copied.append({"src": str(rel), "size": remaining, "truncated_tail": True, "original_size": size})
                total += remaining
            except OSError:
                continue
            break
        shutil.copy2(f, dst)
        copied.append({"src": str(rel), "size": size, "truncated_tail": False})
        total += size
    return {"available": True, "source": str(src), "copied": copied, "bytes": total}


# ---------------------------------------------------------------------------
# bundle assembly
# ---------------------------------------------------------------------------
def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def build_bundle(
    *,
    output_path: Optional[Path] = None,
    workspace: Optional[Path] = None,
) -> Dict[str, Any]:
    """Build a diagnostics tarball.

    ``output_path`` defaults to ``~/.aura/diagnostics/aura-bundle-<ts>.tar.gz``.
    Returns a dict describing the bundle (path, byte size, sha256,
    included files, per-collector errors).
    """
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = (
        Path(output_path)
        if output_path is not None
        else Path.home() / ".aura" / "diagnostics" / f"aura-bundle-{ts}.tar.gz"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    workspace = (
        Path(workspace)
        if workspace is not None
        else Path(tempfile.mkdtemp(prefix="aura_bundle_"))
    )
    workspace.mkdir(parents=True, exist_ok=True)
    bundle_dir = workspace / f"aura-bundle-{ts}"
    bundle_dir.mkdir(parents=True, exist_ok=True)

    errors: Dict[str, str] = {}

    def _step(name: str, fn: Callable[[], Any], target: Path) -> None:
        payload, err = _safe_call(name, fn)
        if err is not None:
            errors[name] = err
            (target.with_suffix(".error.txt")).write_text(err, encoding="utf-8")
            return
        _write_json(target, payload)

    _step("health", collect_health, bundle_dir / "health.json")
    _step("config", collect_config_redacted, bundle_dir / "config.json")
    _step("metrics", collect_metrics, bundle_dir / "metrics.json")
    _step("tasks", collect_tasks, bundle_dir / "tasks.json")
    _step("models", collect_models, bundle_dir / "models.json")
    _step("memory", collect_memory, bundle_dir / "memory.json")
    _step("gateway", collect_gateway, bundle_dir / "gateway.json")
    _step("receipts", collect_recent_receipts, bundle_dir / "receipts.json")
    _step("research_core", collect_research_core, bundle_dir / "research_core.json")

    audit_dir = bundle_dir / "audit_chain"
    audit_dir.mkdir(parents=True, exist_ok=True)
    audit_info, audit_err = _safe_call("audit_chain", lambda: collect_audit_chain(audit_dir))
    if audit_err is not None:
        errors["audit_chain"] = audit_err
        (audit_dir / "_error.txt").write_text(audit_err, encoding="utf-8")
    else:
        _write_json(audit_dir / "info.json", audit_info)

    logs_dir = bundle_dir / "logs"
    log_info, log_err = _safe_call("logs", lambda: collect_logs(logs_dir))
    if log_err is not None:
        errors["logs"] = log_err
        (bundle_dir / "logs.error.txt").write_text(log_err, encoding="utf-8")
    else:
        _write_json(bundle_dir / "logs.json", log_info)

    files_list = sorted(
        str(p.relative_to(bundle_dir))
        for p in bundle_dir.rglob("*")
        if p.is_file()
    )
    # The manifest is being written after this listing, so include it
    # explicitly so an auditor can verify the bundle is self-describing.
    if "bundle_manifest.json" not in files_list:
        files_list.append("bundle_manifest.json")
        files_list.sort()
    manifest = {
        "schema_version": 1,
        "generated_at": ts,
        "platform": _platform_info(),
        "errors": errors,
        "files": files_list,
    }
    _write_json(bundle_dir / "bundle_manifest.json", manifest)

    # Tar it up.
    with tarfile.open(output_path, "w:gz") as tar:
        tar.add(bundle_dir, arcname=bundle_dir.name)

    sha = hashlib.sha256(output_path.read_bytes()).hexdigest()
    return {
        "ok": True,
        "path": str(output_path),
        "bytes": output_path.stat().st_size,
        "sha256": sha,
        "errors": errors,
        "file_count": len(manifest["files"]),
        "included": manifest["files"],
    }


def _platform_info() -> Dict[str, Any]:
    import platform
    import sys

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "node": platform.node(),
    }
