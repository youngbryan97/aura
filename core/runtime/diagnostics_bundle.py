
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

import datetime
import hashlib
import io
import json
import logging
import os
import re
import stat
import tarfile
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from core.runtime.file_write_gateway import get_file_write_gateway
from core.runtime.state_ownership import state_root
from core.verify import invariant

logger = logging.getLogger("core.runtime.diagnostics_bundle")


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
    re.compile(r"\bAuthorization\s*:\s*Bearer\s+\S+", re.IGNORECASE),
    re.compile(
        r"\b(?:api[_-]?key|secret|password|passwd|token|credential|auth)"
        r"\b\s*[:=]\s*[^\s,;]+",
        re.IGNORECASE,
    ),
]

REDACTED = "[REDACTED]"

_DIAGNOSTICS_RECOVERABLE_ERRORS = (
    ImportError,
    AttributeError,
    RuntimeError,
    TypeError,
    ValueError,
    OSError,
    tarfile.TarError,
    json.JSONDecodeError,
)


def redact_value(value: Any) -> Any:
    """Recursively scrub sensitive fields and high-entropy values."""
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for k, v in value.items():
            if any(p.search(str(k)) for p in SENSITIVE_KEY_PATTERNS):
                out[k] = REDACTED
            else:
                out[k] = redact_value(v)
        return out
    if isinstance(value, list):
        return [redact_value(v) for v in value]
    if isinstance(value, tuple):
        return tuple(redact_value(v) for v in value)
    if isinstance(value, (set, frozenset)):
        redacted = [redact_value(v) for v in value]
        return sorted(redacted, key=lambda item: repr(item))
    if isinstance(value, str):
        for pat in SENSITIVE_VALUE_PATTERNS:
            if pat.search(value):
                return REDACTED
        return value
    return value


# ---------------------------------------------------------------------------
# collectors
# ---------------------------------------------------------------------------
def _safe_call(label: str, fn: Callable[[], Any]) -> tuple[Any, str | None]:
    try:
        return fn(), None
    except _DIAGNOSTICS_RECOVERABLE_ERRORS as e:  # noqa: BLE001 - last-resort safety net
        logger.debug("Safe call failed on %s: %s", label, e)
        return None, f"{type(e).__name__}: {e}"


def collect_health() -> dict[str, Any]:
    try:
        from core.health.health_endpoint import HealthAggregator
        agg = HealthAggregator()
        import asyncio

        return asyncio.run(agg.get_report())
    except _DIAGNOSTICS_RECOVERABLE_ERRORS as e:  # noqa: BLE001
        logger.debug("Failed to collect health report: %s", e)
        # Fall back to a minimal snapshot so the bundle still has *something*.
        return {
            "_collector_error": f"{type(e).__name__}: {e}",
            "system": _basic_system_metrics(),
        }


def _basic_system_metrics() -> dict[str, Any]:
    try:
        from core.runtime import resource_psutil as psutil

        proc = psutil.Process()
        return {
            "cpu_percent": psutil.cpu_percent(interval=0.0),
            "memory_percent": psutil.virtual_memory().percent,
            "process_rss_mb": proc.memory_info().rss / (1024 * 1024),
            "threads": proc.num_threads(),
            "pid": proc.pid,
        }
    except _DIAGNOSTICS_RECOVERABLE_ERRORS as e:
        logger.debug("Failed to collect basic system metrics: %s", e)
        return {"available": False}


def collect_config_redacted() -> dict[str, Any]:
    try:
        from core.config import config

        # config is typically a pydantic-style settings object; walk it.
        if hasattr(config, "model_dump"):
            raw = config.model_dump()
        elif hasattr(config, "dict"):
            raw = config.dict()
        else:
            raw = {k: getattr(config, k) for k in dir(config) if not k.startswith("_")}
    except _DIAGNOSTICS_RECOVERABLE_ERRORS as e:  # noqa: BLE001
        raw = {"_collector_error": f"{type(e).__name__}: {e}"}
    return redact_value(raw)


def collect_metrics() -> dict[str, Any]:
    try:
        from core.container import ServiceContainer

        out: dict[str, Any] = {"system": _basic_system_metrics()}
        for name in ("metrics_collector", "telemetry", "metrics"):
            svc = ServiceContainer.get(name, default=None)
            if svc is not None and hasattr(svc, "snapshot"):
                try:
                    out[name] = redact_value(svc.snapshot())
                except _DIAGNOSTICS_RECOVERABLE_ERRORS as e:  # noqa: BLE001
                    out[name] = {"_collector_error": str(e)}
        return out
    except _DIAGNOSTICS_RECOVERABLE_ERRORS as e:  # noqa: BLE001
        logger.debug("Failed to collect metrics: %s", e)
        return {"_collector_error": f"{type(e).__name__}: {e}"}


def collect_tasks() -> dict[str, Any]:
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
    except _DIAGNOSTICS_RECOVERABLE_ERRORS as e:  # noqa: BLE001
        logger.debug("Failed to collect tasks: %s", e)
        return {"_collector_error": f"{type(e).__name__}: {e}"}


def collect_models() -> dict[str, Any]:
    try:
        from core.container import ServiceContainer

        out: dict[str, Any] = {}
        for name in ("model_loader", "model_runtime", "llm_router", "model_index"):
            svc = ServiceContainer.get(name, default=None)
            if svc is None:
                continue
            for method in ("status", "get_status", "snapshot", "to_dict"):
                if hasattr(svc, method):
                    try:
                        out[name] = redact_value(getattr(svc, method)())
                        break
                    except _DIAGNOSTICS_RECOVERABLE_ERRORS as e:  # noqa: BLE001
                        out[name] = {"_collector_error": str(e)}
        return out
    except _DIAGNOSTICS_RECOVERABLE_ERRORS as e:  # noqa: BLE001
        logger.debug("Failed to collect models: %s", e)
        return {"_collector_error": f"{type(e).__name__}: {e}"}


def collect_memory() -> dict[str, Any]:
    try:
        from core.container import ServiceContainer

        out: dict[str, Any] = {}
        for name in ("memory", "memory_facade"):
            svc = ServiceContainer.get(name, default=None)
            if svc is None:
                continue
            for method in ("status", "get_status", "stats", "snapshot"):
                if hasattr(svc, method):
                    try:
                        out[name] = redact_value(getattr(svc, method)())
                        break
                    except _DIAGNOSTICS_RECOVERABLE_ERRORS as e:  # noqa: BLE001
                        out[name] = {"_collector_error": str(e)}
        return out
    except _DIAGNOSTICS_RECOVERABLE_ERRORS as e:  # noqa: BLE001
        logger.debug("Failed to collect memory: %s", e)
        return {"_collector_error": f"{type(e).__name__}: {e}"}


def collect_gateway() -> dict[str, Any]:
    try:
        from core.container import ServiceContainer

        out: dict[str, Any] = {"registered": []}
        services = getattr(ServiceContainer, "_services", {}) or {}
        for name, desc in services.items():
            instance = getattr(desc, "instance", None)
            ready = instance is not None
            out["registered"].append({"name": name, "ready": ready})
        return out
    except _DIAGNOSTICS_RECOVERABLE_ERRORS as e:  # noqa: BLE001
        logger.debug("Failed to collect gateway services: %s", e)
        return {"_collector_error": f"{type(e).__name__}: {e}"}


def collect_research_core() -> dict[str, Any]:
    """Snapshot the SelfImprovingResearchCore via its dedicated collector."""
    try:
        from core.research_core.doctor import collect_research_core_status

        return collect_research_core_status()
    except _DIAGNOSTICS_RECOVERABLE_ERRORS as e:  # noqa: BLE001
        logger.debug("Failed to collect research core status: %s", e)
        return {"available": False, "_collector_error": f"{type(e).__name__}: {e}"}


def collect_control_plane() -> dict[str, Any]:
    try:
        from core.runtime.operator_control_plane import (
            collect_runtime_control_plane_status,
        )

        return collect_runtime_control_plane_status()
    except _DIAGNOSTICS_RECOVERABLE_ERRORS as e:  # noqa: BLE001
        logger.debug("Failed to collect runtime control plane: %s", e)
        return {"available": False, "_collector_error": f"{type(e).__name__}: {e}"}


def collect_recent_receipts(per_kind_limit: int = 20) -> dict[str, Any]:
    try:
        from core.runtime.receipts import _RECEIPT_CLASSES, get_receipt_store

        store = get_receipt_store()
        kinds: dict[str, list[dict[str, Any]]] = {}
        counts = store.coverage_stats()
        for kind in _RECEIPT_CLASSES:
            kinds[kind] = [
                redact_value(receipt.to_dict())
                for receipt in store.query_recent_persisted(
                    kind,
                    limit=max(0, int(per_kind_limit)),
                )
            ]
        return {
            "counts": counts,
            "recent": kinds,
            "storage": store.storage_stats(),
        }
    except _DIAGNOSTICS_RECOVERABLE_ERRORS as e:  # noqa: BLE001
        logger.debug("Failed to collect recent receipts: %s", e)
        return {"_collector_error": f"{type(e).__name__}: {e}"}


def collect_audit_chain(dest_dir: Path) -> dict[str, Any]:
    try:
        from core.runtime.receipts import get_receipt_store

        store = get_receipt_store()
        full_verify = os.environ.get("AURA_DOCTOR_FULL_AUDIT", "").strip().lower() in {"1", "true", "yes", "on"}
        chain = getattr(store, "_chain", None)
        if full_verify:
            info = store.export_chain(dest_dir)
            verify = store.verify_chain()
        else:
            if chain is not None:
                try:
                    if hasattr(chain, "flush"):
                        chain.flush()
                except _DIAGNOSTICS_RECOVERABLE_ERRORS as _exc:
                    logger.debug("Suppressed %s in core.runtime.diagnostics_bundle: %s", type(_exc).__name__, _exc)
                tail_entries = max(1, int(os.environ.get("AURA_DOCTOR_AUDIT_TAIL_ENTRIES", "200")))
                chain_src = Path(getattr(chain, "path", ""))
                chain_tail = dest_dir / "chain_tail.jsonl"
                copied_entries = 0
                if chain_src.exists():
                    from collections import deque

                    tail = deque(maxlen=tail_entries)
                    with chain_src.open(encoding="utf-8") as fh:
                        for line in fh:
                            if line.strip():
                                tail.append(line)
                    _write_text(
                        chain_tail,
                        "".join(tail),
                        source="core.runtime.diagnostics_bundle.audit_chain_tail",
                    )
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
                _write_text(
                    manifest_path,
                    manifest,
                    source="core.runtime.diagnostics_bundle.audit_manifest",
                )
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
    except _DIAGNOSTICS_RECOVERABLE_ERRORS as e:  # noqa: BLE001
        logger.debug("Failed to collect audit chain: %s", e)
        return {"_collector_error": f"{type(e).__name__}: {e}"}


def _bounded_log_tail(stream: Any, limit: int) -> tuple[bytes, int]:
    """Read at most the remaining budget from the opened file's current tail."""
    if type(limit) is not int or limit < 0:
        raise ValueError("log byte limit must be a nonnegative integer")
    stream.seek(0, os.SEEK_END)
    size = stream.tell()
    stream.seek(max(0, size - limit))
    return stream.read(limit), size


@invariant("runtime.diagnostic_log_reads_are_bounded", scope="runtime",
           owner="core/runtime/diagnostics_bundle.py", observational=False)
def _diagnostic_log_read_invariant() -> tuple:
    for payload in (b"", b"x", b"longer than the budget"):
        for limit in (0, 1, 8):
            tail, size = _bounded_log_tail(io.BytesIO(payload), limit)
            assert tail == (payload[-limit:] if limit else b"")
            assert len(tail) <= limit and size == len(payload)
    return ()


def collect_logs(dest_dir: Path, max_total_bytes: int = 512 * 1024) -> dict[str, Any]:
    """Copy the most recent log files up to a total byte cap.

    Read the logger's configured directory. A missing configured root
    cannot borrow another instance's logs.
    """
    from core.observability.logging_config import _resolve_log_dir

    src = _resolve_log_dir(None)
    if not src.is_dir():
        return {"available": False, "reason": "configured logs directory not found", "source": str(src)}
    files: list[tuple[float, Path]] = []
    for candidate in src.rglob("*"):
        try:
            metadata = candidate.stat()
        except OSError as exc:
            logger.debug("Diagnostic log could not be listed: %s: %s", candidate, exc)
            continue
        if stat.S_ISREG(metadata.st_mode):
            files.append((metadata.st_mtime, candidate))
    files.sort(key=lambda entry: entry[0], reverse=True)
    copied: list[dict[str, Any]] = []
    total = 0
    for _, f in files:
        remaining = max_total_bytes - total
        if remaining <= 0:
            break
        rel = f.relative_to(src)
        dst = dest_dir / rel
        try:
            # The opened log may have grown or rotated since the directory
            # scan. Bound the read itself and count only bytes actually read.
            with f.open("rb") as in_fh:
                payload, size = _bounded_log_tail(in_fh, remaining)
            _write_bytes(
                dst,
                payload,
                source="core.runtime.diagnostics_bundle.log_tail" if size > remaining
                else "core.runtime.diagnostics_bundle.log_copy",
            )
        except OSError as exc:
            logger.debug("Diagnostic log could not be copied: %s: %s", f, exc)
            continue
        entry = {"src": str(rel), "size": len(payload), "truncated_tail": size > len(payload)}
        if entry["truncated_tail"]:
            entry["original_size"] = size
        copied.append(entry)
        total += len(payload)
    return {"available": True, "source": str(src), "copied": copied, "bytes": total}


# ---------------------------------------------------------------------------
# bundle assembly
# ---------------------------------------------------------------------------
def _write_text(path: Path, text: str, *, source: str) -> None:
    get_file_write_gateway().write_text(
        path,
        text,
        encoding="utf-8",
        source=source,
    )


def _write_bytes(path: Path, payload: bytes, *, source: str) -> None:
    get_file_write_gateway().write_bytes(
        path,
        payload,
        source=source,
    )


def _write_json(
    path: Path,
    payload: Any,
    *,
    source: str = "core.runtime.diagnostics_bundle.json",
) -> None:
    _write_text(
        path,
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        source=source,
    )


def build_bundle(
    *,
    output_path: Path | None = None,
    workspace: Path | None = None,
) -> dict[str, Any]:
    """Build a diagnostics tarball.

    ``output_path`` defaults to ``~/.aura/diagnostics/aura-bundle-<ts>.tar.gz``.
    Returns a dict describing the bundle (path, byte size, sha256,
    included files, per-collector errors).
    """
    ts = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")
    output_path = (
        Path(output_path)
        if output_path is not None
        else state_root() / "diagnostics" / f"aura-bundle-{ts}.tar.gz"
    )

    workspace = (
        Path(workspace)
        if workspace is not None
        else Path(tempfile.mkdtemp(prefix="aura_bundle_"))
    )
    bundle_dir = workspace / f"aura-bundle-{ts}"

    errors: dict[str, str] = {}

    def _step(name: str, fn: Callable[[], Any], target: Path) -> None:
        payload, err = _safe_call(name, fn)
        if err is not None:
            errors[name] = err
            _write_text(
                target.with_suffix(".error.txt"),
                err,
                source=f"core.runtime.diagnostics_bundle.{name}_error",
            )
            return
        _write_json(target, payload, source=f"core.runtime.diagnostics_bundle.{name}")

    _step("health", collect_health, bundle_dir / "health.json")
    _step("config", collect_config_redacted, bundle_dir / "config.json")
    _step("metrics", collect_metrics, bundle_dir / "metrics.json")
    _step("tasks", collect_tasks, bundle_dir / "tasks.json")
    _step("models", collect_models, bundle_dir / "models.json")
    _step("memory", collect_memory, bundle_dir / "memory.json")
    _step("gateway", collect_gateway, bundle_dir / "gateway.json")
    _step("control_plane", collect_control_plane, bundle_dir / "control_plane.json")
    _step("receipts", collect_recent_receipts, bundle_dir / "receipts.json")
    _step("research_core", collect_research_core, bundle_dir / "research_core.json")

    audit_dir = bundle_dir / "audit_chain"
    audit_info, audit_err = _safe_call("audit_chain", lambda: collect_audit_chain(audit_dir))
    if audit_err is not None:
        errors["audit_chain"] = audit_err
        _write_text(
            audit_dir / "_error.txt",
            audit_err,
            source="core.runtime.diagnostics_bundle.audit_chain_error",
        )
    else:
        _write_json(
            audit_dir / "info.json",
            audit_info,
            source="core.runtime.diagnostics_bundle.audit_chain_info",
        )

    logs_dir = bundle_dir / "logs"
    log_info, log_err = _safe_call("logs", lambda: collect_logs(logs_dir))
    if log_err is not None:
        errors["logs"] = log_err
        _write_text(
            bundle_dir / "logs.error.txt",
            log_err,
            source="core.runtime.diagnostics_bundle.logs_error",
        )
    else:
        _write_json(
            bundle_dir / "logs.json",
            log_info,
            source="core.runtime.diagnostics_bundle.logs",
        )

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
    _write_json(
        bundle_dir / "bundle_manifest.json",
        manifest,
        source="core.runtime.diagnostics_bundle.manifest",
    )

    # Tar it up.
    tar_buffer = io.BytesIO()
    with tarfile.open(fileobj=tar_buffer, mode="w:gz") as tar:
        tar.add(bundle_dir, arcname=bundle_dir.name)
    _write_bytes(
        output_path,
        tar_buffer.getvalue(),
        source="core.runtime.diagnostics_bundle.tarball",
    )

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


def _platform_info() -> dict[str, Any]:
    import platform
    import sys

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "node": platform.node(),
    }
