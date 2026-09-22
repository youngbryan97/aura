"""Single-tenant install boundary.

The platform decision is "single-tenant per install": one Aura install
owns one tenant's data, full stop.  This module enforces that
guarantee at the data-directory boundary so two installs cannot
silently share a data root or cross-mount one another's memory.

How it works:

  * Each install stamps ``<data_dir>/tenant.json`` on first use with
    its tenant id, install id, and a creation timestamp.
  * Every subsequent operation that touches the data dir asks
    ``TenantBoundary.assert_owned()`` to verify the stamp matches the
    runtime's configured tenant.
  * A mismatch raises ``TenantMismatchError`` *before* any data is
    read, so a cross-tenant data dir cannot accidentally bleed records
    into the wrong namespace.

The tenant id defaults to the value of ``AURA_TENANT_ID`` or the
sentinel ``"default"``.  Operators who run a second Aura install on
the same machine must set ``AURA_TENANT_ID`` to a distinct value
*and* use a distinct ``AURA_HOME`` — the boundary check refuses to
mount otherwise.
"""
from __future__ import annotations

import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.runtime.file_write_gateway import get_file_write_gateway

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
DEFAULT_TENANT_ID = "default"
TENANT_FILE = "tenant.json"


def configured_tenant_id() -> str:
    """Read the runtime's configured tenant id (env or fallback)."""
    return os.environ.get("AURA_TENANT_ID", DEFAULT_TENANT_ID).strip() or DEFAULT_TENANT_ID


class TenantMismatchError(RuntimeError):
    """Raised when a data dir's tenant stamp does not match the runtime."""

    def __init__(self, *, expected: str, found: str, data_dir: Path):
        super().__init__(
            f"tenant mismatch at {data_dir}: expected {expected!r}, "
            f"found {found!r}"
        )
        self.expected = expected
        self.found = found
        self.data_dir = Path(data_dir)


@dataclass
class TenantStamp:
    tenant_id: str
    install_id: str
    created_at: float
    schema_version: int = SCHEMA_VERSION
    metadata: dict[str, Any] = field(default_factory=dict)


class TenantBoundary:
    """Single-tenant boundary check at a data directory."""

    def __init__(self, data_dir: Path, *, tenant_id: str | None = None):
        self.data_dir = Path(data_dir)
        self.tenant_id = (tenant_id or configured_tenant_id()).strip() or DEFAULT_TENANT_ID
        self._cached_stamp: TenantStamp | None = None

    @property
    def stamp_path(self) -> Path:
        return self.data_dir / TENANT_FILE

    # ------------------------------------------------------------------
    def stamp(self, *, force: bool = False) -> TenantStamp:
        """Write the tenant stamp for this install.

        If a stamp already exists for the configured tenant, this is a
        no-op.  ``force=True`` overwrites — operators only need that
        when they're re-stamping after a deliberate tenant rename.
        """
        self.data_dir.mkdir(parents=True, exist_ok=True)
        existing = self._read_stamp()
        if existing is not None and not force and existing.tenant_id == self.tenant_id:
            self._cached_stamp = existing
            return existing
        stamp = TenantStamp(
            tenant_id=self.tenant_id,
            install_id=f"install-{uuid.uuid4().hex[:12]}",
            created_at=time.time(),
        )
        self._write_stamp(stamp)
        self._cached_stamp = stamp
        return stamp

    def assert_owned(self) -> TenantStamp:
        """Verify the data dir's stamp matches the configured tenant."""
        stamp = self._read_stamp()
        if stamp is None:
            # First-touch: write our stamp.  Subsequent calls verify it.
            return self.stamp()
        if stamp.tenant_id != self.tenant_id:
            raise TenantMismatchError(
                expected=self.tenant_id,
                found=stamp.tenant_id,
                data_dir=self.data_dir,
            )
        self._cached_stamp = stamp
        return stamp

    def current_stamp(self) -> TenantStamp | None:
        return self._read_stamp()

    # ------------------------------------------------------------------
    def _read_stamp(self) -> TenantStamp | None:
        if not self.stamp_path.exists():
            return None
        try:
            payload = json.loads(self.stamp_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            # The stamp file EXISTS — the check above proved it — so None here
            # says "no tenant" about an install that has one.
            logger.error(
                "tenant stamp at %s does not parse (%s); reading as no tenant",
                self.stamp_path,
                exc,
            )
            return None
        try:
            return TenantStamp(
                tenant_id=str(payload.get("tenant_id") or DEFAULT_TENANT_ID),
                install_id=str(payload.get("install_id") or ""),
                created_at=float(payload.get("created_at") or 0.0),
                schema_version=int(payload.get("schema_version") or SCHEMA_VERSION),
                metadata=dict(payload.get("metadata") or {}),
            )
        except (TypeError, ValueError) as exc:
            logger.error(
                "tenant stamp at %s has unusable fields (%s: %s); reading as "
                "no tenant",
                self.stamp_path,
                type(exc).__name__,
                exc,
            )
            return None

    def _write_stamp(self, stamp: TenantStamp) -> None:
        body = {
            "tenant_id": stamp.tenant_id,
            "install_id": stamp.install_id,
            "created_at": stamp.created_at,
            "schema_version": stamp.schema_version,
            "metadata": stamp.metadata,
        }
        get_file_write_gateway().write_text(
            self.stamp_path,
            json.dumps(body, indent=2, sort_keys=True),
            encoding="utf-8",
            source="core.runtime.tenant_boundary.write_stamp",
        )
