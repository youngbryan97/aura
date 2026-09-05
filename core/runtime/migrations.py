"""Schema migration framework.

Each persistent record is written through the AtomicWriter with a
schema_version envelope. This module declares migration steps that
upgrade ``schema_version=N-1`` records to ``schema_version=N`` and keep
a migration log so partial migrations can resume.
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from core.runtime.atomic_writer import atomic_write_json, read_json_envelope
from core.runtime.errors import record_degradation
from core.runtime.state_ownership import state_root

logger = logging.getLogger("Aura.Migrations")

CURRENT_SCHEMA_VERSION = 1


@dataclass
class MigrationStep:
    from_version: int
    to_version: int
    transform: Callable[[dict[str, Any]], dict[str, Any]]
    description: str = ""


_MIGRATION_REGISTRY: list[MigrationStep] = []


def register_migration(step: MigrationStep) -> None:
    _MIGRATION_REGISTRY.append(step)


def list_migrations() -> list[MigrationStep]:
    return list(_MIGRATION_REGISTRY)


def migrate_payload(payload: dict[str, Any], current_version: int, target_version: int) -> dict[str, Any]:
    if current_version > target_version:
        raise RuntimeError(
            f"refusing to downgrade schema from {current_version} to {target_version}"
        )
    while current_version < target_version:
        step = next(
            (s for s in _MIGRATION_REGISTRY if s.from_version == current_version),
            None,
        )
        if step is None:
            raise RuntimeError(
                f"no migration step from schema_version={current_version}"
            )
        payload = step.transform(payload)
        current_version = step.to_version
    return payload


def run_migrations(*, target_version: int | None = None, dry_run: bool = False) -> dict[str, Any]:
    target = target_version or CURRENT_SCHEMA_VERSION
    home = state_root()
    candidate_dirs = [home / "state", home / "memory", home / "workflows", home / "receipts"]
    migrated: list[str] = []
    skipped: list[str] = []
    failed: list[dict[str, Any]] = []
    for d in candidate_dirs:
        if not d.exists():
            continue
        for jf in d.rglob("*.json"):
            try:
                env = read_json_envelope(jf)
            except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
                record_degradation('migrations', exc)
                failed.append({"path": str(jf), "error": repr(exc)})
                continue
            current = env.get("schema_version", 1)
            if current >= target:
                skipped.append(str(jf))
                continue
            try:
                migrated_payload = migrate_payload(env.get("payload") or {}, current, target)
            except (OSError, ConnectionError, TimeoutError) as exc:
                record_degradation('migrations', exc)
                failed.append({"path": str(jf), "error": repr(exc)})
                continue
            if not dry_run:
                atomic_write_json(jf, migrated_payload, schema_version=target, schema_name=env.get("schema") or "migrated")
            migrated.append(str(jf))
    return {
        "command": "migrate",
        "ok": not failed,
        "target_version": target,
        "dry_run": dry_run,
        "migrated": migrated,
        "skipped": skipped,
        "failed": failed,
    }
