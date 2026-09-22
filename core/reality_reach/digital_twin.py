"""Durable, causal digital twins for Aura's physical Reality Reach graph.

The graph is an operational state projection, not a second source of truth.  Device
identity and topology come from the attachment broker, while channel properties
advance only from validated observations that passed through the historian.  It
therefore cannot actuate hardware or manufacture a healthier state than the
underlying evidence supports.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import math
import os
import re
import secrets
import sqlite3
import stat
import time
import uuid
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from core.reality_reach.contracts import ChannelDeclaration
from core.runtime.audit_chain import canonical_json, sha256_hex
from core.runtime.lockdep import checked_lock
from core.security.zenith_secrets import KeychainBackend, require_keychain_backend

_SCHEMA_VERSION = "4"
_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9_.:-]{0,127}$")
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_MAX_GRAPH_NODES = 16_384
_MAX_RELATIONSHIPS = 65_536
_MAX_EVENTS = 250_000
_MAX_QUERY_NODES = 2_048
_MAX_QUERY_RELATIONSHIPS = 8_192
_MAX_JSON_BYTES = 64 * 1024
_SQLITE_INT_MAX = (1 << 63) - 1
_NO_LIFECYCLE_SHA256 = "sha256:" + "0" * 64
_PRUNABLE_EVENT_KINDS = frozenset({"candidate_seen", "channel_observed"})
_PRIVATE_READ_ACTION = "reality_digital_twin.read_private"
_PRIVATE_READ_SCHEMA = "aura.reality-digital-twin.private-read.v1"
_PRIVATE_READ_SCOPE = "reality_reach.private_twin"
_ARCHIVE_SEGMENT_SCHEMA = "aura.reality-digital-twin.lifecycle-segment.v1"
_ARCHIVE_CHECKPOINT_SCHEMA = "aura.reality-digital-twin.archive-checkpoint.v1"
_ARCHIVE_PENDING_SCHEMA = "aura.reality-digital-twin.archive-pending.v1"
_INTEGRITY_KEY_SCHEMA = "aura.reality-digital-twin.integrity-key.v1"
_INTEGRITY_KEY_SERVICE = "AuraRealityReachDigitalTwin"
_MAX_ARCHIVE_SEGMENTS = 64
_MAX_ARCHIVE_BYTES = 8 * 1024 * 1024
_MAX_INTEGRITY_KEY_BYTES = 8 * 1024
_MAC = re.compile(r"^hmac-sha256:[0-9a-f]{64}$")


class DigitalTwinError(RuntimeError):
    """Base class for bounded digital-twin failures."""


class DigitalTwinCorruptionError(DigitalTwinError):
    """Stored graph state or schema no longer verifies."""


class DigitalTwinConflictError(DigitalTwinError):
    """New evidence conflicts with an existing stable identity or receipt."""


class TwinLifecycle(StrEnum):
    DISCOVERED = "discovered"
    ATTACHED = "attached"
    DETACHED = "detached"
    LOST = "lost"
    DEGRADED = "degraded"
    REVOKED = "revoked"


class TwinHealth(StrEnum):
    UNKNOWN = "unknown"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"
    DETACHED = "detached"
    REVOKED = "revoked"


class TwinDisposition(StrEnum):
    ACCEPTED = "accepted"
    DUPLICATE = "duplicate"
    STALE_IGNORED = "stale_ignored"
    MANIFEST_CONFLICT = "manifest_conflict"


class TwinNodeKind(StrEnum):
    ENTITY = "entity"
    ADAPTER = "adapter"
    CHANNEL = "channel"
    BODY_LIMB = "body_limb"


class TwinRelationshipKind(StrEnum):
    CONTAINS = "contains"
    EXPOSES = "exposes"
    PROJECTS_TO = "projects_to"


@dataclass(frozen=True, slots=True)
class TwinReceipt:
    receipt_id: str
    twin_id: str
    event_id: str
    disposition: TwinDisposition
    accepted: bool
    graph_version: int
    state_sha256: str

    def __post_init__(self) -> None:
        for name, value in (
            ("receipt_id", self.receipt_id),
            ("twin_id", self.twin_id),
            ("event_id", self.event_id),
        ):
            if not _IDENTIFIER.fullmatch(value):
                raise ValueError(f"{name} must be a canonical identifier")
        if not isinstance(self.disposition, TwinDisposition):
            raise TypeError("disposition must be a TwinDisposition")
        if self.graph_version < 0:
            raise ValueError("graph_version must be non-negative")
        if not _DIGEST.fullmatch(self.state_sha256):
            raise ValueError("state_sha256 must be a sha256 digest")

    def to_dict(self) -> dict[str, Any]:
        return {
            "receipt_id": self.receipt_id,
            "twin_id": self.twin_id,
            "event_id": self.event_id,
            "disposition": self.disposition.value,
            "accepted": self.accepted,
            "graph_version": self.graph_version,
            "state_sha256": self.state_sha256,
        }


def _digest(value: Any) -> str:
    return str(sha256_hex(canonical_json(value)))


def _bytes_digest(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _short_digest(value: Any, *, length: int = 32) -> str:
    return _digest(value).removeprefix("sha256:")[:length]


def _identifier(value: Any, *, name: str) -> str:
    normalized = str(value or "").strip().lower()
    if not _IDENTIFIER.fullmatch(normalized):
        raise ValueError(f"{name} must be a canonical identifier")
    return normalized


def _digest_value(value: Any, *, name: str) -> str:
    normalized = str(value or "")
    if not _DIGEST.fullmatch(normalized):
        raise ValueError(f"{name} must be a sha256 digest")
    return normalized


def _bounded_ns(value: Any, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if not minimum <= value <= _SQLITE_INT_MAX:
        raise ValueError(f"{name} lies outside the signed 64-bit range")
    return int(value)


def _canonical_json(value: Any, *, name: str) -> str:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain canonical JSON") from exc
    if len(encoded) > _MAX_JSON_BYTES:
        raise ValueError(f"{name} exceeds the bounded JSON envelope")
    return encoded.decode("utf-8")


def _receipt_id(event_id: str, payload_sha256: str, twin_id: str) -> str:
    return "twin.receipt." + _short_digest(
        {
            "event_id": event_id,
            "payload_sha256": payload_sha256,
            "twin_id": twin_id,
        },
        length=40,
    )


def _event_id(kind: str, evidence: Any) -> str:
    return "twin.event." + _short_digest({"kind": kind, "evidence": evidence}, length=40)


def _twin_id(identity_fingerprint: str, *, persistent: bool, session_id: str) -> str:
    material: dict[str, Any] = {"identity_fingerprint": identity_fingerprint}
    if not persistent:
        material["session_id"] = session_id
    return "twin." + _short_digest(material, length=40)


def _entity_node_id(twin_id: str) -> str:
    return "twin.node.entity." + _short_digest(twin_id, length=32)


def _adapter_node_id(twin_id: str) -> str:
    return "twin.node.adapter." + _short_digest(twin_id, length=32)


def _channel_node_id(twin_id: str, channel_id: str) -> str:
    return "twin.node.channel." + _short_digest(
        {"twin_id": twin_id, "channel_id": channel_id}, length=32
    )


def _body_node_id(twin_id: str, limb_name: str) -> str:
    return "twin.node.body." + _short_digest(
        {"twin_id": twin_id, "limb_name": limb_name}, length=32
    )


def _relationship_id(
    twin_id: str,
    source_node_id: str,
    target_node_id: str,
    kind: TwinRelationshipKind,
) -> str:
    return "twin.relationship." + _short_digest(
        {
            "twin_id": twin_id,
            "source": source_node_id,
            "target": target_node_id,
            "kind": kind.value,
        },
        length=36,
    )


_SCHEMA = """
CREATE TABLE IF NOT EXISTS twin_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS twins (
    twin_id TEXT PRIMARY KEY,
    identity_fingerprint TEXT NOT NULL,
    identity_scope TEXT NOT NULL,
    connector_id TEXT NOT NULL,
    device_identity_sha256 TEXT NOT NULL,
    display_name_sha256 TEXT NOT NULL,
    transport TEXT NOT NULL,
    manifest_sha256 TEXT NOT NULL,
    persistent_identity INTEGER NOT NULL CHECK (persistent_identity IN (0, 1)),
    privacy_sensitive INTEGER NOT NULL CHECK (privacy_sensitive IN (0, 1)),
    lifecycle TEXT NOT NULL CHECK (
        lifecycle IN ('discovered', 'attached', 'detached', 'lost', 'degraded', 'revoked')
    ),
    health TEXT NOT NULL CHECK (
        health IN ('unknown', 'healthy', 'degraded', 'unavailable', 'detached', 'revoked')
    ),
    generation INTEGER NOT NULL CHECK (generation >= 1),
    discovered_at_ns INTEGER NOT NULL CHECK (discovered_at_ns > 0),
    last_seen_at_ns INTEGER NOT NULL CHECK (last_seen_at_ns >= discovered_at_ns),
    attached_at_ns INTEGER NOT NULL CHECK (attached_at_ns >= 0),
    detached_at_ns INTEGER NOT NULL CHECK (detached_at_ns >= 0),
    revoked_at_ns INTEGER NOT NULL CHECK (revoked_at_ns >= 0),
    row_sha256 TEXT NOT NULL,
    UNIQUE(identity_fingerprint, identity_scope)
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS twin_nodes (
    node_id TEXT PRIMARY KEY,
    twin_id TEXT NOT NULL REFERENCES twins(twin_id) ON DELETE CASCADE,
    node_kind TEXT NOT NULL CHECK (node_kind IN ('entity', 'adapter', 'channel', 'body_limb')),
    component_kind TEXT NOT NULL,
    external_id_sha256 TEXT NOT NULL,
    model_sha256 TEXT NOT NULL,
    model_json TEXT NOT NULL,
    enabled INTEGER NOT NULL CHECK (enabled IN (0, 1)),
    generation INTEGER NOT NULL CHECK (generation >= 1),
    row_sha256 TEXT NOT NULL
) WITHOUT ROWID;

CREATE INDEX IF NOT EXISTS twin_nodes_by_twin
ON twin_nodes(twin_id, node_kind, enabled);

CREATE TABLE IF NOT EXISTS twin_relationships (
    relationship_id TEXT PRIMARY KEY,
    twin_id TEXT NOT NULL REFERENCES twins(twin_id) ON DELETE CASCADE,
    source_node_id TEXT NOT NULL REFERENCES twin_nodes(node_id) ON DELETE CASCADE,
    target_node_id TEXT NOT NULL REFERENCES twin_nodes(node_id) ON DELETE CASCADE,
    relationship_kind TEXT NOT NULL CHECK (
        relationship_kind IN ('contains', 'exposes', 'projects_to')
    ),
    properties_json TEXT NOT NULL,
    enabled INTEGER NOT NULL CHECK (enabled IN (0, 1)),
    generation INTEGER NOT NULL CHECK (generation >= 1),
    row_sha256 TEXT NOT NULL,
    UNIQUE(twin_id, source_node_id, target_node_id, relationship_kind)
) WITHOUT ROWID;

CREATE INDEX IF NOT EXISTS twin_relationships_by_source
ON twin_relationships(source_node_id, relationship_kind, enabled);
CREATE INDEX IF NOT EXISTS twin_relationships_by_target
ON twin_relationships(target_node_id, relationship_kind, enabled);

CREATE TABLE IF NOT EXISTS twin_adapter_bindings (
    adapter_id TEXT PRIMARY KEY,
    twin_id TEXT NOT NULL REFERENCES twins(twin_id) ON DELETE CASCADE,
    adapter_node_id TEXT NOT NULL REFERENCES twin_nodes(node_id) ON DELETE CASCADE,
    active INTEGER NOT NULL CHECK (active IN (0, 1)),
    generation INTEGER NOT NULL CHECK (generation >= 1),
    topology_revision INTEGER NOT NULL CHECK (topology_revision >= 1),
    bound_at_ns INTEGER NOT NULL CHECK (bound_at_ns > 0),
    unbound_at_ns INTEGER NOT NULL CHECK (unbound_at_ns >= 0),
    row_sha256 TEXT NOT NULL
) WITHOUT ROWID;

CREATE INDEX IF NOT EXISTS twin_bindings_by_twin
ON twin_adapter_bindings(twin_id, active);

CREATE TABLE IF NOT EXISTS twin_properties (
    node_id TEXT PRIMARY KEY REFERENCES twin_nodes(node_id) ON DELETE CASCADE,
    twin_id TEXT NOT NULL REFERENCES twins(twin_id) ON DELETE CASCADE,
    observation_id TEXT NOT NULL UNIQUE,
    historian_record_id TEXT NOT NULL,
    reading_sha256 TEXT NOT NULL,
    declaration_sha256 TEXT NOT NULL,
    value_json TEXT NOT NULL,
    unit TEXT NOT NULL,
    status TEXT NOT NULL,
    quality TEXT NOT NULL,
    order_basis TEXT NOT NULL,
    order_gap INTEGER NOT NULL CHECK (order_gap IN (0, 1)),
    alarm_codes_json TEXT NOT NULL,
    captured_at_ns INTEGER NOT NULL CHECK (captured_at_ns > 0),
    source_epoch TEXT NOT NULL,
    source_sequence INTEGER NOT NULL CHECK (source_sequence >= 0),
    source_event_id TEXT NOT NULL,
    version INTEGER NOT NULL CHECK (version >= 1),
    updated_at_ns INTEGER NOT NULL CHECK (updated_at_ns > 0),
    row_sha256 TEXT NOT NULL
) WITHOUT ROWID;

CREATE INDEX IF NOT EXISTS twin_properties_by_twin
ON twin_properties(twin_id, updated_at_ns);

CREATE TABLE IF NOT EXISTS twin_events (
    event_id TEXT PRIMARY KEY,
    twin_id TEXT NOT NULL REFERENCES twins(twin_id) ON DELETE CASCADE,
    event_kind TEXT NOT NULL,
    payload_sha256 TEXT NOT NULL,
    receipt_id TEXT NOT NULL UNIQUE,
    disposition TEXT NOT NULL CHECK (
        disposition IN ('accepted', 'duplicate', 'stale_ignored', 'manifest_conflict')
    ),
    accepted INTEGER NOT NULL CHECK (accepted IN (0, 1)),
    graph_version INTEGER NOT NULL CHECK (graph_version >= 0),
    state_sha256 TEXT NOT NULL,
    created_at_ns INTEGER NOT NULL CHECK (created_at_ns > 0),
    lifecycle_sequence INTEGER NOT NULL CHECK (lifecycle_sequence >= 0),
    prior_lifecycle_sha256 TEXT NOT NULL,
    lifecycle_sha256 TEXT NOT NULL,
    row_sha256 TEXT NOT NULL
) WITHOUT ROWID;

CREATE INDEX IF NOT EXISTS twin_events_by_graph_version
ON twin_events(graph_version, created_at_ns);

CREATE TABLE IF NOT EXISTS twin_event_segments (
    segment_sequence INTEGER PRIMARY KEY CHECK (segment_sequence >= 1),
    segment_id TEXT NOT NULL UNIQUE,
    first_lifecycle_sequence INTEGER NOT NULL CHECK (first_lifecycle_sequence >= 1),
    last_lifecycle_sequence INTEGER NOT NULL CHECK (
        last_lifecycle_sequence >= first_lifecycle_sequence
    ),
    event_count INTEGER NOT NULL CHECK (event_count >= 1),
    prior_segment_sha256 TEXT NOT NULL,
    first_prior_lifecycle_sha256 TEXT NOT NULL,
    head_lifecycle_sha256 TEXT NOT NULL,
    events_sha256 TEXT NOT NULL,
    archive_file TEXT NOT NULL UNIQUE,
    archive_file_sha256 TEXT NOT NULL,
    created_at_ns INTEGER NOT NULL CHECK (created_at_ns > 0),
    segment_sha256 TEXT NOT NULL,
    row_sha256 TEXT NOT NULL
) WITHOUT ROWID;

CREATE UNIQUE INDEX IF NOT EXISTS twin_event_segments_by_range
ON twin_event_segments(first_lifecycle_sequence, last_lifecycle_sequence);
"""

_EXPECTED_COLUMNS = {
    "twin_meta": ("key", "value"),
    "twins": (
        "twin_id",
        "identity_fingerprint",
        "identity_scope",
        "connector_id",
        "device_identity_sha256",
        "display_name_sha256",
        "transport",
        "manifest_sha256",
        "persistent_identity",
        "privacy_sensitive",
        "lifecycle",
        "health",
        "generation",
        "discovered_at_ns",
        "last_seen_at_ns",
        "attached_at_ns",
        "detached_at_ns",
        "revoked_at_ns",
        "row_sha256",
    ),
    "twin_nodes": (
        "node_id",
        "twin_id",
        "node_kind",
        "component_kind",
        "external_id_sha256",
        "model_sha256",
        "model_json",
        "enabled",
        "generation",
        "row_sha256",
    ),
    "twin_relationships": (
        "relationship_id",
        "twin_id",
        "source_node_id",
        "target_node_id",
        "relationship_kind",
        "properties_json",
        "enabled",
        "generation",
        "row_sha256",
    ),
    "twin_adapter_bindings": (
        "adapter_id",
        "twin_id",
        "adapter_node_id",
        "active",
        "generation",
        "topology_revision",
        "bound_at_ns",
        "unbound_at_ns",
        "row_sha256",
    ),
    "twin_properties": (
        "node_id",
        "twin_id",
        "observation_id",
        "historian_record_id",
        "reading_sha256",
        "declaration_sha256",
        "value_json",
        "unit",
        "status",
        "quality",
        "order_basis",
        "order_gap",
        "alarm_codes_json",
        "captured_at_ns",
        "source_epoch",
        "source_sequence",
        "source_event_id",
        "version",
        "updated_at_ns",
        "row_sha256",
    ),
    "twin_events": (
        "event_id",
        "twin_id",
        "event_kind",
        "payload_sha256",
        "receipt_id",
        "disposition",
        "accepted",
        "graph_version",
        "state_sha256",
        "created_at_ns",
        "lifecycle_sequence",
        "prior_lifecycle_sha256",
        "lifecycle_sha256",
        "row_sha256",
    ),
    "twin_event_segments": (
        "segment_sequence",
        "segment_id",
        "first_lifecycle_sequence",
        "last_lifecycle_sequence",
        "event_count",
        "prior_segment_sha256",
        "first_prior_lifecycle_sha256",
        "head_lifecycle_sha256",
        "events_sha256",
        "archive_file",
        "archive_file_sha256",
        "created_at_ns",
        "segment_sha256",
        "row_sha256",
    ),
}

_EXPECTED_INDEXES = frozenset(
    {
        "twin_nodes_by_twin",
        "twin_relationships_by_source",
        "twin_relationships_by_target",
        "twin_bindings_by_twin",
        "twin_properties_by_twin",
        "twin_events_by_graph_version",
        "twin_event_segments_by_range",
    }
)

_EXPECTED_FOREIGN_KEYS = {
    "twin_nodes": {("twin_id", "twins", "twin_id", "CASCADE")},
    "twin_relationships": {
        ("twin_id", "twins", "twin_id", "CASCADE"),
        ("source_node_id", "twin_nodes", "node_id", "CASCADE"),
        ("target_node_id", "twin_nodes", "node_id", "CASCADE"),
    },
    "twin_adapter_bindings": {
        ("twin_id", "twins", "twin_id", "CASCADE"),
        ("adapter_node_id", "twin_nodes", "node_id", "CASCADE"),
    },
    "twin_properties": {
        ("node_id", "twin_nodes", "node_id", "CASCADE"),
        ("twin_id", "twins", "twin_id", "CASCADE"),
    },
    "twin_events": {("twin_id", "twins", "twin_id", "CASCADE")},
    "twin_event_segments": set(),
}


class RealityDigitalTwinGraph:
    """Private entity-component graph causally bound to Reality Reach evidence."""

    def __init__(
        self,
        db_path: Path,
        *,
        session_id: str | None = None,
        max_nodes: int = _MAX_GRAPH_NODES,
        max_relationships: int = _MAX_RELATIONSHIPS,
        max_events: int = _MAX_EVENTS,
        max_storage_bytes: int = 256 * 1024 * 1024,
        min_free_bytes: int = 512 * 1024 * 1024,
        max_archive_segments: int = _MAX_ARCHIVE_SEGMENTS,
        max_archive_bytes: int = _MAX_ARCHIVE_BYTES,
        clock_ns: Any = time.time_ns,
        private_read_capability_verifier: Any | None = None,
        migration_authority_verifier: Any | None = None,
        integrity_key_backend: KeychainBackend | None = None,
    ) -> None:
        self.db_path = Path(db_path).expanduser()
        if self.db_path.is_symlink():
            raise DigitalTwinCorruptionError(
                "digital-twin database must not be a symlink"
            )
        self._session_id = _identifier(
            session_id or f"session.{uuid.uuid4().hex}", name="session_id"
        )
        self._max_nodes = max(32, min(int(max_nodes), _MAX_GRAPH_NODES))
        self._max_relationships = max(64, min(int(max_relationships), _MAX_RELATIONSHIPS))
        self._max_events = max(256, min(int(max_events), _MAX_EVENTS))
        self._max_storage_bytes = max(16 * 1024 * 1024, int(max_storage_bytes))
        self._min_free_bytes = max(0, int(min_free_bytes))
        self._max_archive_segments = max(2, min(int(max_archive_segments), 4_096))
        self._max_archive_bytes = min(
            max(128 * 1024, int(max_archive_bytes)),
            max(128 * 1024, self._max_storage_bytes // 2),
        )
        if not callable(clock_ns):
            raise TypeError("clock_ns must be callable")
        if private_read_capability_verifier is not None and not callable(
            getattr(private_read_capability_verifier, "verify", None)
        ):
            raise TypeError("private_read_capability_verifier must expose verify()")
        if migration_authority_verifier is not None and not callable(
            getattr(
                migration_authority_verifier,
                "validate_persisted_manifest_migration",
                None,
            )
        ):
            raise TypeError(
                "migration_authority_verifier must validate persisted migration evidence"
            )
        self._clock_ns = clock_ns
        self._private_read_capability_verifier = private_read_capability_verifier
        self._migration_authority_verifier = migration_authority_verifier
        self._integrity_key_backend = integrity_key_backend or require_keychain_backend()
        self._integrity_key, self._integrity_key_id = self._load_or_create_integrity_key()
        self._lock = checked_lock("reality_digital_twin.graph", reentrant=True)
        self._connection: sqlite3.Connection | None = None
        self._ready = False
        self._last_error = ""
        self._last_integrity_check_ns = 0
        self._graph_version_cache = 0
        self._pending_archive: dict[str, Any] | None = None
        self._initialize()

    @property
    def _integrity_key_account(self) -> str:
        path_digest = hashlib.sha256(
            str(self.db_path.resolve(strict=False)).encode("utf-8")
        ).hexdigest()
        return f"digital-twin-integrity-v1-{path_digest}"

    def _load_or_create_integrity_key(self) -> tuple[bytes, str]:
        account = self._integrity_key_account
        try:
            encoded = self._integrity_key_backend.get_password(
                _INTEGRITY_KEY_SERVICE,
                account,
            )
        except Exception as exc:  # noqa: BLE001 - external credential boundary
            raise DigitalTwinCorruptionError(
                "digital-twin integrity trust root is unavailable"
            ) from exc
        if encoded is None:
            if self.db_path.exists():
                raise DigitalTwinCorruptionError(
                    "digital-twin integrity trust root is missing for existing state"
                )
            key = secrets.token_bytes(32)
            body = {
                "schema": _INTEGRITY_KEY_SCHEMA,
                "key_b64": base64.b64encode(key).decode("ascii"),
                "key_id": "sha256:" + hashlib.sha256(key).hexdigest(),
            }
            encoded = json.dumps(
                body,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
            try:
                saved = self._integrity_key_backend.set_password(
                    _INTEGRITY_KEY_SERVICE,
                    account,
                    encoded,
                )
                confirmed = self._integrity_key_backend.get_password(
                    _INTEGRITY_KEY_SERVICE,
                    account,
                )
            except Exception as exc:  # noqa: BLE001 - external credential boundary
                raise DigitalTwinCorruptionError(
                    "digital-twin integrity trust root could not be provisioned"
                ) from exc
            if saved is not True or confirmed != encoded:
                raise DigitalTwinCorruptionError(
                    "digital-twin integrity trust root could not be confirmed"
                )
        if len(encoded.encode("utf-8")) > _MAX_INTEGRITY_KEY_BYTES:
            raise DigitalTwinCorruptionError(
                "digital-twin integrity trust root exceeds its bounded envelope"
            )
        try:
            document = json.loads(encoded)
            if not isinstance(document, dict) or set(document) != {
                "schema",
                "key_b64",
                "key_id",
            }:
                raise ValueError("integrity key shape differs")
            key = base64.b64decode(str(document["key_b64"]), validate=True)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise DigitalTwinCorruptionError(
                "digital-twin integrity trust root is invalid"
            ) from exc
        key_id = str(document.get("key_id") or "")
        if (
            document.get("schema") != _INTEGRITY_KEY_SCHEMA
            or len(key) != 32
            or key_id != "sha256:" + hashlib.sha256(key).hexdigest()
        ):
            raise DigitalTwinCorruptionError(
                "digital-twin integrity trust root is invalid"
            )
        return key, key_id

    def _integrity_mac(self, purpose: str, value: Any) -> str:
        payload = canonical_json(
            {
                "key_id": self._integrity_key_id,
                "purpose": purpose,
                "value": value,
            }
        )
        return "hmac-sha256:" + hmac.new(
            self._integrity_key,
            payload,
            hashlib.sha256,
        ).hexdigest()

    def _row_hmac(self, table: str, values: Mapping[str, Any]) -> str:
        return self._integrity_mac(
            f"row:{table}",
            {"table": table, "values": dict(values)},
        )

    @property
    def session_id(self) -> str:
        return self._session_id

    def bind_migration_authority_verifier(self, verifier: Any) -> None:
        if not callable(getattr(verifier, "validate_persisted_manifest_migration", None)):
            raise TypeError(
                "migration authority verifier must validate persisted evidence"
            )
        with self._lock:
            existing = self._migration_authority_verifier
            if existing is not None and existing is not verifier:
                raise DigitalTwinConflictError(
                    "digital-twin migration authority verifier is already bound"
                )
            self._migration_authority_verifier = verifier

    def _validate_manifest_migration_authority(
        self,
        *,
        evidence: Mapping[str, Any] | None,
        request_id: str,
        identity_fingerprint: str,
        connector_id: str,
        expected_manifest_sha256: str,
        new_manifest_sha256: str,
        persistent: bool,
    ) -> tuple[str, str]:
        verifier = self._migration_authority_verifier
        if verifier is None or evidence is None:
            raise DigitalTwinConflictError(
                "manifest drift requires exact verified migration authority"
            )
        from core.reality_reach.attachment_authority import (
            build_manifest_migration_authority_intent,
        )

        intent = build_manifest_migration_authority_intent(
            request_id=request_id,
            identity_fingerprint=identity_fingerprint,
            connector_id=connector_id,
            expected_manifest_sha256=expected_manifest_sha256,
            new_manifest_sha256=new_manifest_sha256,
            persistent=persistent,
        )
        validated = verifier.validate_persisted_manifest_migration(
            evidence,
            intent=intent,
            persistent=persistent,
        )
        if not isinstance(validated, Mapping):
            raise DigitalTwinConflictError(
                "manifest migration authority returned no verifiable evidence"
            )
        capability = validated.get("capability")
        receipt_id = (
            str(capability.get("receipt_id") or "")
            if isinstance(capability, Mapping)
            else ""
        )
        authority_receipt_id = _identifier(
            receipt_id,
            name="migration_authority_receipt_id",
        )
        evidence_sha256 = _digest_value(
            validated.get("evidence_sha256"),
            name="migration_authority_evidence_sha256",
        )
        return authority_receipt_id, evidence_sha256

    def manifest_migration_intent(
        self,
        candidate: Any,
        *,
        request_id: str,
    ) -> dict[str, Any] | None:
        fields = self._candidate_fields(candidate)
        canonical_request = _identifier(request_id, name="migration_request_id")
        twin_id = _twin_id(
            fields["identity_fingerprint"],
            persistent=fields["persistent_identity"],
            session_id=self._session_id,
        )
        with self._lock:
            connection = self._connection_or_raise()
            row = connection.execute(
                "SELECT * FROM twins WHERE twin_id=?", (twin_id,)
            ).fetchone()
            if row is None:
                return None
            self._verify_twin_row(row)
            current_manifest = str(row["manifest_sha256"])
            if current_manifest == fields["manifest_sha256"]:
                return None
            if str(row["connector_id"]) != fields["connector_id"]:
                raise DigitalTwinConflictError(
                    "manifest migration cannot silently change connector identity"
                )
        from core.reality_reach.attachment_authority import (
            build_manifest_migration_authority_intent,
        )

        return build_manifest_migration_authority_intent(
            request_id=canonical_request,
            identity_fingerprint=fields["identity_fingerprint"],
            connector_id=fields["connector_id"],
            expected_manifest_sha256=current_manifest,
            new_manifest_sha256=fields["manifest_sha256"],
            persistent=bool(fields["persistent_identity"]),
        )

    def _initialize(self) -> None:
        with self._lock:
            self._prepare_storage()
            connection = self._connect()
            try:
                connection.executescript(_SCHEMA)
                connection.execute(
                    "INSERT OR IGNORE INTO twin_meta(key, value) VALUES('schema_version', ?)",
                    (_SCHEMA_VERSION,),
                )
                connection.execute(
                    "INSERT OR IGNORE INTO twin_meta(key, value) VALUES('graph_version', '0')"
                )
                connection.execute(
                    "INSERT OR IGNORE INTO twin_meta(key, value) VALUES('migration_state', 'complete')"
                )
                connection.execute(
                    "INSERT OR IGNORE INTO twin_meta(key, value) "
                    "VALUES('lifecycle_sequence', '0')"
                )
                connection.execute(
                    "INSERT OR IGNORE INTO twin_meta(key, value) "
                    "VALUES('lifecycle_head_sha256', ?)",
                    (_NO_LIFECYCLE_SHA256,),
                )
                for key, value in (
                    ("lifecycle_archive_segment_sequence", "0"),
                    ("lifecycle_archive_head_sha256", _NO_LIFECYCLE_SHA256),
                    ("lifecycle_archive_last_event_sequence", "0"),
                    ("lifecycle_archive_last_event_head_sha256", _NO_LIFECYCLE_SHA256),
                    ("lifecycle_archive_retained_from_segment_sequence", "1"),
                    ("lifecycle_archive_retained_prior_segment_sha256", _NO_LIFECYCLE_SHA256),
                    ("lifecycle_archive_retained_first_event_sequence", "1"),
                    ("lifecycle_archive_retained_prior_event_sha256", _NO_LIFECYCLE_SHA256),
                    ("lifecycle_archive_retired_segment_count", "0"),
                    ("lifecycle_archive_retired_event_count", "0"),
                    ("lifecycle_archive_checkpoint_hmac_sha256", ""),
                    ("lifecycle_archive_store_id", f"archive.{uuid.uuid4().hex}"),
                ):
                    connection.execute(
                        "INSERT OR IGNORE INTO twin_meta(key, value) VALUES(?, ?)",
                        (key, value),
                    )
                connection.commit()
                self._recover_archive_transaction(connection)
                self._ensure_archive_checkpoint(connection)
                self._cleanup_unreferenced_archive_files(connection)
                self._validate_schema(connection)
                self._verify_integrity(connection, full=True)
                self._connection = connection
                self._ready = True
                self._graph_version_cache = self._graph_version(connection)
                self._reconcile_process_boundary()
                self._last_error = ""
            except Exception:
                self._connection = None
                self._ready = False
                connection.close()
                raise

    def _prepare_storage(self) -> None:
        parent = self.db_path.parent
        if parent.exists() and parent.is_symlink():
            raise DigitalTwinCorruptionError("digital-twin parent must not be a symlink")
        parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            os.chmod(parent, stat.S_IRWXU)
        except OSError as exc:
            raise DigitalTwinCorruptionError(
                "digital-twin parent could not be restricted to its owner"
            ) from exc
        if stat.S_IMODE(parent.stat().st_mode) != stat.S_IRWXU:
            raise DigitalTwinCorruptionError(
                "digital-twin parent is not restricted to its owner"
            )
        if self.db_path.is_symlink():
            raise DigitalTwinCorruptionError("digital-twin database must not be a symlink")
        archive = self._archive_dir
        if archive.exists() and archive.is_symlink():
            raise DigitalTwinCorruptionError("digital-twin archive must not be a symlink")
        archive.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            os.chmod(archive, stat.S_IRWXU)
        except OSError as exc:
            raise DigitalTwinCorruptionError(
                "digital-twin archive could not be restricted to its owner"
            ) from exc
        if stat.S_IMODE(archive.stat().st_mode) != stat.S_IRWXU:
            raise DigitalTwinCorruptionError(
                "digital-twin archive is not restricted to its owner"
            )

    @property
    def _archive_dir(self) -> Path:
        return self.db_path.with_name(f"{self.db_path.name}.lifecycle-archive")

    @property
    def _archive_checkpoint_path(self) -> Path:
        return self._archive_dir / "checkpoint.json"

    @property
    def _archive_pending_path(self) -> Path:
        return self._archive_dir / ".pending.json"

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.db_path,
            timeout=5.0,
            isolation_level=None,
            check_same_thread=False,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=5000")
        mode = str(connection.execute("PRAGMA journal_mode=WAL").fetchone()[0]).lower()
        if mode != "wal":
            connection.close()
            raise DigitalTwinCorruptionError("digital-twin store could not enable WAL")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("PRAGMA trusted_schema=OFF")
        connection.execute("PRAGMA secure_delete=ON")
        try:
            os.chmod(self.db_path, stat.S_IRUSR | stat.S_IWUSR)
        except OSError:
            connection.close()
            raise
        return connection

    def _validate_schema(self, connection: sqlite3.Connection) -> None:
        version = connection.execute(
            "SELECT value FROM twin_meta WHERE key='schema_version'"
        ).fetchone()
        if version is None or str(version[0]) != _SCHEMA_VERSION:
            raise DigitalTwinCorruptionError("unsupported digital-twin schema version")
        migration = connection.execute(
            "SELECT value FROM twin_meta WHERE key='migration_state'"
        ).fetchone()
        if migration is None or str(migration[0]) != "complete":
            raise DigitalTwinCorruptionError("digital-twin migration is incomplete")
        lifecycle_sequence = connection.execute(
            "SELECT value FROM twin_meta WHERE key='lifecycle_sequence'"
        ).fetchone()
        lifecycle_head = connection.execute(
            "SELECT value FROM twin_meta WHERE key='lifecycle_head_sha256'"
        ).fetchone()
        try:
            sequence_value = int(lifecycle_sequence[0]) if lifecycle_sequence is not None else -1
        except (TypeError, ValueError) as exc:
            raise DigitalTwinCorruptionError(
                "digital-twin lifecycle sequence is invalid"
            ) from exc
        if sequence_value < 0:
            raise DigitalTwinCorruptionError("digital-twin lifecycle sequence is invalid")
        if lifecycle_head is None or not _DIGEST.fullmatch(str(lifecycle_head[0])):
            raise DigitalTwinCorruptionError("digital-twin lifecycle head is invalid")
        archive_meta: dict[str, str] = {}
        for key in (
            "lifecycle_archive_segment_sequence",
            "lifecycle_archive_head_sha256",
            "lifecycle_archive_last_event_sequence",
            "lifecycle_archive_last_event_head_sha256",
            "lifecycle_archive_retained_from_segment_sequence",
            "lifecycle_archive_retained_prior_segment_sha256",
            "lifecycle_archive_retained_first_event_sequence",
            "lifecycle_archive_retained_prior_event_sha256",
            "lifecycle_archive_retired_segment_count",
            "lifecycle_archive_retired_event_count",
            "lifecycle_archive_checkpoint_hmac_sha256",
            "lifecycle_archive_store_id",
        ):
            row = connection.execute(
                "SELECT value FROM twin_meta WHERE key=?", (key,)
            ).fetchone()
            if row is None:
                raise DigitalTwinCorruptionError(
                    f"digital-twin lifecycle archive metadata is missing: {key}"
                )
            archive_meta[key] = str(row[0])
        try:
            archive_segment_sequence = int(
                archive_meta["lifecycle_archive_segment_sequence"]
            )
            archive_event_sequence = int(
                archive_meta["lifecycle_archive_last_event_sequence"]
            )
        except (TypeError, ValueError) as exc:
            raise DigitalTwinCorruptionError(
                "digital-twin lifecycle archive sequence is invalid"
            ) from exc
        if archive_segment_sequence < 0 or not 0 <= archive_event_sequence <= sequence_value:
            raise DigitalTwinCorruptionError(
                "digital-twin lifecycle archive sequence is invalid"
            )
        for key in (
            "lifecycle_archive_head_sha256",
            "lifecycle_archive_last_event_head_sha256",
            "lifecycle_archive_retained_prior_segment_sha256",
            "lifecycle_archive_retained_prior_event_sha256",
        ):
            if not _DIGEST.fullmatch(archive_meta[key]):
                raise DigitalTwinCorruptionError(
                    f"digital-twin lifecycle archive digest is invalid: {key}"
                )
        if archive_meta["lifecycle_archive_checkpoint_hmac_sha256"] and not _MAC.fullmatch(
            archive_meta["lifecycle_archive_checkpoint_hmac_sha256"]
        ):
            raise DigitalTwinCorruptionError(
                "digital-twin lifecycle archive checkpoint authentication is invalid"
            )
        try:
            retained_segment = int(
                archive_meta["lifecycle_archive_retained_from_segment_sequence"]
            )
            retained_event = int(
                archive_meta["lifecycle_archive_retained_first_event_sequence"]
            )
            retired_segments = int(
                archive_meta["lifecycle_archive_retired_segment_count"]
            )
            retired_events = int(archive_meta["lifecycle_archive_retired_event_count"])
        except (TypeError, ValueError) as exc:
            raise DigitalTwinCorruptionError(
                "digital-twin lifecycle retention metadata is invalid"
            ) from exc
        if (
            retained_segment != retired_segments + 1
            or retained_event != retired_events + 1
            or retired_segments < 0
            or retired_events < 0
        ):
            raise DigitalTwinCorruptionError(
                "digital-twin lifecycle retention anchors differ"
            )
        _identifier(
            archive_meta["lifecycle_archive_store_id"],
            name="lifecycle_archive_store_id",
        )
        for table, expected in _EXPECTED_COLUMNS.items():
            actual_columns = tuple(
                str(row[1]) for row in connection.execute(f"PRAGMA table_info({table})")
            )
            if actual_columns != expected:
                raise DigitalTwinCorruptionError(
                    f"digital-twin schema drift in {table}: {actual_columns!r}"
                )
        if int(connection.execute("PRAGMA foreign_keys").fetchone()[0]) != 1:
            raise DigitalTwinCorruptionError("digital-twin foreign keys are disabled")
        named_indexes = {
            str(row[0])
            for row in connection.execute(
                """
                SELECT name FROM sqlite_master
                WHERE type='index' AND name NOT LIKE 'sqlite_autoindex_%'
                """
            )
        }
        if named_indexes != _EXPECTED_INDEXES:
            raise DigitalTwinCorruptionError("digital-twin index manifest differs")
        for table, expected_foreign_keys in _EXPECTED_FOREIGN_KEYS.items():
            actual_foreign_keys = {
                (str(row[3]), str(row[2]), str(row[4]), str(row[6]).upper())
                for row in connection.execute(f"PRAGMA foreign_key_list({table})")
            }
            if actual_foreign_keys != expected_foreign_keys:
                raise DigitalTwinCorruptionError(
                    f"digital-twin foreign-key manifest differs in {table}"
                )

    def _connection_or_raise(self) -> sqlite3.Connection:
        if self._connection is None or not self._ready:
            raise DigitalTwinError(self._last_error or "digital-twin graph is unavailable")
        return self._connection

    def _transaction(self) -> sqlite3.Connection:
        connection = self._connection_or_raise()
        self._ensure_storage_headroom()
        connection.execute("BEGIN IMMEDIATE")
        return connection

    def _commit(self, connection: sqlite3.Connection) -> None:
        self._ensure_storage_headroom()
        connection.execute("COMMIT")
        if self._pending_archive is not None:
            try:
                self._finalize_archive_transaction(connection, self._pending_archive)
            except Exception as exc:
                self._ready = False
                self._last_error = (
                    f"archive_reconciliation_pending:{type(exc).__name__}:{exc}"
                )[:320]
                raise

    def _rollback(self, connection: sqlite3.Connection) -> None:
        was_in_transaction = connection.in_transaction
        if was_in_transaction:
            connection.execute("ROLLBACK")
        if was_in_transaction and self._pending_archive is not None:
            self._discard_archive_transaction(self._pending_archive)
        self._graph_version_cache = self._graph_version(connection)

    @staticmethod
    def _archive_json_bytes(value: Any, *, name: str) -> bytes:
        try:
            encoded = json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise DigitalTwinError(f"{name} is not canonical JSON") from exc
        if len(encoded) > (_MAX_JSON_BYTES * 2):
            raise DigitalTwinError(f"{name} exceeds its bounded archive envelope")
        return encoded

    def _write_archive_json(self, path: Path, value: Any, *, name: str) -> None:
        if path.parent != self._archive_dir or path.is_symlink():
            raise DigitalTwinCorruptionError(f"{name} path escaped the archive")
        payload = self._archive_json_bytes(value, name=name)
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(temporary, flags, stat.S_IRUSR | stat.S_IWUSR)
        try:
            with os.fdopen(descriptor, "wb", closefd=True) as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            directory = os.open(self._archive_dir, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            try:
                temporary.unlink(missing_ok=True)
            # not a failure: a temporary already gone is the state this unlink reaches.
            except OSError:
                pass

    def _read_archive_json(self, path: Path, *, name: str) -> dict[str, Any]:
        if path.parent != self._archive_dir or path.is_symlink() or not path.is_file():
            raise DigitalTwinCorruptionError(f"{name} is missing or unsafe")
        try:
            payload = path.read_bytes()
        except OSError as exc:
            raise DigitalTwinCorruptionError(f"{name} cannot be read") from exc
        if len(payload) > (_MAX_JSON_BYTES * 2):
            raise DigitalTwinCorruptionError(f"{name} exceeds its bounded envelope")
        try:
            value = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DigitalTwinCorruptionError(f"{name} is invalid JSON") from exc
        if not isinstance(value, dict):
            raise DigitalTwinCorruptionError(f"{name} must be a JSON object")
        return dict(value)

    @staticmethod
    def _archive_meta_keys() -> tuple[str, ...]:
        return (
            "lifecycle_archive_store_id",
            "lifecycle_archive_segment_sequence",
            "lifecycle_archive_head_sha256",
            "lifecycle_archive_last_event_sequence",
            "lifecycle_archive_last_event_head_sha256",
            "lifecycle_archive_retained_from_segment_sequence",
            "lifecycle_archive_retained_prior_segment_sha256",
            "lifecycle_archive_retained_first_event_sequence",
            "lifecycle_archive_retained_prior_event_sha256",
            "lifecycle_archive_retired_segment_count",
            "lifecycle_archive_retired_event_count",
        )

    def _archive_checkpoint_body(self, connection: sqlite3.Connection) -> dict[str, Any]:
        meta: dict[str, str] = {}
        for key in self._archive_meta_keys():
            row = connection.execute(
                "SELECT value FROM twin_meta WHERE key=?", (key,)
            ).fetchone()
            if row is None:
                raise DigitalTwinCorruptionError(
                    f"digital-twin archive checkpoint metadata is missing: {key}"
                )
            meta[key] = str(row[0])
        manifests = [
            {name: row[name] for name in _EXPECTED_COLUMNS["twin_event_segments"]}
            for row in connection.execute(
                "SELECT * FROM twin_event_segments ORDER BY segment_sequence"
            )
        ]
        return {
            "schema": _ARCHIVE_CHECKPOINT_SCHEMA,
            "meta": meta,
            "retained_manifests": manifests,
        }

    def _archive_checkpoint_envelope(
        self,
        body: Mapping[str, Any],
    ) -> dict[str, Any]:
        canonical_body = dict(body)
        return {
            **canonical_body,
            "checkpoint_hmac_sha256": self._integrity_mac(
                "archive-checkpoint",
                canonical_body,
            ),
        }

    def _ensure_archive_checkpoint(self, connection: sqlite3.Connection) -> None:
        body = self._archive_checkpoint_body(connection)
        expected = self._archive_checkpoint_envelope(body)
        row = connection.execute(
            "SELECT value FROM twin_meta "
            "WHERE key='lifecycle_archive_checkpoint_hmac_sha256'"
        ).fetchone()
        stored = str(row[0]) if row is not None else ""
        if not stored:
            if int(body["meta"]["lifecycle_archive_segment_sequence"]) != 0:
                raise DigitalTwinCorruptionError(
                    "non-empty lifecycle archive has no independent checkpoint"
                )
            self._write_archive_json(
                self._archive_checkpoint_path,
                expected,
                name="lifecycle archive checkpoint",
            )
            connection.execute(
                "UPDATE twin_meta SET value=? "
                "WHERE key='lifecycle_archive_checkpoint_hmac_sha256'",
                (expected["checkpoint_hmac_sha256"],),
            )
            connection.commit()
            stored = str(expected["checkpoint_hmac_sha256"])
        actual = self._read_archive_json(
            self._archive_checkpoint_path,
            name="lifecycle archive checkpoint",
        )
        supplied = str(actual.pop("checkpoint_hmac_sha256", ""))
        expected_mac = self._integrity_mac("archive-checkpoint", actual)
        if (
            not _MAC.fullmatch(supplied)
            or not hmac.compare_digest(supplied, expected_mac)
            or not hmac.compare_digest(supplied, stored)
            or actual != body
        ):
            raise DigitalTwinCorruptionError(
                "digital-twin independent archive checkpoint differs"
            )

    def _cleanup_unreferenced_archive_files(
        self,
        connection: sqlite3.Connection,
    ) -> None:
        referenced = {
            str(row[0])
            for row in connection.execute("SELECT archive_file FROM twin_event_segments")
        }
        for path in self._archive_dir.iterdir():
            if path.name == self._archive_checkpoint_path.name:
                continue
            if path.name == self._archive_pending_path.name:
                raise DigitalTwinCorruptionError(
                    "lifecycle archive recovery intent remained unresolved"
                )
            if path.name in referenced:
                continue
            if re.fullmatch(r"segment-[0-9]{12}-[0-9a-f]{40}\.json", path.name):
                if path.is_symlink() or not path.is_file():
                    raise DigitalTwinCorruptionError(
                        "unreferenced lifecycle archive entry is unsafe"
                    )
                path.unlink()
                continue
            if re.fullmatch(r"\..+\.[0-9a-f]{32}\.tmp", path.name):
                if path.is_symlink() or not path.is_file():
                    raise DigitalTwinCorruptionError(
                        "interrupted lifecycle archive temporary is unsafe"
                    )
                path.unlink()
                continue
            raise DigitalTwinCorruptionError(
                f"unexpected lifecycle archive entry: {path.name}"
            )

    def _recover_archive_transaction(self, connection: sqlite3.Connection) -> None:
        pending_path = self._archive_pending_path
        if not pending_path.exists():
            return
        pending = self._read_archive_json(
            pending_path,
            name="lifecycle archive recovery intent",
        )
        supplied = str(pending.pop("pending_hmac_sha256", ""))
        expected_pending = self._integrity_mac("archive-pending", pending)
        if (
            not _MAC.fullmatch(supplied)
            or not hmac.compare_digest(supplied, expected_pending)
            or pending.get("schema") != _ARCHIVE_PENDING_SCHEMA
        ):
            raise DigitalTwinCorruptionError("lifecycle archive recovery intent differs")
        row = connection.execute(
            "SELECT value FROM twin_meta "
            "WHERE key='lifecycle_archive_checkpoint_hmac_sha256'"
        ).fetchone()
        current = str(row[0]) if row is not None else ""
        target = pending.get("target_checkpoint")
        if not isinstance(target, dict):
            raise DigitalTwinCorruptionError("lifecycle archive recovery target is invalid")
        target_hmac = str(target.get("checkpoint_hmac_sha256") or "")
        if current == target_hmac:
            self._finalize_archive_transaction(connection, pending)
            return
        if current != str(pending.get("prior_checkpoint_hmac_sha256") or ""):
            raise DigitalTwinCorruptionError("lifecycle archive recovery state diverged")
        self._discard_archive_transaction(pending)

    def _finalize_archive_transaction(
        self,
        connection: sqlite3.Connection,
        pending: Mapping[str, Any],
    ) -> None:
        target = pending.get("target_checkpoint")
        if not isinstance(target, Mapping):
            raise DigitalTwinCorruptionError("lifecycle archive target is invalid")
        body = self._archive_checkpoint_body(connection)
        expected = self._archive_checkpoint_envelope(body)
        if dict(target) != expected:
            raise DigitalTwinCorruptionError("lifecycle archive commit target differs")
        self._write_archive_json(
            self._archive_checkpoint_path,
            expected,
            name="lifecycle archive checkpoint",
        )
        for raw_name in pending.get("retire_files", []):
            file_name = str(raw_name)
            if not re.fullmatch(r"segment-[0-9]{12}-[0-9a-f]{40}\.json", file_name):
                raise DigitalTwinCorruptionError("lifecycle archive retire path is invalid")
            (self._archive_dir / file_name).unlink(missing_ok=True)
        self._archive_pending_path.unlink(missing_ok=True)
        self._pending_archive = None

    def _discard_archive_transaction(self, pending: Mapping[str, Any]) -> None:
        for raw_name in pending.get("new_files", []):
            file_name = str(raw_name)
            if re.fullmatch(r"segment-[0-9]{12}-[0-9a-f]{40}\.json", file_name):
                (self._archive_dir / file_name).unlink(missing_ok=True)
        self._archive_pending_path.unlink(missing_ok=True)
        self._pending_archive = None

    def _graph_version(self, connection: sqlite3.Connection) -> int:
        row = connection.execute("SELECT value FROM twin_meta WHERE key='graph_version'").fetchone()
        if row is None:
            raise DigitalTwinCorruptionError("digital-twin graph version is missing")
        return int(row[0])

    def _advance_graph_version(self, connection: sqlite3.Connection) -> int:
        version = self._graph_version(connection) + 1
        connection.execute(
            "UPDATE twin_meta SET value=? WHERE key='graph_version'", (str(version),)
        )
        self._graph_version_cache = version
        return version

    def _reconcile_process_boundary(self) -> None:
        """Fence process-bound adapters left active by an unclean prior exit."""

        connection = self._transaction()
        try:
            active = connection.execute(
                "SELECT * FROM twin_adapter_bindings WHERE active=1 ORDER BY adapter_id"
            ).fetchall()
            for binding in active:
                self._verify_binding_row(binding)
                twin_id = str(binding["twin_id"])
                now_ns = self._now_ns()
                values = self._binding_values(binding)
                values.update({"active": 0, "unbound_at_ns": now_ns})
                values["row_sha256"] = self._row_hmac(
                    "twin_adapter_bindings",
                    {k: v for k, v in values.items() if k != "row_sha256"},
                )
                self._write_binding_values(connection, values)
                twin = connection.execute(
                    "SELECT * FROM twins WHERE twin_id=?", (twin_id,)
                ).fetchone()
                if twin is None:
                    raise DigitalTwinCorruptionError("active binding has no twin")
                self._verify_twin_row(twin)
                twin_values = self._twin_values(twin)
                if twin_values["lifecycle"] != TwinLifecycle.REVOKED.value:
                    twin_values.update(
                        {
                            "lifecycle": TwinLifecycle.LOST.value,
                            "health": TwinHealth.DETACHED.value,
                            "detached_at_ns": now_ns,
                        }
                    )
                    twin_values["row_sha256"] = self._row_hmac(
                        "twins",
                        {key: value for key, value in twin_values.items() if key != "row_sha256"},
                    )
                    self._write_twin_values(connection, twin_values)
                payload_sha256 = _digest(
                    {
                        "twin_id": twin_id,
                        "adapter_id": str(binding["adapter_id"]),
                        "prior_topology_revision": int(binding["topology_revision"]),
                    }
                )
                event_id = _event_id("runtime_restart_lost", payload_sha256)
                if self._existing_event(connection, event_id, payload_sha256) is None:
                    version = self._advance_graph_version(connection)
                    self._record_event(
                        connection,
                        event_id=event_id,
                        twin_id=twin_id,
                        event_kind="runtime_restart_lost",
                        payload_sha256=payload_sha256,
                        disposition=TwinDisposition.ACCEPTED,
                        accepted=True,
                        graph_version=version,
                    )
            self._commit(connection)
        except Exception:
            self._rollback(connection)
            raise

    def _candidate_fields(self, candidate: Any) -> dict[str, Any]:
        fields: dict[str, Any] = {
            "candidate_id": _identifier(
                getattr(candidate, "candidate_id", ""), name="candidate_id"
            ),
            "connector_id": _identifier(
                getattr(candidate, "connector_id", ""), name="connector_id"
            ),
            "device_id": _identifier(getattr(candidate, "device_id", ""), name="device_id"),
            "display_name": str(getattr(candidate, "display_name", "") or ""),
            "transport": _identifier(getattr(candidate, "transport", ""), name="transport"),
            "identity_fingerprint": _digest_value(
                getattr(candidate, "identity_fingerprint", ""),
                name="identity_fingerprint",
            ),
            "manifest_sha256": _digest_value(
                getattr(candidate, "manifest_sha256", ""), name="manifest_sha256"
            ),
            "persistent_identity": bool(getattr(candidate, "persistent_identity", False)),
            "privacy_sensitive": bool(getattr(candidate, "privacy_sensitive", False)),
            "discovered_at_ns": _bounded_ns(
                getattr(candidate, "discovered_at_ns", 0),
                name="discovered_at_ns",
                minimum=1,
            ),
        }
        if not fields["display_name"] or len(fields["display_name"]) > 160:
            raise ValueError("display_name must be present and bounded")
        return fields

    def observe_candidate(self, candidate: Any) -> TwinReceipt:
        fields = self._candidate_fields(candidate)
        twin_id = _twin_id(
            fields["identity_fingerprint"],
            persistent=fields["persistent_identity"],
            session_id=self._session_id,
        )
        payload_sha256 = _digest(
            {
                "identity_fingerprint": fields["identity_fingerprint"],
                "manifest_sha256": fields["manifest_sha256"],
                "candidate_id": fields["candidate_id"],
                "discovered_at_ns": fields["discovered_at_ns"],
            }
        )
        with self._lock:
            connection = self._transaction()
            try:
                for prior_kind in (
                    "candidate_discovered",
                    "candidate_manifest_conflict",
                    "candidate_rediscovered",
                    "candidate_seen",
                ):
                    duplicate = self._existing_event(
                        connection,
                        _event_id(prior_kind, payload_sha256),
                        payload_sha256,
                    )
                    if duplicate is not None:
                        self._commit(connection)
                        return duplicate
                existing = connection.execute(
                    "SELECT * FROM twins WHERE twin_id=?", (twin_id,)
                ).fetchone()
                if (
                    existing is not None
                    and str(existing["manifest_sha256"]) != fields["manifest_sha256"]
                ):
                    event_kind = "candidate_manifest_conflict"
                    receipt = self._record_event(
                        connection,
                        event_id=_event_id(event_kind, payload_sha256),
                        twin_id=twin_id,
                        event_kind=event_kind,
                        payload_sha256=payload_sha256,
                        disposition=TwinDisposition.MANIFEST_CONFLICT,
                        accepted=False,
                        graph_version=self._graph_version(connection),
                    )
                    self._commit(connection)
                    return receipt
                now_ns = max(fields["discovered_at_ns"], self._now_ns())
                if existing is None:
                    event_kind = "candidate_discovered"
                    self._ensure_capacity(connection, nodes=1)
                    self._insert_twin(connection, twin_id=twin_id, fields=fields, now_ns=now_ns)
                    self._upsert_node(
                        connection,
                        node_id=_entity_node_id(twin_id),
                        twin_id=twin_id,
                        node_kind=TwinNodeKind.ENTITY,
                        component_kind="physical_entity",
                        external_id=fields["identity_fingerprint"],
                        model={
                            "manifest_sha256": fields["manifest_sha256"],
                            "persistent_identity": fields["persistent_identity"],
                            "privacy_sensitive": fields["privacy_sensitive"],
                        },
                        enabled=True,
                        generation=1,
                    )
                else:
                    self._verify_twin_row(existing)
                    if str(existing["lifecycle"]) == TwinLifecycle.REVOKED.value:
                        raise DigitalTwinConflictError("revoked twin identity cannot rediscover")
                    was_absent = str(existing["lifecycle"]) in {
                        TwinLifecycle.DETACHED.value,
                        TwinLifecycle.LOST.value,
                    }
                    self._update_twin_seen(connection, existing, fields, now_ns=now_ns)
                    event_kind = "candidate_rediscovered" if was_absent else "candidate_seen"
                    if was_absent:
                        rediscovered = connection.execute(
                            "SELECT * FROM twins WHERE twin_id=?", (twin_id,)
                        ).fetchone()
                        if rediscovered is None:
                            raise DigitalTwinCorruptionError("rediscovered twin disappeared")
                        values = self._twin_values(rediscovered)
                        values.update(
                            {
                                "lifecycle": TwinLifecycle.DISCOVERED.value,
                                "health": TwinHealth.UNKNOWN.value,
                            }
                        )
                        values["row_sha256"] = self._row_hmac(
                            "twins",
                            {key: value for key, value in values.items() if key != "row_sha256"},
                        )
                        self._write_twin_values(connection, values)
                version = self._advance_graph_version(connection)
                receipt = self._record_event(
                    connection,
                    event_id=_event_id(event_kind, payload_sha256),
                    twin_id=twin_id,
                    event_kind=event_kind,
                    payload_sha256=payload_sha256,
                    disposition=TwinDisposition.ACCEPTED,
                    accepted=True,
                    graph_version=version,
                )
                self._prune_events(connection)
                self._commit(connection)
                return receipt
            except Exception:
                self._rollback(connection)
                raise

    def migrate_manifest(
        self,
        identity_fingerprint: str,
        *,
        new_manifest_sha256: str,
        expected_manifest_sha256: str,
        migration_request_id: str,
        migration_authority_evidence: Mapping[str, Any],
        persistent_identity: bool = True,
    ) -> TwinReceipt:
        identity = _digest_value(identity_fingerprint, name="identity_fingerprint")
        new_manifest = _digest_value(new_manifest_sha256, name="new_manifest_sha256")
        expected_manifest = _digest_value(expected_manifest_sha256, name="expected_manifest_sha256")
        request_id = _identifier(migration_request_id, name="migration_request_id")
        twin_id = _twin_id(
            identity,
            persistent=persistent_identity,
            session_id=self._session_id,
        )
        with self._lock:
            connection = self._connection_or_raise()
            authority_row = connection.execute(
                "SELECT * FROM twins WHERE twin_id=?", (twin_id,)
            ).fetchone()
            if authority_row is None:
                raise LookupError("digital twin is unavailable for migration")
            self._verify_twin_row(authority_row)
            if str(authority_row["manifest_sha256"]) != expected_manifest:
                raise DigitalTwinConflictError("manifest migration compare-and-swap failed")
            connector_id = str(authority_row["connector_id"])
        authority, authority_evidence_sha256 = self._validate_manifest_migration_authority(
            evidence=migration_authority_evidence,
            request_id=request_id,
            identity_fingerprint=identity,
            connector_id=connector_id,
            expected_manifest_sha256=expected_manifest,
            new_manifest_sha256=new_manifest,
            persistent=bool(persistent_identity),
        )
        payload_sha256 = _digest(
            {
                "twin_id": twin_id,
                "expected_manifest_sha256": expected_manifest,
                "new_manifest_sha256": new_manifest,
                "authority_receipt_id": authority,
                "authority_evidence_sha256": authority_evidence_sha256,
                "migration_request_id": request_id,
            }
        )
        event_id = _event_id("manifest_migrated", payload_sha256)
        with self._lock:
            connection = self._transaction()
            try:
                duplicate = self._existing_event(connection, event_id, payload_sha256)
                if duplicate is not None:
                    self._commit(connection)
                    return duplicate
                row = connection.execute(
                    "SELECT * FROM twins WHERE twin_id=?", (twin_id,)
                ).fetchone()
                if row is None:
                    raise LookupError("digital twin is unavailable for migration")
                self._verify_twin_row(row)
                if (
                    str(row["manifest_sha256"]) != expected_manifest
                    or str(row["connector_id"]) != connector_id
                    or bool(row["persistent_identity"]) is not bool(persistent_identity)
                ):
                    raise DigitalTwinConflictError("manifest migration compare-and-swap failed")
                values = self._twin_values(row)
                values["manifest_sha256"] = new_manifest
                values["generation"] = int(row["generation"]) + 1
                values["health"] = TwinHealth.UNKNOWN.value
                values["row_sha256"] = self._row_hmac(
                    "twins", {k: v for k, v in values.items() if k != "row_sha256"}
                )
                self._write_twin_values(connection, values)
                version = self._advance_graph_version(connection)
                receipt = self._record_event(
                    connection,
                    event_id=event_id,
                    twin_id=twin_id,
                    event_kind="manifest_migrated",
                    payload_sha256=payload_sha256,
                    disposition=TwinDisposition.ACCEPTED,
                    accepted=True,
                    graph_version=version,
                )
                self._commit(connection)
                return receipt
            except Exception:
                self._rollback(connection)
                raise

    def attach_adapter(
        self,
        candidate: Any,
        adapter: Any,
        *,
        body_projection: Any | None = None,
        migration_request_id: str = "",
        migration_authority_evidence: Mapping[str, Any] | None = None,
    ) -> TwinReceipt:
        fields = self._candidate_fields(candidate)
        adapter_id = _identifier(getattr(adapter, "adapter_id", ""), name="adapter_id")
        declarations_fn = getattr(adapter, "declarations", None)
        if not callable(declarations_fn):
            raise TypeError("attached adapter must expose declarations")
        declarations = tuple(declarations_fn())
        if not declarations or any(
            not isinstance(item, ChannelDeclaration) for item in declarations
        ):
            raise TypeError("attached adapter declarations are invalid")
        twin_id = _twin_id(
            fields["identity_fingerprint"],
            persistent=fields["persistent_identity"],
            session_id=self._session_id,
        )
        validated_migration: tuple[str, str, str, str] | None = None
        with self._lock:
            connection = self._connection_or_raise()
            preflight_row = connection.execute(
                "SELECT * FROM twins WHERE twin_id=?", (twin_id,)
            ).fetchone()
            if preflight_row is not None:
                self._verify_twin_row(preflight_row)
                preflight_manifest = str(preflight_row["manifest_sha256"])
                preflight_connector = str(preflight_row["connector_id"])
            else:
                preflight_manifest = ""
                preflight_connector = fields["connector_id"]
        if preflight_manifest and preflight_manifest != fields["manifest_sha256"]:
            request_id = _identifier(
                migration_request_id,
                name="migration_request_id",
            )
            authority, evidence_sha256 = self._validate_manifest_migration_authority(
                evidence=migration_authority_evidence,
                request_id=request_id,
                identity_fingerprint=fields["identity_fingerprint"],
                connector_id=preflight_connector,
                expected_manifest_sha256=preflight_manifest,
                new_manifest_sha256=fields["manifest_sha256"],
                persistent=bool(fields["persistent_identity"]),
            )
            validated_migration = (
                preflight_manifest,
                request_id,
                authority,
                evidence_sha256,
            )
        declaration_manifest = _digest([item.to_dict() for item in declarations])
        base_payload = {
            "twin_id": twin_id,
            "adapter_id": adapter_id,
            "manifest_sha256": fields["manifest_sha256"],
            "declaration_manifest": declaration_manifest,
            "body_projection": list(tuple(getattr(body_projection, "limb_names", ()) or ())),
        }
        with self._lock:
            connection = self._transaction()
            try:
                row = connection.execute(
                    "SELECT * FROM twins WHERE twin_id=?", (twin_id,)
                ).fetchone()
                if row is None:
                    self._insert_twin(
                        connection,
                        twin_id=twin_id,
                        fields=fields,
                        now_ns=self._now_ns(),
                    )
                    self._upsert_node(
                        connection,
                        node_id=_entity_node_id(twin_id),
                        twin_id=twin_id,
                        node_kind=TwinNodeKind.ENTITY,
                        component_kind="physical_entity",
                        external_id=fields["identity_fingerprint"],
                        model={"manifest_sha256": fields["manifest_sha256"]},
                        enabled=True,
                        generation=1,
                    )
                    row = connection.execute(
                        "SELECT * FROM twins WHERE twin_id=?", (twin_id,)
                    ).fetchone()
                if row is None:
                    raise DigitalTwinCorruptionError("new twin did not persist")
                self._verify_twin_row(row)
                current_manifest = str(row["manifest_sha256"])
                if current_manifest != fields["manifest_sha256"]:
                    if (
                        validated_migration is None
                        or validated_migration[0] != current_manifest
                        or str(row["connector_id"]) != preflight_connector
                    ):
                        raise DigitalTwinConflictError(
                            "manifest drift requires exact verified migration authority"
                        )
                    _, request_id, authority, evidence_sha256 = validated_migration
                    row_values = self._twin_values(row)
                    row_values["manifest_sha256"] = fields["manifest_sha256"]
                    row_values["generation"] = int(row["generation"]) + 1
                    row_values["row_sha256"] = self._row_hmac(
                        "twins",
                        {k: v for k, v in row_values.items() if k != "row_sha256"},
                    )
                    self._write_twin_values(connection, row_values)
                    migration_payload = _digest(
                        {
                            "twin_id": twin_id,
                            "expected_manifest_sha256": current_manifest,
                            "new_manifest_sha256": fields["manifest_sha256"],
                            "authority_receipt_id": authority,
                            "authority_evidence_sha256": evidence_sha256,
                            "migration_request_id": request_id,
                        }
                    )
                    migration_event = _event_id("manifest_migrated", migration_payload)
                    migration_version = self._advance_graph_version(connection)
                    self._record_event(
                        connection,
                        event_id=migration_event,
                        twin_id=twin_id,
                        event_kind="manifest_migrated",
                        payload_sha256=migration_payload,
                        disposition=TwinDisposition.ACCEPTED,
                        accepted=True,
                        graph_version=migration_version,
                    )
                    row = connection.execute(
                        "SELECT * FROM twins WHERE twin_id=?", (twin_id,)
                    ).fetchone()
                if row is None:
                    raise DigitalTwinCorruptionError("attached twin disappeared")
                now_ns = self._now_ns()
                existing_binding = connection.execute(
                    "SELECT * FROM twin_adapter_bindings WHERE adapter_id=?",
                    (adapter_id,),
                ).fetchone()
                attachment_generation = 1
                if existing_binding is not None and str(existing_binding["twin_id"]) == twin_id:
                    self._verify_binding_row(existing_binding)
                    current_generation = int(existing_binding["generation"])
                    if bool(existing_binding["active"]):
                        current_payload = {
                            **base_payload,
                            "attachment_generation": current_generation,
                            "attachment_bound_at_ns": int(existing_binding["bound_at_ns"]),
                        }
                        current_payload_sha256 = _digest(current_payload)
                        duplicate = self._existing_event(
                            connection,
                            _event_id("adapter_attached", current_payload_sha256),
                            current_payload_sha256,
                        )
                        if duplicate is not None:
                            self._commit(connection)
                            return duplicate
                    attachment_generation = current_generation + 1
                self._retire_rebound_binding(
                    connection,
                    adapter_id=adapter_id,
                    destination_twin_id=twin_id,
                    now_ns=now_ns,
                )
                payload_sha256 = _digest(
                    {
                        **base_payload,
                        "attachment_generation": attachment_generation,
                        "attachment_bound_at_ns": now_ns,
                    }
                )
                event_id = _event_id("adapter_attached", payload_sha256)
                duplicate = self._existing_event(connection, event_id, payload_sha256)
                if duplicate is not None:
                    self._commit(connection)
                    return duplicate
                topology_revision = self._graph_version(connection) + 1
                self._bind_adapter_topology(
                    connection,
                    twin_id=twin_id,
                    adapter_id=adapter_id,
                    declarations=declarations,
                    body_projection=body_projection,
                    generation=attachment_generation,
                    topology_revision=topology_revision,
                    now_ns=now_ns,
                )
                updated = self._twin_values(row)
                updated.update(
                    {
                        "connector_id": fields["connector_id"],
                        "transport": fields["transport"],
                        "lifecycle": TwinLifecycle.ATTACHED.value,
                        "health": TwinHealth.HEALTHY.value,
                        "last_seen_at_ns": max(int(row["last_seen_at_ns"]), now_ns),
                        "attached_at_ns": now_ns,
                        "detached_at_ns": 0,
                        "revoked_at_ns": 0,
                    }
                )
                updated["row_sha256"] = self._row_hmac(
                    "twins", {k: v for k, v in updated.items() if k != "row_sha256"}
                )
                self._write_twin_values(connection, updated)
                version = self._advance_graph_version(connection)
                receipt = self._record_event(
                    connection,
                    event_id=event_id,
                    twin_id=twin_id,
                    event_kind="adapter_attached",
                    payload_sha256=payload_sha256,
                    disposition=TwinDisposition.ACCEPTED,
                    accepted=True,
                    graph_version=version,
                )
                self._prune_events(connection)
                self._commit(connection)
                return receipt
            except Exception:
                self._rollback(connection)
                raise

    def reconcile_service(self, service: Any) -> tuple[TwinReceipt, ...]:
        ownership_fn = getattr(service, "adapter_channels", None)
        declarations_fn = getattr(service, "declarations", None)
        if not callable(ownership_fn) or not callable(declarations_fn):
            raise TypeError("Reality Reach service inventory is unavailable")
        declarations = {
            item.channel_id: item
            for item in tuple(declarations_fn())
            if isinstance(item, ChannelDeclaration)
        }
        receipts: list[TwinReceipt] = []
        for adapter_id, channel_ids in sorted(dict(ownership_fn()).items()):
            selected = tuple(
                declarations[channel] for channel in channel_ids if channel in declarations
            )
            if selected:
                receipts.append(self._ensure_runtime_adapter(str(adapter_id), selected))
        return tuple(receipts)

    def binding_context(
        self,
        adapter_id: str,
        declaration: ChannelDeclaration,
    ) -> dict[str, Any]:
        """Return the exact attachment fence to bind into a new observation."""

        canonical = _identifier(adapter_id, name="adapter_id")
        if not isinstance(declaration, ChannelDeclaration):
            raise TypeError("declaration must be a ChannelDeclaration")
        with self._lock:
            connection = self._connection_or_raise()
            binding = connection.execute(
                "SELECT * FROM twin_adapter_bindings WHERE adapter_id=?", (canonical,)
            ).fetchone()
        if binding is None:
            self._ensure_runtime_adapter(canonical, (declaration,))
            with self._lock:
                connection = self._connection_or_raise()
                binding = connection.execute(
                    "SELECT * FROM twin_adapter_bindings WHERE adapter_id=?", (canonical,)
                ).fetchone()
        if binding is None:
            raise DigitalTwinCorruptionError("adapter binding was not materialized")
        with self._lock:
            self._verify_binding_row(binding)
            if not bool(binding["active"]):
                raise DigitalTwinConflictError("adapter binding is not active")
            node_id = _channel_node_id(str(binding["twin_id"]), declaration.channel_id)
            node = (
                self._connection_or_raise()
                .execute("SELECT * FROM twin_nodes WHERE node_id=? AND enabled=1", (node_id,))
                .fetchone()
            )
            if node is None:
                raise DigitalTwinConflictError("adapter does not expose the declared channel")
            self._verify_node_row(node)
            if str(node["model_sha256"]) != declaration.sha256:
                raise DigitalTwinConflictError("live declaration differs from twin topology")
            return {
                "twin_id": str(binding["twin_id"]),
                "attachment_generation": int(binding["generation"]),
                "attachment_bound_at_ns": int(binding["bound_at_ns"]),
                "topology_revision": int(binding["topology_revision"]),
            }

    def _ensure_runtime_adapter(
        self,
        adapter_id: str,
        declarations: tuple[ChannelDeclaration, ...],
    ) -> TwinReceipt:
        canonical_adapter = _identifier(adapter_id, name="adapter_id")
        identity = _digest({"session_id": self._session_id, "runtime_adapter": canonical_adapter})
        twin_id = _twin_id(identity, persistent=False, session_id=self._session_id)
        manifest = _digest([item.to_dict() for item in declarations])
        base_payload = {
            "adapter_id": canonical_adapter,
            "declarations": manifest,
            "session_id": self._session_id,
        }
        fields: dict[str, Any] = {
            "candidate_id": canonical_adapter,
            "connector_id": "runtime.inventory",
            "device_id": canonical_adapter,
            "display_name": canonical_adapter,
            "transport": "runtime",
            "identity_fingerprint": identity,
            "manifest_sha256": manifest,
            "persistent_identity": False,
            "privacy_sensitive": True,
            "discovered_at_ns": self._now_ns(),
        }
        with self._lock:
            connection = self._transaction()
            try:
                row = connection.execute(
                    "SELECT * FROM twins WHERE twin_id=?", (twin_id,)
                ).fetchone()
                if row is None:
                    self._insert_twin(
                        connection,
                        twin_id=twin_id,
                        fields=fields,
                        now_ns=fields["discovered_at_ns"],
                    )
                    self._upsert_node(
                        connection,
                        node_id=_entity_node_id(twin_id),
                        twin_id=twin_id,
                        node_kind=TwinNodeKind.ENTITY,
                        component_kind="runtime_physical_entity",
                        external_id=identity,
                        model={"manifest_sha256": manifest, "identity_strength": "session"},
                        enabled=True,
                        generation=1,
                    )
                now_ns = self._now_ns()
                existing_binding = connection.execute(
                    "SELECT * FROM twin_adapter_bindings WHERE adapter_id=?",
                    (canonical_adapter,),
                ).fetchone()
                attachment_generation = 1
                if existing_binding is not None and str(existing_binding["twin_id"]) == twin_id:
                    self._verify_binding_row(existing_binding)
                    current_generation = int(existing_binding["generation"])
                    if bool(existing_binding["active"]):
                        current_payload = {
                            **base_payload,
                            "attachment_generation": current_generation,
                            "attachment_bound_at_ns": int(existing_binding["bound_at_ns"]),
                        }
                        current_payload_sha256 = _digest(current_payload)
                        duplicate = self._existing_event(
                            connection,
                            _event_id("runtime_adapter_reconciled", current_payload_sha256),
                            current_payload_sha256,
                        )
                        if duplicate is not None:
                            self._commit(connection)
                            return duplicate
                    attachment_generation = current_generation + 1
                self._retire_rebound_binding(
                    connection,
                    adapter_id=canonical_adapter,
                    destination_twin_id=twin_id,
                    now_ns=now_ns,
                )
                payload_sha256 = _digest(
                    {
                        **base_payload,
                        "attachment_generation": attachment_generation,
                        "attachment_bound_at_ns": now_ns,
                    }
                )
                event_id = _event_id("runtime_adapter_reconciled", payload_sha256)
                duplicate = self._existing_event(connection, event_id, payload_sha256)
                if duplicate is not None:
                    self._commit(connection)
                    return duplicate
                self._bind_adapter_topology(
                    connection,
                    twin_id=twin_id,
                    adapter_id=canonical_adapter,
                    declarations=declarations,
                    body_projection=None,
                    generation=attachment_generation,
                    topology_revision=self._graph_version(connection) + 1,
                    now_ns=now_ns,
                )
                row = connection.execute(
                    "SELECT * FROM twins WHERE twin_id=?", (twin_id,)
                ).fetchone()
                if row is None:
                    raise DigitalTwinCorruptionError("runtime twin disappeared")
                values = self._twin_values(row)
                values["lifecycle"] = TwinLifecycle.ATTACHED.value
                values["health"] = TwinHealth.HEALTHY.value
                values["attached_at_ns"] = now_ns
                values["row_sha256"] = self._row_hmac(
                    "twins", {k: v for k, v in values.items() if k != "row_sha256"}
                )
                self._write_twin_values(connection, values)
                version = self._advance_graph_version(connection)
                receipt = self._record_event(
                    connection,
                    event_id=event_id,
                    twin_id=twin_id,
                    event_kind="runtime_adapter_reconciled",
                    payload_sha256=payload_sha256,
                    disposition=TwinDisposition.ACCEPTED,
                    accepted=True,
                    graph_version=version,
                )
                self._commit(connection)
                return receipt
            except Exception:
                self._rollback(connection)
                raise

    def observe_observation(self, observation: Any) -> TwinReceipt:
        adapter_id = _identifier(getattr(observation, "adapter_id", ""), name="adapter_id")
        declaration = getattr(observation, "declaration", None)
        reading = getattr(observation, "reading", None)
        if not isinstance(declaration, ChannelDeclaration) or reading is None:
            raise TypeError("digital-twin observation contract is invalid")
        observation_id = _identifier(
            getattr(observation, "observation_id", ""), name="observation_id"
        )
        reading_sha256 = _digest_value(getattr(reading, "sha256", ""), name="reading_sha256")
        historian_record_id = _identifier(
            getattr(observation, "historian_record_id", ""),
            name="historian_record_id",
        )
        payload_sha256 = _digest(
            {
                "observation_id": observation_id,
                "adapter_id": adapter_id,
                "declaration_sha256": declaration.sha256,
                "reading_sha256": reading_sha256,
                "historian_record_id": historian_record_id,
                "historian_quality": str(getattr(observation, "historian_quality", "") or ""),
                "historian_order_basis": str(
                    getattr(observation, "historian_order_basis", "") or ""
                ),
                "historian_order_gap": bool(getattr(observation, "historian_order_gap", False)),
                "historian_alarm_codes": list(
                    tuple(getattr(observation, "historian_alarm_codes", ()) or ())
                ),
                "twin_id": str(getattr(observation, "twin_id", "") or ""),
                "attachment_generation": int(getattr(observation, "attachment_generation", 0) or 0),
                "attachment_bound_at_ns": int(
                    getattr(observation, "attachment_bound_at_ns", 0) or 0
                ),
                "topology_revision": int(getattr(observation, "topology_revision", 0) or 0),
            }
        )
        event_id = _event_id("channel_observed", observation_id)
        supplied_twin_id = _identifier(getattr(observation, "twin_id", ""), name="twin_id")
        supplied_generation = _bounded_ns(
            getattr(observation, "attachment_generation", 0),
            name="attachment_generation",
            minimum=1,
        )
        supplied_bound_at_ns = _bounded_ns(
            getattr(observation, "attachment_bound_at_ns", 0),
            name="attachment_bound_at_ns",
            minimum=1,
        )
        supplied_revision = _bounded_ns(
            getattr(observation, "topology_revision", 0),
            name="topology_revision",
            minimum=1,
        )
        with self._lock:
            connection = self._transaction()
            try:
                duplicate = self._existing_event(connection, event_id, payload_sha256)
                if duplicate is not None:
                    self._commit(connection)
                    return duplicate
                binding = connection.execute(
                    "SELECT * FROM twin_adapter_bindings WHERE adapter_id=?",
                    (adapter_id,),
                ).fetchone()
                if binding is not None:
                    self._verify_binding_row(binding)
                fence_matches = bool(
                    binding is not None
                    and str(binding["twin_id"]) == supplied_twin_id
                    and int(binding["generation"]) == supplied_generation
                    and int(binding["bound_at_ns"]) == supplied_bound_at_ns
                    and int(binding["topology_revision"]) == supplied_revision
                    and bool(binding["active"])
                )
                if not fence_matches:
                    supplied_twin = connection.execute(
                        "SELECT * FROM twins WHERE twin_id=?", (supplied_twin_id,)
                    ).fetchone()
                    if supplied_twin is None:
                        raise DigitalTwinConflictError(
                            "observation attachment fence names an unknown twin"
                        )
                    self._verify_twin_row(supplied_twin)
                    version = self._graph_version(connection)
                    receipt = self._record_event(
                        connection,
                        event_id=event_id,
                        twin_id=supplied_twin_id,
                        event_kind="channel_observed",
                        payload_sha256=payload_sha256,
                        disposition=TwinDisposition.STALE_IGNORED,
                        accepted=True,
                        graph_version=version,
                    )
                    self._commit(connection)
                    return receipt
                if binding is None:
                    raise DigitalTwinCorruptionError("observation has no twin binding")
                twin_id = str(binding["twin_id"])
                node_id = _channel_node_id(twin_id, declaration.channel_id)
                node = connection.execute(
                    "SELECT * FROM twin_nodes WHERE node_id=?", (node_id,)
                ).fetchone()
                if node is None:
                    self._upsert_node(
                        connection,
                        node_id=node_id,
                        twin_id=twin_id,
                        node_kind=TwinNodeKind.CHANNEL,
                        component_kind=declaration.kind.value,
                        external_id=declaration.channel_id,
                        model=declaration.to_dict(),
                        enabled=True,
                        generation=int(binding["generation"]),
                    )
                    self._upsert_relationship(
                        connection,
                        twin_id=twin_id,
                        source_node_id=str(binding["adapter_node_id"]),
                        target_node_id=node_id,
                        kind=TwinRelationshipKind.EXPOSES,
                        generation=int(binding["generation"]),
                    )
                else:
                    self._verify_node_row(node)
                    if str(node["model_sha256"]) != declaration.sha256:
                        raise DigitalTwinConflictError(
                            "observation declaration differs from attached twin model"
                        )
                existing = connection.execute(
                    "SELECT * FROM twin_properties WHERE node_id=?", (node_id,)
                ).fetchone()
                if existing is not None:
                    self._verify_property_row(existing)
                stale = existing is not None and self._observation_is_stale(existing, reading)
                if stale:
                    version = self._graph_version(connection)
                    receipt = self._record_event(
                        connection,
                        event_id=event_id,
                        twin_id=twin_id,
                        event_kind="channel_observed",
                        payload_sha256=payload_sha256,
                        disposition=TwinDisposition.STALE_IGNORED,
                        accepted=True,
                        graph_version=version,
                    )
                    self._commit(connection)
                    return receipt
                self._write_property(
                    connection,
                    twin_id=twin_id,
                    node_id=node_id,
                    observation=observation,
                    existing=existing,
                )
                self._recompute_health(connection, twin_id)
                version = self._advance_graph_version(connection)
                receipt = self._record_event(
                    connection,
                    event_id=event_id,
                    twin_id=twin_id,
                    event_kind="channel_observed",
                    payload_sha256=payload_sha256,
                    disposition=TwinDisposition.ACCEPTED,
                    accepted=True,
                    graph_version=version,
                )
                self._prune_events(connection)
                self._commit(connection)
                return receipt
            except Exception:
                self._rollback(connection)
                raise

    def detach_adapter(
        self,
        adapter_id: str,
        *,
        reason: str = "detached",
        lost: bool = False,
    ) -> TwinReceipt | None:
        canonical = _identifier(adapter_id, name="adapter_id")
        reason_text = str(reason or "detached")[:160]
        with self._lock:
            connection = self._transaction()
            try:
                binding = connection.execute(
                    "SELECT * FROM twin_adapter_bindings WHERE adapter_id=?", (canonical,)
                ).fetchone()
                if binding is None:
                    self._commit(connection)
                    return None
                self._verify_binding_row(binding)
                twin_id = str(binding["twin_id"])
                payload_sha256 = _digest(
                    {
                        "adapter_id": canonical,
                        "twin_id": twin_id,
                        "attachment_generation": int(binding["generation"]),
                        "attachment_bound_at_ns": int(binding["bound_at_ns"]),
                        "topology_revision": int(binding["topology_revision"]),
                        "reason": reason_text,
                        "lost": bool(lost),
                    }
                )
                event_id = _event_id("adapter_detached", payload_sha256)
                duplicate = self._existing_event(connection, event_id, payload_sha256)
                if duplicate is not None:
                    self._commit(connection)
                    return duplicate
                now_ns = self._now_ns()
                binding_values = self._binding_values(binding)
                binding_values.update({"active": 0, "unbound_at_ns": now_ns})
                binding_values["row_sha256"] = self._row_hmac(
                    "twin_adapter_bindings",
                    {k: v for k, v in binding_values.items() if k != "row_sha256"},
                )
                self._write_binding_values(connection, binding_values)
                row = connection.execute(
                    "SELECT * FROM twins WHERE twin_id=?", (twin_id,)
                ).fetchone()
                if row is None:
                    raise DigitalTwinCorruptionError("detached binding has no twin")
                values = self._twin_values(row)
                values.update(
                    {
                        "lifecycle": (
                            TwinLifecycle.LOST.value if lost else TwinLifecycle.DETACHED.value
                        ),
                        "health": TwinHealth.DETACHED.value,
                        "detached_at_ns": now_ns,
                    }
                )
                values["row_sha256"] = self._row_hmac(
                    "twins", {k: v for k, v in values.items() if k != "row_sha256"}
                )
                self._write_twin_values(connection, values)
                version = self._advance_graph_version(connection)
                receipt = self._record_event(
                    connection,
                    event_id=event_id,
                    twin_id=twin_id,
                    event_kind="adapter_detached",
                    payload_sha256=payload_sha256,
                    disposition=TwinDisposition.ACCEPTED,
                    accepted=True,
                    graph_version=version,
                )
                self._commit(connection)
                return receipt
            except Exception:
                self._rollback(connection)
                raise

    def revoke_identity(self, identity_fingerprint: str, *, reason: str) -> TwinReceipt | None:
        identity = _digest_value(identity_fingerprint, name="identity_fingerprint")
        reason_text = str(reason or "revoked")[:160]
        with self._lock:
            connection = self._transaction()
            try:
                rows = connection.execute(
                    "SELECT * FROM twins WHERE identity_fingerprint=? ORDER BY twin_id",
                    (identity,),
                ).fetchall()
                if not rows:
                    self._commit(connection)
                    return None
                if len(rows) > 1:
                    raise DigitalTwinCorruptionError("identity maps to multiple twin scopes")
                row = rows[0]
                self._verify_twin_row(row)
                twin_id = str(row["twin_id"])
                payload_sha256 = _digest(
                    {"twin_id": twin_id, "identity": identity, "reason": reason_text}
                )
                event_id = _event_id("identity_revoked", payload_sha256)
                duplicate = self._existing_event(connection, event_id, payload_sha256)
                if duplicate is not None:
                    self._commit(connection)
                    return duplicate
                now_ns = self._now_ns()
                values = self._twin_values(row)
                values.update(
                    {
                        "lifecycle": TwinLifecycle.REVOKED.value,
                        "health": TwinHealth.REVOKED.value,
                        "revoked_at_ns": now_ns,
                        "detached_at_ns": now_ns,
                    }
                )
                values["row_sha256"] = self._row_hmac(
                    "twins", {k: v for k, v in values.items() if k != "row_sha256"}
                )
                self._write_twin_values(connection, values)
                for binding in connection.execute(
                    "SELECT * FROM twin_adapter_bindings WHERE twin_id=?", (twin_id,)
                ):
                    binding_values = self._binding_values(binding)
                    binding_values.update({"active": 0, "unbound_at_ns": now_ns})
                    binding_values["row_sha256"] = self._row_hmac(
                        "twin_adapter_bindings",
                        {k: v for k, v in binding_values.items() if k != "row_sha256"},
                    )
                    self._write_binding_values(connection, binding_values)
                version = self._advance_graph_version(connection)
                receipt = self._record_event(
                    connection,
                    event_id=event_id,
                    twin_id=twin_id,
                    event_kind="identity_revoked",
                    payload_sha256=payload_sha256,
                    disposition=TwinDisposition.ACCEPTED,
                    accepted=True,
                    graph_version=version,
                )
                self._commit(connection)
                return receipt
            except Exception:
                self._rollback(connection)
                raise

    def _authorize_private_read(
        self,
        authority_capability: Mapping[str, Any] | Any | None,
        *,
        twin_id: str,
        max_nodes: int,
        max_relationships: int,
        include_property_values: bool,
    ) -> None:
        if authority_capability is None:
            raise PermissionError(
                "private twin access requires a one-use signed capability"
            )
        from core.governance.capability_chain import (
            compute_action_digest,
            get_capability_verifier,
        )

        intent = {
            "schema": _PRIVATE_READ_SCHEMA,
            "action": _PRIVATE_READ_ACTION,
            "twin_id": twin_id,
            "max_nodes": int(max_nodes),
            "max_relationships": int(max_relationships),
            "include_property_values": bool(include_property_values),
            "scope": _PRIVATE_READ_SCOPE,
        }
        verifier = self._private_read_capability_verifier or get_capability_verifier()
        result = verifier.verify(
            authority_capability,
            expected_domain="environment_action",
            expected_action_digest=compute_action_digest(_PRIVATE_READ_ACTION, intent),
            consume=True,
        )
        capability = getattr(result, "capability", None)
        if (
            not bool(getattr(result, "ok", False))
            or capability is None
            or str(getattr(capability, "scope", "")) != _PRIVATE_READ_SCOPE
        ):
            denial = getattr(result, "denial", None)
            denial_text = str(getattr(denial, "value", denial) or "invalid")
            raise PermissionError(f"private twin capability rejected: {denial_text}")

    def snapshot(
        self,
        twin_id: str | None = None,
        *,
        max_nodes: int = 512,
        max_relationships: int = 2_048,
        include_property_values: bool = False,
        include_private_topology: bool = False,
        authority_capability: Mapping[str, Any] | Any | None = None,
    ) -> dict[str, Any]:
        node_limit = max(1, min(int(max_nodes), _MAX_QUERY_NODES))
        relationship_limit = max(1, min(int(max_relationships), _MAX_QUERY_RELATIONSHIPS))
        canonical_twin = _identifier(twin_id, name="twin_id") if twin_id else ""
        private_access = include_property_values or include_private_topology
        if private_access:
            self._authorize_private_read(
                authority_capability,
                twin_id=canonical_twin or "*",
                max_nodes=node_limit,
                max_relationships=relationship_limit,
                include_property_values=include_property_values,
            )
        with self._lock:
            connection = self._connection_or_raise()
            twin_rows = connection.execute(
                "SELECT * FROM twins WHERE (?='' OR twin_id=?) "
                "AND (?=1 OR privacy_sensitive=0) ORDER BY twin_id",
                (canonical_twin, canonical_twin, int(private_access)),
            ).fetchall()
            for row in twin_rows:
                self._verify_twin_row(row)
            nodes = connection.execute(
                """
                SELECT * FROM twin_nodes
                WHERE enabled=1 AND (?='' OR twin_id=?)
                  AND (?=1 OR twin_id IN (
                      SELECT twin_id FROM twins WHERE privacy_sensitive=0
                  ))
                ORDER BY twin_id, node_id LIMIT ?
                """,
                (canonical_twin, canonical_twin, int(private_access), node_limit + 1),
            ).fetchall()
            relationships = connection.execute(
                """
                SELECT * FROM twin_relationships
                WHERE enabled=1 AND (?='' OR twin_id=?)
                  AND (?=1 OR twin_id IN (
                      SELECT twin_id FROM twins WHERE privacy_sensitive=0
                  ))
                ORDER BY twin_id, relationship_id LIMIT ?
                """,
                (
                    canonical_twin,
                    canonical_twin,
                    int(private_access),
                    relationship_limit + 1,
                ),
            ).fetchall()
            if len(nodes) > node_limit or len(relationships) > relationship_limit:
                raise DigitalTwinError("digital-twin query exceeds its bounded envelope")
            properties = connection.execute(
                """
                SELECT * FROM twin_properties
                WHERE (?='' OR twin_id=?)
                  AND (?=1 OR twin_id IN (
                      SELECT twin_id FROM twins WHERE privacy_sensitive=0
                  ))
                ORDER BY twin_id, node_id LIMIT ?
                """,
                (canonical_twin, canonical_twin, int(private_access), node_limit + 1),
            ).fetchall()
            for row in nodes:
                self._verify_node_row(row)
            for row in relationships:
                self._verify_relationship_row(row)
            for row in properties:
                self._verify_property_row(row)
            version = self._graph_version(connection)
            payload = {
                "schema": "aura.reality-digital-twin.snapshot.v2",
                "graph_version": version,
                "private_access": private_access,
                "twins": [self._public_twin(row, private=private_access) for row in twin_rows],
                "nodes": [self._public_node(row, private=private_access) for row in nodes],
                "relationships": [self._public_relationship(row) for row in relationships],
                "properties": [
                    self._public_property(row, include_value=include_property_values)
                    for row in properties
                ],
            }
            payload["snapshot_sha256"] = _digest(payload)
            return payload

    def neighbors(
        self,
        node_id: str,
        *,
        max_depth: int = 2,
        max_nodes: int = 256,
        include_private_topology: bool = False,
        authority_capability: Mapping[str, Any] | Any | None = None,
    ) -> dict[str, Any]:
        start = _identifier(node_id, name="node_id")
        depth_limit = max(0, min(int(max_depth), 8))
        node_limit = max(1, min(int(max_nodes), _MAX_QUERY_NODES))
        with self._lock:
            connection = self._connection_or_raise()
            start_row = connection.execute(
                "SELECT * FROM twin_nodes WHERE node_id=? AND enabled=1", (start,)
            ).fetchone()
            if start_row is None:
                raise LookupError(f"unknown active twin node: {start}")
            self._verify_node_row(start_row)
            twin_row = connection.execute(
                "SELECT * FROM twins WHERE twin_id=?", (str(start_row["twin_id"]),)
            ).fetchone()
            if twin_row is None:
                raise DigitalTwinCorruptionError("neighborhood twin is missing")
            self._verify_twin_row(twin_row)
            sensitive = bool(twin_row["privacy_sensitive"])
            canonical_twin_id = str(start_row["twin_id"])
        private_access = bool(include_private_topology)
        if sensitive and not private_access:
            raise LookupError(f"unknown active twin node: {start}")
        if private_access:
            self._authorize_private_read(
                authority_capability,
                twin_id=canonical_twin_id,
                max_nodes=node_limit,
                max_relationships=min(_MAX_QUERY_RELATIONSHIPS, node_limit * 8),
                include_property_values=False,
            )
        with self._lock:
            connection = self._connection_or_raise()
            start_row = connection.execute(
                "SELECT * FROM twin_nodes WHERE node_id=? AND enabled=1", (start,)
            ).fetchone()
            if start_row is None or str(start_row["twin_id"]) != canonical_twin_id:
                raise DigitalTwinConflictError(
                    "twin topology changed while private read authority was verified"
                )
            self._verify_node_row(start_row)
            visited = {start}
            frontier: deque[tuple[str, int]] = deque([(start, 0)])
            relationships: dict[str, sqlite3.Row] = {}
            while frontier:
                current, depth = frontier.popleft()
                if depth >= depth_limit:
                    continue
                rows = connection.execute(
                    """
                    SELECT * FROM twin_relationships
                    WHERE enabled=1 AND (source_node_id=? OR target_node_id=?)
                    ORDER BY relationship_id
                    """,
                    (current, current),
                ).fetchall()
                for row in rows:
                    self._verify_relationship_row(row)
                    relationships[str(row["relationship_id"])] = row
                    other = (
                        str(row["target_node_id"])
                        if str(row["source_node_id"]) == current
                        else str(row["source_node_id"])
                    )
                    if other not in visited:
                        visited.add(other)
                        if len(visited) > node_limit:
                            raise DigitalTwinError("twin traversal exceeds its node bound")
                        frontier.append((other, depth + 1))
            placeholders = ",".join("?" for _ in visited)
            nodes = connection.execute(
                f"SELECT * FROM twin_nodes WHERE node_id IN ({placeholders}) ORDER BY node_id",
                tuple(sorted(visited)),
            ).fetchall()
            for row in nodes:
                self._verify_node_row(row)
            payload = {
                "schema": "aura.reality-digital-twin.neighborhood.v2",
                "start_node_id": start,
                "max_depth": depth_limit,
                "private_access": private_access,
                "nodes": [self._public_node(row, private=private_access) for row in nodes],
                "relationships": [
                    self._public_relationship(row)
                    for row in sorted(
                        relationships.values(), key=lambda item: str(item["relationship_id"])
                    )
                ],
            }
            payload["snapshot_sha256"] = _digest(payload)
            return payload

    def status(self) -> dict[str, Any]:
        with self._lock:
            try:
                connection = self._connection_or_raise()
                counts = {
                    "twins": int(connection.execute("SELECT COUNT(*) FROM twins").fetchone()[0]),
                    "nodes": int(
                        connection.execute("SELECT COUNT(*) FROM twin_nodes").fetchone()[0]
                    ),
                    "relationships": int(
                        connection.execute("SELECT COUNT(*) FROM twin_relationships").fetchone()[0]
                    ),
                    "properties": int(
                        connection.execute("SELECT COUNT(*) FROM twin_properties").fetchone()[0]
                    ),
                    "events": int(
                        connection.execute("SELECT COUNT(*) FROM twin_events").fetchone()[0]
                    ),
                    "event_segments": int(
                        connection.execute(
                            "SELECT COUNT(*) FROM twin_event_segments"
                        ).fetchone()[0]
                    ),
                    "archived_lifecycle_events": int(
                        connection.execute(
                            "SELECT value FROM twin_meta "
                            "WHERE key='lifecycle_archive_last_event_sequence'"
                        ).fetchone()[0]
                    ),
                }
                lifecycle = {
                    str(row[0]): int(row[1])
                    for row in connection.execute(
                        "SELECT lifecycle, COUNT(*) FROM twins GROUP BY lifecycle"
                    )
                }
                lifecycle_sequence_row = connection.execute(
                    "SELECT value FROM twin_meta WHERE key='lifecycle_sequence'"
                ).fetchone()
                lifecycle_head_row = connection.execute(
                    "SELECT value FROM twin_meta WHERE key='lifecycle_head_sha256'"
                ).fetchone()
                if lifecycle_sequence_row is None or lifecycle_head_row is None:
                    raise DigitalTwinCorruptionError(
                        "digital-twin lifecycle chain head is missing"
                    )
                lifecycle_sequence = int(lifecycle_sequence_row[0])
                lifecycle_head_sha256 = _digest_value(
                    lifecycle_head_row[0], name="lifecycle_head_sha256"
                )
                archive_head_row = connection.execute(
                    "SELECT value FROM twin_meta WHERE key='lifecycle_archive_head_sha256'"
                ).fetchone()
                if archive_head_row is None:
                    raise DigitalTwinCorruptionError(
                        "digital-twin lifecycle archive head is missing"
                    )
                archive_head_sha256 = _digest_value(
                    archive_head_row[0], name="lifecycle_archive_head_sha256"
                )
                storage_files = self._storage_file_bytes()
                lifecycle_drift = int(
                    connection.execute(
                        """
                        SELECT COUNT(*) FROM twin_adapter_bindings b
                        JOIN twins t ON t.twin_id=b.twin_id
                        WHERE (b.active=1 AND t.lifecycle NOT IN ('attached', 'degraded'))
                           OR (b.active=0 AND t.lifecycle='attached')
                        """
                    ).fetchone()[0]
                )
                return {
                    "status": "active" if self._ready else "degraded",
                    "alive": self._connection is not None,
                    "ready": self._ready,
                    "schema_version": _SCHEMA_VERSION,
                    "schema_sha256": _digest(_SCHEMA),
                    "migration_state": "complete",
                    "graph_version": self._graph_version(connection),
                    "counts": counts,
                    "lifecycle": lifecycle,
                    "lifecycle_event_sequence": lifecycle_sequence,
                    "lifecycle_head_sha256": lifecycle_head_sha256,
                    "lifecycle_archive_head_sha256": archive_head_sha256,
                    "invalid_events": 0,
                    "persistent_identity_count": int(
                        connection.execute(
                            "SELECT COUNT(*) FROM twins WHERE persistent_identity=1"
                        ).fetchone()[0]
                    ),
                    "active_bindings": int(
                        connection.execute(
                            "SELECT COUNT(*) FROM twin_adapter_bindings WHERE active=1"
                        ).fetchone()[0]
                    ),
                    "lifecycle_drift": lifecycle_drift,
                    "storage_bytes": sum(storage_files.values()),
                    "storage_files": storage_files,
                    "max_storage_bytes": self._max_storage_bytes,
                    "min_free_bytes": self._min_free_bytes,
                    "disk_free_bytes": self._disk_free_bytes(),
                    "last_integrity_check_ns": self._last_integrity_check_ns,
                    "last_error": self._last_error,
                    "actuation_authority": False,
                }
            except (DigitalTwinError, OSError, sqlite3.Error, TypeError, ValueError) as exc:
                self._ready = False
                self._last_error = f"{type(exc).__name__}:{exc}"[:320]
                return {
                    "status": "degraded",
                    "alive": self._connection is not None,
                    "ready": False,
                    "last_error": self._last_error,
                    "schema_sha256": _digest(_SCHEMA),
                    "migration_state": "failed",
                    "actuation_authority": False,
                }

    def get_status(self) -> dict[str, Any]:
        return self.status()

    def health_snapshot(self) -> dict[str, Any]:
        """Return lock-only readiness for event-loop status aggregation."""

        with self._lock:
            return {
                "status": "active" if self._ready else "degraded",
                "alive": self._connection is not None,
                "ready": self._ready and self._connection is not None,
                "graph_version": self._graph_version_cache,
                "schema_sha256": _digest(_SCHEMA),
                "migration_state": "complete" if self._ready else "failed",
                "last_integrity_check_ns": self._last_integrity_check_ns,
                "last_error": self._last_error,
                "actuation_authority": False,
            }

    def is_alive(self) -> bool:
        with self._lock:
            return self._connection is not None

    def is_ready(self) -> bool:
        with self._lock:
            return self._ready and self._connection is not None

    def probe_health(self, *, full: bool = False) -> bool:
        with self._lock:
            try:
                connection = self._connection
                if connection is None:
                    raise DigitalTwinError("digital-twin graph is unavailable")
                if self._pending_archive is not None or self._archive_pending_path.exists():
                    self._recover_archive_transaction(connection)
                    self._cleanup_unreferenced_archive_files(connection)
                self._validate_schema(connection)
                self._verify_integrity(connection, full=full)
                self._graph_version_cache = self._graph_version(connection)
                self._ready = True
                self._last_error = ""
                return True
            except (DigitalTwinError, OSError, sqlite3.Error, TypeError, ValueError) as exc:
                self._ready = False
                self._last_error = f"{type(exc).__name__}:{exc}"[:320]
                return False

    def close(self) -> None:
        with self._lock:
            if self._connection is not None:
                self._connection.close()
            self._connection = None
            self._ready = False

    def _now_ns(self) -> int:
        value = self._clock_ns()
        return _bounded_ns(value, name="clock_ns", minimum=1)

    def _storage_file_bytes(self) -> dict[str, int]:
        paths = {
            "database": self.db_path,
            "wal": Path(f"{self.db_path}-wal"),
            "shm": Path(f"{self.db_path}-shm"),
        }
        result: dict[str, int] = {}
        for label, path in paths.items():
            try:
                result[label] = int(path.stat().st_size)
            # not a failure: a path that cannot be stat'ed contributes nothing to the count.
            except OSError:
                result[label] = 0
        archive_bytes = 0
        try:
            for path in self._archive_dir.iterdir():
                if path.is_file() and not path.is_symlink():
                    archive_bytes += int(path.stat().st_size)
        # not a failure: a path that cannot be stat'ed contributes nothing to the count.
        except OSError:
            archive_bytes = 0
        result["lifecycle_archive"] = archive_bytes
        return result

    def _disk_free_bytes(self) -> int:
        try:
            values = os.statvfs(self.db_path.parent)
        except OSError as exc:
            raise DigitalTwinError("digital-twin disk headroom is unavailable") from exc
        return int(values.f_bavail * values.f_frsize)

    def _ensure_storage_headroom(self) -> None:
        used = sum(self._storage_file_bytes().values())
        if used > self._max_storage_bytes:
            raise DigitalTwinError("digital-twin storage bound exceeded")
        if self._disk_free_bytes() < self._min_free_bytes:
            raise DigitalTwinError("digital-twin disk headroom is below its reserve")

    def _ensure_capacity(
        self,
        connection: sqlite3.Connection,
        *,
        nodes: int = 0,
        relationships: int = 0,
    ) -> None:
        node_count = int(connection.execute("SELECT COUNT(*) FROM twin_nodes").fetchone()[0])
        relation_count = int(
            connection.execute("SELECT COUNT(*) FROM twin_relationships").fetchone()[0]
        )
        if node_count + nodes > self._max_nodes:
            raise DigitalTwinError("digital-twin node capacity exhausted")
        if relation_count + relationships > self._max_relationships:
            raise DigitalTwinError("digital-twin relationship capacity exhausted")

    def _insert_twin(
        self,
        connection: sqlite3.Connection,
        *,
        twin_id: str,
        fields: Mapping[str, Any],
        now_ns: int,
    ) -> None:
        scope = "persistent" if fields["persistent_identity"] else self._session_id
        values: dict[str, Any] = {
            "twin_id": twin_id,
            "identity_fingerprint": fields["identity_fingerprint"],
            "identity_scope": scope,
            "connector_id": fields["connector_id"],
            "device_identity_sha256": _digest(str(fields["device_id"])),
            "display_name_sha256": _digest(str(fields["display_name"])),
            "transport": fields["transport"],
            "manifest_sha256": fields["manifest_sha256"],
            "persistent_identity": int(bool(fields["persistent_identity"])),
            "privacy_sensitive": int(bool(fields["privacy_sensitive"])),
            "lifecycle": TwinLifecycle.DISCOVERED.value,
            "health": TwinHealth.UNKNOWN.value,
            "generation": 1,
            "discovered_at_ns": int(fields["discovered_at_ns"]),
            "last_seen_at_ns": max(int(fields["discovered_at_ns"]), now_ns),
            "attached_at_ns": 0,
            "detached_at_ns": 0,
            "revoked_at_ns": 0,
        }
        values["row_sha256"] = self._row_hmac("twins", values)
        columns = ",".join(values)
        placeholders = ",".join("?" for _ in values)
        connection.execute(
            f"INSERT INTO twins({columns}) VALUES({placeholders})", tuple(values.values())
        )

    def _update_twin_seen(
        self,
        connection: sqlite3.Connection,
        row: sqlite3.Row,
        fields: Mapping[str, Any],
        *,
        now_ns: int,
    ) -> None:
        values = self._twin_values(row)
        values.update(
            {
                "connector_id": fields["connector_id"],
                "device_identity_sha256": _digest(str(fields["device_id"])),
                "display_name_sha256": _digest(str(fields["display_name"])),
                "transport": fields["transport"],
                "privacy_sensitive": max(
                    int(bool(row["privacy_sensitive"])),
                    int(bool(fields["privacy_sensitive"])),
                ),
                "last_seen_at_ns": max(int(row["last_seen_at_ns"]), now_ns),
            }
        )
        values["row_sha256"] = self._row_hmac(
            "twins", {k: v for k, v in values.items() if k != "row_sha256"}
        )
        self._write_twin_values(connection, values)

    @staticmethod
    def _twin_values(row: sqlite3.Row) -> dict[str, Any]:
        return {name: row[name] for name in _EXPECTED_COLUMNS["twins"]}

    @staticmethod
    def _binding_values(row: sqlite3.Row) -> dict[str, Any]:
        return {name: row[name] for name in _EXPECTED_COLUMNS["twin_adapter_bindings"]}

    def _write_twin_values(
        self,
        connection: sqlite3.Connection,
        values: Mapping[str, Any],
    ) -> None:
        existing = connection.execute(
            "SELECT * FROM twins WHERE twin_id=?", (values["twin_id"],)
        ).fetchone()
        if existing is None:
            raise DigitalTwinCorruptionError("twin disappeared before update")
        self._verify_twin_row(existing)
        assignments = ",".join(f"{key}=?" for key in values if key != "twin_id")
        parameters = [values[key] for key in values if key != "twin_id"]
        parameters.append(values["twin_id"])
        connection.execute(f"UPDATE twins SET {assignments} WHERE twin_id=?", tuple(parameters))

    def _write_binding_values(
        self,
        connection: sqlite3.Connection,
        values: Mapping[str, Any],
    ) -> None:
        existing = connection.execute(
            "SELECT * FROM twin_adapter_bindings WHERE adapter_id=?",
            (values["adapter_id"],),
        ).fetchone()
        if existing is None:
            raise DigitalTwinCorruptionError("adapter binding disappeared before update")
        self._verify_binding_row(existing)
        assignments = ",".join(f"{key}=?" for key in values if key != "adapter_id")
        parameters = [values[key] for key in values if key != "adapter_id"]
        parameters.append(values["adapter_id"])
        connection.execute(
            f"UPDATE twin_adapter_bindings SET {assignments} WHERE adapter_id=?",
            tuple(parameters),
        )

    def _retire_rebound_binding(
        self,
        connection: sqlite3.Connection,
        *,
        adapter_id: str,
        destination_twin_id: str,
        now_ns: int,
    ) -> None:
        binding = connection.execute(
            "SELECT * FROM twin_adapter_bindings WHERE adapter_id=?", (adapter_id,)
        ).fetchone()
        if binding is None or str(binding["twin_id"]) == destination_twin_id:
            return
        self._verify_binding_row(binding)
        prior_twin_id = str(binding["twin_id"])
        prior_twin = connection.execute(
            "SELECT * FROM twins WHERE twin_id=?", (prior_twin_id,)
        ).fetchone()
        if prior_twin is None:
            raise DigitalTwinCorruptionError("rebound adapter has no prior twin")
        self._verify_twin_row(prior_twin)
        prior_values = self._twin_values(prior_twin)
        if prior_values["lifecycle"] != TwinLifecycle.REVOKED.value:
            prior_values.update(
                {
                    "lifecycle": TwinLifecycle.LOST.value,
                    "health": TwinHealth.DETACHED.value,
                    "detached_at_ns": now_ns,
                }
            )
            prior_values["row_sha256"] = self._row_hmac(
                "twins",
                {key: value for key, value in prior_values.items() if key != "row_sha256"},
            )
            self._write_twin_values(connection, prior_values)
        payload_sha256 = _digest(
            {
                "adapter_id": adapter_id,
                "prior_twin_id": prior_twin_id,
                "destination_twin_id": destination_twin_id,
                "prior_topology_revision": int(binding["topology_revision"]),
            }
        )
        event_id = _event_id("adapter_rebound", payload_sha256)
        if self._existing_event(connection, event_id, payload_sha256) is None:
            version = self._advance_graph_version(connection)
            self._record_event(
                connection,
                event_id=event_id,
                twin_id=prior_twin_id,
                event_kind="adapter_rebound",
                payload_sha256=payload_sha256,
                disposition=TwinDisposition.ACCEPTED,
                accepted=True,
                graph_version=version,
            )
        connection.execute("DELETE FROM twin_adapter_bindings WHERE adapter_id=?", (adapter_id,))

    def _bind_adapter_topology(
        self,
        connection: sqlite3.Connection,
        *,
        twin_id: str,
        adapter_id: str,
        declarations: tuple[ChannelDeclaration, ...],
        body_projection: Any | None,
        generation: int,
        topology_revision: int,
        now_ns: int,
    ) -> None:
        existing_binding = connection.execute(
            "SELECT * FROM twin_adapter_bindings WHERE adapter_id=?", (adapter_id,)
        ).fetchone()
        if existing_binding is not None:
            self._verify_binding_row(existing_binding)
            if str(existing_binding["twin_id"]) != twin_id:
                raise DigitalTwinConflictError(
                    "adapter binding must be retired before reassignment"
                )
        entity_node = _entity_node_id(twin_id)
        adapter_node = _adapter_node_id(twin_id)
        existing_nodes = {
            str(row[0])
            for row in connection.execute(
                "SELECT node_id FROM twin_nodes WHERE twin_id=?", (twin_id,)
            )
        }
        desired_nodes = {entity_node, adapter_node}
        desired_nodes.update(_channel_node_id(twin_id, item.channel_id) for item in declarations)
        limb_names = tuple(getattr(body_projection, "limb_names", ()) or ())
        desired_nodes.update(_body_node_id(twin_id, name) for name in limb_names)
        self._ensure_capacity(connection, nodes=len(desired_nodes - existing_nodes))
        self._upsert_node(
            connection,
            node_id=adapter_node,
            twin_id=twin_id,
            node_kind=TwinNodeKind.ADAPTER,
            component_kind="reality_reach_adapter",
            external_id=adapter_id,
            model={
                "adapter_id_sha256": _digest(adapter_id),
                "declaration_manifest_sha256": _digest([item.to_dict() for item in declarations]),
            },
            enabled=True,
            generation=generation,
        )
        self._upsert_relationship(
            connection,
            twin_id=twin_id,
            source_node_id=entity_node,
            target_node_id=adapter_node,
            kind=TwinRelationshipKind.CONTAINS,
            generation=generation,
        )
        for declaration in declarations:
            channel_node = _channel_node_id(twin_id, declaration.channel_id)
            self._upsert_node(
                connection,
                node_id=channel_node,
                twin_id=twin_id,
                node_kind=TwinNodeKind.CHANNEL,
                component_kind=declaration.kind.value,
                external_id=declaration.channel_id,
                model=declaration.to_dict(),
                enabled=True,
                generation=generation,
            )
            self._upsert_relationship(
                connection,
                twin_id=twin_id,
                source_node_id=adapter_node,
                target_node_id=channel_node,
                kind=TwinRelationshipKind.EXPOSES,
                generation=generation,
            )
        for index, limb_name in enumerate(limb_names):
            if not isinstance(limb_name, str) or not limb_name:
                raise ValueError("body projection limb names must be non-empty")
            limb_node = _body_node_id(twin_id, limb_name)
            self._upsert_node(
                connection,
                node_id=limb_node,
                twin_id=twin_id,
                node_kind=TwinNodeKind.BODY_LIMB,
                component_kind="somatic_projection",
                external_id=limb_name,
                model={"limb_name_sha256": _digest(limb_name)},
                enabled=True,
                generation=generation,
            )
            if index < len(declarations):
                self._upsert_relationship(
                    connection,
                    twin_id=twin_id,
                    source_node_id=_channel_node_id(twin_id, declarations[index].channel_id),
                    target_node_id=limb_node,
                    kind=TwinRelationshipKind.PROJECTS_TO,
                    generation=generation,
                )
        for node_id in existing_nodes - desired_nodes:
            node = connection.execute(
                "SELECT * FROM twin_nodes WHERE node_id=?", (node_id,)
            ).fetchone()
            if node is None or str(node["node_kind"]) == TwinNodeKind.ENTITY.value:
                continue
            self._set_node_enabled(connection, node, enabled=False)
            self._disable_node_relationships(connection, node_id)
        binding_values = {
            "adapter_id": adapter_id,
            "twin_id": twin_id,
            "adapter_node_id": adapter_node,
            "active": 1,
            "generation": generation,
            "topology_revision": topology_revision,
            "bound_at_ns": now_ns,
            "unbound_at_ns": 0,
        }
        binding_values["row_sha256"] = self._row_hmac(
            "twin_adapter_bindings", binding_values
        )
        columns = ",".join(binding_values)
        placeholders = ",".join("?" for _ in binding_values)
        updates = ",".join(f"{key}=excluded.{key}" for key in binding_values if key != "adapter_id")
        connection.execute(
            f"""
            INSERT INTO twin_adapter_bindings({columns}) VALUES({placeholders})
            ON CONFLICT(adapter_id) DO UPDATE SET {updates}
            """,
            tuple(binding_values.values()),
        )

    def _upsert_node(
        self,
        connection: sqlite3.Connection,
        *,
        node_id: str,
        twin_id: str,
        node_kind: TwinNodeKind,
        component_kind: str,
        external_id: str,
        model: Mapping[str, Any],
        enabled: bool,
        generation: int,
    ) -> None:
        existing = connection.execute(
            "SELECT * FROM twin_nodes WHERE node_id=?", (node_id,)
        ).fetchone()
        if existing is not None:
            self._verify_node_row(existing)
            if str(existing["twin_id"]) != twin_id:
                raise DigitalTwinConflictError("node identity crosses twin boundaries")
        model_json = _canonical_json(dict(model), name="twin node model")
        values: dict[str, Any] = {
            "node_id": node_id,
            "twin_id": twin_id,
            "node_kind": node_kind.value,
            "component_kind": _identifier(component_kind, name="component_kind"),
            "external_id_sha256": (
                external_id if _DIGEST.fullmatch(external_id) else _digest(external_id)
            ),
            "model_sha256": _digest(json.loads(model_json)),
            "model_json": model_json,
            "enabled": int(enabled),
            "generation": generation,
        }
        values["row_sha256"] = self._row_hmac("twin_nodes", values)
        columns = ",".join(values)
        placeholders = ",".join("?" for _ in values)
        updates = ",".join(
            f"{key}=excluded.{key}" for key in values if key not in {"node_id", "twin_id"}
        )
        connection.execute(
            f"""
            INSERT INTO twin_nodes({columns}) VALUES({placeholders})
            ON CONFLICT(node_id) DO UPDATE SET {updates}
            """,
            tuple(values.values()),
        )

    def _set_node_enabled(
        self, connection: sqlite3.Connection, row: sqlite3.Row, *, enabled: bool
    ) -> None:
        self._verify_node_row(row)
        values = {name: row[name] for name in _EXPECTED_COLUMNS["twin_nodes"]}
        values["enabled"] = int(enabled)
        values["row_sha256"] = self._row_hmac(
            "twin_nodes", {k: v for k, v in values.items() if k != "row_sha256"}
        )
        connection.execute(
            "UPDATE twin_nodes SET enabled=?, row_sha256=? WHERE node_id=?",
            (values["enabled"], values["row_sha256"], values["node_id"]),
        )

    def _disable_node_relationships(self, connection: sqlite3.Connection, node_id: str) -> None:
        rows = connection.execute(
            """
            SELECT * FROM twin_relationships
            WHERE source_node_id=? OR target_node_id=?
            """,
            (node_id, node_id),
        ).fetchall()
        for row in rows:
            self._verify_relationship_row(row)
            values = {name: row[name] for name in _EXPECTED_COLUMNS["twin_relationships"]}
            values["enabled"] = 0
            values["row_sha256"] = self._row_hmac(
                "twin_relationships",
                {key: value for key, value in values.items() if key != "row_sha256"},
            )
            connection.execute(
                """
                UPDATE twin_relationships SET enabled=?, row_sha256=?
                WHERE relationship_id=?
                """,
                (0, values["row_sha256"], values["relationship_id"]),
            )

    def _upsert_relationship(
        self,
        connection: sqlite3.Connection,
        *,
        twin_id: str,
        source_node_id: str,
        target_node_id: str,
        kind: TwinRelationshipKind,
        generation: int,
    ) -> None:
        source = connection.execute(
            "SELECT * FROM twin_nodes WHERE node_id=?",
            (source_node_id,),
        ).fetchone()
        target = connection.execute(
            "SELECT * FROM twin_nodes WHERE node_id=?",
            (target_node_id,),
        ).fetchone()
        if source is None or target is None:
            raise DigitalTwinConflictError("relationship endpoint is missing")
        self._verify_node_row(source)
        self._verify_node_row(target)
        if str(source["twin_id"]) != twin_id or str(target["twin_id"]) != twin_id:
            raise DigitalTwinConflictError("relationship crosses twin boundaries")
        allowed_endpoints = {
            TwinRelationshipKind.CONTAINS: (
                TwinNodeKind.ENTITY.value,
                TwinNodeKind.ADAPTER.value,
            ),
            TwinRelationshipKind.EXPOSES: (
                TwinNodeKind.ADAPTER.value,
                TwinNodeKind.CHANNEL.value,
            ),
            TwinRelationshipKind.PROJECTS_TO: (
                TwinNodeKind.CHANNEL.value,
                TwinNodeKind.BODY_LIMB.value,
            ),
        }
        if (str(source["node_kind"]), str(target["node_kind"])) != allowed_endpoints[kind]:
            raise DigitalTwinConflictError("relationship endpoint types are invalid")
        relationship_id = _relationship_id(twin_id, source_node_id, target_node_id, kind)
        existing = connection.execute(
            "SELECT * FROM twin_relationships WHERE relationship_id=?",
            (relationship_id,),
        ).fetchone()
        if existing is not None:
            self._verify_relationship_row(existing)
            if str(existing["twin_id"]) != twin_id:
                raise DigitalTwinConflictError(
                    "relationship identity crosses twin boundaries"
                )
        else:
            self._ensure_capacity(connection, relationships=1)
        values: dict[str, Any] = {
            "relationship_id": relationship_id,
            "twin_id": twin_id,
            "source_node_id": source_node_id,
            "target_node_id": target_node_id,
            "relationship_kind": kind.value,
            "properties_json": "{}",
            "enabled": 1,
            "generation": generation,
        }
        values["row_sha256"] = self._row_hmac("twin_relationships", values)
        columns = ",".join(values)
        placeholders = ",".join("?" for _ in values)
        updates = ",".join(
            f"{key}=excluded.{key}" for key in values if key not in {"relationship_id", "twin_id"}
        )
        connection.execute(
            f"""
            INSERT INTO twin_relationships({columns}) VALUES({placeholders})
            ON CONFLICT(relationship_id) DO UPDATE SET {updates}
            """,
            tuple(values.values()),
        )

    @staticmethod
    def _observation_is_stale(existing: sqlite3.Row, reading: Any) -> bool:
        existing_epoch = str(existing["source_epoch"])
        incoming_epoch = str(getattr(reading, "source_epoch", "") or "")
        existing_sequence = int(existing["source_sequence"])
        incoming_sequence = int(getattr(reading, "source_sequence", 0) or 0)
        if existing_epoch and incoming_epoch and existing_epoch == incoming_epoch:
            if existing_sequence and incoming_sequence:
                return incoming_sequence < existing_sequence
        captured_at_ns = int(getattr(reading, "captured_at_ns", 0) or 0)
        return captured_at_ns < int(existing["captured_at_ns"])

    def _write_property(
        self,
        connection: sqlite3.Connection,
        *,
        twin_id: str,
        node_id: str,
        observation: Any,
        existing: sqlite3.Row | None,
    ) -> None:
        reading = observation.reading
        if existing is not None:
            self._verify_property_row(existing)
            existing_epoch = str(existing["source_epoch"])
            incoming_epoch = str(getattr(reading, "source_epoch", "") or "")
            existing_sequence = int(existing["source_sequence"])
            incoming_sequence = int(getattr(reading, "source_sequence", 0) or 0)
            if (
                existing_epoch
                and incoming_epoch == existing_epoch
                and existing_sequence
                and incoming_sequence == existing_sequence
                and str(existing["reading_sha256"]) != reading.sha256
            ):
                raise DigitalTwinConflictError(
                    "same source epoch and sequence carry different readings"
                )
        value = getattr(reading, "value", None)
        if value is not None and (isinstance(value, bool) or not math.isfinite(float(value))):
            raise ValueError("digital-twin numeric property must be finite")
        values: dict[str, Any] = {
            "node_id": node_id,
            "twin_id": twin_id,
            "observation_id": observation.observation_id,
            "historian_record_id": observation.historian_record_id,
            "reading_sha256": reading.sha256,
            "declaration_sha256": observation.declaration.sha256,
            "value_json": _canonical_json(value, name="twin property value"),
            "unit": str(reading.unit),
            "status": str(reading.status.value),
            "quality": str(observation.historian_quality),
            "order_basis": str(observation.historian_order_basis),
            "order_gap": int(bool(observation.historian_order_gap)),
            "alarm_codes_json": _canonical_json(
                list(observation.historian_alarm_codes), name="twin alarm codes"
            ),
            "captured_at_ns": int(reading.captured_at_ns),
            "source_epoch": str(getattr(reading, "source_epoch", "") or ""),
            "source_sequence": int(getattr(reading, "source_sequence", 0) or 0),
            "source_event_id": str(getattr(reading, "source_event_id", "") or "")[:192],
            "version": 1 if existing is None else int(existing["version"]) + 1,
            "updated_at_ns": self._now_ns(),
        }
        values["row_sha256"] = self._row_hmac("twin_properties", values)
        columns = ",".join(values)
        placeholders = ",".join("?" for _ in values)
        updates = ",".join(f"{key}=excluded.{key}" for key in values if key != "node_id")
        connection.execute(
            f"""
            INSERT INTO twin_properties({columns}) VALUES({placeholders})
            ON CONFLICT(node_id) DO UPDATE SET {updates}
            """,
            tuple(values.values()),
        )

    def _recompute_health(self, connection: sqlite3.Connection, twin_id: str) -> None:
        row = connection.execute("SELECT * FROM twins WHERE twin_id=?", (twin_id,)).fetchone()
        if row is None:
            raise DigitalTwinCorruptionError("property twin is missing")
        self._verify_twin_row(row)
        property_rows = connection.execute(
            "SELECT * FROM twin_properties WHERE twin_id=?", (twin_id,)
        ).fetchall()
        for property_row in property_rows:
            self._verify_property_row(property_row)
        statuses = {str(item["status"]) for item in property_rows}
        qualities = {str(item["quality"]) for item in property_rows}
        if statuses & {"unavailable", "permission_denied"}:
            health = TwinHealth.UNAVAILABLE
            lifecycle = TwinLifecycle.DEGRADED
        elif statuses & {"degraded", "stale", "uncalibrated"} or qualities & {
            "bad",
            "stale",
            "uncertain",
        }:
            health = TwinHealth.DEGRADED
            lifecycle = TwinLifecycle.DEGRADED
        else:
            health = TwinHealth.HEALTHY
            lifecycle = TwinLifecycle.ATTACHED
        values = self._twin_values(row)
        values.update({"health": health.value, "lifecycle": lifecycle.value})
        values["row_sha256"] = self._row_hmac(
            "twins", {k: v for k, v in values.items() if k != "row_sha256"}
        )
        self._write_twin_values(connection, values)

    def _record_event(
        self,
        connection: sqlite3.Connection,
        *,
        event_id: str,
        twin_id: str,
        event_kind: str,
        payload_sha256: str,
        disposition: TwinDisposition,
        accepted: bool,
        graph_version: int,
    ) -> TwinReceipt:
        self._make_event_space(connection, event_kind=event_kind)
        state_sha256 = self._state_sha256(connection, twin_id)
        receipt_id = _receipt_id(event_id, payload_sha256, twin_id)
        canonical_kind = _identifier(event_kind, name="event_kind")
        created_at_ns = self._now_ns()
        lifecycle_sequence = 0
        prior_lifecycle_sha256 = _NO_LIFECYCLE_SHA256
        lifecycle_sha256 = _NO_LIFECYCLE_SHA256
        if canonical_kind not in _PRUNABLE_EVENT_KINDS:
            sequence_row = connection.execute(
                "SELECT value FROM twin_meta WHERE key='lifecycle_sequence'"
            ).fetchone()
            head_row = connection.execute(
                "SELECT value FROM twin_meta WHERE key='lifecycle_head_sha256'"
            ).fetchone()
            if sequence_row is None or head_row is None:
                raise DigitalTwinCorruptionError("digital-twin lifecycle chain head is missing")
            try:
                lifecycle_sequence = int(sequence_row[0]) + 1
            except (TypeError, ValueError) as exc:
                raise DigitalTwinCorruptionError(
                    "digital-twin lifecycle sequence is invalid"
                ) from exc
            prior_lifecycle_sha256 = _digest_value(
                head_row[0], name="prior_lifecycle_sha256"
            )
            lifecycle_sha256 = _digest(
                {
                    "event_id": event_id,
                    "twin_id": twin_id,
                    "event_kind": canonical_kind,
                    "payload_sha256": payload_sha256,
                    "receipt_id": receipt_id,
                    "disposition": disposition.value,
                    "accepted": int(accepted),
                    "graph_version": graph_version,
                    "state_sha256": state_sha256,
                    "created_at_ns": created_at_ns,
                    "lifecycle_sequence": lifecycle_sequence,
                    "prior_lifecycle_sha256": prior_lifecycle_sha256,
                }
            )
            connection.execute(
                "UPDATE twin_meta SET value=? WHERE key='lifecycle_sequence'",
                (str(lifecycle_sequence),),
            )
            connection.execute(
                "UPDATE twin_meta SET value=? WHERE key='lifecycle_head_sha256'",
                (lifecycle_sha256,),
            )
        values: dict[str, Any] = {
            "event_id": event_id,
            "twin_id": twin_id,
            "event_kind": canonical_kind,
            "payload_sha256": payload_sha256,
            "receipt_id": receipt_id,
            "disposition": disposition.value,
            "accepted": int(accepted),
            "graph_version": graph_version,
            "state_sha256": state_sha256,
            "created_at_ns": created_at_ns,
            "lifecycle_sequence": lifecycle_sequence,
            "prior_lifecycle_sha256": prior_lifecycle_sha256,
            "lifecycle_sha256": lifecycle_sha256,
        }
        values["row_sha256"] = self._row_hmac("twin_events", values)
        columns = ",".join(values)
        placeholders = ",".join("?" for _ in values)
        connection.execute(
            f"INSERT INTO twin_events({columns}) VALUES({placeholders})",
            tuple(values.values()),
        )
        return TwinReceipt(
            receipt_id=receipt_id,
            twin_id=twin_id,
            event_id=event_id,
            disposition=disposition,
            accepted=accepted,
            graph_version=graph_version,
            state_sha256=state_sha256,
        )

    def _existing_event(
        self,
        connection: sqlite3.Connection,
        event_id: str,
        payload_sha256: str,
    ) -> TwinReceipt | None:
        row = connection.execute(
            "SELECT * FROM twin_events WHERE event_id=?", (event_id,)
        ).fetchone()
        if row is None:
            row = self._archived_event(connection, event_id)
            if row is None:
                return None
        self._verify_event_row(row)
        if str(row["payload_sha256"]) != payload_sha256:
            raise DigitalTwinConflictError("event identity was replayed with new evidence")
        return TwinReceipt(
            receipt_id=str(row["receipt_id"]),
            twin_id=str(row["twin_id"]),
            event_id=str(row["event_id"]),
            disposition=TwinDisposition.DUPLICATE,
            accepted=bool(row["accepted"]),
            graph_version=int(row["graph_version"]),
            state_sha256=str(row["state_sha256"]),
        )

    def _archived_event(
        self,
        connection: sqlite3.Connection,
        event_id: str,
    ) -> Mapping[str, Any] | None:
        segments = connection.execute(
            "SELECT * FROM twin_event_segments ORDER BY segment_sequence DESC"
        ).fetchall()
        for segment in segments:
            events = self._verify_segment_row(segment)
            for event in events:
                if str(event.get("event_id") or "") == event_id:
                    return event
        return None

    def _state_sha256(self, connection: sqlite3.Connection, twin_id: str) -> str:
        twin = connection.execute("SELECT * FROM twins WHERE twin_id=?", (twin_id,)).fetchone()
        if twin is None:
            raise DigitalTwinCorruptionError("event twin is missing")
        self._verify_twin_row(twin)
        nodes = connection.execute(
            "SELECT * FROM twin_nodes WHERE twin_id=? ORDER BY node_id", (twin_id,)
        ).fetchall()
        relationships = connection.execute(
            "SELECT * FROM twin_relationships WHERE twin_id=? ORDER BY relationship_id",
            (twin_id,),
        ).fetchall()
        properties = connection.execute(
            "SELECT * FROM twin_properties WHERE twin_id=? ORDER BY node_id", (twin_id,)
        ).fetchall()
        for row in nodes:
            self._verify_node_row(row)
        for row in relationships:
            self._verify_relationship_row(row)
        for row in properties:
            self._verify_property_row(row)
        return _digest(
            {
                "twin": str(twin["row_sha256"]),
                "nodes": [str(row["row_sha256"]) for row in nodes],
                "relationships": [str(row["row_sha256"]) for row in relationships],
                "properties": [str(row["row_sha256"]) for row in properties],
            }
        )

    def _prune_events(self, connection: sqlite3.Connection) -> None:
        count = int(connection.execute("SELECT COUNT(*) FROM twin_events").fetchone()[0])
        excess = count - self._max_events
        if excess <= 0:
            return
        cursor = connection.execute(
            """
            DELETE FROM twin_events WHERE event_id IN (
                SELECT event_id FROM twin_events
                WHERE event_kind IN ('candidate_seen', 'channel_observed')
                ORDER BY graph_version, created_at_ns, event_id LIMIT ?
            )
            """,
            (excess,),
        )
        remaining = excess - max(0, cursor.rowcount)
        if remaining > 0:
            self._archive_lifecycle_events(connection, minimum_to_free=remaining)

    def _make_event_space(self, connection: sqlite3.Connection, *, event_kind: str) -> None:
        count = int(connection.execute("SELECT COUNT(*) FROM twin_events").fetchone()[0])
        if count < self._max_events:
            return
        cursor = connection.execute(
            """
            DELETE FROM twin_events WHERE event_id=(
                SELECT event_id FROM twin_events
                WHERE event_kind IN ('candidate_seen', 'channel_observed')
                ORDER BY graph_version, created_at_ns, event_id LIMIT 1
            )
            """
        )
        if cursor.rowcount != 1:
            self._archive_lifecycle_events(connection, minimum_to_free=1)
        remaining = int(connection.execute("SELECT COUNT(*) FROM twin_events").fetchone()[0])
        if remaining >= self._max_events:
            raise DigitalTwinError(
                f"digital-twin event rollover did not create space before {event_kind}"
            )

    def _archive_lifecycle_events(
        self,
        connection: sqlite3.Connection,
        *,
        minimum_to_free: int,
    ) -> None:
        if self._pending_archive is not None:
            raise DigitalTwinCorruptionError(
                "multiple lifecycle archive transactions cannot overlap"
            )
        requested = max(1, int(minimum_to_free))
        target = max(requested, min(64, max(1, self._max_events // 8)))
        rows = connection.execute(
            """
            SELECT * FROM twin_events
            WHERE lifecycle_sequence > 0
            ORDER BY lifecycle_sequence, created_at_ns, event_id
            LIMIT ?
            """,
            (target,),
        ).fetchall()
        if len(rows) < requested:
            raise DigitalTwinError(
                "digital-twin event capacity cannot be rolled over without lifecycle evidence"
            )
        for row in rows:
            self._verify_event_row(row)

        selected = rows
        events_json = ""
        events: list[dict[str, Any]] = []
        while selected:
            events = [
                {name: row[name] for name in _EXPECTED_COLUMNS["twin_events"]}
                for row in selected
            ]
            try:
                events_json = _canonical_json(events, name="lifecycle archive segment")
                break
            except ValueError:
                if len(selected) <= requested:
                    raise DigitalTwinError(
                        "digital-twin lifecycle archive segment exceeds its bounded envelope"
                    ) from None
                selected = selected[: max(requested, len(selected) // 2)]
        if not selected or not events_json:
            raise DigitalTwinError("digital-twin lifecycle archive segment is empty")

        segment_sequence_row = connection.execute(
            "SELECT value FROM twin_meta WHERE key='lifecycle_archive_segment_sequence'"
        ).fetchone()
        prior_segment_row = connection.execute(
            "SELECT value FROM twin_meta WHERE key='lifecycle_archive_head_sha256'"
        ).fetchone()
        last_event_sequence_row = connection.execute(
            "SELECT value FROM twin_meta WHERE key='lifecycle_archive_last_event_sequence'"
        ).fetchone()
        last_event_head_row = connection.execute(
            "SELECT value FROM twin_meta WHERE key='lifecycle_archive_last_event_head_sha256'"
        ).fetchone()
        if any(
            row is None
            for row in (
                segment_sequence_row,
                prior_segment_row,
                last_event_sequence_row,
                last_event_head_row,
            )
        ):
            raise DigitalTwinCorruptionError(
                "digital-twin lifecycle archive metadata is incomplete"
            )
        checkpoint_row = connection.execute(
            "SELECT value FROM twin_meta "
            "WHERE key='lifecycle_archive_checkpoint_hmac_sha256'"
        ).fetchone()
        if checkpoint_row is None:
            raise DigitalTwinCorruptionError(
                "digital-twin lifecycle archive checkpoint is missing"
            )
        prior_checkpoint_hmac = str(checkpoint_row[0])
        if not _MAC.fullmatch(prior_checkpoint_hmac):
            raise DigitalTwinCorruptionError(
                "digital-twin lifecycle archive checkpoint authentication is invalid"
            )
        segment_sequence = int(segment_sequence_row[0]) + 1
        prior_segment_sha256 = _digest_value(
            prior_segment_row[0], name="prior_segment_sha256"
        )
        archived_event_sequence = int(last_event_sequence_row[0])
        archived_event_head = _digest_value(
            last_event_head_row[0], name="archived_event_head_sha256"
        )
        first_sequence = int(selected[0]["lifecycle_sequence"])
        last_sequence = int(selected[-1]["lifecycle_sequence"])
        if (
            first_sequence != archived_event_sequence + 1
            or str(selected[0]["prior_lifecycle_sha256"]) != archived_event_head
        ):
            raise DigitalTwinCorruptionError(
                "digital-twin lifecycle archive does not continue its retained head"
            )
        events_sha256 = _digest(events)
        created_at_ns = self._now_ns()
        segment_body: dict[str, Any] = {
            "segment_sequence": segment_sequence,
            "segment_id": "twin.segment."
            + _short_digest(
                {
                    "segment_sequence": segment_sequence,
                    "first": first_sequence,
                    "last": last_sequence,
                    "events_sha256": events_sha256,
                },
                length=40,
            ),
            "first_lifecycle_sequence": first_sequence,
            "last_lifecycle_sequence": last_sequence,
            "event_count": len(selected),
            "prior_segment_sha256": prior_segment_sha256,
            "first_prior_lifecycle_sha256": archived_event_head,
            "head_lifecycle_sha256": str(selected[-1]["lifecycle_sha256"]),
            "events_sha256": events_sha256,
            "created_at_ns": created_at_ns,
        }
        segment_body["segment_sha256"] = _digest(segment_body)
        archive_file = (
            f"segment-{segment_sequence:012d}-"
            f"{str(segment_body['segment_sha256']).removeprefix('sha256:')[:40]}.json"
        )
        archive_payload = {
            "schema": _ARCHIVE_SEGMENT_SCHEMA,
            "segment": dict(segment_body),
            "events": events,
        }
        archive_file_sha256 = _bytes_digest(
            self._archive_json_bytes(
                archive_payload,
                name="lifecycle archive segment",
            )
        )
        self._write_archive_json(
            self._archive_dir / archive_file,
            archive_payload,
            name="lifecycle archive segment",
        )
        segment_values = {
            **segment_body,
            "archive_file": archive_file,
            "archive_file_sha256": archive_file_sha256,
        }
        segment_values["row_sha256"] = self._row_hmac(
            "twin_event_segments", segment_values
        )
        columns = ",".join(segment_values)
        placeholders = ",".join("?" for _ in segment_values)
        connection.execute(
            f"INSERT INTO twin_event_segments({columns}) VALUES({placeholders})",
            tuple(segment_values.values()),
        )
        event_ids = tuple(str(row["event_id"]) for row in selected)
        delete_placeholders = ",".join("?" for _ in event_ids)
        cursor = connection.execute(
            f"DELETE FROM twin_events WHERE event_id IN ({delete_placeholders})",
            event_ids,
        )
        if cursor.rowcount != len(event_ids):
            raise DigitalTwinCorruptionError(
                "digital-twin lifecycle archive did not retire its complete source segment"
            )
        for key, value in (
            ("lifecycle_archive_segment_sequence", str(segment_sequence)),
            ("lifecycle_archive_head_sha256", segment_body["segment_sha256"]),
            ("lifecycle_archive_last_event_sequence", str(last_sequence)),
            (
                "lifecycle_archive_last_event_head_sha256",
                segment_body["head_lifecycle_sha256"],
            ),
        ):
            connection.execute("UPDATE twin_meta SET value=? WHERE key=?", (value, key))
        retire_files = self._enforce_archive_retention(connection)
        target_body = self._archive_checkpoint_body(connection)
        target_checkpoint = self._archive_checkpoint_envelope(target_body)
        connection.execute(
            "UPDATE twin_meta SET value=? "
            "WHERE key='lifecycle_archive_checkpoint_hmac_sha256'",
            (target_checkpoint["checkpoint_hmac_sha256"],),
        )
        pending_body: dict[str, Any] = {
            "schema": _ARCHIVE_PENDING_SCHEMA,
            "prior_checkpoint_hmac_sha256": prior_checkpoint_hmac,
            "target_checkpoint": target_checkpoint,
            "new_files": [archive_file],
            "retire_files": retire_files,
        }
        pending = {
            **pending_body,
            "pending_hmac_sha256": self._integrity_mac(
                "archive-pending",
                pending_body,
            ),
        }
        self._write_archive_json(
            self._archive_pending_path,
            pending,
            name="lifecycle archive recovery intent",
        )
        self._pending_archive = pending

    def _enforce_archive_retention(self, connection: sqlite3.Connection) -> list[str]:
        retired_files: list[str] = []
        while True:
            rows = connection.execute(
                "SELECT * FROM twin_event_segments ORDER BY segment_sequence"
            ).fetchall()
            total_bytes = 0
            for row in rows:
                self._verify_segment_row(row)
                try:
                    total_bytes += int(
                        (self._archive_dir / str(row["archive_file"])).stat().st_size
                    )
                except OSError as exc:
                    raise DigitalTwinCorruptionError(
                        "retained lifecycle archive segment is unavailable"
                    ) from exc
            if (
                len(rows) <= self._max_archive_segments
                and total_bytes <= self._max_archive_bytes
            ):
                return retired_files
            if len(rows) <= 1:
                raise DigitalTwinError(
                    "one lifecycle archive segment exceeds the retention byte budget"
                )
            oldest = rows[0]
            retired_files.append(str(oldest["archive_file"]))
            retired_segment_count = int(
                connection.execute(
                    "SELECT value FROM twin_meta "
                    "WHERE key='lifecycle_archive_retired_segment_count'"
                ).fetchone()[0]
            )
            retired_event_count = int(
                connection.execute(
                    "SELECT value FROM twin_meta "
                    "WHERE key='lifecycle_archive_retired_event_count'"
                ).fetchone()[0]
            )
            updates = {
                "lifecycle_archive_retained_from_segment_sequence": str(
                    int(oldest["segment_sequence"]) + 1
                ),
                "lifecycle_archive_retained_prior_segment_sha256": str(
                    oldest["segment_sha256"]
                ),
                "lifecycle_archive_retained_first_event_sequence": str(
                    int(oldest["last_lifecycle_sequence"]) + 1
                ),
                "lifecycle_archive_retained_prior_event_sha256": str(
                    oldest["head_lifecycle_sha256"]
                ),
                "lifecycle_archive_retired_segment_count": str(
                    retired_segment_count + 1
                ),
                "lifecycle_archive_retired_event_count": str(
                    retired_event_count + int(oldest["event_count"])
                ),
            }
            for key, value in updates.items():
                connection.execute(
                    "UPDATE twin_meta SET value=? WHERE key=?", (value, key)
                )
            connection.execute(
                "DELETE FROM twin_event_segments WHERE segment_sequence=?",
                (int(oldest["segment_sequence"]),),
            )

    def _verify_integrity(self, connection: sqlite3.Connection, *, full: bool) -> None:
        quick = connection.execute("PRAGMA quick_check").fetchone()
        if quick is None or str(quick[0]).lower() != "ok":
            raise DigitalTwinCorruptionError("SQLite quick_check rejected twin graph")
        foreign = connection.execute("PRAGMA foreign_key_check").fetchall()
        if foreign:
            raise DigitalTwinCorruptionError("digital-twin foreign-key check failed")
        if full:
            for row in connection.execute("SELECT * FROM twins"):
                self._verify_twin_row(row)
            for row in connection.execute("SELECT * FROM twin_nodes"):
                self._verify_node_row(row)
            for row in connection.execute("SELECT * FROM twin_relationships"):
                self._verify_relationship_row(row)
            for row in connection.execute("SELECT * FROM twin_adapter_bindings"):
                self._verify_binding_row(row)
            for row in connection.execute("SELECT * FROM twin_properties"):
                self._verify_property_row(row)
            for row in connection.execute("SELECT * FROM twin_event_segments"):
                self._verify_segment_row(row)
            self._verify_event_ledger(connection)
        self._last_integrity_check_ns = self._now_ns()

    def _verify_row(
        self,
        table: str,
        row: Mapping[str, Any] | sqlite3.Row,
    ) -> None:
        values = {name: row[name] for name in _EXPECTED_COLUMNS[table]}
        supplied = str(values.pop("row_sha256"))
        expected = self._row_hmac(table, values)
        if not _MAC.fullmatch(supplied) or not hmac.compare_digest(supplied, expected):
            raise DigitalTwinCorruptionError(f"{table} row authentication differs")

    def _verify_twin_row(self, row: sqlite3.Row) -> None:
        self._verify_row("twins", row)

    def _verify_node_row(self, row: sqlite3.Row) -> None:
        self._verify_row("twin_nodes", row)
        try:
            model = json.loads(str(row["model_json"]))
        except json.JSONDecodeError as exc:
            raise DigitalTwinCorruptionError("twin node model is invalid JSON") from exc
        if _digest(model) != str(row["model_sha256"]):
            raise DigitalTwinCorruptionError("twin node model digest differs")

    def _verify_relationship_row(self, row: sqlite3.Row) -> None:
        self._verify_row("twin_relationships", row)

    def _verify_binding_row(self, row: sqlite3.Row) -> None:
        self._verify_row("twin_adapter_bindings", row)

    def _verify_property_row(self, row: sqlite3.Row) -> None:
        self._verify_row("twin_properties", row)

    def _verify_event_row(self, row: Mapping[str, Any] | sqlite3.Row) -> None:
        self._verify_row("twin_events", row)
        event_id = _identifier(row["event_id"], name="event_id")
        twin_id = _identifier(row["twin_id"], name="twin_id")
        event_kind = _identifier(row["event_kind"], name="event_kind")
        payload_sha256 = _digest_value(row["payload_sha256"], name="payload_sha256")
        receipt_id = _identifier(row["receipt_id"], name="receipt_id")
        if receipt_id != _receipt_id(event_id, payload_sha256, twin_id):
            raise DigitalTwinCorruptionError("digital-twin event receipt differs")
        _digest_value(row["state_sha256"], name="state_sha256")
        lifecycle_sequence = int(row["lifecycle_sequence"])
        prior = _digest_value(row["prior_lifecycle_sha256"], name="prior_lifecycle_sha256")
        lifecycle = _digest_value(row["lifecycle_sha256"], name="lifecycle_sha256")
        if event_kind in _PRUNABLE_EVENT_KINDS:
            if (
                lifecycle_sequence != 0
                or prior != _NO_LIFECYCLE_SHA256
                or lifecycle != _NO_LIFECYCLE_SHA256
            ):
                raise DigitalTwinCorruptionError(
                    "prunable event improperly participates in the lifecycle chain"
                )
        elif lifecycle_sequence < 1:
            raise DigitalTwinCorruptionError("lifecycle event has no chain sequence")

    def _verify_segment_row(
        self,
        row: Mapping[str, Any] | sqlite3.Row,
    ) -> tuple[dict[str, Any], ...]:
        self._verify_row("twin_event_segments", row)
        archive_file = str(row["archive_file"])
        if not re.fullmatch(r"segment-[0-9]{12}-[0-9a-f]{40}\.json", archive_file):
            raise DigitalTwinCorruptionError(
                "digital-twin lifecycle archive file identity differs"
            )
        archive_path = self._archive_dir / archive_file
        if archive_path.is_symlink() or not archive_path.is_file():
            raise DigitalTwinCorruptionError(
                "digital-twin lifecycle archive segment is missing or unsafe"
            )
        try:
            raw_payload = archive_path.read_bytes()
        except OSError as exc:
            raise DigitalTwinCorruptionError(
                "digital-twin lifecycle archive segment is missing"
            ) from exc
        if len(raw_payload) > (_MAX_JSON_BYTES * 2):
            raise DigitalTwinCorruptionError(
                "digital-twin lifecycle archive segment is unsafe"
            )
        if _bytes_digest(raw_payload) != str(row["archive_file_sha256"]):
            raise DigitalTwinCorruptionError(
                "digital-twin lifecycle archive file digest differs"
            )
        try:
            archive_payload = json.loads(raw_payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DigitalTwinCorruptionError(
                "digital-twin lifecycle segment contains invalid JSON"
            ) from exc
        if (
            not isinstance(archive_payload, dict)
            or archive_payload.get("schema") != _ARCHIVE_SEGMENT_SCHEMA
            or not isinstance(archive_payload.get("segment"), dict)
        ):
            raise DigitalTwinCorruptionError(
                "digital-twin lifecycle archive segment envelope differs"
            )
        events_value = archive_payload.get("events")
        if not isinstance(events_value, list) or not events_value:
            raise DigitalTwinCorruptionError("digital-twin lifecycle segment is empty")
        events: list[dict[str, Any]] = []
        expected_event_fields = set(_EXPECTED_COLUMNS["twin_events"])
        for raw_event in events_value:
            if not isinstance(raw_event, dict) or set(raw_event) != expected_event_fields:
                raise DigitalTwinCorruptionError(
                    "digital-twin lifecycle segment event shape differs"
                )
            event = dict(raw_event)
            self._verify_event_row(event)
            if int(event["lifecycle_sequence"]) <= 0:
                raise DigitalTwinCorruptionError(
                    "digital-twin lifecycle segment contains a prunable event"
                )
            events.append(event)
        if (
            int(row["event_count"]) != len(events)
            or int(row["first_lifecycle_sequence"])
            != int(events[0]["lifecycle_sequence"])
            or int(row["last_lifecycle_sequence"])
            != int(events[-1]["lifecycle_sequence"])
            or str(row["first_prior_lifecycle_sha256"])
            != str(events[0]["prior_lifecycle_sha256"])
            or str(row["head_lifecycle_sha256"])
            != str(events[-1]["lifecycle_sha256"])
            or str(row["events_sha256"]) != _digest(events)
        ):
            raise DigitalTwinCorruptionError(
                "digital-twin lifecycle segment summary differs"
            )
        segment_body = {
            name: row[name]
            for name in _EXPECTED_COLUMNS["twin_event_segments"]
            if name
            not in {
                "archive_file",
                "archive_file_sha256",
                "segment_sha256",
                "row_sha256",
            }
        }
        if str(row["segment_sha256"]) != _digest(segment_body):
            raise DigitalTwinCorruptionError(
                "digital-twin lifecycle segment digest differs"
            )
        if dict(archive_payload["segment"]) != {
            **segment_body,
            "segment_sha256": str(row["segment_sha256"]),
        }:
            raise DigitalTwinCorruptionError(
                "digital-twin lifecycle archive manifest differs from its file"
            )
        expected_segment_id = "twin.segment." + _short_digest(
            {
                "segment_sequence": int(row["segment_sequence"]),
                "first": int(row["first_lifecycle_sequence"]),
                "last": int(row["last_lifecycle_sequence"]),
                "events_sha256": str(row["events_sha256"]),
            },
            length=40,
        )
        if str(row["segment_id"]) != expected_segment_id:
            raise DigitalTwinCorruptionError(
                "digital-twin lifecycle segment identity differs"
            )
        for name in (
            "prior_segment_sha256",
            "first_prior_lifecycle_sha256",
            "head_lifecycle_sha256",
            "events_sha256",
            "archive_file_sha256",
            "segment_sha256",
        ):
            _digest_value(row[name], name=name)
        return tuple(events)

    def _verify_event_ledger(self, connection: sqlite3.Connection) -> None:
        archive_meta = {
            key: connection.execute(
                "SELECT value FROM twin_meta WHERE key=?", (key,)
            ).fetchone()
            for key in self._archive_meta_keys()
        }
        if any(row is None for row in archive_meta.values()):
            raise DigitalTwinCorruptionError(
                "digital-twin lifecycle archive metadata is incomplete"
            )
        archive_values = {
            key: str(row[0]) for key, row in archive_meta.items() if row is not None
        }
        expected_segment_sequence = (
            int(archive_values["lifecycle_archive_retained_from_segment_sequence"])
            - 1
        )
        prior_segment = archive_values[
            "lifecycle_archive_retained_prior_segment_sha256"
        ]
        expected_sequence = (
            int(archive_values["lifecycle_archive_retained_first_event_sequence"])
            - 1
        )
        prior = archive_values["lifecycle_archive_retained_prior_event_sha256"]

        segments = connection.execute(
            "SELECT * FROM twin_event_segments ORDER BY segment_sequence"
        ).fetchall()
        for segment in segments:
            events = self._verify_segment_row(segment)
            expected_segment_sequence += 1
            if int(segment["segment_sequence"]) != expected_segment_sequence:
                raise DigitalTwinCorruptionError(
                    "digital-twin lifecycle archive segment sequence has a gap"
                )
            if str(segment["prior_segment_sha256"]) != prior_segment:
                raise DigitalTwinCorruptionError(
                    "digital-twin lifecycle archive predecessor differs"
                )
            if str(segment["first_prior_lifecycle_sha256"]) != prior:
                raise DigitalTwinCorruptionError(
                    "digital-twin lifecycle archive event predecessor differs"
                )
            for event in events:
                expected_sequence += 1
                prior = self._verify_lifecycle_event_link(
                    event,
                    expected_sequence=expected_sequence,
                    prior=prior,
                )
            if str(segment["head_lifecycle_sha256"]) != prior:
                raise DigitalTwinCorruptionError(
                    "digital-twin lifecycle archive retained head differs"
                )
            prior_segment = str(segment["segment_sha256"])

        if (
            int(archive_values["lifecycle_archive_segment_sequence"])
            != expected_segment_sequence
            or archive_values["lifecycle_archive_head_sha256"] != prior_segment
            or int(archive_values["lifecycle_archive_last_event_sequence"])
            != expected_sequence
            or archive_values["lifecycle_archive_last_event_head_sha256"] != prior
        ):
            raise DigitalTwinCorruptionError(
                "digital-twin lifecycle archive head differs"
            )
        self._ensure_archive_checkpoint(connection)

        rows = connection.execute(
            "SELECT * FROM twin_events ORDER BY lifecycle_sequence, created_at_ns, event_id"
        ).fetchall()
        for row in rows:
            self._verify_event_row(row)
            lifecycle_sequence = int(row["lifecycle_sequence"])
            if lifecycle_sequence == 0:
                continue
            expected_sequence += 1
            prior = self._verify_lifecycle_event_link(
                row,
                expected_sequence=expected_sequence,
                prior=prior,
            )
        sequence_row = connection.execute(
            "SELECT value FROM twin_meta WHERE key='lifecycle_sequence'"
        ).fetchone()
        head_row = connection.execute(
            "SELECT value FROM twin_meta WHERE key='lifecycle_head_sha256'"
        ).fetchone()
        if sequence_row is None or head_row is None:
            raise DigitalTwinCorruptionError("digital-twin lifecycle chain head is missing")
        try:
            stored_sequence = int(sequence_row[0])
        except (TypeError, ValueError) as exc:
            raise DigitalTwinCorruptionError(
                "digital-twin lifecycle sequence is invalid"
            ) from exc
        if stored_sequence != expected_sequence or str(head_row[0]) != prior:
            raise DigitalTwinCorruptionError("digital-twin lifecycle chain head differs")

    @staticmethod
    def _verify_lifecycle_event_link(
        row: Mapping[str, Any] | sqlite3.Row,
        *,
        expected_sequence: int,
        prior: str,
    ) -> str:
        lifecycle_sequence = int(row["lifecycle_sequence"])
        if lifecycle_sequence != expected_sequence:
            raise DigitalTwinCorruptionError("digital-twin lifecycle sequence has a gap")
        if str(row["prior_lifecycle_sha256"]) != prior:
            raise DigitalTwinCorruptionError(
                "digital-twin lifecycle chain predecessor differs"
            )
        expected_lifecycle_sha256 = _digest(
            {
                "event_id": str(row["event_id"]),
                "twin_id": str(row["twin_id"]),
                "event_kind": str(row["event_kind"]),
                "payload_sha256": str(row["payload_sha256"]),
                "receipt_id": str(row["receipt_id"]),
                "disposition": str(row["disposition"]),
                "accepted": int(row["accepted"]),
                "graph_version": int(row["graph_version"]),
                "state_sha256": str(row["state_sha256"]),
                "created_at_ns": int(row["created_at_ns"]),
                "lifecycle_sequence": lifecycle_sequence,
                "prior_lifecycle_sha256": prior,
            }
        )
        if str(row["lifecycle_sha256"]) != expected_lifecycle_sha256:
            raise DigitalTwinCorruptionError("digital-twin lifecycle chain digest differs")
        return expected_lifecycle_sha256

    @staticmethod
    def _public_twin(row: sqlite3.Row, *, private: bool) -> dict[str, Any]:
        if bool(row["privacy_sensitive"]) and not private:
            raise DigitalTwinCorruptionError(
                "privacy-sensitive twin reached the public projection"
            )
        result = {
            "twin_id": str(row["twin_id"]),
            "identity_scope": str(row["identity_scope"]),
            "persistent_identity": bool(row["persistent_identity"]),
            "privacy_sensitive": bool(row["privacy_sensitive"]),
            "lifecycle": str(row["lifecycle"]),
            "health": str(row["health"]),
            "generation": int(row["generation"]),
            "discovered_at_ns": int(row["discovered_at_ns"]),
            "last_seen_at_ns": int(row["last_seen_at_ns"]),
        }
        if private or not bool(row["privacy_sensitive"]):
            result.update(
                {
                    "connector_id": str(row["connector_id"]),
                    "transport": str(row["transport"]),
                    "manifest_sha256": str(row["manifest_sha256"]),
                }
            )
        return result

    @staticmethod
    def _public_node(row: sqlite3.Row, *, private: bool) -> dict[str, Any]:
        result = {
            "node_id": str(row["node_id"]),
            "twin_id": str(row["twin_id"]),
            "node_kind": str(row["node_kind"]),
            "component_kind": str(row["component_kind"]),
            "generation": int(row["generation"]),
        }
        if private:
            result.update(
                {
                    "model_sha256": str(row["model_sha256"]),
                    "model": json.loads(str(row["model_json"])),
                }
            )
        return result

    @staticmethod
    def _public_relationship(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "relationship_id": str(row["relationship_id"]),
            "twin_id": str(row["twin_id"]),
            "source_node_id": str(row["source_node_id"]),
            "target_node_id": str(row["target_node_id"]),
            "relationship_kind": str(row["relationship_kind"]),
            "generation": int(row["generation"]),
        }

    @staticmethod
    def _public_property(row: sqlite3.Row, *, include_value: bool) -> dict[str, Any]:
        result = {
            "node_id": str(row["node_id"]),
            "twin_id": str(row["twin_id"]),
            "observation_id": str(row["observation_id"]),
            "historian_record_id": str(row["historian_record_id"]),
            "unit": str(row["unit"]),
            "status": str(row["status"]),
            "quality": str(row["quality"]),
            "order_basis": str(row["order_basis"]),
            "order_gap": bool(row["order_gap"]),
            "alarm_codes": json.loads(str(row["alarm_codes_json"])),
            "captured_at_ns": int(row["captured_at_ns"]),
            "version": int(row["version"]),
        }
        if include_value:
            result.update(
                {
                    "reading_sha256": str(row["reading_sha256"]),
                    "declaration_sha256": str(row["declaration_sha256"]),
                    "value": json.loads(str(row["value_json"])),
                }
            )
        return result


__all__ = [
    "DigitalTwinConflictError",
    "DigitalTwinCorruptionError",
    "DigitalTwinError",
    "RealityDigitalTwinGraph",
    "TwinDisposition",
    "TwinHealth",
    "TwinLifecycle",
    "TwinNodeKind",
    "TwinReceipt",
    "TwinRelationshipKind",
]
