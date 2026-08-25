"""core/security/governance_vault.py -- Cryptographic Governance Vault
=====================================================================
Thread-safe, tamper-evident storage for constitutional artifacts.

Uses SQLite in WAL mode for ACID transactions with SHA-256 content hashing.
Any modification detected on load triggers an ExternalTamperingScar via the
scar formation system.

Protected artifacts:
  - canonical_self.json   (the authoritative self-model)
  - constitutional_core.json (constitutional constraints)
  - Any artifact registered via seal()

Design principles:
  1. FAIL-CLOSED: if verification fails, raise SecurityException
  2. THREAD-SAFE: all operations are serialized through SQLite WAL
  3. TAMPER-EVIDENT: every read verifies content hash
  4. ATOMIC: writes are transactional (commit or rollback, never partial)
"""
from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

from core.runtime.lockdep import LockRank, checked_lock
from core.runtime.state_ownership import state_root

logger = logging.getLogger("Aura.GovernanceVault")


class SecurityException(RuntimeError):  # noqa: N818 - public compatibility name
    """Raised when a governance integrity violation is detected."""


class TamperDetected(SecurityException):
    """Raised when artifact content hash does not match sealed hash."""


_DB_DIR = state_root() / "data" / "governance"
_DB_PATH = _DB_DIR / "vault.db"

_SCHEMA_VERSION = 1
_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS sealed_artifacts (
    artifact_id   TEXT PRIMARY KEY,
    content_hash  TEXT NOT NULL,
    content       TEXT NOT NULL,
    sealed_at     REAL NOT NULL,
    updated_at    REAL NOT NULL,
    version       INTEGER NOT NULL DEFAULT 1,
    seal_chain    TEXT NOT NULL DEFAULT '[]'
);

CREATE TABLE IF NOT EXISTS vault_metadata (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


def _sha256(data: str) -> str:
    """Compute SHA-256 hex digest of a string."""
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _canonical_json(obj: Any) -> str:
    """Deterministic JSON encoding for hashing."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, default=str)


class GovernanceVault:
    """Cryptographic vault for constitutional artifacts.

    Usage:
        vault = get_governance_vault()
        vault.seal("canonical_self", content_dict)
        content = vault.unseal("canonical_self")  # verifies integrity
    """

    def __init__(self, db_path: Path | None = None) -> None:
        self._db_path = Path(db_path or _DB_PATH)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn: sqlite3.Connection | None = None
        self._init_db()
        logger.info("GovernanceVault initialized at %s", self._db_path)

    def _init_db(self) -> None:
        """Initialize the SQLite database with WAL mode."""
        with self._lock:
            self._conn = sqlite3.connect(
                str(self._db_path),
                check_same_thread=False,
                timeout=10.0,
            )
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute("PRAGMA busy_timeout=5000")
            self._conn.executescript(_SCHEMA_SQL)

            # Set schema version
            self._conn.execute(
                "INSERT OR IGNORE INTO vault_metadata (key, value) VALUES (?, ?)",
                ("schema_version", str(_SCHEMA_VERSION)),
            )
            self._conn.commit()

    def seal(self, artifact_id: str, content: Any) -> str:
        """Seal an artifact into the vault with cryptographic hash.

        Args:
            artifact_id: Unique identifier for the artifact
            content: Dict/list/str to seal (will be canonical-JSON encoded)

        Returns:
            The SHA-256 content hash

        Raises:
            SecurityException: if the write fails
        """
        if not artifact_id:
            raise SecurityException("artifact_id is required")

        content_str = _canonical_json(content)
        content_hash = _sha256(content_str)
        now = time.time()

        with self._lock:
            try:
                cursor = self._conn.execute(
                    "SELECT version, seal_chain FROM sealed_artifacts WHERE artifact_id = ?",
                    (artifact_id,),
                )
                row = cursor.fetchone()

                if row is not None:
                    version = row[0] + 1
                    chain = json.loads(row[1])
                    chain.append({
                        "version": version,
                        "hash": content_hash,
                        "timestamp": now,
                    })
                    # Keep last 50 chain entries
                    chain = chain[-50:]

                    self._conn.execute(
                        """UPDATE sealed_artifacts
                           SET content_hash = ?, content = ?, updated_at = ?,
                               version = ?, seal_chain = ?
                           WHERE artifact_id = ?""",
                        (content_hash, content_str, now, version,
                         json.dumps(chain), artifact_id),
                    )
                else:
                    chain = [{"version": 1, "hash": content_hash, "timestamp": now}]
                    self._conn.execute(
                        """INSERT INTO sealed_artifacts
                           (artifact_id, content_hash, content, sealed_at,
                            updated_at, version, seal_chain)
                           VALUES (?, ?, ?, ?, ?, ?, ?)""",
                        (artifact_id, content_hash, content_str, now, now, 1,
                         json.dumps(chain)),
                    )

                self._conn.commit()
                logger.debug(
                    "Sealed artifact '%s' (hash=%s...)",
                    artifact_id, content_hash[:12],
                )
                return content_hash

            except (sqlite3.Error, OSError) as exc:
                self._conn.rollback()
                raise SecurityException(
                    f"Failed to seal artifact '{artifact_id}': {exc}"
                ) from exc

    def unseal(self, artifact_id: str, *, verify: bool = True) -> Any:
        """Retrieve and verify a sealed artifact.

        Args:
            artifact_id: The artifact to retrieve
            verify: If True (default), verify content hash matches

        Returns:
            The deserialized content

        Raises:
            TamperDetected: if content hash does not match
            KeyError: if artifact_id not found
        """
        with self._lock:
            cursor = self._conn.execute(
                "SELECT content, content_hash FROM sealed_artifacts WHERE artifact_id = ?",
                (artifact_id,),
            )
            row = cursor.fetchone()

        if row is None:
            raise KeyError(f"Artifact '{artifact_id}' not found in vault")

        content_str, stored_hash = row

        if verify:
            computed_hash = _sha256(content_str)
            if computed_hash != stored_hash:
                self._trigger_tamper_scar(artifact_id, stored_hash, computed_hash)
                raise TamperDetected(
                    f"Artifact '{artifact_id}' content hash mismatch: "
                    f"stored={stored_hash[:12]}... computed={computed_hash[:12]}..."
                )

        return json.loads(content_str)

    def verify_integrity(self, artifact_id: str) -> tuple[bool, str]:
        """Verify artifact integrity without retrieving content.

        Returns:
            (is_valid, message)
        """
        with self._lock:
            cursor = self._conn.execute(
                "SELECT content, content_hash, version FROM sealed_artifacts WHERE artifact_id = ?",
                (artifact_id,),
            )
            row = cursor.fetchone()

        if row is None:
            return False, f"Artifact '{artifact_id}' not found"

        content_str, stored_hash, version = row
        computed_hash = _sha256(content_str)

        if computed_hash != stored_hash:
            return False, (
                f"TAMPER DETECTED: artifact '{artifact_id}' v{version} "
                f"hash mismatch (stored={stored_hash[:12]}... "
                f"computed={computed_hash[:12]}...)"
            )

        return True, f"Artifact '{artifact_id}' v{version} integrity verified"

    def verify_all(self) -> tuple[bool, list[dict[str, Any]]]:
        """Verify integrity of all sealed artifacts.

        Returns:
            (all_valid, list of results)
        """
        with self._lock:
            cursor = self._conn.execute(
                "SELECT artifact_id FROM sealed_artifacts"
            )
            artifact_ids = [row[0] for row in cursor.fetchall()]

        results = []
        all_valid = True
        for artifact_id in artifact_ids:
            valid, message = self.verify_integrity(artifact_id)
            results.append({
                "artifact_id": artifact_id,
                "valid": valid,
                "message": message,
            })
            if not valid:
                all_valid = False

        return all_valid, results

    def get_seal_chain(self, artifact_id: str) -> list[dict[str, Any]]:
        """Get the seal history chain for an artifact."""
        with self._lock:
            cursor = self._conn.execute(
                "SELECT seal_chain FROM sealed_artifacts WHERE artifact_id = ?",
                (artifact_id,),
            )
            row = cursor.fetchone()

        if row is None:
            return []
        return json.loads(row[0])

    def list_artifacts(self) -> list[dict[str, Any]]:
        """List all sealed artifacts with metadata."""
        with self._lock:
            cursor = self._conn.execute(
                "SELECT artifact_id, content_hash, sealed_at, updated_at, version "
                "FROM sealed_artifacts ORDER BY updated_at DESC"
            )
            rows = cursor.fetchall()

        return [
            {
                "artifact_id": row[0],
                "content_hash": row[1],
                "sealed_at": row[2],
                "updated_at": row[3],
                "version": row[4],
            }
            for row in rows
        ]

    def has_artifact(self, artifact_id: str) -> bool:
        """Check if an artifact exists in the vault."""
        with self._lock:
            cursor = self._conn.execute(
                "SELECT 1 FROM sealed_artifacts WHERE artifact_id = ?",
                (artifact_id,),
            )
            return cursor.fetchone() is not None

    def _trigger_tamper_scar(
        self, artifact_id: str, expected_hash: str, actual_hash: str
    ) -> None:
        """Form a behavioral scar when tampering is detected."""
        try:
            from core.memory.scar_formation import ScarDomain, get_scar_formation
            scar_system = get_scar_formation()
            scar_system.form_scar(
                domain=ScarDomain.CONSTITUTION_MODIFIED_EXTERNALLY,
                description=(
                    f"External tampering detected on governance artifact '{artifact_id}'. "
                    f"Expected hash {expected_hash[:16]}..., got {actual_hash[:16]}... "
                    f"This indicates unauthorized modification of a constitutional file."
                ),
                avoidance_tag=f"tampered_artifact_{artifact_id}",
                severity=0.9,
                heal_rate=0.001,  # Heals very slowly
                context={
                    "artifact_id": artifact_id,
                    "expected_hash": expected_hash,
                    "actual_hash": actual_hash,
                    "detected_at": time.time(),
                },
                verified_threat=True,
                confidence=0.95,
            )
            logger.critical(
                "TAMPERING SCAR FORMED for artifact '%s'", artifact_id
            )
        except (ImportError, AttributeError, RuntimeError) as exc:
            logger.error(
                "Failed to form tampering scar for '%s': %s",
                artifact_id, exc,
            )

    def close(self) -> None:
        """Close the vault database connection."""
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    def is_open(self) -> bool:
        """Return whether this instance still owns a usable SQLite handle."""
        with self._lock:
            if self._conn is None:
                return False
            try:
                self._conn.execute("SELECT 1")
            except sqlite3.ProgrammingError:
                return False
            return True

    def get_status(self) -> dict[str, Any]:
        """Return vault status for health endpoints."""
        all_valid, results = self.verify_all()
        return {
            "db_path": str(self._db_path),
            "artifact_count": len(results),
            "all_valid": all_valid,
            "artifacts": results,
        }


# ── Singleton ─────────────────────────────────────────────────────────────

_instance: GovernanceVault | None = None
_shutdown_registered = False
_instance_lock = checked_lock(
    "governance_vault.singleton", rank=LockRank.REGISTRY, reentrant=True
)


def _register_vault_lifecycle() -> None:
    global _shutdown_registered
    if _shutdown_registered:
        return
    try:
        from core.runtime.shutdown_coordinator import get_shutdown_coordinator

        get_shutdown_coordinator().register(
            close_governance_vault,
            phase="state_vault",
            name="governance_vault",
            timeout=5.0,
        )
        _shutdown_registered = True
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
        logger.debug("GovernanceVault shutdown registration deferred: %s", exc)


def close_governance_vault() -> dict[str, object]:
    """Close the singleton without constructing it during shutdown."""
    global _instance
    with _instance_lock:
        instance = _instance
        _instance = None
        if instance is None:
            return {"clean": True, "closed": False, "reason": "not_initialized"}
        try:
            instance.close()
        except (sqlite3.Error, OSError) as exc:
            return {
                "clean": False,
                "closed": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        return {"clean": True, "closed": True}


def get_governance_vault() -> GovernanceVault:
    """Get or create the singleton GovernanceVault."""
    global _instance
    with _instance_lock:
        # A test, fork boundary, or legacy bound shutdown callback can leave a
        # singleton object whose SQLite handle has already been closed. Never
        # hand that corpse to identity verification: replace it atomically.
        if _instance is None or not _instance.is_open():
            _instance = GovernanceVault()
        _register_vault_lifecycle()
        return _instance
