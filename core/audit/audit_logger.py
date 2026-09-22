import hashlib
import hmac
import json
import logging
import os
import re
import sqlite3
import threading
import time
from typing import Any

from core.config import config
from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.Audit")

_SECRET_KEY_MARKERS = ("api_key", "secret", "password", "passwd", "token", "credential", "auth")
# A URL carrying inline credentials, e.g. https://user:pass@host/...
_URL_CRED_RE = re.compile(r"([a-z]+://)[^/\s:@]+:[^/\s@]+@", re.IGNORECASE)
# key=VALUE / key: VALUE where the key names a secret.
_SECRET_ASSIGN_RE = re.compile(
    r"\b(" + "|".join(_SECRET_KEY_MARKERS) + r")\b\s*[:=]\s*\S+", re.IGNORECASE
)


class AuditLogger:
    """
    Immutable structured event store for all critical operations (self-modification, admin).
    Uses SQLite WAL mode with HMAC-SHA256 signatures per entry to detect tampering.
    """
    def __init__(self, db_path: str | None = None, hmac_secret: str | None = None):
        self.db_path = db_path or os.environ.get("AURA_AUDIT_DB", str(config.paths.data_dir / "audit.db"))
        # Honor an explicitly injected secret (e.g. from a trust root) rather
        # than always reading the environment (05fd4428).
        raw_secret = hmac_secret if hmac_secret else os.environ.get("AURA_AUDIT_HMAC_SECRET")
        if not raw_secret:
            logger.error("❌ CRITICAL: no audit HMAC secret provided (arg or AURA_AUDIT_HMAC_SECRET).")
            raise RuntimeError("CRITICAL: audit HMAC secret is not set. Halting boot.")
        self.hmac_secret = raw_secret.encode("utf-8")
        self._lock = threading.Lock()
        self._closed = False

        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        # Long-lived connection to avoid file-descriptor churn and latency spikes.
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._init_db()

    def _init_db(self):
        # Fail CLOSED: a logger whose schema/WAL setup failed is unusable, so
        # construction must not succeed with a broken connection (e001f7fa).
        try:
            with self._lock:
                self._conn.execute("PRAGMA journal_mode=WAL")
                self._conn.execute("""
                    CREATE TABLE IF NOT EXISTS audit_events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp REAL NOT NULL,
                        actor TEXT NOT NULL,
                        action TEXT NOT NULL,
                        target TEXT,
                        context JSON,
                        signature TEXT NOT NULL
                    )
                """)
                self._conn.commit()
        except (sqlite3.Error, OSError) as e:
            record_degradation("audit_logger", e, severity="critical",
                               action="failed audit-logger construction because the DB could not initialize")
            logger.error("Failed to initialize Audit DB: %s", e, exc_info=True)
            try:
                self._conn.close()
            except (sqlite3.Error, OSError) as exc:
                logger.debug(
                    "the audit connection did not close after a failed init (%s: %s)",
                    type(exc).__name__,
                    exc,
                )
            raise RuntimeError(f"audit_db_init_failed: {e}") from e

    def _sign(self, timestamp: float, actor: str, action: str, target: str, context_str: str) -> str:
        payload = f"{timestamp}|{actor}|{action}|{target}|{context_str}".encode()
        return hmac.new(key=self.hmac_secret, msg=payload, digestmod=hashlib.sha256).hexdigest()

    @staticmethod
    def _redact_str(value: str) -> str:
        redacted = _URL_CRED_RE.sub(r"\1***:***@", value)
        redacted = _SECRET_ASSIGN_RE.sub(lambda m: f"{m.group(1)}=[REDACTED]", redacted)
        return redacted

    def _redact(self, context: Any) -> Any:
        """Redact secret-bearing keys AND values from context, recursively."""
        if isinstance(context, dict):
            return {
                k: ("[REDACTED]" if isinstance(k, str) and any(s in k.lower() for s in _SECRET_KEY_MARKERS)
                    else self._redact(v))
                for k, v in context.items()
            }
        if isinstance(context, (list, tuple)):
            return [self._redact(item) for item in context]
        if isinstance(context, str):
            return self._redact_str(context)
        return context

    def log(self, action: str, actor: str = "system", target: str = "", context: dict[str, Any] | None = None) -> bool:
        """Record an immutable event. Returns True only when the row is committed."""
        if self._closed:
            return False
        timestamp = time.time()
        redacted_context = self._redact(context or {})
        context_str = json.dumps(redacted_context, sort_keys=True, default=str)
        signature = self._sign(timestamp, actor, action, target, context_str)

        try:
            with self._lock:
                self._conn.execute("""
                    INSERT INTO audit_events (timestamp, actor, action, target, context, signature)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (timestamp, actor, action, target, context_str, signature))
                self._conn.commit()
            logger.debug("Audit event recorded: [%s] by [%s] on [%s]", action, actor, target)
            return True
        except (sqlite3.Error, OSError) as e:
            # A swallowed write is a silent hole in the audit trail — surface it.
            record_degradation("audit_logger", e, severity="critical",
                               action="failed to durably write an audit event")
            logger.error("CRITICAL: Failed to write to audit log: %s", e, exc_info=True)
            return False

    def verify_integrity(self) -> bool:
        """Verifies that no rows in the audit database have been tampered with."""
        try:
            with self._lock:
                cursor = self._conn.execute(
                    "SELECT id, timestamp, actor, action, target, context, signature FROM audit_events ORDER BY id"
                )
                for row in cursor.fetchall():
                    evt_id, ts, actor, action, target, ctx_str, stored_sig = row
                    expected_sig = self._sign(ts, actor, action, target, ctx_str)
                    if not hmac.compare_digest(stored_sig, expected_sig):
                        logger.critical("AUDIT INTEGRITY VIOLATION DETECTED: Row %s tampered.", evt_id)
                        return False
            logger.info("Audit log integrity verified: SUCCESS")
            return True
        except (sqlite3.Error, OSError) as e:
            record_degradation("audit_logger", e)
            logger.error("Failed to verify audit log integrity: %s", e)
            return False

    def close(self) -> None:
        """Checkpoint the WAL and close the long-lived connection on shutdown."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            try:
                self._conn.execute("PRAGMA wal_checkpoint(FULL)")
                self._conn.commit()
            except (sqlite3.Error, OSError) as e:
                record_degradation("audit_logger", e,
                                   action="audit WAL checkpoint failed during shutdown")
            finally:
                try:
                    self._conn.close()
                except (sqlite3.Error, OSError) as exc:
                    logger.debug(
                        "the audit connection did not close at shutdown (%s: %s)",
                        type(exc).__name__,
                        exc,
                    )


# Global instance
_audit_logger = None
_audit_logger_lock = threading.Lock()


def get_audit_logger() -> AuditLogger:
    global _audit_logger
    if _audit_logger is None:
        with _audit_logger_lock:
            if _audit_logger is None:
                _audit_logger = AuditLogger()
    return _audit_logger
