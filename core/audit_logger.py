from core.runtime.errors import record_degradation
import os
import sqlite3
import hmac
import hashlib
import json
import time
import logging
from typing import Any, Dict, Optional
from core.config import config

logger = logging.getLogger("Aura.Audit")

class AuditLogger:
    """
    Immutable structured event store for all critical operations (self-modification, admin).
    Uses SQLite WAL mode with HMAC-SHA256 signatures per entry to detect tampering.
    """
    def __init__(self, db_path: Optional[str] = None, hmac_secret: Optional[str] = None):
        self.db_path = db_path or os.environ.get("AURA_AUDIT_DB", str(config.paths.data_dir / "audit.db"))
        raw_secret = os.environ.get("AURA_AUDIT_HMAC_SECRET")
        if not raw_secret:
            logger.error("❌ CRITICAL: AURA_AUDIT_HMAC_SECRET environment variable is not set.")
            raise RuntimeError("CRITICAL: AURA_AUDIT_HMAC_SECRET environment variable is not set. Halting boot.")
        self.hmac_secret = raw_secret.encode('utf-8')
        import threading
        self._lock = threading.Lock()
        
        # Ensure directory exists before connecting
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        # Use a long-lived connection to prevent file descriptor churn and latency spikes
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._init_db()

    def _init_db(self):
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
            record_degradation('audit_logger', e)
            logger.error(f"Failed to initialize Audit DB: {e}", exc_info=True)

    def _sign(self, timestamp: float, actor: str, action: str, target: str, context_str: str) -> str:
        payload = f"{timestamp}|{actor}|{action}|{target}|{context_str}".encode('utf-8')
        return hmac.new(key=self.hmac_secret, msg=payload, digestmod=hashlib.sha256).hexdigest()

    def _redact(self, context: Any) -> Any:
        """Redacts sensitive keys from context recursively."""
        if isinstance(context, dict):
            return {k: ("[REDACTED]" if any(s in k.lower() for s in ["api_key", "secret", "password", "token", "credential"]) else self._redact(v)) for k, v in context.items()}
        elif isinstance(context, list):
            return [self._redact(item) for item in context]
        return context

    def log(self, action: str, actor: str = "system", target: str = "", context: Optional[Dict[str, Any]] = None):
        """Record an immutable event."""
        timestamp = time.time()
        redacted_context = self._redact(context or {})
        context_str = json.dumps(redacted_context, sort_keys=True)
        signature = self._sign(timestamp, actor, action, target, context_str)
        
        try:
            with self._lock:
                self._conn.execute("""
                    INSERT INTO audit_events (timestamp, actor, action, target, context, signature)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (timestamp, actor, action, target, context_str, signature))
                self._conn.commit()
                logger.debug("Audit event recorded: [%s] by [%s] on [%s]", action, actor, target)
        except (sqlite3.Error, OSError) as e:
            record_degradation('audit_logger', e)
            logger.error(f"CRITICAL: Failed to write to audit log: {e}", exc_info=True)

    def verify_integrity(self) -> bool:
        """Verifies that no rows in the audit database have been tampered with."""
        try:
            with self._lock:
                cursor = self._conn.execute("SELECT id, timestamp, actor, action, target, context, signature FROM audit_events ORDER BY id")
                for row in cursor.fetchall():
                    evt_id, ts, actor, action, target, ctx_str, stored_sig = row
                    expected_sig = self._sign(ts, actor, action, target, ctx_str)
                    if not hmac.compare_digest(stored_sig, expected_sig):
                        logger.critical("AUDIT INTEGRITY VIOLATION DETECTED: Row %s tampered.", evt_id)
                        return False
            logger.info("Audit log integrity verified: SUCCESS")
            return True
        except (sqlite3.Error, OSError) as e:
            record_degradation('audit_logger', e)
            logger.error("Failed to verify audit log integrity: %s", e)
            return False

# Global instance
_audit_logger = None

def get_audit_logger() -> AuditLogger:
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger()
    return _audit_logger
