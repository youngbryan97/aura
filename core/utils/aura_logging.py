from __future__ import annotations

import json
import logging
import os
import queue
import sqlite3
import threading
from datetime import datetime
from pathlib import Path

from core.runtime.errors import record_degradation
from core.runtime.flags import FlagKind, aura_root_override, declare
from core.runtime.network_gateway import get_network_gateway
from core.runtime.state_ownership import state_root

_SQLITE_LOGGING_FLAG = declare(
    "AURA_LOG_SQLITE_ENABLED",
    kind=FlagKind.BOOL,
    default=True,
    description="Persist enhanced core logs through the asynchronous SQLite sink",
    owner="core.utils.aura_logging",
)
_ALERTS_WEBHOOK_FLAG = declare(
    "AURA_ALERTS_WEBHOOK",
    kind=FlagKind.STRING,
    default="",
    description="Optional webhook URL for critical Aura log alerts",
    owner="core.utils.aura_logging",
)
WEBHOOK_URL = str(_ALERTS_WEBHOOK_FLAG.value() or "").strip()


def _default_db_file() -> Path:
    override = aura_root_override()
    root = Path(override).expanduser() if override else state_root()
    return root.resolve() / "data" / "aura_memory.db"


def _proof_logging_active() -> bool:
    return any(os.environ.get(name) for name in ("AURA_PROOF_RUN", "AURA_AGI_MAX_TASKS", "AURA_TESTING"))

class SQLiteMemoryHandler(logging.Handler):
    """Persist INFO+ logs to SQLite without ever blocking the emitting thread.

    emit() is a bounded put_nowait; a single daemon writer thread owns one
    WAL-mode connection and batch-inserts. The previous implementation opened
    a fresh connection and committed (fsync) PER RECORD on the caller's
    thread — on the event loop under disk contention that is a multi-second
    stall per log line.
    """

    _QUEUE_CAPACITY = 5_000
    _BATCH_MAX = 200

    def __init__(self, db_path: str | Path | None = None):
        super().__init__()
        self.db_path = (
            Path(db_path).expanduser() if db_path is not None else _default_db_file()
        )
        self.dropped = 0
        self._queue: queue.Queue = queue.Queue(maxsize=self._QUEUE_CAPACITY)
        self._stop = threading.Event()
        self._writer = threading.Thread(
            target=self._drain_until_stopped, name="aura-log-sqlite-writer", daemon=True
        )
        self._writer.start()

    def emit(self, record):
        try:
            row = (
                datetime.now().isoformat(),
                record.levelname,
                record.module,
                record.getMessage(),
            )
        except (TypeError, ValueError):
            return
        try:
            self._queue.put_nowait(row)
        except queue.Full:
            self.dropped += 1  # never block or recurse from a log call

    def _open_writer_connection(self) -> sqlite3.Connection:
        from core.governance_context import local_internal_governed_scope
        from core.runtime.file_write_gateway import get_file_write_gateway

        with local_internal_governed_scope("aura_logging", domain="state_mutation"):
            gateway = get_file_write_gateway()
            with gateway.open_owned_binary(
                self.db_path,
                mode="a+b",
                permissions=0o600,
                source="aura_logging.sqlite_database",
            ):
                pass
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS system_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                level TEXT,
                module TEXT,
                message TEXT
            )
            """
        )
        conn.commit()
        return conn

    def _drain_until_stopped(self) -> None:
        conn: sqlite3.Connection | None = None
        while not self._stop.is_set():
            try:
                batch = [self._queue.get(timeout=1.0)]
            except queue.Empty:
                continue
            try:
                while len(batch) < self._BATCH_MAX:
                    batch.append(self._queue.get_nowait())
            except queue.Empty:
                pass
            try:
                if conn is None:
                    conn = self._open_writer_connection()
                conn.executemany(
                    "INSERT INTO system_logs (timestamp, level, module, message) VALUES (?, ?, ?, ?)",
                    batch,
                )
                conn.commit()
            except (ImportError, RuntimeError, sqlite3.Error, OSError) as e:
                if conn is not None:
                    try:
                        conn.close()
                    except sqlite3.Error:
                        pass
                    conn = None
                record_degradation(
                    'aura_logging', e,
                    action=f"dropped {len(batch)} buffered log rows after sqlite write failure",
                )
        if conn is not None:
            try:
                conn.close()
            except sqlite3.Error:
                pass

    def close(self) -> None:
        """Stop the writer thread and flush; safe to call more than once."""
        self._stop.set()
        writer = getattr(self, "_writer", None)
        if writer is not None and writer.is_alive():
            writer.join(timeout=3.0)
        super().close()

class WebhookAlertHandler(logging.Handler):
    """
    Sends logs of ERROR level and above to a Discord/Slack webhook.
    """
    def __init__(self, webhook_url: str | None = WEBHOOK_URL):
        super().__init__()
        self.webhook_url = webhook_url

    def emit(self, record):
        if not self.webhook_url:
            return

        try:
            log_entry = self.format(record)
            payload = {
                "content": f"🚨 **AURA CRITICAL ALERT** 🚨\n```text\n{log_entry}\n```"
            }
            # Short timeout to avoid hanging the main loop
            get_network_gateway().request(
                "POST",
                self.webhook_url,
                data=json.dumps(payload),
                headers={"Content-Type": "application/json"},
                timeout=2.0,
                operational_telemetry=True,
                source="observability:aura_logging.webhook",
            )
        except (OSError, ConnectionError, TimeoutError) as e:
            record_degradation('aura_logging', e)
            import sys
            print(f"FAILED TO SEND WEBHOOK ALERT: {e}", file=sys.stderr)

def setup_enhanced_logging(logger_name: str = "Aura"):
    """
    Configures the given logger with standard console output, 
    SQLite persistent memory (INFO+), and Webhook alerts (ERROR+).
    """
    logger = logging.getLogger(logger_name)
    proof_logging = _proof_logging_active()
    logger.setLevel(logging.INFO if proof_logging else logging.DEBUG)
    
    # Avoid duplicate handlers if setup is called multiple times
    if logger.handlers:
        return logger

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # 1. Console Handler (DEBUG+ in dev, INFO+ in proof/eval runs).
    # If root logging is already configured, let records propagate there.
    # Adding a second child console handler produces duplicate live logs and
    # can saturate the UI log ring during long proof/soak runs.
    if not logging.getLogger().handlers:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO if proof_logging else logging.DEBUG)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # 2. SQLite Handler (INFO+). Initialization and all I/O stay on its writer
    # thread; importing logging on an event loop must never open or fsync a DB.
    if bool(_SQLITE_LOGGING_FLAG.value()):
        db_handler = SQLiteMemoryHandler()
        db_handler.setLevel(logging.INFO)
        logger.addHandler(db_handler)

    # 3. Webhook Handler (ERROR+)
    if WEBHOOK_URL:
        alert_handler = WebhookAlertHandler(WEBHOOK_URL)
        alert_handler.setLevel(logging.ERROR)
        alert_handler.setFormatter(formatter)
        logger.addHandler(alert_handler)
        logger.info("📡 Webhook Alerting system active.")
    else:
        logger.info("Webhook alerting disabled (AURA_ALERTS_WEBHOOK not configured).")

    return logger

# Singleton setup for the core logger
core_logger = setup_enhanced_logging("Aura.Core")
