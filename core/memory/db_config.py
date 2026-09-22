"""Database Configuration & Optimization
-------------------------------------
Centralizes SQLite configuration to ensure WAL (Write-Ahead Logging) mode is enabled.
WAL mode significantly improves concurrency, allowing readers to not block writers.
"""

import logging
import sqlite3
import threading
from pathlib import Path
from typing import Any

from core.db.pool import pool as aio_pool
from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.DBConfig")

class _ConnectionLocal(threading.local):
    """Typed per-thread cache backed by the process-wide ownership registry."""

    def __init__(self) -> None:
        self.connections: dict[str, sqlite3.Connection] = {}


_thread_local = _ConnectionLocal()
_registry_lock = threading.RLock()
_connection_registry: dict[tuple[int, str], sqlite3.Connection] = {}
_registered_shutdown_coordinator: Any | None = None


def _connection_is_open(connection: sqlite3.Connection) -> bool:
    try:
        connection.execute("SELECT 1")
    # not a failure: a closed connection is what this asks about, and False is the
    # answer.
    except sqlite3.ProgrammingError:
        return False
    return True


def _commit_and_close(
    connections: list[sqlite3.Connection],
) -> dict[str, Any]:
    failures: list[str] = []
    closed = 0
    for connection in connections:
        try:
            connection.commit()
        except sqlite3.Error as exc:
            failures.append(f"commit:{type(exc).__name__}: {exc}")
        try:
            connection.close()
            closed += 1
        except sqlite3.Error as exc:
            failures.append(f"close:{type(exc).__name__}: {exc}")
    return {
        "clean": not failures,
        "registered": len(connections),
        "closed": closed,
        "failures": failures,
    }


def close_connections_for_path(db_path: str | Path) -> dict[str, Any]:
    """Release all thread-local connections for one durable store."""

    normalized_path = str(Path(db_path).expanduser())
    with _registry_lock:
        matching_keys = [
            key for key in _connection_registry if key[1] == normalized_path
        ]
        connections = list(
            dict.fromkeys(_connection_registry.pop(key) for key in matching_keys)
        )
        _thread_local.connections.pop(normalized_path, None)
    report = _commit_and_close(connections)
    report["path"] = normalized_path
    return report


def close_all_connections() -> dict[str, Any]:
    """Commit and close every synchronous SQLite connection owned here."""

    with _registry_lock:
        connections = list(dict.fromkeys(_connection_registry.values()))
        _connection_registry.clear()
        _thread_local.connections.clear()
    return _commit_and_close(connections)


def _shutdown_close_all_connections() -> None:
    report = close_all_connections()
    if not report["clean"]:
        raise RuntimeError(f"SQLite connection shutdown failed: {report['failures']}")


def _ensure_shutdown_registration() -> None:
    global _registered_shutdown_coordinator

    try:
        from core.runtime.shutdown_coordinator import get_shutdown_coordinator

        coordinator = get_shutdown_coordinator()
        if _registered_shutdown_coordinator is coordinator:
            return
        coordinator.register(
            _shutdown_close_all_connections,
            phase="state_vault",
            name="db_config_connections",
            timeout=5.0,
        )
        _registered_shutdown_coordinator = coordinator
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
        logger.debug("SQLite shutdown registration deferred: %s", exc)


def configure_connection(db_path: str) -> sqlite3.Connection:
    """Creates a connection to the SQLite DB and enables WAL mode (synchronous)."""
    normalized_path = str(Path(db_path).expanduser())
    with _registry_lock:
        cached = _thread_local.connections.get(normalized_path)
        if cached is not None:
            if _connection_is_open(cached):
                _ensure_shutdown_registration()
                return cached
            _thread_local.connections.pop(normalized_path, None)
            _connection_registry.pop((threading.get_ident(), normalized_path), None)

    try:
        from core.runtime.shutdown_coordinator import is_shutdown_requested

        if is_shutdown_requested():
            raise RuntimeError(f"refused new SQLite connection during shutdown: {normalized_path}")
    # not a failure: no shutdown coordinator means no shutdown to refuse for.
    except ImportError:
        pass

    path = Path(normalized_path)
    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        
    conn = sqlite3.connect(str(path), check_same_thread=False)
    
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA busy_timeout=30000;")
        conn.execute("PRAGMA cache_size=-8000;")
        conn.execute("PRAGMA wal_autocheckpoint=1000;")
        conn.commit()
    except (sqlite3.Error, OSError) as e:
        record_degradation('db_config', e)
        logger.warning("Failed to set PRAGMA options on %s: %s", db_path, e)

    with _registry_lock:
        _thread_local.connections[normalized_path] = conn
        _connection_registry[(threading.get_ident(), normalized_path)] = conn
    _ensure_shutdown_registration()
    return conn


async def configure_connection_async(db_path: str) -> Any:
    """Creates an aiosqlite connection and enables WAL mode.
    Now routes through the centralized core.db.pool to prevent connection churn.
    """
    path = Path(db_path)
    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        
    conn = await aio_pool.acquire(str(path))
    
    try:
        await conn.execute("PRAGMA journal_mode=WAL;")
        await conn.execute("PRAGMA synchronous=NORMAL;")
        await conn.execute("PRAGMA busy_timeout=30000;")  # 30s — prevents lock errors under heavy metabolic load
        await conn.execute("PRAGMA cache_size=-8000;")     # 8MB cache — reduces I/O pressure
        await conn.execute("PRAGMA wal_autocheckpoint=100;")  # Limit WAL file growth under sustained writes
        await conn.commit()
    except (sqlite3.Error, OSError) as e:
        record_degradation('db_config', e)
        logger.warning("Failed to set async PRAGMA options on %s: %s", db_path, e)
        
    return conn
