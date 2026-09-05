"""core/security/egress_monitor.py
Monitors outbound egress traffic and active network connections.
"""
import time
from typing import Any


class EgressConnectionMonitor:
    """Logs outgoing network sockets for auditing purposes."""

    def __init__(self):
        self._connections: list[dict[str, Any]] = []

    def record_connection(self, host: str, port: int) -> None:
        self._connections.append({
            "timestamp": time.time(),
            "host": host,
            "port": port
        })

    def get_recent_connections(self, limit: int = 50) -> list[dict[str, Any]]:
        return self._connections[-limit:]
