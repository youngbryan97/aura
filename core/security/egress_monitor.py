"""core/security/egress_monitor.py
Monitors outbound egress traffic and active network connections.
"""
from typing import Dict, Any, List
import time


class EgressConnectionMonitor:
    """Logs outgoing network sockets for auditing purposes."""

    def __init__(self):
        self._connections: List[Dict[str, Any]] = []

    def record_connection(self, host: str, port: int) -> None:
        self._connections.append({
            "timestamp": time.time(),
            "host": host,
            "port": port
        })

    def get_recent_connections(self, limit: int = 50) -> List[Dict[str, Any]]:
        return self._connections[-limit:]
