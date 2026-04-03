import logging
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger("Core.Antibody")

# Environment variable names that may indicate hostile instrumentation
_SUSPICIOUS_ENV_VARS = {
    "LD_PRELOAD", "DYLD_INSERT_LIBRARIES", "PYTHONDONTWRITEBYTECODE",
    "http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY",
    "DEBUGGER", "PYDEVD_USE_FRAME_EVAL",
}

class Antibody:
    """Active Defense System (v4.0).
    Monitors for file modifications, unexpected processes, and sensor anomalies.
    """

    def __init__(self, immune_system):
        self.immune = immune_system
        self.watch_list = [
            Path("core/prime_directives.py"),
            Path("core/immune_system.py"),
            Path(".env")
        ]
        self.baseline_stats = self._get_stats()
        self._known_ports: set = set()
        logger.info("🛡️ Antibody Deployed: Passive Monitoring Active")

    def _get_stats(self) -> Dict[Path, float]:
        stats = {}
        for p in self.watch_list:
            if p.exists():
                stats[p] = p.stat().st_mtime
        return stats

    def check_integrity(self) -> bool:
        """Scan for unauthorized changes."""
        current_stats = self._get_stats()
        for p, mtime in current_stats.items():
            if mtime != self.baseline_stats.get(p):
                logger.warning("⚠️ UNAUTHORIZED MODIFICATION DETECTED: %s", p)
                return False
        return True

    def scan_environment(self) -> Dict[str, Any]:
        """Check for potentially dangerous environment changes.

        Returns a dict with:
            alerts   – list of human-readable warnings
            clean    – True if no alerts were raised
        """
        alerts: List[str] = []

        # 1. Check for suspicious environment variables
        for var in _SUSPICIOUS_ENV_VARS:
            val = os.environ.get(var)
            if val:
                alerts.append(f"Suspicious env var set: {var}={val[:60]}")

        # 2. Detect unexpected listening ports (requires lsof)
        if shutil.which("lsof"):
            try:
                result = subprocess.run(
                    ["lsof", "-iTCP", "-sTCP:LISTEN", "-nP"],
                    capture_output=True, text=True, timeout=5,
                )
                current_ports: set = set()
                for line in result.stdout.splitlines()[1:]:
                    parts = line.split()
                    if len(parts) >= 9:
                        current_ports.add(parts[8])  # e.g. *:8000

                # First scan — establish baseline
                if not self._known_ports:
                    self._known_ports = current_ports
                else:
                    new_ports = current_ports - self._known_ports
                    for port in new_ports:
                        alerts.append(f"New listening port detected: {port}")
                    # Update baseline to include newly seen ports
                    self._known_ports = current_ports
            except (subprocess.TimeoutExpired, OSError) as e:
                logger.debug("Port scan unavailable: %s", e)

        if alerts:
            for a in alerts:
                logger.warning("🛡️ Antibody Alert: %s", a)

        return {"alerts": alerts, "clean": len(alerts) == 0}