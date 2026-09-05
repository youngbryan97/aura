"""core/forge/weakness_detector.py — Weakness Detector."""
from __future__ import annotations

import logging
from typing import Any, Dict, List

logger = logging.getLogger("Aura.WeaknessDetector")


class WeaknessDetector:
    """Scans error logs, test failure counts, and trace records to identify codebase weaknesses."""

    @staticmethod
    def scan_for_weaknesses(recent_logs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyzes log sequences and identifies systems with high failure rates."""
        weaknesses = []
        failures_by_module: Dict[str, int] = {}
        error_details: Dict[str, List[str]] = {}

        for log in recent_logs:
            # Check for failure markers
            if not log.get("ok", True) or log.get("outcome") == "failure" or "error" in log:
                module = log.get("module") or log.get("executor_name") or "unknown"
                error_msg = log.get("error") or log.get("error_status") or "Generic error"
                
                failures_by_module[module] = failures_by_module.get(module, 0) + 1
                error_details.setdefault(module, []).append(str(error_msg))

        # Flag modules exceeding failure threshold (e.g. > 2 failures)
        for module, count in failures_by_module.items():
            if count >= 2:
                logger.warning("⚠️ High failure rates identified in module '%s': %d failures", module, count)
                weaknesses.append({
                    "module": module,
                    "failure_count": count,
                    "suggested_fix": f"Review error logs and patch semantic flaws in {module}",
                    "details": error_details[module][:5],
                })

        return weaknesses
