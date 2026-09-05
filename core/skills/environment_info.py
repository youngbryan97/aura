import os
import platform
import socket
from pathlib import Path
from typing import Any, Dict

from core.runtime.errors import record_degradation
from core.skills.base_skill import BaseSkill


class EnvironmentSkill(BaseSkill):
    name = "environment_info"
    description = "Self-Diagnostic: Returns information about the current server environment, location, and identity."
    effect_scope = "read_only"
    inputs = {
        "detail": "basic | full (default: basic)"
    }
    output = "Dictionary of system information."

    def match(self, goal: Dict[str, Any]) -> bool:
        obj = goal.get("objective", "").lower()
        env_keywords = ["environment", "system", "os", "platform", "hostname", "diagnostic", "where am i", "what system"]
        return any(kw in obj for kw in env_keywords)

    async def execute(self, goal: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        detail = goal.get("params", {}).get("detail", "basic")
        
        # Issue 63 Fix: Robust user detection with getpass
        try:
            import getpass
            current_user = getpass.getuser()
        except (ImportError, AttributeError, RuntimeError):
            current_user = os.getenv("USER") or os.getenv("USERNAME") or "aura_node"

        info = {
            "os": platform.system(),
            "os_release": platform.release(),
            "hostname": socket.gethostname(),
            "cwd": os.getcwd(),
            "user": current_user,
            "python_version": platform.python_version(),
            "processor": platform.processor()
        }
        
        # Detect "Cloud" vs "Local" heuristic
        if "compute" in info["hostname"] or "ec2" in info["hostname"]:
            info["environment_type"] = "Cloud/Server"
        else:
            info["environment_type"] = "Local/Workstation"

        if detail == "full":
            info.update(self._full_detail())

        return {"ok": True, "result": info, "summary": f"Running on {info['hostname']} ({info['environment_type']})"}

    @staticmethod
    def _full_detail() -> Dict[str, Any]:
        """Local-only extended diagnostics: memory, disk, CPU, uptime, battery.

        Never makes network calls — an environment self-report must not leak
        the probe itself as an external side effect.
        """
        detail: Dict[str, Any] = {}
        try:
            from core.runtime import resource_psutil as psutil
            from core.runtime.resource_observation import get_resource_observer

            vm = psutil.virtual_memory()
            detail["memory"] = {
                "total_gb": round(vm.total / (1024 ** 3), 1),
                "available_gb": round(vm.available / (1024 ** 3), 1),
                "used_percent": vm.percent,
            }
            disk = psutil.disk_usage(str(Path.home()))
            detail["disk"] = {
                "total_gb": round(disk.total / (1024 ** 3), 1),
                "free_gb": round(disk.free / (1024 ** 3), 1),
                "used_percent": disk.percent,
            }
            detail["cpu"] = {
                "logical_cores": psutil.cpu_count(logical=True),
                "load_avg_1m": round(
                    get_resource_observer().compute().load_1m,
                    2,
                ),
            }
            provenance = get_resource_observer().provenance
            detail["observation_source"] = provenance.source.value
            detail["observation_scenario_id"] = provenance.scenario_id
            import time

            boot_ts = psutil.boot_time()
            detail["uptime_hours"] = round((time.time() - boot_ts) / 3600, 1)
            battery = psutil.sensors_battery()
            if battery is not None:
                detail["battery"] = {
                    "percent": battery.percent,
                    "plugged_in": bool(battery.power_plugged),
                }
            detail["process_count"] = len(psutil.pids())
        except (ImportError, AttributeError, RuntimeError, OSError, ValueError) as exc:
            record_degradation(
                "environment_info",
                exc,
                severity="warning",
                action="returned partial full-detail environment report",
            )
            detail["full_detail_error"] = str(exc)[:200]
        return detail
