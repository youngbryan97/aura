from core.runtime.errors import record_degradation
import logging
import os
import platform
import socket
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from pathlib import Path

from core.skills.base_skill import BaseSkill


class EnvironmentSkill(BaseSkill):
    name = "environment_info"
    description = "Self-Diagnostic: Returns information about the current server environment, location, and identity."
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
        except Exception:
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
            try:
                # Add Public IP (External Request)
                # Keep it safe/fast, optional
                pass  # no-op: intentional
            except Exception as _e:  # Non-critical, fallback handled
                logging.debug('Ignored Exception in environment_info.py: %s', _e)

        return {"ok": True, "result": info, "summary": f"Running on {info['hostname']} ({info['environment_type']})"}