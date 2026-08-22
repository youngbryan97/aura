"""skills/network_ops.py
Real Network Operations using OS commands.
"""
import asyncio
import logging
import platform
import socket
import subprocess
from typing import Any

from core.runtime.subprocess_gateway import get_subprocess_gateway
from infrastructure import BaseSkill

logger = logging.getLogger("Skills.NetworkOps")

class NetworkOpsSkill(BaseSkill):
    name = "network_ops"
    description = "Checks network connectivity and interface status using real system calls."
    effect_scope = "privileged_mutation"

    async def execute(self, goal: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        action = goal.get("action", "check_connection")
        
        if action == "check_connection":
            return await self._check_connection()
        elif action == "list_interfaces":
            return await self._list_interfaces()
        else:
            return {"ok": False, "error": f"Unknown action: {action}"}

    async def _check_connection(self, host="8.8.8.8", port=53, timeout_s=3) -> dict[str, Any]:
        """Checks actual internet connectivity via socket (async offload)."""
        def _sync_check():
            sock = None
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(timeout_s)
                sock.connect((host, port))
                return True, "Real connection established."
            except OSError as ex:
                return False, str(ex)
            finally:
                if sock:
                    sock.close()
        
        success, msg = await asyncio.to_thread(_sync_check)
        if success:
            return {
                "ok": True,
                "status": "connected", 
                "target": host,
                "message": msg
            }
        else:
            return {
                "ok": False,
                "status": "disconnected",
                "error": msg,
                "message": "Failed to connect to real external host."
            }

    async def _list_interfaces(self) -> dict[str, Any]:
        """Runs OS command to verify network hardware (async offload)."""
        system = platform.system()
        cmd = []
        if system == "Darwin": # macOS
            cmd = ["networksetup", "-listallhardwareports"]
        elif system == "Linux":
            cmd = ["ip", "link", "show"]
        elif system == "Windows":
             cmd = ["ipconfig"]
        
        if not cmd:
             return {"ok": False, "error": "Unsupported OS"}

        try:
             result = await get_subprocess_gateway().run_async(
                 cmd,
                 capture_output=True,
                 check=True,
                 read_only=True,
                 source="skills.network_ops:list_interfaces",
                 accelerator_capability="none",
             )
             return {
                 "ok": True,
                 "output": result.stdout[:1000] # Truncate for safety
             }
        except subprocess.CalledProcessError as e:
             return {"ok": False, "error": str(e)}
