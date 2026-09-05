"""core/body/file_motor.py
Workspace file actuator executing create, write, and delete operations.
"""
import logging
from pathlib import Path
from typing import Any

from core.body.motor_controller import BaseMotor
from core.runtime.errors import record_degradation
from core.runtime.file_write_gateway import get_file_write_gateway

logger = logging.getLogger("Body.FileMotor")

_FILE_MOTOR_ERRORS = (OSError, RuntimeError, TimeoutError, TypeError, ValueError)


class FileMotor(BaseMotor):
    """Executes filesystem changes inside workspace boundaries."""

    @property
    def name(self) -> str:
        return "file"

    async def actuate(self, params: dict[str, Any]) -> dict[str, Any]:
        action = params.get("action", "write")
        path_str = params.get("path")
        content = params.get("content", "")

        if not path_str:
            return {"status": "error", "message": "Missing file path"}

        path = Path(path_str).resolve()
        
        try:
            gateway = get_file_write_gateway()
            if action == "write":
                await gateway.write_text_async(path, content, source="organism.file_motor")
                return {
                    "status": "success",
                    "action": "write",
                    "path": str(path),
                    "bytes": len(content)
                }
            elif action == "delete":
                if gateway.delete_file(path, source="organism.file_motor"):
                    return {"status": "success", "action": "delete", "path": str(path)}
                return {"status": "ignored", "message": "File does not exist"}
        except _FILE_MOTOR_ERRORS as e:
            record_degradation("body.file_motor", e)
            logger.error("File motor activation failed: %s", e)
            return {"status": "error", "message": str(e)}

        return {"status": "error", "message": f"Unsupported action: {action}"}
