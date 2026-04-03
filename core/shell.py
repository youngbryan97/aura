import shlex
import subprocess
import asyncio
import logging
from typing import Dict, Any


class ShellInterface:
    def __init__(self, timeout=30):
        self.timeout = timeout
        self.logger = logging.getLogger("Shell")

    async def execute(self, command):
        """Executes a shell command safely (async offload).
        Captures stdout and stderr.
        """
        self.logger.info("Executing: %s", command)
        try:
            # Tokenize command safely
            if isinstance(command, str):
                tokens = shlex.split(command)
            else:
                tokens = list(command)
            if not tokens:
                return {"success": False, "error": "Empty command.", "stdout": "", "stderr": "", "code": -1}
            
            result = await asyncio.to_thread(
                subprocess.run,
                tokens,
                shell=False,
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            return {
                "success": result.returncode == 0,
                "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip(),
                "code": result.returncode
            }
        except asyncio.TimeoutError:
            return {"success": False, "error": "Command timed out.", "stdout": "", "stderr": "", "code": -1}
        except Exception as e:
            return {"success": False, "error": str(e), "stdout": "", "stderr": "", "code": -1}

    async def write_file_safe(self, target_file: str, content: str) -> Dict[str, Any]:
        """Write content to a file with an automated safety snapshot."""
        from core.resilience.sandbox_manager import sandbox_manager
        
        # 1. Snapshot
        snapshot = sandbox_manager.create_snapshot(target_file)
        
        # 2. Write
        try:
            full_path = (sandbox_manager.base_dir / target_file).resolve()
            if not str(full_path).startswith(str(sandbox_manager.base_dir.resolve())):
                return {"success": False, "error": f"Path traversal attempt blocked: {target_file}"}
                
            with open(full_path, "w") as f:
                f.write(content)
            
            # 3. Verify
            health = await sandbox_manager.verify_integrity()
            if not health["integrity_ok"]:
                logging.getLogger("Shell").error("Integrity check FAILED after write to %s. Rolling back!", target_file)
                if snapshot:
                    sandbox_manager.restore_snapshot(target_file, snapshot)
                return {"success": False, "error": "Integrity check failed. Rollback triggered."}
                
            return {"success": True, "file": target_file}
        except Exception as e:
            logging.getLogger("Shell").error("Failed to write %s safely: %s", target_file, e)
            if snapshot:
                sandbox_manager.restore_snapshot(target_file, snapshot)
            return {"success": False, "error": str(e)}