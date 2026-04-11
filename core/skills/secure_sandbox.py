"""Secure Docker Sandbox for Sovereign Code Execution.
Force-disables networking and restricts resources.
"""

import io
import logging
import os
import shutil
import tarfile
from pathlib import Path
from typing import Any, Dict, Optional

import docker

logger = logging.getLogger("Aura.SecureSandbox")

class SecureDockerSandbox:
    """Executes untrusted code in a network-isolated Docker container.
    """
    
    def __init__(self, image_name: str = "aura-sovereign-sandbox:latest"):
        try:
            self.client = docker.from_env()
            self.image_name = image_name
        except Exception as e:
            logger.error("Docker initialization failed: %s", e)
            self.client = None

    def execute_code(self, code: str, workspace_path: str, timeout: int = 30) -> Dict[str, Any]:
        """Run code in the isolated container.
        """
        if not self.client:
            return {"ok": False, "error": "Docker not available on host."}

        container = None
        try:
            # Create container with strict limits
            container = self.client.containers.run(
                image=self.image_name,
                command=["python3", "-c", code],
                network_disabled=True,      # KEY REQUIREMENT: Zero network access
                mem_limit="1g",             # Resource limit: Memory (M5/64GB)
                nano_cpus=2000000000,       # Resource limit: 2.0 CPU (M5)
                detach=True,
                remove=False,
                stderr=True,
                stdout=True
            )

            # Wait for completion
            try:
                result = container.wait(timeout=timeout)
                logs = container.logs().decode("utf-8")
                exit_code = result.get("StatusCode", 1)
                
                return {
                    "ok": exit_code == 0,
                    "exit_code": exit_code,
                    "output": logs
                }
            except Exception as e:
                container.kill()
                return {"ok": False, "error": f"Execution timeout or error: {str(e)}"}

        except Exception as e:
            logger.error("Sandbox execution fatal error: %s", e)
            return {"ok": False, "error": str(e)}
        finally:
            if container:
                try:
                    container.remove(force=True)
                except Exception as exc:
                    logger.debug("Suppressed: %s", exc)

    def verify_safety(self) -> bool:
        """v6.1: Verify network isolation using exit code detection (SK-03).
        A non-zero exit code means networking was successfully blocked.
        """
        test_code = (
            "import urllib.request, sys\n"
            "try:\n"
            "    # Attempt connection to a public IP\n"
            "    urllib.request.urlopen('http://1.1.1.1', timeout=2)\n"
            "    sys.exit(0)  # Network accessible — UNSAFE\n"
            "except Exception:\n"
            "    sys.exit(1)  # Network blocked — SAFE\n"
        )
        result = self.execute_code(test_code, "/tmp", timeout=10)
        
        # exit_code 1 (non-zero) means the exception was caught -> Network BLOCKED -> SAFE
        is_safe = result.get("exit_code", 0) != 0
        
        if is_safe:
            logger.info("Sandbox Safety Verified: Network access blocked.")
        else:
            logger.critical("🛡️ SANDBOX SAFETY CHECK FAILED: Network may be accessible!")
            
        return is_safe