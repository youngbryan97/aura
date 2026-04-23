"""Install Package Skill
Allows Aura to install Python packages into the Sovereign Sandbox.
Essential for upgrading the "Body" (Perception libraries).
"""
import inspect
import logging
import sys
from typing import Any, Dict

from pydantic import BaseModel, Field

from core.skills.base_skill import BaseSkill

from ..sovereign.local_sandbox import LocalSandbox
from .active_coding import get_sandbox

logger = logging.getLogger("Skills.InstallPackage")

class InstallPackageParams(BaseModel):
    package_name: str = Field(..., description="The name of the Python package to install via pip.")

class InstallPackageSkill(BaseSkill):
    name = "install_package"
    description = "Installs Python packages into the Sandbox using pip."
    input_model = InstallPackageParams
    
    def __init__(self):
        super().__init__()

        
    async def execute(self, params: InstallPackageParams, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the package installation.
        """
        # Legacy support
        if isinstance(params, dict):
            try:
                params = InstallPackageParams(**params)
            except Exception as e:
                return {"ok": False, "error": f"Invalid input: {e}"}

        package_name = params.package_name
        
        if not package_name:
            return {"ok": False, "error": "No package_name provided"}
            
        import re
        if not re.match(r"^[a-zA-Z0-9_\-\.\[\]]+$", package_name):
            logger.warning("🚨 Blocked suspicious package name: %s", package_name)
            return {"ok": False, "error": f"Invalid package name: {package_name}"}
            
        try:
            sandbox = get_sandbox()
            logger.info("Installing %s in sandbox...", package_name)
            
            import shlex
            safe_package = shlex.quote(package_name)
            # Run pip install
            result = sandbox.run_command(
                f"{shlex.quote(sys.executable)} -m pip install {safe_package}",
                timeout=300,
            )
            if inspect.isawaitable(result):
                result = await result

            return {
                "ok": result.exit_code == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "exit_code": result.exit_code,
                "message": f"Installed {package_name}" if result.exit_code == 0 else f"Failed to install {package_name}"
            }
        except Exception as e:
            return {"ok": False, "error": str(e)}
