import asyncio
import logging
import re
import subprocess
import sys
from typing import Optional, List, Dict

logger = logging.getLogger("Optimizer.PatchLibrary")

class PatchStrategy:
    name = "base_patch"
    
    def match(self, failure_reason: str) -> bool:
        return False
        
    async def apply(self, failure_reason: str) -> bool:
        """Applies the fix. Returns True if successful, False otherwise."""
        raise NotImplementedError(f"{type(self).__name__} must implement apply()")

class GitInitPatch(PatchStrategy):
    name = "git_init_fix"
    
    def match(self, failure_reason: str) -> bool:
        return "not a git repository" in failure_reason.lower()
        
    async def apply(self, failure_reason: str) -> bool:
        logger.warning("⚙️ Autonomic Core engaging 'git init' self-repair...")
        try:
            await asyncio.to_thread(subprocess.run, ["git", "init"], check=True, capture_output=True)
            await asyncio.to_thread(subprocess.run, ["git", "add", "."], check=True, capture_output=True)
            await asyncio.to_thread(
                subprocess.run,
                ["git", "commit", "-m", "Auto-Healer: Re-init corrupted repository"],
                check=True, capture_output=True,
            )
            logger.info("✅ Autonomic Core successfully repaired local Git repository.")
            return True
        except subprocess.CalledProcessError as e:
            logger.error("❌ Git repair failed: %s", e.stderr.decode() if e.stderr else e)
            return False

class PipInstallPatch(PatchStrategy):
    name = "pip_install_fix"
    
    def match(self, failure_reason: str) -> bool:
        return "modulenotfounderror" in failure_reason.lower()
        
    async def apply(self, failure_reason: str) -> bool:
        # Extract module name matches "No module named 'xyz'"
        match = re.search(r"No module named '(\w+)'", failure_reason)
        if match:
            module = match.group(1)
            
            IMPORT_TO_PIP: Dict[str, str] = {
                "aiohttp": "aiohttp",
                "google": "google-generativeai",
                "pydantic": "pydantic",
                "structlog": "structlog",
                "psutil": "psutil",
                "webrtcvad": "webrtcvad",
                "pyaudio": "PyAudio",
                "numpy": "numpy",
            }
            
            if module not in IMPORT_TO_PIP:
                logger.error("🛑 SECURITY: Blocked autonomous installation of '%s'", module)
                return False
            
            pip_package = IMPORT_TO_PIP[module]
            logger.warning("⚙️ Autonomic Core attempting to install missing module: %s (as %s)", module, pip_package)
            try:
                result = await asyncio.to_thread(
                    subprocess.run,
                    [sys.executable, "-m", "pip", "install", pip_package],
                    check=True, capture_output=True,
                )
                logger.info("✅ Autonomic Core successfully installed missing package '%s'", pip_package)
                return True
            except subprocess.CalledProcessError as e:
                logger.error("❌ Autonomic Core failed to install '%s': %s", pip_package, e.stderr.decode() if e.stderr else e)
                return False
        return False

# Registry
def get_patches() -> List[PatchStrategy]:
    return [GitInitPatch(), PipInstallPatch()]