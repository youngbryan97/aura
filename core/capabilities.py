from core.runtime.errors import record_degradation
import logging
import subprocess
import asyncio
from typing import Dict, List, Optional, Tuple

import requests

logger = logging.getLogger("Kernel.Capabilities")

class Shell:
    def __init__(self, cwd: str, allowed_commands: Optional[List[str]] = None, timeout: int = 30):
        self.cwd = cwd
        self.allowed_commands = allowed_commands or []
        self.timeout = timeout

    def _is_allowed(self, cmd: List[str]) -> bool:
        if not self.allowed_commands:
            return True # No restrictions if list is empty
        base_cmd = cmd[0]
        return any(base_cmd == allowed or base_cmd.endswith("/" + allowed) for allowed in self.allowed_commands)

    async def run(self, cmd: List[str]) -> Tuple[bool, str]:
        if not self._is_allowed(cmd):
            logger.warning("Blocked unauthorized shell command: %s", cmd)
            return False, f"Command {cmd[0]} not in allowlist"
        
        logger.info("Shell.run: %s", ' '.join(cmd[:5]))
        try:
            result = await asyncio.to_thread(
                subprocess.run,
                cmd,
                cwd=self.cwd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            out = (result.stdout + "\n" + result.stderr).strip()
            return result.returncode == 0, out
        except Exception as e:
            record_degradation('capabilities', e)
            logger.error("Shell error: %s", e)
            return False, str(e)


class WebClient:
    def __init__(self, allowed_domains: Optional[List[str]] = None, timeout: int = 10):
        self.allowed_domains = allowed_domains or []
        self.timeout = timeout

    def _is_allowed(self, url: str) -> bool:
        if not self.allowed_domains:
            return True
        from urllib.parse import urlparse
        domain = urlparse(url).netloc
        return any(domain == d or domain.endswith("." + d) for d in self.allowed_domains)

    async def get(self, url: str, headers: Optional[Dict[str, str]] = None) -> Tuple[bool, str]:
        if not self._is_allowed(url):
            logger.warning("Blocked unauthorized domain access: %s", url)
            return False, f"Domain not in allowlist: {url}"
            
        logger.info("WebClient.get: %s", url[:80])
        try:
            resp = await asyncio.to_thread(requests.get, url, headers=headers, timeout=self.timeout)
            return True, resp.text
        except Exception as e:
            record_degradation('capabilities', e)
            logger.error("Web error: %s", e)
            return False, str(e)

