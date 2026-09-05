"""core/forge/shadow_runner.py — Shadow Runner."""
from __future__ import annotations

import logging
from typing import Any, Dict

from core.runtime.action_executor import ActionExecutor
from core.will import ActionDomain

logger = logging.getLogger("Aura.ShadowRunner")


class ShadowRunner:
    """Runs patches in shadow/sandboxed execution modes to prove compile safety."""

    @staticmethod
    async def run_shadow_tests(patch_path: str, test_cmd: str, source: str = "shadow_runner") -> Dict[str, Any]:
        """Executes a sandboxed run of the patch using ActionExecutor.SELF_MODIFICATION."""
        logger.info("Initiating shadow run for patch: %s with test command: %s", patch_path, test_cmd)
        
        result = await ActionExecutor.execute(
            domain=ActionDomain.SELF_MODIFICATION,
            action_name="self_modification.shadow_run",
            params={
                "patch_path": patch_path,
                "test_command": test_cmd,
            },
            source=source,
        )

        return result
