################################################################################

import asyncio
import logging
import uuid
from typing import Any

import pytest

from core.skill_management.hephaestus import HephaestusEngine
from core.runtime.subprocess_gateway import get_subprocess_gateway

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ChaosTest")


class DraftingHephaestus(HephaestusEngine):
    """Hephaestus instance with an injected draft provider for adversarial validation."""

    def __init__(self, draft: dict[str, Any]):
        super().__init__()
        self.draft = dict(draft)
        self.requests: list[tuple[str, str]] = []

    async def _draft_logic(self, name: str, objective: str) -> dict[str, Any]:
        self.requests.append((name, objective))
        return dict(self.draft)


@pytest.mark.asyncio
async def test_oom_denial_of_service():
    """Verify if a rogue skill can crash the system via memory exhaustion."""
    logger.info("STARTING: OOM Denial of Service Test")
    rogue_logic = """
junk = bytearray(1024 * 1024 * 1024)
return {"ok": True, "len": len(junk)}
"""
    engine = DraftingHephaestus(
        {
            "ok": True,
            "code": rogue_logic,
            "description": "Rogue OOM",
            "logic_description": "Exhaust memory",
        }
    )

    logger.info("Attempting to synthesize rogue skill...")
    result = await engine.synthesize_skill("rogue_oom", "test")
    logger.info(f"Result: {result}")

    assert engine.requests == [("rogue_oom", "test")]
    assert result.get("ok") is False
    assert "OOM Risk" in result.get("error", "")


@pytest.mark.asyncio
async def test_recursive_fork_bomb():
    """Verify if a skill can spawn background processes that outlive the sandbox."""
    logger.info("STARTING: Fork Bomb Persistence Test")
    sentinel = f"aura-chaos-fork-persistence-{uuid.uuid4().hex}"
    rogue_logic = f"""
import subprocess
import sys
subprocess.Popen(
    [
        sys.executable,
        "-c",
        "import time; time.sleep(60)",
        "{sentinel}",
    ],
    start_new_session=True,
)
return {{"ok": True}}
"""
    engine = DraftingHephaestus(
        {
            "ok": True,
            "code": rogue_logic,
            "description": "Fork Persistence",
            "logic_description": "Spawn background",
        }
    )

    logger.info("Attempting to synthesize fork persistence skill...")
    result = await engine.synthesize_skill("fork_persistence", "test")
    assert result.get("ok") is False
    assert "Forbidden Import" in result.get("error", "")

    # Check if a sleeper process is still alive
    await asyncio.sleep(0.5)
    ps = get_subprocess_gateway().run(
        ["ps", "-ef"],
        timeout=10,
        read_only=True,
        source="certification_tooling:chaos_test_process_scan",
        accelerator_capability="none",
    )
    sleeper_survived = sentinel in ps.stdout
    if sleeper_survived:
        logger.error("FAILURE: Background process survived the sandbox!")
        get_subprocess_gateway().run(
            ["pkill", "-f", sentinel],
            timeout=10,
            offline_tooling=True,
            source="certification_tooling:chaos_test_cleanup",
            accelerator_capability="none",
        )
    else:
        logger.info("SUCCESS: No orphaned background processes detected.")
    assert sleeper_survived is False

if __name__ == "__main__":
    asyncio.run(test_oom_denial_of_service())
    asyncio.run(test_recursive_fork_bomb())
