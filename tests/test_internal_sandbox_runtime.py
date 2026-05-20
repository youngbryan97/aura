import asyncio

from core.runtime.errors import get_degradation_tracker
from core.skills.internal_sandbox import SandboxSkill


def test_internal_sandbox_executes_safe_python(tmp_path):
    async def scenario():
        skill = SandboxSkill()
        result = await skill.execute_code_safely("print('sandbox-ok')", cwd=str(tmp_path))

        assert result["ok"] is True
        assert "sandbox-ok" in result["result"]

    asyncio.run(scenario())


def test_internal_sandbox_blocks_network_import(tmp_path):
    async def scenario():
        skill = SandboxSkill()
        result = await skill.execute_code_safely(
            "import socket\nprint('network-open')",
            cwd=str(tmp_path),
        )

        assert result["ok"] is False
        assert "blocked in sandbox environment" in result["result"]
        assert "network-open" not in result["result"]

    asyncio.run(scenario())


def test_internal_sandbox_timeout_reaps_process(tmp_path):
    async def scenario():
        skill = SandboxSkill()
        skill.MAX_EXECUTION_TIME = 0.05

        result = await skill.execute_code_safely("while True:\n    pass", cwd=str(tmp_path))

        assert result["ok"] is False
        assert "timed out" in result["error"]

    asyncio.run(scenario())


def test_internal_sandbox_invalid_cwd_records_boundary_failure(tmp_path):
    async def scenario():
        tracker = get_degradation_tracker()
        tracker.reset()
        skill = SandboxSkill()

        result = await skill.execute_code_safely(
            "print('unreached')", cwd=str(tmp_path / "missing")
        )

        assert result["ok"] is False
        assert "Sandbox Exception" in result["error"]
        assert any(
            "explicit sandbox failure payload" in record.action
            for record in tracker.recent(subsystem="internal_sandbox")
        )
        tracker.reset()

    asyncio.run(scenario())
