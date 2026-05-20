import asyncio

from core.runtime.errors import get_degradation_tracker
from core.self_modification import boot_validator as boot_module
from core.self_modification.boot_validator import GhostBootValidator


class ProcessReturnsHeartbeat:
    def __init__(self):
        self.killed = False

    async def communicate(self):
        return b"GHOST_HEARTBEAT_STABLE\n", b""

    def kill(self):
        self.killed = True


class ProcessWaitsUntilKilled:
    def __init__(self):
        self.killed = False

    async def communicate(self):
        if self.killed:
            return b"", b"terminated"
        await asyncio.sleep(60)
        return b"", b""

    def kill(self):
        self.killed = True


def test_overlay_escape_is_rejected_without_promoting_candidate(tmp_path):
    async def scenario():
        tracker = get_degradation_tracker()
        tracker.reset()
        sandbox = tmp_path / "sandbox"
        staging = tmp_path / "candidate.py"
        staging.write_text("value = 1\n", encoding="utf-8")

        validator = GhostBootValidator(project_root=tmp_path)
        ok, message = await validator.validate_boot(
            sandbox,
            overlay_file=("../escape.py", str(staging)),
        )

        assert ok is False
        assert "escapes sandbox" in message
        assert not (tmp_path / "escape.py").exists()
        assert not (sandbox / ".ghost_boot.py").exists()
        assert any(
            "without promoting candidate" in record.action
            for record in tracker.recent(subsystem="boot_validator")
        )
        tracker.reset()

    asyncio.run(scenario())


def test_overlay_file_is_removed_after_successful_trial(monkeypatch, tmp_path):
    async def scenario():
        process = ProcessReturnsHeartbeat()

        async def create_process(*_args, **_kwargs):
            return process

        monkeypatch.setattr(boot_module.asyncio, "create_subprocess_exec", create_process)
        sandbox = tmp_path / "sandbox"
        staging = tmp_path / "candidate.py"
        staging.write_text("value = 2\n", encoding="utf-8")

        validator = GhostBootValidator(project_root=tmp_path)
        ok, message = await validator.validate_boot(
            sandbox,
            overlay_file=("core/changed.py", str(staging)),
        )

        assert ok is True
        assert process.killed is False
        assert message == "Stable boot reached"
        assert not (sandbox / "core" / "changed.py").exists()
        assert not (sandbox / ".ghost_boot.py").exists()

    asyncio.run(scenario())


def test_existing_overlay_file_is_restored_after_successful_trial(monkeypatch, tmp_path):
    async def scenario():
        process = ProcessReturnsHeartbeat()

        async def create_process(*_args, **_kwargs):
            return process

        monkeypatch.setattr(boot_module.asyncio, "create_subprocess_exec", create_process)
        sandbox = tmp_path / "sandbox"
        original = sandbox / "core" / "changed.py"
        original.parent.mkdir(parents=True)
        original.write_text("value = 'original'\n", encoding="utf-8")
        staging = tmp_path / "candidate.py"
        staging.write_text("value = 'candidate'\n", encoding="utf-8")

        validator = GhostBootValidator(project_root=tmp_path)
        ok, message = await validator.validate_boot(
            sandbox,
            overlay_file=("core/changed.py", str(staging)),
        )

        assert ok is True
        assert process.killed is False
        assert message == "Stable boot reached"
        assert original.read_text(encoding="utf-8") == "value = 'original'\n"
        assert not (sandbox / ".ghost_boot.py").exists()

    asyncio.run(scenario())


def test_timeout_kills_child_process_and_reports_failure(monkeypatch, tmp_path):
    async def scenario():
        process = ProcessWaitsUntilKilled()

        async def create_process(*_args, **_kwargs):
            return process

        monkeypatch.setattr(boot_module.asyncio, "create_subprocess_exec", create_process)
        validator = GhostBootValidator(project_root=tmp_path)

        ok, message = await validator.validate_boot(tmp_path / "sandbox", timeout_s=0.01)

        assert ok is False
        assert process.killed is True
        assert "Timeout reached without heartbeat" in message
        assert not (tmp_path / "sandbox" / ".ghost_boot.py").exists()

    asyncio.run(scenario())
