import asyncio
from pathlib import Path
from types import SimpleNamespace

from core.agency.sandboxed_modifier import SandboxedModifier
from core.tasks.managed_command import ManagedCommandResult


class _AcceptingGuard:
    def validate_modification(self, file_path: str, proposed_code: str, original_code: str = ""):
        return SimpleNamespace(
            approved=True,
            violations=[],
            requires_human=False,
            confidence=0.93,
            notes=[],
        )


def _ok(command: tuple[str, ...], stdout: str = "") -> ManagedCommandResult:
    return ManagedCommandResult(command, 0, stdout, "", 0.01)


def _fail(command: tuple[str, ...], stderr: str) -> ManagedCommandResult:
    return ManagedCommandResult(command, 1, "", stderr, 0.01)


def test_modifier_rejects_paths_outside_repo(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    calls: list[tuple[str, ...]] = []

    async def runner(command: tuple[str, ...], cwd: Path, timeout_s: float) -> ManagedCommandResult:
        calls.append(command)
        return _fail(command, "unexpected command")

    modifier = SandboxedModifier(str(repo), guard=_AcceptingGuard(), command_runner=runner)
    result = asyncio.run(modifier.modify("../escape.py", "VALUE = 2\n", "path containment check"))

    assert not result.success
    assert "Invalid target path" in result.reason
    assert not (tmp_path / "escape.py").exists()
    assert calls == []


def test_direct_apply_restores_original_when_compile_fails(tmp_path: Path):
    repo = tmp_path / "repo"
    target = repo / "pkg" / "module.py"
    target.parent.mkdir(parents=True)
    target.write_text("VALUE = 1\n")

    async def runner(command: tuple[str, ...], cwd: Path, timeout_s: float) -> ManagedCommandResult:
        return _fail(command, "SyntaxError: invalid syntax")

    modifier = SandboxedModifier(str(repo), guard=_AcceptingGuard(), command_runner=runner)
    result = asyncio.run(modifier.modify("pkg/module.py", "VALUE =\n", "reject invalid Python"))

    assert not result.success
    assert "Syntax check failed" in result.reason
    assert target.read_text() == "VALUE = 1\n"
    assert not (repo / "pkg" / "module.py.bak").exists()


def test_direct_rollback_restores_saved_backup(tmp_path: Path):
    repo = tmp_path / "repo"
    target = repo / "pkg" / "module.py"
    target.parent.mkdir(parents=True)
    target.write_text("VALUE = 1\n")

    async def runner(command: tuple[str, ...], cwd: Path, timeout_s: float) -> ManagedCommandResult:
        return _ok(command)

    modifier = SandboxedModifier(str(repo), guard=_AcceptingGuard(), command_runner=runner)
    result = asyncio.run(modifier.modify("pkg/module.py", "VALUE = 2\n", "direct rollback support"))

    assert result.success
    assert result.rollback_available is True
    assert target.read_text() == "VALUE = 2\n"
    assert modifier.rollback("pkg/module.py") is True
    assert target.read_text() == "VALUE = 1\n"
    assert not (repo / "pkg" / "module.py.bak").exists()


def test_worktree_apply_merges_logs_and_removes_worktree_path(tmp_path: Path):
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    target = repo / "pkg" / "module.py"
    target.parent.mkdir(parents=True)
    target.write_text("VALUE = 1\n")
    calls: list[tuple[tuple[str, ...], Path, float]] = []

    async def runner(command: tuple[str, ...], cwd: Path, timeout_s: float) -> ManagedCommandResult:
        calls.append((command, cwd, timeout_s))
        if command[:3] == ("git", "rev-parse", "--short"):
            return _ok(command, "abc123\n")
        return _ok(command)

    modifier = SandboxedModifier(str(repo), guard=_AcceptingGuard(), command_runner=runner)
    result = asyncio.run(modifier.modify("pkg/module.py", "VALUE = 2\n", "raise value"))

    assert result.success
    assert result.commit_hash == "abc123"
    assert result.rollback_available is True
    assert modifier.modification_log[-1]["ref"] == "abc123"

    worktree_add = [command for command, _cwd, _timeout in calls if command[:3] == ("git", "worktree", "add")]
    worktree_remove = [
        command for command, _cwd, _timeout in calls if command[:4] == ("git", "worktree", "remove", "--force")
    ]
    assert len(worktree_add) == 1
    assert len(worktree_remove) == 1
    branch_name = worktree_add[0][4]
    removed_path = Path(worktree_remove[0][4])
    assert removed_path.name == "worktree"
    assert str(removed_path) != branch_name


def test_worktree_commit_failure_is_reported_without_merge(tmp_path: Path):
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    target = repo / "pkg" / "module.py"
    target.parent.mkdir(parents=True)
    target.write_text("VALUE = 1\n")
    calls: list[tuple[str, ...]] = []

    async def runner(command: tuple[str, ...], cwd: Path, timeout_s: float) -> ManagedCommandResult:
        calls.append(command)
        if command[:2] == ("git", "commit"):
            return _fail(command, "nothing to commit")
        return _ok(command)

    modifier = SandboxedModifier(str(repo), guard=_AcceptingGuard(), command_runner=runner)
    result = asyncio.run(modifier.modify("pkg/module.py", "VALUE = 2\n", "commit failure handling"))

    assert not result.success
    assert "Commit failed" in result.reason
    assert not any(command[:2] == ("git", "merge") for command in calls)
    assert any(command[:3] == ("git", "branch", "-D") for command in calls)
