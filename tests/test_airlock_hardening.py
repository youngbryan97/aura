import asyncio
import sys
from pathlib import Path

from core.guardians.airlock import AirlockProtocol
from core.tasks.managed_command import ManagedCommandResult


def _ok(command: tuple[str, ...], stdout: str = "") -> ManagedCommandResult:
    return ManagedCommandResult(command, 0, stdout, "", 0.01)


def _fail(command: tuple[str, ...], stderr: str = "failed") -> ManagedCommandResult:
    return ManagedCommandResult(command, 1, "", stderr, 0.01)


def _git(command: tuple[str, ...]) -> tuple[str, ...]:
    """A git command without its -c settings.

    The airlock passes `-c core.fsmonitor=false` on every call (2026-09-20,
    e3c33f7cc5), which shifted every token and made a test that matched the
    first three read as "no patch check ran".
    """
    if not command or command[0] != "git":
        return command
    rest = list(command[1:])
    while len(rest) >= 2 and rest[0] == "-c":
        del rest[:2]
    return ("git", *rest)



def test_airlock_success_keeps_patch_artifact_outside_worktree(tmp_path: Path):
    repo_root = tmp_path / "repo"
    sandbox_dir = tmp_path / "airlock_worktree"
    repo_root.mkdir()
    calls: list[tuple[tuple[str, ...], Path, float]] = []

    async def runner(command: tuple[str, ...], cwd: Path, timeout_s: float) -> ManagedCommandResult:
        calls.append((command, cwd, timeout_s))
        return _ok(command, "ok")

    airlock = AirlockProtocol(
        repo_root=repo_root,
        sandbox_dir=sandbox_dir,
        command_runner=runner,
        test_timeout_s=12.0,
    )

    result = asyncio.run(airlock.process_mutation("bad/../branch name", "diff --git a/x b/x\n", "tighten branch"))

    assert result["success"] is True
    assert result["branch"] == "mutation/bad-branch-name"
    apply_commands = [command for command, _cwd, _timeout in calls if _git(command)[:3] == ("git", "apply", "--check")]
    assert len(apply_commands) == 1
    patch_path = Path(apply_commands[0][-1])
    assert sandbox_dir not in patch_path.parents
    assert any(_git(command) == ("git", "add", "-A") for command, _cwd, _timeout in calls)
    assert any(command[:3] == (sys.executable, "-m", "pytest") for command, _cwd, _timeout in calls)


def test_airlock_returns_gauntlet_output_on_test_failure(tmp_path: Path):
    repo_root = tmp_path / "repo"
    sandbox_dir = tmp_path / "airlock_worktree"
    repo_root.mkdir()

    async def runner(command: tuple[str, ...], cwd: Path, timeout_s: float) -> ManagedCommandResult:
        if command[:3] == (sys.executable, "-m", "pytest"):
            return ManagedCommandResult(command, 1, "test out", "test err", 0.01)
        return _ok(command)

    airlock = AirlockProtocol(repo_root=repo_root, sandbox_dir=sandbox_dir, command_runner=runner)
    result = asyncio.run(airlock.process_mutation("h1", "diff --git a/x b/x\n", "test failure"))

    assert result["success"] is False
    assert result["reason"] == "Test suite failed."
    assert result["stdout"] == "test out"
    assert result["stderr"] == "test err"


def test_airlock_recovers_after_patch_failure(tmp_path: Path):
    repo_root = tmp_path / "repo"
    sandbox_dir = tmp_path / "airlock_worktree"
    repo_root.mkdir()
    calls: list[tuple[str, ...]] = []

    async def runner(command: tuple[str, ...], cwd: Path, timeout_s: float) -> ManagedCommandResult:
        calls.append(command)
        if _git(command)[:3] == ("git", "apply", "--check"):
            return _fail(command, "patch rejected")
        return _ok(command)

    airlock = AirlockProtocol(repo_root=repo_root, sandbox_dir=sandbox_dir, command_runner=runner)
    result = asyncio.run(airlock.process_mutation("h2", "not a patch", "patch failure"))

    assert result["success"] is False
    assert "patch check failed" in result["reason"]
    assert any(_git(command)[:3] == ("git", "worktree", "remove") for command in calls)


def test_airlock_branch_suffix_is_stable_and_safe():
    assert AirlockProtocol._safe_branch_suffix("../") != ""
    assert AirlockProtocol._safe_branch_suffix("abc DEF/ghi") == "abc-DEF-ghi"
