"""The shell skill runs inside a boundary, or it does not run.

Three properties, each of which was false before and none of which any test
covered — the skill had no test file at all.

1. Containment is ancestry, not a string prefix. ``/allowed/project-evil``
   begins with ``/allowed/project``; it is a sibling, not a child. The ``rm``
   guard compared with ``startswith`` and admitted it.
2. Isolation is the default. ``sandbox`` defaulted to ``False`` and the host
   path needed no separate authorization.
3. The gate is asked. Nothing in the file reached
   ``core.security.execution_authority``, so no standing directive could
   reach it either.

The seatbelt assertions read the generated profile rather than trusting that
the denials were written once, and the macOS test runs a real command under
it, because a profile that never reaches the kernel proves nothing.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from core.security.execution_authority import KIND_SHELL, ExecutionVerdict
from security.sandbox import SecureSandbox, SecurityLevel
from skills.shell import HOST_EXECUTION_PARAM, ShellSkill, _contains


def _approved() -> ExecutionVerdict:
    return ExecutionVerdict(
        approved=True, reason="ok", kind=KIND_SHELL, descriptor="", outcome="approved"
    )


def _denied() -> ExecutionVerdict:
    return ExecutionVerdict(
        approved=False,
        reason="the Will refused",
        kind=KIND_SHELL,
        descriptor="",
        outcome="denied",
    )


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """A workspace root with a sibling whose name shares its prefix."""
    root = tmp_path / "project"
    root.mkdir()
    sibling = tmp_path / "project-evil"
    sibling.mkdir()
    (sibling / "loot.txt").write_text("secret", encoding="utf-8")

    monkeypatch.setattr("skills.shell._workspace_root", lambda: root)
    return root, sibling


# ─────────────────────────────────────────────────────────── containment


def test_a_sibling_sharing_the_prefix_is_not_inside(workspace):
    root, sibling = workspace
    assert _contains(root, root)
    assert _contains(root, root / "sub" / "file.txt")
    assert not _contains(root, sibling)
    assert not _contains(root, sibling / "loot.txt")
    # The exact comparison the old code made, so the regression is named.
    assert str(sibling).startswith(str(root))


def test_rm_refuses_the_prefix_sibling(workspace):
    root, sibling = workspace
    skill = ShellSkill()
    skill.cwd = str(root)

    safe, reason = skill._is_safe_command(f"rm {sibling / 'loot.txt'}")
    assert safe is False
    assert "outside workspace" in reason


def test_rm_inside_the_workspace_still_passes(workspace):
    root, _ = workspace
    skill = ShellSkill()
    skill.cwd = str(root)
    (root / "scratch.txt").write_text("x", encoding="utf-8")

    safe, reason = skill._is_safe_command("rm scratch.txt")
    assert safe is True, reason


def test_cd_cannot_leave_the_workspace(workspace):
    from skills.shell import _resolve_cd_target

    root, sibling = workspace
    _, allowed = _resolve_cd_target(str(root), str(sibling))
    assert allowed is False
    _, allowed_inside = _resolve_cd_target(str(root), ".")
    assert allowed_inside is True


# ──────────────────────────────────────────────────── isolation defaults


@pytest.mark.asyncio
async def test_leaving_the_sandbox_must_be_asked_for_by_name(workspace, monkeypatch):
    """`sandbox=False` alone is refused.

    A caller written before isolation existed passes `sandbox=False` meaning
    "the old behaviour". Honouring that would put every un-updated caller
    back on the host silently, which is the failure this default was changed
    to remove.
    """
    spawned: list = []
    monkeypatch.setattr(
        "skills.shell.authorize_execution",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not reach the gate")),
    )
    skill = ShellSkill()
    result = await skill.execute(
        {"params": {"command": "echo hi", "sandbox": False}}, {}
    )
    assert result["ok"] is False
    assert HOST_EXECUTION_PARAM in result["error"]
    assert not spawned


@pytest.mark.asyncio
async def test_the_default_path_declares_the_isolation_it_asked_for(workspace, monkeypatch):
    seen: dict = {}

    async def _fake_authorize(kind, argv, **kwargs):
        seen["kind"] = kind
        seen["extra"] = kwargs.get("extra", {})
        return _denied()

    monkeypatch.setattr("skills.shell.authorize_execution", _fake_authorize)
    skill = ShellSkill()
    await skill.execute({"params": {"command": "echo hi"}}, {})

    assert seen["kind"] == KIND_SHELL
    # The field a standing directive is written against.
    assert seen["extra"]["isolation"] == "sandbox"


@pytest.mark.asyncio
async def test_host_execution_declares_itself_to_the_gate(workspace, monkeypatch):
    seen: dict = {}

    async def _fake_authorize(kind, argv, **kwargs):
        seen["extra"] = kwargs.get("extra", {})
        return _denied()

    monkeypatch.setattr("skills.shell.authorize_execution", _fake_authorize)
    skill = ShellSkill()
    await skill.execute(
        {"params": {"command": "echo hi", "sandbox": False, HOST_EXECUTION_PARAM: True}},
        {},
    )

    assert seen["extra"]["isolation"] == "host"


@pytest.mark.asyncio
async def test_a_refusal_never_reaches_a_spawn(workspace, monkeypatch):
    async def _fake_authorize(*a, **k):
        return _denied()

    def _explode(*a, **k):
        raise AssertionError("spawned despite a refusal")

    monkeypatch.setattr("skills.shell.authorize_execution", _fake_authorize)
    monkeypatch.setattr("skills.shell.get_subprocess_gateway", _explode)

    skill = ShellSkill()
    result = await skill.execute({"params": {"command": "echo hi"}}, {})
    assert result["ok"] is False
    assert result["governance"]["authorized"] is False


@pytest.mark.asyncio
async def test_a_persistent_session_is_not_reported_as_sandboxed(workspace, monkeypatch):
    """The daemon path keeps a host shell alive; it cannot also be confined."""
    async def _fake_authorize(*a, **k):
        return _approved()

    monkeypatch.setattr("skills.shell.authorize_execution", _fake_authorize)
    monkeypatch.setattr("skills.shell.release_execution", lambda *a, **k: None)

    skill = ShellSkill()
    result = await skill.execute(
        {"params": {"command": "echo hi", "persistent_session_id": "s1"}}, {}
    )
    assert result["ok"] is False
    assert HOST_EXECUTION_PARAM in result["error"]


# ──────────────────────────────────────────────────────── the profile


def _profile_for(tmp_path: Path) -> str:
    workdir = tmp_path / "work"
    workdir.mkdir()
    sandbox = SecureSandbox(
        security_level=SecurityLevel.CONFINED,
        workdir=workdir,
        allowed_paths=[workdir],
        read_paths=[workdir],
    )
    return sandbox.build_seatbelt_profile()


def test_the_profile_denies_the_network_and_defaults(tmp_path):
    profile = _profile_for(tmp_path)
    assert "(deny default)" in profile
    assert "(deny network*)" in profile


def test_naming_read_paths_denies_the_roots_that_hold_user_data(tmp_path):
    """Reads stay open, then the person's files are taken away.

    An allowlist was measured and does not work — dyld aborts the process
    before ``main``. What holds is the deny, which must therefore appear
    AFTER the blanket allow, since seatbelt takes the last matching rule.
    """
    profile = _profile_for(tmp_path)
    allow_at = profile.index("(allow file-read*)")
    deny_at = profile.index("(deny file-read*")
    assert deny_at > allow_at, "the deny is dead unless it follows the allow"
    for root in ("/Users", "/Volumes", "/private/var/root"):
        assert f'(subpath "{root}")' in profile


def test_the_workdir_is_re_allowed_after_the_deny(tmp_path):
    """The workdir normally sits inside a denied root."""
    profile = _profile_for(tmp_path)
    deny_at = profile.index("(deny file-read*")
    workdir = str((tmp_path / "work").resolve())
    assert workdir in profile
    assert profile.index(workdir) > deny_at


def test_writes_do_not_reach_every_temp_container(tmp_path):
    """`/private/var/folders` is every per-user temp dir, not this one's."""
    profile = _profile_for(tmp_path)
    write_block = profile.split("(allow file-write*")[1].split("\n)")[0]
    assert '(subpath "/private/var/folders")' not in write_block
    assert '(subpath "/Users")' not in write_block


def test_confined_carries_no_binary_allowlist():
    """The Will decides what runs; a four-entry list would overrule it."""
    sandbox = SecureSandbox(security_level=SecurityLevel.CONFINED)
    try:
        assert sandbox.allowed_commands is None
    finally:
        sandbox.cleanup()


def test_confined_is_still_kernel_enforced_on_macos():
    sandbox = SecureSandbox(security_level=SecurityLevel.CONFINED)
    try:
        assert sandbox.kernel_enforced() is (sys.platform == "darwin")
    finally:
        sandbox.cleanup()


def test_privileged_is_the_only_level_with_no_boundary():
    sandbox = SecureSandbox(security_level=SecurityLevel.PRIVILEGED)
    try:
        assert sandbox.kernel_enforced() is False
    finally:
        sandbox.cleanup()


def test_the_launch_environment_carries_no_secrets(tmp_path, monkeypatch):
    monkeypatch.setenv("AURA_TEST_API_TOKEN", "sk-should-not-survive")
    monkeypatch.setenv("AURA_TEST_PLAIN", "keep-me")
    workdir = tmp_path / "work"
    workdir.mkdir()
    sandbox = SecureSandbox(security_level=SecurityLevel.CONFINED, workdir=workdir)
    launch = sandbox.prepare_launch(["/bin/echo", "hi"])

    assert "AURA_TEST_API_TOKEN" not in launch.env
    assert launch.env.get("AURA_TEST_PLAIN") == "keep-me"


def test_the_launch_prefixes_the_seatbelt_on_macos(tmp_path):
    workdir = tmp_path / "work"
    workdir.mkdir()
    sandbox = SecureSandbox(security_level=SecurityLevel.CONFINED, workdir=workdir)
    launch = sandbox.prepare_launch(["/bin/echo", "hi"])

    if sys.platform == "darwin":
        assert launch.argv[0] == "sandbox-exec"
        assert launch.kernel_enforced is True
        assert launch.profile_path is not None
        assert oct(os.stat(launch.profile_path).st_mode)[-3:] == "600"
    else:
        assert launch.argv[0] == "/bin/echo"
        assert launch.kernel_enforced is False


# ─────────────────────────────────────────── the kernel, not the profile


@pytest.mark.macos
@pytest.mark.skipif(sys.platform != "darwin", reason="seatbelt is macOS-only")
def test_the_denied_roots_cover_where_a_person_keeps_files():
    """The configuration, checked separately from the mechanism.

    The kernel tests below run under a workdir in the temp tree, so they
    prove the deny works. This proves it is pointed at the right place: a
    home directory lives under /Users, and a mounted disk under /Volumes.
    """
    from security.sandbox import _USER_DATA_READ_ROOTS

    home = Path.home().resolve()
    assert any(
        home == Path(root) or Path(root) in home.parents
        for root in _USER_DATA_READ_ROOTS
    ), f"no denied root contains {home}"
    assert "/Volumes" in _USER_DATA_READ_ROOTS


@pytest.mark.macos
@pytest.mark.skipif(sys.platform != "darwin", reason="seatbelt is macOS-only")
def test_the_kernel_refuses_a_read_outside_the_named_paths(tmp_path):
    """A profile that never reaches the kernel proves nothing.

    Two runs of the same binary under the same sandbox: one file inside the
    granted read scope, one outside it. If both succeed, the profile was
    written and ignored.
    """
    workdir = tmp_path / "work"
    workdir.mkdir()
    inside = workdir / "inside.txt"
    inside.write_text("visible", encoding="utf-8")
    outside = tmp_path / "outside.txt"
    outside.write_text("private", encoding="utf-8")

    sandbox = SecureSandbox(
        security_level=SecurityLevel.CONFINED,
        workdir=workdir,
        allowed_paths=[workdir],
        read_paths=[workdir],
    )
    allowed = sandbox.execute_command(["/bin/cat", str(inside)], timeout=20.0)
    refused = sandbox.execute_command(["/bin/cat", str(outside)], timeout=20.0)

    assert allowed.exit_code == 0, allowed.stderr
    assert "visible" in allowed.stdout
    assert refused.exit_code != 0
    assert "private" not in refused.stdout


@pytest.mark.macos
@pytest.mark.skipif(sys.platform != "darwin", reason="seatbelt is macOS-only")
def test_the_kernel_refuses_a_write_outside_the_workdir(tmp_path):
    workdir = tmp_path / "work"
    workdir.mkdir()
    target = tmp_path / "escaped.txt"

    sandbox = SecureSandbox(
        security_level=SecurityLevel.CONFINED,
        workdir=workdir,
        allowed_paths=[workdir],
        read_paths=[workdir],
    )
    result = sandbox.execute_command(
        ["/usr/bin/tee", str(target)], timeout=20.0, input_data="escaped"
    )
    assert not target.exists(), "the sandbox let a write land outside the workdir"
    assert result.exit_code != 0
