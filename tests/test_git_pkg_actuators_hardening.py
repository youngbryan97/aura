"""CP126 hardening contracts for core/actuators/git_pkg_actuators.py.

Git and pip actuators shell out, so the rails matter: workspace-confined cwd,
option-injection guards (`--`), clone URL scheme/host validation, a pre-op HEAD
guard, bounded + credential-redacted output, and pinned, flag-proof,
non-interactive package installs. A fake subprocess gateway is used — nothing
real is cloned or installed.
"""
from __future__ import annotations

import pytest

import core.actuators.git_pkg_actuators as gp
from core.actuators.git_pkg_actuators import (
    GitActuator,
    PackageInstallActuator,
    _validate_clone_url,
    _validate_git_cwd,
    _validate_ref,
)


@pytest.fixture(autouse=True)
def _registry_authorization():
    """Simulate the ActuatorRegistry's post-AuthorityGateway scope.

    `_aura_authorized` alone is no longer authorization (CP126): the actuator
    verifies a live registry authorization context, so tests standing in for
    the registry must establish one.
    """
    from core.actuators.authority import actuator_authorization

    with actuator_authorization("*"):
        yield



class _FakeResult:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class _FakeGateway:
    def __init__(self):
        self.runs: list[list[str]] = []
        self.result = _FakeResult(0, "ok", "")

    def run(self, cmd, **kwargs):
        self.runs.append(list(cmd))
        # Matched on the subcommand: the options between `git` and it are
        # read settings, and pinning a slice made this silently stop
        # matching when `-c core.fsmonitor=false` was added.
        if cmd[0] == "git" and cmd[-2:] == ["rev-parse", "HEAD"]:
            return _FakeResult(0, "deadbeefcafe1234\n", "")
        return self.result


@pytest.fixture
def gw(monkeypatch, tmp_path):
    monkeypatch.setenv("AURA_GIT_WORKSPACE", str(tmp_path))
    fake = _FakeGateway()
    monkeypatch.setattr(gp, "get_subprocess_gateway", lambda: fake)
    return fake


# ── 2f0ecfec: cwd confinement ──────────────────────────────────────────────


def test_git_cwd_escape_is_refused(tmp_path):
    root = str(tmp_path)
    ok, _ = _validate_git_cwd(str(tmp_path), root)
    assert ok
    bad, err = _validate_git_cwd("/etc", root)
    assert bad is None and "escapes" in err


# ── 44f6e2ad: option injection guards ──────────────────────────────────────


def test_ref_rejects_option_and_traversal():
    bad, err = _validate_ref("--upload-pack=evil", kind="checkout target")
    assert bad is None and "start with" in err
    bad2, _ = _validate_ref("../../etc", kind="checkout target")
    assert bad2 is None


def test_checkout_uses_end_of_options_separator(gw):
    act = GitActuator()
    res = act.execute({"action": "checkout", "branch_or_commit": "main", "allow_mutation": True, "_aura_authorized": True})
    assert res.success is True
    checkout_run = [r for r in gw.runs if r[:2] == ["git", "checkout"]][0]
    assert "--" in checkout_run and checkout_run[-1] == "main"


def test_checkout_option_injection_refused(gw):
    act = GitActuator()
    res = act.execute({"action": "checkout", "branch_or_commit": "--track=evil", "allow_mutation": True, "_aura_authorized": True})
    assert res.success is False


# ── ae34ad76: missing checkout target is a governed failure, not KeyError ──


def test_checkout_without_target_is_governed_failure(gw):
    act = GitActuator()
    res = act.execute({"action": "checkout", "allow_mutation": True, "_aura_authorized": True})
    assert res.success is False and "Invalid git action" in res.message


# ── 512c86b8: clone URL validation ─────────────────────────────────────────


@pytest.mark.parametrize("url", ["file:///etc/passwd", "http://localhost/x", "https://127.0.0.1/x", "-oProxyCommand=evil"])
def test_clone_url_rejects_dangerous(url):
    bad, _ = _validate_clone_url(url)
    assert bad is None


def test_clone_url_accepts_https_and_scp():
    ok, _ = _validate_clone_url("https://github.com/org/repo.git")
    assert ok
    ok2, _ = _validate_clone_url("git@github.com:org/repo.git")
    assert ok2


def test_clone_refuses_bad_url_via_execute(gw):
    act = GitActuator()
    res = act.execute({"action": "clone", "url": "file:///etc", "allow_external_clone": True, "_aura_authorized": True})
    assert res.success is False and "scheme" in res.message


# ── e7b2fdc5: pre-op HEAD + expected_head guard ────────────────────────────


def test_commit_records_pre_op_head(gw):
    act = GitActuator()
    res = act.execute({"action": "commit", "message": "msg", "allow_mutation": True, "_aura_authorized": True})
    assert res.updates.get("pre_op_head", "").startswith("deadbeef")


def test_expected_head_mismatch_refuses(gw):
    act = GitActuator()
    res = act.execute({
        "action": "commit", "message": "m", "allow_mutation": True,
        "expected_head": "0000", "_aura_authorized": True,
    })
    assert res.success is False and "expected HEAD" in res.message


# ── cfd1f0ad + d1be0d7e: bounded, credential-redacted output ────────────────


def test_output_is_bounded_and_redacted(gw):
    gw.result = _FakeResult(0, "https://user:secretpw@host/x " + ("A" * 200000), "")
    act = GitActuator()
    res = act.execute({"action": "status", "_aura_authorized": True})
    out = res.updates["stdout"]
    assert "secretpw" not in out
    assert len(out) < 200000


# ── authority ──────────────────────────────────────────────────────────────


def test_git_requires_authorization():
    assert GitActuator().execute({"action": "status"}).success is False


def test_package_requires_authorization():
    assert PackageInstallActuator().execute({"package_name": "numpy==1.0", "allow_install": True}).success is False


# ── 3a6ae250: package spec rejects flag-like values ────────────────────────


@pytest.mark.parametrize("pkg", ["--upgrade", "-rrequirements.txt", "numpy; rm -rf /", "pkg --index-url http://x"])
def test_package_spec_rejects_flags_and_injection(pkg):
    act = PackageInstallActuator()
    assert act.validate_params({"package_name": pkg, "allow_install": True}) is False


# ── cc1d78b7: unpinned installs refused by default ─────────────────────────


def test_unpinned_install_refused_by_default(gw):
    act = PackageInstallActuator()
    res = act.execute({"package_name": "numpy", "allow_install": True, "_aura_authorized": True})
    assert res.success is False and "pinned" in res.message


def test_pinned_install_uses_end_of_options_and_noninteractive(gw):
    act = PackageInstallActuator()
    # CP126 3a5c4a39: installing into the RUNNING interpreter now requires an
    # explicit acknowledgement, because it mutates Aura's own environment with
    # no rollback. The option-injection property under test is unchanged.
    res = act.execute({
        "package_name": "numpy==1.26.0", "allow_install": True,
        "_aura_authorized": True, "allow_mutating_running_env": True,
    })
    assert res.success is True
    run = [cmd for cmd in gw.runs if "install" in cmd][-1]
    assert "--" in run and run[-1] == "numpy==1.26.0"
    assert "--no-input" in run


def test_installing_into_the_running_interpreter_needs_acknowledgement(gw):
    """CP126 3a5c4a39: a bad resolution degrades Aura until a restart."""
    act = PackageInstallActuator()
    res = act.execute({
        "package_name": "numpy==1.26.0", "allow_install": True, "_aura_authorized": True,
    })
    assert res.success is False
    assert "RUNNING interpreter" in res.message
