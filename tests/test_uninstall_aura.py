"""Uninstall removes what an install put there, to the Trash, keeps her state unless asked, and waits for her to quit.

Q11: installation, update and uninstall validated. There was no uninstall (13 November 2026); tools/uninstall_aura.py is
it, proved here in a sandbox so nothing on the real Mac is touched.
"""
from __future__ import annotations

import json
import socket
import stat
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import uninstall_aura  # noqa: E402


@pytest.fixture
def mac(tmp_path):
    applications = tmp_path / "Applications"
    (applications / "Aura.app" / "Contents" / "MacOS").mkdir(parents=True)
    agents = tmp_path / "home" / "Library" / "LaunchAgents"
    agents.mkdir(parents=True)
    (agents / "com.aura.supervisor.plist").write_text("<plist/>")
    (agents / "com.someone.else.plist").write_text("<plist/>")
    state = tmp_path / "home" / ".aura"
    (state / "data").mkdir(parents=True)
    (state / "data" / "conversations.db").write_text("what she has lived through")
    calls = tmp_path / "launchctl.calls"
    launchctl = tmp_path / "launchctl"
    launchctl.write_text(f"#!/bin/sh\necho \"$@\" >> {calls}\nexit 0\n")
    launchctl.chmod(launchctl.stat().st_mode | stat.S_IEXEC)
    args = ["--applications", str(applications), "--home", str(tmp_path / "home"), "--state-root", str(state),
            "--trash", str(tmp_path / "Trash"), "--launchctl", str(launchctl), "--port", str(_free_port())]
    return {"root": tmp_path, "args": args, "applications": applications, "agents": agents, "state": state,
            "calls": calls}


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def test_a_dry_run_says_what_would_go_and_changes_nothing(mac, capsys):
    assert uninstall_aura.main([*mac["args"], "--dry-run"]) == 0
    said = json.loads(capsys.readouterr().out)
    assert any(path.endswith("Aura.app") for path in said["would_remove"])
    assert any(path.endswith("com.aura.supervisor.plist") for path in said["would_remove"])
    assert (mac["applications"] / "Aura.app").exists() and not (mac["root"] / "Trash").exists()


def test_it_removes_the_app_and_her_launch_agents_to_the_trash_and_keeps_her_state(mac):
    assert uninstall_aura.main(mac["args"]) == 0
    trash = mac["root"] / "Trash"
    assert not (mac["applications"] / "Aura.app").exists() and (trash / "Aura.app").exists()
    assert not (mac["agents"] / "com.aura.supervisor.plist").exists()
    assert (trash / "com.aura.supervisor.plist").exists()
    assert (mac["agents"] / "com.someone.else.plist").exists()
    assert (mac["state"] / "data" / "conversations.db").exists()
    assert "bootout" in mac["calls"].read_text() and "com.aura.supervisor.plist" in mac["calls"].read_text()


def test_her_state_goes_only_when_asked(mac):
    assert uninstall_aura.main([*mac["args"], "--purge-state"]) == 0
    assert not mac["state"].exists()
    assert (mac["root"] / "Trash" / ".aura" / "data" / "conversations.db").read_text() == "what she has lived through"


def test_it_refuses_while_she_is_running(mac):
    with socket.socket() as listening:
        listening.bind(("127.0.0.1", 0))
        listening.listen()
        port = listening.getsockname()[1]
        args = [*mac["args"][:-1], str(port)]
        assert uninstall_aura.main(args) == 2
    assert (mac["applications"] / "Aura.app").exists()
    assert (mac["agents"] / "com.aura.supervisor.plist").exists()


def test_a_second_uninstall_finds_nothing_and_says_so(mac, capsys):
    assert uninstall_aura.main(mac["args"]) == 0
    capsys.readouterr()
    assert uninstall_aura.main(mac["args"]) == 0
    assert json.loads(capsys.readouterr().out)["removed"] == []
