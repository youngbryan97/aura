import os
import signal

import aura_cleanup


def test_get_aura_pids_collects_patterns_ports_and_locks(monkeypatch):
    def fake_run_capture(args, timeout=aura_cleanup.SUBPROCESS_TIMEOUT_S):
        joined = " ".join(args)
        if "pgrep -f aura_main.py" in joined:
            return ["101", "102"]
        if "pgrep -f interface/gui_actor.py" in joined:
            return ["103"]
        if "lsof -nP -iTCP:8000 -t" in joined:
            return ["201"]
        if "lsof -t /tmp/aura_vram.lock" in joined:
            return ["301"]
        return []

    monkeypatch.setattr(aura_cleanup, "_run_capture", fake_run_capture)
    monkeypatch.setattr(
        aura_cleanup.os.path,
        "exists",
        lambda path: path == "/tmp/aura_vram.lock",
    )

    pids = aura_cleanup.get_aura_pids()

    assert pids == {101, 102, 103, 201, 301}


def test_cleanup_terminates_then_force_kills_remaining(monkeypatch):
    monkeypatch.setattr(aura_cleanup, "get_aura_pids", lambda: {111, 222, os.getpid()})
    monkeypatch.setattr(aura_cleanup, "_wait_for_exit", lambda pid, timeout: pid == 111)

    removed = []
    sent = []

    monkeypatch.setattr(aura_cleanup.os.path, "exists", lambda path: True)
    monkeypatch.setattr(aura_cleanup.os, "remove", lambda path: removed.append(path))
    monkeypatch.setattr(aura_cleanup.os, "kill", lambda pid, sig: sent.append((pid, sig)))

    aura_cleanup.cleanup()

    assert (111, signal.SIGTERM) in sent
    assert (222, signal.SIGTERM) in sent
    assert (222, signal.SIGKILL) in sent
    assert any(path.endswith("orchestrator.lock") for path in removed)
    assert any(path.endswith("desktop-app-launch.marker") for path in removed)
