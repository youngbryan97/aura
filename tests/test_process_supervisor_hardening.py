"""CP126 hardening contracts for core/actuators/process_supervisor.py.

The supervisor spawns background OS processes, so its safety rails matter:
executable allowlist, workspace confinement, secret-scrubbed environment,
process-group kill (no orphans), a lock-synchronized registry, bounded log
previews, and spawn/exit/kill receipts. Tests use a fake subprocess gateway —
no real processes are launched.
"""
from __future__ import annotations

import os
import signal

import pytest

import core.actuators.process_supervisor as ps
from core.actuators.process_supervisor import (
    ProcessSupervisorActuator,
    _merge_caller_env,
    _redact_command,
    _scrub_base_env,
    _tail_preview,
    _validate_command,
    _validate_cwd,
)


@pytest.fixture(autouse=True)
def _registry_authorization():
    """Simulate the ActuatorRegistry's post-AuthorityGateway scope.

    `_aura_authorized` alone is no longer authorization (CP126): the actuator
    verifies a live registry authorization context, so tests standing in for
    the registry must establish one.
    """
    from core.actuators.authority import actuator_authorization

    with actuator_authorization("process_supervisor"):
        yield



class _FakeProc:
    def __init__(self, pid: int = 4321):
        self.pid = pid
        self._exit = None
        self._aura_gateway_streams = ()
        self.signals: list[int] = []

    def poll(self):
        return self._exit

    def send_signal(self, sig):
        self.signals.append(sig)
        self._exit = -sig

    def wait(self, timeout=None):
        return self._exit


class _FakeGateway:
    def __init__(self):
        self.spawns: list[dict] = []
        self.proc = _FakeProc()

    def spawn(self, argv, **kwargs):
        self.spawns.append({"argv": argv, **kwargs})
        return self.proc


@pytest.fixture
def actuator(tmp_path, monkeypatch):
    monkeypatch.setenv("AURA_PROCESS_WORKSPACE", str(tmp_path))
    gw = _FakeGateway()
    monkeypatch.setattr(ps, "get_subprocess_gateway", lambda: gw)
    act = ProcessSupervisorActuator(logs_dir=str(tmp_path / "logs"))
    act._gateway = gw  # type: ignore[attr-defined]
    return act


def _spawn(act, command="python -c 'print(1)'", **extra):
    params = {"action": "spawn", "allow_spawn": True, "command": command, "_aura_authorized": True}
    params.update(extra)
    return act.execute(params)


# ── 12f0bb7d: executable allowlist ─────────────────────────────────────────


def test_command_allowlist_accepts_python_rejects_arbitrary():
    argv, err = _validate_command("python -c 'print(1)'")
    assert argv == ["python", "-c", "print(1)"], err
    bad, err2 = _validate_command("rm -rf /")
    assert bad is None and "allowlist" in err2


def test_spawn_refuses_non_allowlisted_command(actuator):
    # Refused — validate_params rejects it up front (defense in depth), and the
    # gateway is never reached.
    res = _spawn(actuator, command="/bin/rm -rf /tmp/x")
    assert res.success is False
    assert actuator._gateway.spawns == []


def test_command_with_control_characters_is_refused():
    bad, err = _validate_command(["python", "-c", "print(1)\x00"])
    assert bad is None and "control" in err


# ── 24ec128e: workspace confinement ────────────────────────────────────────


def test_cwd_outside_workspace_is_refused(tmp_path):
    root = str(tmp_path)
    ok, _ = _validate_cwd(str(tmp_path), root)
    assert ok == os.path.realpath(root)
    bad, err = _validate_cwd("/etc", root)
    assert bad is None and "escapes" in err


def test_spawn_refuses_cwd_escape(actuator):
    res = _spawn(actuator, cwd="/etc")
    assert res.success is False and "escapes" in res.message


# ── eb4bd20a + faeb289b: environment scrubbing & validation ────────────────


def test_scrub_base_env_drops_secrets(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret")
    monkeypatch.setenv("MY_AUTH_TOKEN", "t")
    monkeypatch.setenv("HARMLESS_VAR", "ok")
    env = _scrub_base_env()
    assert "OPENAI_API_KEY" not in env
    assert "MY_AUTH_TOKEN" not in env
    assert env.get("HARMLESS_VAR") == "ok"


def test_merge_caller_env_rejects_dangerous_and_non_string():
    base = {"PATH": "/usr/bin"}
    bad, err = _merge_caller_env(dict(base), {"LD_PRELOAD": "/evil.so"})
    assert bad is None and "not permitted" in err
    bad2, err2 = _merge_caller_env(dict(base), {"X": 123})
    assert bad2 is None and "string" in err2
    ok, _ = _merge_caller_env(dict(base), {"SAFE": "1"})
    assert ok["SAFE"] == "1"


def test_spawn_does_not_inherit_secret_env(actuator, monkeypatch):
    monkeypatch.setenv("SECRET_MODEL_KEY", "leak-me")
    _spawn(actuator)
    spawned_env = actuator._gateway.spawns[0]["env"]
    assert "SECRET_MODEL_KEY" not in spawned_env


# ── 9dba5760: log directory anchored, not cwd-relative ─────────────────────


def test_logs_dir_anchors_under_aura_log_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("AURA_LOG_DIR", str(tmp_path))
    act = ProcessSupervisorActuator(logs_dir="processes")
    assert act.logs_dir.startswith(str(tmp_path))


# ── authority contract (unchanged) ─────────────────────────────────────────


def test_execute_requires_authorization(actuator):
    assert actuator.execute({"action": "list"}).success is False


def test_spawn_requires_allow_spawn(actuator):
    res = actuator.execute({"action": "spawn", "command": "python x.py", "_aura_authorized": True})
    assert res.success is False


# ── b4c30a87: concurrency budget ───────────────────────────────────────────


def test_spawn_refuses_beyond_max_jobs(actuator, monkeypatch):
    monkeypatch.setattr(ps, "_MAX_JOBS", 2)
    # The fake gateway returns the SAME proc; give each spawn a distinct running proc.
    procs = [_FakeProc(pid=i) for i in range(5)]
    calls = {"n": 0}

    def _spawn_proc(argv, **kwargs):
        p = procs[calls["n"]]
        calls["n"] += 1
        return p

    monkeypatch.setattr(actuator._gateway, "spawn", _spawn_proc)
    assert _spawn(actuator).success is True
    assert _spawn(actuator).success is True
    third = _spawn(actuator)
    assert third.success is False and "max concurrent" in third.message


# ── 2d4588b7 + e3e1d6be: kill hits the group, no KeyError after success ────


def test_kill_signals_process_group_without_keyerror(actuator, monkeypatch):
    killpg_calls: list[tuple[int, int]] = []
    proc = actuator._gateway.proc

    monkeypatch.setattr(os, "getpgid", lambda pid: pid)

    def _fake_killpg(pgid, sig):
        killpg_calls.append((pgid, sig))
        proc._exit = 0  # the group died

    monkeypatch.setattr(os, "killpg", _fake_killpg)

    spawn_res = _spawn(actuator)
    pid = spawn_res.updates["process_id"]
    res = actuator.execute({"action": "kill", "process_id": pid, "_aura_authorized": True})
    assert res.success is True
    assert killpg_calls and killpg_calls[0][1] == signal.SIGTERM


def test_kill_missing_process_is_reported(actuator):
    res = actuator.execute({"action": "kill", "process_id": "nope", "_aura_authorized": True})
    assert res.success is False and "not found" in res.message


# ── b5de2121: bounded log preview ──────────────────────────────────────────


def test_tail_preview_is_bounded(tmp_path):
    p = tmp_path / "big.log"
    p.write_text("\n".join(f"line{i}" for i in range(10_000)))
    preview = _tail_preview(str(p))
    assert preview.count("\n") <= ps._PREVIEW_LINES
    assert "line9999" in preview  # tail, not head


# ── 4fa30720: command redaction in echoed output ───────────────────────────


def test_redact_command_hides_secret_assignments():
    red = _redact_command(["python", "run.py", "API_KEY=sk-123"])
    assert "sk-123" not in " ".join(red)
    assert any("redacted" in part for part in red)


# ── 998948f0: governor is leased once, not on every spawn ──────────────────


def test_governor_leased_once(actuator, monkeypatch):
    applied: list[int] = []

    class _Gov:
        def apply_volition_profile(self, level):
            applied.append(level)

    # The supervisor reaches the substrate governor through its accessor; nothing registers it as "governor".
    monkeypatch.setattr(ps, "_the_governor", lambda: _Gov())

    procs = [_FakeProc(pid=i) for i in range(3)]
    calls = {"n": 0}
    monkeypatch.setattr(actuator._gateway, "spawn", lambda a, **k: procs[calls.__setitem__("n", calls["n"] + 1) or (calls["n"] - 1)])

    _spawn(actuator)
    _spawn(actuator)
    # Focus profile (3) applied only once across two spawns while jobs run.
    assert applied.count(3) == 1


# ── e9ff4d9b: close() drains live children ─────────────────────────────────


def test_close_terminates_live_children(actuator, monkeypatch):
    monkeypatch.setattr(os, "getpgid", lambda pid: pid)
    killed: list[int] = []
    proc = actuator._gateway.proc
    monkeypatch.setattr(os, "killpg", lambda pgid, sig: (killed.append(sig), setattr(proc, "_exit", 0)))

    _spawn(actuator)
    actuator.close()
    assert killed  # a terminate signal was sent
    assert actuator._processes == {}
