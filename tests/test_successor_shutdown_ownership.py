import os
from types import SimpleNamespace

import pytest

from core.runtime.runtime_hygiene import RuntimeHygieneManager


def test_relaunch_executes_python_not_the_script(monkeypatch):
    from core.runtime import runtime_relaunch, runtime_hygiene, subprocess_gateway
    child = Child(99101)
    manager = RuntimeHygieneManager()
    commands = []

    def spawn(command, **kwargs):
        commands.append(command)
        manager.register_process_handle(child)
        return child

    monkeypatch.setattr(runtime_relaunch, "_why_this_process_must_not_replace_itself", lambda argv: "")
    monkeypatch.setattr(subprocess_gateway, "get_subprocess_gateway", lambda: SimpleNamespace(spawn=spawn))
    monkeypatch.setattr(runtime_hygiene, "get_runtime_hygiene", lambda: manager)
    receipt = runtime_relaunch.schedule_relaunch(argv=["aura_main.py", "--desktop"], executable="/python")
    assert receipt["scheduled"]
    command = commands[0]
    assert command[command.index("--") + 1:] == ["/python", "aura_main.py", "--desktop"]


class Child:
    def __init__(self, pid):
        self.pid = pid
        self.returncode = None
        self.terminated = False

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = 0

    def wait(self, timeout=None):
        return self.returncode


@pytest.mark.asyncio
async def test_cleanup_preserves_only_the_exact_handed_off_successor():
    manager = RuntimeHygieneManager()
    successor, ordinary = Child(99101), Child(99102)
    for child in (successor, ordinary):
        manager.register_process_handle(child)
    manager.handoff_successor(successor, predecessor_pid=os.getpid())
    await manager._cleanup_child_processes()
    assert not successor.terminated
    assert ordinary.terminated
    assert manager.process_handle_is_registered(successor)


def test_handoff_rejects_unregistered_or_foreign_handles():
    manager = RuntimeHygieneManager()
    child = Child(99101)
    with pytest.raises(ValueError, match="not_registered"):
        manager.handoff_successor(child, predecessor_pid=os.getpid())
    manager.register_process_handle(child)
    with pytest.raises(ValueError, match="current_process"):
        manager.handoff_successor(child, predecessor_pid=os.getpid() + 1)
    with pytest.raises(ValueError, match="not_registered"):
        manager.handoff_successor(SimpleNamespace(pid=child.pid), predecessor_pid=os.getpid())
