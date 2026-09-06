import os
from types import SimpleNamespace

import pytest

from core.runtime.runtime_hygiene import RuntimeHygieneManager


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
