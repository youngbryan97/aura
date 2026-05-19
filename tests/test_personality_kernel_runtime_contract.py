from __future__ import annotations

import pytest

from core.brain.personality_kernel import PersonalityKernel
from core.runtime.errors import get_degradation_tracker


class KeyParent:
    def mkdir(self, parents: bool, exist_ok: bool) -> None:
        assert parents is True
        assert exist_ok is True


class FailingKeyFile:
    parent = KeyParent()

    def __init__(self) -> None:
        self.write_attempted = False

    def exists(self) -> bool:
        return False

    def write_bytes(self, _content: bytes) -> None:
        self.write_attempted = True
        raise OSError("identity volume locked")


class LockdownRecorder:
    def __init__(self) -> None:
        self.reason = ""

    def __call__(self, reason: str) -> None:
        self.reason = reason
        raise SystemExit(reason)


def test_personality_kernel_fails_closed_when_identity_key_cannot_persist():
    tracker = get_degradation_tracker()
    tracker.reset()
    kernel = PersonalityKernel.__new__(PersonalityKernel)
    key_file = FailingKeyFile()
    lockdown = LockdownRecorder()

    kernel.key_file = key_file
    kernel._execute_emergency_lockdown = lockdown

    with pytest.raises(SystemExit) as raised:
        kernel._load_or_generate_key()

    assert key_file.write_attempted is True
    assert "IDENTITY_KEY_WRITE_FAILED" in lockdown.reason
    assert "IDENTITY_KEY_WRITE_FAILED" in str(raised.value)
    recent = tracker.recent(subsystem="personality_kernel", limit=1)
    assert recent
    assert recent[0].severity == "critical"
    assert "identity key could not be persisted" in recent[0].action
