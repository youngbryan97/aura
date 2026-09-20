"""Index enumeration neither needs fsmonitor nor splits valid path names."""

import subprocess
from types import SimpleNamespace

import pytest

from tools import lint_ai_writing


def test_index_inventory_bypasses_monitor_and_preserves_paths(monkeypatch):
    def run(command, **options):
        assert command == ["git", "-c", "core.fsmonitor=false", "ls-files", "-z", "--", "*.py"]
        assert options["check"] and options["timeout"] == 60
        return SimpleNamespace(stdout="core/one.py\0core/two words.py\0core/new\nline.py\0")

    monkeypatch.setattr(lint_ai_writing.subprocess, "run", run)
    assert lint_ai_writing._tracked("*.py") == ["core/one.py", "core/two words.py", "core/new\nline.py"]


def test_failed_enumeration_cannot_become_an_empty_passing_gate(monkeypatch):
    def run(command, **options):
        raise subprocess.TimeoutExpired(command, options["timeout"])

    monkeypatch.setattr(lint_ai_writing.subprocess, "run", run)
    with pytest.raises(subprocess.TimeoutExpired):
        lint_ai_writing._tracked("*.md")
