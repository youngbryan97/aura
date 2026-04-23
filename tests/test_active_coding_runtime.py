from pathlib import Path

import core.skills.active_coding as active_coding


def test_active_coding_sandbox_uses_gitignored_runtime_workspace(monkeypatch, tmp_path):
    monkeypatch.setenv("AURA_ROOT", str(tmp_path))
    monkeypatch.setattr(active_coding, "_sandbox", None)

    sandbox = active_coding.get_sandbox()
    try:
        assert sandbox.work_path == tmp_path / ".aura_runtime" / "active_coding"
        assert sandbox.work_path.parent == tmp_path / ".aura_runtime"
        assert "aura_main" not in str(sandbox.work_path)
    finally:
        sandbox.stop()
        monkeypatch.setattr(active_coding, "_sandbox", None)
