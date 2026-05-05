from pathlib import Path


def test_no_raw_environment_effects_outside_terminal_adapter():
    for path in Path("core/environment").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "pexpect.spawn" not in text
        assert "child.send" not in text
        assert "subprocess.run" not in text
