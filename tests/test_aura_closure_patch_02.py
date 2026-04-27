from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_aura_main_installs_task_patch_and_requires_python_312():
    text = (ROOT / "aura_main.py").read_text(encoding="utf-8")
    assert "import core.utils.asyncio_patch" in text
    assert "if sys.version_info < (3, 12):" in text
    assert "Aura requires Python 3.12+" in text


def test_asyncio_patch_is_reentry_guarded():
    text = (ROOT / "core" / "utils" / "asyncio_patch.py").read_text(encoding="utf-8")
    assert "_REENTRY" in text
    assert "tracker.create_task" in text
    assert "__aura_task_patch__" in text


def test_morphogenesis_status_exposes_direct_state_counters():
    text = (ROOT / "core" / "morphogenesis" / "registry.py").read_text(encoding="utf-8")
    assert '"quarantined": by_state.get("quarantined", 0)' in text
    assert '"dead": by_state.get("dead", 0)' in text


def test_morphogenesis_organ_episode_uses_task_tracker():
    text = (ROOT / "core" / "morphogenesis" / "runtime.py").read_text(encoding="utf-8")
    assert "get_task_tracker().create_task(record_organ_formation_episode" in text
    assert "asyncio.ensure_future(record_organ_formation_episode" not in text


def test_terminal_monitor_blacklist_write_is_atomic_when_available():
    text = (ROOT / "core" / "terminal_monitor.py").read_text(encoding="utf-8")
    assert "atomic_write_json" in text
    assert "terminal_error_blacklist" in text


def test_docker_python_matches_operator_runtime_contract():
    text = (ROOT / "docker" / "Dockerfile").read_text(encoding="utf-8")
    assert "FROM python:3.12-slim" in text
