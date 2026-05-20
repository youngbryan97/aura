import signal

from core import reaper
from core.reaper import ReaperManifest
from core.runtime.errors import get_degradation_tracker


def test_reaper_keeps_pid_manifest_entry_when_kill_fails(monkeypatch, tmp_path):
    get_degradation_tracker().reset()
    manifest = ReaperManifest(tmp_path / "reaper.json")
    manifest.register_pid(4242)

    def _kill(pid, sig):
        assert pid == 4242
        assert sig == signal.SIGTERM
        raise RuntimeError("permission model changed")

    monkeypatch.setattr(reaper.os, "kill", _kill)

    summary = reaper._execute_cleanup(manifest)

    assert summary["failed_pids"] == [4242]
    assert manifest._data["child_pids"] == [4242]
    assert manifest.path.exists()
    last = get_degradation_tracker().recent(subsystem="reaper")[-1]
    assert last.action == "kept PID in reaper manifest for a future cleanup attempt"
    get_degradation_tracker().reset()


def test_reaper_keeps_shared_memory_manifest_entry_when_unlink_fails(monkeypatch, tmp_path):
    get_degradation_tracker().reset()
    manifest = ReaperManifest(tmp_path / "reaper.json")
    manifest.register_shm("aura-shm")

    class _BrokenSharedMemory:
        def __init__(self, name):
            assert name == "aura-shm"
            raise RuntimeError("shm namespace unavailable")

    monkeypatch.setattr(reaper.shm_lib, "SharedMemory", _BrokenSharedMemory)

    summary = reaper._execute_cleanup(manifest)

    assert summary["failed_shm"] == ["aura-shm"]
    assert manifest._data["shm_names"] == ["aura-shm"]
    assert manifest.path.exists()
    last = get_degradation_tracker().recent(subsystem="reaper")[-1]
    assert last.action == "kept shared-memory name in reaper manifest for a future cleanup attempt"
    get_degradation_tracker().reset()


def test_reaper_removes_manifest_only_after_all_resources_are_clean(tmp_path):
    get_degradation_tracker().reset()
    manifest = ReaperManifest(tmp_path / "reaper.json")
    manifest._save()

    summary = reaper._execute_cleanup(manifest)

    assert summary["manifest_removed"] is True
    assert not manifest.path.exists()
    get_degradation_tracker().reset()
