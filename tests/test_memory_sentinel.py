from __future__ import annotations

import json
import os
import resource

import pytest

import aura_main
from core.container import ServiceContainer
from core.runtime.resource_observation import (
    ObservationProvenance,
    ObservationSource,
    ProcessObservation,
    ProcessTableObservation,
)
from core.runtime.resource_stage_guard import (
    ack_path,
    lease_ack_path,
    lease_request_path,
    publish_compute_lease_request,
    publish_ready_marker,
    read_armed_ack,
    read_compute_lease_ack,
)
from tools import memory_sentinel
from tools.memory_sentinel import should_kill_for_memory


@pytest.fixture(autouse=True)
def isolated_container():
    env_keys = (
        "AURA_MLX_MEMORY_LIMIT_GB",
        "AURA_PROCESS_RSS_LIMIT_GB",
    )
    previous_env = {key: os.environ.get(key) for key in env_keys}
    ServiceContainer.clear()
    yield
    ServiceContainer.clear()
    for key, value in previous_env.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def test_memory_sentinel_waits_for_normal_lethal_confirmation():
    assert (
        should_kill_for_memory(
            managed_mb=46_500.0,
            lethal_mb=46_000.0,
            consecutive_over=1,
        )
        is False
    )
    assert (
        should_kill_for_memory(
            managed_mb=46_500.0,
            lethal_mb=46_000.0,
            consecutive_over=2,
        )
        is True
    )


def test_memory_sentinel_kills_large_overshoot_immediately():
    assert (
        should_kill_for_memory(
            managed_mb=54_000.0,
            lethal_mb=46_000.0,
            consecutive_over=1,
        )
        is True
    )


def test_memory_sentinel_samples_only_protected_tree(monkeypatch):
    provenance = ObservationProvenance(
        source=ObservationSource.HOST,
        scenario_id="sentinel-test",
    )
    table = ProcessTableObservation(
        provenance=provenance,
        processes=(
            ProcessObservation(
                provenance=provenance,
                pid=11,
                ppid=1,
                create_time=100.0,
                status="running",
                name="",
                cmdline=(),
                rss_bytes=100 * 1024 * 1024,
            ),
            ProcessObservation(
                provenance=provenance,
                pid=12,
                ppid=11,
                create_time=101.0,
                status="running",
                name="",
                cmdline=(),
                rss_bytes=20 * 1024 * 1024,
            ),
        ),
    )
    monkeypatch.setattr(
        memory_sentinel,
        "phys_footprint_mb",
        lambda pid: {11: 150.0, 12: 25.0}[pid],
    )

    assert memory_sentinel.tree_rss_mb(11, table=table) == (100.0, 20.0, 2, 175.0)


def test_memory_sentinel_ring_appends_and_compacts_periodically(tmp_path):
    ring = tmp_path / "ring.jsonl"
    memory_sentinel._RING_LINE_COUNTS.clear()

    for index in range(130):
        memory_sentinel.write_ring(ring, {"sample": index}, max_lines=60)

    lines = ring.read_text(encoding="utf-8").splitlines()
    assert 60 <= len(lines) < 120
    assert '"sample":129' in lines[-1]


def test_memory_sentinel_rejects_reused_target_pid():
    provenance = ObservationProvenance(
        source=ObservationSource.HOST,
        scenario_id="sentinel-test",
    )
    target = ProcessObservation(
        provenance=provenance,
        pid=11,
        ppid=1,
        create_time=200.0,
        status="running",
        name="",
        cmdline=(),
        rss_bytes=1,
    )

    assert memory_sentinel.target_process_is_current(target, 100.0) is False


def test_memory_sentinel_identity_survives_transient_rss_failure(monkeypatch):
    class Process:
        pid = 11

        def oneshot(self):
            class Context:
                def __enter__(self):
                    return None

                def __exit__(self, *_args):
                    return False

            return Context()

        def ppid(self):
            return 1

        def create_time(self):
            return 100.0

        def status(self):
            return "running"

        def memory_info(self):
            raise OSError("transient rss failure")

    observed = memory_sentinel._OBSERVER._lightweight_process_from_handle(Process())

    assert observed is not None
    assert observed.pid == 11
    assert observed.create_time == 100.0
    assert observed.rss_bytes == 0


def test_memory_sentinel_confirms_identity_after_missing_tree_sample(resource_observer):
    # target_identity_state reads the canonical resource observer, not psutil
    # directly — observe pid 11 as running with the expected create_time.
    resource_observer.configure_processes(
        [
            ProcessObservation(
                provenance=resource_observer.provenance,
                pid=11,
                ppid=os.getpid(),
                create_time=100.0,
                status="running",
                name="target",
                cmdline=("python",),
                rss_bytes=1024,
            )
        ]
    )

    assert memory_sentinel.target_identity_state(11, 100.0) == "current"


def test_memory_sentinel_transitions_from_startup_to_steady_guard(
    tmp_path,
    monkeypatch,
):
    provenance = ObservationProvenance(
        source=ObservationSource.HOST,
        scenario_id="sentinel-stage-test",
    )
    target = ProcessObservation(
        provenance=provenance,
        pid=123,
        ppid=1,
        create_time=100.0,
        status="running",
        name="trainer",
        cmdline=("python", "trainer.py"),
        rss_bytes=100 * 1024 * 1024,
    )
    live = ProcessTableObservation(provenance=provenance, processes=(target,))
    dead = ProcessTableObservation(provenance=provenance, processes=())

    class Observer:
        def __init__(self):
            self.calls = 0

        def process_tree(self, _pid, *, recursive):
            assert recursive is True
            self.calls += 1
            return live if self.calls == 1 else dead

    marker_path = tmp_path / "ready.json"
    _marker, marker_raw = publish_ready_marker(
        marker_path,
        target_pid=123,
        trainer_sha256="d" * 64,
    )
    ring = tmp_path / "ring.jsonl"
    monkeypatch.setattr(memory_sentinel, "_OBSERVER", Observer())
    monkeypatch.setattr(
        memory_sentinel,
        "tree_rss_mb",
        lambda *_args, **_kwargs: (56000.0, 0.0, 1, 56000.0),
    )
    monkeypatch.setattr(memory_sentinel.time, "sleep", lambda _seconds: None)

    assert (
        memory_sentinel.main(
            [
                "--pid",
                "123",
                "--lethal-mb",
                "59392",
                "--startup-lethal-mb",
                "73728",
                "--steady-marker",
                str(marker_path),
                "--ring",
                str(ring),
                "--tombstone-dir",
                str(tmp_path),
            ]
        )
        == 0
    )

    acknowledgement, _raw = read_armed_ack(
        marker_path,
        marker_raw=marker_raw,
        expected_target_pid=123,
        startup_lethal_mb=73728.0,
        steady_lethal_mb=59392.0,
    )
    assert acknowledgement["stage"] == "steady_memory_guard_armed"
    assert ack_path(marker_path).is_file()
    sample = json.loads(ring.read_text().splitlines()[0])
    assert sample["guard_stage"] == "steady"
    assert sample["active_lethal_mb"] == 59392.0
    assert sample["marker_observed"] is True


def test_memory_sentinel_kills_target_on_invalid_stage_handshake(
    tmp_path,
    monkeypatch,
):
    provenance = ObservationProvenance(
        source=ObservationSource.HOST,
        scenario_id="sentinel-invalid-stage-test",
    )
    target = ProcessObservation(
        provenance=provenance,
        pid=123,
        ppid=1,
        create_time=100.0,
        status="running",
        name="trainer",
        cmdline=("python", "trainer.py"),
        rss_bytes=100 * 1024 * 1024,
    )
    table = ProcessTableObservation(provenance=provenance, processes=(target,))

    class Observer:
        def process_tree(self, _pid, *, recursive):
            assert recursive is True
            return table

    marker_path = tmp_path / "ready.json"
    marker_path.write_text("{}\n", encoding="utf-8")
    killed: list[int] = []
    monkeypatch.setattr(memory_sentinel, "_OBSERVER", Observer())
    monkeypatch.setattr(
        memory_sentinel,
        "tree_rss_mb",
        lambda *_args, **_kwargs: (56000.0, 0.0, 1, 56000.0),
    )
    monkeypatch.setattr(
        memory_sentinel,
        "kill_tree",
        lambda pid: killed.append(pid) or [pid],
    )

    assert (
        memory_sentinel.main(
            [
                "--pid",
                "123",
                "--lethal-mb",
                "59392",
                "--startup-lethal-mb",
                "73728",
                "--steady-marker",
                str(marker_path),
                "--ring",
                str(tmp_path / "ring.jsonl"),
                "--tombstone-dir",
                str(tmp_path),
            ]
        )
        == 2
    )
    assert killed == [123]
    tombstone = next(tmp_path.glob("sentinel_tombstone_*.json"))
    assert "invalid steady-stage" in tombstone.read_text(encoding="utf-8")
    assert tombstone.stat().st_mode & 0o777 == 0o400
    assert tombstone.parent.stat().st_mode & 0o777 == 0o700
    payload = json.loads(tombstone.read_text(encoding="ascii"))
    assert tombstone.read_bytes() == (
        json.dumps(
            payload,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        + b"\n"
    )


def test_memory_sentinel_enforces_compute_lease_and_low_water_rearm(
    tmp_path,
    monkeypatch,
):
    provenance = ObservationProvenance(
        source=ObservationSource.HOST,
        scenario_id="sentinel-compute-lease-test",
    )
    target = ProcessObservation(
        provenance=provenance,
        pid=123,
        ppid=1,
        create_time=100.0,
        status="running",
        name="trainer",
        cmdline=("python", "trainer.py"),
        rss_bytes=100 * 1024 * 1024,
    )
    live = ProcessTableObservation(provenance=provenance, processes=(target,))
    dead = ProcessTableObservation(provenance=provenance, processes=())

    class Observer:
        def __init__(self):
            self.calls = 0

        def process_tree(self, _pid, *, recursive):
            assert recursive is True
            self.calls += 1
            return live if self.calls <= 4 else dead

    marker_path = tmp_path / "ready.json"
    _marker, marker_raw = publish_ready_marker(
        marker_path,
        target_pid=123,
        trainer_sha256="e" * 64,
    )
    ring = tmp_path / "ring.jsonl"
    sleeps = {"count": 0}

    def advance_trainer(_seconds):
        sleeps["count"] += 1
        if sleeps["count"] == 1:
            _initial, initial_ack_raw = read_armed_ack(
                marker_path,
                marker_raw=marker_raw,
                expected_target_pid=123,
                startup_lethal_mb=73728.0,
                steady_lethal_mb=59392.0,
            )
            publish_compute_lease_request(
                marker_path,
                marker_raw=marker_raw,
                target_pid=123,
                sequence=1,
                workload="training_step",
                action="acquire",
                predecessor_ack_raw=initial_ack_raw,
            )
        elif sleeps["count"] == 2:
            acquire_path = lease_request_path(
                marker_path,
                sequence=1,
                action="acquire",
            )
            acquire_raw = acquire_path.read_bytes()
            _acquire, acquire_ack_raw = read_compute_lease_ack(
                acquire_path,
                request_raw=acquire_raw,
                expected_target_pid=123,
                sequence=1,
                workload="training_step",
                action="acquire",
                active_lethal_mb=73728.0,
            )
            publish_compute_lease_request(
                marker_path,
                marker_raw=marker_raw,
                target_pid=123,
                sequence=1,
                workload="training_step",
                action="release",
                predecessor_ack_raw=acquire_ack_raw,
            )

    monkeypatch.setattr(memory_sentinel, "_OBSERVER", Observer())
    monkeypatch.setattr(
        memory_sentinel,
        "tree_rss_mb",
        lambda *_args, **_kwargs: (56000.0, 0.0, 1, 56000.0),
    )
    monkeypatch.setattr(memory_sentinel.time, "sleep", advance_trainer)

    assert (
        memory_sentinel.main(
            [
                "--pid",
                "123",
                "--lethal-mb",
                "59392",
                "--startup-lethal-mb",
                "73728",
                "--steady-marker",
                str(marker_path),
                "--interval",
                "0.5",
                "--ring",
                str(ring),
                "--tombstone-dir",
                str(tmp_path),
            ]
        )
        == 0
    )

    release_path = lease_request_path(
        marker_path,
        sequence=1,
        action="release",
    )
    assert lease_ack_path(release_path).is_file()
    stages = [json.loads(line)["guard_stage"] for line in ring.read_text().splitlines()]
    assert stages == ["steady", "compute", "draining", "steady"]


def test_memory_sentinel_tombstones_invalid_compute_acquire(
    tmp_path,
    monkeypatch,
):
    provenance = ObservationProvenance(
        source=ObservationSource.HOST,
        scenario_id="sentinel-invalid-compute-test",
    )
    target = ProcessObservation(
        provenance=provenance,
        pid=123,
        ppid=1,
        create_time=100.0,
        status="running",
        name="trainer",
        cmdline=("python", "trainer.py"),
        rss_bytes=100 * 1024 * 1024,
    )
    table = ProcessTableObservation(provenance=provenance, processes=(target,))

    class Observer:
        def process_tree(self, _pid, *, recursive):
            assert recursive is True
            return table

    marker_path = tmp_path / "ready.json"
    _marker, marker_raw = publish_ready_marker(
        marker_path,
        target_pid=123,
        trainer_sha256="f" * 64,
    )
    sleeps = {"count": 0}

    def write_invalid_acquire(_seconds):
        sleeps["count"] += 1
        if sleeps["count"] == 1:
            lease_request_path(
                marker_path,
                sequence=1,
                action="acquire",
            ).write_text("{}\n", encoding="utf-8")

    killed: list[int] = []
    monkeypatch.setattr(memory_sentinel, "_OBSERVER", Observer())
    monkeypatch.setattr(
        memory_sentinel,
        "tree_rss_mb",
        lambda *_args, **_kwargs: (56000.0, 0.0, 1, 56000.0),
    )
    monkeypatch.setattr(memory_sentinel.time, "sleep", write_invalid_acquire)
    monkeypatch.setattr(
        memory_sentinel,
        "kill_tree",
        lambda pid: killed.append(pid) or [pid],
    )

    assert (
        memory_sentinel.main(
            [
                "--pid",
                "123",
                "--lethal-mb",
                "59392",
                "--startup-lethal-mb",
                "73728",
                "--steady-marker",
                str(marker_path),
                "--interval",
                "0.5",
                "--ring",
                str(tmp_path / "ring.jsonl"),
                "--tombstone-dir",
                str(tmp_path),
            ]
        )
        == 2
    )
    assert killed == [123]
    tombstone = next(tmp_path.glob("sentinel_tombstone_*.json"))
    payload = json.loads(tombstone.read_text(encoding="utf-8"))
    assert payload["reason"] == ("invalid compute-acquire resource-guard handshake")


def test_memory_sentinel_default_ceiling_is_host_safe_on_64gb_node(monkeypatch):
    monkeypatch.delenv("AURA_ALLOW_UNSAFE_MEMORY_LIMITS", raising=False)

    ceiling = aura_main._bounded_memory_ceiling_mb(64 * 1024.0)

    assert ceiling == 57_344.0


def test_memory_sentinel_clamps_excessive_env_override(monkeypatch):
    monkeypatch.delenv("AURA_ALLOW_UNSAFE_MEMORY_LIMITS", raising=False)

    ceiling = aura_main._bounded_memory_ceiling_mb(64 * 1024.0, "120000")

    assert ceiling == 57_344.0


def test_memory_sentinel_allows_explicit_unsafe_override_only_when_opted_in(monkeypatch):
    monkeypatch.setenv("AURA_ALLOW_UNSAFE_MEMORY_LIMITS", "1")

    ceiling = aura_main._bounded_memory_ceiling_mb(64 * 1024.0, "120000")

    assert ceiling == 120000.0


def test_memory_sentinel_malformed_override_falls_back_to_safe_ceiling(monkeypatch):
    monkeypatch.delenv("AURA_ALLOW_UNSAFE_MEMORY_LIMITS", raising=False)

    ceiling = aura_main._bounded_memory_ceiling_mb(64 * 1024.0, "not-a-number")

    assert ceiling == 57_344.0


def test_desktop_boot_memory_protection_registers_armed_external_sentinel(monkeypatch):
    popen_calls: list[list[str]] = []

    class FakePopen:
        pid = os.getpid()

        def __init__(self, args, **kwargs):
            popen_calls.append([str(arg) for arg in args])
            stdout = kwargs.get("stdout")
            if stdout is not None:
                stdout.close()

        def poll(self):
            return None

    monkeypatch.setenv("AURA_MEMORY_SENTINEL", "1")
    monkeypatch.setenv("AURA_MEMWATCH_LETHAL_MB", "46080")
    monkeypatch.setenv("AURA_MEMORY_SENTINEL_INTERVAL_S", "0.5")
    monkeypatch.setattr(
        resource, "getrlimit", lambda _kind: (resource.RLIM_INFINITY, resource.RLIM_INFINITY)
    )
    monkeypatch.setattr(resource, "setrlimit", lambda _kind, _limits: None)
    monkeypatch.setattr(aura_main.subprocess, "Popen", FakePopen)

    aura_main._install_systemwide_memory_protection()

    sentinel = ServiceContainer.get("external_memory_sentinel")
    assert sentinel.is_armed() is True
    assert sentinel.get_status()["armed"] is True
    assert sentinel.get_status()["lethal_mb"] == pytest.approx(46_080.0)
    assert popen_calls
    assert "memory_sentinel.py" in " ".join(popen_calls[0])
    assert "--pid" in popen_calls[0]
    assert str(os.getpid()) in popen_calls[0]
    assert "--lethal-mb" in popen_calls[0]
    assert str(46_080.0) in popen_calls[0]


def test_disabled_external_memory_sentinel_is_never_reported_armed(monkeypatch):
    monkeypatch.setenv("AURA_MEMORY_SENTINEL", "0")
    monkeypatch.setattr(
        resource, "getrlimit", lambda _kind: (resource.RLIM_INFINITY, resource.RLIM_INFINITY)
    )
    monkeypatch.setattr(resource, "setrlimit", lambda _kind, _limits: None)

    aura_main._install_systemwide_memory_protection()

    sentinel = ServiceContainer.get("external_memory_sentinel")
    assert sentinel.is_armed() is False
    assert sentinel.get_status()["armed"] is False


class _DeadProc:
    """A sentinel process that has already exited."""

    pid = 4242

    def poll(self):
        return 1


class _LiveProc:
    pid = 4343

    def poll(self):
        return None


def _pid_exists_stub(monkeypatch, value=True):
    import psutil

    monkeypatch.setattr(psutil, "pid_exists", lambda _pid: value)


def _observe_running_pid(resource_observer, pid):
    """is_armed() consults the canonical observer census (71b5598f); a
    respawned sentinel is armed only if its pid is observed running."""
    from core.runtime.resource_observation import ProcessObservation

    resource_observer.configure_processes(
        [
            ProcessObservation(
                provenance=resource_observer.provenance,
                pid=pid,
                ppid=os.getpid(),
                create_time=1_700_000_000.0,
                status="running",
                name="memory-sentinel",
                cmdline=("python", "-m", "tools.memory_sentinel"),
                rss_bytes=1024,
            )
        ]
    )


class TestSentinelRearm:
    """A dead contract-CRITICAL guardian must come back, boundedly.

    Live incident: the sentinel died 85 seconds after arming and the desktop
    ran CRITICAL — and unprotected — for two hours with nothing respawning it.
    """

    def _status(self, monkeypatch, *, spawn_results=None):
        _pid_exists_stub(monkeypatch)
        spawned = []
        results = list(spawn_results or [])

        def spawner():
            if results:
                item = results.pop(0)
                if isinstance(item, Exception):
                    raise item
                spawned.append(item)
                return item
            proc = _LiveProc()
            spawned.append(proc)
            return proc

        status = aura_main._ExternalMemorySentinelStatus(
            _DeadProc(), lethal_mb=1000.0, interval_s=1.0, spawner=spawner
        )
        return status, spawned

    def test_rearm_respawns_dead_sentinel(self, monkeypatch, resource_observer):
        status, spawned = self._status(monkeypatch)
        _observe_running_pid(resource_observer, _LiveProc.pid)
        assert status.is_armed() is False
        assert status.rearm() is True
        assert status.is_armed() is True
        assert len(spawned) == 1
        assert status.pid == _LiveProc.pid
        assert status.get_status()["rearms_last_hour"] == 1

    def test_rearm_is_noop_while_armed(self, monkeypatch, resource_observer):
        status, spawned = self._status(monkeypatch)
        _observe_running_pid(resource_observer, _LiveProc.pid)
        status.rearm()
        assert status.rearm() is True
        assert len(spawned) == 1, "an armed sentinel must not be respawned"

    def test_rearm_respects_min_interval(self, monkeypatch):
        status, spawned = self._status(monkeypatch)
        status.rearm()
        status.proc = _DeadProc()  # dies again immediately
        status.pid = _DeadProc.pid
        assert status.rearm() is False, "respawn faster than 30s must be refused"
        assert len(spawned) == 1

    def test_rearm_budget_is_bounded_per_hour(self, monkeypatch):
        status, spawned = self._status(monkeypatch)
        base = 1_000_000.0
        clock = {"now": base}
        monkeypatch.setattr(aura_main.time, "time", lambda: clock["now"])
        for i in range(status.REARM_HOURLY_BUDGET):
            clock["now"] = base + i * 60.0
            assert status.rearm() is True
            status.proc = _DeadProc()
        clock["now"] = base + status.REARM_HOURLY_BUDGET * 60.0
        assert status.rearm() is False, "hourly budget must stop a crash-looping sentinel"
        assert len(spawned) == status.REARM_HOURLY_BUDGET
        # After the oldest attempt ages out of the window, respawns resume.
        clock["now"] = base + 3601.0
        assert status.rearm() is True

    def test_rearm_spawn_failure_records_and_returns_false(self, monkeypatch):
        status, spawned = self._status(
            monkeypatch, spawn_results=[OSError("spawn refused"), _LiveProc()]
        )
        assert status.rearm() is False
        assert status.is_armed() is False
        assert len(spawned) == 0

    def test_disabled_sentinel_never_rearms(self, monkeypatch):
        _pid_exists_stub(monkeypatch)
        status = aura_main._ExternalMemorySentinelStatus(None)
        assert status.rearm() is False
        assert status.is_armed() is False


def test_boot_registers_supervised_sentinel(monkeypatch):
    """The install path must hand the status object a working respawner."""
    popen_procs = []

    class FakePopen:
        def __init__(self, args, **kwargs):
            self.pid = os.getpid()
            popen_procs.append(self)
            self._dead = False

        def poll(self):
            return 1 if self._dead else None

    supervisors = []
    monkeypatch.setattr(
        aura_main, "_start_memory_sentinel_supervisor", lambda status: supervisors.append(status)
    )
    monkeypatch.setenv("AURA_MEMORY_SENTINEL", "1")
    monkeypatch.setattr(
        resource, "getrlimit", lambda _kind: (resource.RLIM_INFINITY, resource.RLIM_INFINITY)
    )
    monkeypatch.setattr(resource, "setrlimit", lambda _kind, _limits: None)
    monkeypatch.setattr(aura_main.subprocess, "Popen", FakePopen)

    aura_main._install_systemwide_memory_protection()

    sentinel = ServiceContainer.get("external_memory_sentinel")
    assert supervisors == [sentinel], "supervisor must watch the registered status object"
    assert sentinel.is_armed() is True
    # Kill it; rearm must produce a fresh process through the real spawner.
    popen_procs[0]._dead = True
    assert sentinel.is_armed() is False
    assert sentinel.rearm() is True
    assert len(popen_procs) == 2
    assert sentinel.is_armed() is True
