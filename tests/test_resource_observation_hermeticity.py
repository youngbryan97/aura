"""Hermetic resource-policy and test-isolation closure checks."""

from __future__ import annotations

import concurrent.futures
import contextlib
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import pytest

from core.brain.lane_admission import ActiveLane, LaneAdmissionController, QoSClass
from core.resource.resource_governor import ResourceGovernor, ThermalState
from core.runtime import resource_psutil
from core.runtime.model_lane_control import ModelLaneController
from core.runtime.receipts import ReceiptStore
from core.runtime.resource_observation import (
    HostResourceObserver,
    NetworkConnectionObservation,
    ObservationSource,
    ProcessObservation,
    ProcessTableObservation,
    SimulatedResourceObserver,
    assert_live_pressure_observer,
)
from core.runtime.runtime_pressure import UnifiedRuntimePressure
from core.runtime.subprocess_gateway import get_subprocess_gateway

GIB = 1024**3


def test_resource_and_state_leak_guards_are_enforcing_by_default():
    from tests import conftest

    marker = conftest.hermetic_resource_sandbox._fixture_function_marker
    assert marker.autouse is True
    assert os.environ["AURA_TEST_STATE_GUARD"] == "fail"


def _process(
    observer: SimulatedResourceObserver,
    *,
    pid: int,
    ppid: int = 1,
    ancestors: tuple[int, ...] = (),
    rss_bytes: int = 1024,
) -> ProcessObservation:
    return ProcessObservation(
        provenance=observer.provenance,
        pid=pid,
        ppid=ppid,
        create_time=float(pid),
        status="running",
        name=f"process-{pid}",
        cmdline=(sys.executable, "worker.py", str(pid)),
        rss_bytes=rss_bytes,
        ancestor_pids=ancestors,
        cwd="aura-hermetic",
    )


def test_simulated_snapshot_attributes_every_resource_dimension(resource_observer):
    resource_observer.configure_memory(
        total_bytes=64 * GIB,
        available_bytes=20 * GIB,
        percent=68.75,
        process_rss_bytes=2 * GIB,
        process_tree_rss_bytes=5 * GIB,
        swap_total_bytes=4 * GIB,
        swap_used_bytes=9 * GIB,
    )
    resource_observer.configure_disk(total_bytes=500 * GIB, free_bytes=125 * GIB)
    resource_observer.configure_thermal(2)
    resource_observer.configure_accelerator(
        active_bytes=6 * GIB,
        cache_bytes=2 * GIB,
        peak_bytes=9 * GIB,
    )
    resource_observer.configure_compute(cpu_percent=73.0, cpu_count=12, load_1m=4.5)
    resource_observer.configure_power(
        battery_percent=37.0,
        plugged=False,
        seconds_left=1800,
    )
    resource_observer.configure_processes(
        [_process(resource_observer, pid=100, rss_bytes=3 * GIB)]
    )
    resource_observer.configure_connections(
        [
            NetworkConnectionObservation(
                provenance=resource_observer.provenance,
                pid=100,
                fd=8,
                family="AF_INET",
                socket_type="SOCK_STREAM",
                local_host="127.0.0.1",
                local_port=8123,
                status="LISTEN",
            )
        ]
    )
    resource_observer.configure_open_files(["aura-hermetic/state.json"])

    snapshot = resource_observer.snapshot(path="/tmp", include_processes=True)
    provenances = (
        snapshot.provenance,
        snapshot.memory.provenance,
        snapshot.disk.provenance,
        snapshot.thermal.provenance,
        snapshot.accelerator.provenance,
        snapshot.compute.provenance,
        snapshot.power.provenance,
        snapshot.processes[0].provenance,
        resource_observer.connection_table().provenance,
        resource_observer.open_file_table().provenance,
    )

    assert all(item.source is ObservationSource.SIMULATED for item in provenances)
    assert all(item.scenario_id == resource_observer.provenance.scenario_id for item in provenances)
    assert snapshot.memory.swap_used_bytes == 4 * GIB
    assert snapshot.memory.swap_free_bytes == 0
    assert snapshot.memory.swap_percent == 100.0
    assert snapshot.disk.percent == 75.0
    assert snapshot.thermal.level == 2
    assert snapshot.accelerator.active_bytes == 6 * GIB
    assert snapshot.compute.cpu_percent == 73.0
    assert snapshot.power.battery_percent == 37.0
    assert snapshot.power.plugged is False
    assert resource_observer.process_table().available is True
    assert resource_observer.connection_table().connections[0].local_port == 8123
    assert resource_observer.open_file_table().paths == ("aura-hermetic/state.json",)


def test_live_pressure_evidence_rejects_simulation_and_plain_host(resource_observer):
    with pytest.raises(ValueError, match="HostResourceObserver"):
        assert_live_pressure_observer(resource_observer)

    host = HostResourceObserver(
        source=ObservationSource.HOST,
        scenario_id="ordinary-host-observation",
    )
    with pytest.raises(ValueError, match="source=live_pressure"):
        assert_live_pressure_observer(host)

    live = HostResourceObserver(
        source=ObservationSource.LIVE_PRESSURE,
        scenario_id="bounded-live-pressure-proof",
    )
    assert_live_pressure_observer(live)
    assert live.provenance.qualifies_as_live_pressure is True


def test_host_observer_process_tree_uses_lightweight_targeted_probe(monkeypatch):
    from core.runtime import resource_observation

    class _Memory:
        rss = 32 * 1024 * 1024

    class _Handle:
        def __init__(self, pid, ppid):
            self.pid = pid
            self._ppid = ppid

        def oneshot(self):
            return contextlib.nullcontext()

        def ppid(self):
            return self._ppid

        def create_time(self):
            return float(self.pid)

        def status(self):
            return "running"

        def memory_info(self):
            return _Memory()

        def children(self, *, recursive):
            assert recursive is True
            return [_Handle(12, self.pid)]

    monkeypatch.setattr(resource_observation.psutil, "Process", lambda pid: _Handle(pid, 1))
    observer = HostResourceObserver(
        source=ObservationSource.HOST,
        scenario_id="targeted-tree-test",
    )

    table = observer.process_tree(11)

    assert table.available is True
    assert [process.pid for process in table.processes] == [11, 12]
    assert all(process.cmdline == () for process in table.processes)
    assert all(process.name == "" for process in table.processes)


def test_host_process_ids_never_enriches_process_table(monkeypatch):
    from core.runtime import resource_observation

    monkeypatch.setattr(resource_observation.psutil, "pids", lambda: [9, 3, 9, 5])
    monkeypatch.setattr(
        resource_observation.psutil,
        "process_iter",
        lambda: (_ for _ in ()).throw(AssertionError("full process table was scanned")),
    )

    observation = HostResourceObserver().process_ids()

    assert observation.available is True
    assert observation.pids == (3, 5, 9)


def test_host_process_table_skips_native_process_probe_system_error(monkeypatch):
    from core.runtime import resource_observation

    class _ProtectedHandle:
        pid = 41

        def oneshot(self):
            return contextlib.nullcontext()

        def ppid(self):
            return 1

        def create_time(self):
            return 41.0

        def status(self):
            return "running"

        def name(self):
            raise SystemError("sysctl(KERN_PROCARGS2) returned an exception")

    monkeypatch.setattr(
        resource_observation.psutil,
        "process_iter",
        lambda: iter((_ProtectedHandle(),)),
    )

    observation = HostResourceObserver().process_table()

    assert observation.available is True
    assert observation.processes == ()


def test_legacy_pid_census_does_not_call_process_table(monkeypatch, resource_observer):
    resource_observer.configure_processes(
        [
            _process(resource_observer, pid=11),
            _process(resource_observer, pid=17),
        ]
    )
    monkeypatch.setattr(
        resource_observer,
        "process_table",
        lambda: (_ for _ in ()).throw(AssertionError("full process table was scanned")),
    )

    assert resource_psutil.pids() == [11, 17]


def test_legacy_resource_facade_never_reaches_sabotaged_host_apis(
    monkeypatch,
    resource_observer,
):
    resource_observer.configure_memory(
        total_bytes=48 * GIB,
        available_bytes=24 * GIB,
        percent=50.0,
    )
    resource_observer.configure_disk(total_bytes=400 * GIB, free_bytes=300 * GIB)
    resource_observer.configure_compute(cpu_percent=12.5, cpu_count=6, load_1m=1.0)
    resource_observer.configure_power(
        battery_percent=42.0,
        plugged=False,
        seconds_left=1200,
    )
    resource_observer.configure_processes(
        [
            ProcessObservation(
                provenance=resource_observer.provenance,
                pid=os.getpid(),
                ppid=os.getppid(),
                create_time=123.0,
                status="running",
                name="hermetic-test-process",
                cmdline=(sys.executable, "-m", "pytest"),
                rss_bytes=321 * 1024**2,
                memory_percent=2.5,
                cpu_percent=7.5,
                cpu_user_seconds=12.0,
                cpu_system_seconds=3.0,
                num_threads=9,
                num_fds=4,
            )
        ]
    )

    def host_access_forbidden(*_args, **_kwargs):
        raise AssertionError("simulated policy reached a host resource API")

    # Armed only around the calls under test. conftest's teardown reads the
    # host after the test body and before a plain monkeypatch is undone, and
    # met this tripwire there.
    with monkeypatch.context() as tripwire:
        tripwire.setattr("core.runtime.resource_observation.psutil.virtual_memory", host_access_forbidden)
        tripwire.setattr("core.runtime.resource_observation.psutil.process_iter", host_access_forbidden)
        tripwire.setattr("core.runtime.resource_observation.shutil.disk_usage", host_access_forbidden)
        tripwire.setattr("core.runtime.resource_observation.os.getloadavg", host_access_forbidden)
        tripwire.setattr("core.runtime.resource_observation.psutil.sensors_battery", host_access_forbidden)
        tripwire.setattr("core.runtime.resource_psutil._psutil.disk_io_counters", host_access_forbidden)
        tripwire.setattr("core.runtime.resource_psutil._psutil.net_io_counters", host_access_forbidden)
        tripwire.setattr("core.runtime.resource_psutil._psutil.net_if_addrs", host_access_forbidden)

        memory = resource_psutil.virtual_memory()
        disk = resource_psutil.disk_usage("/")

        assert memory.total == 48 * GIB
        assert memory.available == 24 * GIB
        assert memory.percent == 50.0
        assert disk.free == 300 * GIB
        assert resource_psutil.cpu_percent() == 12.5
        assert resource_psutil.cpu_count() == 6
        assert resource_psutil.sensors_battery().percent == 42.0
        assert resource_psutil.disk_io_counters().read_bytes == 0
        assert resource_psutil.net_io_counters().bytes_recv == 0
        assert resource_psutil.net_if_addrs() == {}
        process = resource_psutil.Process(os.getpid())
        assert process.memory_info().rss == 321 * 1024**2
        assert process.cpu_times().user == 12.0
        assert process.num_threads() == 9
        assert process.num_fds() == 4


def test_virtual_memory_fast_path_skips_recursive_process_accounting(
    monkeypatch,
    resource_observer,
):
    calls = []

    class RecordingObserver:
        def memory(self, **kwargs):
            calls.append(kwargs)
            return resource_observer.memory(**kwargs)

    monkeypatch.setattr(
        resource_psutil,
        "get_resource_observer",
        lambda: RecordingObserver(),
    )

    resource_psutil.virtual_memory()

    assert calls == [{"include_process_tree": False}]


def test_legacy_process_facade_scopes_reads_and_native_actions(
    monkeypatch,
    resource_observer,
):
    root_pid = os.getpid()
    child_pid = root_pid + 10_000
    resource_observer.configure_processes(
        [
            _process(resource_observer, pid=root_pid),
            _process(
                resource_observer,
                pid=child_pid,
                ppid=root_pid,
                ancestors=(root_pid,),
                rss_bytes=1234,
            ),
        ]
    )
    resource_observer.configure_connections(
        [
            NetworkConnectionObservation(
                provenance=resource_observer.provenance,
                pid=root_pid,
                fd=4,
                family="AF_INET",
                socket_type="SOCK_STREAM",
                local_host="127.0.0.1",
                local_port=8000,
                status="LISTEN",
            ),
            NetworkConnectionObservation(
                provenance=resource_observer.provenance,
                pid=child_pid,
                fd=5,
                family="AF_INET",
                socket_type="SOCK_STREAM",
                local_host="127.0.0.1",
                local_port=9000,
                status="LISTEN",
            ),
        ]
    )

    root = resource_psutil.Process(root_pid)
    children = root.children(recursive=True)
    assert [child.pid for child in children] == [child_pid]
    assert children[0].as_dict(attrs=["pid", "memory_info"])["memory_info"].rss == 1234
    assert [process.pid for process in resource_psutil.process_iter(attrs=["pid"])] == [
        root_pid,
        child_pid,
    ]
    assert [item.laddr.port for item in root.net_connections()] == [8000]
    assert [item.laddr.port for item in children[0].net_connections()] == [9000]

    calls: list[tuple[str, object]] = []

    class _NativeActionHandle:
        def terminate(self):
            calls.append(("terminate", None))

        def kill(self):
            calls.append(("kill", None))

        def wait(self, timeout=None):
            calls.append(("wait", timeout))
            return 17

    monkeypatch.setattr(
        resource_psutil._psutil,
        "Process",
        lambda pid: _NativeActionHandle(),
    )
    root.terminate()
    root.kill()
    assert root.wait(timeout=2.0) == 17
    assert calls == [("terminate", None), ("kill", None), ("wait", 2.0)]


def _decision_signature(controller: LaneAdmissionController) -> tuple[object, ...]:
    decision = controller.admit(
        model_path="Aura-32B",
        request_gb=22.0,
        active=(
            ActiveLane(
                lane="trainer",
                qos=QoSClass.BEST_EFFORT,
                footprint_gb=18.0,
                model_path="training-job",
            ),
            ActiveLane(
                lane="brainstem",
                qos=QoSClass.BURSTABLE,
                footprint_gb=8.0,
                model_path="brainstem-7B",
            ),
        ),
    )
    return (
        decision.admitted,
        decision.reason,
        decision.budget_gb,
        decision.evict_first,
        decision.observation_source,
        decision.resource_observation_available,
    )


def test_lane_decisions_are_invariant_to_host_api_failure(
    monkeypatch,
    resource_observer,
):
    resource_observer.configure_memory(total_bytes=64 * GIB, available_bytes=8 * GIB)
    expected = _decision_signature(LaneAdmissionController(observer=resource_observer))

    # The patch lands on the psutil module, which every thread in the process
    # shares. In a loaded run a thread an earlier test left polling memory hit
    # it and this test errored with lane admission nowhere on the stack.
    # `admit` runs on the caller's thread, so only this thread is forbidden.
    import threading

    from core.runtime.resource_observation import psutil as host

    here = threading.get_ident()

    def forbidden_here(real):
        def guarded(*args, **kwargs):
            if threading.get_ident() == here:
                raise AssertionError("lane admission reached host state")
            return real(*args, **kwargs)

        return guarded

    monkeypatch.setattr(host, "virtual_memory", forbidden_here(host.virtual_memory))
    monkeypatch.setattr(host, "process_iter", forbidden_here(host.process_iter))

    actual = _decision_signature(LaneAdmissionController(observer=resource_observer))
    assert actual == expected
    assert actual[4] == ObservationSource.SIMULATED.value


def test_lane_decisions_are_order_and_thread_invariant(resource_observer):
    resource_observer.configure_memory(total_bytes=64 * GIB, available_bytes=16 * GIB)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        signatures = list(
            pool.map(
                lambda _index: _decision_signature(
                    LaneAdmissionController(observer=resource_observer)
                ),
                range(64),
            )
        )
    assert len(set(signatures)) == 1

    baseline = signatures[0]
    indices = list(range(64))
    for seed in range(20):
        random.Random(seed).shuffle(indices)
        observed = {
            index: _decision_signature(
                LaneAdmissionController(observer=resource_observer)
            )
            for index in indices
        }
        assert set(observed.values()) == {baseline}


class _UnavailableObserver(SimulatedResourceObserver):
    def memory(self, *, root_pid: int | None = None):
        del root_pid
        raise RuntimeError("memory-probe-failed")

    def thermal(self, *, max_age_s: float = 5.0):
        del max_age_s
        raise RuntimeError("thermal-probe-failed")

    def process_table(self) -> ProcessTableObservation:
        return ProcessTableObservation(
            provenance=self.provenance,
            processes=(),
            available=False,
            error="process-table-failed",
        )


def test_policy_surfaces_fail_closed_when_observation_raises():
    observer = _UnavailableObserver(scenario_id="induced-observer-failure")

    pressure = UnifiedRuntimePressure(observer=observer).runtime_pressure_snapshot()
    governor = ResourceGovernor(observer=observer).sample()

    assert pressure["pressure_ok"] is False
    assert "memory_observation_unavailable" in pressure["red_zones"]
    assert "thermal_observation_unavailable" in pressure["red_zones"]
    assert pressure["observation_source"] == ObservationSource.SIMULATED.value
    assert governor.memory_percent == 100.0
    assert governor.thermal_state is ThermalState.CRITICAL
    assert governor.observation_available is False
    assert observer.process_table().available is False


def test_resource_ledgers_and_receipts_are_scoped_to_each_test(
    resource_observer,
    tmp_path,
):
    runtime_root = Path(os.environ["AURA_TEST_RUNTIME_ROOT"])
    receipt_store = ReceiptStore()
    controller = ModelLaneController(
        receipt_store=receipt_store,
        process_discovery=None,
        observer=resource_observer,
    )
    try:
        snapshot = controller.snapshot()
        assert runtime_root.is_relative_to(tmp_path)
        assert controller.state_path == runtime_root / "model_lane_control.json"
        assert receipt_store.root == runtime_root / "receipts"
        assert snapshot["observation_source"] == ObservationSource.SIMULATED.value
        assert not (Path.home() / ".aura" / "run" / "model_lane_control.json").samefile(
            controller.state_path
        ) if controller.state_path.exists() else True
    finally:
        receipt_store.close()


@pytest.mark.host_leak_observation
def test_host_leak_guard_observes_listener_and_open_file(hermetic_resource_sandbox):
    leaked_file = hermetic_resource_sandbox.root / "held-open.txt"
    with leaked_file.open("w", encoding="utf-8") as handle:
        with hermetic_resource_sandbox.listening_socket() as listener:
            port = listener.getsockname()[1]
            deadline = time.monotonic() + 2.0
            leaks = hermetic_resource_sandbox.leaks()
            while time.monotonic() < deadline and not leaks["listeners"]:
                time.sleep(0.05)
                leaks = hermetic_resource_sandbox.leaks()
            assert any(int(item[3]) == port for item in leaks["listeners"])
            assert str(leaked_file) in leaks["open_files"]
        handle.flush()

    assert hermetic_resource_sandbox.leaks() == {
        "children": set(),
        "listeners": set(),
        "open_files": set(),
    }


@pytest.mark.host_leak_observation
def test_host_leak_guard_observes_spawned_child(hermetic_resource_sandbox):
    child = get_subprocess_gateway().spawn(
        [sys.executable, "-c", "import time; time.sleep(10)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        read_only=True,
        offline_tooling=True,
        source="proof_tooling:resource_observation_hermeticity.spawned_child",
        accelerator_capability="none",
    )
    try:
        deadline = time.monotonic() + 2.0
        leaks = hermetic_resource_sandbox.leaks()
        while time.monotonic() < deadline and not leaks["children"]:
            time.sleep(0.05)
            leaks = hermetic_resource_sandbox.leaks()
        assert any(int(identity[0]) == child.pid for identity in leaks["children"])
    finally:
        child.terminate()
        child.wait(timeout=2.0)

    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and hermetic_resource_sandbox.leaks()["children"]:
        time.sleep(0.05)
    assert hermetic_resource_sandbox.leaks()["children"] == set()
