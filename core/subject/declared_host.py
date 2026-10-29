"""The host a run declares, and the one observer every arm reads it through.

The campaign must not read the machine it happens to run on: two arms of an
intervention, or two machines running one commit, would otherwise see different
bodies. So the run declares a host, and every question the organism asks the
resource observer is answered from that declaration. Moved out of
core/subject/snapshot.py, which re-exports both classes.
"""
from __future__ import annotations

from typing import Any


class _HeldObserver:
    """The shared host observer, answering every question the same way.

    A thin memoising face over the real one rather than a second
    implementation of the protocol: the first call of each shape passes
    through and its answer stands until the hold is released. What the machine
    is doing is the environment, and an experiment that compares two arms has
    to hold the environment still or the comparison is between two machines.
    """


    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self._answers: dict[tuple, Any] = {}

    @property
    def provenance(self) -> Any:
        return self._inner.provenance

    @property
    def machine(self) -> Any:
        """The real observer underneath, for questions about the machine rather than about her."""
        return getattr(self._inner, "machine", self._inner)

    def _held(self, name: str, args: tuple, kwargs: dict) -> Any:
        key = (name, args, tuple(sorted(kwargs.items())))
        if key not in self._answers:
            self._answers[key] = getattr(self._inner, name)(*args, **kwargs)
        return self._answers[key]

    def __getattr__(self, name: str) -> Any:
        attribute = getattr(self._inner, name)
        if not callable(attribute):
            return attribute
        return lambda *args, **kwargs: self._held(name, args, kwargs)


def _delegate_to_the_real_observer() -> None:
    """Give the held face a real method per protocol member.

    `isinstance` against a runtime-checkable protocol looks the members up
    statically, so `__getattr__` alone does not satisfy it and the installer
    refuses the object. Written out of the protocol's own attribute list rather
    than by hand, so the face cannot fall behind what the protocol asks for.
    """
    try:
        from core.runtime.resource_observation import ResourceObserver
    except (ImportError, AttributeError):  # pragma: no cover - shipped together
        # not a failure: with no protocol to read, there are no members to
        # install on the face.
        return
    for name in getattr(ResourceObserver, "__protocol_attrs__", ()):  # type: ignore[attr-defined]
        if name == "provenance" or hasattr(_HeldObserver, name):
            continue

        def make(method: str) -> Any:
            def call(self, *args: Any, **kwargs: Any) -> Any:
                return self._held(method, args, kwargs)

            call.__name__ = method
            call.__qualname__ = f"_HeldObserver.{method}"
            return call

        setattr(_HeldObserver, name, make(name))


_delegate_to_the_real_observer()


#: The thermal module's own thresholds for calling a temperature serious and
#: critical (core/runtime/thermal.py), so a declared temperature reads as the
#: level a host at that temperature would report.
_SERIOUS_C = 78.0
_CRITICAL_C = 90.0


class _DeclaredHost(_HeldObserver):
    """The host a run declares, standing in for the machine for the whole run.

    Each turn's condition prepares a host reading and the recorded body keeps
    it. The organs that guard against load did not read the body: they asked
    the process-wide observer, which answered from the real machine. When
    another campaign loaded the machine during the seed-7 run, proprioception's
    reflex dropped metabolic load and inhibited world decay on 241 turns,
    homeostasis throttled cognitive depth on 280, and the defensive resource
    monitor, sampling from its own thread between turns, pushed the emergency
    protocol into minimal mode. The recorded body read the prepared host
    throughout, so none of it could be explained from the recording.

    Installed once for the run and told each turn's prepared reading, this
    answers compute, memory and thermal from that reading to every reader, in
    any thread and between turns. Every other question, processes,
    connections, open files, disk, power and the machine's traffic counters,
    goes to the real observer once and is held from then on, as the held face
    holds every answer. Her interoception feels disk traffic, process count and
    power, so these cannot move between two arms either: the traffic counters
    went around the observer until 22 October, and two untouched forks of one
    anchor parted at the mesh's sensory input.

    Her own process's size is held at what it was when the run declared its
    host. It had passed straight through, and in an offline run the process is
    mostly the harness: every anchor's snapshot is in it. So it rose through a
    run like a clock, differed from arm to arm, and crossed the 3,500 MB line
    at which homeostasis throttles how deeply she thinks. On the seed-7 run of
    25 September at 21fec3d95 the homeostatic coupling's copy of it was the
    largest single leak into her core: a gain of 0.0148, where the whole leak
    read 0.0452 against a shuffled floor of 0.0028.
    """

    def __init__(self, inner: Any, host: dict[str, float]) -> None:
        super().__init__(inner)
        from core.runtime.lockdep import LockRank, checked_lock

        self._declared_lock = checked_lock("subject.snapshot.declared_host", rank=LockRank.LEAF)
        self._held_rss: tuple[int, int] | None = None
        self.declare(host)

    def declare(self, host: dict[str, float]) -> None:
        """Make this reading the host until the next one."""
        inner = self._inner
        from core.runtime.resource_observation import (
            ComputeObservation,
            MemoryObservation,
            ObservationProvenance,
            ObservationSource,
            ThermalObservation,
        )

        provenance = ObservationProvenance(
            source=ObservationSource.SIMULATED,
            scenario_id="subject-core-declared-host",
            observer="core.subject.snapshot._DeclaredHost",
        )
        real_compute = inner.compute()
        real_memory = inner.memory(include_process_tree=False)
        if self._held_rss is None:
            self._held_rss = (
                int(getattr(real_memory, "process_rss_bytes", 0) or 0),
                int(getattr(real_memory, "process_tree_rss_bytes", 0) or 0),
            )
        cpu = max(0.0, min(100.0, float(host.get("cpu_usage", 0.0) or 0.0)))
        cores = max(1, int(getattr(real_compute, "cpu_count", 1) or 1))
        load = cpu / 100.0 * cores
        compute = ComputeObservation(
            provenance=provenance,
            cpu_percent=cpu,
            cpu_count=cores,
            load_1m=load,
            load_5m=load,
            load_15m=load,
            boot_time=float(getattr(real_compute, "boot_time", 0.0) or 0.0),
        )
        percent = float(host.get("ram_usage") or host.get("vram_usage") or 0.0)
        percent = max(0.0, min(100.0, percent))
        total = int(getattr(real_memory, "total_bytes", 0) or 0)
        available = int(total * (1.0 - percent / 100.0))
        memory = MemoryObservation(
            provenance=provenance,
            total_bytes=total,
            available_bytes=available,
            used_bytes=total - available,
            free_bytes=available,
            active_bytes=total - available,
            percent=percent,
            process_rss_bytes=self._held_rss[0],
            process_tree_rss_bytes=self._held_rss[1],
        )
        temperature = float(host.get("temperature", 0.0) or 0.0)
        level = 3 if temperature >= _CRITICAL_C else 2 if temperature >= _SERIOUS_C else 0
        thermal = ThermalObservation(provenance=provenance, level=level, provider="declared")
        with self._declared_lock:
            self._compute, self._memory, self._thermal = compute, memory, thermal

    def compute(self) -> Any:
        with self._declared_lock:
            return self._compute

    def memory(self, *args: Any, **kwargs: Any) -> Any:
        with self._declared_lock:
            return self._memory

    def thermal(self, *args: Any, **kwargs: Any) -> Any:
        with self._declared_lock:
            return self._thermal
