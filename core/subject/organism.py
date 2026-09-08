"""Bringing up enough of the organism that the measurement is of the organism.

The first battery run said the ten domains barely influence each other. That
was true of what the harness had assembled and false of Aura. Instrumenting the
service lookups during a driven turn found two hundred and seventy two requests
for an inhibition manager that was not there, and thirty for a neural mesh, and
a long tail after them: the phases were running against a container that
answered None to most of what they asked, and the graph recorded under those
conditions was a graph of a partly assembled machine.

So this brings the layers up in the order the desktop boot brings them up, and
then takes the free-running loops back down.

The loops have to go. Two arms of an intervention are only comparable if the
same amount of computation happened in each, and a twenty-hertz substrate loop
plus a heartbeat on its own timer put an uncontrolled amount of both into
whichever arm ran while the machine was busy. The driver calls the same tick
functions once per turn instead, so the computation is the runtime's and the
timing is the experiment's.

Cancelling the two I knew about left eleven more. The consciousness bridge
starts a loop per layer — neural mesh, neurochemistry, interoception,
oscillatory binding, the unified field, substrate evolution — and the closed
causal loop runs its own prediction cycle, and none of them exposes a per-tick
entry point that could be called instead. Extracting one from each would be a
refactor of production code for the harness's convenience, which is the wrong
trade, so they are stopped and named.

That is a real limit and it goes in the report rather than in a footnote: those
organs are constructed and initialised but not integrating while the
measurement runs, so an edge that depends on their continuous operation reads
as absent here. It is the same kind of limit as holding the model constant —
the alternative is not a better measurement, it is arms that cannot be
compared.

No model is loaded and no port is left open. The consciousness layers include
an inter-instance protocol listener, which `start` binds unconditionally; a
measurement harness advertising itself as an Aura instance is wrong on its own
terms and would collide with the live desktop runtime on the same port, so it
is stopped as soon as it comes up, along with the free-running loops.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

__all__ = ["Organism", "bring_up", "quiesce", "wind_down"]

logger = logging.getLogger("Aura.Subject.Organism")


@dataclass
class Organism:
    """What came up, what did not, and the handles the driver needs."""

    consciousness: Any = None
    bridge: Any = None
    heartbeat: Any = None
    substrate: Any = None
    up: list[str] = field(default_factory=list)
    down: dict[str, str] = field(default_factory=dict)
    #: The free-running cognitive loops this stopped, by name. They are part of
    #: the organism and they are not deterministic, and a measurement has to
    #: choose; this one chooses comparable arms and says what it gave up.
    stopped_loops: list[str] = field(default_factory=list)
    #: What is still running after the wind-down. Reported rather than assumed:
    #: the consciousness boot starts several things and stopping the ones I
    #: know about is not the same as knowing what is left.
    still_running: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        return {
            "up": sorted(self.up),
            "down": dict(sorted(self.down.items())),
            "stopped_loops": sorted(self.stopped_loops),
            "still_running": sorted(self.still_running),
        }


def _note(organism: Organism, name: str, exc: BaseException | None = None) -> None:
    if exc is None:
        organism.up.append(name)
    else:
        organism.down[name] = f"{type(exc).__name__}: {exc}"[:200]
        logger.warning("subject-core organism: %s did not come up: %s", name, exc)


async def bring_up(*, with_bridge: bool = True, quiet: bool = False) -> Organism:
    """Register the container and start the consciousness layers.

    ``quiet`` stops the free-running loops. Leave it off for a recording: an
    observational measure needs a trajectory, not paired arms, and the
    trajectory is more of the organism with the loops running. Turn it on
    before interventions, where two arms have to see the same computation and a
    loop running at whatever rate the machine allows makes them incomparable.

    Which way round matters. Most of the cross-domain coupling flows through
    those loops — stopping them for the whole run dropped the partition score
    from +0.002 to -0.042 — so measuring the transition law with them stopped
    would be measuring a quieter organism than the one that exists.
    """
    organism = Organism()

    from core.container import ServiceContainer
    from core.service_registration import register_all_services

    try:
        register_all_services()
        _note(organism, "container")
    except Exception as exc:  # noqa: BLE001
        _note(organism, "container", exc)
        return organism

    from types import SimpleNamespace

    orchestrator = SimpleNamespace(
        affect_engine=ServiceContainer.get("affect_engine", default=None),
        substrate=ServiceContainer.get("conscious_substrate", default=None),
        self_model=ServiceContainer.get("self_model", default=None),
        state=None,
    )

    try:
        from core.consciousness.system import ConsciousnessSystem

        organism.consciousness = ConsciousnessSystem(orchestrator)
        _note(organism, "consciousness_system")
    except Exception as exc:  # noqa: BLE001
        _note(organism, "consciousness_system", exc)
        return organism

    try:
        await organism.consciousness.start()
        _note(organism, "consciousness_layers")
    except Exception as exc:  # noqa: BLE001
        _note(organism, "consciousness_layers", exc)

    organism.heartbeat = getattr(organism.consciousness, "heartbeat", None)
    organism.substrate = getattr(organism.consciousness, "liquid_substrate", None)

    # The affect phase pushes valence and arousal into the substrate under the
    # key `liquid_substrate`, which only the desktop boot registers. Offline
    # that lookup returned None and the whole affect-to-substrate channel was
    # dead, which would have been recorded as an absent edge rather than as a
    # missing alias.
    if organism.substrate is not None:
        try:
            ServiceContainer.register_instance("liquid_substrate", organism.substrate)
            _note(organism, "liquid_substrate_alias")
        except Exception as exc:  # noqa: BLE001
            _note(organism, "liquid_substrate_alias", exc)

    for key, value in (
        ("consciousness", organism.consciousness),
        ("consciousness_system", organism.consciousness),
        ("global_workspace", getattr(organism.consciousness, "global_workspace", None)),
        ("attention_schema", getattr(organism.consciousness, "attention_schema", None)),
    ):
        if value is not None:
            try:
                ServiceContainer.register_instance(key, value)
            except Exception as exc:  # noqa: BLE001
                _note(organism, f"register:{key}", exc)

    if with_bridge:
        try:
            from core.consciousness.consciousness_bridge import ConsciousnessBridge

            organism.bridge = ConsciousnessBridge(organism.consciousness)
            await organism.bridge.start()
            _note(organism, "consciousness_bridge")
        except Exception as exc:  # noqa: BLE001
            _note(organism, "consciousness_bridge", exc)

    for name, holder in (("heartbeat_loop", organism.consciousness), ("bridge_loop", organism.bridge)):
        await _cancel_loop(organism, name, holder)

    # The inter-instance protocol listener. `ConsciousnessSystem.start` binds
    # it unconditionally, so a battery run was holding a port that the live
    # desktop runtime uses for the same purpose.
    protocol = getattr(organism.consciousness, "aura_protocol", None)
    if protocol is not None:
        try:
            await protocol.stop()
            _note(organism, "protocol_listener_stopped")
        except Exception as exc:  # noqa: BLE001
            _note(organism, "protocol_listener_stopped", exc)

    if organism.substrate is not None:
        try:
            await organism.substrate.stop()
            _note(organism, "substrate_loop_stopped")
        except Exception as exc:  # noqa: BLE001
            _note(organism, "substrate_loop_stopped", exc)

    if quiet:
        organism.stopped_loops = await quiesce()
    organism.still_running = _live_tasks()
    return organism


#: Tasks left alone. Infrastructure the state layer needs to commit a write,
#: not cognition — stopping these would break the run rather than steady it.
KEPT_TASKS: tuple[str, ...] = (
    "state_registry.notification_dispatcher",
)


async def quiesce() -> list[str]:
    """Cancel every free-running cognitive loop, and say which.

    Cancellation is a request, not an event: the task does not end until the
    loop it is suspended in gets to run and raise. So this waits for them,
    briefly, and whatever is still alive afterwards appears in `still_running`
    where it can be argued with rather than in nothing.
    """
    import asyncio

    stopped: set[str] = set()
    doomed: list[Any] = []
    try:
        current = asyncio.current_task()
        for task in asyncio.all_tasks():
            name = task.get_name()
            if task is current or task.done() or name in KEPT_TASKS:
                continue
            task.cancel()
            doomed.append(task)
            stopped.add(name)
    except RuntimeError:
        return sorted(stopped)
    if doomed:
        await asyncio.wait(doomed, timeout=2.0)
    stopped_list = sorted(stopped)
    if stopped_list:
        logger.info(
            "subject-core: stopped %d background loops so two arms see the same "
            "computation: %s",
            len(stopped_list),
            ", ".join(stopped_list),
        )
    return stopped_list


def _live_tasks() -> list[str]:
    """Every asyncio task still alive after the wind-down, by name.

    Stopping the loops I know about is not the same as knowing what is left,
    and a run whose numbers were shaped by a background task nobody listed is
    a run that cannot be repeated. So the list goes in the report.
    """
    import asyncio

    try:
        current = asyncio.current_task()
        return sorted(
            {
                task.get_name()
                for task in asyncio.all_tasks()
                if task is not current and not task.done()
            }
        )
    except RuntimeError:
        return []


async def _cancel_loop(organism: Organism, name: str, holder: Any) -> None:
    """Take one free-running loop down without disturbing what it registered."""
    if holder is None:
        return
    task = getattr(holder, "_task", None)
    beat = getattr(holder, "heartbeat", None)
    if beat is not None and hasattr(beat, "stop"):
        try:
            beat.stop()
        except Exception as exc:  # noqa: BLE001
            _note(organism, f"{name}:stop", exc)
    if task is None:
        return
    try:
        task.cancel()
        holder._task = None
        _note(organism, name)
    except Exception as exc:  # noqa: BLE001
        _note(organism, name, exc)


async def wind_down(organism: Organism) -> None:
    """Stop what was started. Called on the way out of a battery run."""
    for holder in (organism.bridge, organism.consciousness):
        if holder is None:
            continue
        try:
            await holder.stop()
        except Exception as exc:  # noqa: BLE001
            logger.debug("wind down: %s", exc)
