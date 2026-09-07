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

Nothing here starts a server, opens a port, or loads a model. The live desktop
instance is untouched.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

__all__ = ["Organism", "bring_up", "wind_down"]

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

    def summary(self) -> dict[str, Any]:
        return {"up": sorted(self.up), "down": dict(sorted(self.down.items()))}


def _note(organism: Organism, name: str, exc: BaseException | None = None) -> None:
    if exc is None:
        organism.up.append(name)
    else:
        organism.down[name] = f"{type(exc).__name__}: {exc}"[:200]
        logger.warning("subject-core organism: %s did not come up: %s", name, exc)


async def bring_up(*, with_bridge: bool = True) -> Organism:
    """Register the container, start the consciousness layers, stop their loops."""
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

    if organism.substrate is not None:
        try:
            await organism.substrate.stop()
            _note(organism, "substrate_loop_stopped")
        except Exception as exc:  # noqa: BLE001
            _note(organism, "substrate_loop_stopped", exc)

    return organism


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
