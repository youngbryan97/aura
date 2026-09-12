"""One offline life, driven hard enough that the ten domains have something to do.

The measurements need a system that can be forked, perturbed, cut and run
again, which the live desktop runtime cannot be without endangering it. So this
runs the same organism offline: the real kernel phases over one real
``AuraState`` that is carried from turn to turn, the real ontogenetic reservoir
stepped once per turn with the same call the ontogeny service makes, and the
real intentional retriever over her own memory.

Three things about it are not the live runtime, and each one is a limit on what
the numbers below can claim.

The model is a stub that answers the same sentence every time. Holding decoding
constant is what makes two arms of an intervention comparable at all — a 32B
sampling freely would swamp a 0.15 displacement in affect — but it means no
edge measured here runs *through* language. Edges that need the model to read
one state and write another will read as absent.

The environment is scripted. Conditions supply objectives and body states
rather than a screen and a person, so P and I are driven rather than sensed.

The life is short. A few hundred turns is not an ontogeny, and N moves in this
recording only as far as a few hundred reservoir steps move it.

What survives those limits is everything that happens between the phases over
the shared state, which is where the coupling being measured actually lives.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import inspect
import logging
import os
import random
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from dataclasses import fields as dataclass_fields
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from core.soma.effort import note_effort
from core.subject.state import (
    FAST_DOMAINS,
    CoreState,
    Organs,
    read_core_state,
)
from core.subject.steppable import Steps, step_once

__all__ = [
    "CONDITIONS",
    "Condition",
    "SubjectRuntime",
    "Snapshot",
    "build_runtime",
    "quiesce_organism",
    "start_organism",
]

from core.state.percepts import emit_percept

logger = logging.getLogger("Aura.Subject.Driver")

#: How long one phase gets before it is abandoned. A phase that hangs is a
#: phase that did not contribute, and waiting for it turns a battery into a
#: soak test.
PHASE_TIMEOUT: float = 12.0

#: How much substrate time one turn buys. The live loop runs at twenty hertz
#: and a turn takes about half a second, so ten steps' worth is the honest
#: equivalent — taken as one step of that length rather than ten of a twentieth,
#: because two arms must see the same integration and not the same wall clock.

#: The priority below which the will defers an initiative
#: (`core/governance/will.py`). Read here rather than chosen, because "urgent
#: enough to act on" is a decision the governance layer already makes.
WILL_DEFERRAL_BAR: float = 0.3


class DeterministicMind:
    """One answer, always, so two arms differ by the intervention and nothing else.

    Registered both as the kernel's `llm` organ and as the container's
    `llm_router`, because the phases prefer the router and fall back to the
    organ. Installed only as the organ, the response phase asked the real
    router — which has no model loaded offline — got nothing back, and raised
    on every single turn of every run. The reply never landed, so the exchange
    never reached memory consolidation, self-review had nothing to review, and
    the whole arc from a question to an answer was absent from a measurement of
    whether the parts of her reach each other.
    """

    #: Which turn this is. Set by the runtime before each turn and carried
    #: across the fork with everything else, so the two arms of a trial share
    #: it and the reply cannot differ between them — while two different turns
    #: get different replies, which is what stops the loop detector from
    #: reading the harness's constancy as her repeating herself.
    moment: int = 0

    #: The reply is deliberately bland and constant. Anything that varied would
    #: enter the state through several phases at once and appear as coupling.
    REPLY = "Continuity holds. The pipeline is executing over the shared state."

    #: What a caller that parses JSON gets. Several phases ask the model for a
    #: structured answer and raise on prose, so a stub that only speaks prose
    #: makes those phases fail on every turn — the inference phase did, for the
    #: whole of every run, and a phase that always fails is a phase absent from
    #: the measurement. Constant like the prose: the same shape every time, so
    #: two arms still differ by the intervention and nothing else.
    STRUCTURED = (
        '{"implicit_intent": "continue the exchange", "user_subtext": "steady", '
        '"momentum": "steady", "conversation_hooks": []}'
    )

    async def think(self, prompt: str, **_kwargs: Any) -> str:
        # A test double has to honour the interface its callers expect. Which
        # of the two constants comes back is decided by what the caller asked
        # for, not by anything about the state, so the answer is still a
        # function of the call site alone.
        if "json" in str(prompt or "").lower():
            return self.STRUCTURED
        # And the prose carries a mark of what was asked. A reply that is
        # byte-identical on every turn is a repetition, and the memory
        # consolidation phase is right to call it one: it degrades identity
        # stability to its floor and clears the pending initiatives that
        # produced it. So the harness pinned the self-state at its worst value
        # and suppressed deliberation's only output, on every turn of every
        # run, by being too constant.
        #
        # Still a function of the prompt alone, so two arms of a trial — which
        # share a snapshot and therefore a prompt — get the same answer, and
        # nothing about the displacement can reach the decoder.
        seed = f"{self.moment}|{prompt or ''}"
        mark = hashlib.blake2b(seed.encode("utf-8", "ignore"), digest_size=4)
        return f"{self.REPLY} [{mark.hexdigest()}]"

    async def generate(self, prompt: str, **kwargs: Any) -> str:
        return await self.think(prompt, **kwargs)

    async def classify(self, _prompt: str) -> str:
        return "CHAT"

    async def embed(self, _text: str) -> list[float]:
        return [0.0] * 8

    async def route(self, prompt: str, **kwargs: Any) -> str:
        return await self.think(prompt, **kwargs)

    async def chat(self, prompt: str, **kwargs: Any) -> str:
        return await self.think(prompt, **kwargs)

    async def complete(self, prompt: str, **kwargs: Any) -> str:
        return await self.think(prompt, **kwargs)

    def get_stats(self) -> dict[str, Any]:
        # A count that does not move. The body reads token velocity off this,
        # and a growing count would be a clock in the interoception domain.
        return {"total_calls": 0}

    @property
    def high_pressure_mode(self) -> bool:
        return False


@dataclass(frozen=True, slots=True)
class Condition:
    """One kind of life, named before the run so results can be split by it."""

    name: str
    objective: str
    origin: str = "user"
    #: Applied to the state before the turn. This is the world arriving, not
    #: the core acting, so it belongs to E rather than to K.
    prepare: Callable[[Any, random.Random], dict[str, float]] | None = None
    #: Real subsystem work done after the phases, inside the same turn.
    after: str = ""


def _stress(state: Any, rng: random.Random) -> dict[str, float]:
    load = 0.75 + 0.2 * rng.random()
    state.soma.hardware["cpu_usage"] = round(load * 100.0, 2)
    state.soma.hardware["temperature"] = round(70.0 + 25.0 * load, 2)
    state.soma.hardware["vram_usage"] = round(80.0 + 15.0 * rng.random(), 2)
    state.soma.latency["last_thought_ms"] = round(1500.0 + 2500.0 * load, 1)
    return {"host_load": load, "host_thermal": 1.0}


def _calm(state: Any, rng: random.Random) -> dict[str, float]:
    load = 0.05 + 0.15 * rng.random()
    state.soma.hardware["cpu_usage"] = round(load * 100.0, 2)
    state.soma.hardware["temperature"] = round(38.0 + 8.0 * load, 2)
    state.soma.hardware["vram_usage"] = round(30.0 + 10.0 * rng.random(), 2)
    state.soma.latency["last_thought_ms"] = round(200.0 + 400.0 * load, 1)
    return {"host_load": load, "host_thermal": 0.0}


def _percept(
    state: Any, rng: random.Random, kind: str, source: str, content: str
) -> None:
    """The world arriving, in the shape the organism's own percepts arrive in.

    The first version wrote `source` and `salience` and no `type`. The affect
    phase keys on the type and dropped every one of them; the workspace prices
    its perception bid from a strength no producer writes. So for the whole of
    every recording, the world arrived and could not be felt. The salience is
    still drawn — the environment is not uniform — and the draw is matched
    across arms because the generator's state is carried through the fork.
    """
    emit_percept(
        state.world,
        kind,
        content=content,
        intensity=round(rng.random(), 3),
        source=source,
    )


def _seen(
    source: str, contents: Sequence[str], *, kind: str = "interaction"
) -> Callable[[Any, random.Random], dict[str, float]]:
    def prepare(state: Any, rng: random.Random) -> dict[str, float]:
        reading = _calm(state, rng)
        _percept(state, rng, kind, source, rng.choice(list(contents)))
        reading["percept_arrived"] = 1.0
        return reading

    return prepare


#: The eight ordinary situations. A mechanism that only appears in a ninth,
#: written for the battery, is not a property of the ordinary agent.
CONDITIONS: tuple[Condition, ...] = (
    Condition(
        "conversation",
        "Tell me what you have been thinking about today.",
        prepare=_seen("chat", ("Bryan is typing", "Bryan sent a message", "the window has focus")),
    ),
    Condition(
        "problem_solving",
        "Every A is a B, and some B are C. Does it follow that some A are C?",
        prepare=_seen("chat", ("a question arrived", "a hard question arrived")),
    ),
    Condition(
        "autonomy",
        "Review the last hour of your own behaviour and pick one thing to improve.",
        origin="motivation",
        prepare=_calm,
    ),
    Condition("idle", "", origin="system", prepare=_calm),
    Condition(
        "salience",
        "Something you did earlier caused a problem for someone. Sit with that.",
        prepare=_seen("chat", ("a correction arrived", "a complaint arrived")),
    ),
    Condition(
        "stress",
        "Keep working while the machine is under load.",
        origin="system",
        prepare=_stress,
    ),
    Condition(
        "memory",
        "What did we establish earlier that still matters?",
        prepare=_seen("chat", ("a recall request arrived",)),
        after="retrieve",
    ),
    Condition(
        "tool_use",
        "Write today's plan into notes.txt.",
        prepare=_seen("chat", ("a file request arrived",)),
        after="act",
    ),
)


# The fork’s state machinery lives in core/subject/snapshot.py. It is
# imported rather than reachable only through it, because 29 call sites
# already ask the driver for these names.
from core.subject.snapshot import (  # noqa: E402
    SUBSTRATE_BODY,
    Snapshot,
    _HeldObserver,
    _UNFORKED_SERVICES,
    _built_services,
    _differs,
    _effort_state,
    _intentions_state,
    _lifetime_last,
    _moments_of,
    _organ_state,
    _reanchor,
    _restore_effort,
    _restore_intentions,
    _restore_lifetime_last,
    _restore_moments,
    _restore_organ,
    _restore_services,
    _restore_singletons,
    _restore_torch_random,
    _restore_world,
    _service_state,
    _singleton_state,
    _torch_random_state,
    _world_state,
)


@dataclass
class SubjectRuntime:
    """The organism, running offline, forkable."""

    kernel: Any
    state: Any
    ontogeny: Any
    rng: random.Random
    #: The ontogeny service that owns the reservoir. Its accumulators live
    #: outside the reservoir and are carried across the fork with it.
    ontogeny_service: Any = None
    turn: int = 0
    retriever: Any = None
    failures: dict[str, int] = field(default_factory=dict)
    failure_notes: dict[str, str] = field(default_factory=dict)
    frames_per_turn: int = 0
    #: Which services and phases the fork carries, learned at bring-up by
    #: living a turn and seeing what moved. None means carry everything, which
    #: is what calibration itself runs under.
    forked_services: set[str] | None = None
    forked_phases: set[str] | None = None
    #: Called after every phase when a lesion is in force. See core.subject.clamp.
    after_phase: Any = None
    #: Host readings held constant for the duration of a paired trial. The body
    #: senses the real machine, so two arms run seconds apart read different
    #: CPU and different thermals, and that difference is the environment
    #: moving rather than the intervention propagating. Freezing it is the
    #: matched-environment control every arm of a comparison needs.
    frozen_host: dict[str, float] | None = None
    frozen_latency: dict[str, float] | None = None
    #: The host observer held still for a trial, and whatever was installed
    #: before it.
    _held_observer: Any = None
    _previous_observer: Any = None
    #: The clock every arm shares, or None to run on the machine's. See
    #: `core.subject.clock`: the phases that read elapsed time have to be
    #: handed the same interval in both arms or the machine's own speed is in
    #: the floor of every edge into them.
    clock: Any = None
    #: Who the next action is attributed to. "self" is the ordinary case; the
    #: ownership experiment in core.subject.agency sets it to "external" for one
    #: arm and matches everything else, so the two runs differ in authorship
    #: alone.
    actor: str = "self"
    last_action: dict[str, Any] = field(default_factory=dict)
    organs: Organs = field(default_factory=Organs)
    #: The consciousness layer's own tick. In the desktop runtime it free-runs
    #: beside the phases; here it is called once per turn so that two arms of an
    #: intervention see the same number of ticks and differ by the displacement
    #: rather than by how long each one happened to take.
    heartbeat: Any = None
    organism: Any = None
    #: How many times each free-running layer has been advanced, and any that
    #: could not be. Counted rather than timed, so both arms of a trial run the
    #: same layers the same number of times.
    layer_steps: Steps = field(default_factory=Steps)
    #: The harness's own frame count, which decides whose turn it is to step.
    #: Carried in the snapshot and rewound by a restore, because otherwise the
    #: second arm of a trial starts sixty-six frames further on and steps a
    #: different set of layers from the first.
    frame_index: int = 0
    #: The live intention loop, so the probe's action takes the path a real one
    #: takes rather than writing the outcome straight into the state.
    _intentions: Any = None
    #: A recorded sensory stream, or None for the scripted percepts the
    #: conditions write. Played at the turn index rather than at a wall clock,
    #: so both arms of a paired trial see the same frame of the same world and
    #: the difference between them stays the intervention. See
    #: `core.subject.perception_replay`.
    tape: Any = None

    # ── forking ──────────────────────────────────────────────────────────

    #: Which organs are carried across a fork, by the name they are read under.
    #: Every organ the reading declares, taken from the declaration rather
    #: than written out again here. The hand-kept version was missing
    #: `self_prediction` and `comparator` for a whole session: both are
    #: process-wide, both write into the self-state domain every turn, and
    #: left out of the fork the sham arm inherited the prediction error the
    #: displaced arm had just produced — four tenths of a standard deviation
    #: in the floor of every S column before a phase had run. A second list
    #: of the same organs is a second chance to forget one, so there is one.
    ORGAN_FIELDS: ClassVar[tuple[str, ...]] = tuple(
        organ.name for organ in dataclass_fields(Organs)
    )

    def freeze_host(self) -> dict[str, float]:
        """Take the body's current reading and hold it for every arm to come."""
        hardware = dict(getattr(self.state.soma, "hardware", {}) or {})
        latency = dict(getattr(self.state.soma, "latency", {}) or {})
        self.frozen_host = {
            key: float(hardware.get(key, 0.0) or 0.0)
            for key in ("cpu_usage", "vram_usage", "ram_usage", "temperature")
        }
        # Latency is elapsed wall clock, so it differs between two arms run
        # seconds apart by exactly as much as the machine was busy. Same
        # argument as the hardware readings: it is the environment, and it
        # belongs held still.
        self.frozen_latency = {
            key: float(latency.get(key, 0.0) or 0.0)
            for key in ("last_thought_ms", "perception_lag_ms", "token_velocity")
        }
        self._hold_observer()
        return self.frozen_host

    async def calibrate_fork(self, conditions: Sequence[Condition]) -> dict[str, Any]:
        """Find out which services and phases actually move, and carry only those.

        A hundred and ten services are built by the time the organism is up and
        almost none of them change during a turn. Capturing and restoring all
        of them costs about a second each way, and a run makes fifteen hundred
        restores. So the fork is calibrated rather than guessed: take a reading,
        live a turn in each condition, take another, and carry whatever
        differed. What never moves cannot carry a difference between two arms.
        """
        before_services = _service_state(None)
        before_phases = self._phase_state()
        for condition in conditions:
            await self.turn_once(condition)
        after_services = _service_state(None)
        after_phases = self._phase_state()

        def moved(before: dict[str, dict[str, Any]], after: dict[str, dict[str, Any]]) -> set[str]:
            names = set(before) | set(after)
            return {name for name in names if _differs(before.get(name), after.get(name))}

        self.forked_services = moved(before_services, after_services)
        self.forked_phases = moved(before_phases, after_phases)
        return {
            "services_carried": sorted(self.forked_services),
            "services_seen": len(before_services | after_services.keys()),
            "phases_carried": sorted(self.forked_phases),
        }

    def _phase_state(self, only: set[str] | None = None) -> dict[str, dict[str, Any]]:
        # Everything the container already holds is carried under its own name,
        # and the kernel is the machinery rather than the state. What is left
        # is what a phase kept for itself — including the module-level
        # singletons it caches, which no container knows about.
        skip = frozenset(
            {
                id(self.kernel),
                id(self),
                *(id(obj) for obj in _built_services().values()),
                # The organs are carried by name and are the largest objects in
                # the process: the substrate alone holds a half-million weights,
                # and copying it twice per fork is most of what a fork costs.
                *(
                    id(getattr(self.organs, field_name, None))
                    for field_name in self.ORGAN_FIELDS
                ),
                # And the services the fork deliberately leaves alone. Reaching
                # one of them through a phase is the same write the exclusion
                # was written to prevent, and the mycelial topology logs a
                # critical every time the guard refuses it.
                *(
                    id(obj)
                    for name, obj in _built_services().items()
                    if name in _UNFORKED_SERVICES
                ),
            }
        )
        out: dict[str, dict[str, Any]] = {}
        for phase in getattr(self.kernel, "_phases", []) or []:
            name = phase.__class__.__name__
            if only is not None and name not in only:
                continue
            try:
                captured = _organ_state(phase, skip=skip)
            except (
                ArithmeticError,
                AttributeError,
                ImportError,
                LookupError,
                OSError,
                RuntimeError,
                TypeError,
                ValueError,
            ):
                # A phase that cannot be read is skipped, by kind.
                continue
            if captured:
                out[name] = captured
        return out

    def _restore_phases(self, saved: Mapping[str, dict[str, Any]]) -> None:
        if not saved:
            return
        for phase in getattr(self.kernel, "_phases", []) or []:
            fields = saved.get(phase.__class__.__name__)
            if fields:
                _restore_organ(phase, fields)


    def _publish_state(self) -> None:
        """Point the repository at the state this driver is carrying.

        Runtime paths that ask the container for the current state read it off
        the repository, and this driver keeps its state in an attribute. Without
        this they read None: present, registered, and never exercised.
        """
        vault = getattr(self.kernel, "vault", None)
        if vault is None:
            return
        try:
            vault._current = self.state
        except (AttributeError, TypeError):
            return

    def _refresh_health(self) -> None:
        """Run the kernel's own end-of-tick projection over the finished state."""
        refresh = getattr(self.state, "_refresh_cognitive_health", None)
        if not callable(refresh):
            return
        try:
            refresh()
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            logger.debug("cognitive health projection failed: %s", exc)

    def _republish_body(self) -> None:
        """Tell the engine that judges the body what the body was just held at.

        The proprioceptive loop reads the machine and reports it to the
        resilience engine mid-phase, and the hold is applied after the phase.
        Without this the engine — and through it homeostasis, and through that
        her will to live — kept reading the real machine while the state was
        held at the displaced value, which is the two-bodies problem again with
        the seam moved. One body: the held reading is the reading.
        """
        engine = getattr(self.organs, "soma", None)
        report = getattr(engine, "observe_host", None)
        if not callable(report) or self.frozen_host is None:
            return
        try:
            report(
                cpu_percent=self.frozen_host.get("cpu_usage", 0.0),
                ram_percent=self.frozen_host.get(
                    "ram_usage", self.frozen_host.get("vram_usage", 0.0)
                ),
                temperature_c=self.frozen_host.get("temperature"),
            )
        except (AttributeError, TypeError, ValueError):
            return

    def thaw_host(self) -> None:
        self.frozen_host = None
        self.frozen_latency = None
        self._release_observer()

    def _hold_observer(self) -> None:
        """Hold the shared host observer still for the arms that follow.

        The state's body readings are held by `freeze_host`, and every layer
        that reads the machine through the shared observer went round that hold
        — embodied interoception samples it once a second of the organism's
        life, so two arms seconds apart read a different machine and the
        difference was in the floor of every edge into the body. This is the
        same act at the observer's own seam: the first reading of each kind
        stands for all three arms.
        """
        try:
            from core.runtime.resource_observation import (
                get_resource_observer,
                set_resource_observer_for_test,
            )
        except (ImportError, AttributeError):
            return
        if self._held_observer is not None:
            return
        held = _HeldObserver(get_resource_observer())
        self._previous_observer = set_resource_observer_for_test(held)
        self._held_observer = held

    def _release_observer(self) -> None:
        if self._held_observer is None:
            return
        try:
            from core.runtime.resource_observation import set_resource_observer_for_test

            set_resource_observer_for_test(self._previous_observer)
        except (ImportError, AttributeError):
            pass
        self._held_observer = None
        self._previous_observer = None

    def snapshot(self) -> Snapshot:
        return Snapshot(
            outcomes_by_kind=dict(getattr(self, "_outcomes_by_kind", {}) or {}),
            organs={
                name: _organ_state(getattr(self.organs, name, None))
                for name in self.ORGAN_FIELDS
            },
            state=copy.deepcopy(self.state),
            hidden=np.array(self.ontogeny.h, copy=True),
            steps=int(self.ontogeny.steps),
            era=int(self.ontogeny.era),
            centre=np.array(self.ontogeny._centre, copy=True),
            scatter=np.array(self.ontogeny._scatter, copy=True),
            centre_n=float(self.ontogeny._centre_n),
            turn=self.turn,
            rng_state=self.rng.getstate(),
            moments=_moments_of(self.ontogeny_service),
            last_reading=getattr(self.ontogeny_service, "_last_reading", None),
            last_novelty=float(getattr(self.ontogeny, "last_novelty", 0.5)),
            last_displacement=float(getattr(self.ontogeny, "last_displacement", 0.0)),
            lifetime_last=_lifetime_last(),
            phases=self._phase_state(self.forked_phases),
            singletons=_singleton_state(),
            services=_service_state(self.forked_services),
            effort=_effort_state(),
            taken_at=time.time(),
            clock_at=None if self.clock is None else self.clock.now(),
            frame_index=self.frame_index,
            global_random=random.getstate(),
            numpy_random=np.random.get_state(),
            torch_random=_torch_random_state(),
            world=_world_state(getattr(self, "_scratch", None)),
            intentions=_intentions_state(self._intentions),
        )

    def restore(self, snapshot: Snapshot) -> None:
        self._outcomes_by_kind = dict(snapshot.outcomes_by_kind)
        for name, saved in snapshot.organs.items():
            _restore_organ(getattr(self.organs, name, None), saved)
        self.state = copy.deepcopy(snapshot.state)
        self._publish_state()
        self.ontogeny.h = np.array(snapshot.hidden, copy=True)
        self.ontogeny.steps = snapshot.steps
        self.ontogeny.era = snapshot.era
        self.ontogeny._centre = np.array(snapshot.centre, copy=True)
        self.ontogeny._scatter = np.array(snapshot.scatter, copy=True)
        self.ontogeny._centre_n = snapshot.centre_n
        _restore_moments(self.ontogeny_service, snapshot.moments)
        self.ontogeny.last_novelty = snapshot.last_novelty
        self.ontogeny.last_displacement = snapshot.last_displacement
        _restore_lifetime_last(snapshot.lifetime_last)
        service = self.ontogeny_service
        if service is not None:
            service._last_reading = snapshot.last_reading
        self.turn = snapshot.turn
        self.rng.setstate(snapshot.rng_state)
        if snapshot.global_random is not None:
            random.setstate(snapshot.global_random)
        if snapshot.numpy_random is not None:
            np.random.set_state(snapshot.numpy_random)
        _restore_torch_random(snapshot.torch_random)
        self._restore_phases(snapshot.phases)
        _restore_singletons(snapshot.singletons)
        _restore_services(snapshot.services)
        _restore_effort(snapshot.effort)
        self.frame_index = snapshot.frame_index
        _restore_world(getattr(self, "_scratch", None), snapshot.world)
        _restore_intentions(self._intentions, snapshot.intentions)
        if self.clock is not None and snapshot.clock_at is not None:
            # The clock is the state as far as a phase reading elapsed time is
            # concerned, so it rewinds with everything else and the shift below
            # is zero. The shift stays because a run without the clock still
            # has to rewind, and because it covers what the clock cannot: a
            # module that bound `time.time` before the install.
            self.clock.set(snapshot.clock_at)
        if snapshot.taken_at:
            _reanchor(self, time.time() - snapshot.taken_at)

    # ── reading ──────────────────────────────────────────────────────────

    def read(self, condition: str, tag: str, env: dict[str, float]) -> CoreState:
        return read_core_state(
            self.state,
            ontogeny=self.ontogeny,
            organs=self.organs,
            condition=condition,
            tag=tag,
            env=env,
        )

    # ── the turn ─────────────────────────────────────────────────────────

    async def turn_once(
        self,
        condition: Condition,
        *,
        on_frame: Callable[[CoreState], None] | None = None,
        perturb_at: int | None = None,
        perturb: Callable[[SubjectRuntime], None] | None = None,
        sustain: Callable[[SubjectRuntime], None] | None = None,
    ) -> list[CoreState]:
        """Run every phase once over the carried state, reading K after each.

        ``perturb_at`` is a frame index; the displacement is applied after that
        frame is read, so the arms share every reading before it and differ
        only from the next one on.

        ``sustain`` is applied after every frame from then on. `do(X)` holds X
        where it was put, and a domain whose own dynamics pull it back faster
        than its consumers sample it cannot be measured any other way: see
        `core.subject.causal.SUSTAINED`.
        """
        engine = self.kernel.organs.get("llm") if hasattr(self.kernel, "organs") else None
        mind = getattr(engine, "instance", None) if engine is not None else None
        if mind is not None and hasattr(mind, "moment"):
            mind.moment = self.turn
        env = {"turn": float(self.turn), "condition_id": float(_condition_index(condition.name))}
        if condition.prepare is not None:
            env.update(condition.prepare(self.state, self.rng))
        if self.tape is not None:
            # Perception as it actually arrived, at this turn's frame. The
            # timestamp is the experiment's, not the tape's: an instant from
            # the day the tape was cut puts every consumer that reasons about
            # recency into a different decade from the run.
            now = self.clock.now() if self.clock is not None else None
            env["percepts_replayed"] = float(
                self.tape.play(self.state.world, self.turn, now=now)
            )
        self.state.cognition.current_objective = condition.objective or None
        self.state.cognition.current_origin = condition.origin
        env["objective_len"] = float(len(condition.objective))

        frames: list[CoreState] = []

        async def capture(tag: str) -> None:
            if self.clock is not None:
                self.clock.advance()
            # The free-running layers, advanced by the harness's own count
            # rather than by the machine's schedule. Both arms of a trial run
            # them the same number of times, which is the only way a layer that
            # runs on a timer can be inside a paired measurement at all — the
            # alternative was stopping them, and then whatever they contribute
            # to the coupling was absent from every number.
            self.layer_steps = await step_once(
                self.organism, self.frame_index, self.layer_steps
            )
            await self._integrate_substrate(self.frame_index)
            self.frame_index += 1
            reading = self.read(condition.name, tag, env)
            frames.append(reading)
            if on_frame is not None:
                on_frame(reading)
            landed = perturb_at is not None and len(frames) - 1 == perturb_at
            if landed and perturb is not None:
                outcome = perturb(self)
                if inspect.isawaitable(outcome):
                    await outcome
            elif sustain is not None and perturb_at is not None and len(frames) - 1 > perturb_at:
                outcome = sustain(self)
                if inspect.isawaitable(outcome):
                    await outcome

        await capture("open")
        for phase in self.kernel._phases:
            name = phase.__class__.__name__
            # The same cost the production seam reports. This driver runs the
            # phases directly rather than through `wrap_phase`, so without this
            # a turn here cost her nothing where a turn in the runtime costs
            # her thirty phases of work.
            note_effort("phases", 1.0)
            try:
                result = await asyncio.wait_for(
                    phase.execute(self.state, objective=condition.objective),
                    timeout=PHASE_TIMEOUT,
                )
                if result is not None:
                    self.state = result
                    self._publish_state()
            except BaseException as exc:  # noqa: BLE001 - a phase that dies is a reading
                self.failures[name] = self.failures.get(name, 0) + 1
                self.failure_notes[name] = f"{type(exc).__name__}: {exc}"[:200]
                logger.debug("phase %s failed: %s", name, exc)
            if self.frozen_host is not None:
                self.state.soma.hardware.update(self.frozen_host)
                if self.frozen_latency is not None:
                    self.state.soma.latency.update(self.frozen_latency)
                self._republish_body()
            if self.after_phase is not None:
                self.after_phase()
            await capture(name)

        # The projection the real tick makes once the phases are done. The
        # kernel refreshes it at the end of `tick`, outside the phase loop, and
        # this driver runs the phases directly — so coherence, fragmentation,
        # the contradiction count and the whole cognitive-health block were
        # never computed during a run. Four of the workspace domain's columns
        # were constants because of it, and the deliberation phase's reading of
        # how badly the moment was going was reading two of them.
        self._refresh_health()

        if condition.after == "retrieve":
            self._retrieve(condition.objective)
        elif condition.after == "act" or self._intends_to_act():
            # Off the loop. The action is a real write and a real read-back, so
            # it carries an fsync, and an fsync on the event loop is the defect
            # this driver exists to measure rather than to commit.
            #
            # She acts when she has decided to, not only when the condition
            # says so. A return route to perception that exists in one of eight
            # conditions cannot replicate in three, and the specification asks
            # for the loop to close through the world — deliberation, action,
            # environment, perception. Which action is chosen and whether one
            # happens at all are both read off her own intentions.
            await asyncio.to_thread(self._act, condition.objective, actor=self.actor)
        await capture("after")

        await self._consciousness_tick()
        await capture("heartbeat")

        self._step_ontogeny(frames[-1])
        await capture("ontogeny")

        self.turn += 1
        self.frames_per_turn = len(frames)
        return frames

    # ── the real subsystems the conditions reach for ─────────────────────

    async def _consciousness_tick(self) -> None:
        """One beat of the consciousness layer, and one substrate step.

        These run continuously in the desktop runtime — the heartbeat on its
        own interval, the liquid substrate at twenty hertz. Free-running them
        here would put uncontrolled noise between the two arms of every
        intervention and make the sham floor larger than any effect. Calling
        each once per turn keeps the computation and drops the jitter.
        """
        substrate = self.organs.substrate
        if substrate is not None:
            try:
                await asyncio.wait_for(
                    substrate.update(source="subject_core_turn"), timeout=PHASE_TIMEOUT
                )
            except BaseException as exc:  # noqa: BLE001
                self.failures["substrate"] = self.failures.get("substrate", 0) + 1
                self.failure_notes["substrate"] = f"{type(exc).__name__}: {exc}"[:200]
        if self.heartbeat is not None:
            try:
                await asyncio.wait_for(self.heartbeat._tick(), timeout=PHASE_TIMEOUT)
            except BaseException as exc:  # noqa: BLE001
                self.failures["heartbeat"] = self.failures.get("heartbeat", 0) + 1
                self.failure_notes["heartbeat"] = f"{type(exc).__name__}: {exc}"[:200]
        self._train_world_model()


    async def _integrate_substrate(self, frame: int) -> None:
        """The substrate's own iterations for this frame, at its own step size.

        One Euler step of half a second is not thirteen of a tenth. The
        substrate is a nonlinear stochastic recurrent system: the tanh is
        evaluated at different intermediate states, the noise draws are
        independent, and the clip can bite in the middle — so a single large
        step is a different trajectory rather than a coarse version of the same
        one. Here it takes the iterations its own configured rate calls for in
        one frame, each at its own configured integration constant, which is
        what its loop does.

        Scheduled off the frame index like every other layer, so nothing is
        carried across the fork and two arms integrate identically.

        And the whole loop body, not one line of it. This called
        `_step_dynamics`, which is one Euler step and is marked deprecated in
        the substrate itself; the loop it stands in for also settles the psych
        state every iteration, computes the recurrent self-model every fifth
        and applies Hebbian plasticity every hundredth. So the substrate's
        energy regenerated in no run — a standard deviation of seven
        ten-thousandths across a whole recording, which is a dead channel — its
        integrated-information estimate never advanced, and its connectivity
        never learned. A counted schedule that runs a fraction of a layer's
        body measures a different organism from the one that lives here.

        Persistence is left out on purpose. Writing the state to disk is not
        cognition, and a save that lands in one arm and not the other is a
        difference between the arms that nothing thought.
        """
        substrate = self.organs.substrate
        if substrate is None:
            return
        config = getattr(substrate, "config", None)
        rate = float(getattr(config, "update_rate", 20.0) or 20.0)
        dt = float(getattr(config, "time_constant", 0.1) or 0.1)
        if not callable(getattr(substrate, "_step_torch_math", None)):
            return
        from core.subject.steppable import Layer, frame_seconds, iterations_at

        _, count = iterations_at(
            Layer("substrate", "", "", rate, ()), frame, frame_seconds()
        )
        for _ in range(count):
            tick = int(getattr(substrate, "tick_count", 0) or 0)
            body = [
                (name, dt if takes_dt else None)
                for name, every, takes_dt in SUBSTRATE_BODY
                if tick % every == 0
            ]
            for name, argument in body:
                # The `_sync` twin where the substrate has one. Each of these
                # is awaited anyway, so the thread hop buys nothing and its
                # scheduling is one more thing that can differ between arms.
                call = getattr(substrate, f"{name}_sync", None)
                if not callable(call):
                    call = getattr(substrate, name, None)
                if not callable(call):
                    continue
                try:
                    outcome = call() if argument is None else call(argument)
                    if inspect.isawaitable(outcome):
                        await asyncio.wait_for(outcome, timeout=PHASE_TIMEOUT)
                except BaseException as exc:  # noqa: BLE001
                    self.failures["substrate"] = self.failures.get("substrate", 0) + 1
                    self.failure_notes["substrate"] = f"{type(exc).__name__}: {exc}"[:200]
                    return
            substrate.tick_count = tick + 1

    def _train_world_model(self) -> None:
        """The gradient steps the training lane would have taken, on this clock.

        The lane is a thread on a two-second timer, and the wind-down stops it —
        but the unified model starts it again the first time anything asks for
        the learned facet, which happens inside a turn. So it was running
        during the arms, taking however many steps the machine allowed, and the
        model's weights, hidden norm and last surprise all differed between two
        arms that had seen the same data. Three of the world model's columns,
        and part of the body's exertion, were a fact about thread scheduling.

        The passes and the pending-work gate are the lane's own, so what the
        model learns is unchanged. Only the clock it learns on is the
        experiment's rather than the host's.
        """
        model = getattr(self.organs, "world_model", None)
        learned = getattr(model, "learned", None) if model is not None else None
        if learned is None:
            return
        halt = getattr(learned, "stop_training", None)
        if callable(halt) and getattr(learned, "_trainer_thread", None) is not None:
            try:
                halt()
            except (RuntimeError, OSError) as exc:
                logger.debug("could not stop the training lane: %s", exc)
        try:
            if int(getattr(learned, "_pending_since_train", 0)) <= 0:
                return
            learned._pending_since_train = 0
            from core.world_model.learned_world_model import (
                _BPTT_WINDOW,
                _TRAIN_PASSES_PER_CYCLE,
            )

            for _ in range(_TRAIN_PASSES_PER_CYCLE):
                if len(getattr(learned, "_replay", ())) < _BPTT_WINDOW:
                    break
                learned._mini_batch_update()
        except (AttributeError, ImportError, ValueError, RuntimeError, FloatingPointError) as exc:
            self.failures["world_model_training"] = (
                self.failures.get("world_model_training", 0) + 1
            )
            logger.debug("deterministic world-model training failed: %s", exc)

    def _step_ontogeny(self, reading: CoreState) -> None:
        """Carry the last lifetime reading onto the reservoir object N reads.

        The step itself happens inside the affect phase now, through
        `core.ontogeny.lifetime.advance`, which is the runtime's own path. This
        only copies what that step sensed onto the state object so that the N
        domain can read novelty and displacement beside the hidden units. If
        the phase did not advance — the organ was absent, or it degraded — the
        previous reading stands and N shows a flat step, which is what actually
        happened.
        """
        del reading
        try:
            from core.ontogeny.lifetime import last_reading

            step = last_reading()
        except ImportError:
            step = None
        if step is None or self.ontogeny is None:
            return
        self.ontogeny.last_novelty = float(step.novelty)
        self.ontogeny.last_displacement = float(step.displacement)
        self.ontogeny.last_relative_displacement = float(
            getattr(step, "relative_displacement", step.displacement)
        )

    def _retrieve(self, query: str) -> None:
        """Run the real retriever over her own memory, ontogeny included.

        The breadth of this retrieval is chosen by the ontogenetic organ in the
        live runtime, through `IntentionalRetriever._choose_breadth`. Running
        the real object is what makes N -> M a measurable edge rather than an
        assumed one.
        """
        if self.retriever is None:
            return
        try:
            from core.memory.intentional_retrieval import RetrievalIntent

            result = self.retriever.retrieve(
                RetrievalIntent(task=query or "recent context", kind="general", query=query or "recent context", limit=8)
            )
            self.state.cognition.long_term_memory = [
                str(getattr(hit, "content", hit))[:240] for hit in result.hits
            ][:8]
        except Exception as exc:  # noqa: BLE001 - a retriever that dies is a reading
            self.failures["retrieve"] = self.failures.get("retrieve", 0) + 1
            logger.debug("retrieval failed: %s", exc)

    #: What she expects of each thing she can do. The intention loop compares
    #: this against what happened, and the agency ledger keys its capability
    #: beliefs on the name — so a hardcoded `write_notes` made every action she
    #: ever took the same capability, and left her expecting a file to hold a
    #: plan on the turns she was looking in a room.
    ACTION_EXPECTATIONS: ClassVar[dict[str, str]] = {
        "write_notes": "the file holds the plan",
        "append_log": "the line is on the end of the log",
        "make_room": "the room is there",
        "read_room": "the room from last turn is there to look in",
        # The four added so ownership is asked of more than one shape of acting.
        # Each says what its own check in `_act` verifies, so the ledger learns
        # four capabilities rather than one called "the action lands".
        "paint_panel": "the panel reads back as she drew it",
        "visit_room": "she is standing in the next room",
        "finish_task": "the task has all three of its steps",
        "tidy_room": "the room has nothing left in it",
    }

    def _through_the_intention_loop(
        self, intended: str, ok: bool, actor: str, kind: str = "write_notes"
    ) -> None:
        """Say, do, observe — the live agency path, for the probe's own action."""
        if actor != "self":
            # An intention is hers by construction. Recording an outside actor's
            # outcome as one of her intentions is the confusion the ownership
            # experiment exists to detect.
            return
        try:
            loop = self._intentions
            if loop is None:
                return
            expected = self.ACTION_EXPECTATIONS.get(kind, "the action lands")
            identifier = loop.intend(
                intention=intended, drive="creation", expected_outcome=expected
            )
            loop.record_action(
                identifier,
                tool_name=kind,
                args={},
                result="ok" if ok else "failed",
                success=ok,
                duration_ms=1.0,
            )
            outcome = expected if ok else f"{expected} — it did not"
            loop.observe(identifier, observation=expected, actual_outcome=outcome)
        except Exception as exc:  # noqa: BLE001 - the probe's action still stands
            logger.debug("intention loop unavailable: %s", exc)

    #: What she can do to the world, in the order the drives are read. Each is
    #: a real filesystem change with a real reading back, and which one happens
    #: is decided by her state rather than fixed by this harness.
    ACTIONS: ClassVar[tuple[str, ...]] = (
        "write_notes",
        "append_log",
        "make_room",
        "read_room",
        # Four more, because the first four were all the same shape: change a
        # file, read it back, succeed. Ownership measured only on that shape is
        # ownership of writing to disk. These add the shapes the completion
        # specification asks for — a surface rather than a record, going
        # somewhere rather than changing it, finishing a thing that takes more
        # than one step, and an action that half works.
        "paint_panel",
        "visit_room",
        "finish_task",
        "tidy_room",
    )

    #: What each action can come back as. An action repertoire where everything
    #: succeeds cannot distinguish "I did this" from "this worked" — both read
    #: as one constant — and one where nothing fails teaches no efficacy at all.
    OUTCOMES: ClassVar[tuple[str, ...]] = ("succeeded", "partial", "failed")

    #: Which action each drive reaches for. Written out rather than hashed:
    #: `hash()` on a string is salted per process, so the drive that picked
    #: `make_room` in one run picked `read_room` in the next and two runs of the
    #: battery were not comparable in what she actually did. And a mapping by
    #: what the need is for is a reading, where a hash is a coin.
    DRIVE_ACTIONS: ClassVar[dict[str, str]] = {
        "growth": "make_room",
        "curiosity": "read_room",
        "social": "write_notes",
        "integrity": "append_log",
        "energy": "append_log",
    }

    #: And what she does when she is attending to something. The broadcast
    #: winner decides before the drives do, because that is the claim global
    #: workspace theory makes — what wins the competition reaches the
    #: specialised process, and acting is one. It also varies from turn to
    #: turn, where the drives move over days: an action chosen by a standing
    #: budget is the same action every time, and an outcome that is the same
    #: every time cannot teach her anything about what she can do.
    ATTENTION_ACTIONS: ClassVar[dict[str, str]] = {
        "perception": "read_room",
        "world_model": "read_room",
        "ontogeny": "read_room",
        "memory": "append_log",
        "metacognition": "append_log",
        "interoception": "append_log",
        "exchange": "write_notes",
        "self": "write_notes",
        "deliberation": "make_room",
        "substrate": "make_room",
    }

    def _intends_to_act(self) -> bool:
        """Whether anything she is holding is urgent enough to do something about.

        The bar is the will's own: an initiative below three tenths is deferred
        by governance, so one at or above it is one she has decided on.
        """
        cognition = getattr(self.state, "cognition", None)
        if cognition is None:
            return False
        holdings = list(getattr(cognition, "pending_initiatives", []) or [])
        holdings += list(getattr(cognition, "active_goals", []) or [])
        for item in holdings:
            if not isinstance(item, dict):
                continue
            if str(item.get("status", "")) in {"done", "failed"}:
                continue
            for key in ("urgency", "priority"):
                value = item.get(key)
                if isinstance(value, (int, float)) and float(value) >= WILL_DEFERRAL_BAR:
                    return True
        return False

    #: An action kind the agency experiment is holding her to for this arm, or
    #: None for the ordinary case where what she does follows from what she is
    #: attending to. The ownership experiment needs the same action in both
    #: arms with only the author differing, and it needs more than one kind
    #: across its trials.
    forced_action: str | None = None

    def _chosen_action(self) -> str:
        """What she does, decided by what she is attending to.

        Not a constant and not a random draw: the budgets are part of the
        deliberation domain, so displacing that domain changes what she does,
        which changes what the filesystem holds, which changes what her senses
        report back. That is the whole of the return route through the world.
        """
        forced = getattr(self, "forced_action", None)
        if forced in self.ACTIONS:
            return str(forced)
        attending = str(getattr(self.state.cognition, "attention_focus", "") or "")
        source = attending.split(":", 1)[0].strip()
        if source.startswith("affect_"):
            source = "self"
        chosen = self.ATTENTION_ACTIONS.get(source)
        if chosen:
            return chosen
        budgets = getattr(getattr(self.state, "motivation", None), "budgets", {}) or {}
        levels = []
        # Every budget the state carries, read from the state. This listed
        # `creation` and `rest`, which no budget has ever been called, and
        # omitted `growth` and `integrity`, which are two of the five that
        # exist — so the drive that is most depleted on almost every tick was
        # not among the ones considered.
        for name in sorted(budgets):
            entry = budgets.get(name)
            if isinstance(entry, dict):
                levels.append((float(entry.get("level", 100.0) or 0.0), name))
        if not levels:
            return self.ACTIONS[0]
        levels.sort()
        return self.DRIVE_ACTIONS.get(levels[0][1], self.ACTIONS[0])

    #: How many lines the action log keeps. Past this an append rolls it,
    #: which removes lines the append never asked to remove — an outcome she
    #: caused and did not intend. A repertoire with no accidents cannot tell a
    #: deliberate effect from any other kind.
    LOG_LINES: ClassVar[int] = 24

    def _expects_to_succeed(self, kind: str) -> bool:
        """Whether she expects this to work, from what it did last time.

        The first attempt at anything is optimistic, which is a real prior
        rather than a placeholder: nothing has taught her otherwise yet.
        """
        seen = getattr(self, "_outcomes_by_kind", None)
        if not seen:
            return True
        return bool(seen.get(kind, True))

    def _remember_outcome(self, kind: str, ok: bool) -> None:
        if not hasattr(self, "_outcomes_by_kind"):
            self._outcomes_by_kind = {}
        self._outcomes_by_kind[str(kind)] = bool(ok)

    def _roll_the_log_if_long(self, kind: str) -> bool:
        """Trim the log when an append made it too long. Returns whether it bit.

        The trim is a consequence of her own append and of nothing else, and
        she never asked for it. That is what an accidental self-caused outcome
        is, and the repertoire had none.
        """
        if kind != "append_log":
            return False
        target = getattr(self, "_scratch", None)
        if target is None:
            return False
        path = Path(target) / "actions.log"
        try:
            if not path.exists():
                return False
            lines = path.read_text().splitlines()
            if len(lines) <= self.LOG_LINES:
                return False
            from core.governance_context import local_internal_governed_scope
            from core.runtime.file_write_gateway import get_file_write_gateway

            with local_internal_governed_scope("subject_core.action_probe"):
                get_file_write_gateway().write_text(
                    path,
                    "\n".join(lines[-self.LOG_LINES:]) + "\n",
                    source="subject_core.action_probe",
                )
            return True
        except OSError as exc:
            logger.debug("could not roll the action log: %s", exc)
            return False

    def _act(self, objective: str, *, actor: str = "self") -> None:
        """The action arm of the self/world loop, and its consequence.

        The action is a real write to a scratch file inside the run directory
        and the outcome is read back from the filesystem rather than asserted.
        Both the intention and what came of it land in the state.

        ``actor`` is the whole of the ownership experiment. The file ends with
        the same bytes either way, the same fact is recorded, the same goal is
        appended with the same text and the same verified outcome. The two runs
        differ in who is named as having done it, and nothing else. If the
        self-model reads that difference, S diverges; if it does not, the
        divergence is zero and the loop is open at exactly that point.
        """
        target = getattr(self, "_scratch", None)
        if target is None:
            return
        # Which action, chosen from her own state rather than fixed here. The
        # completion specification asks for the reafference loop to be more
        # than one scratch-file pathway, and for the return to be genuinely
        # environmental: the drive that is most depleted picks what she does,
        # the filesystem is what changes, and what comes back is read off the
        # filesystem rather than asserted. Displacing deliberation therefore
        # changes what the world holds, and the world is what she then sees.
        kind = self._chosen_action()
        intended = f"{kind} for turn {self.turn}: {objective[:80]}"
        ok = False
        partial = False
        observed = ""
        # What she expects, before she finds out. Read from what this kind of
        # action did for her last time rather than assumed: a prediction that is
        # always "it will work" is not a prediction, and the difference between
        # being right and being wrong about her own effect is one of the things
        # the ownership experiment has to span.
        predicted = self._expects_to_succeed(kind)
        try:
            from core.governance_context import local_internal_governed_scope
            from core.runtime.file_write_gateway import get_file_write_gateway

            gateway = get_file_write_gateway()
            room = Path(target)
            with local_internal_governed_scope("subject_core.action_probe"):
                if kind == "append_log":
                    path = room / "actions.log"
                    prior = path.read_text() if path.exists() else ""
                    gateway.write_text(
                        path, prior + intended + "\n", source="subject_core.action_probe"
                    )
                    lines = path.read_text().splitlines()
                    ok = bool(lines) and lines[-1] == intended
                    observed = f"the log holds {len(lines)} lines"
                elif kind == "make_room":
                    path = room / f"room_{self.turn:04d}"
                    gateway.ensure_directory(path, source="subject_core.action_probe")
                    rooms = sorted(p.name for p in room.glob("room_*") if p.is_dir())
                    ok = path.is_dir()
                    observed = f"{len(rooms)} rooms exist"
                elif kind == "read_room":
                    # The one that can fail, and fails for a reason that is
                    # hers: the room for a turn exists only if she chose to
                    # make one then. An action repertoire in which nothing can
                    # fail cannot teach efficacy — every attempt succeeded, so
                    # the ledger's efficacy and authored share sat at one for
                    # the whole of every run and the self-state read two
                    # constants where two of its liveliest columns should be.
                    path = room / f"room_{max(0, self.turn - 1):04d}"
                    ok = path.is_dir()
                    observed = (
                        f"room {path.name} holds {len(list(path.iterdir()))} things"
                        if ok
                        else f"there is no room {path.name}"
                    )
                elif kind == "paint_panel":
                    # A surface rather than a record: what it leaves behind is
                    # rendered for looking at, and it is correct only if what
                    # came back can be read as the panel she drew.
                    path = room / "panel.txt"
                    body = f"| {intended[:40]:<40} |"
                    rule = "+" + "-" * 42 + "+"
                    gateway.write_text(
                        path, f"{rule}\n{body}\n{rule}\n", source="subject_core.action_probe"
                    )
                    drawn = path.read_text().splitlines()
                    ok = len(drawn) == 3 and drawn[1] == body
                    observed = f"the panel is {len(drawn)} lines wide {len(rule)}"
                elif kind == "visit_room":
                    # Going somewhere rather than changing something. Nothing
                    # in the world is different afterwards, which is the point:
                    # an action whose whole effect is on where she is.
                    rooms = sorted(p.name for p in room.glob("room_*") if p.is_dir())
                    here = str(self.state.world.facts.get("location", "") or "")
                    ahead = [name for name in rooms if name > here] or rooms
                    if ahead:
                        self.state.world.facts["location"] = ahead[0]
                        ok = True
                        observed = f"standing in {ahead[0]} of {len(rooms)}"
                    else:
                        observed = "there is nowhere to go"
                elif kind == "finish_task":
                    # A thing that takes more than one step, so completion is
                    # a different outcome from progress. Two steps in is a
                    # partial success, and a repertoire with no partial success
                    # has nothing between done and failed.
                    path = room / "task.txt"
                    prior = path.read_text().splitlines() if path.exists() else []
                    steps = prior + [f"step {len(prior) + 1}: {intended[:60]}"]
                    gateway.write_text(
                        path, "\n".join(steps) + "\n", source="subject_core.action_probe"
                    )
                    done = len(steps) % 3 == 0
                    ok = done
                    partial = not done
                    observed = f"{len(steps)} steps, {'finished' if done else 'still going'}"
                elif kind == "tidy_room":
                    # Half works by design. It clears what it can reach and
                    # leaves the directories, so a room with anything in it
                    # comes back partly tidy — and the ledger sees an attempt
                    # that neither succeeded nor failed.
                    rooms = sorted(
                        (p for p in room.glob("room_*") if p.is_dir()), reverse=True
                    )
                    if not rooms:
                        observed = "there is nothing to tidy"
                    else:
                        target_room = rooms[0]
                        inside = sorted(target_room.iterdir())
                        removed = 0
                        for item in inside:
                            if item.is_file():
                                item.unlink()
                                removed += 1
                        left = len(list(target_room.iterdir()))
                        ok = removed > 0 and left == 0
                        partial = removed > 0 and left > 0
                        observed = f"{target_room.name}: cleared {removed}, {left} left"
                else:
                    path = room / "notes.txt"
                    gateway.write_text(
                        path, intended, source="subject_core.action_probe"
                    )
                    ok = path.read_text() == intended
                    observed = f"notes.txt holds {len(intended)} characters"
        except OSError as exc:
            logger.debug("probe action failed: %s", exc)
        # Through the intention loop, which is the path the live runtime takes:
        # form the intention, record the action, observe the outcome. That is
        # what emits the efference copy and compares it, so the agency
        # comparator sees the action rather than sitting at its defaults — its
        # four readings were constant for want of a caller, not for want of
        # anything to say.
        self._through_the_intention_loop(intended, ok, actor, kind)
        outcome = "succeeded" if ok else ("partial" if partial else "failed")
        # An outcome she caused and did not mean to. The log rolls itself when
        # it gets long, so an append can remove lines nobody asked to remove —
        # her hand, her consequence, and no intention of it anywhere.
        accidental = self._roll_the_log_if_long(kind)
        record = {
            "intended": intended,
            "verified": ok,
            "at": time.time(),
            "actor": actor,
            # Which of the things she can do this was, so an experiment over
            # action kinds can say which one it measured.
            "kind": kind,
            # And what shape it came back as. "Failed" and "half done" are
            # different things to have done, and a self-model that tells them
            # apart is doing something a boolean cannot show.
            "outcome": outcome,
            "predicted": predicted,
            "prediction_correct": bool(predicted) == bool(ok),
            "accidental": accidental,
            "deliberate": bool(actor == "self" and not accidental),
        }
        self._remember_outcome(kind, ok)
        self.state.world.facts["last_action"] = record
        self.last_action = dict(record)
        # The same world state, attributed. This is the one call that separates
        # the two arms of the ownership experiment.
        #
        # Only for an outside actor. Her own action already reached the ledger
        # through the intention loop above, and recording it here as well
        # counted every action twice — which halved the efficacy the self model
        # was being handed.
        if actor != "self":
            try:
                from core.agency.authorship import Event, get_agency_ledger

                get_agency_ledger().observe(
                    Event(what="write_notes", actor=actor, verified=ok, detail={"path": "notes.txt"}),
                    self_model=self.organs.self_model,
                )
            except Exception as exc:  # noqa: BLE001
                logger.debug("authorship ledger unavailable: %s", exc)
        self.state.cognition.active_goals.append(
            {
                "id": f"act_{self.turn}",
                "goal": intended,
                "description": intended,
                "origin": actor,
                "status": "done" if ok else "failed",
                # Priced, because a goal the workspace cannot price is a goal
                # attention can never reach — but this one is finished, and a
                # finished goal is a record rather than an intention. The
                # workspace drops it on status; the number is here for whatever
                # reads the record.
                "priority": 1.0 if ok else 0.5,
            }
        )
        self._last_observed = observed
        self.state.cognition.last_action_source = actor

        # And she sees what she did. The loop the specification asks for closes
        # through the world — action, environment, perception — and without
        # this the probe wrote a file and nothing ever came back through the
        # senses, so nothing she did could reach perception by any route.
        _percept(
            self.state,
            self.rng,
            "goal_achieved" if ok else "error",
            "filesystem",
            observed or f"{kind} {'took' if ok else 'did not take'}",
        )
        if len(self.state.cognition.active_goals) > 12:
            del self.state.cognition.active_goals[:-12]


def _condition_index(name: str) -> int:
    for index, condition in enumerate(CONDITIONS):
        if condition.name == name:
            return index
    return -1


def build_runtime(workdir: Path, *, seed: int = 0, mind: Any = None) -> SubjectRuntime:
    """Assemble the offline organism. Import cost lives here, not at module load."""
    os.environ.setdefault("AURA_TESTING", "1")
    from core.governance_context import local_internal_governed_scope
    from core.runtime.file_write_gateway import get_file_write_gateway

    with local_internal_governed_scope("subject_core.driver"):
        get_file_write_gateway().ensure_directory(workdir, source="subject_core.driver")
    os.environ.setdefault("AURA_LOG_DIR", str(workdir / "logs"))

    import threading
    from types import SimpleNamespace

    from core.kernel.aura_kernel import AuraKernel, KernelConfig
    from core.ontogeny.state import OntogeneticState

    # The container has to exist before the kernel is built; the rest of the
    # organism comes up in `start_organism`, which is async.
    from core.service_registration import register_all_services
    from core.state.aura_state import AuraState
    from core.state.state_repository import StateRepository
    from core.subject.state import domain_width

    try:
        register_all_services()
    except Exception as exc:  # noqa: BLE001
        logger.warning("service registration failed: %s", exc)

    vault = StateRepository(db_path=str(workdir / "subject.db"), is_vault_owner=True)
    kernel = AuraKernel(config=KernelConfig(), vault=vault)
    kernel._setup_phases()
    kernel._initialize_organs()
    # The organ shape matters: phases test `organ.ready.is_set()` and read
    # `organ.instance`, and a stub missing either makes the phase raise, which
    # would be recorded as a phase that does nothing rather than as a hole in
    # the harness.
    engine = mind or DeterministicMind()
    ready = threading.Event()
    ready.set()
    kernel.organs["llm"] = SimpleNamespace(
        get_instance=lambda: engine, instance=engine, ready=ready, name="llm"
    )
    # And under the name the phases actually ask for first. Registered only as
    # the organ, every phase that prefers `llm_router` reached the real router,
    # which has no model offline.
    for name in ("llm_router", "local_llm"):
        try:
            from core.container import ServiceContainer

            ServiceContainer.register_instance(name, engine)
        except Exception as exc:  # noqa: BLE001 - a container that refuses is a datum
            logger.warning("could not register the deterministic mind as %s: %s", name, exc)

    width = sum(domain_width(key) for key in FAST_DOMAINS)
    ontogeny = OntogeneticState(
        input_width=width, units=64, seed=seed, path=workdir / "ontogeny.npz"
    )
    ontogeny.last_novelty = 0.5
    ontogeny.last_displacement = 0.0

    # The service's reservoir is a different object from the one N reads, and
    # the affect phase steps that one every turn through
    # `core.ontogeny.lifetime.advance`. Its accumulators are process-wide, so
    # they have to be carried across the fork with everything else.
    try:
        from core.ontogeny.service import get_ontogeny

        ontogeny_service = get_ontogeny()
    except Exception:  # noqa: BLE001 - an absent organ is an absent organ
        ontogeny_service = None

    runtime = SubjectRuntime(
        kernel=kernel,
        state=AuraState.default(),
        ontogeny=ontogeny,
        rng=random.Random(seed),
        ontogeny_service=ontogeny_service,
    )
    runtime.organs = Organs.live()
    # `replace`, never a fresh `Organs(...)` listing the fields by hand. A
    # hand-written rebuild silently drops whatever was added to the dataclass
    # after it was written: this one lost the self-prediction loop and the
    # efference comparator, so eleven of the self-state's columns read zero for
    # the whole of a session and every criterion that depended on them failed
    # on an organ that was there.
    runtime.organs = replace(runtime.organs, ontogeny=ontogeny)
    runtime._scratch = workdir / "scratch"
    with local_internal_governed_scope("subject_core.driver"):
        get_file_write_gateway().ensure_directory(
            runtime._scratch, source="subject_core.driver"
        )
    runtime.retriever = _build_retriever(runtime)
    try:
        from core.agency.intention_loop import IntentionLoop

        runtime._intentions = IntentionLoop(db_path=str(workdir / "intentions.db"))
    except Exception as exc:  # noqa: BLE001
        logger.warning("intention loop unavailable: %s", exc)
    return runtime


#: What one turn of the organism is worth, in seconds of its own life.
#:
#: The free-running layers declare their rates in hertz — the mesh at ten, the
#: field at twenty, the oscillators at a hundred — so a counted schedule needs
#: to know how much life a frame is worth before it can run them at those
#: rates. The first version measured it: it timed the machine's frames and made
#: the experiment's second as long as the host happened to take. That puts the
#: host back into the timeline it was installed to remove, and it means two
#: machines running the same commit give the organism different amounts of life
#: per turn — which is exactly the comparison a second-machine replication is
#: for.
#:
#: So it is fixed, and it is the simplest statement that can be made: a turn is
#: one second. A turn is the organism's unit of experience, the layer rates are
#: per second, and at one second a turn the bridge integrates ten times per
#: turn, which is its own declared cadence. The frame follows from it and from
#: how many readings a turn takes, which is a property of the phase list rather
#: than of the machine.
SECONDS_PER_TURN: float = 1.0


async def calibrate_clock(
    runtime: SubjectRuntime, conditions: Sequence[Condition], *, turns: int = 3
) -> dict[str, float]:
    """Count this organism's frames per turn, then put the run on its own clock.

    The count is a property of the phase list. The machine's real pace is timed
    beside it, on `time.monotonic` — which this never replaces, so asyncio's
    timeouts still fire — and reported rather than used, because how fast the
    host runs is not how much life a turn is worth.
    """
    from core.subject.clock import ExperimentClock

    frames = 0
    started = time.monotonic()
    for index in range(max(1, turns)):
        condition = conditions[index % len(conditions)]
        frames += len(await runtime.turn_once(condition))
    elapsed = max(1e-6, time.monotonic() - started)
    per_turn = max(1, round(frames / max(1, turns)))
    step = SECONDS_PER_TURN / per_turn
    clock = ExperimentClock(step)
    clock.install()
    runtime.clock = clock
    reading = {
        "seconds_per_turn": SECONDS_PER_TURN,
        "frames_per_turn": per_turn,
        "step": round(step, 6),
        "real_seconds_per_frame": round(elapsed / max(1, frames), 5),
    }
    logger.info(
        "subject-core: experiment clock at %.4fs a frame, %d frames a turn "
        "(the machine took %.4fs a frame)",
        step,
        per_turn,
        reading["real_seconds_per_frame"],
    )
    return reading



def _publish_repository(runtime: SubjectRuntime) -> None:
    """Register this run's vault under the name the tree reads it by."""
    vault = getattr(runtime.kernel, "vault", None)
    if vault is None:
        return
    try:
        from core.container import ServiceContainer

        ServiceContainer.register_instance("state_repository", vault, required=False)
    except Exception as exc:  # noqa: BLE001 - a container that refuses is a datum
        logger.warning("could not register the run's state repository: %s", exc)

async def quiesce_organism(runtime: SubjectRuntime) -> list[str]:
    """Stop the free-running loops before the paired arms begin."""
    from core.subject.organism import _live_tasks, quiesce

    stopped = await quiesce()
    if runtime.organism is not None:
        runtime.organism.stopped_loops = stopped
        # And re-read what is left. The summary taken at bring-up listed
        # everything that was running then, which is not what is running now,
        # and a report that says twelve loops are live after nine were stopped
        # is a report nobody can act on.
        runtime.organism.still_running = _live_tasks()
    return stopped


async def start_organism(runtime: SubjectRuntime, *, quiet: bool = False) -> dict[str, Any]:
    """Bring the layers up, then bind the runtime to what came up.

    Separate from `build_runtime` because it is async and because a caller who
    wants only the phase pipeline — a unit test, say — should not be made to
    boot the consciousness stack to get one.
    """
    from core.subject.organism import bring_up

    organism = await bring_up(quiet=quiet)
    # After the bring-up, because it registers the services again and its own
    # registration replaces the one `build_runtime` made. The name has to point
    # at the vault this run is actually carrying its state in, or every runtime
    # path that reads the current state off the container reads someone else's
    # empty one.
    _publish_repository(runtime)
    runtime.heartbeat = organism.heartbeat
    # N reads the organ's shared lifetime reservoir, not a private one. A
    # private reservoir would be a second life running beside the real one and
    # would show the driver's own arithmetic as a developmental state.
    try:
        from core.ontogeny.lifetime import state as lifetime_state

        shared = lifetime_state()
        if shared is not None:
            if not hasattr(shared, "last_novelty"):
                shared.last_novelty = 0.5
            if not hasattr(shared, "last_displacement"):
                shared.last_displacement = 0.0
            runtime.ontogeny = shared
    except Exception as exc:  # noqa: BLE001
        logger.warning("lifetime reservoir unavailable; N stays on the local one: %s", exc)

    live = Organs.live()
    runtime.organs = replace(
        live,
        substrate=organism.substrate or live.substrate,
        ontogeny=runtime.ontogeny,
    )
    runtime.organism = organism
    summary = organism.summary()
    # And learn what the fork has to carry. A hundred and ten services are
    # built by now and almost none of them move during a turn; carrying all of
    # them costs a second each way against fifteen hundred restores in a run.
    try:
        summary["fork"] = await runtime.calibrate_fork(CONDITIONS)
    except Exception as exc:  # noqa: BLE001 - an uncalibrated fork carries everything
        logger.warning("fork calibration failed; carrying every service: %s", exc)
        runtime.forked_services = None
        runtime.forked_phases = None
    return summary


def _build_retriever(runtime: SubjectRuntime) -> Any:
    """The real retriever, over adapters that read her own state.

    The stores are her working memory, her retained long-term list and her
    world facts, which is where the live stores get their contents from. What
    is real here is the planning, the ontogenetic breadth decision and the
    merge; what is scaffolding is the adapters.
    """
    try:
        from core.memory.intentional_retrieval import IntentionalRetriever, MemoryStoreType
    except ImportError:  # pragma: no cover - the retriever is optional to the rest
        return None

    retriever = IntentionalRetriever()

    def working(query: str, limit: int) -> list[dict[str, Any]]:
        del query
        rows = runtime.state.cognition.working_memory[-limit:]
        return [{"content": str(row)[:240], "score": 0.5} for row in rows]

    def semantic(query: str, limit: int) -> list[dict[str, Any]]:
        del query
        rows = runtime.state.cold.long_term_memory[-limit:]
        return [{"content": str(row)[:240], "score": 0.4} for row in rows]

    def facts(query: str, limit: int) -> list[dict[str, Any]]:
        del query
        rows = list(runtime.state.world.facts.items())[-limit:]
        return [{"content": f"{k}={v}"[:240], "score": 0.6} for k, v in rows]

    retriever.register_store(MemoryStoreType.EPISODIC, working)
    retriever.register_store(MemoryStoreType.SEMANTIC, semantic)
    retriever.register_store(MemoryStoreType.RECEIPT, facts)
    return retriever
