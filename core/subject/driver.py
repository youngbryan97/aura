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
import inspect
import logging
import os
import random
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from core.subject.state import (
    FAST_DOMAINS,
    CoreState,
    Organs,
    read_core_state,
)

__all__ = [
    "CONDITIONS",
    "Condition",
    "SubjectRuntime",
    "Snapshot",
    "build_runtime",
    "start_organism",
]

logger = logging.getLogger("Aura.Subject.Driver")

#: How long one phase gets before it is abandoned. A phase that hangs is a
#: phase that did not contribute, and waiting for it turns a battery into a
#: soak test.
PHASE_TIMEOUT: float = 12.0


class DeterministicMind:
    """One answer, always, so two arms differ by the intervention and nothing else."""

    #: The reply is deliberately bland and constant. Anything that varied would
    #: enter the state through several phases at once and appear as coupling.
    REPLY = "Continuity holds. The pipeline is executing over the shared state."

    async def think(self, prompt: str, **_kwargs: Any) -> str:
        del prompt
        return self.REPLY

    async def generate(self, prompt: str, **kwargs: Any) -> str:
        return await self.think(prompt, **kwargs)

    async def classify(self, _prompt: str) -> str:
        return "CHAT"

    async def embed(self, _text: str) -> list[float]:
        return [0.0] * 8


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


def _percept(state: Any, rng: random.Random, source: str, content: str) -> None:
    state.world.recent_percepts.append(
        {
            "source": source,
            "content": content,
            "timestamp": time.time(),
            "salience": round(rng.random(), 3),
        }
    )
    state.world.trim_percepts(50)


def _seen(source: str, contents: Sequence[str]) -> Callable[[Any, random.Random], dict[str, float]]:
    def prepare(state: Any, rng: random.Random) -> dict[str, float]:
        reading = _calm(state, rng)
        _percept(state, rng, source, rng.choice(list(contents)))
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


@dataclass
class Snapshot:
    """Everything a fork has to carry for two arms to start from one place."""

    state: Any
    hidden: np.ndarray
    steps: int
    era: int
    centre: np.ndarray
    scatter: np.ndarray
    centre_n: float
    turn: int
    rng_state: tuple


@dataclass
class SubjectRuntime:
    """The organism, running offline, forkable."""

    kernel: Any
    state: Any
    ontogeny: Any
    rng: random.Random
    turn: int = 0
    retriever: Any = None
    failures: dict[str, int] = field(default_factory=dict)
    failure_notes: dict[str, str] = field(default_factory=dict)
    frames_per_turn: int = 0
    #: Called after every phase when a lesion is in force. See core.subject.clamp.
    after_phase: Any = None
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

    # ── forking ──────────────────────────────────────────────────────────

    def snapshot(self) -> Snapshot:
        return Snapshot(
            state=copy.deepcopy(self.state),
            hidden=np.array(self.ontogeny.h, copy=True),
            steps=int(self.ontogeny.steps),
            era=int(self.ontogeny.era),
            centre=np.array(self.ontogeny._centre, copy=True),
            scatter=np.array(self.ontogeny._scatter, copy=True),
            centre_n=float(self.ontogeny._centre_n),
            turn=self.turn,
            rng_state=self.rng.getstate(),
        )

    def restore(self, snapshot: Snapshot) -> None:
        self.state = copy.deepcopy(snapshot.state)
        self.ontogeny.h = np.array(snapshot.hidden, copy=True)
        self.ontogeny.steps = snapshot.steps
        self.ontogeny.era = snapshot.era
        self.ontogeny._centre = np.array(snapshot.centre, copy=True)
        self.ontogeny._scatter = np.array(snapshot.scatter, copy=True)
        self.ontogeny._centre_n = snapshot.centre_n
        self.turn = snapshot.turn
        self.rng.setstate(snapshot.rng_state)

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
    ) -> list[CoreState]:
        """Run every phase once over the carried state, reading K after each.

        ``perturb_at`` is a frame index; the displacement is applied after that
        frame is read, so the arms share every reading before it and differ
        only from the next one on.
        """
        env = {"turn": float(self.turn), "condition_id": float(_condition_index(condition.name))}
        if condition.prepare is not None:
            env.update(condition.prepare(self.state, self.rng))
        self.state.cognition.current_objective = condition.objective or None
        self.state.cognition.current_origin = condition.origin
        env["objective_len"] = float(len(condition.objective))

        frames: list[CoreState] = []

        async def capture(tag: str) -> None:
            reading = self.read(condition.name, tag, env)
            frames.append(reading)
            if on_frame is not None:
                on_frame(reading)
            if perturb_at is not None and perturb is not None and len(frames) - 1 == perturb_at:
                outcome = perturb(self)
                if inspect.isawaitable(outcome):
                    await outcome

        await capture("open")
        for phase in self.kernel._phases:
            name = phase.__class__.__name__
            try:
                result = await asyncio.wait_for(
                    phase.execute(self.state, objective=condition.objective),
                    timeout=PHASE_TIMEOUT,
                )
                if result is not None:
                    self.state = result
            except BaseException as exc:  # noqa: BLE001 - a phase that dies is a reading
                self.failures[name] = self.failures.get(name, 0) + 1
                self.failure_notes[name] = f"{type(exc).__name__}: {exc}"[:200]
                logger.debug("phase %s failed: %s", name, exc)
            if self.after_phase is not None:
                self.after_phase()
            await capture(name)

        if condition.after == "retrieve":
            self._retrieve(condition.objective)
        elif condition.after == "act":
            self._act(condition.objective, actor=self.actor)
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
        intended = f"plan for turn {self.turn}: {objective[:80]}"
        ok = False
        try:
            path = Path(target) / "notes.txt"
            path.write_text(intended)
            ok = path.read_text() == intended
        except OSError as exc:
            logger.debug("probe action failed: %s", exc)
        record = {
            "intended": intended,
            "verified": ok,
            "at": time.time(),
            "actor": actor,
        }
        self.state.world.facts["last_action"] = record
        self.last_action = dict(record)
        # The same world state, attributed. This is the one call that separates
        # the two arms of the ownership experiment.
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
                "origin": actor,
                "status": "done" if ok else "failed",
            }
        )
        self.state.cognition.last_action_source = actor
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
    workdir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("AURA_LOG_DIR", str(workdir / "logs"))

    import threading
    from types import SimpleNamespace

    from core.kernel.aura_kernel import AuraKernel, KernelConfig
    from core.ontogeny.state import OntogeneticState
    from core.state.aura_state import AuraState
    from core.state.state_repository import StateRepository
    from core.subject.state import domain_width

    # The container has to exist before the kernel is built; the rest of the
    # organism comes up in `start_organism`, which is async.
    from core.service_registration import register_all_services

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

    width = sum(domain_width(key) for key in FAST_DOMAINS)
    ontogeny = OntogeneticState(
        input_width=width, units=64, seed=seed, path=workdir / "ontogeny.npz"
    )
    ontogeny.last_novelty = 0.5
    ontogeny.last_displacement = 0.0

    runtime = SubjectRuntime(
        kernel=kernel,
        state=AuraState.default(),
        ontogeny=ontogeny,
        rng=random.Random(seed),
    )
    runtime.organs = Organs.live()
    runtime.organs = Organs(
        workspace=runtime.organs.workspace,
        substrate=runtime.organs.substrate,
        free_energy=runtime.organs.free_energy,
        self_model=runtime.organs.self_model,
        world_model=runtime.organs.world_model,
        ontogeny=ontogeny,
    )
    runtime._scratch = workdir / "scratch"
    runtime._scratch.mkdir(parents=True, exist_ok=True)
    runtime.retriever = _build_retriever(runtime)
    return runtime


async def start_organism(runtime: SubjectRuntime) -> dict[str, Any]:
    """Bring the layers up, then bind the runtime to what came up.

    Separate from `build_runtime` because it is async and because a caller who
    wants only the phase pipeline — a unit test, say — should not be made to
    boot the consciousness stack to get one.
    """
    from core.subject.organism import bring_up

    organism = await bring_up()
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
    runtime.organs = Organs(
        workspace=live.workspace,
        substrate=organism.substrate or live.substrate,
        free_energy=live.free_energy,
        self_model=live.self_model,
        world_model=live.world_model,
        ontogeny=runtime.ontogeny,
        agency=live.agency,
    )
    runtime.organs = Organs(
        workspace=runtime.organs.workspace,
        substrate=runtime.organs.substrate,
        free_energy=runtime.organs.free_energy,
        self_model=runtime.organs.self_model,
        world_model=runtime.organs.world_model,
        ontogeny=runtime.ontogeny,
        agency=runtime.organs.agency,
    )
    runtime.organism = organism
    return organism.summary()


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
