"""Many machines in one: a program meant to run on several computers, tried on a cluster simulated here.

A consensus algorithm for five Raspberry Pis cannot be tried on five Raspberry
Pis that are not here, and run once on real ones it would pass by luck: the
faults that break such programs (a message lost at the wrong moment, a machine
that restarts mid-vote, a network cut in two) are rare on a quiet bench. The
people who build these systems test them the way this does (FoundationDB's
simulation, Jepsen): every machine's program runs here against a network and
clock that are simulated. Messages are delayed, dropped, duplicated and
reordered; machines crash and come back with only what they wrote to disk; the
network is cut and healed. Time is simulated, so a run is exact and repeats
from its seed, and a failure found once is found every time.

A machine's program is any object with these methods (a missing one is skipped):

    on_start(ctx)                     it is switched on, or comes back after a crash
    on_message(ctx, sender, message)  a message arrives
    on_timer(ctx, name)               a timer it set goes off

and ``ctx`` gives it what a machine has:

    ctx.id, ctx.peers, ctx.now        who it is, who else there is, the time (seconds)
    ctx.send(to, message)             a message to one machine; ctx.broadcast(message) to the others
    ctx.set_timer(name, after)        a timer, in seconds; set again it is moved
    ctx.disk                          a dict that survives a crash; everything else is lost
    ctx.random                        its own seeded randomness
    ctx.say(event, **what)            something it did that the properties judge (decided, became leader)

Properties are judged on what the machines said, in order: a safety property
must never be broken (two values decided, two leaders in one term), and a
liveness property must come true once the faults stop. The verdict names the
seed that broke one, with the run's last events, so it can be read and run
again. Nothing here knows any algorithm.
"""
from __future__ import annotations

import heapq
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = ["Faults", "Said", "Verdict", "tried_across"]


@dataclass(frozen=True)
class Faults:
    """What the simulated world does wrong, and for how long."""

    delay: tuple[float, float] = (0.001, 0.05)  # each message takes between these, in seconds
    drop: float = 0.05  # the chance a message is lost
    duplicate: float = 0.02  # the chance it arrives twice
    crashes: int = 1  # machines that crash in a run, each coming back after a while
    partitions: int = 1  # times the network is cut in two
    calm_after: float = 20.0  # when every fault stops: liveness is judged from here
    end: float = 60.0  # how long a run lasts


@dataclass(frozen=True)
class Said:
    at: float
    machine: Any
    event: str
    what: dict[str, Any]


@dataclass
class Verdict:
    held: bool
    runs: int
    broken: str = ""
    seed: int | None = None
    trace: list[str] = field(default_factory=list)

    def says(self) -> str:
        if self.held:
            return f"every property held in {self.runs} runs with lost, late and repeated messages, crashes and a cut network"
        return f"run {self.seed} broke it: {self.broken}. Its last events: " + "; ".join(self.trace[-8:])


Property = Callable[[list[Said]], str]  # what is wrong, or ""


class _Machine:
    def __init__(self, world: _World, machine_id: Any, peers: list[Any], program: Any, seed: int) -> None:
        self.world, self.id, self.peers, self.program = world, machine_id, peers, program
        self.disk: dict[str, Any] = {}
        self.random = random.Random(seed)
        self.up = True
        self.timers: dict[str, int] = {}

    # what the program sees as ctx
    @property
    def now(self) -> float:
        return self.world.now

    def send(self, to: Any, message: Any) -> None:
        self.world.post(self.id, to, message)

    def broadcast(self, message: Any) -> None:
        for peer in self.peers:
            self.send(peer, message)

    def set_timer(self, name: str, after: float) -> None:
        self.timers[name] = self.world.schedule(self.world.now + max(0.0, after), ("timer", self.id, name))

    def say(self, event: str, **what: Any) -> None:
        self.world.said.append(Said(self.world.now, self.id, event, what))

    def call(self, method: str, *args: Any) -> None:
        handler = getattr(self.program, method, None)
        if handler is not None and self.up:
            handler(self, *args)


class _World:
    def __init__(self, make: Callable[[], Any], ids: Sequence[Any], faults: Faults, seed: int) -> None:
        self.random = random.Random(seed)
        self.faults = faults
        self.now = 0.0
        self.queue: list[tuple[float, int, tuple]] = []
        self.seq = 0
        self.said: list[Said] = []
        self.log: list[str] = []
        self.cut: tuple[frozenset, frozenset] | None = None
        self.make = make
        self.machines = {i: _Machine(self, i, [p for p in ids if p != i], make(), seed * 1009 + n) for n, i in enumerate(ids)}
        self._plan(list(ids))

    def schedule(self, at: float, event: tuple) -> int:
        self.seq += 1
        heapq.heappush(self.queue, (at, self.seq, event))
        return self.seq

    def _plan(self, ids: list[Any]) -> None:
        f, r = self.faults, self.random
        for _ in range(f.crashes):
            victim, down = r.choice(ids), r.uniform(0.5, f.calm_after * 0.8)
            self.schedule(down, ("crash", victim))
            self.schedule(min(f.calm_after, down + r.uniform(0.5, 5.0)), ("restart", victim))
        for _ in range(f.partitions):
            start = r.uniform(0.5, f.calm_after * 0.8)
            side = frozenset(r.sample(ids, max(1, len(ids) // 2)))
            self.schedule(start, ("cut", side))
            self.schedule(min(f.calm_after, start + r.uniform(1.0, 8.0)), ("heal",))

    def calm(self) -> bool:
        return self.now >= self.faults.calm_after

    def post(self, sender: Any, to: Any, message: Any) -> None:
        f, r = self.faults, self.random
        if to not in self.machines:
            return
        if self.cut is not None and ((sender in self.cut[0]) != (to in self.cut[0])):
            self.log.append(f"{self.now:.3f} {sender}->{to} cut off: {message!r}")
            return
        copies = 1
        if not self.calm():
            if r.random() < f.drop:
                self.log.append(f"{self.now:.3f} {sender}->{to} lost: {message!r}")
                return
            if r.random() < f.duplicate:
                copies = 2
        for _ in range(copies):
            low, high = f.delay
            self.schedule(self.now + r.uniform(low, high), ("message", sender, to, message))

    def run(self, safety: Sequence[Property], liveness: Sequence[Property]) -> tuple[str, list[str]]:
        for machine in self.machines.values():
            machine.call("on_start")
        while self.queue:
            at, seq, event = heapq.heappop(self.queue)
            if at > self.faults.end:
                break
            self.now = at
            kind = event[0]
            if kind == "message":
                _, sender, to, message = event
                machine = self.machines[to]
                if machine.up:
                    self.log.append(f"{at:.3f} {sender}->{to} {message!r}")
                    machine.call("on_message", sender, message)
            elif kind == "timer":
                _, owner, name = event
                machine = self.machines[owner]
                if machine.up and machine.timers.get(name) == seq:
                    machine.call("on_timer", name)
            elif kind == "crash":
                machine = self.machines[event[1]]
                machine.up, machine.timers = False, {}
                self.log.append(f"{at:.3f} {event[1]} crashed")
            elif kind == "restart":
                old = self.machines[event[1]]
                if not old.up:
                    # Back with what it wrote to disk, and nothing else it held.
                    fresh = _Machine(self, old.id, old.peers, self.make(), self.random.randrange(1 << 30))
                    fresh.disk = old.disk
                    self.machines[old.id] = fresh
                    self.log.append(f"{at:.3f} {old.id} restarted")
                    fresh.call("on_start")
            elif kind == "cut":
                ids = set(self.machines)
                self.cut = (event[1], frozenset(ids - event[1]))
                self.log.append(f"{at:.3f} the network is cut: {sorted(event[1], key=str)} | {sorted(ids - event[1], key=str)}")
            elif kind == "heal":
                self.cut = None
                self.log.append(f"{at:.3f} the network is whole again")
            for prop in safety:
                wrong = prop(self.said)
                if wrong:
                    return wrong, self.log
        for prop in liveness:
            wrong = prop(self.said)
            if wrong:
                return f"{wrong} (by {self.faults.end:g}s, with every fault over at {self.faults.calm_after:g}s)", self.log
        return "", self.log


def tried_across(make: Callable[[], Any], machines: int | Sequence[Any], *, safety: Sequence[Property] = (), liveness: Sequence[Property] = (),
                 faults: Faults | None = None, seeds: Sequence[int] = range(100)) -> Verdict:
    """The program ``make`` builds, run on every machine of a simulated cluster, once per seed; the first property broken, if any."""
    ids = list(range(machines)) if isinstance(machines, int) else list(machines)
    faults = faults or Faults()
    runs = 0
    for seed in seeds:
        runs += 1
        wrong, log = _World(make, ids, faults, seed).run(safety, liveness)
        if wrong:
            return Verdict(False, runs, wrong, seed, log[-40:])
    return Verdict(True, runs)
