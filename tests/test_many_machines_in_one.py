"""A program for many machines is tried on a simulated cluster: lost and repeated messages, crashes, a cut network.

A correct consensus holds every property in every run; one with the classic
mistake (a proposer that ignores what was already accepted) is caught, and the
run that caught it repeats from its seed.
"""
from __future__ import annotations

import pytest

from core.stand_ins.many_machines import Faults, tried_across

pytestmark = pytest.mark.unit


class _Paxos:
    """Single-decree Paxos on every machine: proposer, acceptor and learner at once."""

    def __init__(self, careless: bool = False) -> None:
        self.careless = careless

    def on_start(self, ctx):
        ctx.disk.setdefault("promised", (0, -1))
        ctx.disk.setdefault("accepted", None)
        self.everyone = [ctx.id, *ctx.peers]
        self.majority = len(self.everyone) // 2 + 1
        self.ballot, self.promises, self.asked, self.votes = None, {}, False, {}
        self.decided = ctx.disk.get("decided")
        ctx.set_timer("propose", ctx.random.uniform(0.05, 1.0))

    def on_timer(self, ctx, name):
        if self.decided is not None:
            return
        ctx.disk["round"] = ctx.disk.get("round", 0) + 1
        self.ballot, self.promises, self.asked = (ctx.disk["round"], ctx.id), {}, False
        for machine in self.everyone:
            ctx.send(machine, ("prepare", self.ballot))
        ctx.set_timer("propose", ctx.random.uniform(0.5, 2.0))

    def on_message(self, ctx, sender, message):
        kind, ballot = message[0], message[1]
        if kind == "prepare" and ballot > ctx.disk["promised"]:
            ctx.disk["promised"] = ballot
            ctx.send(sender, ("promise", ballot, ctx.disk["accepted"]))
        elif kind == "promise" and ballot == self.ballot and not self.asked:
            self.promises[sender] = message[2]
            if len(self.promises) >= self.majority:
                accepted = [a for a in self.promises.values() if a is not None]
                value = max(accepted)[1] if accepted and not self.careless else f"value of {ctx.id}"
                self.asked = True
                for machine in self.everyone:
                    ctx.send(machine, ("accept", ballot, value))
        elif kind == "accept" and ballot >= ctx.disk["promised"]:
            ctx.disk["promised"], ctx.disk["accepted"] = ballot, (ballot, message[2])
            for machine in self.everyone:
                ctx.send(machine, ("accepted", ballot, message[2]))
        elif kind == "accepted":
            voters = self.votes.setdefault((ballot, message[2]), set())
            voters.add(sender)
            if len(voters) >= self.majority and self.decided is None:
                self.decided = ctx.disk["decided"] = message[2]
                ctx.say("decided", value=message[2])


def _one_value(said):
    values = {s.what["value"] for s in said if s.event == "decided"}
    return f"two values were decided: {sorted(values)}" if len(values) > 1 else ""


def _someone_decided(said):
    return "" if any(s.event == "decided" for s in said) else "nothing was ever decided"


def test_a_correct_consensus_holds_in_every_run():
    verdict = tried_across(_Paxos, 5, safety=[_one_value], liveness=[_someone_decided], seeds=range(100))
    assert verdict.held, verdict.says()
    assert verdict.runs == 100


def test_the_classic_mistake_is_caught_and_its_run_repeats():
    faults = Faults(drop=0.3, duplicate=0.1, crashes=2, partitions=2)
    verdict = tried_across(lambda: _Paxos(careless=True), 5, safety=[_one_value], faults=faults, seeds=range(300))
    assert not verdict.held and "two values were decided" in verdict.broken
    again = tried_across(lambda: _Paxos(careless=True), 5, safety=[_one_value], faults=faults, seeds=[verdict.seed])
    assert not again.held and again.broken == verdict.broken and again.trace == verdict.trace


class _NeverDecides:
    def on_start(self, ctx):
        ctx.broadcast(("hello",))


def test_a_program_that_never_finishes_fails_its_liveness():
    verdict = tried_across(_NeverDecides, 3, liveness=[_someone_decided], seeds=range(3))
    assert not verdict.held and "nothing was ever decided" in verdict.broken and "every fault over" in verdict.broken


def test_what_a_crash_keeps_is_only_its_disk():
    class _Counter:
        def on_start(self, ctx):
            ctx.disk["starts"] = ctx.disk.get("starts", 0) + 1
            self.in_memory = getattr(self, "in_memory", 0) + 1
            ctx.say("started", starts=ctx.disk["starts"], in_memory=self.in_memory)

    def restarts_remember_disk_only(said):
        for s in said:
            if s.what["in_memory"] != 1:
                return "memory survived a crash"
        return ""

    verdict = tried_across(_Counter, 3, safety=[restarts_remember_disk_only], faults=Faults(crashes=3, partitions=0), seeds=range(20))
    assert verdict.held, verdict.says()
