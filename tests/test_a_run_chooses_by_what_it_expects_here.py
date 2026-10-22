"""A campaign turn's choice weighs what she expects of each act in the world as it stands.

The run's choice was a Borda count over attention, drive, feeling and habit;
the world was not a reason, so choice by consequence never reached a run. The
fifth reason is her feeling about the act times how much likelier this world
makes it to work, learned from her own acts' outcomes in the run.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from core.agency.consequence import ConsequenceModel
from core.subject.choosing import expected_here


def _runtime():
    return SimpleNamespace(state=SimpleNamespace(), ACTIONS=("read_room", "write_notes"))


def test_with_no_record_the_reason_follows_her_feelings(monkeypatch):
    import core.agency.consequence as consequence

    monkeypatch.setattr(consequence, "_MODEL", ConsequenceModel())
    monkeypatch.setattr(consequence, "world_of", lambda state: np.array([1.0, 0.0]))
    runtime = _runtime()
    assert expected_here(runtime, runtime.ACTIONS, [0.5, -0.5]) == [0.75, 0.25]
    assert np.allclose(runtime.world_at_choice, [1.0, 0.0])


def test_an_act_that_works_in_this_world_is_worth_more_here(monkeypatch):
    import core.agency.consequence as consequence

    model = ConsequenceModel()
    rng = np.random.default_rng(0)
    for _ in range(120):
        here = rng.choice([-1.0, 1.0])
        world = np.array([here, rng.normal()])
        model.learn("act:read_room", world, good=bool(here > 0))
        model.learn("act:write_notes", world, good=bool(here < 0))
    monkeypatch.setattr(consequence, "_MODEL", model)
    runtime = _runtime()
    monkeypatch.setattr(consequence, "world_of", lambda state: np.array([1.0, 0.0]))
    there = expected_here(runtime, runtime.ACTIONS, [0.0, 0.0])
    monkeypatch.setattr(consequence, "world_of", lambda state: np.array([-1.0, 0.0]))
    elsewhere = expected_here(runtime, runtime.ACTIONS, [0.0, 0.0])
    assert there[0] > there[1] and elsewhere[1] > elsewhere[0]


def test_her_self_model_is_a_prior_over_her_acts(monkeypatch):
    """Bryan's analysis of 21 October: the self-model as a prior over policies,
    checked by clamping the self-model and reading the choice. Two ledgers that
    differ only in what she has seen herself bring off rank her acts apart."""
    import core.agency.authorship as authorship
    from core.agency.authorship import SELF, AgencyLedger, Event
    from core.subject.choosing import believed_she_can

    def ledger_where(good: str, bad: str) -> AgencyLedger:
        held = AgencyLedger()
        for _ in range(12):
            held.observe(Event(what=good, actor=SELF, verified=True))
            held.observe(Event(what=bad, actor=SELF, verified=False))
        return held

    acts = ("read_room", "write_notes")
    monkeypatch.setattr(authorship, "_LEDGER", ledger_where("read_room", "write_notes"))
    one = believed_she_can(acts)
    monkeypatch.setattr(authorship, "_LEDGER", ledger_where("write_notes", "read_room"))
    other = believed_she_can(acts)
    assert one[0] > one[1] and other[1] > other[0]
