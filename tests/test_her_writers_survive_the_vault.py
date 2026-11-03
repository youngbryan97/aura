"""What wrote a change to her survives the commit's trip to the process that writes the row.

Her lineage judges a step by whether each change to her person has a measured writer. The process that runs her
phases takes that evidence when it commits, and proxies the state to the vault, another process, which writes the
row. The vault called the same commit again and took the evidence a second time, where no phase had run and its
provenance held no receipts, so every change it saw was recorded with no writer. On the day run of 1 November that was
28 of 35 steps, and every person path on every step. These pin the defect and the repair.
"""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest

import core.phases.affect_update  # noqa: F401 - registers the affect phase's contract
from core.runtime.cognitive_provenance import (
    begin_transformation,
    close_tick,
    open_tick,
    reset_provenance_for_test,
)
from core.state.aura_state import AuraState
from core.state.person_transitions import observe_person, take_person_evidence

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _clean_provenance():
    reset_provenance_for_test()
    yield
    reset_provenance_for_test()


def _grown(state: AuraState, by: float = 0.2) -> AuraState:
    """Her growth written by the measured affect phase, with its receipt."""
    graph = open_tick(objective="a turn of her life")
    measuring = begin_transformation("AffectUpdatePhase", state)
    state.identity.evolution_score = float(state.identity.evolution_score or 0.0) + by
    receipt = measuring.complete(state, publish_violation=False)
    close_tick(graph)
    assert "identity.evolution_score" in receipt.observed_writes
    return state


def _engine_and_vault():
    """Her engine and her vault, each with the view of her its own commits gave it, both at the first state."""
    engine, vault = SimpleNamespace(), SimpleNamespace()
    first = AuraState.default()
    take_person_evidence(engine, first)
    take_person_evidence(vault, copy.deepcopy(first))
    return engine, vault, first


def test_the_engine_records_what_wrote_her_growth():
    engine, _vault, first = _engine_and_vault()
    second = _grown(first.derive("cognitive_cycle"))
    take_person_evidence(engine, second)
    assert second.transition_evidence["last_change"]["identity.evolution_score"]["transform"] == "AffectUpdatePhase"


def test_taking_the_evidence_again_in_the_vault_lost_the_writer():
    """The defect as it was: the vault, which ran no phase, overwrote the engine's evidence."""
    engine, vault, first = _engine_and_vault()
    second = _grown(first.derive("cognitive_cycle"))
    take_person_evidence(engine, second)
    arrived = copy.deepcopy(second)
    reset_provenance_for_test()  # the vault process: no phase ran there, so its provenance holds nothing
    take_person_evidence(vault, arrived)
    change = arrived.transition_evidence["last_change"]["identity.evolution_score"]
    assert "transform" not in change


def test_the_vault_keeps_the_evidence_the_engine_took():
    engine, vault, first = _engine_and_vault()
    second = _grown(first.derive("cognitive_cycle"))
    take_person_evidence(engine, second)
    arrived = copy.deepcopy(second)
    reset_provenance_for_test()
    observe_person(vault, arrived)
    change = arrived.transition_evidence["last_change"]["identity.evolution_score"]
    assert change["transform"] == "AffectUpdatePhase" and change["honoured"] is True


def test_the_vault_stays_current_so_its_own_next_commit_finds_no_change():
    engine, vault, first = _engine_and_vault()
    second = _grown(first.derive("cognitive_cycle"))
    take_person_evidence(engine, second)
    arrived = copy.deepcopy(second)
    reset_provenance_for_test()
    observe_person(vault, arrived)
    later = copy.deepcopy(arrived)
    later.transition_evidence = {}
    take_person_evidence(vault, later)
    assert "identity.evolution_score" not in later.transition_evidence["last_change"]


def test_the_vault_commits_a_proxied_state_without_taking_evidence_again():
    import asyncio

    from core.state.vault import StateVaultActor

    seen = {}

    async def commit(state, cause, trace_id=None, **kwargs):
        seen.update(kwargs)
        return state

    async def deserialize(data):
        return AuraState.default()

    stub = SimpleNamespace(
        repo=SimpleNamespace(commit=commit),
        _deserialize_commit_state=deserialize,
        _last_shm_update=float("inf"),
        _track_task=lambda *a, **k: None,
        _update_shared_memory_async=lambda *a, **k: None,
    )
    asyncio.run(StateVaultActor._process_commit_inner(stub, {"state": {}, "cause": "cognitive_cycle"}, "t"))
    assert seen == {"evidence_taken": True}
