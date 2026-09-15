"""Persistent planner state lives in the action domain and moves with it.

The intention loop keeps what she has declared she means to do on disk between
turns. The fork carried it and nothing read it, so displacing deliberation left
her standing plans where they were and a lesion of deliberation held none of
them. These pin the three places it now belongs: the action domain reads how
many she holds, a displacement declares or abandons one through the loop's own
path, and a lesion that holds the domain holds the loop.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.agency.intention_loop import IntentionLoop
from core.state.aura_state import AuraState
from core.subject.clamp import Clamp, organs_read_by
from core.subject.state import Organs, feature_names, perturb_organs, read_core_state


@pytest.fixture
def loop(tmp_path: Path) -> IntentionLoop:
    made = IntentionLoop(db_path=str(tmp_path / "intentions.db"))
    yield made
    made.close()


def _column(reading, name: str) -> float:
    names = feature_names("D")
    return float(reading.values["D"][names.index(name)])


def test_the_action_domain_reads_the_intentions_she_holds(loop: IntentionLoop) -> None:
    state = AuraState.default()
    before = _column(read_core_state(state, organs=Organs(intentions=loop)), "D.durable_intentions_open")
    loop.intend("finish the index", drive="growth")
    loop.intend("write back to Sam", drive="connection")
    after = _column(read_core_state(state, organs=Organs(intentions=loop)), "D.durable_intentions_open")
    assert after > before


def test_a_displacement_towards_deliberation_declares_one_and_away_abandons_the_newest(loop: IntentionLoop) -> None:
    organs = Organs(intentions=loop)
    assert asyncio.run(perturb_organs(organs, "D", 0.2, state=None)) is True
    held = loop.get_open_intentions()
    assert len(held) == 1 and "probe" in held[0].intention
    loop.intend("finish the index", drive="growth")
    assert asyncio.run(perturb_organs(organs, "D", -0.2, state=None)) is True
    remaining = [rec.intention for rec in loop.get_open_intentions()]
    assert remaining == [held[0].intention]


def test_a_lesion_that_holds_deliberation_holds_the_loop(loop: IntentionLoop) -> None:
    assert "intentions" in organs_read_by(["D"])
    holder = SimpleNamespace(state=AuraState.default(), organs=Organs(intentions=loop), ontogeny=None)
    loop.intend("finish the index", drive="growth")
    clamp = Clamp(holder, ["D"])
    loop.intend("something the held arm must not keep", drive="curiosity")
    clamp.apply()
    assert [rec.intention for rec in loop.get_open_intentions()] == ["finish the index"]


def test_without_a_loop_nothing_is_displaced() -> None:
    assert asyncio.run(perturb_organs(Organs(), "D", 0.2, state=None)) is False
