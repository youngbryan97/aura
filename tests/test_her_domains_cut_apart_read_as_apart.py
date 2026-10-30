"""The disconnected-Aura control composes her domains' separate lives and says whether they read as one.

tools/run_disconnected_aura.py runs each of her ten domains alone, the other
nine held, and composes the ten recordings. These pin the composition (each
domain's columns come from the life it lived alone) and the verdict's rule,
which was written before the control first ran.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from core.subject.state import DOMAINS
from tools.run_disconnected_aura import compose_alone, verdict

pytestmark = pytest.mark.unit


@dataclasses.dataclass
class Frame:
    values: dict
    misses: dict = dataclasses.field(default_factory=dict)


def test_each_domain_comes_from_the_life_it_lived_alone() -> None:
    alone = {
        domain: [Frame({key: np.full(2, float(index * 10 + t)) for key in DOMAINS}) for t in range(3)]
        for index, domain in enumerate(DOMAINS)
    }
    composed = compose_alone(alone, list(DOMAINS))
    assert len(composed) == 3
    for t, frame in enumerate(composed):
        for index, domain in enumerate(DOMAINS):
            assert frame.values[domain][0] == index * 10 + t


def _arm(phi: float, se: float = 0.01, passes: bool = False) -> dict:
    return {"phi_do": phi, "standard_error": se, "synergy_passes": {"A,S->G": passes, "P,M->W": False}}


def test_apart_and_quiet_is_as_predicted() -> None:
    read = verdict(_arm(0.03), _arm(0.028), _arm(0.004, se=0.006), {d: {} for d in DOMAINS})
    assert read["as_predicted"] is True


def test_integration_read_on_her_cut_apart_is_named() -> None:
    read = verdict(_arm(0.03), _arm(0.028), _arm(0.02, se=0.004), {d: {} for d in DOMAINS})
    assert read["as_predicted"] is False
    assert "phi_do" in read["why"]


def test_synergy_on_her_cut_apart_is_named() -> None:
    read = verdict(_arm(0.03), _arm(0.028), _arm(0.0, passes=True), {d: {} for d in DOMAINS})
    assert read["as_predicted"] is False
    assert "A,S->G" in read["why"]


def test_a_held_column_that_moved_is_a_channel_the_cut_missed() -> None:
    leaks = {d: {} for d in DOMAINS}
    leaks["M"] = {"M.recall_strength": 0.2}
    read = verdict(_arm(0.03), _arm(0.028), _arm(0.0), leaks)
    assert read["as_predicted"] is False
    assert "['M']" in read["why"]
