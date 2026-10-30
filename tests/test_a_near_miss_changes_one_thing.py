"""A near miss is the reference with one property taken out, and the agent is what its name says.

These check the constructions, not what the battery reads on them: that each near miss differs from the reference
in exactly the property it is named for, that none of them enters the null library a campaign compares against, and
that the agent's model keeps nothing and, with no input, nothing in the agent moves. What the lines read on them is
the experiment (tools/run_near_miss_decoys.py), and its predictions are written in docs/ISC_V6_PREREGISTRATION.md
before it runs.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.subject import near_miss
from core.subject.nulls import ARCHITECTURES, architecture
from core.subject.state import DOMAINS
from core.subject.synergy import TRIPLES

pytestmark = pytest.mark.unit

SEED = 7


@pytest.fixture(scope="module")
def reference():
    return architecture("recurrent", seed=SEED)


def _edges(system) -> set[tuple[str, str]]:
    """Who reads whom, as source -> target."""
    return {(other, key) for key, row in system.coupling.items() for other in row}


def _reachable(edges: set[tuple[str, str]], start: str, removed: str | None = None) -> set[str]:
    seen, todo = {start}, [start]
    while todo:
        here = todo.pop()
        for source, target in edges:
            if source == here and target != removed and target not in seen:
                seen.add(target)
                todo.append(target)
    return seen


def test_none_of_them_enters_the_null_library_a_campaign_compares_against():
    assert not set(near_miss.NAMES) & set(ARCHITECTURES)


@pytest.mark.parametrize("name", near_miss.NAMES)
def test_each_is_built_through_the_null_library_by_name(name):
    assert architecture(name, seed=SEED).name == name


def test_without_synergy_every_coupling_stays_and_every_product_goes(reference):
    system = architecture("near_miss_no_synergy", seed=SEED)
    assert all(reference.gains[triple] > 0 for triple in TRIPLES)
    assert all(system.gains[triple] == 0 for triple in TRIPLES)
    assert _edges(system) == _edges(reference)
    for key, row in reference.coupling.items():
        for other, matrix in row.items():
            np.testing.assert_array_equal(system.coupling[key][other], matrix)


def test_driven_adds_one_remembering_variable_outside_k_and_keeps_k_as_it_was(reference):
    system = architecture("near_miss_driven", seed=SEED)
    assert set(system.start(np.random.default_rng(0))) - set(DOMAINS) == {"_hub"}
    assert system.hub_decay == near_miss.DRIVER_MEMORY
    assert set(system.driver_out) == set(DOMAINS)
    assert _edges(system) == _edges(reference)
    assert system.gains == reference.gains


def test_the_cut_vertex_is_the_only_way_between_its_two_sides(reference):
    system = architecture("near_miss_one_cut_vertex", seed=SEED)
    edges = _edges(system)
    assert all(_reachable(edges, key) == set(DOMAINS) for key in DOMAINS), "still strongly connected"
    rest = [key for key in DOMAINS if key not in near_miss.BEHIND_THE_CUT and key != near_miss.CUT_VERTEX]
    for behind in near_miss.BEHIND_THE_CUT:
        assert not _reachable(edges, behind, removed=near_miss.CUT_VERTEX) & set(rest)
    assert system.gains == reference.gains


def test_one_way_keeps_every_triple_and_has_no_way_back(reference):
    system = architecture("near_miss_one_way", seed=SEED)
    assert not {(d, u) for d in near_miss.DOWNSTREAM for u in near_miss.UPSTREAM} & _edges(system)
    for source_a, source_b, target in TRIPLES:
        for source in (source_a, source_b):
            np.testing.assert_array_equal(system.coupling[target][source], reference.coupling[target][source])
    assert system.gains == reference.gains


def test_one_shock_moves_every_domain_with_the_same_draw(reference):
    system = architecture("near_miss_one_shock", seed=SEED)
    state = system.start(np.random.default_rng(1))
    quiet = architecture("near_miss_one_shock", seed=SEED)
    quiet.noise = 0.0
    shocked = system.step(state, np.random.default_rng(2))
    calm = quiet.step(state, np.random.default_rng(2))
    draws = {
        float(((shocked[key] - calm[key]) / (system.noise * system.channels[key][:, 0])).mean()) for key in DOMAINS
    }
    assert max(draws) - min(draws) < 1e-9
    assert _edges(system) == _edges(reference)


def test_weakly_coupled_keeps_every_path_at_a_fraction_of_its_strength(reference):
    system = architecture("near_miss_weakly_coupled", seed=SEED)
    assert _edges(system) == _edges(reference)
    for key, row in reference.coupling.items():
        for other, matrix in row.items():
            np.testing.assert_allclose(system.coupling[key][other], matrix * near_miss.WEAK_SHARE)
    assert system.gains == reference.gains


def test_with_no_input_nothing_in_the_agent_moves():
    agent = architecture(near_miss.AGENT, seed=SEED)
    agent.input_rate = 0.0
    rng = np.random.default_rng(3)
    start = agent.start(rng)
    state = {key: value.copy() for key, value in start.items()}
    for _ in range(200):
        state = agent.step(state, rng)
    for key in DOMAINS:
        np.testing.assert_array_equal(state[key], start[key])


def test_the_agents_model_keeps_nothing_and_its_persona_never_changes():
    agent = architecture(near_miss.AGENT, seed=SEED)
    state = agent.start(np.random.default_rng(4))
    other = {key: value.copy() for key, value in state.items()}
    other["G"] = other["G"] + 5.0
    a = agent.step(state, np.random.default_rng(5))
    b = agent.step(other, np.random.default_rng(5))
    np.testing.assert_array_equal(a["G"], b["G"])  # the context is rebuilt from input, store and persona
    np.testing.assert_array_equal(a["S"], state["S"])
    assert set(agent.frozen) == {"I", "A", "C", "W", "N", "S"}
