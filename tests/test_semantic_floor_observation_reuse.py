"""Search reuses checked computations without changing their evidentiary status."""

import pytest

from core.learning.procedure_induction import Instruction, Program
from core.learning.semantic_graph_counterexamples import ProgramObservationCache, compare_program_meanings


def programs():
    return Program(2, (Instruction("idiv", (0, 1)),)), Program(2, (Instruction("sub", (0, 1)),))


def test_reused_floor_witness_is_identical_and_executes_no_second_time():
    target, other = programs()
    cache = ProgramObservationCache()
    expected = compare_program_meanings(target, other, [(8, 2)])
    first = compare_program_meanings(target, other, [(8, 2)], observation_cache=cache)
    assert first == expected and first["status"] == "different"
    first["witness"]["outcomes"][0]["result"] = 999
    assert compare_program_meanings(target, other, [(8, 2)], observation_cache=cache) == expected
    assert cache.statistics() == {"hits": 2, "executions": 2, "retained": 2, "capacity": 256}


def test_values_program_and_fuel_are_separate_identities():
    target, other = programs()
    cache = ProgramObservationCache()
    assert cache.observe(target, (8, 2), fuel=100_000)["result"] == 4
    assert cache.observe(target, (10, 2), fuel=100_000)["result"] == 5
    assert cache.observe(other, (8, 2), fuel=100_000)["result"] == 6
    assert cache.observe(target, (8, 2), fuel=1)["status"] == "error"
    assert cache.observe(target, (8, 2), fuel=1)["status"] == "error"
    assert cache.statistics()["hits"] == 0
    assert cache.statistics()["retained"] == 3


def test_checked_undefined_domain_is_reused_but_never_means_equivalence():
    target, _ = programs()
    other = Program(2, (Instruction("mod", (0, 1)),))
    cache = ProgramObservationCache()
    first = compare_program_meanings(target, other, [(8, 0)], observation_cache=cache)
    assert first["status"] == "unknown"
    assert compare_program_meanings(target, other, [(8, 0)], observation_cache=cache) == first
    assert cache.statistics()["hits"] == 2


def test_observation_retention_is_bounded_and_evicted_computation_is_reexecuted():
    target, _ = programs()
    cache = ProgramObservationCache(capacity=1)
    for values in ((8, 2), (10, 2), (8, 2)):
        cache.observe(target, values, fuel=100_000)
    assert cache.statistics() == {"hits": 0, "executions": 3, "retained": 1, "capacity": 1}


@pytest.mark.parametrize("values", [(True, 2), ((False,), 2), ([1, 2], 2)])
def test_values_cannot_alias_a_different_semantic_type(values):
    with pytest.raises(ValueError, match="inputs"):
        ProgramObservationCache().observe(programs()[0], values, fuel=100_000)


@pytest.mark.parametrize("capacity", [0, -1, True])
def test_cache_rejects_invalid_capacity(capacity):
    with pytest.raises(ValueError, match="capacity"):
        ProgramObservationCache(capacity=capacity)
