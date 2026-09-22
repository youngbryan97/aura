"""Partial analogy preserves one-to-one bindings and never invents matches."""

import pytest

from core.cognition.relational_generalization import RelationalCase, RelationalGeneralizer
from core.cognition.structure_mapping import Graph, Relation, _score, map_structures


def test_larger_source_retains_a_real_partial_analogy():
    source = Graph("source", (Relation("follows", ("b", "a")), Relation("follows", ("c", "b"))))
    target = Graph("target", (Relation("behind", ("y", "x")),))
    match = map_structures(source, target)
    assert match is not None
    assert match.score == pytest.approx(0.5)
    assert len(set(match.mapping.values())) == len(match.mapping) == 2


def test_unmapped_object_cannot_match_by_surface_name():
    source = Graph("source", (Relation("r", ("a", "same")),))
    target = Graph("target", (Relation("r", ("x", "same")),))
    assert _score(source, target, {"a": "x"}, {"r": "r"})[0] == 0


def test_omitted_predicate_cannot_match_by_surface_name():
    source = Graph("source", (Relation("r", ("a",)),))
    target = Graph("target", (Relation("r", ("x",)),))
    assert _score(source, target, {"a": "x"}, {})[0] == 0
    assert _score(source, target, {"a": "x"}, None)[0] == 1


def test_relational_adapter_reuses_partial_mapping_without_claiming_equivalence():
    source = RelationalCase("queue", (("a", "person"), ("b", "person"), ("c", "person")),
                            (("follows", "b", "a"), ("follows", "c", "b")), goal="serve first")
    target = RelationalCase("road", (("x", "vehicle"), ("y", "vehicle")),
                            (("behind", "y", "x"),), goal="move first")
    engine = RelationalGeneralizer()
    assert engine.shared_structure(source, target).score == pytest.approx(0.5)
    assert not engine.same_problem(source, target)
