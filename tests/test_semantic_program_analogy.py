"""Structural transfer proposes; the floor decides program equivalence."""

import pytest

from core.learning.procedure_induction import Instruction, Program
from core.learning.semantic_program_analogy import (
    find_program_analogues,
    program_relation_graph,
)


def test_program_analogies_reuse_shared_index_but_refuse_false_equivalence():
    query = Program(2, (Instruction("add", (0, 1)),))
    commuted = Program(2, (Instruction("add", (1, 0)),))
    different = Program(2, (Instruction("mul", (0, 1)),))
    report = find_program_analogues(query, (commuted, different), ((2, 3),), trials=3)
    by_sha = {row["program_sha256"]: row for row in report["retrieved"]}
    assert report["precedents"] == 2
    assert by_sha[commuted.sha()]["equivalent"] is True
    assert by_sha[different.sha()]["equivalent"] is False
    assert by_sha[different.sha()]["meaning"]["status"] == "different"
    assert all("null_mean" in row["analogy"] for row in report["retrieved"])
    assert report["claim"].startswith("structural_hypotheses_only")


def test_program_analogy_requires_connected_typed_graph_and_bounded_search():
    query = Program(2, (Instruction("sub", (0, 1)),))
    graph = program_relation_graph(query)
    assert graph.objects == ("register:0", "register:1", "register:2")
    assert graph.relations[-1].predicate == "result"
    with pytest.raises(ValueError, match="finite retrieval"):
        find_program_analogues(query, (query,), ((2, 1),), top_k=0)
    with pytest.raises(ValueError, match="connected typed"):
        program_relation_graph(Program(2, (Instruction("not-an-op", (0, 1)),)))
