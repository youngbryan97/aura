"""Explore typed program analogies without mistaking resemblance for meaning."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from core.cognition.structure_mapping import Graph, Relation
from core.cognition.transfer_search import DomainIndex
from core.learning.procedure_induction import Program
from core.learning.semantic_graph_counterexamples import compare_program_meanings
from core.learning.semantic_program_floor import semantic_program_structural_key


def program_relation_graph(program: Program) -> Graph:
    """Expose operation, role, dependency and result structure to the shared index."""
    if semantic_program_structural_key(program) is None:
        raise ValueError("analogy needs a connected typed program")
    registers = [f"register:{index}" for index in range(program.n_inputs + len(program.instructions))]
    relations = [Relation(f"input_role:{index}", (registers[index],))
                 for index in range(program.n_inputs)]
    for index, instruction in enumerate(program.instructions):
        relations.append(Relation(f"operation:{instruction.op}",
                                  tuple(registers[arg] for arg in instruction.args)
                                  + (registers[program.n_inputs + index],), order=2))
    relations.append(Relation("result", (registers[-1],), order=3))
    return Graph(name=program.sha(), relations=tuple(relations))


def find_program_analogues(
    query: Program,
    precedents: Sequence[Program],
    probes: Sequence[tuple],
    *,
    top_k: int=5,
    trials: int=20,
) -> dict[str, Any]:
    """Retrieve structural hypotheses, then independently check their meaning.

    An analogy can suggest a comparison, not certify it. Floor equivalence is
    the only positive authority here; a counterexample refutes equivalence.
    Neither result proves that the source utterance intended either program.
    """
    if type(top_k) is not int or not 1 <= top_k <= 16 or type(trials) is not int or not 1 <= trials <= 100:
        raise ValueError("program analogy search needs finite retrieval and null budgets")
    graph = program_relation_graph(query)
    if len(graph.objects) > 7:
        raise ValueError("program graph exceeds exhaustive structure-mapping budget")
    index = DomainIndex()
    by_sha = {}
    for program in precedents:
        if program.n_inputs != query.n_inputs:
            continue
        precedent = program_relation_graph(program)
        if len(precedent.objects) > 7:
            continue
        if precedent.name != graph.name:
            index.add(precedent)
            by_sha[precedent.name] = program
    retrieved = []
    for transfer in index.find_analogues(graph, top_k=top_k, trials=trials):
        other = by_sha[transfer.target]
        meaning = compare_program_meanings(query, other, probes)
        retrieved.append({"program_sha256": other.sha(), "analogy": transfer.to_dict(),
                          "meaning": meaning, "equivalent": meaning["status"] == "equivalent"})
    return {"schema": "aura.semantic_program_analogy.v1", "query_program_sha256": query.sha(),
            "precedents": len(by_sha), "retrieved": retrieved,
            "claim": "structural_hypotheses_only_without_floor_equivalence_and_source_binding"}
