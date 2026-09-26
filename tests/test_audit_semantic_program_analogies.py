"""The analogy audit must grade held constructions after precedent retrieval."""

from core.learning.procedure_induction import Instruction, Program
from tools.audit_semantic_program_analogies import audit


def test_analogy_audit_keeps_fit_precedents_and_held_labels_separate():
    add = Program(2, (Instruction("add", (0, 1)),))
    multiply = Program(2, (Instruction("mul", (0, 1)),))
    rows = [{"source": f"source-{index}", "construction": f"form-{index}",
             "labels": {add.sha(): True, multiply.sha(): False}}
            for index in range(6)]
    banks = {row["source"]: {add.sha(): add, multiply.sha(): multiply} for row in rows}
    report = audit(rows, banks, excluded=set(), seed=0, top_k=2, trials=2)
    assert report["fit_sources"] == 2
    assert report["fit_verified_programs"] == 1
    assert report["admission"]["sources"] == 2
    assert report["admission"]["candidates"] == 4
    assert report["admission"]["unsupported_graphs"] == 0
    assert report["fit_labels_used_for_precedent_selection"] is True
    assert report["admission_labels_used_for_retrieval"] is False
