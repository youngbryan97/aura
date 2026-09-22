"""Cross-method programs are finite proposals, not inferred correct answers."""

from core.learning.procedure_induction import Instruction, Program
from core.learning.semantic_program_composition import compose_semantic_programs
from core.learning.semantic_program_portfolio import select_semantic_program_portfolio


def _program(first, second):
    return Program(3, (Instruction(first, (0, 1)), Instruction(second, (3, 2))))


def test_complementary_program_parts_survive_alongside_both_parents():
    proposals = {"a": _program("sub", "mul"), "b": _program("add", "add")}
    result = compose_semantic_programs(proposals, (7, 3, 2),
                                       max_candidates=8, max_examined=16)
    programs = {candidate.program.sha(): candidate for candidate in result.candidates}
    wanted = _program("add", "mul")
    assert wanted.sha() in programs
    assert programs[wanted.sha()].instruction_sources == (("b",), ("a",))
    assert len(result.candidates) == 2
    assert result.examined == 4
    assert result.search_exhausted
    assert tuple(proposals) == ("a", "b")


def test_crossovers_with_incompatible_floor_types_are_not_retained():
    length_add = Program(2, (Instruction("length", (0,)), Instruction("add", (2, 1))))
    tail_total = Program(2, (Instruction("tail", (0,)), Instruction("total", (2,))))
    result = compose_semantic_programs({"a": length_add, "b": tail_total},
                                       ((3, 4), 2), max_candidates=8, max_examined=16)
    assert not result.candidates
    assert result.examined == 4
    assert result.search_exhausted


def test_duplicate_methods_do_not_multiply_derived_programs():
    proposals = {"a": _program("sub", "mul"), "alias": _program("sub", "mul"),
                 "b": _program("add", "add")}
    result = compose_semantic_programs(proposals, (7, 3, 2),
                                       max_candidates=8, max_examined=16)
    assert len(result.candidates) == 2
    mixed = next(candidate for candidate in result.candidates
                 if candidate.program == _program("add", "mul"))
    assert mixed.instruction_sources == (("b",), ("a", "alias"))


def test_search_allowances_report_incomplete_instead_of_claiming_exhaustion():
    proposals = {"a": _program("sub", "mul"), "b": _program("add", "add")}
    limited = compose_semantic_programs(proposals, (7, 3, 2),
                                        max_candidates=1, max_examined=16)
    assert len(limited.candidates) == 1
    assert not limited.search_exhausted
    examined = compose_semantic_programs(proposals, (7, 3, 2),
                                         max_candidates=8, max_examined=2)
    assert examined.examined == 2
    assert not examined.search_exhausted


def test_portfolio_preserves_incumbent_and_records_derived_parentage():
    proposals = {"a": _program("sub", "mul"), "b": _program("add", "add")}
    result = select_semantic_program_portfolio(
        proposals=proposals, incumbent="a", provenance={"a": "a" * 64, "b": "b" * 64},
        public_inputs=(7, 3, 2), observation_sha256="c" * 64,
        composition_budget=8, composition_examined_limit=16,
    )
    assert result.decision.selected == "a"
    assert result.composition is not None and result.composition.search_exhausted
    assert len(result.proposals) == 4
    wanted = _program("add", "mul")
    name = "composition:" + wanted.sha()[7:]
    assert dict(result.proposals)[name] == wanted
    assert dict(result.executions)[name]["result"] == 20
    assert dict(result.executions)["a"]["result"] == 8
    assert result.composition.candidates[0].provenance_sha256 != "a" * 64
