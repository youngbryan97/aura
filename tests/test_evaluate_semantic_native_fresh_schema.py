"""Fresh-schema scoring cannot borrow source labels or rename its candidates."""

import pytest

from core.learning.procedure_induction import Instruction, Program
from tools.evaluate_semantic_native_checkpoint import digest
from tools.evaluate_semantic_native_fresh_schema import (
    _program_from_record,
    _program_record,
    choose_by_scores,
    frozen_cases,
    verified_row,
)


def test_fresh_three_step_cases_have_disjoint_sources_and_witnessed_contrasts():
    cases = frozen_cases({"fit_ids": [], "calibration_ids": [], "held_ids": []},
                         examples_per_cell=3, contrast_limit=6)
    assert len(cases) == 72
    assert len({case["construction"] for case in cases}) == 24
    assert len({case["source_sha256"] for case in cases}) == 72
    for case in cases:
        programs = [_program_from_record(record) for record in case["candidates"]]
        assert len(programs) == 6
        assert all(program.depth == 3 for program in programs)
        assert case["target_sha256"] in {program.sha() for program in programs}
    with pytest.raises(ValueError, match="overlaps"):
        frozen_cases({"fit_ids": [cases[0]["source_sha256"]], "calibration_ids": [],
                      "held_ids": []}, examples_per_cell=3, contrast_limit=6)


def test_label_blind_scorer_and_replay_require_candidate_identity():
    programs = (
        Program(2, (Instruction("add", (0, 1)),)),
        Program(2, (Instruction("sub", (0, 1)),)),
    )
    case = {"example_id": "example", "source_sha256": "a" * 64,
            "target_sha256": programs[0].sha(),
            "candidates": [_program_record(program) for program in programs]}
    assert choose_by_scores(programs, (-1., -2.)) == programs[0].sha()
    body = {"plan_sha256": "plan", "example_id": "example", "source_sha256": "a" * 64,
            "program_sha256s": [program.sha() for program in programs],
            "scores": [-2., -1.], "unfitted_scores": [-1., -2.],
            "selected_sha256": programs[1].sha(), "unfitted_sha256": programs[0].sha(),
            "selected_correct": False, "unfitted_correct": True,
            "labels_available_to_scorer": False}
    row = {**body, "receipt_sha256": digest(body)}
    assert verified_row(row, case, "plan") == row
    altered = {**body, "selected_correct": True}
    with pytest.raises(ValueError, match="outcome"):
        verified_row({**altered, "receipt_sha256": digest(altered)}, case, "plan")
    altered = {**body, "scores": [-1., -2.]}
    with pytest.raises(ValueError, match="outcome"):
        verified_row({**altered, "receipt_sha256": digest(altered)}, case, "plan")
    changed = {**case, "candidates": list(reversed(case["candidates"]))}
    with pytest.raises(ValueError, match="outcome"):
        verified_row(row, changed, "plan")


def test_program_record_rejects_substitution_and_nonfinite_scores():
    program = Program(2, (Instruction("add", (0, 1)),))
    record = _program_record(program)
    assert _program_from_record(record) == program
    with pytest.raises(ValueError, match="identity"):
        _program_from_record({**record, "sha256": "wrong"})
    with pytest.raises(ValueError, match="finite"):
        choose_by_scores((program,), (float("nan"),))
