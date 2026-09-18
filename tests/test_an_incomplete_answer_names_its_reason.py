"""A generation reported incomplete says which condition it failed.

LIVE 2026-09-17, four times in one evening:

    ended before semantic completion: missing_parts=[] quality=[]
    epistemic_covered=True terminal_boundary=False tokens=244 stop=none
    spent_budget=False deadline=False

Every reported reason says the answer was fine. The warning printed four
of the five conditions the decision is made on, and the two it left out —
unfulfilled discourse commitments, and whether the model's own end of
utterance counted as a boundary — were the only ones that could have
fired. A warning that names four of five reasons reads as a warning with
no reason at all.
"""

from __future__ import annotations

import inspect

# mlx_worker_surface_quality imports names from mlx_worker at module
# scope and mlx_worker imports back, so the pair only resolves when the
# worker is imported first. Pre-existing; noted here so the ordering is
# not mistaken for style.
import core.brain.llm.mlx_worker  # noqa: F401  isort:skip
from core.brain.llm.mlx_worker_surface_quality import (  # noqa: E402
    _semantic_completion_receipt_state,
    semantic_completion_blockers,
)


def _receipt(**overrides):
    state = {
        "semantic_completion_contract": True,
        "semantic_completion_satisfied": False,
        "semantic_completion_missing_part_indexes": [],
        "semantic_completion_quality_reasons": [],
        "semantic_completion_epistemic_partition_covered": True,
        "semantic_completion_unfulfilled_discourse": [],
        "semantic_completion_terminal_boundary": True,
        "semantic_completion_eos_boundary": True,
    }
    state.update(overrides)
    return state


def test_a_satisfied_generation_has_no_blockers():
    assert semantic_completion_blockers(_receipt(semantic_completion_satisfied=True)) == []


def test_a_generation_with_no_contract_has_no_blockers():
    assert semantic_completion_blockers(_receipt(semantic_completion_contract=False)) == []


def test_the_live_shape_names_the_condition_that_fired():
    # The exact live receipt: everything reported passing, and the answer
    # refused. The two unreported conditions are now the reasons.
    blockers = semantic_completion_blockers(
        _receipt(semantic_completion_terminal_boundary=False, semantic_completion_eos_boundary=False)
    )
    assert "no_terminal_boundary" in blockers
    assert "no_eos_boundary" in blockers


def test_unfulfilled_discourse_is_reported_at_all():
    blockers = semantic_completion_blockers(
        _receipt(
            semantic_completion_unfulfilled_discourse=[
                {"kind": "list", "observed_count": 2, "expected_count": 5,
                 "declaration": "five reasons"}
            ]
        )
    )
    assert any(b.startswith("unfulfilled_discourse=") for b in blockers)
    assert "list:2/5" in " ".join(blockers)


def test_each_failing_condition_appears():
    blockers = " ".join(
        semantic_completion_blockers(
            _receipt(
                semantic_completion_missing_part_indexes=[1, 3],
                semantic_completion_quality_reasons=["empty_reply"],
                semantic_completion_epistemic_partition_covered=False,
                semantic_completion_terminal_boundary=False,
                semantic_completion_eos_boundary=False,
            )
        )
    )
    for expected in (
        "unanswered_parts=[1, 3]",
        "empty_reply",
        "epistemic_partition=False",
        "no_terminal_boundary",
        "no_eos_boundary",
    ):
        assert expected in blockers


def test_a_refusal_with_no_named_condition_says_so():
    # The validator and these diagnostics disagreeing is worth more than
    # either of them being quietly wrong.
    assert semantic_completion_blockers(_receipt()) == [
        "validator_refused_with_no_named_condition"
    ]


def test_every_condition_the_contract_reads_is_one_this_can_name():
    # A condition added to the receipt's decision must be nameable here, or
    # the next live warning is unexplained again.
    decision = inspect.getsource(_semantic_completion_receipt_state)
    tail = decision[decision.index("eos_stop_ready = bool(") :]
    named = inspect.getsource(semantic_completion_blockers)
    for key in (
        "missing_indexes",
        "discourse_missing",
        "quality_reasons",
        "epistemic_covered",
        "eos_completion_boundary",
    ):
        assert key in tail
    for reported in (
        "semantic_completion_missing_part_indexes",
        "semantic_completion_unfulfilled_discourse",
        "semantic_completion_quality_reasons",
        "semantic_completion_epistemic_partition_covered",
        "semantic_completion_eos_boundary",
        "semantic_completion_terminal_boundary",
    ):
        assert reported in named


def test_the_worker_logs_the_derived_reasons():
    from core.brain.llm import mlx_worker

    source = inspect.getsource(mlx_worker)
    assert "semantic_completion_blockers(semantic_completion_state)" in source
    assert "because=%s" in source
