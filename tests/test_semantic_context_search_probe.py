"""Constrained selection retains lower-ranked labels without target access."""

from types import SimpleNamespace

import numpy as np
import pytest

from core.learning.semantic_context_search_probe import resolve_operation_scores
from core.learning.semantic_program_transducer_fitting import RegisterUseContract


def model():
    return SimpleNamespace(
        hidden_size=2, inference_step_limit=lambda _: 2,
        register_use_contract=RegisterUseContract(1, 1, 1, 1, True),
        _runtime_input_grounding=lambda *args: ((), (), "pointer"),
        definition_pointer=SimpleNamespace(score_sequence=lambda _: "definitions"),
    )


def evidence():
    return dict(source_token_ids=(1, 2), hidden_states=np.eye(2), public_inputs=(7, 3))


def test_lower_ranked_binary_label_survives_infeasible_unary(monkeypatch):
    import core.learning.semantic_context_search_probe as module

    seen = []

    def assign(**kwargs):
        seen.append(tuple(x.operation for x in kwargs["operation_nodes"]))
        return SimpleNamespace(operation_nodes=kwargs["operation_nodes"], arguments=((0, 1),))

    monkeypatch.setattr(module, "_assign_typed_arguments", assign)
    scores = np.full((2, 1, 2), -np.inf)
    scores[0, 0] = [10., 2.]
    program, refusal, trace = resolve_operation_scores(
        model(), **evidence(), scores=scores, labels=("length", "sub"))
    assert not refusal and program.run((7, 3)) == 4
    assert seen == [("sub",)] and trace["nodes"] == 2
    assert not trace["grammar_exhausted"]


def test_first_infeasible_assignment_does_not_end_search(monkeypatch):
    import core.learning.semantic_context_search_probe as module

    def assign(**kwargs):
        nodes = kwargs["operation_nodes"]
        return None if nodes[0].operation == "sub" else SimpleNamespace(
            operation_nodes=nodes, arguments=((0, 1),))

    monkeypatch.setattr(module, "_assign_typed_arguments", assign)
    scores = np.full((2, 1, 2), -np.inf)
    scores[0, 0] = [10., 2.]
    program, refusal, trace = resolve_operation_scores(
        model(), **evidence(), scores=scores, labels=("sub", "add"))
    assert not refusal and program.run((7, 3)) == 10
    assert trace["feasible_charts"] == 2


def test_budget_exhaustion_is_not_a_grammar_impossibility():
    program, refusal, trace = resolve_operation_scores(
        model(), **evidence(), scores=np.ones((2, 1, 1)), labels=("sub",), max_expansions=1)
    assert program is None and "operation_search_incomplete" in refusal
    assert not trace["grammar_exhausted"]


@pytest.mark.parametrize("invalid", [np.nan, np.inf])
def test_invalid_scores_rejected(invalid):
    with pytest.raises(ValueError, match="invalid labeled"):
        resolve_operation_scores(model(), **evidence(), scores=np.full((2, 1, 1), invalid), labels=("sub",))


def test_out_of_bounds_scores_rejected():
    with pytest.raises(ValueError, match="source bounds"):
        resolve_operation_scores(model(), **evidence(), scores=np.zeros((2, 2, 1)), labels=("sub",))
