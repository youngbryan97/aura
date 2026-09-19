"""Capacity expansion reuses the relation head and begins without a new score term."""

import numpy as np
import pytest

from core.learning.semantic_relation_tissue import DirectionalRelationHead
from core.learning.semantic_relation_graph_learning import RelationEvidenceBank, RelationGraphContrast
from core.learning.semantic_graph_constraints import _fit_graph_parameters


def head():
    return DirectionalRelationHead(np.zeros(9), .2, .3,
        np.array([[1.], [0.], [0.]]), np.array([[.5], [-.5], [0.]]))


def test_expansion_preserves_old_component_and_has_live_gradient():
    original = head()
    wider = original.expanded_rank(3, seed=7)
    np.testing.assert_array_equal(wider.query_projection[:, :1], original.query_projection)
    np.testing.assert_array_equal(wider.definition_projection[:, :1], original.definition_projection)
    np.testing.assert_array_equal(wider.definition_projection[:, 1:], 0.)
    np.testing.assert_allclose(original.query_projection.T @ wider.query_projection[:, 1:], 0., atol=1e-7)
    for reference in np.eye(3):
        for definition in np.eye(3):
            assert wider.score(reference, definition) == pytest.approx(original.score(reference, definition), abs=1e-7)
    bank = RelationEvidenceBank(np.eye(3)[1], np.eye(3), np.zeros(3))
    _, _, gradient = bank.score_gradient(1, wider.query_projection, wider.definition_projection)
    assert np.linalg.norm(gradient[:, 1:]) > 0
    np.testing.assert_array_equal(wider.query_projection, original.expanded_rank(3, seed=7).query_projection)


@pytest.mark.parametrize("rank,seed", [(1, 0), (4, 0), (True, 0), (2, -1), (2, True)])
def test_invalid_rank_or_seed_is_rejected(rank, seed):
    with pytest.raises(ValueError, match="larger supported rank"):
        head().expanded_rank(rank, seed=seed)


def test_widened_existing_optimizer_can_learn_three_distinct_definition_choices():
    original = head()
    wider = original.expanded_rank(3, seed=7)
    banks = [RelationEvidenceBank(reference, np.eye(3), np.zeros(3)) for reference in np.eye(3)]
    rows = tuple(RelationGraphContrast(((bank, i),), ((bank, j),), 0.)
                 for i, bank in enumerate(banks) for j in range(3) if i != j)
    _, receipt = _fit_graph_parameters((wider.query_projection, wider.definition_projection), rows,
                                       steps=40, adaptive_step=True)
    assert receipt["status"] == "retained_constraints_satisfied"
    assert receipt["retained_positive_regressions"] == 0
    assert min(receipt["stored_margins"]) >= .1


def test_expansion_roundtrips_as_a_new_candidate_without_changing_parent():
    from tests.test_semantic_relation_graph_learning import model_examples
    from core.learning.semantic_program_compositional_transducer import compositional_semantic_program_transducer_from_dict

    model, _ = model_examples()
    rank = model.definition_relation_head.query_projection.shape[1]
    previous = model.receipt_sha256
    candidate = model.with_expanded_relation_rank(rank + 1, seed=11)
    assert model.receipt_sha256 == previous != candidate.receipt_sha256
    receipt = candidate.training_receipt["relation_rank_expansions"][-1]
    assert receipt["parent_transducer_receipt_sha256"] == previous
    assert not receipt["serving_authority"] and receipt["numerical_replay_required"]
    restored = compositional_semantic_program_transducer_from_dict(candidate.to_dict())
    assert restored.receipt_sha256 == candidate.receipt_sha256
    assert restored.definition_relation_head.query_projection.shape[1] == rank + 1
    assert candidate.training_receipt["relation_tissue_fit"] == model.training_receipt["relation_tissue_fit"]
    expanded_again = candidate.with_expanded_relation_rank(rank + 2, seed=12)
    assert len(expanded_again.training_receipt["relation_rank_expansions"]) == 2
    assert compositional_semantic_program_transducer_from_dict(expanded_again.to_dict()).receipt_sha256 == expanded_again.receipt_sha256


@pytest.mark.parametrize("field,value", [("rank", 1), ("previous_rank", 0), ("seed", True),
    ("serving_authority", True), ("numerical_replay_required", False),
    ("parent_transducer_receipt_sha256", "unknown")])
def test_invalid_expansion_lineage_cannot_roundtrip(field, value):
    from tests.test_semantic_relation_graph_learning import model_examples
    from core.learning.semantic_program_compositional_transducer import compositional_semantic_program_transducer_from_dict
    from core.learning.semantic_program_campaign import _sha

    model, _ = model_examples()
    candidate = model.with_expanded_relation_rank(model.definition_relation_head.query_projection.shape[1] + 1)
    payload = candidate.to_dict()
    receipt = payload["training_receipt"]
    receipt["relation_rank_expansions"][-1][field] = value
    receipt["receipt_sha256"] = _sha({k: v for k, v in receipt.items() if k != "receipt_sha256"})
    with pytest.raises(ValueError, match="envelope"):
        compositional_semantic_program_transducer_from_dict(payload)


def test_expansion_cli_requires_a_graph_fit(monkeypatch, tmp_path):
    from tools.refit_semantic_argument_proposals import main

    monkeypatch.setattr("sys.argv", ["refit", "--transducer", "unused", "--source-report", "unused",
        "--bundle", "unused", "--output", str(tmp_path / "candidate.json"), "--relation-rank", "32"])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
