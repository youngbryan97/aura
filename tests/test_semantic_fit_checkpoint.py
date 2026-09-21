"""Interrupted numerical search resumes the same protected inequalities."""

from dataclasses import replace

import numpy as np
import pytest

from core.learning.semantic_fit_checkpoint import SemanticFitCheckpoint, fit_identity
from core.learning.semantic_graph_constraints import fit_graph_constraints
from tests.test_semantic_graph_constraints import operation_constraint, simple_model


class InterruptedFit(RuntimeError):
    pass


def test_rejected_projection_keeps_accepted_checkpoint_separate(tmp_path):
    import json
    store = SemanticFitCheckpoint(tmp_path / "fit.npz", "source-fit")
    state = dict(flat=np.ones(2), margins=np.ones(1), floors=np.array([-np.inf]),
                 trace=[], next_step=0, status="running")
    store.save(**state)
    before = store.path.read_bytes()
    path = store.save_projection(normals=[[1., 0.], [-1., 0.]], required=[1., 1.],
        anchor=[0., 0.], receipt={"status": "unverified"}, step=1)
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(archive["metadata"].tobytes())
        arrays = {name: archive[name] for name in ("normals", "required", "anchor")}
    body = {key: value for key, value in metadata.items() if key != "sha256"}
    assert metadata["sha256"] == fit_identity((body, arrays))
    assert metadata["identity"] == "source-fit"
    assert not metadata["serving_authority"]
    assert store.path.read_bytes() == before
    np.testing.assert_array_equal(arrays["normals"], [[1., 0.], [-1., 0.]])


def test_unresolved_fit_exports_its_actual_affine_problem(tmp_path):
    import json
    from core.learning.semantic_graph_constraints import _fit_graph_parameters
    from tests.test_semantic_minimum_change import linear
    path = tmp_path / "fit.npz"
    parameters = (np.zeros((1, 1)), np.zeros((1, 1)), np.zeros(1), np.array(0.))
    _, receipt = _fit_graph_parameters(parameters, (linear([1.]), linear([-1.])),
        steps=1, update_rule="minimum_change", checkpoint_path=path)
    assert receipt["status"] == "local_margin_projection_unverified"
    with np.load(path.with_name(path.name + ".projection.npz"), allow_pickle=False) as data:
        metadata = json.loads(data["metadata"].tobytes())
        np.testing.assert_array_equal(data["required"], [.1, .1])
        assert data["normals"].shape == (2, 4)
        assert not metadata["receipt"]["stored_primal_feasible"]
    assert receipt["accepted_steps"] == []


@pytest.mark.parametrize("objective", ["squared_deficit", "pairwise_logistic"])
def test_interrupted_fit_resumes_bit_identically_without_repeating_updates(tmp_path, objective):
    head, operation = simple_model()
    rows = (operation_constraint([1., 0.]), operation_constraint([0., 1.]))
    options = dict(steps=30, learning_rate=.01, required_margin=.5, objective=objective)
    expected = fit_graph_constraints(head, operation, rows, **options)
    path = tmp_path / "fit.npz"
    seen = []
    def interrupt(row):
        seen.append(row)
        if row["stage"] == "constraint_fit_step" and row["completed"] == 3:
            raise InterruptedFit
    with pytest.raises(InterruptedFit):
        fit_graph_constraints(head, operation, rows, checkpoint_path=path, progress=interrupt, **options)
    assert path.exists()
    resumed = []
    actual = fit_graph_constraints(head, operation, rows, checkpoint_path=path, progress=resumed.append, **options)
    assert resumed[0]["stage"] == "constraint_fit_resumed"
    assert resumed[0]["completed"] == 3
    assert all(row["completed"] > 3 for row in resumed[1:])
    assert expected[2] == actual[2]
    for lhs, rhs in zip(expected[1].heads, actual[1].heads, strict=True):
        np.testing.assert_array_equal(lhs.weight, rhs.weight)
        np.testing.assert_array_equal(lhs.bias, rhs.bias)
    finished = []
    again = fit_graph_constraints(head, operation, rows, checkpoint_path=path, progress=finished.append, **options)
    assert len(finished) == 1
    assert again[2] == expected[2]


@pytest.mark.parametrize("change", ["initial", "evidence", "options", "owner"])
def test_checkpoint_cannot_cross_a_fit_identity(tmp_path, change):
    head, operation = simple_model()
    rows = (operation_constraint([1., 0.]), operation_constraint([0., 1.]))
    path = tmp_path / "fit.npz"
    options = dict(steps=2, checkpoint_path=path, checkpoint_identity="candidate-a")
    fit_graph_constraints(head, operation, rows, **options)
    if change == "initial":
        operation = replace(operation, heads=(replace(operation.heads[0], bias=np.ones(2)),))
    elif change == "evidence":
        rows = (replace(rows[0], weight=2.), rows[1])
    elif change == "options":
        options["steps"] = 3
    else:
        options["checkpoint_identity"] = "candidate-b"
    with pytest.raises(ValueError, match="identity or checksum"):
        fit_graph_constraints(head, operation, rows, **options)


def test_numerical_identity_binds_shapes_types_and_contents():
    value = np.array([[1., 2.]], dtype=np.float64)
    assert fit_identity(value) == fit_identity(value.copy())
    assert fit_identity(value) != fit_identity(value.reshape(2))
    assert fit_identity(value) != fit_identity(value.astype(np.float32))
    assert fit_identity(value) != fit_identity(value + 1)
    with pytest.raises(ValueError, match="object-valued"):
        fit_identity(np.array([object()]))


def test_checkpoint_remains_atomic_on_write_failure(tmp_path, monkeypatch):
    from core.runtime.file_write_gateway import get_file_write_gateway
    path = tmp_path / "state.npz"
    store = SemanticFitCheckpoint(path, "fit-a")
    state = dict(flat=np.ones(2), margins=np.ones(1), floors=np.array([-np.inf]),
                 trace=[], next_step=0, status="running")
    store.save(**state)
    original = path.read_bytes()
    def fail(*args, **kwargs):
        raise OSError("disk unavailable")
    monkeypatch.setattr(get_file_write_gateway(), "write_bytes", fail)
    with pytest.raises(OSError, match="disk unavailable"):
        store.save(**{**state, "flat": np.zeros(2)})
    assert path.read_bytes() == original
    assert store.load()["flat"].tolist() == [1., 1.]
    with pytest.raises(ValueError, match="identity or checksum"):
        SemanticFitCheckpoint(path, "fit-b").load()


@pytest.mark.parametrize("change", ["margin", "floor", "progress"])
def test_resume_rechecks_numerical_state_even_with_a_valid_checksum(tmp_path, change):
    from io import BytesIO
    import json
    head, operation = simple_model()
    rows = (operation_constraint([1., 0.]), operation_constraint([0., 1.]))
    path = tmp_path / "fit.npz"
    fit_graph_constraints(head, operation, rows, steps=2, checkpoint_path=path)
    with np.load(BytesIO(path.read_bytes()), allow_pickle=False) as archive:
        identity = json.loads(archive["metadata"].tobytes())["identity"]
    store = SemanticFitCheckpoint(path, identity)
    state = store.load()
    if change == "margin":
        state["margins"] += 1
    elif change == "floor":
        state["floors"][:] = -np.inf
    else:
        state["next_step"] += 1
    store.save(**{key: state[key] for key in ("flat", "margins", "floors", "trace", "next_step", "status")})
    with pytest.raises(ValueError, match="fit checkpoint"):
        fit_graph_constraints(head, operation, rows, steps=2, checkpoint_path=path)


def test_production_joint_refit_wires_round_checkpoints(tmp_path):
    import hashlib
    import json
    from core.learning.semantic_program_compositional_transducer import compositional_semantic_program_transducer_from_dict
    from core.learning.semantic_joint_graph_learning import refit_compositional_joint_graphs
    from tests.test_semantic_relation_graph_learning import model_examples
    model, examples = model_examples()
    model = model.with_joint_operation_argument_scores().with_source_ordered_definitions()
    progress = []
    options = dict(rounds=1, steps=2, constraint_learning=True, checkpoint_dir=tmp_path)
    first = refit_compositional_joint_graphs(model, examples, progress=progress.append, **options)
    assert (tmp_path / "round-1.npz").exists()
    saved_path = tmp_path / "round-1.candidate.json"
    saved_bytes = saved_path.read_bytes()
    saved = json.loads(saved_bytes)
    assert saved['sha256'] == fit_identity({k: v for k, v in saved.items() if k != 'sha256'})
    assert saved['numerical_checkpoint_sha256'] == hashlib.sha256((tmp_path / "round-1.npz").read_bytes()).hexdigest()
    assert saved['parent'] == model.receipt_sha256 and saved['round'] == 1
    assert not saved['serving_authority'] and not saved['validation_used_for_selection']
    restored = compositional_semantic_program_transducer_from_dict(saved['candidate'])
    assert fit_identity(restored._coefficient_body()) == fit_identity(first._coefficient_body())
    resumed = []
    second = refit_compositional_joint_graphs(model, examples, progress=resumed.append, **options)
    assert first.receipt_sha256 == second.receipt_sha256
    assert any(row["stage"] == "constraint_fit_resumed" and row["round"] == 1 for row in resumed)
    assert not any(row["stage"] == "constraint_fit_step" for row in resumed)
    assert saved_path.read_bytes() == saved_bytes
    from core.learning.semantic_fit_checkpoint import load_round_candidate
    options = dict(expected_parent=model.receipt_sha256, expected_round=1,
                   numerical_checkpoint=tmp_path / "round-1.npz")
    verified = load_round_candidate(saved_path, **options)
    assert verified.receipt_sha256 == restored.receipt_sha256
    from tools.refit_semantic_argument_proposals import load_evaluation_candidate
    assert load_evaluation_candidate(saved_path, model, round_index=1,
        numerical_checkpoint=tmp_path / "round-1.npz").receipt_sha256 == restored.receipt_sha256
    with pytest.raises(ValueError, match="requires its numerical checkpoint"):
        load_evaluation_candidate(saved_path, model, round_index=1)
    for change in (dict(expected_parent="another-parent"), dict(expected_round=2), dict(expected_round=True)):
        with pytest.raises(ValueError, match="identity or checkpoint"):
            load_round_candidate(saved_path, **{**options, **change})
    for field, value in (("serving_authority", True), ("validation_used_for_selection", True), ("round", True)):
        altered = {**saved, field: value}
        altered['sha256'] = fit_identity({k: v for k, v in altered.items() if k != 'sha256'})
        invalid = tmp_path / f"invalid-{field}.json"
        invalid.write_text(json.dumps(altered))
        with pytest.raises(ValueError, match="identity or checkpoint"):
            load_round_candidate(invalid, **options)
    (tmp_path / "round-1.npz").write_bytes(b"different numerical state")
    with pytest.raises(ValueError, match="identity or checkpoint"):
        load_round_candidate(saved_path, **options)


def test_round_candidate_survives_interruption_before_next_round(tmp_path):
    import json
    from core.learning.semantic_joint_graph_learning import refit_compositional_joint_graphs
    from core.learning.semantic_program_compositional_transducer import compositional_semantic_program_transducer_from_dict
    from tests.test_semantic_relation_graph_learning import model_examples

    model, examples = model_examples()
    model = model.with_joint_operation_argument_scores().with_source_ordered_definitions()
    def interrupt(row):
        if row['stage'] == 'joint_graph_fit':
            raise InterruptedFit
    with pytest.raises(InterruptedFit):
        refit_compositional_joint_graphs(model, examples, rounds=2, steps=2,
            constraint_learning=True, checkpoint_dir=tmp_path, progress=interrupt)
    path = tmp_path / 'round-1.candidate.json'
    saved = json.loads(path.read_bytes())
    restored = compositional_semantic_program_transducer_from_dict(saved['candidate'])
    assert restored.model_basis_sha256 == model.model_basis_sha256
    assert not (tmp_path / 'round-2.candidate.json').exists()
    path.chmod(0o600)
    path.write_text('{}')
    with pytest.raises(ValueError, match='saved round candidate differs'):
        refit_compositional_joint_graphs(model, examples, rounds=2, steps=2,
            constraint_learning=True, checkpoint_dir=tmp_path)
    assert path.read_text() == '{}'
