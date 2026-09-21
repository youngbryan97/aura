"""Persist exactly the constraints used by the fitter, without executable payloads."""

import numpy as np
import pytest

from core.learning.semantic_fit_checkpoint import fit_identity
from core.learning.semantic_fit_problem import load_fit_problem, save_fit_problem
from core.learning.semantic_relation_graph_learning import graph_margin_gradient
from tests.test_semantic_graph_batch import problem


def test_roundtrip_preserves_scores_gradients_and_shared_evidence(tmp_path):
    initial, contrasts = problem()
    path = tmp_path / "problem.npz"
    identity = fit_identity((initial, contrasts))
    save_fit_problem(path, identity=identity, initial=initial, contrasts=contrasts, options={"scale": 1.7})
    restored = load_fit_problem(path, expected_identity=identity)
    assert fit_identity((restored["initial"], restored["contrasts"])) == identity
    assert restored["contrasts"][0].positive_operations[0][0] is restored["contrasts"][5].positive_operations[0][0]
    for original, replay in zip(contrasts, restored["contrasts"], strict=True):
        before, gradients = graph_margin_gradient(initial, original, scale=1.7)
        after, replay_gradients = graph_margin_gradient(restored["initial"], replay, **restored["options"])
        assert before == after
        for a, b in zip(gradients, replay_gradients, strict=True):
            np.testing.assert_array_equal(a, b)
    with pytest.raises(ValueError, match="identity"):
        load_fit_problem(path, expected_identity="different")


def test_fit_saves_inputs_before_first_proposal(tmp_path):
    from core.learning.semantic_graph_constraints import _fit_graph_parameters
    from tests.test_semantic_minimum_change import linear

    initial = (np.zeros((1, 1)), np.zeros((1, 1)), np.zeros(1), np.array(0.))
    path = tmp_path / "fit.npz"
    _, receipt = _fit_graph_parameters(initial, (linear([1.], -.5),),
        steps=1, checkpoint_path=path, update_rule="minimum_change")
    from core.learning.semantic_fit_checkpoint import SemanticFitCheckpoint
    import json

    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(archive["metadata"].tobytes())
    problem = load_fit_problem(path.with_suffix(".problem.npz"), expected_identity=metadata["identity"])
    _, replay = _fit_graph_parameters(problem["initial"], problem["contrasts"], **problem["options"])
    assert replay["stored_margins"] == receipt["stored_margins"]
    assert SemanticFitCheckpoint(path, metadata["identity"]).load()["next_step"] == 1
    previous = path.with_suffix(".problem.npz").read_bytes()
    with pytest.raises(ValueError, match="identity"):
        _fit_graph_parameters(initial, (linear([1.], -.6),),
            steps=1, checkpoint_path=path, update_rule="minimum_change")
    assert path.with_suffix(".problem.npz").read_bytes() == previous


def test_evidence_mutation_is_detected(tmp_path):
    from io import BytesIO

    initial, contrasts = problem()
    path = tmp_path / "problem.npz"
    save_fit_problem(path, identity="parent", initial=initial, contrasts=contrasts, options={})
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    arrays["array_0"] = arrays["array_0"] + 1
    output = BytesIO()
    np.savez_compressed(output, **arrays)
    path.write_bytes(output.getvalue())
    with pytest.raises(ValueError, match="checksum"):
        load_fit_problem(path, expected_identity="parent")


def test_problem_survives_an_interrupted_first_projection(tmp_path, monkeypatch):
    import json
    from core.learning import margin_repair
    from core.learning.semantic_graph_constraints import _fit_graph_parameters
    from tests.test_semantic_minimum_change import linear

    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(margin_repair, "minimum_stored_margin_repair", interrupt)
    initial = (np.zeros((1, 1)), np.zeros((1, 1)), np.zeros(1), np.array(0.))
    path = tmp_path / "interrupted.npz"
    with pytest.raises(KeyboardInterrupt):
        _fit_graph_parameters(initial, (linear([1.], -.5),),
            steps=1, checkpoint_path=path, update_rule="minimum_change")
    assert not path.exists()
    problem_path = path.with_suffix(".problem.npz")
    with np.load(problem_path, allow_pickle=False) as archive:
        identity = json.loads(archive["metadata"].tobytes())["identity"]
    restored = load_fit_problem(problem_path, expected_identity=identity)
    assert len(restored["contrasts"]) == 1
    assert restored["options"]["update_rule"] == "minimum_change"
