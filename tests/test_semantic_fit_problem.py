"""Persist exactly the constraints used by the fitter, without executable payloads."""

import numpy as np
import pytest

from core.learning.semantic_fit_checkpoint import fit_identity
from core.learning.semantic_fit_problem import load_fit_problem, save_fit_problem
from core.learning.semantic_relation_graph_learning import graph_margin_gradient
from tests.test_semantic_graph_batch import problem


def test_equal_arrays_share_archive_storage_without_changing_aliases(tmp_path):
    shared = np.arange(100, dtype=np.float64)
    path = tmp_path / "deduplicated.npz"
    save_fit_problem(path, identity="equal-evidence",
        initial=(shared, shared, shared.copy()), contrasts=(), options={})
    with np.load(path, allow_pickle=False) as archive:
        assert set(archive.files) == {"metadata", "array_0"}
    restored = load_fit_problem(path, expected_identity="equal-evidence")["initial"]
    assert restored[0] is restored[1]
    assert restored[0] is not restored[2]
    for value in restored:
        np.testing.assert_array_equal(value, shared)
    restored[0][0] = -1
    assert restored[1][0] == -1
    assert restored[2][0] == 0


def test_single_array_node_keeps_loaded_storage(tmp_path, monkeypatch):
    from core.learning import semantic_fit_problem

    path = tmp_path / "single.npz"
    save_fit_problem(path, identity="single", initial=(np.arange(100),), contrasts=(), options={})
    original_load = np.load
    loaded = {}

    class ObservedArchive:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.archive.close()

        def __init__(self, *args, **kwargs):
            self.archive = original_load(*args, **kwargs)
            self.files = self.archive.files

        def __getitem__(self, name):
            value = self.archive[name]
            if name != "metadata":
                loaded[name] = value
            return value

    monkeypatch.setattr(semantic_fit_problem.np, "load", ObservedArchive)
    restored = load_fit_problem(path, expected_identity="single")["initial"]
    assert restored[0] is loaded["array_0"]


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
