"""A lossy sketch can hide an internal cause; a sufficient read basis cannot."""
import numpy as np

from core.subject.estimate import fit_predict, split_rows
from core.subject.observation_basis import sufficient_basis
from core.subject.sketch import _directions, sketch


def collision(size=16):
    directions = _directions(size)
    constraints = np.vstack([np.ones(size - 2), directions[:, 1:-1]])
    _u, singular, vt = np.linalg.svd(constraints, full_matrices=True)
    rank = int((singular > 1e-12).sum())
    left, right = np.zeros(size), np.zeros(size)
    left[[0, -1]] = right[[0, -1]] = (-3., 3.)
    left[1:-1], right[1:-1] = vt[rank], vt[rank + 1]
    return left, right


def test_the_actual_eight_number_sketch_is_not_an_injective_state_reading():
    left, right = collision()
    assert np.linalg.norm(left - right) > 1.
    np.testing.assert_allclose(list(sketch(left).values()), list(sketch(right).values()), atol=1e-14)
    # An ordinary coordinate read can give different futures although every
    # recorded sketch field agrees. A passive outside copy could then predict
    # that future without causing it.
    index = int(np.argmax(np.abs(left - right)))
    assert abs(left[index] - right[index]) > .1


def test_one_added_read_direction_recovers_the_missing_transition_input():
    left, right = collision()
    index = int(np.argmax(np.abs(left - right)))
    reader = np.eye(len(left))[[index]]
    original = np.vstack([np.ones(len(left)), _directions(len(left))])
    basis = sufficient_basis(original, reader)
    assert basis.added_rank == 1
    assert basis.reconstruction_error < 1e-12
    states = np.vstack([left, right, np.random.default_rng(919).normal(size=(100, len(left)))])
    np.testing.assert_allclose(basis.reconstruct_reads(basis.observe(states)), states @ reader.T, atol=1e-12)


def test_existing_read_directions_are_not_counted_twice():
    original = np.eye(6)[:2]
    readers = np.vstack([original[0] * 1e-15, original[1] * 1e6, np.zeros(6)])
    basis = sufficient_basis(original, readers)
    assert basis.added_rank == 0
    np.testing.assert_allclose(basis.axes @ basis.axes.T, np.eye(2), atol=1e-12)


def test_a_nonlinear_transition_of_preserved_reads_is_also_preserved():
    rng = np.random.default_rng(929)
    readers = rng.normal(size=(3, 40))
    basis = sufficient_basis(np.empty((0, 40)), readers)
    states = rng.normal(size=(100, 40))

    def transition(reads):
        return np.tanh(reads[:, 0]) + reads[:, 1] * reads[:, 2]

    np.testing.assert_allclose(transition(basis.reconstruct_reads(basis.observe(states))),
                               transition(states @ readers.T), atol=1e-11)


def test_a_passive_outside_copy_looks_informative_only_because_the_sketch_lost_state():
    left, right = collision()
    rng = np.random.default_rng(939)
    states = np.stack([left, right])[rng.integers(0, 2, size=600)]
    index = int(np.argmax(np.abs(left - right)))
    target = states[:, [index]]  # the transition reads its own internal state
    passive_copy = target.copy()  # this observation is never read by the transition
    unrelated = rng.normal(size=(600, 1))
    observed = np.array([list(sketch(state).values()) for state in states])
    train, validate, test = split_rows(len(states))

    def loss(features):
        return fit_predict(features, target, train=train, validate=validate, test=test).loss

    narrow = loss(np.hstack([observed, unrelated]))
    wide = loss(np.hstack([observed, unrelated, passive_copy]))
    assert narrow > .9 and wide < .001
    original = np.vstack([np.ones(len(left)), _directions(len(left))])
    basis = sufficient_basis(original, np.eye(len(left))[[index]])
    assert loss(np.hstack([basis.observe(states), unrelated])) < .001


def test_the_registered_invariant_detects_a_broken_read_canary(monkeypatch):
    from core.subject import observation_basis
    from core.verify.runtime_invariants import _declared_linear_reads

    assert observation_basis.declared_read_canary()
    assert not list(_declared_linear_reads())
    monkeypatch.setattr(observation_basis, "declared_read_canary", lambda: False)
    assert len(list(_declared_linear_reads())) == 1


def test_full_array_digests_distinguish_the_production_sketch_collision():
    from types import SimpleNamespace

    from core.subject.declared_observations import ObservationPlan

    left, right = collision()
    plan = ObservationPlan({"version": 1, "max_bytes": 10000, "arrays": [
        {"unit": "organ:substrate", "path": ["x"], "shape": [16], "mode": "exact"},
    ]})
    a = plan.digest_units({"organ:substrate": SimpleNamespace(x=left)})
    b = plan.digest_units({"organ:substrate": SimpleNamespace(x=right)})
    assert a.keys() == b.keys() and a != b
