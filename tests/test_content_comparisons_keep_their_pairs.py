"""Missing presentations and equal distances cannot invent content structure."""
from types import SimpleNamespace

import pytest

from core.subject.content_runtime import ClassSamples, behavioural_geometry
from tools.run_subject_core_content import _relations_kept


def samples(name, keys, recalled):
    return ClassSamples(name, [], [frozenset([s]) for s in recalled], [], keys=keys)


def compare(left, right):
    return behavioural_geometry(
        {"a": left, "b": right}, [SimpleNamespace(name="a"), SimpleNamespace(name="b")]
    )[0]


def test_missing_presentations_match_on_anchor_and_condition():
    a = samples("a", [(0, 0), (1, 0), (2, 1)], ["zero", "one", "two"])
    b = samples("b", [(2, 1), (1, 0)], ["two", "one"])
    assert compare(a, b) == {(0, 1): 0.0}


def test_disjoint_presentations_are_unmeasured():
    assert compare(samples("a", [(0, 0)], ["x"]), samples("b", [(0, 1)], ["x"])) == {}


@pytest.mark.parametrize("keys", [[(0, 0)], [(0, 0), (0, 0)]])
def test_malformed_anchor_keys_cannot_be_scored(keys):
    with pytest.raises(ValueError, match="unique anchor"):
        compare(samples("a", keys, ["x", "y"]), samples("b", [(0, 0)], ["x"]))


def test_old_samples_with_unknown_missing_rows_are_unmeasured():
    assert compare(samples("a", [], ["x", "y"]), samples("b", [], ["y"])) == {}


def test_flat_geometry_has_no_measured_rank_correlation():
    flat = {k: 0.5 for k in range(6)}
    assert _relations_kept(flat, flat) == {"measured": False, "pairs": 6}


def test_ties_have_equal_ranks():
    before = dict(enumerate([0., 0., 1., 1.]))
    after = dict(enumerate([0., 1., 0., 1.]))
    assert _relations_kept(before, after)["rank_correlation"] == 0.0


def test_nonfinite_distance_is_unmeasured():
    before = dict(enumerate([0., 1., float("nan")]))
    assert not _relations_kept(before, before)["measured"]
