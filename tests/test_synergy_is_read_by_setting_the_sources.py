"""ISC-v6's synergy line: two sources set alone and together, from one moment.

The preregistration (docs/ISC_V6_PREREGISTRATION.md) states what each simulated
world must come out as before the line is read on her: worlds with no joint
effect, deterministic divergence among them, pass at most 3% of datasets; a
product along one direction passes at least 95%. The suite runs 200 datasets of
each; tools/synergy_v6_known_answers.py runs the full 1,000.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from core.subject.battery import Verdict, _assemble_v6
from core.subject.interventional_synergy import (
    Samples,
    direction_p,
    line,
    sample_arrays,
    summarise,
    target_columns,
)
from core.subject.synergy import TRIPLES

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def battery():
    name = "synergy_v6_known_answers_under_test"
    spec = importlib.util.spec_from_file_location(name, REPO / "tools" / "synergy_v6_known_answers.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    yield module
    sys.modules.pop(name, None)


def test_every_world_comes_out_as_stated(battery):
    result = battery.run(datasets=200, draws=2000)
    assert all(row["as_stated"] for row in result.values()), result


def test_the_direction_null_flips_whole_vectors():
    """Flipping the projections one by one called 12% of noise a direction (the review of 23 October)."""
    rng = np.random.default_rng(11)
    rejected = sum(
        direction_p(list(rng.normal(size=(48, 8))), seed=index, draws=1000) < 0.05 for index in range(300)
    )
    assert rejected / 300 <= 0.08


def test_a_shared_direction_is_found():
    rng = np.random.default_rng(12)
    direction = np.ones(8) / np.sqrt(8)
    vectors = [3.0 * direction + rng.normal(size=8) for _ in range(48)]
    assert direction_p(vectors, seed=1, draws=2000) < 0.01


def test_the_registers_are_out_of_the_target():
    columns = ["G.bid", "G.conj_attention_k0", "G.winner", "C.mean", "G.conj_attention_k1"]
    assert list(target_columns(columns, slice(0, 3))) == [0, 2]


def test_the_line_needs_every_declared_triple(battery):
    rng = np.random.default_rng(13)
    passing = summarise(battery.world_samples("deterministic_product", rng), seed=6, draws=2000)
    failing = summarise(battery.world_samples("additive", rng), seed=7, draws=2000)
    every = {f"{a},{b}->{y}": passing for a, b, y in TRIPLES}
    assert passing["passes"] and line(every)
    one_short = dict(every)
    one_short[next(iter(one_short))] = failing
    assert not line(one_short)
    assert not line({})


def test_a_run_that_did_not_set_the_sources_fails_the_v6_line():
    verdict = Verdict()
    _assemble_v6(verdict, {})
    assert [c.key for c in verdict.v6_criteria] == ["synergy"] and not verdict.v6_criteria[0].passed


def test_too_few_samples_never_pass():
    samples = Samples()
    zero = np.zeros(8)
    samples.add(untouched=[zero] * 4, first=zero + 1, second=zero + 1, both=zero + 5)
    assert not summarise(samples, seed=0)["passes"]


def test_every_sample_is_kept_with_its_columns():
    """A triple that fails is read column by column, so the probe keeps every vector."""
    samples = Samples()
    rng = np.random.default_rng(14)
    for _ in range(3):
        n1, first, second, both = (rng.normal(size=2) for _ in range(4))
        samples.add(untouched=[n1, n1, n1, n1], first=first, second=second, both=both)
    columns = ["G.bid", "G.conj_attention_k0", "G.winner"]
    arrays = sample_arrays({"A,S->G": samples}, columns, {"G": np.array([0, 2])})
    assert arrays["AS_G.interactions"].shape == (3, 2)
    assert list(arrays["AS_G.columns"]) == ["G.bid", "G.winner"]
    assert np.allclose(arrays["AS_G.interactions"], np.stack(samples.interactions))
    assert np.allclose(arrays["AS_G.firsts"][0], samples.firsts[0])
    assert "PM_W.interactions" not in arrays
