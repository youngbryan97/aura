"""The two organism probes of 21 October read what they say, on cases with known answers.

`tools/run_paired_synergy.py` scores a joint effect by intervention against
the floor of two identical forks; `tools/k_compact_sufficiency.py` moves an
array where the eight-number sketch cannot see it. Both run on her organism;
these pin the arithmetic they rest on.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]


def _tool(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "tools" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_an_interaction_above_the_floor_is_found_and_noise_is_not():
    paired = _tool("run_paired_synergy")
    rng = np.random.default_rng(0)
    direction = rng.standard_normal(6)
    real = [direction + rng.normal(0, 0.1, 6) for _ in range(8)]
    floor = [rng.normal(0, 0.1, 6) for _ in range(8)]
    found = paired.summarise(real, floor, seed=1)
    assert found["p_above_floor"] < 0.01 and found["p_consistent_direction"] < 0.01
    noise = paired.summarise([rng.normal(0, 0.1, 6) for _ in range(8)], floor, seed=1)
    assert noise["p_above_floor"] > 0.05


def test_a_hidden_direction_leaves_all_eight_numbers_where_they_were():
    from core.subject.sketch import sketch

    probe = _tool("k_compact_sufficiency")
    rng = np.random.default_rng(3)
    weights = rng.uniform(-1, 1, (16, 16))
    hidden = probe.hidden_direction(weights, rng)
    step = probe._room(weights, hidden)
    assert step > 0.0
    plus, minus = weights + step * hidden, weights - step * hidden
    assert max(abs(sketch(plus)[k] - sketch(minus)[k]) for k in sketch(plus)) < 1e-9
    x = rng.standard_normal(16)
    assert np.linalg.norm(np.tanh(plus @ x) - np.tanh(minus @ x)) > 1e-3
    visible = probe.visible_direction(weights)
    assert max(abs(sketch(weights + 0.1 * visible)[k] - sketch(weights)[k]) for k in sketch(weights)) > 1e-3


def test_the_probe_finds_every_array_k_reads_through_the_sketch():
    probe = _tool("k_compact_sufficiency")
    sources = probe.sketched_sources()
    assert "organ:substrate.x" in sources and "organ:substrate.W" in sources
    assert all(source.startswith("organ:") for source in sources)
