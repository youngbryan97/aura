"""The two organism probes of 21 October read what they say, on cases with known answers.

`tools/run_paired_synergy.py` scores a joint effect by intervention, by ISC-v6's
design and bars; `tools/k_compact_sufficiency.py` moves an
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


def test_the_paired_probe_scores_by_the_v6_bars_and_keeps_no_arithmetic_of_its_own():
    """Its two-fork floor passed independent noise in every simulated dataset (review of 23 October).

    The arithmetic now lives in core/subject/interventional_synergy.py, whose known
    answers tests/test_synergy_is_read_by_setting_the_sources.py pins.
    """
    paired = _tool("run_paired_synergy")
    assert not hasattr(paired, "summarise") and not hasattr(paired, "sign_flip_p")
    source = (REPO / "tools" / "run_paired_synergy.py").read_text()
    assert "from core.subject.interventional_synergy import collect, line, summarise, target_columns" in source


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
    from core.subject.calibrated_axes import sketched_sources

    sources = sketched_sources()
    assert "organ:substrate.x" in sources and "organ:substrate.W" in sources
    assert all(source.startswith("organ:") for source in sources)


def test_an_array_mostly_at_its_minimum_still_has_room_to_move_unseen():
    """A buffer that is mostly zeros: the hidden step must leave every zero where it is."""
    from core.subject.sketch import sketch

    probe = _tool("k_compact_sufficiency")
    rng = np.random.default_rng(5)
    buffer = np.zeros(1024)
    buffer[rng.choice(1024, 200, replace=False)] = rng.uniform(0.1, 1.0, 200)
    hidden = probe.hidden_direction(buffer, rng)
    step = probe._room(buffer, hidden)
    assert step > 0.0
    plus, minus = buffer + step * hidden, buffer - step * hidden
    assert max(abs(sketch(plus)[k] - sketch(minus)[k]) for k in sketch(plus)) < 1e-9
    visible = probe.visible_direction(buffer)
    assert probe._room(buffer, visible) > 0.0
