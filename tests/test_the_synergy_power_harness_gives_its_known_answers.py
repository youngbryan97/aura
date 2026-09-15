"""The injection that measures synergy's power gives the known answers first.

To ask how strong a coupling synergy on the change can register on Aura's own
recording, a scaled coupling is added to the target's first change component and
the line is scored as it stands. That measurement means something only if the
same injection gives the preregistered known answer on a background where the
answer is known. These pin it on the ISC-v2 known-answer drift background: the
preregistered shape, a linear part plus the product of two correlated sources,
passes once it is as large as the target's own change; a pure product does not
clear the information bars at any scale, because a Gaussian copula estimate
cannot see it, while the held-out interaction check does see it.
"""

from __future__ import annotations

import importlib.util
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from core.subject.state import DOMAINS
from core.subject.synergy import _components, synergy

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[1]


def _background():
    spec = importlib.util.spec_from_file_location("known", REPO / "tests" / "test_synergy_known_answers_for_isc_v2.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module._recording("drift", 1)


def _injected(recording, *, shape: str, scale: float):
    source_a, source_b, target = DOMAINS[0], DOMAINS[3], DOMAINS[6]
    where = recording.slices[target]
    block = recording.x[:, where]
    a = _components(recording.domain(source_a)[:-1])[:, 0]
    b = _components(recording.domain(source_b)[:-1])[:, 0]
    a = (a - a.mean()) / a.std()
    b = (b - b.mean()) / b.std()
    change = np.diff(block, axis=0)
    centred = change - change.mean(axis=0)
    column_spread = np.maximum(centred.std(axis=0), 1e-9)
    _, _, vectors = np.linalg.svd(centred / column_spread, full_matrices=False)
    unit = vectors[0] * column_spread
    unit = unit / np.linalg.norm(unit)
    spread = float(np.std(centred @ unit))
    term = a * b if shape == "product" else (a - b) + a * b
    term = (term - term.mean()) / term.std()
    pushed = block.copy()
    pushed[1:] += np.cumsum(np.outer(scale * spread * term, unit), axis=0)
    x = recording.x.copy()
    x[:, where] = pushed
    return synergy(replace(recording, x=x), source_a, source_b, target, seed=7, of="change")


def test_nothing_injected_does_not_pass() -> None:
    assert not _injected(_background(), shape="known", scale=0.0).passes


@pytest.mark.parametrize("scale", (1.0, 2.0))
def test_the_preregistered_shape_passes_once_it_is_as_large_as_the_change(scale: float) -> None:
    assert _injected(_background(), shape="known", scale=scale).passes


@pytest.mark.parametrize("scale", (1.0, 2.0))
def test_a_pure_product_is_seen_by_the_interaction_check_and_not_by_the_information_bars(scale: float) -> None:
    report = _injected(_background(), shape="product", scale=scale)
    assert report.interaction_lower_bound > 0.0
    assert not report.passes
