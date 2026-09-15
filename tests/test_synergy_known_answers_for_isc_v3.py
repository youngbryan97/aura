"""The known answers ISC-v3's synergy line gives before it reads Aura.

docs/ISC_V3_PREREGISTRATION.md keeps v2's synergy line on the change and
replaces the fraction's two bars against the shifted null with the raw synergy
against a bootstrap null, which simulates the sources' own dynamics without the
target. v2 could not register a coupling of two spreads on sources as slow as
Aura's. On synthetic recordings only, these pin that v3 keeps v2's three known
answers, fails slow sources that carry no coupling and slow sources coupled
additively at four spreads, and registers the preregistered coupling at two
spreads where v2 does not.
"""

from __future__ import annotations

import importlib.util
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from core.subject import synergy as synergy_module
from core.subject.recording import Recording
from core.subject.state import DOMAINS
from core.subject.synergy import _components, synergy

pytestmark = pytest.mark.unit

#: A 300-round run's length, one row per turn.
ROWS = 2400
WIDTH = 3
#: Near the middle of Aura's source domains, whose lag-one autocorrelations run
#: from 0.98 to 0.998.
PERSISTENCE = 0.99
SEEDS = (0, 1, 2)
SOURCE_A, SOURCE_B, TARGET = DOMAINS[2], DOMAINS[5], DOMAINS[3]
V3_FIELDS = {
    "bootstrap_q99", "bootstrap_spread", "bootstrap_draws", "bootstrap_radius", "margin_over_bootstrap", "passes_v3",
}


def _battery_shape(kind: str, seed: int) -> Recording:
    path = Path(__file__).with_name("test_synergy_known_answers_for_isc_v2.py")
    spec = importlib.util.spec_from_file_location("isc_v2_known_answers", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module._recording(kind, seed)


def _background(persistence: float, seed: int) -> Recording:
    rng = np.random.default_rng(seed)
    x = np.zeros((ROWS, WIDTH * len(DOMAINS)))
    noise = rng.normal(size=x.shape)
    for row in range(1, ROWS):
        x[row] = persistence * x[row - 1] + noise[row]
    slices = {key: slice(index * WIDTH, (index + 1) * WIDTH) for index, key in enumerate(DOMAINS)}
    return Recording(
        x=x,
        conditions=tuple("x" for _ in range(ROWS)),
        tags=tuple("" for _ in range(ROWS)),
        times=np.arange(ROWS, dtype=np.float64),
        env=rng.normal(size=(ROWS, 2)),
        env_names=("e0", "e1"),
        columns=tuple(f"{key}.{index}" for key in DOMAINS for index in range(WIDTH)),
        slices=slices,
        notes={},
    )


def _coupled(recording: Recording, scale: float, shape: str) -> Recording:
    """The sources added to the target's first change component, in that component's spread."""
    where = recording.slices[TARGET]
    block = recording.x[:, where]
    a = _components(recording.domain(SOURCE_A)[:-1])[:, 0]
    b = _components(recording.domain(SOURCE_B)[:-1])[:, 0]
    a, b = (a - a.mean()) / a.std(), (b - b.mean()) / b.std()
    change = np.diff(block, axis=0)
    centred = change - change.mean(axis=0)
    column_spread = np.maximum(centred.std(axis=0), 1e-9)
    _, _, vectors = np.linalg.svd(centred / column_spread, full_matrices=False)
    unit = vectors[0] * column_spread
    unit /= np.linalg.norm(unit)
    spread = float(np.std(centred @ unit))
    term = {"preregistered": (a - b) + a * b, "additive": a - b}[shape]
    term = (term - term.mean()) / term.std()
    pushed = block.copy()
    pushed[1:] += np.cumsum(np.outer(scale * spread * term, unit), axis=0)
    x = recording.x.copy()
    x[:, where] = pushed
    return replace(recording, x=x)


def _score(recording: Recording, seed: int):
    return synergy(recording, SOURCE_A, SOURCE_B, TARGET, seed=seed, of="change")


@pytest.mark.parametrize("seed", (1, 2, 3))
def test_v2s_known_answers_hold_under_v3(seed: int) -> None:
    def score(kind: str):
        return synergy(_battery_shape(kind, seed), DOMAINS[0], DOMAINS[3], DOMAINS[6], seed=seed, of="change")

    assert score("product").passes_v3
    assert not score("additive").passes_v3
    assert not score("drift").passes_v3


@pytest.mark.parametrize("seed", SEEDS)
def test_slow_sources_with_no_coupling_fail(seed: int) -> None:
    report = _score(_background(PERSISTENCE, 700 + seed), seed)
    assert not report.passes_v3, report.as_dict()


@pytest.mark.parametrize("seed", SEEDS)
def test_a_coupling_of_two_spreads_on_slow_sources_passes_v3_where_v2_cannot_see_it(seed: int) -> None:
    report = _score(_coupled(_background(PERSISTENCE, 800 + seed), 2.0, "preregistered"), seed)
    assert report.passes_v3, report.as_dict()
    assert not report.passes, report.as_dict()


@pytest.mark.parametrize("seed", SEEDS)
def test_an_additive_coupling_of_four_spreads_on_slow_sources_fails(seed: int) -> None:
    report = _score(_coupled(_background(PERSISTENCE, 800 + seed), 4.0, "additive"), seed)
    assert not report.passes_v3, report.as_dict()


def test_a_level_reading_carries_no_bootstrap_and_cannot_pass_v3() -> None:
    report = synergy(_battery_shape("product", 1), DOMAINS[0], DOMAINS[3], DOMAINS[6], seed=1, of="level")
    assert report.bootstrap_draws == 0
    assert not report.passes_v3


def test_the_bootstrap_leaves_the_v2_reading_as_it_was(monkeypatch: pytest.MonkeyPatch) -> None:
    """Its draws come from a stream of their own, so the shifted null reads what v2 read."""
    recording = _battery_shape("product", 2)
    with_bootstrap = synergy(recording, DOMAINS[0], DOMAINS[3], DOMAINS[6], seed=2, of="change").as_dict()
    monkeypatch.setattr(synergy_module, "_bootstrap_raw_nulls", lambda *args, **kwargs: (np.empty(0), 0.0))
    without = synergy(recording, DOMAINS[0], DOMAINS[3], DOMAINS[6], seed=2, of="change").as_dict()
    assert with_bootstrap["bootstrap_draws"] == synergy_module.BOOTSTRAP_DRAWS
    assert without["passes_v3"] is False
    assert {k: v for k, v in with_bootstrap.items() if k not in V3_FIELDS} == {
        k: v for k, v in without.items() if k not in V3_FIELDS
    }
