"""Adding a counter to a domain does not change what synergy says about it.

ISC-v5 reads synergy with the counters out of every domain
(core.subject.synergy.without_clocks). The known answers are an invariance:
on the ISC-v2 background, with running totals appended to every domain, the
clocks-out reading gives the verdicts the clean background gives. A product
passes, the same sources acting additively fail, and drift alone fails.
"""

from __future__ import annotations

import importlib.util
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from core.subject.state import DOMAINS
from core.subject.synergy import synergy, without_clocks

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[1]
SEEDS = (1, 2, 3)
CLOCKS_PER_DOMAIN = 3


def _v2_background():
    spec = importlib.util.spec_from_file_location("known_v2", REPO / "tests" / "test_synergy_known_answers_for_isc_v2.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module._recording


def _with_clocks(recording, seed: int):
    """Each domain's columns, then three running totals of its own."""
    rng = np.random.default_rng([seed, 5])
    rows = recording.x.shape[0]
    blocks, slices, start = [], {}, 0
    for key in DOMAINS:
        own = recording.x[:, recording.slices[key]]
        # Counters the way hers move: whole steps, taken some turns and not others.
        clocks = np.cumsum(rng.poisson(rng.uniform(0.2, 2.0, CLOCKS_PER_DOMAIN), size=(rows, CLOCKS_PER_DOMAIN)), axis=0)
        block = np.hstack([own, clocks.astype(np.float64)])
        blocks.append(block)
        slices[key] = slice(start, start + block.shape[1])
        start += block.shape[1]
    x = np.hstack(blocks)
    return replace(recording, x=x, slices=slices, columns=tuple(f"c{i}" for i in range(x.shape[1])))


@pytest.fixture(scope="module")
def backgrounds():
    make = _v2_background()
    return {(kind, seed): make(kind, seed) for kind in ("product", "additive", "drift") for seed in SEEDS}


def _passes(recording, seed: int, *, clocks_out: bool) -> bool:
    return synergy(recording, DOMAINS[0], DOMAINS[3], DOMAINS[6], seed=seed, of="change", clocks_out=clocks_out).passes_v3


def test_the_clocks_added_here_are_the_ones_the_recording_calls_clocks(backgrounds) -> None:
    clean = backgrounds[("product", 1)]
    clocked = _with_clocks(clean, 1)
    assert int(clocked.monotone_columns().sum()) == CLOCKS_PER_DOMAIN * len(DOMAINS)
    assert np.array_equal(without_clocks(clean).x, clean.x)


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize(("kind", "expected"), [("product", True), ("additive", False), ("drift", False)])
def test_with_counters_in_every_domain_the_clocks_out_reading_gives_the_clean_answer(
    backgrounds, kind: str, expected: bool, seed: int
) -> None:
    clean = backgrounds[(kind, seed)]
    assert _passes(clean, seed, clocks_out=False) is expected
    assert _passes(_with_clocks(clean, seed), seed, clocks_out=True) is expected


def test_with_the_clocks_left_in_a_real_product_is_missed(backgrounds) -> None:
    """Why the line reads without them: counters cost it the product on two seeds of three."""
    found = [_passes(_with_clocks(backgrounds[("product", seed)], seed), seed, clocks_out=False) for seed in SEEDS]
    assert sum(found) < len(SEEDS)
