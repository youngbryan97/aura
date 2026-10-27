"""The content run's internal geometry is averaged over fold assignments.

One assignment made a run's "moves together" a draw: twelve fold seeds read the
seed 31 control at 0.151 to 0.491 (27 October). Repeated cross-fitting keeps
what is estimated and averages away which rows shared a fold.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from core.subject.content_runtime import ClassSamples, internal_geometry, repeated_internal_geometry

pytestmark = pytest.mark.unit


def _samples(seed: int) -> tuple[dict[str, ClassSamples], list[SimpleNamespace]]:
    rng = np.random.default_rng(seed)
    classes = [SimpleNamespace(name=f"c{i}") for i in range(4)]
    out = {}
    for i, cls in enumerate(classes):
        keys = [(a, c) for a in range(12) for c in range(2)]
        futures = [rng.normal(size=6) + i * 0.3 for _ in keys]
        out[cls.name] = ClassSamples(
            name=cls.name, futures=futures, retrieved=[frozenset() for _ in keys], context=[],
            futures_b=[f + rng.normal(scale=0.1, size=6) for f in futures], keys=keys,
        )
    return out, classes


def test_one_repeat_is_one_fold_assignment() -> None:
    samples, classes = _samples(1)
    once, _ = repeated_internal_geometry(samples, classes, seed=7, paired=True, with_context=False, repeats=1)
    direct, _ = internal_geometry(samples, classes, seed=7000, paired=True, with_context=False)
    assert once == pytest.approx(direct)


def test_averaging_narrows_what_the_fold_assignment_decides() -> None:
    samples, classes = _samples(2)
    key = (0, 1)
    single = [internal_geometry(samples, classes, seed=s, paired=True, with_context=False)[0][key] for s in range(8)]
    averaged = [
        repeated_internal_geometry(samples, classes, seed=s, paired=True, with_context=False, repeats=8)[0][key]
        for s in range(8)
    ]
    assert np.std(averaged) < np.std(single)
