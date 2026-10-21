"""Each direction of a cut is read on its own side's columns.

The cut arm runs each side with the other held at its anchor and composes the
two. Where the composed run departs from the untouched one on the left's
columns, holding the right took something from the left; on the right's
columns, the other way. A one-way influence decides one direction and not the
other.
"""
from __future__ import annotations

import numpy as np

from core.subject.state import DOMAINS, domain_slices
from core.subject.v25_cut import directional_decisions


def _slot(anchors: int = 48, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    width = max(s.stop for s in domain_slices().values())
    intact = rng.standard_normal((anchors, width))
    cut = intact.copy()
    # Holding the rest moved C: rest -> C carries something, C -> rest nothing.
    c = domain_slices()["C"]
    cut[:, c.start : c.start + 6] += 1.0
    return {
        "context": rng.standard_normal((anchors, 8)),
        "intact": intact,
        "cut": cut,
        "sham_a": intact,
        "sham_b": intact.copy(),
    }


def test_a_one_way_influence_decides_its_own_direction_and_not_the_other():
    rest = tuple(d for d in DOMAINS if d != "C")
    out = directional_decisions(_slot(), ("C",), rest, tau_seconds=1.0, draws=200, alpha=0.05 / 6, paired=True)
    assert out["into_left"]["decided"] is True
    assert out["into_right"]["decided"] is False
