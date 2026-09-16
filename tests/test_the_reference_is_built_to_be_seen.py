"""The battery's positive control, and the properties it is built out of.

`beats_every_null` is "no null passes the conjunction and the reference does",
so a reference that fails a line the battery asks about blocks the criterion
whatever Aura does. The recurrent reference had no designed interaction, and
whether the synergy lines passed on it was a matter of the seed.

These pin what the rebuilt reference is made of: both sources of every declared
triple are their target's parents, level in the information they carry about
it; one bounded bilinear product carries the interaction; no target drives its
own sources; and the wiring is one strongly connected system held at a stated
spectral radius. The slow lines — the synergy suite and irreducibility on a
recording — are measured in the campaign's own null table, not here.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.subject.nulls import (
    REFERENCE_CHANNELS,
    REFERENCE_DOSE,
    REFERENCE_RADIUS,
    _as_matrix,
    _domain_slices,
    architecture,
    predicted_synergy,
    toy_recording,
)
from core.subject.state import DOMAINS
from core.subject.synergy import TRIPLES, synergy_suite

pytestmark = pytest.mark.unit

SEEDS = (1, 3, 7)


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("of", ["change", "level"])
def test_both_sources_carry_the_same_information_about_the_target(seed: int, of: str) -> None:
    """The fraction the line reads is a ratio, so what it needs is two level sources."""
    lines = predicted_synergy(architecture("recurrent", seed=seed), of=of)
    for triple, row in lines.items():
        larger = max(row["mi_a"], row["mi_b"])
        smaller = min(row["mi_a"], row["mi_b"])
        assert smaller > 0.0, triple
        # Level enough that the fraction is about two sources rather than one.
        # Each round of the solve moves a path a quarter of the way and is
        # clipped, so a seed can end a little off level; what the line reads is
        # the fraction below.
        assert larger / smaller < 2.5, (triple, of, row)
        # Two level sources read near a half; the line's floor is a tenth.
        assert row["fraction"] >= 0.20, (triple, of, row)


@pytest.mark.parametrize("seed", SEEDS)
def test_a_target_does_not_drive_its_own_sources(seed: int) -> None:
    """A path back puts the product into both sources and they stop being two."""
    system = architecture("recurrent", seed=seed)
    for source_a, source_b, target in TRIPLES:
        assert target not in system.coupling[source_a], (source_a, target)
        assert target not in system.coupling[source_b], (source_b, target)
        assert source_a in system.coupling[target]
        assert source_b in system.coupling[target]


@pytest.mark.parametrize("seed", SEEDS)
def test_the_wiring_is_one_system_at_the_radius_it_declares(seed: int) -> None:
    system = architecture("recurrent", seed=seed)
    radius = float(np.max(np.abs(np.linalg.eigvals(_as_matrix(system, _domain_slices(system.widths))))))
    assert radius == pytest.approx(REFERENCE_RADIUS, abs=1e-6)
    reached = {key: set(system.coupling.get(key, {})) for key in DOMAINS}
    for start in DOMAINS:
        seen, stack = {start}, [start]
        while stack:
            for other in reached[stack.pop()]:
                if other not in seen:
                    seen.add(other)
                    stack.append(other)
        assert seen == set(DOMAINS), (start, sorted(seen))


@pytest.mark.parametrize("seed", SEEDS)
def test_each_domain_speaks_in_as_many_voices_as_it_declares(seed: int) -> None:
    system = architecture("recurrent", seed=seed)
    for key in DOMAINS:
        shocks = system.channels[key]
        assert shocks.shape[1] == REFERENCE_CHANNELS
        assert np.allclose(np.linalg.norm(shocks, axis=0), 1.0)


@pytest.mark.parametrize("seed", SEEDS)
def test_the_product_is_what_the_interaction_line_reads(seed: int) -> None:
    """The control the preregistration rests on, on the conjunction as a whole.

    Not on one triple's sign. The interaction check reads a small positive gain
    where there is no interaction at all — a linear system this predictable
    gives the wider model room to be chosen on the validation rows — and over
    twelve seeds it read a positive bound on 21 of 48 triples with the product
    removed. What separates the two arms is the size and the conjunction: the
    gain's median is 0.176 with the product and 0.006 without, and all four
    triples pass on 12 seeds of 12 with it and on none without it.
    """
    system = architecture("recurrent", seed=seed)
    assert all(gain == pytest.approx(REFERENCE_DOSE) for gain in system.gains.values())

    def reads(carrying: bool) -> list:
        system.gains = {triple: (REFERENCE_DOSE if carrying else 0.0) for triple in TRIPLES}
        return synergy_suite(toy_recording(system, steps=2500, seed=seed), seed=seed, of="change")

    with_product = reads(carrying=True)
    assert all(report.passes_v3 for report in with_product), [r.as_dict() for r in with_product]
    without = reads(carrying=False)
    assert not all(report.passes_v3 for report in without)
    # And it is the interaction the product carries, not the information: the
    # fraction and the raw bar still hold on every triple with it gone.
    assert all(report.normalised >= 0.10 for report in without)
    assert all(report.synergy > report.raw_null_q99 for report in without)
    # And on every triple the product is what the gain reads: larger than the
    # same triple reads without it, with a bound the line can see.
    for carried, bare in zip(with_product, without, strict=True):
        assert carried.interaction_gain > bare.interaction_gain
        assert carried.interaction_lower_bound > 0.0
