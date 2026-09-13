"""A lesion that leaks is a lesion that was not made.

The cut is built by holding one side still and letting the other run, then
taking each side's columns from the arm where it was free. That only removes
the cross-partition channels if holding a domain still actually holds it. What
"still" means is `CLAMPED_FIELDS`, and what the domain *is* is its schema — so
any source a schema reads and the clamp does not hold is a field that keeps
moving inside its own lesion, and information keeps crossing the cut.

Twenty of them were, across six domains. Interoception's effort ledger, which
is the only channel the body has while the host is frozen. Active memory's
recall scores, which the memory displacement writes. The workspace's six
modifiers, which are what attention does to everything downstream. Affect's
standing mood and its three physiological channels. Perception's objective. And
four of development's thirteen columns: the reservoir's units were held and its
step count, era and last reading were not.

Every lesion measured before this under-cut.
"""

from __future__ import annotations

import pytest

from core.subject.clamp import CLAMPED_FIELDS, RESERVOIR_FIELDS
from core.subject.state import DOMAINS, schema

pytestmark = pytest.mark.unit

#: How the reservoir's fields appear when read as schema sources.
_RESERVOIR_SOURCES = {"ontogeny.h", "ontogeny.novelty", "ontogeny.displacement"} | {
    f"ontogeny.{name}" for name in RESERVOIR_FIELDS
}


def _held(domain: str) -> set[str]:
    held = set(CLAMPED_FIELDS.get(domain, ()))
    if domain == "N":
        # Held through the runtime rather than through the state.
        held |= _RESERVOIR_SOURCES
    return held


def _state_sources(domain: str) -> set[str]:
    """Schema sources that name a state attribute, not an organ call."""
    return {
        str(source).split("[", 1)[0]
        for source in schema(domain).sources
        if not str(source).startswith("organ:")
    }


def _unheld(domain: str) -> list[str]:
    held = _held(domain)
    return sorted(
        source
        for source in _state_sources(domain)
        if source not in held
        and not any(source.startswith(f"{parent}.") for parent in held)
    )


@pytest.mark.parametrize("domain", list(DOMAINS))
def test_every_source_a_domain_reads_is_held_when_it_is_clamped(domain: str) -> None:
    loose = _unheld(domain)
    assert not loose, (
        f"clamping {domain} does not hold {loose}, so those columns keep moving "
        "inside their own lesion and information crosses the cut"
    )


def test_the_effort_ledger_is_held_with_the_body() -> None:
    """The only channel the body has while the host is frozen."""
    assert "soma.exertion" in CLAMPED_FIELDS["I"]
    assert "soma.effort" in CLAMPED_FIELDS["I"]


def test_the_recall_scores_are_held_with_memory() -> None:
    """The memory displacement writes them and the workspace prices its bid on them."""
    assert "cognition.memory_scores" in CLAMPED_FIELDS["M"]


def test_the_workspace_modifiers_are_held_with_the_workspace() -> None:
    """They are what attention does to everything downstream."""
    for name in ("temperature_mod", "depth_mod", "focus_mod", "creativity_mod"):
        assert f"cognition.modifiers.{name}" in CLAMPED_FIELDS["G"]


def test_the_standing_mood_is_held_with_affect() -> None:
    assert "affect.mood_baselines" in CLAMPED_FIELDS["A"]


def test_development_holds_more_than_its_hidden_units() -> None:
    """Four of N's thirteen columns read the counters beside the reservoir."""
    assert "steps" in RESERVOIR_FIELDS
    assert "era" in RESERVOIR_FIELDS
    assert "last_novelty" in RESERVOIR_FIELDS


def test_a_clamped_reservoir_puts_its_counters_back() -> None:
    from types import SimpleNamespace

    import numpy as np

    from core.subject.clamp import Clamp

    ontogeny = SimpleNamespace(
        h=np.ones(4), steps=10, era=2, last_novelty=0.5,
        last_displacement=0.1, last_relative_displacement=0.2,
    )
    holder = SimpleNamespace(state=SimpleNamespace(), ontogeny=ontogeny)
    clamp = Clamp(holder, ["N"])

    ontogeny.h = np.zeros(4)
    ontogeny.steps = 99
    ontogeny.era = 7
    ontogeny.last_novelty = 0.99
    clamp.apply()

    assert np.allclose(ontogeny.h, np.ones(4))
    assert ontogeny.steps == 10
    assert ontogeny.era == 2
    assert ontogeny.last_novelty == pytest.approx(0.5)


def test_an_organ_read_is_a_residual_this_cannot_hold() -> None:
    """Named rather than hidden: a clamp reaches state, not an organ's insides.

    What limits the leak is that the state the organ computes from is held, so
    its output is largely held with it. Largely is not exactly, and saying so
    is the difference between a known residual and an unknown one.
    """
    organ_reads = {
        domain: sorted(
            str(source)
            for source in schema(domain).sources
            if str(source).startswith("organ:")
        )
        for domain in DOMAINS
    }
    assert any(organ_reads.values()), "no domain reads an organ, so this is stale"
