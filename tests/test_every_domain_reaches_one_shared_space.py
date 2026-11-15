"""Her domains reached each other through named channels a few numbers wide.

Affect wrote valence, arousal and curiosity; motivation wrote two budgets;
chemistry wrote a mood; the intention band at offset 68 carried three scalars.
Outside those bands the substrate's five hundred and twelve neurons were driven
only by the mesh. A cut that takes a domain away loses three or four numbers, so
it is cheap; and two channels that never meet cannot carry anything jointly.

A thalamus is what biology reached for: overlapping projections into one
high-dimensional space that every region reads back out of. `all_to_all` in the
null suite is the degenerate version — one shared signal, effective dimension
1.38, bound and empty — and a relay with per-domain projections is bound and
rich at once.
"""

from __future__ import annotations

import contextlib

import numpy as np
import pytest

from core.consciousness.domain_relay import DOMAINS, DomainRelay, relay_strength, summarise


class _Substrate:
    def __init__(self, n: int = 512) -> None:
        self.x = np.zeros(n)
        self.marked: list[str] = []

    @property
    def sync_lock(self):
        return contextlib.nullcontext()

    def mark_state_mutated_locked(self, why: str) -> None:
        self.marked.append(why)


def _readings(**moved: float) -> dict[str, list[float]]:
    out = {domain: [0.5, 0.5, 0.5] for domain in DOMAINS}
    for domain, value in moved.items():
        out[domain] = [value, value, value]
    return out


def test_off_unless_a_run_asks_for_it(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AURA_DOMAIN_RELAY", raising=False)
    assert relay_strength() == 0.0
    monkeypatch.setenv("AURA_DOMAIN_RELAY", "1")
    assert relay_strength() == 1.0
    monkeypatch.setenv("AURA_DOMAIN_RELAY", "nonsense")
    assert relay_strength() == 0.0


def test_it_writes_the_second_half_and_never_a_named_band() -> None:
    """0 to 6 are the psychological state, 8 up is telemetry, 68 is intention."""
    relay = DomainRelay(512)
    assert relay.base == 256
    assert relay.width == 256
    substrate = _Substrate()
    relay.into(substrate, _readings(), 1.0)
    assert np.allclose(substrate.x[:256], 0.0)
    assert np.any(substrate.x[256:] != 0.0)
    assert "domain_relay" in substrate.marked


def _alone(domain: str) -> dict[str, list[float]]:
    """Only this domain has anything to say, so its own direction is what comes out."""
    return {d: ([1.0] if d == domain else []) for d in DOMAINS}


def test_every_domain_has_its_own_direction_and_they_overlap() -> None:
    """Private slices would restore the cheap cut this exists to close."""
    relay = DomainRelay(512)
    directions = {d: relay.drive(_alone(d)) for d in DOMAINS}
    for domain, vector in directions.items():
        # Each domain reaches the whole block, not a slice of it.
        assert np.count_nonzero(vector) > relay.width * 0.9, domain
    pairs = [
        abs(float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))))
        for i, a in enumerate(directions.values())
        for b in list(directions.values())[i + 1 :]
    ]
    # Overlapping but not aligned: random directions in 256 dimensions.
    assert max(pairs) < 0.3, max(pairs)


def test_taking_a_domain_away_changes_the_mixture() -> None:
    """Which is the whole point: the mixture is what everything downstream reads."""
    relay = DomainRelay(512)
    whole = relay.drive(_readings())
    for domain in DOMAINS:
        without = dict(_readings())
        without[domain] = []
        assert not np.allclose(whole, relay.drive(without)), domain


def test_a_domain_with_nothing_to_say_contributes_nothing() -> None:
    relay = DomainRelay(512)
    assert np.allclose(relay.drive({d: [] for d in DOMAINS}), 0.0)
    assert np.allclose(relay.drive({}), 0.0)


def test_one_domain_of_step_counts_does_not_own_the_mixture() -> None:
    """Her numbers run from a share to a step count; the size is squashed."""
    relay = DomainRelay(512)
    ordinary = relay.drive(_readings())
    huge = relay.drive(_readings(N=1e6))
    assert float(np.abs(huge).max()) < 3.0 * float(np.abs(ordinary).max())


def test_the_same_relay_is_rebuilt_identically() -> None:
    """A relay that rewires itself between two arms of a trial is not one relay."""
    a, b = DomainRelay(512), DomainRelay(512)
    assert np.allclose(a.drive(_readings(A=0.7)), b.drive(_readings(A=0.7)))


def test_a_relay_that_is_off_touches_nothing() -> None:
    substrate = _Substrate()
    assert DomainRelay(512).into(substrate, _readings(), 0.0) == 0.0
    assert np.allclose(substrate.x, 0.0)


def test_every_domain_is_summarised_from_her_state() -> None:
    class _Empty:
        pass

    readings = summarise(_Empty())
    assert set(readings) == set(DOMAINS)


def test_the_last_phase_drives_it(monkeypatch) -> None:
    """A pass of the consciousness phase, with none of her services up, still puts every domain into the block."""
    import asyncio

    from core.phases import consciousness_phase
    from core.phases.consciousness_phase import ConsciousnessPhase
    from core.state.aura_state import AuraState

    driven: list[object] = []
    monkeypatch.setattr(consciousness_phase, "get_consciousness_integration", lambda: None)
    monkeypatch.setattr(consciousness_phase, "get_runtime_service", lambda name, default=None: default)
    monkeypatch.setattr(ConsciousnessPhase, "_every_domain_into_the_shared_block",
                        classmethod(lambda cls, state: driven.append(state)))
    returned = asyncio.run(ConsciousnessPhase.__new__(ConsciousnessPhase).execute(AuraState.default()))
    assert driven == [returned]
