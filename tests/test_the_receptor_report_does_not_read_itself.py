"""interiority.worst_tolerance read red at 0.95 on every boot from 23 September.

LIVE 2026-10-04, from /api/health/interiority: the one channel at the gain
floor was f18_receptor_adjustment, the faculty that reports the receptor
bank. Its intensity was the worst tolerance or withdrawal over every channel,
its own included, and its report passes through the bank like every
faculty's. Two quiet channels (f16_neat, f27_pursuit_gait) reported
withdrawal near 0.5 with nothing internalised, so f18 fired, its channel
desensitized under the firing, and from then on its own tolerance kept it
firing.
"""

from __future__ import annotations

from core.interiority.receptors import Receptor, ReceptorBank


def test_a_channel_that_never_carried_a_signal_misses_nothing():
    quiet = Receptor("never_driven")
    for _ in range(600):
        quiet.transduce(0.0, dt=1.0)

    assert quiet.scale > 1.0, "a long silence raises the gain"
    assert quiet.withdrawal() == 0.0


def test_a_signal_that_stops_is_still_missed():
    shouted = Receptor("driven")
    for _ in range(60):
        shouted.transduce(1.0, dt=1.0)
    for _ in range(30):
        shouted.transduce(0.0, dt=1.0)

    assert shouted.internalised > 0.0
    assert shouted.withdrawal() > 0.0


def test_the_report_of_the_bank_does_not_read_its_own_channel(monkeypatch):
    from core.interiority.faculties.f18_receptor_adjustment import ReceptorAdjustment

    bank = ReceptorBank()
    own = bank.receptor(ReceptorAdjustment.id)
    for _ in range(120):
        own.transduce(1.0, dt=1.0)
    assert own.tolerance() > 0.9
    bank.receptor("f06_sympathetic_concern").transduce(0.2, dt=1.0)

    class _Context:
        def receptors(self):
            return bank

    activation = ReceptorAdjustment().compute(_Context())

    assert activation.intensity < 0.5
    assert ReceptorAdjustment.id not in activation.receipt["gains"]


def test_a_release_that_clears_does_not_hold_the_channel_for_the_whole_interval():
    """LIVE 2026-10-04: two interior ticks in 72 s left f16_neat and f27_pursuit_gait fully internalised.

    A release clears at four per second. Adapting as though it had been there
    for the 60 s since the last tick took the channel to the gain floor.
    """
    import random

    from core.interiority.cleft import SynapticCleft

    bank = ReceptorBank()
    medium = SynapticCleft(bank=bank, rng=random.Random(7))
    for _ in range(3):
        medium.release("f16_neat", 0.2, dt=36.0)

    assert bank.receptor("f16_neat").tolerance() < 0.5


def test_a_long_interval_does_not_overshoot_the_steady_state():
    held = Receptor("held")
    for _ in range(60):
        held.transduce(1.0, dt=1.0)
    stepped = held.phosphorylated, held.internalised

    jumped = Receptor("jumped")
    jumped.transduce(1.0, dt=60.0)

    assert abs(jumped.phosphorylated - stepped[0]) < 0.05
    assert abs(jumped.internalised - stepped[1]) < 0.1
