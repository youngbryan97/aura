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
