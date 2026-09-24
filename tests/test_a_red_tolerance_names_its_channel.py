"""interiority.worst_tolerance goes red with a number and no name.

LIVE 2026-09-23: red_high at 0.95 three minutes after boot, and nothing that
could be asked said which channel had adapted away its gain.
"""

from __future__ import annotations


def test_the_diagnostic_route_carries_the_receptor_bank():
    from core.interiority.receptors import get_receptor_bank
    from interface.routes.system import _interiority_receptors

    bank = get_receptor_bank()
    said = _interiority_receptors()
    assert set(said) == set(bank.snapshot())
    assert "channels" in said
