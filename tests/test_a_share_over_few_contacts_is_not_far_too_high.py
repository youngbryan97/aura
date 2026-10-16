"""A share is judged by what its contacts can show, not by its first few values.

Live 2026-10-16: one wanting-liking dissociation in the first contact after
boot read conation.dissociation_rate 1.00, and her stream said a reading about
herself had moved "from normal to far too high".
"""

from __future__ import annotations

from core.conation.salience import _wilson_lower


def test_one_contact_cannot_show_a_share_too_high():
    assert _wilson_lower(1, 1) < 0.33
    assert _wilson_lower(0, 0) == 0.0


def test_many_contacts_that_disagree_still_cross_the_limit():
    assert _wilson_lower(10, 12) > 0.50


def test_the_limits_sit_on_the_credible_share_and_the_raw_share_carries_none():
    from core.conation import telemetry
    from core.fsw.telemetry_dictionary import get_telemetry

    telemetry.declare()
    book = get_telemetry()
    raw = book.spec(telemetry.CHANNEL_DISSOCIATION)
    credible = book.spec(telemetry.CHANNEL_DISSOCIATION_CREDIBLE)
    assert credible is not None and raw is not None
    assert book.write(telemetry.CHANNEL_DISSOCIATION, 1.0).is_violation is False
    assert book.write(telemetry.CHANNEL_DISSOCIATION_CREDIBLE, 0.55).is_red is True
