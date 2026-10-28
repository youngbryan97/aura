"""A run permitted to continue after a failure cannot settle the carrier."""
from core.subject.bridge import carrier_term


def test_a_degradation_override_cannot_settle_the_carrier():
    report = {"authority": {"authoritative": True, "allowed_degraded": True,
                             "blockers": ["clock.valve"]},
              "exclusion": {"status": "FOUND", "frontier": ["K"]}}
    result = carrier_term(report, {"verdict": "CLOSED"})
    assert result.status == "UNRESOLVED"
    assert result.blockers == ("clock.valve",)


def test_a_clean_carrier_still_settles():
    report = {"authority": {"authoritative": True, "allowed_degraded": False, "blockers": []},
              "exclusion": {"status": "FOUND", "frontier": ["K"]}}
    assert carrier_term(report, {"verdict": "CLOSED"}).status == "FOUND"
