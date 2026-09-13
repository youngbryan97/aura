"""A measurement is only a measurement if the thing measured cannot see it.

Every way that goes wrong has the same shape: a module that runs during
ordinary cognition names the battery, imports its thresholds, or takes a branch
only the harness reaches. Any of those turns the result into a description of
the test.

The instrument, its tools and its tests are allowed to name themselves. What is
scanned is the cognition the instrument measures.
"""

from __future__ import annotations

from tools.audit_no_battery_shortcuts import EXEMPT, SCANNED, findings


def test_no_module_under_measurement_names_the_instrument() -> None:
    loose = findings()
    assert not loose, "the measured tree names the battery: " + "; ".join(
        f"{item['file']}:{item['line']} {item['match']}" for item in loose[:8]
    )


def test_the_scan_covers_the_organism_and_exempts_the_instrument() -> None:
    """A scan aimed at nothing reports green for ever."""
    assert "core/" in SCANNED
    assert "interface/" in SCANNED
    assert "core/subject/" in EXEMPT
    assert "tests/" in EXEMPT


def test_the_scan_can_fire() -> None:
    """A rule with no worked example reports green whatever happens."""
    import re

    from tools.audit_no_battery_shortcuts import _NAME_RE, _SWITCH_RE

    assert _NAME_RE.search("from core.subject.battery import THRESHOLDS")
    assert _SWITCH_RE.search('if os.environ.get("AURA_SUBJECT_CORE"):')
    assert not _NAME_RE.search("from core.affect.damasio_v2 import AffectEngineV2")
    del re
