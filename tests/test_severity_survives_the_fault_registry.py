"""A deliberate de-escalation was silently undone by a missing dict key.

turn_outcome records an empty cognitive cycle at severity="info" on purpose.
Its comment explains why: recording it any higher once produced 231 CRITICAL
SERVICE FAILUREs in a day, took long-term memory consolidation down with them,
and had the healer dispatching repairs at severity=emergency for a cycle that
simply produced no text.

record_degradation honoured that. The fault registry did not. Its severity map
had no "info" key, so the lookup fell to a MARGINAL default and the same event
that had been carefully classified as ordinary reappeared as
"FAULT RUNTIME-COGNITIVE_ENGINE [MARGINAL]" — 916 in one sampled window, plus
391 of its sibling. Every info-severity degradation in the runtime shared the
fate, so the classification existed and could not be expressed.

An UNRECOGNISED severity should still be MARGINAL: not knowing is not evidence
of harmlessness. A recognised one that says ordinary has to be allowed to.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from core.resilience.fault_taxonomy import FaultSeverity

_SOURCE = Path("core/runtime/errors.py").read_text(encoding="utf-8")


def _severity_map() -> dict[str, str]:
    match = re.search(r"_sev_map = \{(.*?)\}", _SOURCE, re.S)
    assert match, "severity map not found"
    return dict(re.findall(r'"(\w+)":\s*FaultSeverity\.(\w+)', match.group(1)))


@pytest.mark.parametrize(
    ("severity", "expected"),
    [
        ("critical", "CRITICAL"),
        ("degraded", "MARGINAL"),
        ("warning", "MARGINAL"),
        ("info", "NEGLIGIBLE"),
        ("debug", "NEGLIGIBLE"),
    ],
)
def test_every_severity_the_runtime_emits_is_mapped(severity, expected):
    """A severity that is emitted but unmapped is a silent re-escalation."""
    assert _severity_map().get(severity) == expected, severity


def test_info_is_not_marginal():
    """The exact regression: 916 ordinary cycles reported as faults."""
    assert _severity_map()["info"] == "NEGLIGIBLE"


def test_an_unknown_severity_still_defaults_to_marginal():
    """Not recognising a severity is not evidence that it is harmless."""
    assert "FaultSeverity.MARGINAL)" in _SOURCE


def test_the_mapped_names_are_real_severities():
    """A typo here would fall to the default and look like it worked."""
    for name in _severity_map().values():
        assert hasattr(FaultSeverity, name), name


def test_turn_outcome_still_classifies_an_empty_cycle_as_ordinary():
    """The upstream half of the contract this test exists to protect."""
    source = Path("core/runtime/turn_outcome.py").read_text(encoding="utf-8")

    block = source[source.index('elif receipt.rationale == "nothing_served"') :][:1400]
    assert 'severity = "info"' in block
