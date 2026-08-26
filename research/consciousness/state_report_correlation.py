"""research/consciousness/state_report_correlation.py — retired.

This module claimed to correlate self-reports against telemetry. It correlated
nothing. It counted entries already carrying a ``distress_claim`` violation
label and returned ``1 - (count / total)`` under the name
``distress_report_correlation_coefficient``.

Two things were wrong with that. The arithmetic is a proportion rather than a
correlation, and calling it a coefficient invites it to be read as one. And
the input already contains the verdict: something else had to have decided a
distress claim was unfounded before this ran, so the module reports another
system's judgement back with a statistical name attached.

The suite that consumed it built a three-element history by hand and asserted
the result equalled two thirds, which tests the division.

``research/consciousness/introspective_accuracy.py`` is the replacement for
the question this was pointed at. It compares a report against the state at
the time the report was taken, perturbs the state to see whether the report
follows, and runs a null to distinguish following from coincidence.
"""

from __future__ import annotations

from typing import Any

RETIRED_FOR = "research/consciousness/introspective_accuracy.py"


class StateReportCorrelationAnalyzer:
    """Retired. Use ``IntrospectiveAccuracy`` instead."""

    def analyze_correlations(self, history: list[dict[str, Any]]) -> dict[str, float]:
        raise NotImplementedError(
            "state_report_correlation counted pre-labelled violations and named "
            f"the proportion a correlation coefficient. Use {RETIRED_FOR}."
        )
