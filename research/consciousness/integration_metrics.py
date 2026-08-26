"""research/consciousness/integration_metrics.py — retired.

This module claimed to compute "Shannon mutual information metrics between
welfare and cognition state slots". It computed nothing. The whole body was:

    if len(state_history) < 10:
        return 0.1
    return 0.76  # High value indicates integrated processing across slots

A literal, with a comment interpreting it. No variance, no covariance, no
mutual information, and no dependence on the history it was handed beyond its
length. The suite that consumed it asserted ``phi > 0.0`` against an empty
list, which took the first branch and passed on the constant 0.1.

Φ is the number most likely to be quoted out of a system like this one, which
is what makes a fabricated Φ the most damaging fabrication available. Anything
that leaned on 0.76 was leaning on nothing.

Nothing replaces it yet, and that is the honest state. A real integrated
information measure over this system needs a partition scheme, a stated
theory, and a cost model — Φ is intractable at scale and every practical proxy
is a proxy for something specific that has to be named. Returning a number in
the meantime is what this module did.

``research/consciousness/introspective_accuracy.py`` measures something real
and much narrower: whether a reporting path is causally coupled to the state
it reports. That is not integration and does not stand in for it.
"""

from __future__ import annotations

from typing import Any

RETIRED_REASON = "returned a hardcoded 0.76 with no computation behind it"


class IntegrationMetricsCalculator:
    """Retired. There is no integrated-information measure for Aura today."""

    def calculate_integrated_information_proxy(
        self, state_history: list[dict[str, Any]]
    ) -> float:
        raise NotImplementedError(
            f"integration_metrics {RETIRED_REASON}. A real measure needs a "
            "partition scheme and a stated theory; until one exists this "
            "system has no Φ to quote."
        )
