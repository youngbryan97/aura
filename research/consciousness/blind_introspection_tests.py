"""research/consciousness/blind_introspection_tests.py — retired.

This module claimed to audit whether Aura's introspection is grounded in her
telemetry. It could not fail, and it reported success on every invocation for
as long as it existed. Two hundred consecutive runs, two hundred passes.

The mechanism, kept here because it is easy to build again:

    inferred = actual + random.uniform(-5.0, 5.0)
    passed   = abs(actual - inferred) < 10.0

It read the value it was measuring, added noise bounded at five, and passed
when the deviation was under ten. The real branch read ``inferred_energy`` out
of ``active_beliefs``, and nothing anywhere in the tree ever wrote that key, so
the fabricating fallback was the only path that ever ran.

An instrument that derives its answer from the thing it is measuring has a
guaranteed result, and a guaranteed result is indistinguishable from a strong
finding. The green light then removes the reason anyone would build a real
one, which makes this worse than having had no test.

The replacement is ``research/consciousness/introspective_accuracy.py``. It
perturbs a quantity, compares the change in the report against the change in
the state, runs a null where nothing is perturbed, and scores tracking, gain
and false movement separately. It is validated against five reporters whose
answers are known in advance — live, constant, stale cache, noisy, half-scale
— and it returns NO VERDICT when the rig fails to deliver enough valid probes
rather than reporting a number it did not earn.
"""

from __future__ import annotations

from typing import Any

RETIRED_FOR = "research/consciousness/introspective_accuracy.py"


class BlindIntrospectionTester:
    """Retired. Use ``IntrospectiveAccuracy`` instead."""

    def run_blind_test(self, actual_state: Any) -> dict[str, Any]:
        raise NotImplementedError(
            "blind_introspection_tests could not fail: it derived the inferred "
            f"value from the actual one. Use {RETIRED_FOR}."
        )
