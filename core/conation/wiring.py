"""core/conation/wiring.py — where the motivational organ meets the rest.

An organ nothing reads is a claim rather than a capability, so this file is
the honest part of the package. It says exactly which live paths conation
touches and what each one gets, and the list is short on purpose: the organ
should reach as far as it is useful and no further.

**The live mind snapshot** gets a compact conative section. That snapshot is
what grounds a desktop conversation turn, so this is the path by which "why do
I want this" reaches her speech as evidence rather than as instruction. The
direction is one-way and enforced by
``core/conation/invariants.py::conation.text_never_writes_motivation``.

**The affect layer** gets the conative arousal term. It already owns a heart
rate, a galvanic response and the arousal axis that sets sampling temperature,
so conation contributes the rise-in-motive term and lets that layer decide
what a body does with it.

**Initiative generation** gets typed motives instead of one score. A
spontaneous goal that carries its origin can be argued with; a spontaneous
goal carrying a number cannot.

**Telemetry** gets eight channels and four events.

Two uses reach outside the mind entirely, which is the point of building the
mechanism generally rather than as an emotion feature:

**Retry policy.** Frustration is a principled account of when to stop trying,
and its shape is a measured inverted U rather than a fixed attempt count. Any
subsystem that retries can ask ``should_disengage`` and get an answer that
accounts for how much the thing was wanted.

**Source quality.** The noisy-television detector identifies inputs whose
unpredictability never falls with exposure. That is a general statement about
a data source, and it is useful anywhere something decides whether to keep
sampling one.
"""

from __future__ import annotations

import logging
from typing import Any

from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.Conation.Wiring")

SERVICE_NAME = "conation"

_wired = False


def register(container: Any) -> bool:
    """Register the conation engine as a singleton service."""
    try:
        from core.container import ServiceLifetime

        container.register(
            SERVICE_NAME,
            lambda: __import__(
                "core.conation.engine", fromlist=["get_conation"]
            ).get_conation(),
            lifetime=ServiceLifetime.SINGLETON,
            required=False,
        )
        return True
    except (ImportError, AttributeError, TypeError, ValueError) as exc:
        record_degradation("conation_wiring", exc, severity="warning",
                           action="conation service not registered")
        return False


def boot() -> dict[str, Any]:
    """Declare telemetry and load invariants. Idempotent, safe to call late."""
    global _wired
    if _wired:
        return {"already": True}
    result: dict[str, Any] = {}
    try:
        from core.conation.telemetry import declare

        result["telemetry"] = declare()
    except (ImportError, RuntimeError, TypeError, ValueError) as exc:
        record_degradation("conation_wiring", exc, severity="debug",
                           action="conation telemetry not declared")
        result["telemetry"] = []
    try:
        import core.conation.invariants  # noqa: F401 — registration by import

        result["invariants"] = True
    except (ImportError, RuntimeError, TypeError, ValueError) as exc:
        record_degradation("conation_wiring", exc, severity="debug",
                           action="conation invariants not registered")
        result["invariants"] = False
    _wired = True
    return result


def snapshot() -> dict[str, Any]:
    """Compact conative readout for the live mind snapshot.

    Deliberately small. The full status is a debugging surface; what a
    conversation turn needs is what is wanted right now, why, whether any of
    it is borrowed, and what is being refused or held back.
    """
    try:
        from core.conation.engine import get_conation

        engine = get_conation()
        status = engine.status()
        last = status.get("last") or {}
        access = status.get("access", {})
        dynamics = status.get("dynamics", {})
        return {
            "present": True,
            "wanting": last.get("wanting"),
            "dominant_origin": last.get("dominant_origin"),
            "topology": last.get("topology"),
            "phase": last.get("phase"),
            "why": last.get("why"),
            "borrowed_fraction": last.get("borrowed_fraction"),
            "refusals": last.get("refusals", []),
            "arousal": dynamics.get("arousal"),
            "blocked_wants": access.get("blocked_wants", [])[:3],
            "frustrated": [row.get("key") for row in dynamics.get("frustrated", [])][:3],
            "overvalued": status.get("salience", {}).get("overvalued", [])[:3],
            "noisy_sources": status.get("epistemic", {}).get("noisy_sources", [])[:3],
        }
    except (ImportError, AttributeError, KeyError, TypeError, ValueError) as exc:
        record_degradation("conation_wiring", exc, severity="debug",
                           action="conation snapshot unavailable this turn")
        return {"present": False, "reason": "conation unavailable"}


def tick() -> dict[str, Any]:
    """One maintenance pass: publish telemetry, couple arousal to the soma."""
    try:
        from core.conation.engine import get_conation
        from core.conation.telemetry import publish

        engine = get_conation()
        return {
            "published": publish(engine),
            "coupled": engine.couple(),
        }
    except (ImportError, AttributeError, TypeError, ValueError) as exc:
        record_degradation("conation_wiring", exc, severity="debug",
                           action="conation tick skipped")
        return {"published": False, "coupled": {"delivered": False}}


# ── general-purpose readers, for callers outside the mind ────────────────


def should_disengage(key: str) -> bool:
    """Whether repeated failure at this thing has passed the stopping point.

    A retry policy that accounts for how much the thing was wanted. An agent
    that answers every failure with another attempt has no way to ever stop,
    and a fixed attempt count cannot tell a valued goal from an idle one.
    """
    try:
        from core.conation.engine import get_conation

        return get_conation().dynamics.frustration(key).should_disengage()
    except (ImportError, AttributeError, TypeError, ValueError):
        return False


def should_change_approach(key: str) -> bool:
    """Whether the current method has had a fair test and should be swapped."""
    try:
        from core.conation.engine import get_conation

        return get_conation().dynamics.frustration(key).should_switch_strategy()
    except (ImportError, AttributeError, TypeError, ValueError):
        return False


def record_attempt(key: str, *, succeeded: bool, wanting: float = 0.5) -> None:
    """Report an attempt at something, for the retry policy above."""
    try:
        from core.conation.engine import get_conation

        get_conation().dynamics.observe_attempt(
            key, wanting=wanting, succeeded=succeeded
        )
    except (ImportError, AttributeError, TypeError, ValueError):
        return


def source_is_noise(key: str) -> bool:
    """Whether this input's unpredictability has refused to fall with exposure.

    A general statement about a data source rather than about a motive.
    Useful anywhere something decides whether to keep sampling one.
    """
    try:
        from core.conation.engine import get_conation

        return key in get_conation().epistemic.noisy_sources()
    except (ImportError, AttributeError, TypeError, ValueError):
        return False


def observe_source_error(key: str, prediction_error: float) -> None:
    """Feed one prediction error from a source into the noise detector."""
    try:
        from core.conation.engine import get_conation

        get_conation().epistemic.observe_error(key, prediction_error)
    except (ImportError, AttributeError, TypeError, ValueError):
        return
