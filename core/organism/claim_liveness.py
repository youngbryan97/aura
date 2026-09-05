"""core/organism/claim_liveness.py — evidence that expires.

Clean-room adoption of Hexis's capability-maturity scorecard (MIT), whose
one structural idea this codebase was missing: **the level is derived from
live deployment truth, not from what the author wrote down.** Their top
level is never "the code exists" or "a test passes" — it requires a row in
an audit table proving the behaviour actually happened in production.

:class:`core.organism.model_validation.Evidence` already reasons about
evidence more finely than that ladder does. What it cannot do is change
its mind. ``MEASURED_LIVE`` is set once, by the person registering the
claim, and stays set forever — including after the live path that earned it
stops running. Every entry in this repository's failure ledger with the
shape "a claim that outlived the code making it true" lives in that gap:

* two flagship measurements had writers and no reader at all, while a live
  writer went on reporting them
* a steering claim survived the A/B that refuted it
* fifteen guarded imports named symbols that had never existed

None of those were lies when written. They decayed, and nothing was
watching the decay.

A claim may now name the telemetry channels that carry its evidence. The
channels are already declared, already limit-checked, and already know
when they have gone silent (``ChannelSpec.stale_after_s``). This module
reads that and returns the claim's EFFECTIVE evidence:

* a bound channel that is fresh leaves the declared evidence standing
* a bound channel that has gone silent demotes the claim to ``UNMEASURED``
* a bound channel that was never declared demotes it too, and says so

That last case matters more than it looks. ``TelemetryDictionary.state()``
answers ``NOMINAL`` for a channel it has never heard of, so binding a claim
to a misspelled name would otherwise read as permanently healthy — an
absence reported as a pass, which is the exact inversion this codebase
keeps finding in its own gates. Liveness therefore checks *declaration*
first and treats an unknown channel as evidence of nothing.

Demotion is not deletion. The declared evidence is preserved alongside the
effective one, so the report can say "this was measured live and has been
silent for six hours" rather than quietly rewriting history.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from core.runtime.errors import record_degradation

__all__ = [
    "Liveness",
    "channel_liveness",
    "effective_evidence",
    "liveness_report",
]


@dataclass(frozen=True)
class Liveness:
    """Whether the telemetry behind a claim is still arriving."""

    channel: str
    declared: bool
    fresh: bool
    state: str
    age_s: float | None
    stale_after_s: float | None

    @property
    def supports(self) -> bool:
        """The channel exists and has been written recently enough."""
        return self.declared and self.fresh

    def reason(self) -> str:
        if not self.declared:
            return f"channel {self.channel!r} is not declared"
        if not self.fresh:
            if self.age_s is None:
                return f"channel {self.channel!r} has never been written"
            return (
                f"channel {self.channel!r} last wrote {self.age_s:.0f}s ago "
                f"(stale after {self.stale_after_s:.0f}s)"
            )
        return f"channel {self.channel!r} is live"

    def to_dict(self) -> dict[str, Any]:
        return {
            "channel": self.channel,
            "declared": self.declared,
            "fresh": self.fresh,
            "state": self.state,
            "age_s": None if self.age_s is None else round(self.age_s, 1),
            "stale_after_s": self.stale_after_s,
            "supports": self.supports,
            "reason": self.reason(),
        }


def channel_liveness(name: str) -> Liveness:
    """Read one channel's declaration and freshness.

    Never raises: a claim whose liveness cannot be established is reported
    as unsupported, which is the safe direction. A liveness check that
    could take down the validation suite would be worse than the drift it
    exists to catch.
    """
    channel = str(name or "").strip()
    if not channel:
        return Liveness(channel, False, False, "undeclared", None, None)
    try:
        from core.fsw.telemetry_dictionary import get_telemetry

        dictionary = get_telemetry()
        spec = dictionary.spec(channel)
        if spec is None:
            return Liveness(channel, False, False, "undeclared", None, None)
        sample = dictionary.value(channel)
        if sample is None:
            return Liveness(channel, True, False, "never_written", None, spec.stale_after_s)
        age = max(0.0, time.time() - float(sample.at))
        fresh = age <= float(spec.stale_after_s)
        return Liveness(
            channel,
            True,
            fresh,
            "live" if fresh else "stale",
            age,
            float(spec.stale_after_s),
        )
    except Exception as exc:
        record_degradation(
            "claim_liveness",
            exc,
            action="treated the claim's evidence as unverifiable",
        )
        return Liveness(channel, False, False, "unreadable", None, None)


def effective_evidence(
    declared: Any,
    channels: Sequence[str],
) -> tuple[Any, str, list[Liveness]]:
    """Resolve a claim's evidence against the channels that carry it.

    Returns ``(evidence, note, liveness)``. With no bound channels the
    declared evidence stands unchanged — binding is opt-in, and a claim
    that names nothing is exactly as trustworthy as it was before, which
    is to say: only as good as the person who wrote it.
    """
    from core.organism.model_validation import Evidence

    bound = [str(c) for c in channels if str(c or "").strip()]
    if not bound:
        return declared, "", []

    liveness = [channel_liveness(c) for c in bound]
    failing = [entry for entry in liveness if not entry.supports]
    if not failing:
        return declared, "", liveness

    # Only a live-measurement claim can decay into an unmeasured one.
    # Something already marked SYNTHETIC or RETRACTED is not made worse by
    # a silent channel, and overwriting it would lose the stronger fact.
    if declared is not Evidence.MEASURED_LIVE:
        return declared, "", liveness

    note = (
        "declared measured_live, but the telemetry behind it is not arriving: "
        + "; ".join(entry.reason() for entry in failing)
    )
    return Evidence.UNMEASURED, note, liveness


def liveness_report(claims: Sequence[Any]) -> dict[str, Any]:
    """Summarise which registered claims have decayed off their evidence."""
    from core.organism.model_validation import Evidence

    decayed: list[dict[str, Any]] = []
    bound = 0
    for claim in claims:
        channels = tuple(getattr(claim, "live_channels", ()) or ())
        if not channels:
            continue
        bound += 1
        declared = getattr(claim, "evidence", Evidence.MEASURED_LIVE)
        resolved, note, liveness = effective_evidence(declared, channels)
        if resolved is declared:
            continue
        decayed.append(
            {
                "statement": getattr(claim, "statement", ""),
                "test": getattr(claim, "test", ""),
                "owner": getattr(claim, "owner", ""),
                "declared_evidence": str(declared),
                "effective_evidence": str(resolved),
                "note": note,
                "channels": [entry.to_dict() for entry in liveness],
            }
        )
    return {
        "claims_bound_to_telemetry": bound,
        "decayed": decayed,
        "decayed_count": len(decayed),
    }


def liveness_of(claim: Any) -> Mapping[str, Any]:
    """Per-claim liveness detail, for a report that wants to show its work."""
    channels = tuple(getattr(claim, "live_channels", ()) or ())
    return {c: channel_liveness(c).to_dict() for c in channels}
