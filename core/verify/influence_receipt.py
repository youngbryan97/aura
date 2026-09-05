"""core/verify/influence_receipt.py

The receipt a subsystem is allowed to attach to its output.

Aura's existing receipts are assertions the code makes about itself.
``live_mind_controls_bound: True`` was set by the same function that would have
set it False, from a dict it had just filled with constants when the real one
came back empty. Nothing downstream could tell the difference between that and
a mind that actually steered the reply, because the two produce identical JSON.

An :class:`InfluenceReceipt` cannot do that. Every field on it is derived from
:mod:`core.verify.causal_influence` verdicts, which are derived from paired
trials, which require a registered lesion and a measured null. There is no
argument that makes ``bound`` True; it is True when the measurements say so and
there is no other way to get there.

The vocabulary is three-valued on purpose. ``influential`` and ``inert`` are
both measurements. ``unmeasured`` is the absence of one, and it is the honest
answer for most of this system right now — which is the finding, not a bug in
the receipt.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from core.verify.causal_influence import (
    ChannelVerdict,
    Verdict,
    get_influence_ledger,
)
from core.verify.lesion_registry import get_lesion_registry

__all__ = [
    "InfluenceReceipt",
    "build_influence_receipt",
    "channel_is_influential",
    "unfalsifiable_channels",
]


@dataclass(frozen=True)
class InfluenceReceipt:
    """What is actually known about a turn's causal channels."""

    channels: dict[str, dict[str, Any]]
    influential: tuple[str, ...]
    inert: tuple[str, ...]
    unmeasured: tuple[str, ...]
    #: Channels nothing can ever measure, because no lesion is registered for
    #: them. A claim about one of these is unfalsifiable by construction.
    unfalsifiable: tuple[str, ...]
    source: str

    @property
    def bound(self) -> bool:
        """Did any claimed channel measurably shape the output?

        The field the old receipt set from a constant.
        """

        return bool(self.influential)

    @property
    def status(self) -> str:
        if self.influential:
            return "measured_influential"
        if self.unfalsifiable:
            return "unfalsifiable_no_lesion_registered"
        if self.inert and not self.unmeasured:
            return "measured_inert"
        return "unmeasured"

    def as_dict(self) -> dict[str, Any]:
        return {
            "bound": self.bound,
            "status": self.status,
            "influential": list(self.influential),
            "inert": list(self.inert),
            "unmeasured": list(self.unmeasured),
            "unfalsifiable": list(self.unfalsifiable),
            "channels": self.channels,
            "source": self.source,
        }


def build_influence_receipt(
    channels: Iterable[str],
    *,
    source: str,
) -> InfluenceReceipt:
    """Build the receipt for the channels a turn claims to have run through."""

    ledger = get_influence_ledger()
    registry = get_lesion_registry()

    resolved: dict[str, dict[str, Any]] = {}
    influential: list[str] = []
    inert: list[str] = []
    unmeasured: list[str] = []
    unfalsifiable: list[str] = []

    for name in dict.fromkeys(str(c) for c in channels if str(c).strip()):
        verdict: ChannelVerdict = ledger.verdict(name)
        entry = verdict.as_dict()
        registered = registry.is_registered(name)
        entry["lesion_registered"] = registered
        handle = registry.get(name)
        entry["direct_actuation"] = bool(handle.direct_actuation) if handle else False
        resolved[name] = entry

        if not registered:
            unfalsifiable.append(name)
            # An unregistered channel is reported as unmeasured too: it has no
            # verdict and must never be counted toward `bound`.
            unmeasured.append(name)
            continue
        if verdict.verdict is Verdict.INFLUENTIAL:
            influential.append(name)
        elif verdict.verdict is Verdict.INERT:
            inert.append(name)
        else:
            unmeasured.append(name)

    return InfluenceReceipt(
        channels=resolved,
        influential=tuple(influential),
        inert=tuple(inert),
        unmeasured=tuple(unmeasured),
        unfalsifiable=tuple(unfalsifiable),
        source=str(source),
    )


def channel_is_influential(channel: str) -> bool:
    """The single-channel gate. False unless measurement says otherwise."""

    if not get_lesion_registry().is_registered(channel):
        return False
    return get_influence_ledger().verdict(channel).is_influential


def unfalsifiable_channels(channels: Iterable[str]) -> tuple[str, ...]:
    """Channels with no registered lesion — claims about them cannot be checked."""

    registry = get_lesion_registry()
    return tuple(
        sorted(
            {
                str(c)
                for c in channels
                if str(c).strip() and not registry.is_registered(str(c))
            }
        )
    )
