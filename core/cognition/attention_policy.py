"""core/cognition/attention_policy.py — one policy for what is worth processing.

Attention is allocated in three places that do not know about each other. The
global workspace scores candidates by salience with hand-tuned weights. The
AtomSpace runs an ECAN economy over STI and LTI. Perception prioritises regions
by its own rules. Each is defensible and none of them learns: a weight that was
right when it was written stays right forever, and a component that stopped
mattering keeps its share.

One policy, over one currency. Each channel - a percept, a memory, an atom, a
cognitive candidate - gets an allocation, and the allocation is updated by what
the channel's attention actually bought downstream. That is the only signal
that means anything: attention is not a preference, it is a bet, and a bet is
settled by the outcome.

Learning without thrashing
--------------------------
Two guards, because an attention policy that overreacts is worse than a fixed
one. Updates are exponentially smoothed, so one lucky turn does not move the
allocation far. And every channel keeps a floor: a channel starved to zero can
never produce the evidence that would earn its share back, which is how a
learned policy locks in whatever it believed early.

The comparison that matters
---------------------------
:meth:`AttentionPolicy.against_static` scores the learned allocation against
the fixed weights it replaced, at equal total compute. A learned policy that
does not beat the weights somebody tuned by hand is a more complicated way to
get the same answer.
"""

from __future__ import annotations

from core.runtime.lockdep import checked_lock
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = ["Channel", "AttentionPolicy", "get_attention_policy", "reset_attention_policy_for_test"]

#: Smallest share a channel may fall to. A starved channel cannot generate the
#: evidence that would earn its share back.
FLOOR = 0.02

#: How fast the allocation follows the evidence. Low, because one lucky turn is
#: not a reason to reallocate a mind.
RATE = 0.1


@dataclass
class Channel:
    """One thing attention can be spent on, and what spending it has bought."""

    name: str
    kind: str
    share: float = 0.0
    spent: float = 0.0
    returned: float = 0.0
    observations: int = 0

    @property
    def value_per_unit(self) -> float | None:
        return self.returned / self.spent if self.spent > 0 else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "share": self.share,
            "spent": self.spent,
            "returned": self.returned,
            "value_per_unit": self.value_per_unit,
            "observations": self.observations,
        }


class AttentionPolicy:
    """What to process, learned from what processing it returned."""

    def __init__(self, *, floor: float = FLOOR, rate: float = RATE) -> None:
        self._lock = checked_lock("core.cognition.attention_policy.AttentionPolicy", reentrant=True)
        self._channels: dict[str, Channel] = {}
        self._floor = float(floor)
        self._rate = float(rate)
        self._static: dict[str, float] = {}
        self._history: list[tuple[float, float]] = []

    def register(self, name: str, kind: str, *, static_weight: float = 0.0) -> Channel:
        """Add a channel, recording the fixed weight it is replacing."""
        with self._lock:
            channel = self._channels.setdefault(name, Channel(name=name, kind=kind))
            if static_weight:
                self._static[name] = static_weight
            self._normalise_locked()
            return channel

    def observe(self, name: str, *, spent: float, returned: float) -> None:
        """Record what attending to this channel cost and what it bought."""
        with self._lock:
            channel = self._channels.get(name)
            if channel is None:
                return
            channel.spent += max(0.0, spent)
            channel.returned += returned
            channel.observations += 1

    def reallocate(self) -> dict[str, float]:
        """Move shares toward measured return, smoothly and above the floor."""
        with self._lock:
            measured = {
                name: c.value_per_unit
                for name, c in self._channels.items()
                if c.value_per_unit is not None
            }
            if not measured:
                return self.allocation()
            total = sum(max(0.0, v) for v in measured.values())
            for name, channel in self._channels.items():
                if name in measured and total > 0:
                    target = max(0.0, measured[name]) / total
                else:
                    target = channel.share
                channel.share += self._rate * (target - channel.share)
            self._normalise_locked()
            return self.allocation()

    def _normalise_locked(self) -> None:
        channels = list(self._channels.values())
        if not channels:
            return
        for channel in channels:
            channel.share = max(self._floor, channel.share)
        total = sum(c.share for c in channels)
        if total <= 0:
            for channel in channels:
                channel.share = 1.0 / len(channels)
            return
        for channel in channels:
            channel.share /= total

    def allocation(self) -> dict[str, float]:
        with self._lock:
            return {name: c.share for name, c in sorted(self._channels.items())}

    def against_static(self, learned_score: float, static_score: float) -> dict[str, Any]:
        """Did learning the allocation beat the weights it replaced.

        Recorded rather than computed, because the two arms have to be run at
        equal compute and this class cannot check that. What it can do is keep
        the comparison beside the policy, so a learned allocation with no such
        comparison is visibly unjustified.
        """
        with self._lock:
            self._history.append((learned_score, static_score))
        return {
            "learned": learned_score,
            "static": static_score,
            "delta": learned_score - static_score,
            "worth_learning": learned_score > static_score,
            "comparisons": len(self._history),
        }

    def report(self) -> dict[str, Any]:
        with self._lock:
            channels = list(self._channels.values())
            history = list(self._history)
        starved = [c.name for c in channels if c.share <= self._floor + 1e-9]
        return {
            "channels": len(channels),
            "by_kind": {
                kind: sorted(c.name for c in channels if c.kind == kind)
                for kind in sorted({c.kind for c in channels})
            },
            "allocation": {c.name: c.share for c in sorted(channels, key=lambda c: -c.share)},
            "at_the_floor": starved,
            "unmeasured": [c.name for c in channels if c.value_per_unit is None],
            "static_comparisons": [
                {"learned": a, "static": b, "delta": a - b} for a, b in history
            ],
            "beats_static": (
                all(a > b for a, b in history) if history else None
            ),
        }


_lock = checked_lock("core.cognition.attention_policy.singleton")
_policy: AttentionPolicy | None = None


def get_attention_policy() -> AttentionPolicy:
    global _policy
    with _lock:
        if _policy is None:
            _policy = AttentionPolicy()
        return _policy


def reset_attention_policy_for_test(**kwargs: Any) -> AttentionPolicy:
    global _policy
    with _lock:
        _policy = AttentionPolicy(**kwargs)
        return _policy
