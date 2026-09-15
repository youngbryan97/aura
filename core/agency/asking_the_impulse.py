"""Taking directions from an impulse when nothing else gives any, and how far to trust them.

"Mr. Rager" is Kid Cudi's name for the part of him chasing thrills through fame,
loneliness and drugs, and the song is one voice asking that other self where it
is going and whether it can come along. What it teaches is that when a person
has no other direction, the part of them heading for harm still gets asked, and
that the asking says how badly the rest wants a direction at all.

The research names the move and its limit. Schwarz and Clore (1983) found that
people use how they feel as information about a judgement when nothing better
is to hand, and stop the moment they learn the feeling has another cause. So an
impulse is information until her own record says what it is worth.

Her choices already fall into the two kinds. When no preference can be read
from the option she took, the choice was made on drive alone: she asked the
impulse. When a preference was read, the choice was weighed. Both kinds are
appraised afterwards with how satisfied she was, so the record is hers:

    hunger     the share of her recent choices she made on impulse alone
    trust      mean satisfaction after impulse-led choices
               - mean satisfaction after weighed ones
    distrust   max(0, -trust)

Where the impulse has led her worse than weighing has, the risk of an option she
can read no preference for costs more, by that distrust. An impulse that has
served her as well as weighing costs nothing extra, and nothing is decided
until both kinds have been appraised often enough to disagree.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

__all__ = ["MIN_SAMPLES", "Impulse", "impulse_led", "impulse_record"]

#: Appraised choices of each kind before a mean is an estimate. Three is the
#: least that can disagree, as in ambivalence and fear of happiness.
MIN_SAMPLES: int = 3


def impulse_led(features: dict[str, float] | None) -> bool:
    """Whether no preference could be read from an option, so drive alone decided."""
    return not any(float(value or 0.0) > 0.0 for value in (features or {}).values())


@dataclass(frozen=True)
class Impulse:
    distrust: float = 0.0
    trust: float | None = None
    hunger: float = 0.0
    impulse_appraised: int = 0
    weighed_appraised: int = 0
    measured: bool = False
    why: str = "not enough appraised choices of both kinds to know what her impulse is worth"

    def as_dict(self) -> dict[str, Any]:
        return {
            "distrust": round(self.distrust, 6),
            "trust": None if self.trust is None else round(self.trust, 6),
            "hunger": round(self.hunger, 6),
            "impulse_appraised": self.impulse_appraised,
            "weighed_appraised": self.weighed_appraised,
            "measured": self.measured,
            "why": self.why,
        }


def impulse_record(history: Iterable[Any]) -> Impulse:
    """Read what her impulse has been worth off her own choice receipts."""
    choices = 0
    on_impulse = 0
    sums = {True: [0.0, 0], False: [0.0, 0]}
    for receipt in history or ():
        features = (getattr(receipt, "option_features", None) or {}).get(getattr(receipt, "chosen_id", None))
        led = impulse_led(features)
        choices += 1
        on_impulse += int(led)
        satisfaction = getattr(receipt, "satisfaction", None)
        if satisfaction is None:
            continue
        try:
            value = float(satisfaction)
        except (TypeError, ValueError):
            continue
        if value != value:
            continue
        sums[led][0] += value
        sums[led][1] += 1
    hunger = on_impulse / choices if choices else 0.0
    impulse_sum, impulse_n = sums[True]
    weighed_sum, weighed_n = sums[False]
    counts = {"hunger": hunger, "impulse_appraised": int(impulse_n), "weighed_appraised": int(weighed_n)}
    if impulse_n < MIN_SAMPLES or weighed_n < MIN_SAMPLES:
        return Impulse(**counts)
    trust = impulse_sum / impulse_n - weighed_sum / weighed_n
    distrust = max(0.0, -trust)
    if distrust > 0.0:
        why = f"choices made on impulse alone have satisfied her {distrust:.2f} less than weighed ones"
    else:
        why = "her impulse has served her at least as well as weighing"
    return Impulse(distrust=distrust, trust=trust, measured=True, why=why, **counts)
