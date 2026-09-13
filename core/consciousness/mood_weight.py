"""A mood activation as a signed steering weight.

Every mood dimension is a 0..1 activation — `_neutral_reference_state` says so
in as many words — and the weight read the raw value as though it were already
signed. Neutral, 0.5, weighed +0.46. A low state pushed the SAME way as a high
one, only less hard. So the channel carried a constant offset whose magnitude
wobbled rather than her state.

The fusion certificate is what caught it, on 2026-09-13: two opposing states
moved the distribution apart by 0.171 of how far either moved at all, against a
floor of half, and `carries_content` refused to open the channel.
"""

from __future__ import annotations

import numpy as np

#: Neutral on a mood dimension: the midpoint of the declared 0..1 range.
NEUTRAL_MOOD = 0.5


def signed_weight(raw: float, substrate_fn: str) -> float:
    """Centred on neutral, so below it steers the other way."""
    centred = 2.0 * (float(raw) - NEUTRAL_MOOD)
    if substrate_fn == "linear_half":
        return float(np.clip(centred, -1.0, 1.0))
    return float(np.tanh(centred))


__all__ = ["NEUTRAL_MOOD", "signed_weight"]
