"""Amortize optional work against measured essential work, without waiting.

One indivisible operation can overdraw its allowance. Its observed cost
must be repaid before another starts. This is a work-sharing policy, not
a bound on one operation's latency or a measurement of host CPU use.
"""
from __future__ import annotations

import math


class OptionalWorkShare:
    """A bounded credit account with one outstanding optional operation."""

    def __init__(self, *, fraction: float = .25, credit_cap_s: float = .1,
                 admission_s: float = .025) -> None:
        values = (fraction, credit_cap_s, admission_s)
        if (any(type(v) not in (int, float) or not math.isfinite(v) for v in values)
                or not 0 < fraction <= .5 or not 0 < admission_s <= credit_cap_s <= 1.):
            raise ValueError('optional work needs a finite minority share and bounded credit')
        self.ratio = fraction/(1.-fraction)
        self.cap, self.admission = credit_cap_s, admission_s
        self.credit = 0.
        self.active = False

    def reserve(self, essential_s: float) -> bool:
        """Earn from one measured essential interval; never queue an operation."""
        if (self.active or type(essential_s) not in (int, float)
                or not math.isfinite(essential_s) or essential_s < 0):
            return False
        self.credit = min(self.cap, self.credit+essential_s*self.ratio)
        if self.credit < self.admission:
            return False
        self.active = True
        return True

    def spent(self, seconds: float) -> None:
        """Charge actual elapsed work, including a failed operation."""
        if not self.active:
            raise ValueError('an optional cost needs its outstanding reservation')
        self.active = False
        if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds < 0:
            self.credit = -math.inf
            raise ValueError('an unmeasured optional cost cannot release its budget')
        self.credit -= seconds
