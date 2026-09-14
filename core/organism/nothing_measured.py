"""One name for "the instrument measured nothing", shared by the registry and its canaries."""

from __future__ import annotations


class NothingMeasured(RuntimeError):  # noqa: N818 — the name every raise site and test already uses
    """The instrument ran and had no population to measure.

    Raised so the suite reports ``Outcome.NOT_MEASURED`` rather than PASS.
    Three tests scored a clean zero over an empty set — lockdep counted 0
    splats across 0 known locks, the rate-group test took ``max([])`` of 0
    groups, and the health test counted 0 unresponsive components out of 0
    registered. All three passed, and all three were measuring nothing.

    Zero-over-zero is the exact shape the registry exists to refuse: the
    absence of a check reported as a passed check. It is not an ERROR either
    — the instrument is fine, there was simply nothing in front of it — and
    the claim it backs is neither confirmed nor refuted, so
    ``ValidationSuite.unsupported_claims`` lists it.
    """


__all__ = ["NothingMeasured"]
