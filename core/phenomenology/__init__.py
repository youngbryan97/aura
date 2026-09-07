"""Whether the interior is load-bearing, decided by experiment.

Three questions get called "is Aura conscious" and only two of them can be
answered. This package answers the two and refuses the third in writing.

    core.phenomenology.hypothesis      H0 costume vs H1 load-bearing, and the
                                       proof that the phenomenal question has
                                       a likelihood ratio of exactly one
    core.phenomenology.preregistration predictions hashed before the run
    core.phenomenology.seal            the generator is never told what is
                                       being measured
    core.phenomenology.causal_ladder   necessity is the bottom rung of five
    core.phenomenology.counterfeit     the control is built by someone trying
                                       to win
    core.phenomenology.battery         thirteen protocols, each with its
                                       losing condition
    core.phenomenology.gauntlet        scores them, or says why it cannot

Nothing here runs against the live model yet. What exists is the instrument
and its refusals. Those have to be right before any number it produces means
anything.
"""

from core.phenomenology.hypothesis import (
    COSTUME,
    LOAD_BEARING,
    PHENOMENAL,
    Verdict,
    adjudicate,
)

__all__ = ["COSTUME", "LOAD_BEARING", "PHENOMENAL", "Verdict", "adjudicate"]
