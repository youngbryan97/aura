"""core/conation — Aura's motivational faculty.

Conation is the third term of the classical trilogy of mind. Cognition asks
what is the case, affect asks how it feels, and conation asks what is being
pursued and on whose authority. Aura had the first two.

The gap showed up as a measurement. Four motivational situations that behave
nothing alike — a want borrowed from another child, a cue-triggered appetite,
a hand going out to a snail, a friend flustered on purpose — return
bit-identical values from the affect appraisal path: v=0.000, a=0.410,
e=0.525, six of six pairs at L2 distance zero. Valence and arousal have no
axis on which those differ, because what differs is where the value was
manufactured and whose mind had to be involved to manufacture it.

Public surface:

``ValueOrigin``      where value came from: homeostatic, epistemic,
                     aesthetic, vicarious, enactive.
``MindTopology``     whose mind it passes through.
``Incentive``        a candidate for motivation.
``ConativeState``    the typed stance toward one, with wanting and liking
                     held apart and every origin's evidence attached.
``ConationEngine``   assembles states, arbitrates among them, and learns from
                     what happens.
``get_conation()``   the process-wide engine.
"""

from core.conation.access import AccessLedger, Blocker, GrantResponse
from core.conation.aesthetic import AestheticValuation
from core.conation.dynamics import ConativeDynamics
from core.conation.enactive import (
    EnactiveValuation,
    PlayFrame,
    Refusal,
    TargetForecast,
)
from core.conation.engine import (
    Choice,
    ConationEngine,
    get_conation,
    reset_conation_for_test,
)
from core.conation.epistemic import EpistemicValuation, wundt_curve
from core.conation.homeostatic import HomeostaticValuation
from core.conation.origins import (
    EVIDENCE_REQUIRED,
    SOCIAL_ORIGINS,
    ConativePhase,
    Instrumentality,
    MindTopology,
    OriginReading,
    ValueOrigin,
)
from core.conation.salience import IncentiveSalience, SalienceCalibration
from core.conation.state import (
    VECTOR_FIELDS,
    ConativeState,
    Incentive,
    OutcomeReport,
)
from core.conation.vicarious import VicariousValuation

__all__ = [
    "AccessLedger",
    "AestheticValuation",
    "Blocker",
    "Choice",
    "ConationEngine",
    "ConativeDynamics",
    "ConativePhase",
    "ConativeState",
    "EVIDENCE_REQUIRED",
    "EnactiveValuation",
    "EpistemicValuation",
    "GrantResponse",
    "HomeostaticValuation",
    "Incentive",
    "IncentiveSalience",
    "Instrumentality",
    "MindTopology",
    "OriginReading",
    "OutcomeReport",
    "PlayFrame",
    "Refusal",
    "SOCIAL_ORIGINS",
    "SalienceCalibration",
    "TargetForecast",
    "VECTOR_FIELDS",
    "ValueOrigin",
    "VicariousValuation",
    "get_conation",
    "reset_conation_for_test",
    "wundt_curve",
]
