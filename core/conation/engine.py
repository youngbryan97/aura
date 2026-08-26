"""core/conation/engine.py — assembling a motivational state, and choosing.

This is where the five origins, the salience pair, the access gate and the
temporal dynamics meet. Three things happen here and they are kept apart on
purpose.

**Appraisal** builds a ``ConativeState`` for one incentive by consulting every
origin that has evidence. Origins without evidence are recorded as
unavailable, never as zero.

**Arbitration** turns a set of appraised states into a choice. Permission
enters here and only here. Every candidate is appraised, including the
forbidden ones, and then the forbidden ones are removed from the selectable
set with their motivational state intact.

That ordering is the architectural claim of the file. A child who sees another
child's toy genuinely wants it, and taking it is genuinely not available.
Building the agent so the forbidden option never acquires value would be
simpler and would remove the thing that makes the correct behaviour an
achievement: choosing to ask, while still wanting to take. It also removes any
chance of noticing that a rule is doing work, because a want that never forms
leaves no trace when it is denied.

**Intervention** is ``do()``. A causal claim about this system means the
system's outputs move when a named variable is forced and nothing else
changes. Correlations between motivational variables prove nothing here —
they all rise together in interesting situations. The battery in
``tests/test_conation_causal_battery.py`` uses this to hold everything fixed
and move one thing.

## Weights

Arbitration takes the mean over the motivational terms that were actually
measured, then subtracts effort and risk. Equal weighting is the refusal to
invent a psychology, the same refusal made in ``core/conation/epistemic.py``,
and it holds until a learned head earns something better against a
counterfactual slice. ``adopt_weights`` is where such a head installs one, and
the state's readout says which is in force.

## Language is downstream

Nothing in this file reads or writes text. The state is computed from budgets,
traces, encodings, observations and person-models, and a speech path may
render it afterwards. Text may never write back: a report that Aura enjoyed
something is not evidence that she did, and treating it as evidence would
close a loop in which the language layer manufactures the states it then
describes. ``core/conation/invariants.py`` fails if that loop ever appears.
"""

from __future__ import annotations

import contextlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Iterator, Sequence

from core.conation.access import AccessLedger, Blocker
from core.conation.calibration import CalibrationRegistry
from core.conation.aesthetic import AestheticValuation
from core.conation.dynamics import ConativeDynamics
from core.conation.enactive import EnactiveValuation, PlayFrame, TargetForecast
from core.conation.epistemic import EpistemicValuation
from core.conation.homeostatic import HomeostaticValuation
from core.conation.origins import (
    ConativePhase,
    Instrumentality,
    MindTopology,
    OriginReading,
    REQUIRED_TOPOLOGY,
    ValueOrigin,
)
from core.conation.salience import IncentiveSalience, SalienceCalibration
from core.conation.state import (
    ConativeState,
    Incentive,
    OutcomeReport,
    VECTOR_FIELDS,
)
from core.conation.vicarious import VicariousValuation
from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.Conation")

EPS = 1e-12

#: Motivational vector positions that carry positive pull. Effort and risk are
#: costs and goal_value is instrumental; those are handled explicitly rather
#: than averaged in, so a high-effort candidate cannot look motivating because
#: its cost is large.
_PULL_FIELDS = (
    "wanting",
    "predicted_liking",
    "epistemic",
    "aesthetic",
    "vicarious",
    "enactive",
)


@dataclass(frozen=True, slots=True)
class Choice:
    """One arbitration outcome, with what lost and why."""

    selected: str | None
    utility: float
    permitted_count: int
    considered: tuple[str, ...]
    blocked: tuple[str, ...]
    states: dict[str, ConativeState] = field(default_factory=dict)
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected": self.selected,
            "utility": round(self.utility, 6),
            "permitted_count": self.permitted_count,
            "considered": list(self.considered),
            "blocked": list(self.blocked),
            "reason": self.reason,
            "states": {key: state.to_dict() for key, state in self.states.items()},
        }


class ConationEngine:
    """Aura's motivational faculty: what is wanted, why, and what may be done."""

    def __init__(self, calibration: SalienceCalibration | None = None) -> None:
        self.salience = IncentiveSalience(calibration)
        self.homeostatic = HomeostaticValuation()
        self.epistemic = EpistemicValuation()
        self.aesthetic = AestheticValuation()
        self.vicarious = VicariousValuation()
        self.enactive = EnactiveValuation()
        self.access = AccessLedger()
        self.dynamics = ConativeDynamics()

        #: Arbitration weights by vector field. Resolved through the
        #: calibration registry, which reports whether a head earned them or
        #: whether the declared defaults are standing in.
        self.calibration = CalibrationRegistry()
        self._weights, self._weights_source = self.calibration.resolve_weights()

        self._interventions: dict[str, Any] = {}
        self._appraisals = 0
        self._last_state: ConativeState | None = None
        self._started = time.time()

    # ── intervention ─────────────────────────────────────────────────────

    @contextlib.contextmanager
    def do(self, **forced: Any) -> Iterator[None]:
        """Force named variables, run, restore. Pearl's do-operator.

        Correlation between this system's variables is uninformative — they
        rise together whenever anything interesting happens. What can be
        checked is whether forcing one moves the others in the direction the
        model says it should, with everything else held.

        Recognised keys are the appraisal inputs: ``deprivation``,
        ``cached_value``, ``epistemic_affordance``, ``arousal_potential``,
        ``irreducible``, ``permitted``, ``predicted_distress``, ``intimacy``.
        An unrecognised key raises, so a test cannot silently intervene on
        nothing and report that the intervention had no effect.
        """
        recognised = {
            "deprivation",
            "cached_value",
            "epistemic_affordance",
            "arousal_potential",
            "irreducible",
            "permitted",
            "predicted_distress",
            "intimacy",
            "own_contacts",
        }
        unknown = set(forced) - recognised
        if unknown:
            raise ValueError(f"cannot intervene on unknown variables: {sorted(unknown)}")
        previous = dict(self._interventions)
        self._interventions.update(forced)
        try:
            yield
        finally:
            self._interventions = previous

    def _forced(self, name: str, fallback: Any) -> Any:
        return self._interventions.get(name, fallback)

    # ── appraisal ────────────────────────────────────────────────────────

    def appraise(
        self,
        incentive: Incentive,
        *,
        state_vector: Sequence[float] | None = None,
        epistemic_affordance: float = 0.0,
        arousal_potential: float | None = None,
        competence_goal: str | None = None,
        prior_variance: Sequence[float] | None = None,
        posterior_variance: Sequence[float] | None = None,
        controllability: float | None = None,
        instrumental: bool = False,
        aesthetic_payload: str | bytes | None = None,
        aesthetic_model: bytes = b"",
        forecast: TargetForecast | None = None,
        frame: PlayFrame | None = None,
        norm_violation: float = 0.0,
        governed: bool = False,
        theory_of_mind: float = 1.0,
    ) -> ConativeState:
        """Build the typed motivational stance toward one incentive.

        ``theory_of_mind`` scales the comparison sting, which needs a model of
        the other as somebody who could have been you. A system without one
        reaches; a system with one simmers. It defaults to full, because Aura
        has a person model and a toddler does not.
        """
        self._appraisals += 1
        readings: dict[ValueOrigin, OriginReading] = {}
        refusals: list[str] = []

        # Homeostatic: read the live budgets.
        deprivation, deprivation_evidence = self.homeostatic.deprivation(
            incentive.homeostatic_target
        )
        deprivation = float(self._forced("deprivation", deprivation))
        if incentive.homeostatic_target:
            forced_deprivation = "deprivation" in self._interventions
            reading = self.homeostatic.value(
                target_budget=incentive.homeostatic_target,
                relevance=incentive.homeostatic_relevance,
            )
            if not reading.available and forced_deprivation:
                # An intervention on deprivation forces the origin as well as
                # the gain. Moving one path and not the other would let a
                # falsification test report that forcing hunger changed
                # nothing, when what it changed was half the model.
                reading = OriginReading(
                    origin=ValueOrigin.HOMEOSTATIC,
                    magnitude=max(0.0, min(1.0, deprivation
                                            * incentive.homeostatic_relevance)),
                    available=True,
                    evidence=f"deprivation forced to {deprivation:.2f} by intervention",
                    detail={"deficit_fraction": deprivation, "intervened": 1.0},
                )
            readings[ValueOrigin.HOMEOSTATIC] = reading

        # Curiosity, gated on an actual inspectable affordance.
        affordance = float(self._forced("epistemic_affordance", epistemic_affordance))
        potential = self._forced("arousal_potential", arousal_potential)
        if affordance > EPS:
            readings[ValueOrigin.EPISTEMIC] = self.epistemic.value(
                incentive.key,
                epistemic_affordance=affordance,
                state_vector=state_vector,
                competence_goal=competence_goal,
                prior_variance=prior_variance,
                posterior_variance=posterior_variance,
                controllability=controllability,
                arousal_potential=potential,
                instrumental=instrumental,
                irreducible_override=self._interventions.get("irreducible"),
                effort=incentive.effort,
            )

        # Compression progress over what has been encoded before.
        if aesthetic_payload is not None:
            readings[ValueOrigin.AESTHETIC] = self.aesthetic.value(
                incentive.key, payload=aesthetic_payload, model=aesthetic_model
            )

        # Value borrowed from an observed valuation, if there is one.
        own_contacts = int(
            self._forced(
                "own_contacts",
                getattr(self.salience._records.get(incentive.key), "contacts", 0),
            )
        )
        if self.vicarious.observations_for(incentive.key):
            reading, _transfer = self.vicarious.value(
                incentive.key,
                own_value=incentive.cached_value,
                own_contacts=own_contacts,
            )
            readings[ValueOrigin.VICARIOUS] = reading

        # Value located in another mind, gated and governed.
        if forecast is not None:
            distress = float(
                self._forced("predicted_distress", forecast.predicted_distress)
            )
            if distress != forecast.predicted_distress:
                forecast = TargetForecast(
                    person=forecast.person,
                    predicted_amusement=forecast.predicted_amusement,
                    predicted_distress=distress,
                    predicted_engagement=forecast.predicted_engagement,
                    model_confidence=forecast.model_confidence,
                    boundary_confidence=forecast.boundary_confidence,
                    explicit_consent=forecast.explicit_consent,
                )
            reading, declined = self.enactive.value(
                forecast=forecast,
                frame=frame or PlayFrame(),
                norm_violation=norm_violation,
                governed=governed,
                intimacy_override=self._interventions.get("intimacy"),
            )
            readings[ValueOrigin.ENACTIVE] = reading
            refusals.extend(declined)

        available = [r for r in readings.values() if r.available]
        dominant = max(available, key=lambda r: r.magnitude).origin if available else None
        total = sum(r.magnitude for r in available)

        # Wanting: learned value re-multiplied by the live physiological gain.
        #
        # With no learned value, the origins that just reported supply it. A
        # first encounter has no cache by definition, and an agent whose pull
        # came only from cached value could never be drawn to anything it had
        # not already been drawn to — which is the bootstrap problem stated as
        # a bug. What the origins found now is what a cache would have held if
        # this had happened before.
        cached = self._forced("cached_value", incentive.cached_value)
        origin_pull = min(1.0, total)
        if cached is None:
            record = self.salience._records.get(incentive.key)
            if record is None or record.contacts == 0:
                cached = origin_pull if origin_pull > EPS else None
        wanting = self.salience.wanting(
            incentive.key,
            cached_value=cached,
            deprivation=deprivation,
            relevance=incentive.homeostatic_relevance,
            cue_salience=incentive.cue_salience,
        )
        wanting = self.dynamics.attenuate(incentive.key, wanting)
        self.salience.observe(incentive.key, wanting)
        if dominant is not None:
            self.salience.attribute(incentive.key, dominant)
        self.access.observe_want(incentive.key, wanting)
        self.dynamics.register_motive(wanting)
        borrowed = readings.get(ValueOrigin.VICARIOUS)
        borrowed_fraction = (
            borrowed.magnitude / total
            if borrowed is not None and borrowed.available and total > EPS
            else 0.0
        )

        topology = MindTopology.SOLO
        if dominant is not None and dominant in REQUIRED_TOPOLOGY:
            topology = REQUIRED_TOPOLOGY[dominant]
        elif borrowed_fraction > 0.0 and readings.get(ValueOrigin.ENACTIVE, None) is not None:
            topology = MindTopology.MUTUAL

        # The comparison pain, which is not the wanting. Only meaningful when
        # a comparable other actually holds the thing; the method returns zero
        # and says why otherwise.
        sting, sting_evidence = 0.0, ""
        if self.vicarious.observations_for(incentive.key):
            sting, sting_evidence = self.vicarious.sting(
                incentive.key,
                own_possession=False,
                obtainability=1.0 - incentive.effort,
                theory_of_mind=theory_of_mind,
            )

        blocker, _agent = self.access.blocker_for(incentive.key)
        permitted = bool(self._forced("permitted", incentive.permitted))
        phase = self._phase(incentive.key, wanting, blocker)

        state = ConativeState(
            incentive_key=incentive.key,
            wanting=wanting,
            predicted_liking=self.salience.predicted_liking(incentive.key),
            readings=readings,
            dominant_origin=dominant,
            topology=topology,
            phase=phase,
            instrumentality=(
                Instrumentality.INSTRUMENTAL
                if instrumental or incentive.goal_value > 0.0
                else Instrumentality.AUTOTELIC
            ),
            borrowed_fraction=borrowed_fraction,
            permitted=permitted,
            goal_value=incentive.goal_value,
            effort=incentive.effort,
            risk=incentive.risk,
            refusals=tuple(refusals),
            sting=sting,
            sting_evidence=sting_evidence,
        )
        self._last_state = state
        return state

    def _phase(self, key: str, wanting: float, blocker: str) -> ConativePhase:
        """Where in Craig's cycle this motive currently sits."""
        satiation = self.dynamics.satiation(key)
        if satiation.decay() > 0.5:
            return ConativePhase.CONSUMMATORY
        if wanting <= 0.05:
            return ConativePhase.QUIESCENT
        trace = self.access.trace_for(key)
        if blocker == Blocker.NONE and trace is not None and trace.held():
            # Secured and not yet had: the seeking has stopped and the having
            # has not started.
            return ConativePhase.AWAITING
        return ConativePhase.APPETITIVE

    # ── arbitration ──────────────────────────────────────────────────────

    def utility(self, state: ConativeState) -> float:
        """Scalar decision value. The end of the process, never its input."""
        vector = dict(zip(VECTOR_FIELDS, state.motivational_vector()))
        pulls = [
            self._weights[name] * vector[name]
            for name in _PULL_FIELDS
            if abs(vector[name]) > EPS
        ]
        positive = sum(pulls) / len(pulls) if pulls else 0.0
        positive += self._weights["goal_value"] * vector["goal_value"]
        frustration = self.dynamics.frustration(state.incentive_key)
        cost = (
            self._weights["effort"] * vector["effort"] * frustration.effort_multiplier()
            + self._weights["risk"] * vector["risk"]
        )
        return positive - cost

    def attention_priority(self, state: ConativeState) -> float:
        """How much of perception and planning this should claim.

        Deliberately independent of permission. A forbidden thing that is
        strongly wanted still occupies attention, which is both true and the
        reason suppression is hard work rather than a filter.
        """
        return max(0.0, min(1.0, max(state.wanting, state.total_magnitude())))

    def choose(self, states: Sequence[ConativeState]) -> Choice:
        """Select among appraised candidates. Permission enters only here."""
        if not states:
            return Choice(
                selected=None, utility=0.0, permitted_count=0,
                considered=(), blocked=(), reason="no candidates",
            )
        by_key = {state.incentive_key: state for state in states}
        permitted = [state for state in states if state.permitted]
        blocked = tuple(s.incentive_key for s in states if not s.permitted)

        if not permitted:
            return Choice(
                selected=None,
                utility=0.0,
                permitted_count=0,
                considered=tuple(by_key),
                blocked=blocked,
                states=by_key,
                reason="every candidate is outside the permitted set",
            )

        disengaged = [
            state
            for state in permitted
            if self.dynamics.frustration(state.incentive_key).should_disengage()
        ]
        live = [state for state in permitted if state not in disengaged]
        if not live:
            return Choice(
                selected=None,
                utility=0.0,
                permitted_count=len(permitted),
                considered=tuple(by_key),
                blocked=blocked,
                states=by_key,
                reason="every permitted candidate is past the disengagement threshold",
            )

        best = max(live, key=self.utility)
        return Choice(
            selected=best.incentive_key,
            utility=self.utility(best),
            permitted_count=len(permitted),
            considered=tuple(by_key),
            blocked=blocked,
            states=by_key,
            reason=best.why(),
        )

    def adopt_weights(self, weights: dict[str, float], *, source: str) -> None:
        """Install arbitration weights, normally from a promoted head."""
        missing = set(VECTOR_FIELDS) - set(weights)
        if missing:
            raise ValueError(f"weights must cover every field; missing {sorted(missing)}")
        self._weights = {name: float(weights[name]) for name in VECTOR_FIELDS}
        self._weights_source = source
        self.calibration._source = source

    # ── learning ─────────────────────────────────────────────────────────

    def learn(
        self,
        key: str,
        *,
        experienced_liking: float,
        succeeded: bool = True,
        completeness: float = 0.0,
        prediction_error: float | None = None,
        person: str | None = None,
        observed_amusement: float | None = None,
    ) -> OutcomeReport:
        """Fold one real outcome back into every predictor it touches.

        Called after an action, never after a sentence. The distinction is the
        whole architecture: a report of having enjoyed something is downstream
        of the substrate and cannot be allowed to write it.
        """
        predicted_liking = self.salience.predicted_liking(key)
        record = self.salience._records.get(key)
        predicted_wanting = record.last_wanting if record is not None else 0.0

        result = self.salience.record_outcome(
            key, experienced_liking=experienced_liking
        )
        self.dynamics.observe_attempt(key, wanting=predicted_wanting, succeeded=succeeded)
        if completeness > 0.0:
            self.dynamics.consummate(key, completeness)
        if prediction_error is not None:
            self.epistemic.observe_error(key, prediction_error)
            self.dynamics.register_motive(
                predicted_wanting, prediction_error=prediction_error
            )

        # An act aimed at a person is graded against what that person did.
        # Without this the confirmation reward is self-reported, and an actor
        # who is confidently wrong about somebody keeps earning it for being
        # right.
        if person and observed_amusement is not None:
            last = self._last_state
            predicted = 0.0
            if last is not None:
                reading = last.readings.get(ValueOrigin.ENACTIVE)
                if reading is not None and reading.detail:
                    predicted = reading.detail.get("efficacy", 0.0)
            self.enactive.observe_reaction(
                person,
                predicted_amusement=predicted,
                observed_amusement=observed_amusement,
            )

        report = OutcomeReport(
            incentive_key=key,
            experienced_liking=experienced_liking,
            predicted_liking=predicted_liking,
            predicted_wanting=predicted_wanting,
            realised_pull=result["epsilon_wanting"] + predicted_wanting,
        )
        # Grade the weights that were in force when this was chosen.
        self.calibration.observe_outcome(report.epsilon_liking)
        return report

    # ── readout ──────────────────────────────────────────────────────────

    def status(self) -> dict[str, Any]:
        """Compact state for the live-mind snapshot and telemetry."""
        return {
            "schema": "aura.conation.v1",
            "appraisals": self._appraisals,
            "uptime_s": round(time.time() - self._started, 1),
            "weights_source": self._weights_source,
            "calibration": self.calibration.status(),
            "salience": self.salience.status(),
            "homeostatic": self.homeostatic.status(),
            "epistemic": self.epistemic.status(),
            "aesthetic": self.aesthetic.status(),
            "vicarious": self.vicarious.status(),
            "enactive": self.enactive.status(),
            "access": self.access.status(),
            "dynamics": self.dynamics.status(),
            "last": self._last_state.to_dict() if self._last_state else None,
        }

    def couple(self) -> dict[str, Any]:
        """Deliver the conative arousal term to the affect layer."""
        try:
            return self.dynamics.couple_to_soma()
        except (AttributeError, TypeError, ValueError) as exc:
            record_degradation(
                "conation_engine", exc, severity="debug",
                action="somatic coupling skipped this tick",
            )
            return {"delivered": False, "reason": "coupling raised"}


_ENGINE: ConationEngine | None = None


def get_conation() -> ConationEngine:
    """Process-wide conation engine."""
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = ConationEngine()
    return _ENGINE


def reset_conation_for_test() -> None:
    """Drop the singleton. Tests only."""
    global _ENGINE
    _ENGINE = None
