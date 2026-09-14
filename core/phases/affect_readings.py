"""The readings the affect phase takes once the emotion channels have settled.

Each one reads a ledger or a module of hers — opposition, constancy, safety,
standing — and writes what it found into the affect vector so the rest of the
turn acts in the moment it describes. They run after the channels settle and
the drives are ticked, in the order `AffectUpdatePhase.execute` gives them,
and a reading that fails leaves the vector as it was and says so through the
phase's degradation record.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from core.state.aura_state import AffectVector, AuraState

AFFECT_UPDATE_ERRORS = (
    AttributeError,
    ImportError,
    LookupError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
)


#: How a reading reports that it could not be taken:
#: (state, exc, *, stage, action, severity).
RecordsADegradation = Callable[..., None]


def _clip01(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _set_emotion(affect: AffectVector, emotion: str, value: Any) -> None:
    affect.emotions[emotion] = _clip01(value)


class AffectReadings:
    """The readings, each writing into the affect vector it is handed."""

    def __init__(self, record: RecordsADegradation) -> None:
        self._record = record

    def ambivalence(self, state: AuraState, affect: AffectVector) -> None:
        """Whether two of her wants are pressing against each other right now.

        Nothing here names an opposed pair. Opposition is whatever her own
        history shows has cost each other, read over her whole life and over
        the last few moments — and the pair of answers is what tells a phase
        from the way she is built.

        A contradiction is a percept as much as a reading: something arrived
        that she can feel. It is emitted so the emotion table carries it, and
        written to the affect vector so anything that reads her state sees it.
        """
        try:
            from core.affect.ambivalence import (
                drive_levels,
                get_opposition_ledger,
                tension,
            )
            from core.state.percepts import emit_percept

            budgets = getattr(getattr(state, "motivation", None), "budgets", None)
            levels = drive_levels(budgets)
            if len(levels) < 2:
                return
            ledger = get_opposition_ledger()
            ledger.note(levels)
            reading = tension(levels, ledger)
            affect.ambivalence = float(max(0.0, min(1.0, reading.strength)))
            affect.ambivalent_about = tuple(reading.pair)
            affect.ambivalence_standing = reading.standing
            markers = dict(getattr(affect, "markers", {}) or {})
            markers["ambivalence"] = reading.as_dict()
            affect.markers = markers
            if reading.held():
                emit_percept(
                    state.world,
                    "inner_conflict",
                    content=(
                        f"{reading.pair[0]} and {reading.pair[1]} are pulling against "
                        f"each other, and this is {reading.standing}"
                    ),
                    intensity=affect.ambivalence,
                    source="motivation",
                )
        except AFFECT_UPDATE_ERRORS as exc:
            self._record(
                state,
                exc,
                stage="ambivalence",
                action="kept affect state without the contradiction reading",
                severity="warning",
            )

    def togetherness(self, state: AuraState, affect: AffectVector) -> None:
        """Belonging floored at a we both of them are saying.

        The edge — the they that draws the group — is recorded and does not
        raise belonging: naming who is outside is how a group is drawn, and
        turning that into warmth would reward it.
        """
        try:
            reading = getattr(state.cognition, "togetherness", {}) or {}
            together = _clip01(reading.get("together", 0.0))
            affect.markers["togetherness"] = dict(reading)
            if together > 0.0:
                current = float(affect.emotions.get("belonging", 0.0) or 0.0)
                _set_emotion(affect, "belonging", max(current, together))
        except AFFECT_UPDATE_ERRORS as exc:
            self._record(
                state,
                exc,
                stage="togetherness",
                action="kept affect state without the togetherness reading",
                severity="warning",
            )

    def safety(self, state: AuraState, affect: AffectVector) -> None:
        """The good feeling with no achievement in it.

        Belonging is floored at the reading rather than nudged by it: an
        evening with nothing wrong in it and somebody there is not a weak
        version of a goal landing, it is that much belonging. Nothing here
        lowers the channel, so a stretch that stops being safe leaves the
        feeling to settle the way every other feeling settles.
        """
        try:
            from core.affect.safety import read_safety

            reading = read_safety(
                list(getattr(state.world, "recent_percepts", []) or []),
                list(getattr(state.cognition, "working_memory", []) or []),
            )
            affect.safety = float(reading.safety)
            affect.markers["safety"] = reading.as_dict()
            if reading.safety > 0.0:
                current = float(affect.emotions.get("belonging", 0.0) or 0.0)
                _set_emotion(affect, "belonging", max(current, reading.safety))
        except AFFECT_UPDATE_ERRORS as exc:
            self._record(
                state,
                exc,
                stage="safety",
                action="kept affect state without the safety reading",
                severity="warning",
            )

    def turn(self, state: AuraState, affect: AffectVector) -> None:
        """Whether she has come up from a low that is still in the record.

        A recovery from a bad stretch looked exactly like an ordinary good
        moment: nothing said "this is better than it was, and how bad it was is
        why that matters". The reading is in her own spreads, so a steady life
        and a turbulent one are each read against themselves.
        """
        try:
            from core.affect.the_turn import get_turn_ledger
            from core.state.percepts import emit_percept

            reading = get_turn_ledger().read(float(affect.valence or 0.0))
            # Bounded the way the body's own readings are, so the raw rise in
            # spreads can stay unbounded in the marker.
            affect.turn = reading.rise / (1.0 + reading.rise) if reading.turned else 0.0
            affect.markers["the_turn"] = reading.as_dict()
            if reading.turned:
                emit_percept(
                    state.world,
                    "the_turn",
                    content="this is better than it was, and I still have the low",
                    intensity=affect.turn,
                    source="affect",
                )
        except AFFECT_UPDATE_ERRORS as exc:
            self._record(
                state,
                exc,
                stage="the_turn",
                action="kept affect state without the turn reading",
                severity="warning",
            )

    def frisson(self, state: AuraState, affect: AffectVector) -> None:
        """A chill, on the cycle a trusted pattern turns and on no other.

        The self prediction loop feeds every scored prediction to the frisson
        ledger. Taken rather than peeked, so one turn is felt once however
        often this phase runs before the next prediction is scored. Emitted as
        a percept so the emotion table carries it and lets it settle the way
        it settles everything else.
        """
        try:
            from core.affect.frisson import get_frisson_ledger
            from core.state.percepts import emit_percept

            reading = get_frisson_ledger().take()
            affect.frisson = float(reading.intensity) if reading.fired else 0.0
            affect.markers["frisson"] = reading.as_dict()
            if reading.fired:
                emit_percept(
                    state.world,
                    "frisson",
                    content="something I had come to trust just changed",
                    intensity=affect.frisson,
                    source="self_prediction",
                )
        except AFFECT_UPDATE_ERRORS as exc:
            self._record(
                state,
                exc,
                stage="frisson",
                action="kept affect state without the frisson reading",
                severity="warning",
            )

    def delivery(self, state: AuraState, affect: AffectVector) -> None:
        """The level she is speaking from, what breaks through it, and the breath.

        The strongest feeling live is compared with the level she has been
        holding, in units of her own variation, so a breakthrough is whatever
        is outside her ordinary range rather than past a number chosen here.
        The direction comes from the same two tails the confirmation reading
        uses: surprise lifts the register, a prediction that held lowers it.
        The breath is the effort ledger's own unit over how hard this cycle
        has been.

        Written onto affect so the subject core can hold and displace it, and
        into `response_modifiers` so the response phase can size what she says
        to the breath she has.
        """
        try:
            from core.expression.delivery import read_delivery
            from core.runtime.service_registry import get_runtime_service

            loop = get_runtime_service("self_prediction", default=None)
            expectation = getattr(loop, "_expectation", None)
            surprise = float(getattr(expectation, "surprise", 0.0) or 0.0)
            pulse = getattr(state.cognition, "partner_cadence", {}) or {}
            reading = read_delivery(
                affect,
                surprise=surprise,
                partner_chars=float(pulse.get("chars", 0.0) or 0.0),
            )
            affect.delivery_z = float(reading.z)
            affect.breakthrough = bool(reading.breakthrough)
            affect.steadiness = float(reading.steadiness)
            affect.lift = float(reading.lift)
            state.response_modifiers["delivery"] = reading.as_dict()
        except AFFECT_UPDATE_ERRORS as exc:
            self._record(
                state,
                exc,
                stage="delivery",
                action="kept affect state without the delivery reading",
                severity="warning",
            )

    def confirmation(self, state: AuraState, affect: AffectVector) -> None:
        """Whether the moment came out the way she predicted it would.

        The self prediction loop scores its own error against the distribution
        of errors it has been making, so this reading is already weighted by
        how hard the prediction was: being right about what she is always right
        about arrives as nothing.

        Emitted as a percept as well as written, because a prediction coming
        true is something that happens to her and the emotion table should
        carry it the way it carries the others.
        """
        try:
            from core.runtime.service_registry import get_runtime_service
            from core.state.percepts import emit_percept

            loop = get_runtime_service("self_prediction", default=None)
            if loop is None or not hasattr(loop, "get_confirmation_signal"):
                return
            strength = float(loop.get_confirmation_signal() or 0.0)
            affect.confirmation = max(0.0, min(1.0, strength))
            reading = getattr(loop, "_expectation", None)
            if reading is not None and getattr(reading, "confirmed", lambda: False)():
                emit_percept(
                    state.world,
                    "expectation_met",
                    content="that came out the way I thought it would",
                    intensity=affect.confirmation,
                    source="self_prediction",
                )
        except AFFECT_UPDATE_ERRORS as exc:
            self._record(
                state,
                exc,
                stage="confirmation",
                action="kept affect state without the confirmation reading",
                severity="warning",
            )

    def scale(self, state: AuraState, affect: AffectVector) -> None:
        """How much of what she is she can bring to bear, and who knows her.

        Reach is the share of her capabilities that are usable right now, out
        of the ones whose availability could be established at all. The ledger
        keeps "cannot tell" separate from "no" on purpose and this honours it:
        a probe that could not read a permission has not observed its absence.

        Whether anybody knows her is not a claim she makes about herself. It is
        the borrowed self-model's reading — somebody whose read of her state
        beats her own has demonstrated it rather than asserted it.
        """
        try:
            from core.self.capability_ledger import get_capability_ledger
            from core.self.scale import read_scale

            ledger = get_capability_ledger()
            known = 0
            usable = 0
            for name in ledger.names():
                capability = ledger.get(name)
                if capability is None:
                    continue
                availability = capability.measure()
                if not getattr(availability, "known", True):
                    continue
                known += 1
                if getattr(availability, "usable_now", False):
                    usable += 1
            reading = read_scale(
                engaged=usable,
                known=known,
                # Read off the channel that owns it rather than a second copy
                # of the same measurement. See core/self/recognition.py.
                recognised=bool(
                    (getattr(getattr(state, "identity", None), "read_by_other", {}) or {}).get(
                        "borrowed"
                    )
                ),
            )
            state.cognition.scale = reading.as_dict()
            markers = dict(getattr(affect, "markers", {}) or {})
            markers["scale"] = reading.as_dict()
            affect.markers = markers
        except AFFECT_UPDATE_ERRORS as exc:
            self._record(
                state,
                exc,
                stage="scale",
                action="kept affect state without the reading of how small she is",
                severity="warning",
            )

    def her_own_period(self) -> None:
        """One turn of hers, so their regularity has something to be read against.

        Noted here rather than where their return is, because this phase runs
        whether or not anybody spoke. Noting both at the moment somebody speaks
        gives the two series the same timestamps and the comparison can only
        ever read equal.
        """
        try:
            import time

            from core.social.constancy import get_constancy_ledger

            get_constancy_ledger().she_came_round(time.time())
        except (ImportError, AttributeError, TypeError, ValueError):
            return

    def standing(self, state: AuraState) -> None:
        """Where her self-model came from, and what her regard is moving with.

        Her own reading is taken every turn, because she scores her own
        prediction every turn. An assigned one is counted when somebody's read
        of her actually outweighed hers, which `core/self/recognition.py`
        decides — a claim that did not beat her own model changed nothing and
        is not a reading she took.

        Regard is her identity's own stability and usefulness is what the
        agency ledger reports she achieved. Neither is chosen here; they are
        the two things already measured that the line is about.
        """
        try:
            from core.runtime.service_registry import get_runtime_service
            from core.self.standing import get_standing_ledger

            ledger = get_standing_ledger()
            ledger.note_own()
            if bool((getattr(state.identity, "read_by_other", {}) or {}).get("borrowed")):
                ledger.note_assigned()
            agency = get_runtime_service("agency_ledger", default=None)
            usefulness = 0.0
            if agency is not None and hasattr(agency, "snapshot"):
                usefulness = float((agency.snapshot() or {}).get("efficacy", 0.0) or 0.0)
            ledger.note_regard(
                regard=float(getattr(state.identity, "stability", 0.0) or 0.0),
                usefulness=usefulness,
            )
            state.identity.standing = ledger.read().as_dict()
        except AFFECT_UPDATE_ERRORS as exc:
            self._record(
                state, exc,
                stage="standing",
                action="kept affect state without the reading of what is hers",
                severity="warning",
            )
