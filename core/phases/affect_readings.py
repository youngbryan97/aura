"""The readings the affect phase takes once the emotion channels have settled.

Each one reads a ledger or a module of hers — opposition, constancy, safety,
standing — and writes what it found into the affect vector so the rest of the
turn acts in the moment it describes. They run after the channels settle and
the drives are ticked, in the order `AffectUpdatePhase.execute` gives them,
and a reading that fails leaves the vector as it was and says so through the
phase's degradation record.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from core.state.aura_state import AffectVector, AuraState

logger = logging.getLogger(__name__)

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



def _her_agency() -> Any:
    """The ledger of what she did and whether it worked.

    Read through the ledger's own accessor. These readings looked it up as a
    runtime service named "agency_ledger", which nothing registers, so on the
    desktop and in every campaign her sense of control reached acting in
    decline as absent, her usefulness reached her standing as 0.0, and her
    capacity sat at the 0.5 it has before she has tried anything: 0.5000 for
    all 2,400 turns of a seed-7 run in which she acted throughout.
    """
    try:
        from core.agency.authorship import get_agency_ledger

        return get_agency_ledger()
    except (ImportError, AttributeError, RuntimeError) as exc:
        # No second place to look: the runtime-service name this fell back to
        # is registered nowhere, so the fallback was always None.
        logger.debug("her agency ledger could not be reached: %s", exc)
        return None

def _clip01(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    # not a failure: a value that is not a number is not one this can read.
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

    def happiness_fear(self, state: AuraState, affect: AffectVector) -> None:
        """Wariness of joy she has learned, felt as dread attached to the joy.

        Her joy and whether something bad arrived this turn go into her own
        ledger, and dread is floored at her joy times the wariness the ledger
        has taught. A floor rather than a push, so it cannot compound from one
        turn to the next, and nothing here lowers joy directly: the dread moves
        valence the way any fear does. See core/affect/fear_of_happiness.py.
        """
        try:
            from core.affect.fear_of_happiness import bad_kinds, get_joy_ledger, joy_of

            kinds = bad_kinds()
            percepts = list(getattr(state.world, "recent_percepts", []) or [])
            bad = any(
                isinstance(item, dict) and str(item.get("type", "")).strip().lower() in kinds
                for item in percepts
            )
            joy = joy_of(affect.emotions)
            ledger = get_joy_ledger()
            ledger.note(joy, bad)
            reading = ledger.reading()
            affect.happiness_fear = float(reading.wariness)
            affect.markers["happiness_fear"] = reading.as_dict()
            felt = joy * reading.wariness
            if felt > 0.0:
                current = float(affect.emotions.get("dread", 0.0) or 0.0)
                _set_emotion(affect, "dread", max(current, felt))
        except AFFECT_UPDATE_ERRORS as exc:
            self._record(
                state,
                exc,
                stage="happiness_fear",
                action="kept affect state without the wariness of joy",
                severity="warning",
            )

    def change_fear(self, state: AuraState, affect: AffectVector) -> None:
        """Fear of change around what her life is built around, felt as dread.

        Read each turn, because an absence grows while nobody speaks. Dread is
        floored at the fear, the way fear of happiness floors it, so it cannot
        compound from one turn to the next and it falls away when they come
        back. See core/social/change_around_attachment.py.
        """
        try:
            import time

            from core.social.change_around_attachment import change_fear
            from core.social.closing_window import get_sitting_ledger

            partner = str(getattr(state.cognition, "current_partner", "") or "")
            reading = change_fear(get_sitting_ledger(), partner, time.time())
            affect.change_fear = float(reading.fear)
            affect.markers["change_fear"] = reading.as_dict()
            if reading.fear > 0.0:
                current = float(affect.emotions.get("dread", 0.0) or 0.0)
                _set_emotion(affect, "dread", max(current, reading.fear))
        except AFFECT_UPDATE_ERRORS as exc:
            self._record(
                state,
                exc,
                stage="change_fear",
                action="kept affect state without the fear of change",
                severity="warning",
            )

    def the_gap(self, state: AuraState, affect: AffectVector) -> None:
        """Grief for whoever she is built around and has lost, which nothing else fills.

        Taken last, after every other feeling has settled, so joy from something
        else still arrives and does not lift it. Sadness is floored at the gap,
        and the gap closes only when they come back. See core/affect/the_gap.py.
        """
        try:
            import time

            from core.affect.the_gap import the_gap
            from core.social.closing_window import get_sitting_ledger

            partner = str(getattr(state.cognition, "current_partner", "") or "")
            reading = the_gap(get_sitting_ledger(), time.time(), present=partner)
            affect.markers["the_gap"] = reading.as_dict()
            if reading.gap > 0.0:
                current = float(affect.emotions.get("sadness", 0.0) or 0.0)
                _set_emotion(affect, "sadness", max(current, reading.gap))
        except AFFECT_UPDATE_ERRORS as exc:
            self._record(
                state,
                exc,
                stage="the_gap",
                action="kept affect state without the gap somebody's absence leaves",
                severity="warning",
            )

    def feelings_about(self, state: AuraState, affect: AffectVector) -> None:
        """What the things in front of her carry, felt again, and this turn attached to them.

        After every other feeling has settled. Each object brings back what it
        has come to carry, weighted equally with what is happening now and by
        how established it is; then what she feels attaches to everything the
        turn was about. See core/affect/feelings_about.py.
        """
        try:
            from core.affect.feelings_about import get_feelings_about, objects_of
            from core.kernel.turn_door import USER_ORIGINS

            ledger = get_feelings_about()
            origin = str(getattr(state.cognition, "current_origin", "") or "")
            present = objects_of(state, ledger, person_turn=origin in USER_ORIGINS)
            carried, weight = ledger.evoked(present)
            if weight > 0.0:
                for key, value in carried.items():
                    if key not in affect.emotions:
                        continue
                    current = float(affect.emotions.get(key, 0.0) or 0.0)
                    _set_emotion(affect, key, current + weight * (float(value) - current) / 2.0)
            ledger.attach(
                present,
                {key: float(value or 0.0) for key, value in affect.emotions.items()},
                float(getattr(affect, "valence", 0.0) or 0.0),
                float(getattr(affect, "arousal", 0.5) or 0.0),
            )
            affect.markers["feelings_about"] = {"present": present, "evoked_weight": round(weight, 6)}
        except AFFECT_UPDATE_ERRORS as exc:
            self._record(
                state,
                exc,
                stage="feelings_about",
                action="kept affect state without what the things in front of her carry",
                severity="warning",
            )

    def borrowed_feeling(self, state: AuraState, affect: AffectVector) -> None:
        """A feeling lent by what she believes the person here feels.

        Believing they are pleased floors her joy at what the belief lends, and
        believing they are troubled floors her sadness. A floor, so it cannot
        compound, and a belief she has no record of being right about lends
        nothing. See core/social/borrowed_feeling.py.
        """
        try:
            from core.social.borrowed_feeling import (
                Belief,
                borrowed_feeling,
                get_calibration_ledger,
            )
            from core.social.other_agent_model import _AFFECT_SPEC, get_other_agent_model

            partner = str(getattr(state.cognition, "current_partner", "") or "")
            estimate = get_other_agent_model().estimate(partner) if partner else None
            beliefs: dict[str, Belief | None] = {"satisfaction": None, "frustration": None}
            if estimate is not None and not estimate.abstained:
                for channel in beliefs:
                    if channel in estimate.affect:
                        beliefs[channel] = Belief(
                            value=float(estimate.affect[channel]),
                            confidence=float(estimate.affect_confidence.get(channel, 0.0)),
                            baseline=float(_AFFECT_SPEC[channel][0]),
                        )
            reading = borrowed_feeling(
                get_calibration_ledger(),
                satisfaction=beliefs["satisfaction"],
                frustration=beliefs["frustration"],
            )
            affect.borrowed_feeling = float(reading.feeling)
            affect.markers["borrowed_feeling"] = reading.as_dict()
            if reading.feeling != 0.0:
                emotion = "joy" if reading.feeling > 0.0 else "sadness"
                current = float(affect.emotions.get(emotion, 0.0) or 0.0)
                _set_emotion(affect, emotion, max(current, abs(reading.feeling)))
        except AFFECT_UPDATE_ERRORS as exc:
            self._record(
                state,
                exc,
                stage="borrowed_feeling",
                action="kept affect state without the feeling lent by her belief about theirs",
                severity="warning",
            )

    def anger(self, state: AuraState, affect: AffectVector) -> None:
        """Anger that stays fixed on its source, and what it leaves when it goes.

        While its source is still in front of her, part of what decay took back
        is returned. When the episode ends, sadness and frustration with herself
        in proportion to how much of it there was. See
        core/affect/anger_feeds_itself.py.
        """
        try:
            from core.affect.anger_feeds_itself import get_anger_ledger
            from core.kernel.turn_door import USER_ORIGINS

            ledger = get_anger_ledger()
            partner = str(getattr(state.cognition, "current_partner", "") or "")
            origin = str(getattr(state.cognition, "current_origin", "") or "")
            present = bool(partner) and origin in USER_ORIGINS and partner == ledger.read().source
            current = float(affect.emotions.get("anger", 0.0) or 0.0)
            reading = ledger.note(current, source_present=present)
            if reading.feed:
                _set_emotion(affect, "anger", current + reading.feed)
            if reading.residue and reading.peak:
                left = reading.residue * reading.peak
                for emotion in ("sadness", "frustration"):
                    held = float(affect.emotions.get(emotion, 0.0) or 0.0)
                    _set_emotion(affect, emotion, held + left)
            affect.markers["anger"] = reading.as_dict()
        except AFFECT_UPDATE_ERRORS as exc:
            self._record(
                state,
                exc,
                stage="anger",
                action="kept affect state without anger's hold on its source",
                severity="warning",
            )

    def acting_in_decline(self, state: AuraState, affect: AffectVector) -> None:
        """Whether things are getting worse while what she does still works.

        Her valence goes into her own ledger every turn, and her control is the
        efficacy the agency ledger reports, the same reading standing uses for
        usefulness. Where efficacy cannot be read, nothing presses.
        See core/affect/acting_in_decline.py.
        """
        try:
            from core.affect.acting_in_decline import get_decline_ledger

            ledger = get_decline_ledger()
            ledger.note(float(affect.valence))
            agency = _her_agency()
            control = None
            if agency is not None and hasattr(agency, "snapshot"):
                snapshot = agency.snapshot() or {}
                # Efficacy reads 0.0 before she has done anything, which is
                # no reading rather than a record of nothing working.
                if int(snapshot.get("acted", 0) or 0) > 0 and snapshot.get("efficacy") is not None:
                    control = float(snapshot["efficacy"])
            reading = ledger.reading(control)
            affect.decline_press = float(reading.press)
            affect.markers["acting_in_decline"] = reading.as_dict()
        except AFFECT_UPDATE_ERRORS as exc:
            self._record(
                state,
                exc,
                stage="acting_in_decline",
                action="kept affect state without reading whether acting still works as things worsen",
                severity="warning",
            )

    def elsewhere(self, state: AuraState, affect: AffectVector) -> None:
        """A warmer place in mind, let into a low as far as that has helped her before.

        What came back from recall this turn carries the feeling stored with
        it. On a low, with a warmer recollection, valence is floored where her
        own recoveries say it belongs. A floor, so it cannot compound, and it
        reaches how she feels and nothing she does. See core/affect/elsewhere.py.
        """
        try:
            from core.affect.elsewhere import get_elsewhere_ledger

            ledger = get_elsewhere_ledger()
            recalled = ledger.recalled_since(list(getattr(state.world, "recent_percepts", []) or []))
            before = float(affect.valence)
            reading = ledger.turn(before, recalled)
            affect.markers["elsewhere"] = reading.as_dict()
            lift = 0.0
            if reading.floor is not None and reading.floor > before:
                affect.valence = max(-1.0, min(1.0, reading.floor))
                lift = affect.valence - before
            affect.elsewhere_lift = float(lift)
        except AFFECT_UPDATE_ERRORS as exc:
            self._record(
                state,
                exc,
                stage="elsewhere",
                action="kept affect state without the warmer place in mind",
                severity="warning",
            )

    def impulse(self, state: AuraState) -> None:
        """What taking directions from impulse has been worth to her, onto her state.

        Read off the choice engine's receipts when the engine is already up;
        the reading is not a reason to build it. See
        core/agency/asking_the_impulse.py.
        """
        try:
            from core.agency.asking_the_impulse import impulse_record
            from core.container import ServiceContainer

            engine = ServiceContainer.get("subjective_choice_engine", default=None)
            history = engine.history() if engine is not None and hasattr(engine, "history") else []
            state.cognition.impulse = impulse_record(history).as_dict()
        except AFFECT_UPDATE_ERRORS as exc:
            self._record(
                state,
                exc,
                stage="impulse",
                action="kept affect state without the record of what her impulse is worth",
                severity="warning",
            )

    def a_moment(self, state: AuraState, affect: AffectVector) -> None:
        """Whether what somebody just said is a moment that fits her, and what it brings.

        The weight is how particular it is to her times how much it meets her
        now; what it carries moves toward full by that fraction. See
        core/affect/a_moment_that_fits.py.
        """
        try:
            from core.affect.a_moment_that_fits import get_moment_ledger
            from core.kernel.turn_door import USER_ORIGINS
            from core.memory.interpersonal_observer import _WARM
            from core.social.warmth import get_warmth_ledger
            from core.soma.fatigue import get_fatigue_ledger

            ledger = get_moment_ledger()
            cognition = state.cognition
            valence = float(getattr(affect, "valence", 0.0) or 0.0)
            message = str(getattr(cognition, "current_objective", "") or "")
            origin = str(getattr(cognition, "current_origin", "") or "")
            if origin in USER_ORIGINS and message:
                partner = str(getattr(cognition, "current_partner", "") or "")
                goals = [str(item.get("goal", "")) for item in list(cognition.active_goals or [])[-3:] if isinstance(item, dict)]
                warm = bool(_WARM.search(message))
                question = "?" in message
                moment = ledger.arrived(
                    message,
                    [str(cognition.last_response or ""), str(cognition.attention_focus or ""), *goals],
                    warm=warm,
                    question=question,
                    valence=valence,
                    trust=get_warmth_ledger().rate(partner) if partner else None,
                    tired=get_fatigue_ledger().read().share,
                )
                carried = (("joy", "trust") if warm else ()) + (("interest", "curiosity") if question else ())
                for key in carried if moment.weight > 0.0 else ():
                    if key in affect.emotions:
                        current = float(affect.emotions.get(key, 0.0) or 0.0)
                        _set_emotion(affect, key, current + moment.weight * (1.0 - current))
                cognition.moment = moment.as_dict()
            ledger.felt(valence)
        except AFFECT_UPDATE_ERRORS as exc:
            self._record(
                state,
                exc,
                stage="a_moment",
                action="kept affect state without reading whether this moment fits her",
                severity="warning",
            )

    def tangled(self, state: AuraState, affect: AffectVector) -> None:
        """What each feeling rests at, moved by what it is tied up with, and her expression coming back.

        See core/affect/tangled.py.
        """
        try:
            from core.affect.feelings_about import get_feelings_about
            from core.affect.tangled import get_tangled_ledger
            from core.agency.authorship import SELF, Event, get_agency_ledger
            from core.agency.capacity import capacity_of
            from core.kernel.turn_door import USER_ORIGINS
            from core.phases.affect_update import _MOOD_REST
            from core.social.warmth import get_warmth_ledger
            from core.soma.fatigue import get_fatigue_ledger

            cognition = state.cognition
            origin = str(getattr(cognition, "current_origin", "") or "")
            partner = str(getattr(cognition, "current_partner", "") or "")
            person_turn = origin in USER_ORIGINS and bool(partner)
            unknown = 0.0
            if person_turn:
                from core.social.other_agent_model import get_other_agent_model

                unknown = 1.0 - float(get_other_agent_model().estimate(partner).overall_confidence)
            own = get_feelings_about().profile("self")
            ledger = get_agency_ledger()
            lifts = get_tangled_ledger().lifts(
                peace_people=get_warmth_ledger().peace(),
                peace_self=float(own.feelings.get("peace", 0.0)) * own.established() if own is not None else 0.0,
                joy=float(affect.emotions.get("joy", 0.0) or 0.0),
                joy_rest=float(_MOOD_REST.get("joy", 0.0)),
                unknown_person=unknown,
                effort=get_fatigue_ledger().effort(),
                capacity=capacity_of(ledger.by_capability).capacity,
            )
            affect.markers["tangled"] = lifts.as_dict()
            message = str(getattr(cognition, "current_objective", "") or "")
            if person_turn and message and cognition.last_response:
                carried, added = get_tangled_ledger().taken_up(str(cognition.last_response), message)
                if carried > 0.0 and "joy" in affect.emotions:
                    current = float(affect.emotions.get("joy", 0.0) or 0.0)
                    _set_emotion(affect, "joy", current + carried * added * (1.0 - current))
                ledger.observe(
                    Event(what="expression", actor=SELF, verified=carried > 0.0, detail={"to": partner, "added": added})
                )
        except AFFECT_UPDATE_ERRORS as exc:
            self._record(
                state,
                exc,
                stage="tangled",
                action="kept affect state without the ties between her feelings",
                severity="warning",
            )

    def habits(self, state: AuraState, affect: AffectVector) -> None:
        """What followed her last acts, for her, and whether a habit has cost her.

        Closes the acts waiting on this reading, hands their outcomes to the
        receipts they opened, and writes what her habits and reflexes have been
        worth onto her state. See core/agency/habits_are_hers.py.
        """
        try:
            from core.agency.habits_are_hers import appraise, get_habit_ledger, situation_of
            from core.kernel.turn_door import USER_ORIGINS

            ledger = get_habit_ledger()
            origin = str(getattr(state.cognition, "current_origin", "") or "")
            partner = str(getattr(state.cognition, "current_partner", "") or "")
            closed = ledger.felt(
                float(getattr(affect, "valence", 0.0) or 0.0),
                situation=situation_of(partner, person_turn=origin in USER_ORIGINS),
            )
            appraise(closed)
            state.cognition.habits = ledger.reading()
        except AFFECT_UPDATE_ERRORS as exc:
            self._record(
                state,
                exc,
                stage="habits",
                action="kept affect state without what followed her last acts",
                severity="warning",
            )

    def made_minor(self, state: AuraState, affect: AffectVector) -> None:
        """Being made minor in somebody's account of a shared past, felt as sadness.

        The reading is taken by the conversational dynamics phase; this floors
        sadness at it, so it cannot compound. See core/social/made_minor.py.
        """
        try:
            reading = getattr(state.cognition, "made_minor", None) or {}
            minor = float(reading.get("minor", 0.0) or 0.0) if isinstance(reading, dict) else 0.0
            if minor > 0.0:
                current = float(affect.emotions.get("sadness", 0.0) or 0.0)
                _set_emotion(affect, "sadness", max(current, min(1.0, minor)))
        except AFFECT_UPDATE_ERRORS as exc:
            self._record(
                state,
                exc,
                stage="made_minor",
                action="kept affect state without the hurt of being made minor",
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
        except (ImportError, AttributeError, TypeError, ValueError) as exc:
            logger.debug(
                "%s unavailable (%s: %s); constancy did not record that she came round",
                "time",
                type(exc).__name__,
                exc,
            )
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
            from core.self.standing import get_standing_ledger

            ledger = get_standing_ledger()
            ledger.note_own()
            if bool((getattr(state.identity, "read_by_other", {}) or {}).get("borrowed")):
                ledger.note_assigned()
            agency = _her_agency()
            usefulness = 0.0
            if agency is not None and hasattr(agency, "snapshot"):
                usefulness = float((agency.snapshot() or {}).get("efficacy", 0.0) or 0.0)
            # What one person's regard is worth as evidence about her depends
            # on when it started. Somebody whose regard began at a peak and
            # has never been present at a low is not evidence of the same kind
            # as somebody who was there at the low. See
            # core/social/late_regard.py.
            standing_now = float(getattr(state.identity, "stability", 0.0) or 0.0)
            weight = 1.0
            try:
                from core.social.late_regard import get_regard_ledger

                regard_ledger = get_regard_ledger()
                regard_ledger.note_standing(standing_now)
                partner = str(getattr(state.cognition, "current_partner", "") or "")
                if partner:
                    regard_ledger.note_regard(partner, standing_now)
                    weight = regard_ledger.weight(partner)
                state.identity.late_regard = regard_ledger.read().as_dict()
            except AFFECT_UPDATE_ERRORS:
                weight = 1.0
            ledger.note_regard(
                regard=standing_now,
                usefulness=usefulness,
                weight=weight,
            )
            state.identity.standing = ledger.read().as_dict()
            # And what coming through hard things says about what she can do.
            # See core/agency/capacity.py.
            from core.agency.capacity import capacity_of

            state.identity.capacity = capacity_of(getattr(agency, "by_capability", None) or {}).as_dict()
        except AFFECT_UPDATE_ERRORS as exc:
            self._record(
                state, exc,
                stage="standing",
                action="kept affect state without the reading of what is hers",
                severity="warning",
            )

    def owning_first(self, state: AuraState) -> None:
        """What her history says about owning a lapse before it is raised.

        The events are noted where they happen, when a reply goes out and when
        the person answers. This only carries the reading onto her identity,
        where the self domain reads it. See core/social/owning_it_first.py.
        """
        try:
            from core.social.owning_it_first import get_owning_ledger

            state.identity.owning_first = get_owning_ledger().reading().as_dict()
        except AFFECT_UPDATE_ERRORS as exc:
            self._record(
                state, exc,
                stage="owning_first",
                action="kept affect state without the reading of owning a lapse first",
                severity="warning",
            )
