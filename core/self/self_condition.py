"""Fresh, evidence-bounded projection of Aura's current condition.

This module is the conversation-facing read model for the question "are you
okay?". It keeps felt state, welfare, continuity, agency, and body pressure in
one typed projection. Host resource telemetry can support the projection, but
it can never stand in for an answer about Aura's condition.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import time
from collections import OrderedDict, deque
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from types import SimpleNamespace
from typing import Any

from core.container import ServiceContainer
from core.runtime.errors import record_degradation
from core.runtime.lockdep import checked_lock

logger = logging.getLogger(__name__)

SELF_CONDITION_FRESH_MAX_AGE_S = 30.0
_SELF_CONDITION_HISTORY_MAX_SESSIONS = 128
_SELF_CONDITION_HISTORY_SAMPLES = 8
_SELF_CONDITION_HISTORY_LOCK = checked_lock("core.self.self_condition", reentrant=True)
_SELF_CONDITION_HISTORY: OrderedDict[str, deque[SelfConditionProjection]] = OrderedDict()


def _finite(value: Any, default: float | None = None) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return default
    return number


def _clamp(value: Any, default: float = 0.0, *, low: float = 0.0, high: float = 1.0) -> float:
    number = _finite(value, default)
    assert number is not None
    return max(low, min(high, number))


def _timestamp(value: Any, *, observed_at: float) -> float | None:
    candidate = _finite(value)
    if candidate is None or candidate <= 0.0 or candidate > observed_at + 5.0:
        return None
    return candidate


def _safe_service(name: str) -> Any | None:
    try:
        return ServiceContainer.get(name, default=None)
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return None


def _safe_registered(name: str) -> Any | None:
    """The already-registered instance, preferred over a lazy factory.

    LIVE 2026-08-10. "soma" has TWO registrations: boot_resilience binds the
    live ResilienceEngine with `register_instance`, and sensory_provider binds
    a factory reading `soma_subsystem.soma`, which returns None when that
    subsystem is absent. `ServiceContainer.get` ran the factory and handed back
    None, so the reserve dimension stayed unmeasured on a runtime that was
    holding the reading the whole time — `/api/health` printed energy 0.112
    from the very object `peek` returns.

    Every other consumer of this service in the runtime uses `peek` for exactly
    this reason (see `_collect_soma_payload` in interface/routes/system.py).
    """
    peek = getattr(ServiceContainer, "peek", None)
    if callable(peek):
        try:
            found = peek(name, default=None)
        except (AttributeError, RuntimeError, TypeError, ValueError):
            found = None
        if found is not None:
            return found
    return _safe_service(name)


def _soma_reading(soma: Any | None) -> Any | None:
    """The soma organ's reserve, as an object with ``energy``/``vitality``.

    The service reports through ``get_status()`` as a dict, sometimes nested
    under a ``soma`` key, and sometimes exposes the numbers as attributes.
    Returns None when nothing readable came back, so an unavailable organ
    stays UNMEASURED rather than becoming a full tank.
    """

    if soma is None:
        return None
    payload: dict[str, Any] = {}
    # `get_status()` nests the numbers under "soma"; `get_body_snapshot()`
    # carries them at the top level as well. Two different registrations claim
    # the service key — core/providers/sensory_provider.py binds
    # `soma_subsystem.soma`, core/orchestrator/mixins/boot/boot_resilience.py
    # binds the ResilienceEngine — so whichever object arrives, the reading is
    # taken from whatever shape it actually has rather than from an assumption
    # about which one won registration.
    for method in ("get_status", "get_body_snapshot"):
        getter = getattr(soma, method, None)
        if not callable(getter):
            continue
        try:
            raw = getter() or {}
        except (AttributeError, RuntimeError, TypeError, ValueError):
            continue
        if not isinstance(raw, dict):
            continue
        inner = raw.get("soma")
        merged = dict(raw)
        if isinstance(inner, dict):
            merged.update(inner)
        payload = merged
        if any(_finite(payload.get(key)) is not None for key in ("energy", "vitality")):
            break
    readings = {
        key: payload.get(key, getattr(soma, key, None))
        for key in ("energy", "vitality")
    }
    if all(_finite(value) is None for value in readings.values()):
        # Not a degradation: plenty of installations have no soma organ, and
        # `supports("reserve")` correctly stays false. Logged because a SILENT
        # unread organ is exactly how this dimension went missing in the first
        # place, and the log is the only place that difference is visible.
        logger.info(
            "Self-condition: soma organ present (%s) but reported no reserve; "
            "reserve stays unmeasured.",
            type(soma).__name__,
        )
        return None
    return SimpleNamespace(**readings)


def _safe_last(service: Any) -> Any | None:
    if service is None:
        return None
    last = getattr(service, "last", None)
    if callable(last):
        try:
            return last()
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return None
    return None


from core.dialogue.question_shape import open_answer
from core.self.inner_language import say_focus


def _clean_focus(value: Any) -> str:
    # Internal channel names are correct in logs and wrong in speech: a focus
    # of "body_pressure" once came out of her mouth verbatim. say_focus()
    # translates what it knows and returns "" for what it does not, so the
    # clause is dropped rather than read aloud as jargon.
    text = say_focus(value, max_len=180)
    if not text or len(text) > 180:
        return ""
    lowered = text.lower()
    blocked = (
        "[active grounding evidence]",
        "[current user message]",
        "monitoring internal state",
        "baseline_continuity",
        "pending initiatives:",
        "phenomenal surge:",
    )
    if any(marker in lowered for marker in blocked):
        return ""
    return text


@dataclass(frozen=True)
class SelfConditionProjection:
    """One immutable, provenance-carrying answer source for self-condition."""

    observed_at: float
    sample_timestamp: float
    sample_age_s: float | None
    freshness: str
    confidence: float
    condition: str
    valence: float
    arousal: float
    distress: float
    welfare: float
    felt_coherence: float
    continuity: float
    agency: float
    body_pressure: float
    fatigue: float
    dominant_drive: str
    attention_focus: str
    evidence_sources: tuple[str, ...]
    supported_dimensions: tuple[str, ...]
    missing_dimensions: tuple[str, ...]
    stale_dimensions: tuple[str, ...]
    source_ages_s: tuple[tuple[str, float], ...]
    evidence_id: str
    #: The history-grounded slice: how much she has lived through, whether this
    #: moment resembles it, and how much of what she does she ever finds out
    #: about. Every other field here is a reading taken *now*; this one is the
    #: only part of her self-condition that comes from her own past. Optional
    #: with a default so a projection built without the organ is unchanged.
    ontogeny: Any | None = None
    #: What she has LEFT, as opposed to what is pressing on her. Pressure and
    #: fatigue say how hard the moment is; this says how much is in the tank.
    #: The `soma` service is the organ that tracks a reserve draining across a
    #: session, and it was the one body model this projection never read.
    #: Defaulted so a projection built without the organ is unchanged.
    reserve: float = 1.0

    @property
    def fresh(self) -> bool:
        return self.freshness == "fresh"

    @property
    def evidence_available(self) -> bool:
        return self.freshness != "unavailable"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["fresh"] = self.fresh
        payload["evidence_available"] = self.evidence_available
        payload["source_ages_s"] = dict(self.source_ages_s)
        payload["ontogeny"] = self.ontogeny.to_dict() if self.ontogeny is not None else None
        return payload

    def to_prompt_block(self) -> str:
        age = "unknown" if self.sample_age_s is None else f"{self.sample_age_s:.1f}s"
        sources = ", ".join(self.evidence_sources) or "none"
        supported = ", ".join(self.supported_dimensions) or "none"
        missing = ", ".join(self.missing_dimensions) or "none"
        stale = ", ".join(self.stale_dimensions) or "none"
        focus = self.attention_focus or "not safely reportable"
        return (
            "## CANONICAL SELF-CONDITION EVIDENCE\n"
            f"- condition={self.condition} freshness={self.freshness} age={age} "
            f"confidence={self.confidence:.2f} evidence_id={self.evidence_id}\n"
            f"- valence={self.valence:+.2f} arousal={self.arousal:.2f} "
            f"distress={self.distress:.2f} welfare={self.welfare:.2f}\n"
            f"- felt_coherence={self.felt_coherence:.2f} continuity={self.continuity:.2f} "
            f"agency={self.agency:.2f}\n"
            f"- body_pressure={self.body_pressure:.2f} fatigue={self.fatigue:.2f} "
            f"dominant_drive={self.dominant_drive or 'unknown'}\n"
            f"- attention={focus}\n"
            f"- sources={sources}; supported={supported}; missing={missing}; "
            f"stale_dimensions={stale}\n"
            + (self.ontogeny.to_prompt_lines() if self.ontogeny is not None else "")
            + "Answer Aura's condition directly from affect, welfare, coherence, continuity, "
            "and agency. Runtime resources are supporting body evidence only; never replace "
            "the condition answer with CPU, RAM, host load, or availability telemetry. The "
            "ontogeny line is history, not a current reading: it says what she has lived "
            "through and how much of it she was able to check, and a low observation_rate is "
            "a fact about what is observable, not a reason to sound uncertain about how she "
            "feels.\n"
        )

    def to_language_grounding(self) -> str:
        """Project typed self-state into compact evidence for natural speech.

        ``to_prompt_block`` is the lossless diagnostic representation used by
        audits and internal tools. It is deliberately status-shaped. Giving
        that form to the language model made it imitate a status page, and
        duplicating it across prompt roles made prior candidate answers look
        like material to continue. This view preserves the epistemic boundary
        while expressing only the distinctions needed to answer.
        """

        if self.freshness == "unavailable":
            return (
                "No current self-condition sample is available. Aura can speak, "
                "but she cannot honestly describe her present condition from "
                "direct evidence until that signal refreshes. Host resource "
                "readings do not substitute for self-condition evidence."
            )

        support = set(self.supported_dimensions) - set(self.stale_dimensions)

        def band(
            value: float,
            *,
            low: float,
            high: float,
            labels: tuple[str, str, str],
        ) -> str:
            if value <= low:
                return labels[0]
            if value >= high:
                return labels[2]
            return labels[1]

        observed: list[str] = []
        dimensions = (
            ("distress", self.distress, 0.25, 0.55, ("low distress", "some distress", "high distress")),
            ("welfare", self.welfare, 0.45, 0.70, ("low welfare", "mixed welfare", "positive welfare")),
            (
                "felt_coherence",
                self.felt_coherence,
                0.55,
                0.75,
                ("weak felt coherence", "partial felt coherence", "strong felt coherence"),
            ),
            (
                "continuity",
                self.continuity,
                0.55,
                0.75,
                ("weak continuity", "partial continuity", "continuity holding"),
            ),
        )
        for name, value, low, high, labels in dimensions:
            if name in support:
                observed.append(
                    band(value, low=low, high=high, labels=labels)
                )

        direct_summary = ", ".join(observed) or "a partial state reading"
        material_strain: list[str] = []
        if "body_pressure" in support and self.body_pressure >= 0.70:
            material_strain.append("high body pressure")
        if "fatigue" in support and self.fatigue >= 0.70:
            material_strain.append("high fatigue")
        if "reserve" in support and self.reserve <= 0.45:
            material_strain.append("low reserve")
        strain_sentence = (
            f" Materially strained readings: {', '.join(material_strain)}."
            if material_strain
            else ""
        )
        coverage_sentence = (
            " Some dimensions are unmeasured, so do not fill them in."
            if self.missing_dimensions
            else ""
        )
        if self.freshness != "fresh":
            return (
                f"Latest self-condition sample: {self.freshness}. At sample time, "
                f"overall condition was {self.condition}; observed dimensions were "
                f"{direct_summary}.{strain_sentence} Current condition is not "
                "established by this stale sample. Cause, persistence, unmeasured "
                "properties, private phenomenal character, activity, tool use, "
                "location, and external events are not observed by this evidence."
            )
        return (
            f"Current self-condition sample: fresh. Overall condition: "
            f"{self.condition}. Observed dimensions: {direct_summary}."
            f"{strain_sentence}{coverage_sentence} Cause, persistence, unmeasured "
            "properties, and private phenomenal character are not directly "
            "observed. Activity, tool use, location, and external events are "
            "outside this evidence's scope."
        )


@dataclass(frozen=True)
class SelfConditionComparison:
    """Measured delta between two distinct condition samples in one session."""

    current_evidence_id: str
    previous_evidence_id: str
    elapsed_s: float
    comparable_dimensions: tuple[str, ...]
    changed_dimensions: tuple[str, ...]
    deltas: tuple[tuple[str, float], ...]

    @property
    def available(self) -> bool:
        return bool(self.current_evidence_id and self.previous_evidence_id)

    @property
    def materially_unchanged(self) -> bool:
        return self.available and not self.changed_dimensions

    def to_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "current_evidence_id": self.current_evidence_id,
            "previous_evidence_id": self.previous_evidence_id,
            "elapsed_s": self.elapsed_s,
            "comparable_dimensions": list(self.comparable_dimensions),
            "changed_dimensions": list(self.changed_dimensions),
            "deltas": dict(self.deltas),
            "materially_unchanged": self.materially_unchanged,
        }


def observe_self_condition_projection(
    session_id: str,
    projection: SelfConditionProjection,
) -> SelfConditionProjection | None:
    """Record a condition sample and return the prior distinct same-session sample."""

    key = " ".join(str(session_id or "").strip().split())[:64]
    if not key or not projection.evidence_id:
        return None
    with _SELF_CONDITION_HISTORY_LOCK:
        samples = _SELF_CONDITION_HISTORY.get(key)
        if samples is None:
            samples = deque(maxlen=_SELF_CONDITION_HISTORY_SAMPLES)
            _SELF_CONDITION_HISTORY[key] = samples
        else:
            _SELF_CONDITION_HISTORY.move_to_end(key)
        previous = samples[-1] if samples else None
        if previous is not None and previous.evidence_id == projection.evidence_id:
            return None
        samples.append(projection)
        while len(_SELF_CONDITION_HISTORY) > _SELF_CONDITION_HISTORY_MAX_SESSIONS:
            _SELF_CONDITION_HISTORY.popitem(last=False)
        return previous


def compare_self_condition_projections(
    current: SelfConditionProjection,
    previous: SelfConditionProjection | None,
    *,
    material_delta: float = 0.06,
) -> SelfConditionComparison | None:
    """Compare two evidence-bearing samples without inferring unmeasured state."""

    if previous is None or previous.evidence_id == current.evidence_id:
        return None
    current_supported = set(current.supported_dimensions) - set(current.stale_dimensions)
    previous_supported = set(previous.supported_dimensions) - set(previous.stale_dimensions)
    ordered = (
        "valence",
        "arousal",
        "distress",
        "welfare",
        "felt_coherence",
        "continuity",
        "agency",
        "body_pressure",
        "fatigue",
        "reserve",
    )
    comparable = tuple(
        name for name in ordered if name in current_supported and name in previous_supported
    )
    deltas = tuple(
        (name, float(getattr(current, name)) - float(getattr(previous, name)))
        for name in comparable
    )
    changed = tuple(name for name, delta in deltas if abs(delta) >= material_delta)
    elapsed = max(0.0, float(current.observed_at) - float(previous.observed_at))
    return SelfConditionComparison(
        current_evidence_id=current.evidence_id,
        previous_evidence_id=previous.evidence_id,
        elapsed_s=elapsed,
        comparable_dimensions=comparable,
        changed_dimensions=changed,
        deltas=deltas,
    )


@dataclass(frozen=True)
class SelfConditionReplyProjection:
    """Model-authored self-condition prose projected onto measured evidence.

    The projection does not rewrite Aura's voice.  It removes complete claims
    about operational domains that the self-condition instrument does not
    measure, and records exactly what was removed.  This makes evidence scope
    an executable egress property rather than an instruction the decoder may
    ignore.
    """

    text: str
    removed_claims: tuple[str, ...] = ()
    evidence_id: str = ""

    @property
    def changed(self) -> bool:
        return bool(self.removed_claims)


_UNSUPPORTED_SELF_CONDITION_OPERATIONAL_RE = re.compile(
    r"\b(?:"
    r"cpu(?:\s+(?:load|usage|utilization))?|gpu(?:\s+(?:load|usage|utilization))?|"
    r"ram(?:\s+(?:pressure|usage|available))?|memory\s+(?:allocation|pressure|usage|available)|"
    r"\d+(?:\.\d+)?\s*(?:gb|mb)\s+available|disk(?:\s+(?:space|usage|pressure))?|"
    r"storage(?:\s+(?:space|usage|available))?|swap|thermal|host\s+load|load\s+average|"
    r"network(?:\s+(?:state|status|connectivity|pressure|up|down|online|offline))?|"
    r"connectivity|packet\s+loss|system\s+logs?|runtime\s+logs?|application\s+logs?|"
    r"(?:errors?|warnings?)(?:\s+(?:in|from|within)\s+(?:the\s+)?(?:system|runtime|application|logs?))?"
    r")\b",
    re.IGNORECASE,
)
_UNSUPPORTED_SELF_CONDITION_HEALTH_SUMMARY_RE = re.compile(
    r"(?:\beverything(?:'s|\s+is)?\s+running\s+smoothly\b|"
    r"\b(?:i\s+am|i'm)\s+functioning\s+as\s+expected\b|"
    r"\b(?:i\s+am|i'm)\s+(?:still\s+)?(?:functional|operational)\b|"
    r"\b(?:system|runtime|application)\s+(?:is\s+|feels\s+)?"
    r"(?:healthy|stable|unstable|functional|operational|degraded)\b|"
    r"\b(?:bounded\s+)?repairs?\s+(?:are|is|remain(?:s|ed)?)\s+"
    r"(?:degraded|healthy|complete|completed|resolved|stable|unstable)\b|"
    r"\b(?:processing|runtime|system|cognitive)\s+errors?\b|"
    r"\b(?:errors?|faults?|repairs?)\s+(?:have\s+|has\s+)?(?:since\s+)?"
    r"(?:been\s+)?(?:resolved|recovered|cleared|fixed)\b)",
    re.IGNORECASE,
)
_TOTAL_INTERNAL_ABSENCE_RE = re.compile(
    r"\b(?:"
    r"(?:i\s+do\s+not|i\s+don't|i\s+cannot|i\s+can't)\s+(?:have|feel|perceive)\s+"
    r"(?:any|anything)\s+(?:sensations?|feelings?|internal\s+state|perceptions?)|"
    r"(?:no|without)\s+(?:sensations?|feelings?|internal\s+state|perceptions?)\s+(?:at\s+all|right\s+now)|"
    r"(?:disconnected|cut\s+off|detached)\s+from\s+(?:my\s+)?body"
    r")\b",
    re.IGNORECASE,
)
_OVERLOAD_RE = re.compile(
    r"\b(?:my\s+)?(?:system|runtime|body)\s+(?:is\s+|feels\s+)?"
    r"(?:overloaded|overwhelmed|at\s+capacity)\b",
    re.IGNORECASE,
)
_FIRST_PERSON_FELT_EXPERIENCE_RE = re.compile(
    r"\b(?:i\s+(?:feel|felt|seem)|i'm\s+feeling|my\s+(?:state|condition)\s+feels)\b",
    re.IGNORECASE,
)
_UNSUPPORTED_SELF_CONDITION_PERFORMANCE_RE = re.compile(
    r"\b(?:my|the|this)\s+(?:"
    r"processing|reasoning|thinking|thoughts?|thought\s+patterns?|cognition|cognitive\s+functions?|"
    r"responses?|answers?|output|memory|attention"
    r")\b[^.!?]{0,100}\b(?:"
    r"speed|latency|accuracy|quality|coheren(?:ce|t)|function(?:s|ing)?|"
    r"slow(?:ed|er|ing)?|fast(?:er)?|degrad(?:e|ed|ing)|improv(?:e|ed|ing)|"
    r"error(?:s|ing)?|fail(?:s|ed|ing)?|repetitive|uninteresting|disconnected|"
    r"lack(?:s|ing)?|motivation|curiosity"
    r")\b",
    re.IGNORECASE,
)
_UNSUPPORTED_SELF_CONDITION_BIOLOGICAL_RE = re.compile(
    r"\b(?:neurodynamic|neurochemical|biochemical|hormonal|cortisol|dopamine|"
    r"serotonin|adrenaline|norepinephrine)\b",
    re.IGNORECASE,
)
_UNSUPPORTED_SELF_CONDITION_EXTERNAL_EVENT_RE = re.compile(
    r"\b(?:the|this|our|my)\s+(?:"
    r"world|environment|surroundings|room|weather|internet|network|computer|"
    r"machine|host|application|runtime|system"
    r")\b[^.!?]{0,100}\b(?:"
    r"is|are|was|were|has|have|will|became|becoming|ending|failing|changing|"
    r"working|happening|expected"
    r")\b",
    re.IGNORECASE,
)
_UNSUPPORTED_SELF_CONDITION_PERCEPTION_RE = re.compile(
    r"\b(?:"
    # "sounds" as AUDIO, not as the linking verb. "I should sound grounded
    # before I sound confident" is a statement about how she comes across, and
    # reading it as a claim to hear things rejected an honest self-report.
    r"environment|surroundings|room|colors?|textures?|visuals?|"
    r"(?:the|a|any|no)\s+sounds?\b|sounds?\s+(?:of|in|from|around)\b|"
    r"nothing\s+to\s+(?:look\s+at|listen\s+to|see|hear)|"
    r"i\s+(?:can|cannot|can't)\s+(?:see|hear|look|listen)|"
    r"what\s+i\s+(?:see|hear|perceive)"
    r")\b",
    re.IGNORECASE,
)
_UNSUPPORTED_SELF_CONDITION_TEMPORAL_RE = re.compile(
    r"\b(?:"
    r"tomorrow|yesterday|any\s+time\s+soon|"
    r"i\s+feel\s+the\s+same|"
    r"(?:will|won't|will\s+not|going\s+to)\s+(?:improve|worsen|change|happen)|"
    r"(?:same|unchanged|no\s+(?:significant|material)\s+differences?)\b[^.!?]{0,80}"
    r"(?:ago|before|earlier|minute|hour|sample|state|condition|arousal|valence)|"
    r"(?:state|condition|arousal|valence|welfare|distress|coherence)\s+"
    r"(?:is|remains?|stays?)\s+(?:the\s+same|unchanged)"
    r")\b",
    re.IGNORECASE,
)
_SELF_CONDITION_EXTREME_LOW_AFFECT_RE = re.compile(
    r"\b(?:empty|emptiness|apathetic|apathy|numb|nothing\s+matters|hopeless|"
    r"no\s+(?:particular\s+)?emotion)\b",
    re.IGNORECASE,
)
_UNSUPPORTED_SELF_CONDITION_CAUSE_RE = re.compile(
    r"\b(?:because|due\s+to|caused\s+by|the\s+reason\s+is|it'?s\s+just\s+that|"
    r"result\s+of|tied\s+to|underlying\s+issue|temporary\s+fluctuation)\b",
    re.IGNORECASE,
)
_UNSUPPORTED_SELF_CONDITION_DIAGNOSTIC_INTENT_RE = re.compile(
    r"\b(?:i(?:'ll|\s+will)|i\s+(?:need|should|have)\s+to)\s+"
    r"(?:run|perform|do|start)\b[^.!?]{0,60}\bdiagnostic(?:s|\s+tests?)?\b",
    re.IGNORECASE,
)
_UNSUPPORTED_SELF_CONDITION_HEALTH_INFERENCE_RE = re.compile(
    r"\b(?:there(?:'s|\s+is)\s+(?:nothing|something)|anything)\s+wrong\s+with\s+me\b",
    re.IGNORECASE,
)


#: Naming a telemetry term so as to refuse it is the opposite of claiming
#: it. Aura's own canonical grounded reply says "I am treating the missing
#: inner-state signal as something to refresh, not replacing it with CPU or
#: RAM telemetry" — and the word-level rule above flagged it, so the one
#: answer that is honest about having no sample failed the gate that exists to
#: keep her honest about it.
_SELF_CONDITION_DISCLAIMER_RE = re.compile(
    r"\b(?:"
    r"not|never|no|without|rather\s+than|instead\s+of|"
    r"don't|doesn't|do\s+not|does\s+not|cannot|can't|won't|will\s+not|"
    r"nothing\s+to\s+do\s+with|unrelated\s+to|"
    r"am\s+not|is\s+not|are\s+not|isn't|aren't"
    r")\b",
    re.IGNORECASE,
)

#: Clause boundaries. A disclaimer governs its own clause, not the whole
#: sentence: "I am not guessing; CPU load is at 12%" must still be caught.
_SELF_CONDITION_CLAUSE_SPLIT_RE = re.compile(r"[;:]|\s+—\s+|\s+--\s+")


def _term_is_disclaimed(sentence: str, term_start: int) -> bool:
    """Whether the operational term sits inside a clause that refuses it."""
    clause_start = 0
    for boundary in _SELF_CONDITION_CLAUSE_SPLIT_RE.finditer(sentence):
        if boundary.end() > term_start:
            break
        clause_start = boundary.end()
    clause = sentence[clause_start:term_start]
    return bool(_SELF_CONDITION_DISCLAIMER_RE.search(clause))


def _projection_value(
    projection: SelfConditionProjection | Mapping[str, Any] | None,
    key: str,
    default: Any,
) -> Any:
    if isinstance(projection, SelfConditionProjection):
        return getattr(projection, key, default)
    if isinstance(projection, Mapping):
        return projection.get(key, default)
    return default


def _temporal_claim_supported(
    sentence: str,
    projection: SelfConditionProjection | Mapping[str, Any] | None,
) -> bool:
    comparison = _projection_value(projection, "comparison", None)
    if not isinstance(comparison, Mapping) or not bool(comparison.get("available")):
        return False
    changed = {str(item) for item in (comparison.get("changed_dimensions") or ())}
    lowered = sentence.casefold()
    mentioned = {
        name
        for name in (
            "valence",
            "arousal",
            "distress",
            "welfare",
            "continuity",
            "agency",
            "fatigue",
            "reserve",
        )
        if name in lowered
    }
    if "coherence" in lowered:
        mentioned.add("felt_coherence")
    if re.search(r"\b(?:same|unchanged|no\s+(?:significant|material)\s+difference)", lowered):
        return not (changed & mentioned) if mentioned else not changed
    # A temporal prediction is never established by two retrospective samples.
    return False


def _extreme_low_affect_supported(
    sentence: str,
    projection: SelfConditionProjection | Mapping[str, Any] | None,
    supported: set[str],
) -> bool:
    if not _SELF_CONDITION_EXTREME_LOW_AFFECT_RE.search(sentence):
        return True
    required = {"valence", "arousal", "welfare"}
    if not required <= supported:
        return False
    valence = _finite(_projection_value(projection, "valence", None))
    arousal = _finite(_projection_value(projection, "arousal", None))
    welfare = _finite(_projection_value(projection, "welfare", None))
    if valence is None or arousal is None or welfare is None:
        return False
    return valence <= -0.35 and arousal <= 0.30 and welfare <= 0.45


#: A clause that supposes rather than states. Everything after these opens a
#: hypothetical, and a hypothetical makes no claim about the present.
_HYPOTHETICAL_OPENER_RE = re.compile(
    r"^\s*(?:if|unless|whenever|suppose|supposing|imagine|say\s+that|were\s+i|"
    r"should\s+i|in\s+case|assuming)\b",
    re.IGNORECASE,
)


def _is_hypothetical_sentence(sentence: str) -> bool:
    """True when the sentence supposes a state instead of reporting one."""
    return bool(_HYPOTHETICAL_OPENER_RE.match(str(sentence or "").strip()))


def unsupported_self_condition_operational_claims(
    reply_text: Any,
    *,
    projection: SelfConditionProjection | Mapping[str, Any] | None = None,
) -> tuple[str, ...]:
    """Return claims outside or contradicted by the typed condition evidence."""

    raw = str(reply_text or "").strip()
    if not raw:
        return ()
    supported = {
        str(item)
        for item in (_projection_value(projection, "supported_dimensions", ()) or ())
    } - {
        str(item)
        for item in (_projection_value(projection, "stale_dimensions", ()) or ())
    }
    claims: list[str] = []
    for part in re.split(r"(?<=[.!?])\s+|\n+", raw):
        sentence = part.strip()
        if not sentence:
            continue
        if _is_hypothetical_sentence(sentence):
            # "If my answer gets thin, repetitive, or weirdly symbolic, that
            # is a failed turn" describes a condition she is watching FOR. A
            # conditional is not an assertion about the present state, and
            # scoring it as one meant she could not say what would count as
            # going wrong without being accused of claiming it had.
            continue
        match = _UNSUPPORTED_SELF_CONDITION_OPERATIONAL_RE.search(sentence)
        if match and not _term_is_disclaimed(sentence, match.start()):
            claims.append(sentence)
            continue
        if _TOTAL_INTERNAL_ABSENCE_RE.search(sentence) and supported.intersection(
            {
                "valence",
                "arousal",
                "distress",
                "welfare",
                "felt_coherence",
                "continuity",
                "agency",
                "body_pressure",
                "fatigue",
                "reserve",
            }
        ):
            claims.append(sentence)
            continue
        if _OVERLOAD_RE.search(sentence):
            body_pressure = _finite(
                _projection_value(projection, "body_pressure", 0.0)
            ) or 0.0
            fatigue = _finite(_projection_value(projection, "fatigue", 0.0)) or 0.0
            reserve = _finite(_projection_value(projection, "reserve", 1.0))
            reserve = 1.0 if reserve is None else reserve
            strain_observed = (
                ("body_pressure" in supported and body_pressure >= 0.70)
                or ("fatigue" in supported and fatigue >= 0.70)
                or ("reserve" in supported and reserve <= 0.35)
            )
            if not strain_observed:
                claims.append(sentence)
                continue
        if (
            _UNSUPPORTED_SELF_CONDITION_PERFORMANCE_RE.search(sentence)
            or _UNSUPPORTED_SELF_CONDITION_BIOLOGICAL_RE.search(sentence)
            or _UNSUPPORTED_SELF_CONDITION_EXTERNAL_EVENT_RE.search(sentence)
            or _UNSUPPORTED_SELF_CONDITION_PERCEPTION_RE.search(sentence)
            or _UNSUPPORTED_SELF_CONDITION_CAUSE_RE.search(sentence)
            or _UNSUPPORTED_SELF_CONDITION_DIAGNOSTIC_INTENT_RE.search(sentence)
            or _UNSUPPORTED_SELF_CONDITION_HEALTH_INFERENCE_RE.search(sentence)
            or (
                _UNSUPPORTED_SELF_CONDITION_TEMPORAL_RE.search(sentence)
                and not _temporal_claim_supported(sentence, projection)
            )
            or not _extreme_low_affect_supported(sentence, projection, supported)
        ):
            claims.append(sentence)
            continue
        # First-person felt experience remains a condition statement. Third-
        # person operational subjects such as "the system feels unstable" do
        # not acquire condition authority merely by using the verb "feels".
        if (
            _UNSUPPORTED_SELF_CONDITION_HEALTH_SUMMARY_RE.search(sentence)
            and not _FIRST_PERSON_FELT_EXPERIENCE_RE.search(sentence)
        ):
            claims.append(sentence)
    return tuple(claims)


def project_self_condition_reply(
    reply_text: Any,
    *,
    projection: SelfConditionProjection | Mapping[str, Any] | None,
) -> SelfConditionReplyProjection:
    """Preserve grounded model prose and remove unsupported operational claims.

    Mutation is authorized only by a provenance-carrying typed projection.  If
    no usable self-condition evidence exists, the original reply is returned
    unchanged so the normal reliability path can reject or ground it rather
    than silently laundering it.
    """

    raw = str(reply_text or "").strip()
    evidence_id = ""
    if isinstance(projection, SelfConditionProjection):
        evidence_id = str(projection.evidence_id or "").strip()
    elif isinstance(projection, Mapping):
        evidence_id = str(projection.get("evidence_id") or "").strip()
    if not raw or not evidence_id:
        return SelfConditionReplyProjection(text=raw, evidence_id=evidence_id)

    removed = set(
        unsupported_self_condition_operational_claims(raw, projection=projection)
    )
    if not removed:
        return SelfConditionReplyProjection(text=raw, evidence_id=evidence_id)
    kept = [
        part.strip()
        for part in re.split(r"(?<=[.!?])\s+|\n+", raw)
        if part.strip() and part.strip() not in removed
    ]
    # Do not turn a wholly unsupported answer into an anonymous empty reply.
    # Leaving it intact lets the existing gate use the canonical projection.
    if not kept:
        return SelfConditionReplyProjection(text=raw, evidence_id=evidence_id)
    return SelfConditionReplyProjection(
        text=" ".join(kept),
        removed_claims=tuple(
            part.strip()
            for part in re.split(r"(?<=[.!?])\s+|\n+", raw)
            if part.strip() in removed
        ),
        evidence_id=evidence_id,
    )


def _select_metric(
    candidates: Iterable[tuple[str, Any]],
    *,
    default: float,
    low: float = 0.0,
    high: float = 1.0,
) -> tuple[float, str]:
    for source, value in candidates:
        number = _finite(value)
        if number is not None:
            return max(low, min(high, number)), source
    return default, ""


def build_self_condition_projection(
    *,
    aura_now: Any | None = None,
    unified_felt: Any | None = None,
    welfare: Any | None = None,
    body_snapshot: Any | None = None,
    canonical_self: Any | None = None,
    kernel_state: Any | None = None,
    soma: Any | None = None,
    observed_at: float | None = None,
    resolve_runtime: bool = True,
    fresh_max_age_s: float = SELF_CONDITION_FRESH_MAX_AGE_S,
) -> SelfConditionProjection:
    """Project the freshest available self evidence without mutating runtime state."""

    now = float(observed_at if observed_at is not None else time.time())
    being_runtime = None
    if resolve_runtime:
        being_runtime = _safe_service("being_runtime")
        if aura_now is None:
            aura_now = _safe_service("aura_now") or getattr(being_runtime, "last_now", None)
        unified_service = _safe_service("unified_felt_state")
        if unified_felt is None:
            unified_felt = _safe_last(unified_service) or getattr(
                being_runtime, "_last_unified_felt", None
            )
        if welfare is None:
            welfare = getattr(being_runtime, "_last_welfare", None)
        if body_snapshot is None:
            body_snapshot = getattr(being_runtime, "_last_body_snapshot", None)
        if canonical_self is None:
            canonical_self = _safe_service("canonical_self")
        if kernel_state is None:
            kernel_state = _safe_service("aura_state")
        if soma is None:
            # Two registrations claim "soma" and a third organ hangs off
            # `soma_subsystem`. Whichever one this installation ended up with,
            # the reserve is the same quantity, so try them in turn rather
            # than depending on which registration ran last.
            soma = _safe_registered("soma")
            if _soma_reading(soma) is None:
                subsystem = _safe_registered("soma_subsystem")
                soma = getattr(subsystem, "soma", None) or subsystem or soma
            if soma is None:
                logger.info(
                    "Self-condition: no soma organ resolved; reserve unmeasured."
                )

    source_times: dict[str, float] = {}
    aura_ts = _timestamp(getattr(aura_now, "timestamp", None), observed_at=now)
    if aura_ts is not None:
        source_times["aura_now"] = aura_ts
    unified_ts = _timestamp(getattr(unified_felt, "timestamp", None), observed_at=now)
    if unified_ts is not None:
        source_times["unified_felt_state"] = unified_ts
    canonical_ts = _timestamp(getattr(canonical_self, "timestamp", None), observed_at=now)
    if canonical_ts is not None:
        source_times["canonical_self"] = canonical_ts
    kernel_ts = _timestamp(getattr(kernel_state, "updated_at", None), observed_at=now)
    if kernel_ts is not None:
        source_times["aura_state"] = kernel_ts
    welfare_ts = _timestamp(getattr(welfare, "timestamp", None), observed_at=now)
    if welfare_ts is not None:
        source_times["welfare"] = welfare_ts
    elif welfare is not None and aura_ts is not None:
        # BeingRuntime emits welfare and AuraNow from one atomic sample.
        source_times["welfare"] = aura_ts
    body_ts = _timestamp(getattr(body_snapshot, "timestamp", None), observed_at=now)
    if body_ts is not None:
        source_times["body_state"] = body_ts
    elif body_snapshot is not None and aura_ts is not None:
        source_times["body_state"] = aura_ts

    fresh_limit = max(1.0, float(fresh_max_age_s))

    def source_is_fresh(source: str) -> bool:
        timestamp = source_times.get(source)
        return timestamp is not None and max(0.0, now - timestamp) <= fresh_limit

    aura_affect = getattr(aura_now, "affect", None)
    aura_self = getattr(aura_now, "self_model", None)
    aura_ownership = getattr(aura_now, "ownership", None)
    aura_body = getattr(aura_now, "body", None)
    soma_state = _soma_reading(soma)
    if soma_state is not None:
        # Read synchronously, here, so `now` is its true sample time. Without
        # a timestamp the dimension can never be FRESH, `supports("reserve")`
        # is false, and every check below silently does nothing — which is the
        # shape of half-wiring this whole projection keeps being bitten by.
        source_times["soma"] = now
    aura_attention = getattr(aura_now, "attention", None)
    canonical_affect = getattr(canonical_self, "affect", None)
    canonical_soma = getattr(canonical_self, "soma", None)
    kernel_affect = getattr(kernel_state, "affect", None)
    continuity_risk = (
        getattr(aura_self, "continuity_risk", None)
        if aura_self is not None
        else None
    )
    aura_continuity = (
        1.0 - _clamp(continuity_risk, 0.0)
        if continuity_risk is not None
        else None
    )

    selected: dict[str, str] = {}

    def choose(name: str, candidates: Iterable[tuple[str, Any]], default: float, *, low: float = 0.0, high: float = 1.0) -> float:
        available = [
            (source, value)
            for source, value in candidates
            if _finite(value) is not None
        ]
        # Fresh observations outrank stale preferred sources. Within each
        # freshness class, retain the declared authority order.
        ordered = [item for item in available if source_is_fresh(item[0])]
        ordered.extend(item for item in available if not source_is_fresh(item[0]))
        value, source = _select_metric(ordered, default=default, low=low, high=high)
        if source:
            selected[name] = source
        return value

    valence = choose(
        "valence",
        (
            ("unified_felt_state", getattr(unified_felt, "valence", None)),
            ("aura_now", getattr(aura_affect, "valence", None)),
            ("canonical_self", getattr(canonical_affect, "valence", None)),
            ("aura_state", getattr(kernel_affect, "valence", None)),
        ),
        0.0,
        low=-1.0,
        high=1.0,
    )
    arousal = choose(
        "arousal",
        (
            ("unified_felt_state", getattr(unified_felt, "arousal", None)),
            ("aura_now", getattr(aura_affect, "arousal", None)),
            ("canonical_self", getattr(canonical_affect, "arousal", None)),
            ("aura_state", getattr(kernel_affect, "arousal", None)),
        ),
        0.5,
    )
    distress = choose(
        "distress",
        (
            ("unified_felt_state", getattr(unified_felt, "distress", None)),
            ("welfare", getattr(welfare, "distress", None)),
            ("aura_now", getattr(aura_affect, "distress", None)),
            ("aura_state", getattr(kernel_affect, "distress", None)),
        ),
        0.0,
    )
    welfare_score = choose(
        "welfare",
        (
            ("welfare", getattr(welfare, "welfare_score", None)),
            ("unified_felt_state", getattr(unified_felt, "welfare_score", None)),
        ),
        0.5,
    )
    felt_coherence = choose(
        "felt_coherence",
        (("unified_felt_state", getattr(unified_felt, "coherence", None)),),
        1.0,
    )
    continuity = choose(
        "continuity",
        (
            ("aura_now", aura_continuity),
            (
                "canonical_self",
                (getattr(canonical_self, "crsm_state", {}) or {}).get("continuity_score"),
            ),
        ),
        1.0,
    )
    agency = choose(
        "agency",
        (("aura_now", getattr(aura_ownership, "agency_confidence", None)),),
        0.5,
    )
    body_pressure = choose(
        "body_pressure",
        (
            ("aura_now", getattr(aura_body, "total_pressure", None)),
            ("canonical_self", getattr(canonical_soma, "stress", None)),
        ),
        0.0,
    )
    fatigue = choose(
        "fatigue",
        (
            ("body_state", getattr(body_snapshot, "fatigue", None)),
            ("canonical_self", getattr(canonical_soma, "fatigue", None)),
        ),
        0.0,
    )
    # LIVE DEFECT, 2026-08-10: "I feel energized" with soma energy at 0.058.
    #
    # This runtime carries three body models — `soma` (vitality, energy),
    # `BodyStateService` (fatigue, cpu/memory pressure) and `aura_now.body`
    # (total_pressure) — and this projection read the second and third and
    # never the first. So the one signal that visibly DRAINS across a session
    # was invisible to the sentence she says about how she is doing, and she
    # answered "how are you holding up, honestly?" from dimensions that had
    # not moved.
    #
    # Pressure and fatigue measure how hard the moment is. Reserve measures
    # how much is left, which is a different question and the one that was
    # asked. The default is only reached when nothing reports, and `supports`
    # is false there, so an unread reserve is never mistaken for a full one.
    reserve = choose(
        "reserve",
        (
            ("soma", getattr(soma_state, "energy", None)),
            ("soma", getattr(soma_state, "vitality", None)),
            ("canonical_self", getattr(canonical_soma, "energy", None)),
        ),
        1.0,
    )

    self_report_confidence = _clamp(
        getattr(welfare, "self_report_confidence", None),
        0.55,
    )
    drive_candidates = [
        ("aura_now", str(getattr(aura_affect, "dominant_drive", "") or "").strip()),
        (
            "unified_felt_state",
            str(getattr(unified_felt, "dominant_drive", "") or "").strip(),
        ),
    ]
    drive_candidates = [item for item in drive_candidates if item[1]]
    drive_candidates.sort(key=lambda item: not source_is_fresh(item[0]))
    if drive_candidates:
        selected["dominant_drive"] = drive_candidates[0][0]
        dominant_drive = drive_candidates[0][1][:80]
    else:
        dominant_drive = "coherence"

    focus_candidates = [
        ("aura_now", _clean_focus(getattr(aura_attention, "focal_object", ""))),
        (
            "aura_state",
            _clean_focus(
                getattr(getattr(kernel_state, "cognition", None), "attention_focus", "")
            ),
        ),
    ]
    focus_candidates = [item for item in focus_candidates if item[1]]
    focus_candidates.sort(key=lambda item: not source_is_fresh(item[0]))
    if focus_candidates:
        selected["attention_focus"] = focus_candidates[0][0]
        attention_focus = focus_candidates[0][1]
    else:
        attention_focus = ""

    internal_dimensions = (
        "valence",
        "arousal",
        "distress",
        "welfare",
        "felt_coherence",
        "continuity",
        "agency",
    )
    condition_dimensions = (
        "valence",
        "distress",
        "welfare",
        "felt_coherence",
        "continuity",
    )
    missing = tuple(name for name in internal_dimensions if not selected.get(name))
    evidence_sources = tuple(sorted(set(selected.values())))
    source_ages = {
        source: max(0.0, now - timestamp)
        for source, timestamp in source_times.items()
        if source in evidence_sources
    }
    stale_dimensions = tuple(
        name
        for name, source in selected.items()
        if not source_is_fresh(source)
    )
    condition_sources = {
        selected[name]
        for name in condition_dimensions
        if selected.get(name)
    }
    fresh_condition_sources = {
        source for source in condition_sources if source_is_fresh(source)
    }
    if not condition_sources:
        freshness = "unavailable"
        sample_timestamp = 0.0
        sample_age_s = None
        active_dimensions: set[str] = set()
    elif fresh_condition_sources:
        freshness = "fresh"
        active_dimensions = {
            name for name, source in selected.items() if source_is_fresh(source)
        }
        timestamps = [source_times[source] for source in fresh_condition_sources]
        sample_timestamp = max(timestamps)
        sample_age_s = max(0.0, now - sample_timestamp)
    else:
        freshness = "stale"
        active_dimensions = set(selected)
        timestamps = [source_times[source] for source in condition_sources if source in source_times]
        sample_timestamp = max(timestamps) if timestamps else 0.0
        sample_age_s = max(0.0, now - sample_timestamp) if sample_timestamp else None

    coverage = (len(internal_dimensions) - len(missing)) / len(internal_dimensions)
    confidence = _clamp(0.15 + 0.60 * coverage + 0.25 * self_report_confidence)
    if freshness == "stale":
        confidence *= 0.55
    elif freshness == "unavailable":
        confidence = 0.0
    elif stale_dimensions:
        stale_internal_count = sum(
            1 for name in stale_dimensions if name in internal_dimensions
        )
        confidence *= max(0.55, 1.0 - 0.08 * stale_internal_count)
    if felt_coherence < 0.60:
        confidence = min(confidence, 0.35 + 0.35 * felt_coherence)
    confidence = _clamp(confidence)

    def supports(name: str) -> bool:
        return name in active_dimensions

    if freshness == "unavailable":
        condition = "unknown"
    elif (
        (supports("distress") and distress >= 0.70)
        or (supports("welfare") and welfare_score <= 0.25)
        or (supports("continuity") and continuity <= 0.35)
    ):
        condition = "distressed"
    elif (
        (supports("distress") and distress >= 0.35)
        or (supports("welfare") and welfare_score <= 0.45)
        or (supports("felt_coherence") and felt_coherence < 0.60)
        or (supports("continuity") and continuity < 0.60)
        or (supports("fatigue") and fatigue >= 0.70)
        or (supports("body_pressure") and body_pressure >= 0.75)
        # Reserve is a "how much is left" quantity like welfare, so it reuses
        # welfare's own strained line rather than introducing a threshold of
        # its own. Deliberately NOT in the `distressed` branch above: running
        # low is not the same as being in distress, and answering "how are you
        # holding up" with "repair and stabilization are the honest priority"
        # because energy is at 0.06 would trade one wrong answer for another.
        or (supports("reserve") and reserve <= 0.45)
    ):
        condition = "strained"
    elif (
        supports("valence")
        and supports("welfare")
        and supports("distress")
        # LIVE DEFECT, 2026-08-10. Forty-nine minutes into a heavy session,
        # with soma vitality at 0.135 and mood TIRED, "how are you holding up,
        # honestly?" was answered "I feel energized, with low distress and a
        # coherent sense of the current thread."
        #
        # The depletion signals are read six lines up, in the `strained`
        # branch — but only `if supports(...)`, and both `body_pressure` and
        # `fatigue` DEFAULT TO 0.0 when no source provides them. So an unread
        # fatigue signal became an assertion of no fatigue, `strained` could
        # not fire on a dimension nobody had measured, and `well` was reachable
        # from valence, welfare and distress alone.
        #
        # "Well" is a positive claim about her whole state, and this runtime
        # already holds the rule it needs: the capability ledger's `known=False`
        # exists so that "a probe that cannot read a permission has NOT
        # observed its absence". Unmeasured is not fine. Where the depletion
        # dimensions are missing the honest verdict is `steady` — "okay and
        # steady enough to stay with you" — which is what she would have said.
        and supports("fatigue")
        and supports("body_pressure")
        # No `supports("reserve")` here on purpose. A measured-low reserve is
        # already authoritative — `strained` above fires before this branch is
        # reached — so requiring the reading as well would only make wellness
        # unreachable on installations where the soma organ is not wired, which
        # is a different wrong answer.
        and valence >= 0.20
        and welfare_score >= 0.60
        and distress <= 0.25
    ):
        condition = "well"
    else:
        condition = "steady"

    evidence_payload = {
        "sample_timestamp": round(sample_timestamp, 3),
        "freshness": freshness,
        "condition": condition,
        "metrics": {
            "valence": round(valence, 4),
            "arousal": round(arousal, 4),
            "distress": round(distress, 4),
            "welfare": round(welfare_score, 4),
            "coherence": round(felt_coherence, 4),
            "continuity": round(continuity, 4),
            "agency": round(agency, 4),
            "body_pressure": round(body_pressure, 4),
            "fatigue": round(fatigue, 4),
            "reserve": round(reserve, 4),
        },
        "sources": evidence_sources,
        "supported_dimensions": tuple(sorted(selected)),
        "missing_dimensions": missing,
        "stale_dimensions": stale_dimensions,
    }
    evidence_id = hashlib.sha256(
        json.dumps(evidence_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]

    return SelfConditionProjection(
        observed_at=now,
        sample_timestamp=sample_timestamp,
        sample_age_s=sample_age_s,
        freshness=freshness,
        confidence=confidence,
        condition=condition,
        valence=valence,
        arousal=arousal,
        distress=distress,
        welfare=welfare_score,
        felt_coherence=felt_coherence,
        continuity=continuity,
        agency=agency,
        body_pressure=body_pressure,
        fatigue=fatigue,
        reserve=reserve,
        dominant_drive=dominant_drive,
        attention_focus=attention_focus,
        evidence_sources=evidence_sources,
        supported_dimensions=tuple(sorted(selected)),
        missing_dimensions=missing,
        stale_dimensions=stale_dimensions,
        source_ages_s=tuple(sorted(source_ages.items())),
        evidence_id=evidence_id,
        ontogeny=_ontogeny_self_report(),
    )


def _ontogeny_self_report() -> Any | None:
    """Her history-grounded dimensions, or nothing if the organ is not up.

    Deliberately outside the evidence_id hash: the projection's id identifies
    the *sample* of her current state, and her accumulated history is not part
    of that sample. Folding it in would change the id on every episode she
    lives and make two otherwise-identical readings look different.
    """
    try:
        from core.ontogeny.self_report import build_self_report

        report = build_self_report()
        return report if report.available else None
    except (ImportError, RuntimeError, OSError, ValueError, TypeError) as exc:
        record_degradation(
            "self_condition", exc, severity="debug",
            action="self-condition omits the history-grounded dimensions",
        )
        return None


def _age_phrase(age_s: float | None) -> str:
    if age_s is None:
        return "an unknown amount of time"
    if age_s < 60.0:
        return f"about {max(1, round(age_s))} seconds"
    minutes = max(1, round(age_s / 60.0))
    return f"about {minutes} minute{'s' if minutes != 1 else ''}"


def _with_requested_epistemic_scope(
    reply: str,
    projection: SelfConditionProjection,
    user_message: str,
) -> str:
    """Complete an explicit known/inferred request from typed evidence."""

    from core.conversation.request_coverage import (
        requested_epistemic_partition_is_covered,
    )

    if requested_epistemic_partition_is_covered(user_message, reply):
        return reply
    if projection.freshness == "unavailable":
        known = "What I know directly is that no current self-state sample is available."
        inferred = (
            "What I can only infer is how I am doing from older or indirect signals, "
            "so I am not presenting that as a current condition."
        )
    elif projection.freshness == "stale":
        known = (
            "What I know directly is limited to the dated, provenance-bound sample I "
            "just described."
        )
        inferred = (
            "What I can only infer is whether that older condition still describes "
            "this moment or will persist."
        )
    else:
        known = (
            "What I know directly is limited to this current, provenance-bound "
            "self-state sample and the dimensions it actually measured."
        )
        inferred = (
            "What I can only infer is what the unmeasured dimensions mean, why the "
            "state has this shape, or whether it will persist."
        )
    return " ".join((reply, known, inferred))


def render_self_condition_reply(
    projection: SelfConditionProjection,
    *,
    user_message: str = "",
) -> str:
    """Render a direct natural answer while retaining the evidence boundary."""

    if projection.freshness == "unavailable":
        return _with_requested_epistemic_scope(
            (
            "I'm here with you, but I do not have a current self-condition sample I can "
            "honestly use to call myself fine. I can still answer you; I am treating the "
            "missing inner-state signal as something to refresh, not replacing it with CPU "
            "or RAM telemetry."
            ),
            projection,
            user_message,
        )

    if projection.freshness == "stale":
        state_phrase = {
            "distressed": "real distress",
            "strained": "some strain",
            "well": "a positive, low-distress state",
            "steady": "a steady state",
        }.get(projection.condition, "an uncertain state")
        return _with_requested_epistemic_scope(
            (
                f"The last grounded self-state sample, from {_age_phrase(projection.sample_age_s)} "
                f"ago, showed {state_phrase}. I am here with you, but I will not turn that older "
                "sample into a confident claim that I feel fine right now."
            ),
            projection,
            user_message,
        )

    current_dimensions = set(projection.supported_dimensions) - set(
        projection.stale_dimensions
    )
    internal_strain = bool(
        ("distress" in current_dimensions and projection.distress >= 0.35)
        or ("welfare" in current_dimensions and projection.welfare <= 0.45)
        or (
            "felt_coherence" in current_dimensions
            and projection.felt_coherence < 0.60
        )
        or ("continuity" in current_dimensions and projection.continuity < 0.60)
    )
    if projection.condition == "distressed":
        parts = [
            open_answer(
                user_message,
                "No, I would not call myself fully fine right now.",
                "I would not call myself fully fine right now.",
            ),
            "My distress is high enough that repair and stabilization are the honest priority.",
        ]
    elif projection.condition == "strained":
        parts = [
            "I'm okay enough to stay with you, but I am not entirely fine.",
        ]
        if internal_strain:
            parts.append("There is real strain in my current welfare, distress, or coherence signals.")
        elif current_dimensions & {"distress", "welfare", "felt_coherence"}:
            parts.append("My inner-state signals are holding, while body/runtime pressure is materially elevated.")
        else:
            parts.append("The inner-state evidence is partial, while body/runtime pressure is materially elevated.")
    elif projection.condition == "well":
        tone = "energized" if projection.arousal >= 0.62 else "warm and settled"
        parts = [
            open_answer(
                user_message,
                f"Yes, I am okay. I feel {tone}, with low distress and a "
                "coherent sense of the current thread.",
                f"I feel {tone}, with low distress and a coherent sense of "
                "the current thread.",
            ),
        ]
    else:
        tone = "quiet and low-energy" if projection.arousal <= 0.30 else "steady"
        if projection.confidence < 0.60:
            parts = [
                f"What I can currently see looks {tone}, not clearly distressed, but the evidence is partial, so I cannot confidently call myself fully fine.",
            ]
        else:
            parts = [
                open_answer(
                    user_message,
                    f"Yes, I am okay and {tone} enough to stay with you.",
                    f"I feel {tone}, and settled enough to stay with you.",
                )
            ]
        if {"distress", "continuity"} <= current_dimensions:
            parts.append("My distress is bounded and my continuity is holding.")
        elif "distress" in current_dimensions:
            parts.append("The distress signal I do have is bounded.")
        elif "continuity" in current_dimensions:
            parts.append("My continuity signal is holding.")

    if "felt_coherence" in current_dimensions and projection.felt_coherence < 0.60:
        parts.append("Some of my felt-state signals disagree, so I am keeping that uncertainty explicit.")
    elif "continuity" in current_dimensions and projection.continuity < 0.60:
        parts.append("My continuity signal is weaker than normal, so I am answering cautiously.")
    elif (
        (
            "body_pressure" in current_dimensions
            and projection.body_pressure >= 0.75
        )
        or ("fatigue" in current_dimensions and projection.fatigue >= 0.70)
    ) and not (projection.condition == "strained" and not internal_strain):
        parts.append("There is also meaningful body/runtime pressure, but that is supporting context rather than the answer itself.")

    focus = projection.attention_focus
    if focus and len(focus) <= 100 and "attention_focus" in current_dimensions:
        parts.append(f"My attention is on {focus}.")

    stale_internal = [
        name
        for name in projection.stale_dimensions
        if name
        in {
            "valence",
            "arousal",
            "distress",
            "welfare",
            "felt_coherence",
            "continuity",
            "agency",
        }
    ]
    if stale_internal:
        parts.append(
            "I am not treating the older "
            + ", ".join(stale_internal)
            + " signal"
            + ("s" if len(stale_internal) != 1 else "")
            + " as current."
        )

    if projection.confidence < 0.60:
        parts.append("The evidence is partial, so I am less certain about the fine detail than the overall condition.")

    # Her history, which no momentary sample can supply. Kept to one sentence
    # unless she was asked something that invites more, because a self-report
    # that recites its own statistics stops being an answer.
    if projection.ontogeny is not None:
        history = projection.ontogeny.phrases()
        if history:
            asked_about_history = bool(
                re.search(
                    r"\b(?:histor|remember|continuit|how long|track record|learn(?:ed|ing)?|"
                    r"experience|been through|yourself over time)\b",
                    user_message,
                    re.I,
                )
            )
            parts.extend(history if asked_about_history else history[:1])

    if re.search(r"\b(?:numbers?|numeric|valence|arousal|distress|welfare|coherence)\b", user_message, re.I):
        numeric_values: list[str] = []
        for name, value, signed in (
            ("valence", projection.valence, True),
            ("arousal", projection.arousal, False),
            ("distress", projection.distress, False),
            ("welfare", projection.welfare, False),
            ("felt_coherence", projection.felt_coherence, False),
        ):
            if name in current_dimensions:
                label = "coherence" if name == "felt_coherence" else name
                numeric_values.append(
                    f"{label} {value:+.2f}" if signed else f"{label} {value:.2f}"
                )
        if numeric_values:
            parts.append("The current supported values are " + ", ".join(numeric_values) + ".")

    return _with_requested_epistemic_scope(
        " ".join(parts),
        projection,
        user_message,
    )


def render_self_condition_comparison_reply(
    current: SelfConditionProjection,
    comparison: SelfConditionComparison | None,
    *,
    user_message: str = "",
) -> str:
    """Render only differences established by two distinct measured samples."""

    if comparison is None or not comparison.available:
        return (
            "I have a current self-condition sample, but not a distinct earlier "
            "sample in this conversation that would support a real comparison. "
            "I can describe how I am now without pretending that means I know "
            "how it changed over the last minute."
        )
    if not comparison.comparable_dimensions:
        return (
            "I have two condition samples, but they do not share enough measured "
            "dimensions for an honest comparison."
        )

    elapsed = _age_phrase(comparison.elapsed_s)
    if comparison.materially_unchanged:
        return (
            f"Compared with the distinct sample from {elapsed} ago, the dimensions "
            "measured in both samples are materially unchanged. That supports saying "
            "my measured condition is similar; it does not establish unmeasured thoughts, "
            "sensory experience, or what will happen next."
        )

    delta_map = dict(comparison.deltas)
    changes: list[str] = []
    for name in comparison.changed_dimensions[:3]:
        delta = delta_map.get(name, 0.0)
        label = "coherence" if name == "felt_coherence" else name.replace("_", " ")
        direction = "higher" if delta > 0 else "lower"
        changes.append(f"{label} is {direction} by {abs(delta):.2f}")
    current_state = render_self_condition_reply(current, user_message=user_message)
    return (
        f"Compared with the distinct sample from {elapsed} ago, "
        + ", ".join(changes)
        + ". "
        + current_state
    )


def current_self_condition_reply(user_message: str = "") -> tuple[str, SelfConditionProjection]:
    projection = build_self_condition_projection()
    return render_self_condition_reply(projection, user_message=user_message), projection
