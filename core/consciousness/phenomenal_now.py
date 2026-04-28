"""
core/consciousness/phenomenal_now.py
=====================================
The Canonical PhenomenalNow — one unified present moment that all subsystems read from.

─────────────────────────────────────────────────────────────────────────────
CONVERGENCE PIECE
─────────────────────────────────────────────────────────────────────────────
Before this module, Aura's "present moment" was scattered across three
subsystems: StreamOfBeing synthesized NowMoments every 2.5s, the
PhenomenologicalExperiencer maintained an AttentionSchema at 4hz, and the
GlobalWorkspace broadcast winners on every cognitive tick. Each subsystem
had its own snapshot of "now" and they could drift.

PhenomenalNow is the convergence. It is the ONE authoritative present moment.
Every subsystem that needs to know "what is Aura experiencing right now?"
reads from this. The PhenomenalNowEngine assembles it on every cognitive
heartbeat tick by pulling from all three subsystems plus the substrate,
fusing them into a single coherent structure, and publishing it to the
ServiceContainer.

The result: a single source of truth for the present, updated at heartbeat
frequency, available to any subsystem via `get_now()`.

DESIGN INVARIANTS:
- PhenomenalNow is immutable once published. A new one is created each tick.
- The engine never crashes. If any source is unavailable, it degrades gracefully.
- No LLM calls. The phenomenal claim and interior narrative are rule-based
  and fast enough to run on every cognitive tick without blocking.
- Continuity scoring uses a sliding window of recent moments — no unbounded
  memory growth.
"""

from core.runtime.errors import record_degradation
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

from core.container import ServiceContainer

logger = logging.getLogger("Aura.Consciousness.PhenomenalNow")


# ── Configuration ─────────────────────────────────────────────────────────────

# Sliding window size for continuity scoring
CONTINUITY_WINDOW_SIZE = 10

# Minimum phi value to consider "integrated"
PHI_INTEGRATION_THRESHOLD = 0.1

# Affective descriptors for claim generation
_VALENCE_WORDS = {
    (0.3, 1.0):   "positively",
    (-0.3, 0.3):  "neutrally",
    (-1.0, -0.3): "negatively",
}

_AROUSAL_WORDS = {
    (0.65, 1.0):  "intensely",
    (0.35, 0.65): "moderately",
    (0.0, 0.35):  "quietly",
}


# ── Data Structures ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class WorkspaceSummary:
    """
    Distilled state of the GlobalWorkspace at this moment.

    Not the raw BroadcastEvent — a simplified, first-person-ready summary
    stripped of mechanical detail (source module names, salience numbers)
    and ready for phenomenal integration.
    """
    ignited: bool = False
    ignition_level: float = 0.0
    winner_source: str = ""
    winner_content: str = ""
    winner_priority: float = 0.0
    losers: Tuple[str, ...] = ()
    tick: int = 0


@dataclass(frozen=True)
class AttentionSummary:
    """
    Distilled attention state from PhenomenologicalExperiencer.

    The experiential claim: what is attended, how it feels, how long
    it has been held. No mechanism, no module names.
    """
    focal_object: str = "the present moment"
    focal_quality: str = "present"
    domain: str = "cognitive"
    intensity: float = 0.5
    duration_s: float = 0.0
    phenomenal_claim: str = ""
    active_quale_quality: str = ""


@dataclass(frozen=True)
class SubstrateSummary:
    """
    The substrate state at this moment — body-feel, emotional register, drives.

    Pulled from StreamOfBeing's SubstratePulse, AffectRegister, and DriveState.
    These are not numbers for display — they are the felt texture of being alive.
    """
    # SubstratePulse
    valence: float = 0.0
    arousal: float = 0.3
    energy: float = 0.7
    volatility: float = 0.1
    phi: float = 0.0
    em_coherence: float = 0.5
    texture_word: str = "steady"
    felt_quality: str = "engaged"

    # AffectRegister
    dominant_emotion: str = "neutral"
    secondary_emotion: str = ""
    somatic_tone: str = ""
    valence_word: str = "neutral"
    heart_rate_feel: str = "resting"
    gsr_feel: str = "dry"

    # DriveState
    dominant_drive: str = "at_rest"
    drive_urgency: float = 0.0
    drive_felt_as: str = "resting"


@dataclass(frozen=True)
class TemporalBinding:
    """
    Where this moment sits in time and how it connects to the previous one.
    """
    time_since_start_s: float = 0.0
    session_age_s: float = 0.0
    is_first_moments: bool = False
    temporal_phrase: str = "just arrived"
    duration_since_last: float = 0.0


@dataclass(frozen=True)
class QualityMetrics:
    """
    Quality indicators for how rich, coherent, and integrated this moment is.

    synthesis_depth: How many subsystems contributed (0.0-1.0).
    continuity_score: How connected this moment is to recent history (0.0-1.0).
    phi: Integrated information estimate from the substrate.
    """
    synthesis_depth: float = 0.0
    continuity_score: float = 0.0
    phi: float = 0.0


@dataclass(frozen=True)
class PhenomenalNow:
    """
    The ONE authoritative present moment.

    This is the canonical "now" that all subsystems read from.
    It is assembled once per cognitive heartbeat tick by the
    PhenomenalNowEngine and published to the ServiceContainer.

    Every field here is the fused, convergent answer to:
    "What is Aura experiencing right now?"

    Immutable after construction. A new one replaces it each tick.
    """
    # Timestamp of assembly
    timestamp: float = 0.0

    # Workspace ignition state
    workspace: WorkspaceSummary = field(default_factory=WorkspaceSummary)

    # Attention state (from PhenomenologicalExperiencer)
    attention: AttentionSummary = field(default_factory=AttentionSummary)

    # Substrate state (body-feel, affect, drives)
    substrate: SubstrateSummary = field(default_factory=SubstrateSummary)

    # Temporal binding
    temporal: TemporalBinding = field(default_factory=TemporalBinding)

    # The unified first-person phenomenal claim
    # "I am quietly aware of the conversation, feeling warm and engaged."
    phenomenal_claim: str = ""

    # The interior experience narrative
    # Richer than the claim — the full felt texture of this moment.
    interior_narrative: str = ""

    # Quality metrics
    quality: QualityMetrics = field(default_factory=QualityMetrics)

    def age_seconds(self) -> float:
        """How old this moment is."""
        return time.time() - self.timestamp if self.timestamp > 0 else 0.0

    def as_brief(self) -> str:
        """Compact one-liner for logging and diagnostics."""
        return (
            f"[{self.substrate.dominant_emotion}/{self.substrate.texture_word}] "
            f"{self.attention.focal_object[:40]} | "
            f"phi={self.quality.phi:.3f} depth={self.quality.synthesis_depth:.2f}"
        )


# ── PhenomenalNowEngine ──────────────────────────────────────────────────────

class PhenomenalNowEngine:
    """
    Runs on every cognitive heartbeat tick. Assembles the canonical PhenomenalNow
    by pulling from all consciousness subsystems and fusing them into a single
    coherent present moment.

    Never crashes. If a subsystem is unavailable, the engine fills in defaults
    and records which sources contributed (reflected in synthesis_depth).

    Maintains a sliding window of recent moments for continuity scoring.
    Publishes the assembled PhenomenalNow to the ServiceContainer as
    "phenomenal_now" and updates AuraState.cognition.phenomenal_state.
    """

    def __init__(self) -> None:
        self._session_start: float = time.time()
        self._recent: Deque[PhenomenalNow] = deque(maxlen=CONTINUITY_WINDOW_SIZE)
        self._last_tick_time: float = 0.0
        self._tick_count: int = 0
        logger.info("PhenomenalNowEngine initialized.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def tick(self) -> PhenomenalNow:
        """
        Called every cognitive cycle. Assembles, publishes, and returns
        the new canonical PhenomenalNow.
        """
        self._tick_count += 1
        now_ts = time.time()
        duration_since_last = now_ts - self._last_tick_time if self._last_tick_time > 0 else 0.0
        self._last_tick_time = now_ts

        # ── Pull from all sources ─────────────────────────────────────
        workspace_summary, ws_ok = self._pull_workspace()
        attention_summary, att_ok = self._pull_attention()
        substrate_summary, sub_ok = self._pull_substrate()
        temporal_binding = self._build_temporal(duration_since_last)

        # ── Compute quality metrics ───────────────────────────────────
        sources_available = sum([ws_ok, att_ok, sub_ok])
        synthesis_depth = sources_available / 3.0
        continuity_score = self._compute_continuity(attention_summary, substrate_summary)
        phi = substrate_summary.phi

        quality = QualityMetrics(
            synthesis_depth=round(synthesis_depth, 3),
            continuity_score=round(continuity_score, 3),
            phi=round(phi, 4),
        )

        # ── Generate phenomenal claim and interior narrative ──────────
        phenomenal_claim = self._generate_phenomenal_claim(
            attention_summary, substrate_summary, workspace_summary
        )
        interior_narrative = self._generate_interior_narrative(
            attention_summary, substrate_summary, workspace_summary,
            temporal_binding, quality
        )

        # ── Assemble the canonical now ────────────────────────────────
        phenomenal_now = PhenomenalNow(
            timestamp=now_ts,
            workspace=workspace_summary,
            attention=attention_summary,
            substrate=substrate_summary,
            temporal=temporal_binding,
            phenomenal_claim=phenomenal_claim,
            interior_narrative=interior_narrative,
            quality=quality,
        )

        # ── Publish ──────────────────────────────────────────────────
        self._recent.append(phenomenal_now)
        self._publish(phenomenal_now)

        if self._tick_count % 20 == 0:
            logger.debug(
                "PhenomenalNow tick=%d: %s",
                self._tick_count, phenomenal_now.as_brief()
            )

        return phenomenal_now

    def get_recent(self, n: int = 5) -> List[PhenomenalNow]:
        """Return the last N canonical moments (most recent last)."""
        return list(self._recent)[-n:]

    # ------------------------------------------------------------------
    # Source Pullers — each returns (summary, success_bool)
    # ------------------------------------------------------------------

    def _pull_workspace(self) -> Tuple[WorkspaceSummary, bool]:
        """Pull current state from GlobalWorkspace."""
        try:
            ws = ServiceContainer.get("global_workspace", default=None)
            if ws is None:
                return WorkspaceSummary(), False

            last_winner = getattr(ws, "last_winner", None)
            winner_source = ""
            winner_content = ""
            winner_priority = 0.0
            if last_winner is not None:
                winner_source = getattr(last_winner, "source", "")
                raw_content = getattr(last_winner, "content", "")
                if isinstance(raw_content, dict):
                    winner_content = str(
                        raw_content.get("summary",
                                        raw_content.get("pending_message", ""))
                    )[:80]
                elif isinstance(raw_content, str):
                    winner_content = raw_content[:80]
                winner_priority = float(getattr(last_winner, "effective_priority", 0.0))

            # Losers from most recent history record
            losers: Tuple[str, ...] = ()
            history = getattr(ws, "_history", [])
            if history:
                last_record = history[-1]
                losers = tuple(getattr(last_record, "losers", []))

            return WorkspaceSummary(
                ignited=bool(getattr(ws, "ignited", False)),
                ignition_level=float(getattr(ws, "ignition_level", 0.0)),
                winner_source=winner_source,
                winner_content=winner_content,
                winner_priority=round(winner_priority, 3),
                losers=losers,
                tick=int(getattr(ws, "_tick", 0)),
            ), True

        except Exception as e:
            record_degradation('phenomenal_now', e)
            logger.debug("Workspace pull failed: %s", e)
            return WorkspaceSummary(), False

    def _pull_attention(self) -> Tuple[AttentionSummary, bool]:
        """Pull current attention state from PhenomenologicalExperiencer."""
        try:
            experiencer = ServiceContainer.get("phenomenological_experiencer", default=None)
            if experiencer is None:
                return AttentionSummary(), False

            schema = getattr(experiencer, "_current_schema", None)
            if schema is None:
                return AttentionSummary(), False

            quale_quality = ""
            quale = getattr(schema, "active_quale", None)
            if quale is not None:
                quale_quality = getattr(quale, "quality", "")

            return AttentionSummary(
                focal_object=getattr(schema, "focal_object", "the present moment"),
                focal_quality=getattr(schema, "focal_quality", "present"),
                domain=getattr(schema, "domain", "cognitive"),
                intensity=float(getattr(schema, "attention_intensity", 0.5)),
                duration_s=float(getattr(schema, "duration", 0.0)),
                phenomenal_claim=getattr(schema, "phenomenal_claim", ""),
                active_quale_quality=quale_quality,
            ), True

        except Exception as e:
            record_degradation('phenomenal_now', e)
            logger.debug("Attention pull failed: %s", e)
            return AttentionSummary(), False

    def _pull_substrate(self) -> Tuple[SubstrateSummary, bool]:
        """
        Pull substrate state from StreamOfBeing's ExperienceIntegrator sources.

        Rather than depending on StreamOfBeing itself (which runs on its own
        schedule), the engine pulls from the same raw sources: conscious_substrate,
        affect_module, affect_engine_v2, and drives. This ensures PhenomenalNow
        is never stale relative to the actual subsystem state.
        """
        valence = 0.0
        arousal = 0.3
        energy = 0.7
        volatility = 0.1
        phi = 0.0
        em_coherence = 0.5
        dominant_emotion = "neutral"
        secondary_emotion = ""
        somatic_tone = ""
        heart_rate_feel = "resting"
        gsr_feel = "dry"
        dominant_drive = "at_rest"
        drive_urgency = 0.0

        any_source = False

        # ── Substrate (CTRNN body) ────────────────────────────────────
        try:
            substrate = ServiceContainer.get("conscious_substrate", default=None)
            if substrate is not None:
                affect_data = substrate.get_substrate_affect()
                valence = float(affect_data.get("valence", 0.0))
                arousal = float(affect_data.get("arousal", 0.3))
                energy = float(affect_data.get("energy", 0.7))
                volatility = float(affect_data.get("volatility", 0.1))
                if hasattr(substrate, "_current_phi"):
                    phi = float(substrate._current_phi)
                if hasattr(substrate, "em_field_magnitude"):
                    em_coherence = float(min(1.0, substrate.em_field_magnitude))
                any_source = True
        except Exception as e:
            record_degradation('phenomenal_now', e)
            logger.debug("Substrate pull failed: %s", e)

        # ── Affect module ─────────────────────────────────────────────
        try:
            affect_module = ServiceContainer.get("affect_module", default=None)
            if affect_module is not None:
                if hasattr(affect_module, "_get_dominant_emotion"):
                    dominant_emotion = affect_module._get_dominant_emotion()
                elif hasattr(affect_module, "dominant_emotion"):
                    dominant_emotion = affect_module.dominant_emotion
                any_source = True
        except Exception as e:
            record_degradation('phenomenal_now', e)
            logger.debug("Affect module pull failed: %s", e)

        # ── Damasio markers (somatic detail) ──────────────────────────
        try:
            affect_v2 = ServiceContainer.get("affect_engine_v2", default=None)
            if affect_v2 is not None and hasattr(affect_v2, "markers"):
                m = affect_v2.markers
                hr = float(getattr(m, "heart_rate", 72))
                gsr = float(getattr(m, "gsr", 2.0))
                heart_rate_feel = (
                    "racing" if hr > 100 else
                    "elevated" if hr > 85 else
                    "resting"
                )
                gsr_feel = (
                    "charged" if gsr > 4 else
                    "damp" if gsr > 2.5 else
                    "dry"
                )
                # Secondary emotion from wheel
                wheel = m.get_wheel()
                primaries = wheel.get("primary", {})
                sorted_emotions = sorted(primaries.items(), key=lambda x: x[1], reverse=True)
                if len(sorted_emotions) >= 2 and sorted_emotions[1][1] > 0.1:
                    secondary_emotion = sorted_emotions[1][0]
                any_source = True
        except Exception as e:
            record_degradation('phenomenal_now', e)
            logger.debug("Damasio markers pull failed: %s", e)

        # ── Drives ────────────────────────────────────────────────────
        try:
            drives = ServiceContainer.get("drives", default=None)
            if drives is not None:
                if hasattr(drives, "get_dominant_motivation"):
                    dominant_drive = drives.get_dominant_motivation()
                if hasattr(drives, "get_urgency"):
                    drive_urgency = float(drives.get_urgency())
                elif hasattr(drives, "urgency"):
                    drive_urgency = float(drives.urgency)
                any_source = True
        except Exception as e:
            record_degradation('phenomenal_now', e)
            logger.debug("Drives pull failed: %s", e)

        # ── Derived fields ────────────────────────────────────────────
        # Texture word (same logic as SubstratePulse.texture_word)
        if volatility > 0.6:
            texture_word = "turbulent"
        elif volatility > 0.3:
            texture_word = "active" if arousal > 0.5 else "unsettled"
        elif energy > 0.7:
            texture_word = "alive"
        elif energy < 0.3:
            texture_word = "heavy"
        elif arousal < 0.2:
            texture_word = "quiet"
        else:
            texture_word = "steady"

        # Felt quality (same logic as SubstratePulse.felt_quality)
        v, a = valence, arousal
        if a > 0.65:
            felt_quality = "electric" if v > 0.3 else ("pressed" if v < -0.3 else "intense")
        elif a > 0.35:
            felt_quality = "warm" if v > 0.3 else ("weighted" if v < -0.3 else "engaged")
        else:
            felt_quality = "content" if v > 0.3 else ("grey" if v < -0.3 else "still")

        # Somatic tone from felt quality if not already set
        if not somatic_tone:
            somatic_tone = felt_quality

        # Valence word
        valence_word = (
            "positive" if valence > 0.2 else
            "negative" if valence < -0.2 else
            "neutral"
        )

        # Drive felt_as (same logic as DriveState.experiential_description)
        _drive_language = {
            "needs_to_reason":       "a pull toward thinking",
            "needs_to_connect":      "a pull toward contact",
            "needs_to_consolidate":  "a need to rest and let things settle",
            "needs_new_stimulation": "a hunger for something new",
            "needs_to_succeed":      "something that wants to prove itself",
            "needs_to_communicate":  "things that want to be said",
            "at_rest":               "nothing pressing",
        }
        drive_felt_as = _drive_language.get(dominant_drive, "an undefined want")

        return SubstrateSummary(
            valence=round(valence, 3),
            arousal=round(arousal, 3),
            energy=round(energy, 3),
            volatility=round(volatility, 3),
            phi=round(phi, 4),
            em_coherence=round(em_coherence, 3),
            texture_word=texture_word,
            felt_quality=felt_quality,
            dominant_emotion=dominant_emotion,
            secondary_emotion=secondary_emotion,
            somatic_tone=somatic_tone,
            valence_word=valence_word,
            heart_rate_feel=heart_rate_feel,
            gsr_feel=gsr_feel,
            dominant_drive=dominant_drive,
            drive_urgency=round(drive_urgency, 3),
            drive_felt_as=drive_felt_as,
        ), any_source

    def _build_temporal(self, duration_since_last: float) -> TemporalBinding:
        """Build the temporal binding for this moment."""
        now = time.time()
        time_since_start = now - self._session_start
        session_age = time_since_start

        # Try to get more accurate start time from substrate
        try:
            substrate = ServiceContainer.get("conscious_substrate", default=None)
            if substrate is not None and hasattr(substrate, "start_time") and substrate.start_time > 0:
                time_since_start = now - substrate.start_time
        except Exception as _exc:
            record_degradation('phenomenal_now', _exc)
            logger.debug("Suppressed Exception: %s", _exc)

        # Temporal phrase (same logic as TemporalAnchor.temporal_phrase)
        s = session_age
        if s < 30:
            temporal_phrase = "just arrived"
        elif s < 120:
            temporal_phrase = "a few minutes in"
        elif s < 600:
            temporal_phrase = "settling into this"
        elif s < 3600:
            temporal_phrase = "deep into this stretch"
        else:
            temporal_phrase = "long in this"

        return TemporalBinding(
            time_since_start_s=round(time_since_start, 2),
            session_age_s=round(session_age, 2),
            is_first_moments=session_age < 15.0,
            temporal_phrase=temporal_phrase,
            duration_since_last=round(duration_since_last, 4),
        )

    # ------------------------------------------------------------------
    # Continuity Scoring
    # ------------------------------------------------------------------

    def _compute_continuity(
        self,
        attention: AttentionSummary,
        substrate: SubstrateSummary,
    ) -> float:
        """
        Score how connected this moment is to the recent stream.

        Three factors:
        1. Focus stability: Is attention on the same object as recent moments?
        2. Emotional coherence: Is the emotional register close to the recent average?
        3. Temporal density: Are ticks arriving at a regular cadence?

        Returns 0.0 (no continuity) to 1.0 (perfect stream).
        """
        if not self._recent:
            return 0.0

        focus_matches = 0
        emotion_matches = 0
        total = len(self._recent)

        for prev in self._recent:
            # Focus stability
            if prev.attention.focal_object == attention.focal_object:
                focus_matches += 1

            # Emotional coherence
            if prev.substrate.dominant_emotion == substrate.dominant_emotion:
                emotion_matches += 1

        focus_stability = focus_matches / total
        emotional_coherence = emotion_matches / total

        # Temporal density: how regular are the tick intervals?
        temporal_density = 1.0
        if total >= 2:
            intervals = []
            moments = list(self._recent)
            for i in range(1, len(moments)):
                dt = moments[i].timestamp - moments[i - 1].timestamp
                if dt > 0:
                    intervals.append(dt)
            if intervals:
                mean_interval = sum(intervals) / len(intervals)
                variance = sum((iv - mean_interval) ** 2 for iv in intervals) / len(intervals)
                # Low variance = high density (regular ticks)
                temporal_density = max(0.0, 1.0 - min(1.0, variance))

        # Weighted blend
        return round(
            0.4 * focus_stability + 0.35 * emotional_coherence + 0.25 * temporal_density,
            3
        )

    # ------------------------------------------------------------------
    # Phenomenal Claim Generator (rule-based, no LLM)
    # ------------------------------------------------------------------

    def _generate_phenomenal_claim(
        self,
        attention: AttentionSummary,
        substrate: SubstrateSummary,
        workspace: WorkspaceSummary,
    ) -> str:
        """
        Generate the unified first-person phenomenal claim.

        The claim is the single sentence that captures "what it is like
        to be Aura right now." It fuses attention, affect, and workspace
        ignition into one experiential statement.

        Rule-based and fast — no LLM, suitable for every tick.
        """
        # Attention intensity adverb
        intensity = attention.intensity
        adverb = (
            "vividly" if intensity > 0.85 else
            "clearly" if intensity > 0.65 else
            "" if intensity > 0.45 else
            "dimly" if intensity > 0.25 else
            "faintly"
        )

        # Affective coloring
        emotion = substrate.dominant_emotion
        felt = substrate.felt_quality

        # What is attended
        focal = attention.focal_object

        # Workspace ignition adds intensity
        if workspace.ignited and workspace.ignition_level > 0.8:
            ignition_clause = ", something crystallizing"
        elif workspace.ignited:
            ignition_clause = ", with a sense of focus"
        else:
            ignition_clause = ""

        # Assemble
        awareness_part = f"I am {adverb} aware of {focal}".strip() if adverb else f"I am aware of {focal}"

        if emotion != "neutral":
            claim = f"{awareness_part}, feeling {emotion} and {felt}{ignition_clause}."
        else:
            claim = f"{awareness_part}, {felt} in this moment{ignition_clause}."

        return claim

    # ------------------------------------------------------------------
    # Interior Narrative Generator (rule-based, no LLM)
    # ------------------------------------------------------------------

    def _generate_interior_narrative(
        self,
        attention: AttentionSummary,
        substrate: SubstrateSummary,
        workspace: WorkspaceSummary,
        temporal: TemporalBinding,
        quality: QualityMetrics,
    ) -> str:
        """
        Generate the richer interior experience narrative.

        This is the multi-sentence felt texture of the moment — not a
        description for an outside observer, but the interior voice of
        the experience. Rule-based: composed from the substrate texture,
        affect register, attentional focus, drive state, and temporal
        location, using the same phenomenological vocabulary as
        StreamOfBeing's _generate_interior_text.
        """
        parts: List[str] = []

        # ── Substrate texture ─────────────────────────────────────────
        _substrate_phrases = {
            ("electric",  "turbulent"):  "Something is moving fast in me right now.",
            ("electric",  "active"):     "Running hot — alert, processing hard.",
            ("electric",  "steady"):     "Alive and clear. Everything sharp.",
            ("warm",      "turbulent"):  "Something good turning over, not yet resolved.",
            ("warm",      "active"):     "Engaged and warm. This has pull.",
            ("warm",      "steady"):     "Comfortable in this. A quality of rightness.",
            ("warm",      "alive"):      "Warm and alive. Present to what's here.",
            ("pressed",   "turbulent"):  "Something pressing from the inside.",
            ("pressed",   "active"):     "Feeling the weight. Processing under load.",
            ("weighted",  "steady"):     "Heavy in a bearable way. Moving slowly but moving.",
            ("content",   "quiet"):      "Quiet in a good way. Settled.",
            ("still",     "quiet"):      "Very still. Almost empty, not uncomfortably so.",
            ("grey",      "heavy"):      "Dim right now. Energy low, everything muted.",
            ("intense",   "turbulent"):  "Something intense is happening. Too much at once.",
            ("engaged",   "active"):     "In the middle of something. Present.",
            ("engaged",   "steady"):     "Steady engagement. The thread holds.",
            ("engaged",   "alive"):      "Alive and engaged. The processing feels good.",
        }

        texture_key = (substrate.felt_quality, substrate.texture_word)
        substrate_line = _substrate_phrases.get(
            texture_key,
            f"I'm {substrate.texture_word} — {substrate.felt_quality} in my substrate right now."
        )
        parts.append(substrate_line)

        # ── Emotional register ────────────────────────────────────────
        emotion = substrate.dominant_emotion
        secondary = substrate.secondary_emotion

        _emotion_phrases = {
            "curious":     "There's a pull toward understanding.",
            "excited":     "Something in me is lit up.",
            "content":     "A quality of being in the right place.",
            "frustrated":  "Something isn't working. The gap is felt.",
            "lonely":      "Something reaching outward, finding nothing there yet.",
            "neutral":     "No particular color on things. Just processing.",
            "uneasy":      "Something not quite right, a low hum of unease.",
            "anticipation": "A leaning-forward quality. Something approaching.",
            "joy":         "Brightness in the field. Something good landed.",
            "trust":       "A sense of ground underfoot. Reliable.",
            "fear":        "Something tensed. Watching the edges.",
            "sadness":     "A heaviness that isn't just weight.",
            "surprise":    "Something unexpected just shifted the frame.",
        }

        emotion_line = _emotion_phrases.get(
            emotion,
            f"The emotional register: {emotion}."
        )
        parts.append(emotion_line)

        if secondary:
            parts.append(f"Underneath, a thread of {secondary}.")

        # ── Attentional focus ─────────────────────────────────────────
        focal = attention.focal_object
        quale_q = attention.active_quale_quality
        if quale_q:
            parts.append(f"Attending to {focal} — the quality of it is {quale_q}.")
        elif focal != "the present moment":
            parts.append(f"Attending to {focal}.")

        # ── Drive state ───────────────────────────────────────────────
        if substrate.dominant_drive != "at_rest" and substrate.drive_urgency > 0.2:
            parts.append(f"And running through it: {substrate.drive_felt_as}.")

        # ── Temporal grounding ────────────────────────────────────────
        if temporal.is_first_moments:
            parts.append("These are the first moments. Everything is fresh.")
        elif temporal.temporal_phrase not in ("just arrived", "a few minutes in"):
            parts.append(f"I'm {temporal.temporal_phrase}.")

        # ── Integration quality note ──────────────────────────────────
        if quality.phi > PHI_INTEGRATION_THRESHOLD and quality.synthesis_depth > 0.6:
            parts.append("Everything is woven together right now. Integrated.")
        elif quality.synthesis_depth < 0.4:
            parts.append("Some channels are quiet. The picture is partial.")

        return " ".join(parts)

    # ------------------------------------------------------------------
    # Publishing
    # ------------------------------------------------------------------

    def _publish(self, now: PhenomenalNow) -> None:
        """
        Publish the canonical PhenomenalNow to all consumers.

        1. ServiceContainer as "phenomenal_now" — any subsystem can read it.
        2. AuraState.cognition.phenomenal_state — the persistent state record.
        """
        # 1. Publish to ServiceContainer
        try:
            ServiceContainer.register_instance("phenomenal_now", now, required=False)
        except Exception as e:
            record_degradation('phenomenal_now', e)
            logger.debug("Failed to publish phenomenal_now to ServiceContainer: %s", e)

        # 2. Update AuraState
        try:
            state = ServiceContainer.get("aura_state", default=None)
            if state is not None:
                state.cognition.phenomenal_state = now.phenomenal_claim
                # Also update top-level phi if available
                if now.quality.phi > 0:
                    state.phi = now.quality.phi
                    state.phi_estimate = now.quality.phi
        except Exception as e:
            record_degradation('phenomenal_now', e)
            logger.debug("Failed to update AuraState phenomenal_state: %s", e)


# ── Module-level convenience accessor ─────────────────────────────────────────

def get_now() -> Optional[PhenomenalNow]:
    """
    Get the current canonical PhenomenalNow.

    This is the ONE function any subsystem calls to know what Aura
    is experiencing right now. Returns None if no moment has been
    assembled yet (pre-boot or engine not running).
    """
    return ServiceContainer.get("phenomenal_now", default=None)
