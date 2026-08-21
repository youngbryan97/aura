"""The organ itself: one object, wired into the live path, doing its job from episode one.

There is a version of this project that sits in shadow for six months writing
predictions nobody reads, and then asks to be trusted. That version is inert,
and inert is a polite word for useless. This one is causal on the first
episode, through three surfaces that need three different amounts of trust:

**Now, needing no trust at all — novelty.** The reservoir has a state whether
or not any head is fitted, and how far that state sits from the centre of her
lived distribution is a fact, not a claim. "I have not been anywhere like this
before" is exactly the signal that should buy a situation more thought than its
surface suggests, and it is available on day one.

**Now, needing only counting — the track record.** In situations like this one,
how often did it actually go well, and how sure can she be given how few times
she has been here? That is arithmetic over her own history with a Wilson
interval, and it is the honest source for how confident she sounds. It is also
willing to say the unflattering thing, which is what separates a track record
from a mood.

**Later, needing earned trust — the decision itself.** The head predicts
whether an episode will go well given each candidate action, and climbs the
ladder in ``authority.py`` from shadow to advisory to deciding, on held-out
evidence, with the incumbent keeping a permanent slice so the comparison never
becomes unrepeatable.

Every path through this file degrades to the incumbent. A broken organ costs
Aura the learning, never the decision.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from core.ontogeny import telemetry
from core.ontogeny.authority import AuthorityLedger, AuthorityStage, get_authority_ledger
from core.ontogeny.authority_observation import AuthorityObservationMixin
from core.ontogeny.calibration import (
    CANDIDATE_VALIDATION,
    OPERATIONAL_SHADOW,
    CalibrationMonitor,
    CalibrationObservation,
    TrackRecord,
    TrackRecordIndex,
)
from core.ontogeny.experience import (
    Episode,
    ExperienceSpine,
    Outcome,
    OutcomeKind,
    Provenance,
    get_experience_spine,
)
from core.ontogeny.features import (
    EXECUTIVE_ADMISSION,
    FeatureSchema,
    RunningMoments,
    design_row,
)
from core.ontogeny.heads import PredictionHead
from core.ontogeny.reservation import Decider, ExplorationReservation, Reservation, get_reservation
from core.ontogeny.resolution import OutcomeSweeper, ResolverRegistry, get_resolvers
from core.ontogeny.state import DEFAULT_SEED, DEFAULT_UNITS, OntogeneticState, StateReading
from core.ontogeny.trainer import Trainer, TrainingResult, design_names, design_width
from core.runtime.errors import record_degradation
from core.runtime.foreground_guard import foreground_activity_reason
from core.runtime.lockdep import LockRank, checked_lock
from core.runtime.sqlite_support import connecting

logger = logging.getLogger("Aura.Ontogeny")

#: How often the organ refits, when there is new evidence to refit on.
TRAIN_INTERVAL_S = 1800.0

#: New graded episodes required before a refit is worth the cycles.
TRAIN_MIN_NEW_EVIDENCE = 50

#: A zero-length sleep can immediately reschedule the calling worker on macOS
#: and leave the event loop starved for the whole recurrent replay. A tenth of
#: a millisecond is long enough to hand the core back without materially
#: changing a half-hour maintenance fit.
TRAIN_COOPERATIVE_YIELD_S = 0.0001

#: How often the state checkpoint is written. The state is small; losing an
#: hour of it to a crash would be a needless amputation.
CHECKPOINT_INTERVAL_S = 300.0

#: Episodes whose bucket is remembered so a later resolution can find its
#: tally. Bounded: an outcome that lands after this many decisions have gone
#: by is folded in by the next rehydration instead.
_BUCKET_MEMORY = 20_000


@dataclass(frozen=True)
class _PendingEpisode:
    """Decision-time evidence retained until its outcome lands."""

    control_point: str
    bucket: str
    confidence: float | None
    predicted_success: bool | None
    decided_at: float
    runtime_revision: str
    head_version: int


@dataclass
class ControlPoint:
    """One decision surface the organ watches, and eventually may make.

    There is one head *per action*, not one head with the action as an input.
    That is not a stylistic choice — a single model with the action bolted on
    as a one-hot can only shift its intercept per action, so it is structurally
    incapable of learning that approving is right when priority outweighs
    pressure and deferring is right when it does not. Whatever the situation,
    such a model must rank the actions in the same order, which is another way
    of saying it can never recommend anything the base rates did not already
    recommend. Per-action heads let each action have its own opinion about
    which situations suit it, which is the entire question.

    It also makes missing evidence visible rather than silently extrapolated:
    an action nobody has ever taken simply has no fitted head, and the organ
    says so instead of inventing a score for it.
    """

    name: str
    schema: FeatureSchema
    actions: tuple[str, ...]
    #: The subset a random-exploration episode may draw from. Exploration is
    #: how the corpus gets its only causally clean evidence, but it is not a
    #: licence to take any action at random: the explorable set holds the ones
    #: whose worst case is a delay, and leaves out the ones that create
    #: obligations or consequences somebody has to unwind. Empty means the
    #: whole action set is explorable.
    explorable: tuple[str, ...] = ()
    horizon_s: float = 900.0
    heads: dict[str, PredictionHead] = field(default_factory=dict)
    moments: RunningMoments | None = None
    #: Graded episodes at the last fit, so a refit only runs on new evidence.
    evidence_at_last_fit: int = 0

    def ensure_heads(self, units: int) -> dict[str, PredictionHead]:
        width = design_width(self.schema, units)
        names = design_names(self.schema, units)
        for action in self.actions:
            if action not in self.heads:
                self.heads[action] = PredictionHead(
                    control_point=f"{self.name}/{action}",
                    # The target is the outcome, never the action taken.
                    options=("failure", "success"),
                    input_width=width,
                    input_names=names,
                )
        if self.moments is None:
            self.moments = RunningMoments(len(self.schema.names))
        return self.heads

    @property
    def explorable_actions(self) -> tuple[str, ...]:
        return self.explorable or self.actions

    @property
    def scorable(self) -> tuple[str, ...]:
        """Actions with enough lived evidence to be scored at all."""
        return tuple(a for a, h in self.heads.items() if h.ready)

    @property
    def ready(self) -> bool:
        """A choice needs at least two actions the organ can actually compare."""
        return len(self.scorable) >= 2


@dataclass
class Verdict:
    """What the organ concluded, and what it wants the caller to do about it."""

    control_point: str
    #: The action to take. Equals the incumbent's choice unless a head holds
    #: authority, or a probe reservation put the challenger in the seat.
    choice: str
    decider: str
    stage: AuthorityStage
    reservation: Reservation
    episode_id: str | None
    novelty: float
    #: The world model's prediction error for this moment, when it is running.
    #: Complements novelty: unfamiliar *state* versus unexpected *outcome*.
    surprise: float | None = None
    #: P(success) per candidate action, when a head exists. The shadow record.
    scores: dict[str, float] = field(default_factory=dict)
    track: TrackRecord | None = None
    advice: str | None = None
    attribution: list[tuple[str, float]] = field(default_factory=list)

    @property
    def overridden(self) -> bool:
        """True when the organ, not the rules, chose."""
        return self.decider.startswith("ontogeny")

    def as_dict(self) -> dict[str, Any]:
        return {
            "control_point": self.control_point,
            "choice": self.choice,
            "decider": self.decider,
            "stage": str(self.stage),
            "reservation": {"decider": str(self.reservation.decider), "reason": self.reservation.reason},
            "episode_id": self.episode_id,
            "novelty": round(self.novelty, 4),
            "surprise": round(self.surprise, 4) if self.surprise is not None else None,
            "scores": {k: round(v, 4) for k, v in self.scores.items()},
            "track": self.track.as_dict() if self.track else None,
            "advice": self.advice,
            "attribution": [(n, round(v, 4)) for n, v in self.attribution[:5]],
        }


class OntogenyCore(AuthorityObservationMixin):
    """The whole organ. One instance per process, alive for the life of the runtime."""

    def __init__(
        self,
        *,
        spine: ExperienceSpine | None = None,
        authority: AuthorityLedger | None = None,
        reservation: ExplorationReservation | None = None,
        resolvers: ResolverRegistry | None = None,
        units: int = DEFAULT_UNITS,
        seed: int = DEFAULT_SEED,
        autostart: bool = True,
    ) -> None:
        self._spine = spine or get_experience_spine()
        self._authority = authority or get_authority_ledger()
        self._reservation = reservation or get_reservation()
        self._resolvers = resolvers or get_resolvers()
        self._candidate_calibration = CalibrationMonitor(provenance=CANDIDATE_VALIDATION)
        self._operational_calibration = CalibrationMonitor(provenance=OPERATIONAL_SHADOW)
        # Compatibility name for internal callers. Authority deliberately sees
        # only candidate validation; deployed drift is a separate evidence
        # plane and must not be contaminated by post-fit rescoring.
        self._calibration = self._candidate_calibration
        self._authority.attach_calibration(self._candidate_calibration)
        self._runtime_revision = _runtime_revision()
        self._units = int(units)
        self._seed = int(seed)
        self._lock = checked_lock("ontogeny.core", rank=LockRank.LEAF, reentrant=True)
        self._control_points: dict[str, ControlPoint] = {}
        self._state: OntogeneticState | None = None
        self._trainer = Trainer(
            self._spine,
            self._authority,
            self._candidate_calibration,
            units=self._units,
            seed=self._seed,
        )
        self._last_reading: StateReading | None = None
        self._last_train = 0.0
        self._last_checkpoint = time.time()
        self._episodes_seen = 0
        self._sweeper: OutcomeSweeper | None = None
        self._maintenance: threading.Thread | None = None
        self._stopped = threading.Event()
        self._started_at = time.time()
        self._track = TrackRecordIndex()
        #: The VRNN, or ``False`` once it has proved unavailable.
        self._world_model: Any = None
        self._episode_buckets: OrderedDict[str, _PendingEpisode] = OrderedDict()
        self._spine.on_resolve(self._note_resolution)

        self.register(
            ControlPoint(
                name=EXECUTIVE_ADMISSION.control_point,
                schema=EXECUTIVE_ADMISSION,
                actions=("approved", "deferred", "degraded", "rejected"),
                # Never explores by rejecting: a rejection records a failure
                # obligation and cancels work outright, where a deferral or a
                # constrained approval only costs time.
                explorable=("approved", "deferred", "degraded"),
                horizon_s=900.0,
            )
        )
        self._load_heads()
        self._activate_operational_cohorts()
        if autostart:
            self.start()

    # ── registration ─────────────────────────────────────────────────────

    def register(self, control_point: ControlPoint) -> ControlPoint:
        with self._lock:
            control_point.ensure_heads(self._units)
            self._control_points[control_point.name] = control_point
        return control_point

    @staticmethod
    def _head_version(cp: ControlPoint) -> int:
        """A single version number for the whole control point's model set."""
        return max((h.version for h in cp.heads.values()), default=0)

    def control_points(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(self._control_points)

    @property
    def resolvers(self) -> ResolverRegistry:
        """Where a subsystem registers how to find out what came of its decisions."""
        return self._resolvers

    @property
    def authority(self) -> AuthorityLedger:
        return self._authority

    def reservation_report(self) -> dict[str, Any]:
        return self._reservation.report()

    def _state_for(self, schema: FeatureSchema) -> OntogeneticState:
        """The reservoir. Width is fixed by the first schema that asks for it.

        All control points share one state on purpose: it is *her* state, not a
        per-subsystem scratchpad, and what happened in the executive is
        legitimately context for what happens in memory retrieval.
        """
        if self._state is None:
            # Rooted at the spine, like the heads and for the same reason: a
            # sandbox has to be total. A state grown from simulated episodes
            # and checkpointed to the live path would be picked up by the real
            # instance at next boot as though she had lived it.
            self._state = OntogeneticState(
                input_width=schema.width, units=self._units, seed=self._seed,
                path=self._spine.db_path.parent / "state.npz",
            )
            self._state.load()
        return self._state

    # ── the live seam ────────────────────────────────────────────────────

    def consider(
        self,
        control_point: str,
        features: Mapping[str, float],
        *,
        incumbent_choice: str,
        seed: str,
        stakes: float = 0.5,
        context: Mapping[str, Any] | None = None,
        provenance: Provenance = Provenance.LIVE,
    ) -> Verdict:
        """The one call the live path makes. Cheap, total, and never raises.

        Returns the action to take. Unless a head has earned authority — or a
        reservation deliberately handed this episode to the challenger so its
        counterfactual gets observed — that action is exactly the incumbent's.
        """
        try:
            return self._consider(
                control_point, features, incumbent_choice=incumbent_choice, seed=seed,
                stakes=stakes, context=context, provenance=provenance,
            )
        except (RuntimeError, ValueError, TypeError, KeyError, AttributeError, IndexError) as exc:
            record_degradation(
                "ontogeny", exc, severity="warning",
                action=f"ontogeny fell back to the incumbent for {control_point}",
            )
            return Verdict(
                control_point=control_point, choice=incumbent_choice, decider="incumbent",
                stage=AuthorityStage.OBSERVE,
                reservation=Reservation(Decider.INCUMBENT, "organ_error"),
                episode_id=None, novelty=0.5,
            )

    def _consider(
        self,
        control_point: str,
        features: Mapping[str, float],
        *,
        incumbent_choice: str,
        seed: str,
        stakes: float,
        context: Mapping[str, Any] | None,
        provenance: Provenance,
    ) -> Verdict:
        with self._lock:
            cp = self._control_points.get(control_point)
        if cp is None:
            return Verdict(
                control_point=control_point, choice=incumbent_choice, decider="incumbent",
                stage=AuthorityStage.OBSERVE,
                reservation=Reservation(Decider.INCUMBENT, "unregistered"),
                episode_id=None, novelty=0.5,
            )

        heads = cp.ensure_heads(self._units)
        state = self._state_for(cp.schema)
        moments = cp.moments or RunningMoments(len(cp.schema.names))
        cp.moments = moments

        vector = cp.schema.vector(features)
        base = design_row(vector, moments, update=True)
        reading = state.step(base)
        self._last_reading = reading
        self._episodes_seen += 1

        # Score every action she has evidence about: P(success | situation, action).
        row = np.concatenate([base, reading.hidden])
        scores: dict[str, float] = {}
        attribution: list[tuple[str, float]] = []
        for action in cp.scorable:
            prediction = heads[action].predict(row)
            scores[action] = float(prediction.probabilities.get("success", 0.0))
        if scores:
            best = max(scores, key=lambda a: scores[a])
            attributed = heads[best].predict(row, attribute=True)
            if attributed.attribution:
                attribution = attributed.attribution.top()
        else:
            best = incumbent_choice

        stage = self._authority.stage(control_point)
        reservation = self._reservation.decide(
            control_point,
            seed=seed,
            stakes=float(stakes),
            has_authority=stage.decides,
            challenger_ready=cp.ready,
        )

        if reservation.decider is Decider.RANDOM:
            # Uniform over the *explorable* actions, deterministically in the
            # seed so a replay reproduces it. This slice exists to break the
            # confound between what the situation was and what was done, and it
            # buys that evidence only with actions whose worst case is a delay.
            pool = cp.explorable_actions
            choice = pool[_stable_index(f"{control_point}|{seed}", len(pool))]
            decider = "explore:random"
        elif reservation.decider is Decider.CHALLENGER and cp.ready:
            choice = best
            decider = f"ontogeny:{control_point}@{self._head_version(cp)}"
        else:
            choice = incumbent_choice
            decider = "incumbent"

        advice = None
        if stage.advises and cp.ready and scores:
            advice = self._phrase_advice(scores, incumbent_choice, choice)

        track = self.track_record(control_point, choice)

        head_version = self._head_version(cp)
        episode_context = dict(context or {})
        episode_context.setdefault("runtime_revision", self._runtime_revision)
        episode_context.setdefault("ontogeny_head_version", head_version)
        episode = Episode(
            control_point=control_point,
            features=dict(features),
            decision=choice,
            options=cp.actions,
            decider=decider,
            exploration=reservation.reserved,
            shadow=dict(scores) if scores else None,
            shadow_version=head_version if cp.ready else None,
            stakes=float(stakes),
            horizon_s=cp.horizon_s,
            provenance=provenance,
            feature_schema=cp.schema.schema_id,
            context=episode_context,
        )
        episode_id = self._spine.record(episode)
        if episode_id == episode.episode_id:
            self._remember_episode(episode)
        surprise = self._feed_world_model(base, cp.actions, choice)

        return Verdict(
            control_point=control_point,
            choice=choice,
            decider=decider,
            stage=stage,
            reservation=reservation,
            episode_id=episode_id,
            novelty=reading.novelty,
            surprise=surprise,
            scores=scores,
            track=track,
            advice=advice,
            attribution=attribution,
        )

    def _feed_world_model(
        self, base: np.ndarray, actions: Sequence[str], choice: str
    ) -> float | None:
        """Give the VRNN a real observation stream, and take its surprise back.

        The world model has existed for months without ever being fed: its
        checkpoint directory was empty and its hidden state was reset to zeros
        every boot. Every episode the organ handles is exactly the shape it
        wanted — a standardised observation and the action actually taken — so
        the two organs are wired together rather than each half-working alone.

        The signal that comes back is different in kind from the reservoir's
        novelty. Novelty asks "have I been in a state like this?"; surprise
        asks "did what just happened match what I expected?" A familiar state
        with a surprising outcome is the interesting case, and only having both
        can tell them apart.
        """
        if self._world_model is False:
            return None
        try:
            if self._world_model is None:
                from core.world_model.learned_world_model import get_learned_world_model

                model = get_learned_world_model()
                model.start_training()
                self._world_model = model
            encoded = np.zeros(len(actions), dtype=np.float64)
            if choice in actions:
                encoded[list(actions).index(choice)] = 1.0
            prediction = self._world_model.observe(base, encoded, learn=True)
            return float(prediction.surprise)
        except (ImportError, RuntimeError, ValueError, TypeError, AttributeError, MemoryError) as exc:
            record_degradation(
                "ontogeny", exc, severity="warning",
                action="world model not fed; ontogeny continues without surprise",
            )
            self._world_model = False
            return None

    def resolve(self, episode_id: str, outcome: Outcome) -> None:
        """Attach a real outcome. The only way a label enters the corpus."""
        if not episode_id:
            return
        self._spine.resolve(episode_id, outcome)

    def resolve_success(self, episode_id: str, success: bool, resolver: str, **detail: Any) -> None:
        self.resolve(
            episode_id,
            Outcome(
                kind=OutcomeKind.SUCCESS if success else OutcomeKind.FAILURE,
                utility=1.0 if success else 0.0,
                resolver=resolver,
                detail=detail,
            ),
        )

    # ── the day-one surfaces ─────────────────────────────────────────────

    def novelty(self) -> float:
        """How unlike her ordinary life this moment is. 0.5 until she has one."""
        reading = self._last_reading
        return reading.novelty if reading else 0.5

    def state_reading(self) -> StateReading | None:
        return self._last_reading

    def track_record(self, control_point: str, decision: str) -> TrackRecord | None:
        """What actually happened, last time she did this. Counting, not modelling.

        Reads the live index, never the database — this is called on the
        decision path, and a decision path that queries sqlite is a decision
        path that will one day block the event loop.
        """
        return self._track.get(control_point, decision)

    def _note_resolution(self, episode_id: str, outcome: Outcome) -> None:
        """Fold a landed outcome into tallies and decision-time calibration."""
        with self._lock:
            located = self._episode_buckets.pop(episode_id, None)
        if located is None:
            return
        self._track.observe(located.control_point, located.bucket, outcome.kind)
        if (
            outcome.kind.is_evidence
            and located.confidence is not None
            and located.predicted_success is not None
        ):
            actual_success = outcome.kind is OutcomeKind.SUCCESS
            self._operational_calibration.observe(
                located.control_point,
                episode_id=episode_id,
                confidence=located.confidence,
                correct=located.predicted_success == actual_success,
                decided_at=located.decided_at,
                observed_at=outcome.resolved_at,
                runtime_revision=located.runtime_revision,
                head_version=located.head_version,
                action=located.bucket,
                provenance=OPERATIONAL_SHADOW,
            )

    def _remember_episode(self, episode: Episode) -> None:
        """Keep the immutable decision-time claim until resolution."""
        probability = None
        if episode.shadow is not None:
            candidate = episode.shadow.get(episode.decision)
            if candidate is not None:
                probability = min(1.0, max(0.0, float(candidate)))
        pending = _PendingEpisode(
            control_point=episode.control_point,
            bucket=episode.decision,
            confidence=(max(probability, 1.0 - probability) if probability is not None else None),
            predicted_success=(probability >= 0.5 if probability is not None else None),
            decided_at=episode.decided_at,
            runtime_revision=str(
                (episode.context or {}).get("runtime_revision") or self._runtime_revision
            ),
            head_version=int(episode.shadow_version or 0),
        )
        with self._lock:
            self._episode_buckets[episode.episode_id] = pending
            while len(self._episode_buckets) > _BUCKET_MEMORY:
                self._episode_buckets.popitem(last=False)

    def rehydrate_track_records(self, limit: int = 6000) -> dict[str, int]:
        """Rebuild the tallies from the corpus. Slow, so it runs on maintenance."""
        rebuilt: dict[str, int] = {}
        with self._lock:
            names = list(self._control_points)
        for name in names:
            try:
                episodes = self._spine.episodes(name, limit=limit)
            except (RuntimeError, OSError, ValueError) as exc:
                record_degradation("ontogeny", exc, severity="debug",
                                   action=f"track-record rehydration skipped for {name}")
                continue
            rebuilt[name] = self._track.hydrate(name, episodes)
        return rebuilt

    def rehydrate_operational_calibration(self, limit: int = _BUCKET_MEMORY) -> dict[str, int]:
        """Rebuild operational cohorts from immutable decision-time shadows.

        The episode reader intentionally exposes a narrow projection and older
        versions omitted ``context_json`` from that projection. Read only that
        provenance column here so restart recovery remains source-bound without
        rewriting or deleting historical incidents.
        """
        rebuilt: dict[str, int] = {}
        with self._lock:
            control_points = list(self._control_points.values())
        for cp in control_points:
            try:
                episodes = self._spine.episodes(cp.name, evidence_only=True, limit=limit)
                contexts = self._episode_contexts([episode.episode_id for episode in episodes])
            except (RuntimeError, OSError, ValueError, sqlite3.Error) as exc:
                record_degradation(
                    "ontogeny", exc, severity="debug",
                    action=f"operational calibration rehydration skipped for {cp.name}",
                )
                continue
            observations: list[CalibrationObservation] = []
            for episode in episodes:
                if episode.outcome is None or not episode.outcome.kind.is_evidence:
                    continue
                probability = (episode.shadow or {}).get(episode.decision)
                if probability is None or episode.shadow_version is None:
                    continue
                probability = min(1.0, max(0.0, float(probability)))
                context = contexts.get(episode.episode_id, episode.context or {})
                revision = str(context.get("runtime_revision") or "legacy-unbound")
                predicted_success = probability >= 0.5
                observations.append(CalibrationObservation(
                    episode_id=episode.episode_id,
                    control_point=episode.control_point,
                    confidence=max(probability, 1.0 - probability),
                    correct=predicted_success == (episode.outcome.kind is OutcomeKind.SUCCESS),
                    decided_at=episode.decided_at,
                    observed_at=episode.outcome.resolved_at,
                    runtime_revision=revision,
                    head_version=int(episode.shadow_version),
                    action=episode.decision,
                    provenance=OPERATIONAL_SHADOW,
                ))
            rebuilt[cp.name] = self._operational_calibration.replace_observations(
                cp.name,
                observations,
                provenance=OPERATIONAL_SHADOW,
            )
            self._operational_calibration.activate(
                cp.name,
                runtime_revision=self._runtime_revision,
                head_version=self._head_version(cp),
                provenance=OPERATIONAL_SHADOW,
            )
        return rebuilt

    def _episode_contexts(self, episode_ids: Sequence[str]) -> dict[str, dict[str, Any]]:
        """Read persisted provenance for restart rehydration, in bounded chunks."""
        if not episode_ids:
            return {}
        contexts: dict[str, dict[str, Any]] = {}
        uri = f"file:{self._spine.db_path}?mode=ro"
        with connecting(sqlite3.connect(uri, uri=True, timeout=2.0)) as conn:
            for offset in range(0, len(episode_ids), 500):
                chunk = list(episode_ids[offset:offset + 500])
                placeholders = ",".join("?" for _ in chunk)
                for episode_id, raw in conn.execute(
                    f"SELECT episode_id, context_json FROM episodes WHERE episode_id IN ({placeholders})",
                    chunk,
                ):
                    try:
                        value = json.loads(raw or "{}")
                    except (TypeError, ValueError):
                        value = {}
                    contexts[str(episode_id)] = value if isinstance(value, dict) else {}
        return contexts

    def grounded_confidence(self, control_point: str, decision: str) -> str | None:
        """Her track record as a sentence, or nothing when she has no standing to speak.

        This is the surface that reaches Bryan. It is deliberately willing to
        report a bad record: a self-assessment that only speaks when flattering
        is not a self-assessment.
        """
        record = self.track_record(control_point, decision)
        if record is None or record.graded == 0:
            return None
        return record.phrase()

    # ── learning ─────────────────────────────────────────────────────────

    def train(
        self,
        control_point: str | None = None,
        *,
        yield_to_foreground: bool = False,
    ) -> dict[str, TrainingResult]:
        """Refit heads from the corpus. Off the cognitive path, always."""
        with self._lock:
            targets = (
                [self._control_points[control_point]]
                if control_point and control_point in self._control_points
                else list(self._control_points.values())
            )
        results: dict[str, TrainingResult] = {}
        stop_callback = None
        cooperate = None
        if yield_to_foreground:
            def _foreground_active() -> bool:
                return foreground_activity_reason() == "foreground_chat_active"

            stop_callback = _foreground_active
            # The maintenance lane is a native thread, but the recurrent replay
            # and per-example AdaGrad loops execute thousands of small Python /
            # NumPy operations. Explicit GIL handoffs keep those bounded loops
            # from starving the asyncio control plane while preserving the
            # exact fit and its chronological corpus contract.
            def _cooperate() -> None:
                time.sleep(TRAIN_COOPERATIVE_YIELD_S)

            cooperate = _cooperate
        for cp in targets:
            if yield_to_foreground and foreground_activity_reason():
                break
            heads = cp.ensure_heads(self._units)
            try:
                result = self._trainer.train(
                    cp.name,
                    cp.schema,
                    heads,
                    cp.actions,
                    should_stop=stop_callback,
                    cooperate=cooperate,
                )
            except (RuntimeError, ValueError, TypeError, MemoryError, np.linalg.LinAlgError) as exc:
                record_degradation(
                    "ontogeny", exc, severity="warning",
                    action=f"training pass failed for {cp.name}; the running head is unchanged",
                )
                continue
            results[cp.name] = result
            if result.reason == "foreground_preempted":
                logger.info(
                    "ontogeny: training yielded to an active foreground chat turn"
                )
                break
            if result.fitted:
                cp.evidence_at_last_fit = (
                    result.samples + result.temperature_samples + result.holdout_samples
                )
                self._save_head(cp)
                self._operational_calibration.activate(
                    cp.name,
                    runtime_revision=self._runtime_revision,
                    head_version=self._head_version(cp),
                    provenance=OPERATIONAL_SHADOW,
                )
        if not any(result.reason == "foreground_preempted" for result in results.values()):
            self._last_train = time.time()
        return results

    def _new_evidence(self) -> int:
        total = 0
        with self._lock:
            control_points = list(self._control_points.values())
        for cp in control_points:
            stats = self._spine.stats(cp.name)
            if stats.get("available"):
                total += max(0, int(stats.get("evidence_rows", 0)) - cp.evidence_at_last_fit)
        return total

    # ── lifecycle ────────────────────────────────────────────────────────

    def start(self) -> None:
        """Bring the organ up: sweeper, maintenance loop, resolver defaults."""
        if self._sweeper is None:
            self._sweeper = OutcomeSweeper(self._resolvers, self._spine, interval_s=60.0)
            self._sweeper.start()
        if self._maintenance is None:
            self._maintenance = threading.Thread(
                target=self._maintenance_loop, name="ontogeny-maintenance", daemon=True
            )
            self._maintenance.start()

    def _maintenance_loop(self) -> None:
        cycles = 0
        while not self._stopped.wait(60.0):
            now = time.time()
            cycles += 1
            try:
                if foreground_activity_reason():
                    continue
                if self._state is not None and now - self._last_checkpoint >= CHECKPOINT_INTERVAL_S:
                    self._state.save()
                    self._last_checkpoint = now
                if cycles == 1:
                    self.rehydrate_operational_calibration()
                self._enforce_authority_observation()
                telemetry.sample(self.report())
                if cycles % 10 == 0:
                    # Slow, authoritative rebuild of the tallies from the
                    # ledger, so incremental counting cannot drift unnoticed.
                    self.rehydrate_track_records()
                if now - self._last_train >= TRAIN_INTERVAL_S and self._new_evidence() >= TRAIN_MIN_NEW_EVIDENCE:
                    self.train(yield_to_foreground=True)
            except (RuntimeError, OSError, ValueError, TypeError) as exc:
                record_degradation(
                    "ontogeny", exc, severity="warning",
                    action="ontogeny maintenance cycle failed; retrying next minute",
                )

    def stop(self) -> None:
        self._stopped.set()
        if self._sweeper is not None:
            self._sweeper.stop()
        if self._state is not None:
            self._state.save()
        self._spine.close()

    # ── head persistence ─────────────────────────────────────────────────

    def _heads_dir(self) -> Path:
        """Head checkpoints live beside the corpus that produced them.

        Deriving this from the spine rather than from config is the whole
        point. The provenance gate keeps test episodes out of the live corpus,
        but a head is *derived* from a corpus, and a head fitted on simulated
        episodes and written to the live directory is the same contamination
        one level up — it would be loaded by the real instance at next boot and
        would start scoring real decisions from things that never happened.
        Tying every artefact to the store's own root makes a sandbox total
        instead of partial.
        """
        return self._spine.db_path.parent / "heads"

    def _save_head(self, cp: ControlPoint) -> None:
        if not cp.heads:
            return
        try:
            from core.governance_context import local_internal_governed_scope
            from core.runtime.file_write_gateway import get_file_write_gateway

            payload: dict[str, Any] = {
                "schema_id": cp.schema.schema_id,
                "actions": list(cp.actions),
                "moments": cp.moments.state_dict() if cp.moments else {},
                "heads": {action: head.state_dict() for action, head in cp.heads.items()},
            }
            gateway = get_file_write_gateway()
            target = self._heads_dir() / f"{cp.name.replace('.', '_')}.json"
            with local_internal_governed_scope(
                "ontogeny_head", domain="state_mutation", receipt_prefix="ontogeny-head"
            ):
                gateway.ensure_directory(target.parent, source="ontogeny_head")
                gateway.write_text(
                    target, json.dumps(payload, ensure_ascii=False), source="ontogeny_head"
                )
        except (ImportError, OSError, RuntimeError, ValueError, TypeError) as exc:
            record_degradation(
                "ontogeny", exc, severity="warning",
                action=f"head checkpoint for {cp.name} not written",
            )

    def _load_heads(self) -> None:
        directory = self._heads_dir()
        if not directory.exists():
            return
        with self._lock:
            control_points = list(self._control_points.values())
        for cp in control_points:
            target = directory / f"{cp.name.replace('.', '_')}.json"
            if not target.exists():
                continue
            try:
                payload = json.loads(target.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                record_degradation("ontogeny", exc, severity="warning",
                                   action=f"head checkpoint for {cp.name} unreadable")
                continue
            if payload.get("schema_id") != cp.schema.schema_id:
                logger.info(
                    "ontogeny: heads for %s were fitted on schema %s, now %s — discarding them",
                    cp.name, payload.get("schema_id"), cp.schema.schema_id,
                )
                continue
            heads = cp.ensure_heads(self._units)
            restored = 0
            for action, state in (payload.get("heads") or {}).items():
                head = heads.get(action)
                if head is not None and head.load_state(state):
                    restored += 1
            if restored and cp.moments is not None:
                cp.moments.load_state(payload.get("moments", {}))
            logger.info("ontogeny: restored %d/%d heads for %s", restored, len(heads), cp.name)

    def _activate_operational_cohorts(self) -> None:
        with self._lock:
            control_points = list(self._control_points.values())
        for cp in control_points:
            self._operational_calibration.activate(
                cp.name,
                runtime_revision=self._runtime_revision,
                head_version=self._head_version(cp),
                provenance=OPERATIONAL_SHADOW,
            )

    # ── reporting ────────────────────────────────────────────────────────

    def health_report(self) -> dict[str, Any]:
        """Bounded in-memory state for high-frequency health polling.

        ``report()`` is a diagnostic surface: its corpus section performs
        SQLite aggregates for every registered control point.  Boot health
        needs neither those aggregates nor the training and history payloads,
        and polling them made the health worker compete with the live mind.
        Keep this projection at the owner so health cannot accidentally grow
        another database dependency by slicing the full report downstream.
        """
        with self._lock:
            control_points = tuple(self._control_points)
            state = self._state
            episodes_seen = self._episodes_seen
            last_reading = self._last_reading
            world_model = self._world_model

        authority = self._authority.report()
        calibration = self._candidate_calibration.all_reports()
        resolution = self._resolvers.report()
        return {
            "schema": "aura.ontogeny.health.v1",
            "episodes_seen": episodes_seen,
            "novelty": round(last_reading.novelty if last_reading else 0.5, 4),
            "state": state.report() if state else None,
            "stages": {
                name: str(self._authority.stage(name)) for name in control_points
            },
            "frozen": authority.get("frozen"),
            "observation_rate": resolution.get("observation_rate"),
            "calibration": {
                name: {
                    "ece": report.get("ece"),
                    "overconfidence": report.get("overconfidence"),
                }
                for name, report in calibration.items()
            },
            "world_model": (
                world_model.get_status()
                if world_model not in (None, False) else None
            ),
        }

    def report(self) -> dict[str, Any]:
        """Everything the organ knows about itself, for health and for Bryan."""
        with self._lock:
            control_points = list(self._control_points.values())
        heads = {}
        for cp in control_points:
            stats = self._spine.stats(cp.name)
            heads[cp.name] = {
                "heads": {a: h.report() for a, h in cp.heads.items()},
                "scorable_actions": list(cp.scorable),
                "ready": cp.ready,
                "stage": str(self._authority.stage(cp.name)),
                "corpus": stats,
                "last_training": (
                    self._trainer.last_result[cp.name].as_dict()
                    if cp.name in self._trainer.last_result else None
                ),
            }
        return {
            "schema": "aura.ontogeny.report.v1",
            "uptime_s": round(time.time() - self._started_at, 1),
            "episodes_seen": self._episodes_seen,
            "state": self._state.report() if self._state else None,
            "novelty": round(self.novelty(), 4),
            "world_model": (
                self._world_model.get_status()
                if self._world_model not in (None, False) else None
            ),
            "control_points": heads,
            "authority": self._authority.report(),
            "authority_observation": self.authority_observation_report(),
            "reservation": self._reservation.report(),
            "resolution": self._resolvers.report(),
            "sweeper": self._sweeper.report() if self._sweeper else None,
            "calibration": self._candidate_calibration.all_reports(),
            "candidate_validation": self._candidate_calibration.all_reports(),
            "operational_calibration": self._operational_calibration.all_reports(),
            "operational_calibration_history": self._operational_calibration.cohort_reports(),
            "track_records": self._track.report(),
        }
    def summary(self) -> str:
        """One line, for a log or a status pane."""
        state = self._state
        if state is None:
            return "ontogeny: not yet stepped"
        stages = {
            cp: str(self._authority.stage(cp)) for cp in self.control_points()
        }
        return (
            f"ontogeny: {state.steps} episodes lived, era {state.era}, "
            f"novelty {self.novelty():.2f}, stages {stages}"
        )

    @staticmethod
    def _phrase_advice(scores: Mapping[str, float], incumbent: str, chosen: str) -> str | None:
        """What the head would say, when it is allowed to say anything."""
        if not scores:
            return None
        best = max(scores, key=lambda k: scores[k])
        if best == incumbent:
            return f"agrees with {incumbent} (p_success {scores[best]:.2f})"
        return (
            f"would prefer {best} (p_success {scores[best]:.2f}) over "
            f"{incumbent} ({scores.get(incumbent, 0.0):.2f})"
            + ("" if chosen == best else "; incumbent stands")
        )


def _runtime_revision() -> str:
    """Stable source identity supplied by the signed launcher, without git I/O."""
    for name in (
        "AURA_LAUNCH_EXPECTED_COMMIT",
        "AURA_RUNTIME_SOURCE_COMMIT",
        "AURA_SOURCE_COMMIT",
    ):
        value = str(os.environ.get(name) or "").strip().lower()
        if value:
            return value
    return "runtime-unbound"


def _stable_index(seed: str, modulus: int) -> int:
    """Deterministic uniform draw. A replay of the corpus reproduces it exactly."""
    import hashlib

    digest = hashlib.sha256(seed.encode("utf-8")).digest()
    return int.from_bytes(digest[4:8], "big") % max(1, modulus)


_core: OntogenyCore | None = None
_core_lock = threading.Lock()


def get_ontogeny() -> OntogenyCore:
    global _core
    if _core is None:
        with _core_lock:
            if _core is None:
                _core = OntogenyCore()
    return _core


def ontogeny_report() -> dict[str, Any]:
    """Module-level surface for the health report and the live mind snapshot."""
    try:
        return get_ontogeny().report()
    except (RuntimeError, OSError, ValueError, TypeError, AttributeError) as exc:
        record_degradation("ontogeny", exc, severity="warning",
                           action="ontogeny report unavailable")
        return {"available": False, "error": type(exc).__name__}


def ontogeny_health_report() -> dict[str, Any]:
    """Bounded module-level projection for runtime health polling."""
    try:
        return get_ontogeny().health_report()
    except (RuntimeError, OSError, ValueError, TypeError, AttributeError) as exc:
        record_degradation(
            "ontogeny",
            exc,
            severity="warning",
            action="ontogeny health report unavailable",
        )
        return {"available": False, "error": type(exc).__name__}


def reset_ontogeny_for_test(core: OntogenyCore | None = None) -> None:
    global _core
    with _core_lock:
        if _core is not None and core is not _core:
            _core.stop()
        _core = core


__all__ = [
    "CHECKPOINT_INTERVAL_S",
    "TRAIN_INTERVAL_S",
    "ControlPoint",
    "OntogenyCore",
    "Verdict",
    "get_ontogeny",
    "ontogeny_health_report",
    "ontogeny_report",
    "reset_ontogeny_for_test",
]
