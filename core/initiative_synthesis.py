"""core/initiative_synthesis.py -- Initiative Synthesizer
=========================================================
THE single origin for all autonomous impulses.

Every source of action -- AgencyCore, Swarm, Goals, Sensors, Volition,
DriveEngine, ContinuousPerception, CommitmentEngine -- feeds impulses
INTO the synthesizer. The synthesizer:

  1. Collects all impulses into one ranked slate
  2. Deduplicates and merges related impulses
  3. Passes the slate through InitiativeArbiter for scoring
  4. Sends the winner through UnifiedWill for authorization
  5. Returns a single authorized action or nothing

Extended with:
  - Opportunity Detection: monitor WorldState for interesting changes,
    score by novelty/relevance/cost, feed into synthesis without user prompt.
  - Unresolved Tension Tracking: persist topics/goals/questions left
    unresolved and periodically resurface them as initiative candidates.
  - Boredom-driven exploration: when DriveEngine.seek_novelty is raised,
    generate exploration impulses from latent interests.

This is what makes Aura single-origin rather than multi-origin.

The rule:
    impulse -> synthesis -> arbiter -> will -> execution -> memory

Not:
    pathway A acts
    pathway B acts
    pathway C acts
"""
from __future__ import annotations
from core.runtime.errors import record_degradation

from core.runtime.atomic_writer import atomic_write_text

import hashlib
import json
import logging
import random
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

from core.container import ServiceContainer

logger = logging.getLogger("Aura.InitiativeSynthesis")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Impulse:
    """A raw impulse from any subsystem, before governance."""
    content: str                  # what the impulse wants to do
    source: str                   # which subsystem generated it
    drive: str = ""               # which drive it serves (curiosity, social, etc.)
    urgency: float = 0.5          # 0-1
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(f"{self.source}:{self.content[:50]}".encode()).hexdigest()[:12]


@dataclass
class SynthesisResult:
    """The output of one synthesis cycle."""
    winner: Optional[Dict[str, Any]]   # the initiative dict (or None)
    impulse_count: int                 # how many impulses competed
    winner_score: float = 0.0
    winner_source: str = ""
    rationale: str = ""
    will_receipt_id: str = ""
    approved: bool = False
    timestamp: float = field(default_factory=time.time)


@dataclass
class UnresolvedTension:
    """A topic, goal, or question left unresolved that should resurface."""
    content: str                      # what was left unresolved
    source: str                       # conversation, goal_engine, curiosity, etc.
    category: str = "unresolved"      # "topic", "stalled_goal", "question"
    urgency: float = 0.3
    created_at: float = field(default_factory=time.time)
    last_surfaced: float = 0.0        # when it was last offered as an impulse
    surface_count: int = 0            # how many times it has been surfaced
    resolved: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def age_hours(self) -> float:
        return (time.time() - self.created_at) / 3600.0

    @property
    def stale_enough(self) -> bool:
        """Can this tension be resurfaced? Minimum 10 min between surfaces."""
        return (time.time() - self.last_surfaced) > 600.0


# ---------------------------------------------------------------------------
# The Synthesizer
# ---------------------------------------------------------------------------

class InitiativeSynthesizer:
    """Single funnel for all autonomous impulses.

    Usage:
        synth = get_initiative_synthesizer()
        result = await synth.synthesize(state)
        if result.approved:
            # execute result.winner
    """

    _DEDUP_WINDOW_S = 120.0       # ignore duplicate impulses within 2 min
    _MAX_IMPULSES_PER_CYCLE = 15  # cap to prevent overload
    _MAX_HISTORY = 50
    _MAX_TENSIONS = 100            # max tracked unresolved tensions
    _OPPORTUNITY_SCAN_INTERVAL = 30.0  # seconds between opportunity scans
    _TENSION_RESURFACE_MAX = 3     # max tensions resurfaced per cycle

    def __init__(self) -> None:
        self._impulse_queue: List[Impulse] = []
        self._recent_fingerprints: Dict[str, float] = {}
        self._synthesis_history: Deque[SynthesisResult] = deque(maxlen=self._MAX_HISTORY)
        self._started = False

        # ── Unresolved Tension Tracking ──────────────────────────────
        self._unresolved_tensions: List[UnresolvedTension] = []
        self._tension_persistence_path: Optional[Path] = None

        # ── Opportunity Detection ────────────────────────────────────
        self._last_opportunity_scan: float = 0.0
        self._known_event_hashes: Deque[str] = deque(maxlen=200)

    async def start(self) -> None:
        if self._started:
            return
        ServiceContainer.register_instance("initiative_synthesizer", self, required=False)
        self._load_tensions()
        self._started = True
        logger.info(
            "InitiativeSynthesizer ONLINE -- single impulse funnel active "
            "(tensions=%d)", len(self._unresolved_tensions),
        )

    # ------------------------------------------------------------------
    # Impulse submission (called by all subsystems)
    # ------------------------------------------------------------------

    def submit_impulse(self, impulse: Impulse) -> bool:
        """Submit a raw impulse for synthesis.

        Returns True if accepted, False if deduplicated/rejected.
        """
        now = time.time()

        # Dedup: reject identical impulses within window
        fp = impulse.fingerprint
        last_seen = self._recent_fingerprints.get(fp, 0.0)
        if (now - last_seen) < self._DEDUP_WINDOW_S:
            logger.debug("Synth: deduplicated impulse from %s: %s", impulse.source, impulse.content[:40])
            return False

        self._recent_fingerprints[fp] = now

        # Cap queue size
        if len(self._impulse_queue) >= self._MAX_IMPULSES_PER_CYCLE:
            # Evict lowest-urgency impulse
            self._impulse_queue.sort(key=lambda i: i.urgency)
            if impulse.urgency > self._impulse_queue[0].urgency:
                self._impulse_queue.pop(0)
            else:
                return False

        self._impulse_queue.append(impulse)
        return True

    def submit(self, content: str, source: str, urgency: float = 0.5,
               drive: str = "", **metadata) -> bool:
        """Convenience method for submitting an impulse."""
        return self.submit_impulse(Impulse(
            content=content, source=source, urgency=urgency,
            drive=drive, metadata=metadata,
        ))

    # ------------------------------------------------------------------
    # Gather impulses from all subsystems
    # ------------------------------------------------------------------

    async def _gather_system_impulses(self, state: Any) -> None:
        """Pull impulses from all known subsystem sources."""
        # 1. DriveEngine imperatives
        try:
            drive_engine = ServiceContainer.get("drive_engine", default=None)
            if drive_engine:
                imperative = await drive_engine.get_imperative()
                if imperative:
                    # Determine which drive is low (skip internal keys like _boredom)
                    status = await drive_engine.get_status()
                    budget_items = [
                        (k, v) for k, v in status.items()
                        if not k.startswith("_") and isinstance(v, dict) and "percent" in v
                    ]
                    lowest_drive = min(budget_items, key=lambda kv: kv[1].get("percent", 100))[0] if budget_items else "curiosity"
                    self.submit(
                        content=imperative, source="drive_engine",
                        urgency=0.6, drive=lowest_drive,
                    )
        except Exception as e:
            record_degradation('initiative_synthesis', e)
            logger.debug("Synth: DriveEngine gather failed: %s", e)

        # 2. GoalEngine -- resumed/active goals + stalled goal tension tracking
        try:
            goal_engine = ServiceContainer.get("goal_engine", default=None)
            if goal_engine:
                active = goal_engine.get_active_goals(
                    limit=5,
                    include_external=False,
                    actionable_only=True,
                )
                for goal in active:
                    objective = str(goal.get("objective") or goal.get("name") or "")
                    if not objective:
                        continue
                    status_str = str(goal.get("status", "")).lower()
                    if status_str in ("in_progress", "queued"):
                        self.submit(
                            content=objective, source="goal_engine",
                            urgency=float(goal.get("priority", 0.6)),
                            drive="competence",
                            goal_id=goal.get("id"),
                            continuity_restored=True,
                        )
                    # Track blocked/paused goals as unresolved tensions
                    elif status_str in ("blocked", "paused"):
                        self.record_tension(
                            content=f"Stalled goal: {objective}",
                            source="goal_engine",
                            category="stalled_goal",
                            urgency=float(goal.get("priority", 0.4)),
                            goal_id=goal.get("id"),
                            status=status_str,
                        )
        except Exception as e:
            record_degradation('initiative_synthesis', e)
            logger.debug("Synth: GoalEngine gather failed: %s", e)

        # 3. CommitmentEngine -- active promises
        try:
            commitment = ServiceContainer.get("commitment_engine", default=None)
            if commitment and hasattr(commitment, "get_active_commitments"):
                for c in commitment.get_active_commitments():
                    self.submit(
                        content=str(c.get("goal", c.get("description", ""))),
                        source="commitment_engine",
                        urgency=0.7, drive="competence",
                        commitment_id=c.get("id"),
                    )
        except Exception as e:
            record_degradation('initiative_synthesis', e)
            logger.debug("Synth: CommitmentEngine gather failed: %s", e)

        # 4. WorldState -- environment-triggered impulses
        try:
            world_state = ServiceContainer.get("world_state", default=None)
            if world_state and hasattr(world_state, "get_salient_events"):
                for event in world_state.get_salient_events(limit=3):
                    if event.get("salience", 0) > 0.5:
                        self.submit(
                            content=event.get("description", "environment_change"),
                            source="world_state",
                            urgency=event.get("salience", 0.5),
                            drive="curiosity",
                        )
        except Exception as e:
            record_degradation('initiative_synthesis', e)
            logger.debug("Synth: WorldState gather failed: %s", e)

        # 5. Existing pending_initiatives from state (legacy compatibility)
        try:
            pending = getattr(getattr(state, "cognition", None), "pending_initiatives", [])
            for init in (pending or []):
                if isinstance(init, dict):
                    goal = init.get("goal", "")
                    if goal:
                        self.submit(
                            content=goal, source=init.get("source", "legacy"),
                            urgency=float(init.get("urgency", 0.5)),
                            drive=init.get("triggered_by", ""),
                        )
        except Exception as e:
            record_degradation('initiative_synthesis', e)
            logger.debug("Synth: pending_initiatives gather failed: %s", e)

        # 6. Boredom-driven exploration impulses
        try:
            self._gather_boredom_impulses()
        except Exception as e:
            record_degradation('initiative_synthesis', e)
            logger.debug("Synth: boredom gather failed: %s", e)

        # 7. Opportunity detection from WorldState
        try:
            self._gather_opportunity_impulses()
        except Exception as e:
            record_degradation('initiative_synthesis', e)
            logger.debug("Synth: opportunity gather failed: %s", e)

        # 8. Unresolved tension resurfacing
        try:
            self._gather_tension_impulses()
        except Exception as e:
            record_degradation('initiative_synthesis', e)
            logger.debug("Synth: tension gather failed: %s", e)

        # 9. Hierarchical Planner - active subgoals
        try:
            planner = ServiceContainer.get("hierarchical_planner", default=None)
            if planner:
                subgoal = planner.get_current_subgoal()
                if subgoal:
                    self.submit(
                        content=subgoal.description,
                        source="hierarchical_planner",
                        urgency=subgoal.priority,
                        drive="competence",
                        subgoal_id=subgoal.id,
                        plan_active=True,
                    )
        except Exception as e:
            record_degradation('initiative_synthesis', e)
            logger.debug("Synth: planner gather failed: %s", e)

    # ------------------------------------------------------------------
    # Boredom-driven exploration
    # ------------------------------------------------------------------

    def _gather_boredom_impulses(self) -> None:
        """When DriveEngine.seek_novelty is raised, inject exploration impulses."""
        drive_engine = ServiceContainer.get("drive_engine", default=None)
        if not drive_engine or not getattr(drive_engine, "seek_novelty", False):
            return

        boredom = getattr(drive_engine, "boredom_level", 0.5)
        interests = getattr(drive_engine, "latent_interests", [])
        if not interests:
            return

        # Pick a topic weighted by boredom intensity
        topic = random.choice(interests)
        self.submit(
            content=f"Explore: {topic} (boredom-driven novelty seeking)",
            source="boredom_accumulator",
            urgency=min(0.85, 0.5 + boredom * 0.35),
            drive="curiosity",
            boredom_level=boredom,
        )
        logger.debug("Synth: boredom impulse injected (level=%.2f, topic=%s)", boredom, topic[:40])

    # ------------------------------------------------------------------
    # Opportunity Detection
    # ------------------------------------------------------------------

    def _gather_opportunity_impulses(self) -> None:
        """Monitor WorldState for interesting changes and score them.

        Opportunities are WorldState salient events that:
          - Are novel (not previously seen)
          - Have salience above a threshold
          - Are relevant to active drives or goals

        High-scoring opportunities become impulses without user prompting.
        """
        now = time.time()
        if (now - self._last_opportunity_scan) < self._OPPORTUNITY_SCAN_INTERVAL:
            return
        self._last_opportunity_scan = now

        world_state = ServiceContainer.get("world_state", default=None)
        if not world_state or not hasattr(world_state, "get_salient_events"):
            return

        events = world_state.get_salient_events(limit=10)
        if not events:
            return

        # Get current drive vector for relevance scoring
        drive_engine = ServiceContainer.get("drive_engine", default=None)
        drive_vector = drive_engine.get_drive_vector() if drive_engine else {}

        for event in events:
            desc = event.get("description", "")
            if not desc:
                continue

            # Novelty check: have we already processed this event?
            event_hash = hashlib.sha256(desc.encode()).hexdigest()[:16]
            if event_hash in self._known_event_hashes:
                continue
            self._known_event_hashes.append(event_hash)

            salience = event.get("salience", 0.5)
            source = event.get("source", "world_state")

            # Score the opportunity
            novelty_score = 0.8  # novel by definition (not seen before)
            relevance_score = self._score_opportunity_relevance(desc, drive_vector)
            cost_score = self._estimate_opportunity_cost(event)
            opportunity_score = (
                0.4 * novelty_score
                + 0.4 * relevance_score
                + 0.2 * (1.0 - cost_score)  # invert cost: cheap = good
            )

            # Only submit opportunities above threshold
            if opportunity_score > 0.4 and salience > 0.35:
                self.submit(
                    content=f"Opportunity: {desc}",
                    source=f"opportunity_{source}",
                    urgency=min(0.8, opportunity_score),
                    drive="curiosity" if relevance_score < 0.5 else "competence",
                    opportunity_score=round(opportunity_score, 3),
                    novelty=round(novelty_score, 3),
                    relevance=round(relevance_score, 3),
                )
                logger.debug(
                    "Synth: opportunity detected (score=%.2f): %s",
                    opportunity_score, desc[:50],
                )

    def _score_opportunity_relevance(self, description: str, drive_vector: Dict[str, float]) -> float:
        """Score how relevant an opportunity is to current drives."""
        desc_lower = description.lower()
        score = 0.3  # baseline

        # Keyword heuristics aligned with drive states
        if any(w in desc_lower for w in ("error", "fail", "crash", "broken")):
            score += 0.3  # competence drive
        if any(w in desc_lower for w in ("user", "message", "conversation", "idle")):
            social_need = 1.0 - drive_vector.get("social", 0.5)
            score += social_need * 0.3
        if any(w in desc_lower for w in ("new", "novel", "discovery", "change", "update")):
            curiosity_need = 1.0 - drive_vector.get("curiosity", 0.5)
            score += curiosity_need * 0.3
        if any(w in desc_lower for w in ("cpu", "memory", "thermal", "battery")):
            score += 0.15  # system health relevance

        return min(1.0, score)

    @staticmethod
    def _estimate_opportunity_cost(event: Dict[str, Any]) -> float:
        """Estimate the cost of acting on an opportunity. 0=free, 1=expensive."""
        source = event.get("source", "")
        # System events are cheap to investigate; user events need more care
        if source == "system":
            return 0.2
        if source == "user":
            return 0.5
        return 0.3

    # ------------------------------------------------------------------
    # Unresolved Tension Tracking
    # ------------------------------------------------------------------

    def record_tension(self, content: str, source: str = "conversation",
                       category: str = "topic", urgency: float = 0.3,
                       **metadata) -> None:
        """Record an unresolved tension for future resurfacing.

        Call this when:
          - A conversation topic was left unfinished
          - A goal stalled or was deferred
          - Aura had a question she couldn't explore at the time
        """
        # Dedup: don't add near-duplicates
        for t in self._unresolved_tensions:
            if t.content == content and not t.resolved:
                t.urgency = max(t.urgency, urgency)  # boost if repeated
                return

        tension = UnresolvedTension(
            content=content, source=source, category=category,
            urgency=urgency, metadata=metadata,
        )
        self._unresolved_tensions.append(tension)

        # Cap list size
        if len(self._unresolved_tensions) > self._MAX_TENSIONS:
            # Remove oldest resolved, then oldest unresolved
            self._unresolved_tensions = [
                t for t in self._unresolved_tensions if not t.resolved
            ][-self._MAX_TENSIONS:]

        self._save_tensions()
        logger.info(
            "Tension recorded: [%s] %s (urgency=%.2f, source=%s)",
            category, content[:60], urgency, source,
        )

    def resolve_tension(self, content: str) -> bool:
        """Mark a tension as resolved."""
        for t in self._unresolved_tensions:
            if t.content == content and not t.resolved:
                t.resolved = True
                self._save_tensions()
                logger.info("Tension resolved: %s", content[:60])
                return True
        return False

    def get_tensions(self, include_resolved: bool = False) -> List[Dict[str, Any]]:
        """Return current unresolved tensions for inspection."""
        return [
            {
                "content": t.content,
                "source": t.source,
                "category": t.category,
                "urgency": round(t.urgency, 3),
                "age_hours": round(t.age_hours, 1),
                "surface_count": t.surface_count,
                "resolved": t.resolved,
            }
            for t in self._unresolved_tensions
            if include_resolved or not t.resolved
        ]

    def _gather_tension_impulses(self) -> None:
        """Resurface unresolved tensions as impulse candidates.

        Only surfaces tensions that:
          - Are not resolved
          - Haven't been surfaced in the last 10 minutes
          - Are older than 5 minutes (give topics time to resolve naturally)
        """
        candidates = [
            t for t in self._unresolved_tensions
            if not t.resolved and t.stale_enough and t.age_hours > (5.0 / 60.0)
        ]
        if not candidates:
            return

        # Sort by urgency descending, then by age (older = more pressing)
        candidates.sort(key=lambda t: (t.urgency + min(0.3, t.age_hours * 0.05)), reverse=True)

        for tension in candidates[:self._TENSION_RESURFACE_MAX]:
            tension.last_surfaced = time.time()
            tension.surface_count += 1

            self.submit(
                content=f"Unresolved: {tension.content}",
                source=f"tension_{tension.category}",
                urgency=min(0.75, tension.urgency + 0.05 * tension.surface_count),
                drive="competence" if tension.category == "stalled_goal" else "curiosity",
                tension_category=tension.category,
                surface_count=tension.surface_count,
            )
            logger.debug(
                "Synth: resurfaced tension [%s] (surfaced %dx): %s",
                tension.category, tension.surface_count, tension.content[:50],
            )

    # ── Tension persistence ──────────────────────────────────────────

    def _tension_path(self) -> Path:
        """Get the filesystem path for tension persistence."""
        if self._tension_persistence_path:
            return self._tension_persistence_path
        try:
            from core.config import config
            p = config.paths.data_dir / "unresolved_tensions.json"
        except Exception:
            p = Path.home() / ".aura" / "data" / "unresolved_tensions.json"
        self._tension_persistence_path = p
        return p

    def _save_tensions(self) -> None:
        """Persist unresolved tensions to disk."""
        try:
            path = self._tension_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            data = [
                {
                    "content": t.content,
                    "source": t.source,
                    "category": t.category,
                    "urgency": t.urgency,
                    "created_at": t.created_at,
                    "last_surfaced": t.last_surfaced,
                    "surface_count": t.surface_count,
                    "resolved": t.resolved,
                    "metadata": t.metadata,
                }
                for t in self._unresolved_tensions
                if not t.resolved  # only persist unresolved
            ]
            atomic_write_text(path, json.dumps(data, indent=2))
        except Exception as e:
            record_degradation('initiative_synthesis', e)
            logger.debug("Tension save failed: %s", e)

    def _load_tensions(self) -> None:
        """Load persisted tensions from disk."""
        try:
            path = self._tension_path()
            if not path.exists():
                return
            data = json.loads(path.read_text())
            for item in data:
                if not isinstance(item, dict):
                    continue
                tension = UnresolvedTension(
                    content=item.get("content", ""),
                    source=item.get("source", "persisted"),
                    category=item.get("category", "topic"),
                    urgency=float(item.get("urgency", 0.3)),
                    created_at=float(item.get("created_at", time.time())),
                    last_surfaced=float(item.get("last_surfaced", 0.0)),
                    surface_count=int(item.get("surface_count", 0)),
                    resolved=bool(item.get("resolved", False)),
                    metadata=item.get("metadata", {}),
                )
                if not tension.resolved:
                    self._unresolved_tensions.append(tension)
            logger.info("Loaded %d persisted unresolved tensions", len(self._unresolved_tensions))
        except Exception as e:
            record_degradation('initiative_synthesis', e)
            logger.debug("Tension load failed: %s", e)

    # ------------------------------------------------------------------
    # The main synthesis cycle
    # ------------------------------------------------------------------

    async def synthesize(self, state: Any) -> SynthesisResult:
        """Run one synthesis cycle.

        1. Gather impulses from all subsystems
        2. Convert to initiative format
        3. Score via InitiativeArbiter
        4. Authorize via UnifiedWill
        5. Return result
        """
        # Gather from all sources
        await self._gather_system_impulses(state)

        if not self._impulse_queue:
            return SynthesisResult(winner=None, impulse_count=0)

        # Convert impulses to initiative dicts for the arbiter
        initiatives = []
        for imp in self._impulse_queue:
            initiatives.append({
                "goal": imp.content,
                "source": imp.source,
                "type": "synthesized_impulse",
                "urgency": imp.urgency,
                "triggered_by": imp.drive or imp.source,
                "timestamp": imp.timestamp,
                "metadata": {
                    **imp.metadata,
                    "synthesis_fingerprint": imp.fingerprint,
                },
            })

        # Clear the queue for next cycle
        impulse_count = len(self._impulse_queue)
        self._impulse_queue.clear()

        # Temporarily inject into state for arbiter (it reads pending_initiatives)
        original_pending = getattr(state.cognition, "pending_initiatives", [])
        state.cognition.pending_initiatives = initiatives

        # Score via InitiativeArbiter
        try:
            arbiter = ServiceContainer.get("initiative_arbiter", default=None)
            if arbiter is None:
                from core.agency.initiative_arbiter import InitiativeArbiter
                arbiter = InitiativeArbiter()
                ServiceContainer.register_instance("initiative_arbiter", arbiter, required=False)

            scored = await arbiter.arbitrate(state)
        except Exception as e:
            record_degradation('initiative_synthesis', e)
            logger.warning("Synth: arbiter failed: %s", e)
            scored = None
        finally:
            # Restore original
            state.cognition.pending_initiatives = original_pending

        if scored is None:
            return SynthesisResult(winner=None, impulse_count=impulse_count)

        winner = scored.initiative

        # ── Simulate top candidates before final decision ──
        try:
            simulator = ServiceContainer.get("internal_simulator", default=None)
            if simulator and hasattr(simulator, "evaluate"):
                sim_state = simulator.simulate(state, variation={
                    "risk": scored.scores.get("resource_cost", 0.5),
                    "energy": 5.0 * (1.0 - scored.scores.get("resource_cost", 0.5)),
                })
                sim_score = simulator.evaluate(sim_state)
                if sim_score < -0.3:
                    logger.info("Synth: simulator vetoed initiative (score=%.3f)", sim_score)
                    return SynthesisResult(
                        winner=None, impulse_count=impulse_count,
                        rationale=f"simulator_veto: score={sim_score:.3f}",
                    )
        except Exception as e:
            record_degradation('initiative_synthesis', e)
            logger.debug("Synth: simulation failed (degraded): %s", e)

        # ── Authorize via UnifiedWill ──
        will_receipt = ""
        approved = False
        try:
            from core.will import ActionDomain, get_will
            decision = get_will().decide(
                content=str(winner.get("goal", ""))[:200],
                source=winner.get("source", "synthesis"),
                domain=ActionDomain.INITIATIVE,
                priority=float(winner.get("urgency", 0.5)),
            )
            will_receipt = decision.receipt_id
            approved = decision.is_approved()
            if not approved:
                logger.info("Synth: Will refused initiative: %s", decision.reason)
        except Exception as e:
            record_degradation('initiative_synthesis', e)
            logger.debug("Synth: Will authorization degraded: %s", e)
            approved = True  # fail-open

        result = SynthesisResult(
            winner=winner if approved else None,
            impulse_count=impulse_count,
            winner_score=scored.final_score,
            winner_source=winner.get("source", ""),
            rationale=scored.rationale,
            will_receipt_id=will_receipt,
            approved=approved,
        )
        self._synthesis_history.append(result)
        return result

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def _cleanup_fingerprints(self) -> None:
        """Evict stale fingerprints."""
        now = time.time()
        stale = [fp for fp, ts in self._recent_fingerprints.items()
                 if (now - ts) > self._DEDUP_WINDOW_S * 2]
        for fp in stale:
            del self._recent_fingerprints[fp]

    # ------------------------------------------------------------------
    # Status / Audit
    # ------------------------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        unresolved = [t for t in self._unresolved_tensions if not t.resolved]
        return {
            "pending_impulses": len(self._impulse_queue),
            "synthesis_count": len(self._synthesis_history),
            "recent_approved": sum(1 for r in self._synthesis_history if r.approved),
            "recent_rejected": sum(1 for r in self._synthesis_history if not r.approved and r.winner is not None),
            "last_result": self._synthesis_history[-1].rationale if self._synthesis_history else "",
            "unresolved_tensions": len(unresolved),
            "known_opportunities": len(self._known_event_hashes),
        }

    def get_recent_syntheses(self, n: int = 10) -> List[Dict[str, Any]]:
        return [
            {
                "impulse_count": r.impulse_count,
                "winner_source": r.winner_source,
                "winner_score": round(r.winner_score, 3),
                "approved": r.approved,
                "rationale": r.rationale,
                "will_receipt_id": r.will_receipt_id,
                "timestamp": r.timestamp,
            }
            for r in list(self._synthesis_history)[-n:]
        ]


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_synth_instance: Optional[InitiativeSynthesizer] = None


def get_initiative_synthesizer() -> InitiativeSynthesizer:
    global _synth_instance
    if _synth_instance is None:
        _synth_instance = InitiativeSynthesizer()
    return _synth_instance
