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

This is what makes Aura single-origin rather than multi-origin.

The rule:
    impulse -> synthesis -> arbiter -> will -> execution -> memory

Not:
    pathway A acts
    pathway B acts
    pathway C acts
"""
from __future__ import annotations

import hashlib
import logging
import time
from collections import deque
from dataclasses import dataclass, field
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

    def __init__(self) -> None:
        self._impulse_queue: List[Impulse] = []
        self._recent_fingerprints: Dict[str, float] = {}
        self._synthesis_history: Deque[SynthesisResult] = deque(maxlen=self._MAX_HISTORY)
        self._started = False

    async def start(self) -> None:
        if self._started:
            return
        ServiceContainer.register_instance("initiative_synthesizer", self, required=False)
        self._started = True
        logger.info("InitiativeSynthesizer ONLINE -- single impulse funnel active")

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
                    # Determine which drive is low
                    status = await drive_engine.get_status()
                    lowest_drive = min(status.items(), key=lambda kv: kv[1].get("percent", 100))[0]
                    self.submit(
                        content=imperative, source="drive_engine",
                        urgency=0.6, drive=lowest_drive,
                    )
        except Exception as e:
            logger.debug("Synth: DriveEngine gather failed: %s", e)

        # 2. GoalEngine -- resumed/active goals
        try:
            goal_engine = ServiceContainer.get("goal_engine", default=None)
            if goal_engine:
                active = goal_engine.get_active_goals(limit=3, include_external=False)
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
        except Exception as e:
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
            logger.debug("Synth: pending_initiatives gather failed: %s", e)

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
        return {
            "pending_impulses": len(self._impulse_queue),
            "synthesis_count": len(self._synthesis_history),
            "recent_approved": sum(1 for r in self._synthesis_history if r.approved),
            "recent_rejected": sum(1 for r in self._synthesis_history if not r.approved and r.winner is not None),
            "last_result": self._synthesis_history[-1].rationale if self._synthesis_history else "",
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
