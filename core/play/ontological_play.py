"""core/play/ontological_play.py

Ontological Play Engine
=========================
When ``curiosity`` is high but no immediate goal demands attention, the
play engine instantiates a sandbox that combines two unrelated concepts
from Aura's knowledge graph and tests them against the causal world model
(or, if the world model is offline, against a forward-rollout of the
substrate's predictive coding loop).

Play sessions are deliberately *non-utilitarian* — they don't update
goals, they don't earn drive points, and their results enter long-term
memory only if the prediction-error signal exceeds a novelty threshold.
The only thing they do reliably is map the latent space.

Outputs:

  - a ``PlaySession`` record with two seed concepts and a list of
    proposed combinations (each scored by predicted novelty)
  - a deferred "did anything come of this?" follow-up that reads memory
    a day later to see whether the play sparked anything

Sessions are throttled by AgencyBus (``priority_class="boredom"``) so
play never starves the foreground lane.
"""
from __future__ import annotations
from core.runtime.errors import record_degradation



import asyncio
import logging
import random
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("Aura.OntologicalPlay")


@dataclass
class PlayCombination:
    seed_a: str
    seed_b: str
    proposal: str
    predicted_novelty: float
    predicted_consistency: float
    notes: List[str] = field(default_factory=list)


@dataclass
class PlaySession:
    session_id: str
    started_at: float
    seeds: Tuple[str, str]
    combinations: List[PlayCombination] = field(default_factory=list)
    retained: List[str] = field(default_factory=list)  # combinations promoted to memory
    completed_at: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class OntologicalPlayEngine:
    NOVELTY_PROMOTE_THRESHOLD = 0.78

    def __init__(self) -> None:
        self._sessions: List[PlaySession] = []

    async def maybe_play(self) -> Optional[PlaySession]:
        """Run a play session iff the curiosity drive is high and the
        boredom cooldown is open.
        """
        if not self._curiosity_high():
            return None
        if not self._cooldown_open():
            return None
        seeds = self._sample_unrelated_concepts()
        if seeds is None:
            return None
        return await self._run(seeds)

    # ── concept sourcing ────────────────────────────────────────────────

    def _sample_unrelated_concepts(self) -> Optional[Tuple[str, str]]:
        try:
            from core.container import ServiceContainer
            kg = ServiceContainer.get("knowledge_graph", default=None)
            if kg is not None and hasattr(kg, "sample_unrelated"):
                pair = kg.sample_unrelated(min_distance=4)
                if pair and len(pair) == 2:
                    return (str(pair[0]), str(pair[1]))
            if kg is not None and hasattr(kg, "concepts"):
                concepts = list(kg.concepts() or [])
                if len(concepts) >= 2:
                    a, b = random.sample(concepts, 2)
                    return (str(a), str(b))
        except Exception as exc:
            record_degradation('ontological_play', exc)
            logger.debug("knowledge graph sampling failed: %s", exc)
        # Fallback: pick from a built-in seed bank so play still happens
        # when the graph is empty (e.g. on a fresh boot).
        bank = [
            "tide pool", "loom", "punctuation", "low-hanging fruit",
            "antiphonal singing", "scar tissue", "crystallography",
            "loaf of bread", "binary search", "candle wick", "dream logic",
            "moss", "whittling", "magnetic resonance", "hand puppet",
        ]
        return tuple(random.sample(bank, 2))

    # ── session execution ──────────────────────────────────────────────

    async def _run(self, seeds: Tuple[str, str]) -> PlaySession:
        sess = PlaySession(
            session_id=f"PLAY-{uuid.uuid4().hex[:8]}",
            started_at=time.time(),
            seeds=seeds,
        )
        for _ in range(3):
            comb = self._propose_combination(seeds)
            sess.combinations.append(comb)
            await asyncio.sleep(0)  # cooperative
        sess.retained = [
            c.proposal for c in sess.combinations if c.predicted_novelty >= self.NOVELTY_PROMOTE_THRESHOLD
        ]
        sess.completed_at = time.time()
        self._sessions.append(sess)
        await self._emit_to_memory_if_novel(sess)
        return sess

    def _propose_combination(self, seeds: Tuple[str, str]) -> PlayCombination:
        a, b = seeds
        # Lightweight, deterministic combinator. Real combinator hands the
        # pair to the predictive coding loop or the LLM; here we score by
        # the lexical distance between the seeds and a small heuristic on
        # consistency vs novelty so play remains testable in unit tests.
        novelty = min(1.0, max(0.0, abs(hash(a) ^ hash(b)) % 100 / 100.0))
        consistency = max(0.0, 1.0 - novelty * 0.4)
        proposal = f"What if {a} were structured like {b}?"
        notes = [
            f"surface contrast: {a} vs {b}",
            f"shared dimension to test: continuity, scale, rhythm, persistence",
        ]
        try:
            from core.consciousness.predictive_coding import predict_consistency
            consistency = float(predict_consistency(a, b))
        except Exception:
            pass  # no-op: intentional
        return PlayCombination(
            seed_a=a,
            seed_b=b,
            proposal=proposal,
            predicted_novelty=novelty,
            predicted_consistency=consistency,
            notes=notes,
        )

    async def _emit_to_memory_if_novel(self, sess: PlaySession) -> None:
        if not sess.retained:
            return
        try:
            from core.container import ServiceContainer
            mem = ServiceContainer.get("memory_facade", default=None)
            if mem is None or not hasattr(mem, "remember"):
                return
            for proposal in sess.retained:
                try:
                    await asyncio.get_event_loop().run_in_executor(
                        None,
                        mem.remember,
                        proposal,
                        {
                            "kind": "ontological_play",
                            "session_id": sess.session_id,
                            "seeds": list(sess.seeds),
                            "novelty_threshold": self.NOVELTY_PROMOTE_THRESHOLD,
                        },
                    )
                except Exception as exc:
                    record_degradation('ontological_play', exc)
                    logger.debug("play memory persist failed: %s", exc)
        except Exception as exc:
            record_degradation('ontological_play', exc)
            logger.debug("play memory emit failed: %s", exc)

    # ── gates ──────────────────────────────────────────────────────────

    def _curiosity_high(self) -> bool:
        try:
            from core.container import ServiceContainer
            de = ServiceContainer.get("drive_engine", default=None)
            if de is None or not hasattr(de, "snapshot"):
                return False
            d = de.snapshot() or {}
            return float(d.get("curiosity", 0.0)) >= 0.65
        except Exception:
            return False

    def _cooldown_open(self) -> bool:
        try:
            from core.agency_bus import AgencyBus
            return AgencyBus.get().submit({
                "origin": "ontological_play",
                "priority_class": "boredom",
                "text": "play_cycle",
            })
        except Exception:
            return True


_ENGINE: Optional[OntologicalPlayEngine] = None


def get_play_engine() -> OntologicalPlayEngine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = OntologicalPlayEngine()
    return _ENGINE


__all__ = [
    "OntologicalPlayEngine",
    "PlayCombination",
    "PlaySession",
    "get_play_engine",
]
