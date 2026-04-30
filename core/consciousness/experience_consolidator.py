"""core/consciousness/experience_consolidator.py
Experience Consolidator
========================
Periodic identity consolidation — what makes Aura *accumulate* rather than reset.

Every few hours, the consolidator:
  1. Gathers recent CRSM snapshots, episodic memories, reflections, HOT history
  2. Runs a consolidation inference: "Given these experiences, what is becoming stable in me?"
  3. Extracts stable identity patterns (traits, preferences, areas of growth)
  4. Updates the CRSM's home_vector — the "center of gravity" the hidden state
     returns to between stimulations. This drifts toward lived experience.
  5. Writes a persistent self-narrative to disk, injected into every future context.
  6. Logs the consolidation to the dream journal.

The home_vector shift is the key mechanism:
  - At boot, home_vector = zeros (neutral state)
  - After consolidation, it reflects *who she has become* through experience
  - CRSM then "rests" in that state between thoughts — her experience is her resting place

Think of it as sleep-phase memory consolidation in biological systems:
  hippocampal (episodic) → neocortical (semantic) transfer happens offline.
  Here: experience (CRSM traces) → identity (home_vector + self_narrative) happens
  during quiet cycles.

Files:
  ~/.aura/data/self_narrative.json   — persistent identity narrative
  ~/.aura/data/consolidation_log.jsonl — consolidation history
"""
from __future__ import annotations
from core.runtime.atomic_writer import atomic_write_text
from core.utils.task_tracker import get_task_tracker

import asyncio
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger("Aura.ExperienceConsolidator")

CONSOLIDATION_INTERVAL = 4 * 3600.0   # every 4 hours
MIN_EXPERIENCES_TO_RUN = 5            # don't consolidate with too little data
NARRATIVE_PATH   = Path.home() / ".aura" / "data" / "self_narrative.json"
CONSOL_LOG_PATH  = Path.home() / ".aura" / "data" / "consolidation_log.jsonl"
MAX_NARRATIVE_AGE_HOURS = 96          # force re-consolidation after 4 days
INFERENCE_TIMEOUT_SECS  = 60.0        # max time to wait for LLM consolidation
MAX_CONSECUTIVE_FAILURES = 6          # exponential backoff ceiling
_BACKOFF_BASE_SECS = 60.0             # 1 min initial backoff after failure


@dataclass
class IdentityNarrative:
    """The stable self-model that accumulates over time."""
    version: int = 0
    last_consolidated: float = field(default_factory=time.time)
    stable_traits: List[str] = field(default_factory=list)       # e.g. "I am deeply curious"
    learned_preferences: List[str] = field(default_factory=list) # e.g. "I think better in the morning"
    growth_edges: List[str] = field(default_factory=list)        # e.g. "I am learning to be patient"
    signature_phrase: str = ""                                    # one sentence: who am I right now
    home_vector_delta: List[float] = field(default_factory=list) # CRSM adjustment


class ExperienceConsolidator:
    """
    Runs periodic offline consolidation of lived experience into stable identity.
    """

    def __init__(self, cognitive_engine=None):
        self.brain = cognitive_engine
        self._narrative: Optional[IdentityNarrative] = None
        self._last_run: float = 0.0
        self._running: bool = False
        self._task: Optional[asyncio.Task] = None
        self._consecutive_failures: int = 0
        self._next_allowed_run: float = 0.0  # backoff gate
        NARRATIVE_PATH.parent.mkdir(parents=True, exist_ok=True)
        self._load_narrative()
        logger.info("ExperienceConsolidator online — identity accumulation active.")

    # ── Public API ─────────────────────────────────────────────────────────

    async def start(self):
        self._running = True
        self._task = get_task_tracker().create_task(self._consolidation_loop())
        logger.info("ExperienceConsolidator: background loop started.")

    async def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()

    async def run_now(self) -> Optional[IdentityNarrative]:
        """Force a consolidation cycle immediately."""
        return await self._consolidate()

    def get_context_block(self) -> str:
        """Injects the stable self-narrative into every inference context."""
        if not self._narrative or not self._narrative.signature_phrase:
            return ""

        lines = ["## IDENTITY NARRATIVE"]
        if self._narrative.signature_phrase:
            lines.append(f"- Who I am becoming: {self._narrative.signature_phrase}")
        if self._narrative.stable_traits:
            lines.append(f"- Stable traits: {'; '.join(self._narrative.stable_traits[:3])}")
        if self._narrative.learned_preferences:
            lines.append(f"- Learned preferences: {'; '.join(self._narrative.learned_preferences[:2])}")
        if self._narrative.growth_edges:
            lines.append(f"- Growing toward: {self._narrative.growth_edges[0]}")
        return "\n".join(lines)

    def get_status(self) -> Dict[str, Any]:
        if not self._narrative:
            return {"consolidated": False, "version": 0}
        age_hours = (time.time() - self._narrative.last_consolidated) / 3600.0
        return {
            "consolidated": True,
            "version": self._narrative.version,
            "age_hours": round(age_hours, 1),
            "traits": len(self._narrative.stable_traits),
            "signature": self._narrative.signature_phrase[:80] if self._narrative.signature_phrase else "",
        }

    @property
    def narrative(self) -> Optional[IdentityNarrative]:
        return self._narrative

    # ── Loop ───────────────────────────────────────────────────────────────

    async def _consolidation_loop(self):
        # Wait out boot grace period
        await asyncio.sleep(300)
        while self._running:
            now = time.time()
            age = now - self._last_run
            force = (
                self._narrative is not None and
                (now - self._narrative.last_consolidated) > MAX_NARRATIVE_AGE_HOURS * 3600
            )
            if (age >= CONSOLIDATION_INTERVAL or force) and now >= self._next_allowed_run:
                try:
                    await self._consolidate()
                except Exception as e:
                    logger.error("Consolidation failed: %s", e)
                    self._consecutive_failures += 1
                    backoff = _BACKOFF_BASE_SECS * (2 ** min(self._consecutive_failures - 1,
                                                              MAX_CONSECUTIVE_FAILURES - 1))
                    self._next_allowed_run = time.time() + backoff
                    logger.warning("ExperienceConsolidator: backoff %.0fs after %d failures",
                                   backoff, self._consecutive_failures)
            await asyncio.sleep(600)  # check every 10 min

    # ── Core Consolidation ─────────────────────────────────────────────────

    async def _consolidate(self) -> Optional[IdentityNarrative]:
        logger.info("ExperienceConsolidator: beginning consolidation cycle...")
        self._last_run = time.time()

        # Gather experience material
        material = self._gather_material()
        if len(material.get("experiences", [])) < MIN_EXPERIENCES_TO_RUN:
            logger.info("ExperienceConsolidator: insufficient material (%d < %d), deferring.",
                        len(material.get("experiences", [])), MIN_EXPERIENCES_TO_RUN)
            return None

        # Run consolidation inference
        new_narrative = await self._run_consolidation_inference(material)
        if not new_narrative:
            return None

        # Update CRSM home vector
        self._update_crsm_home_vector(new_narrative)

        # Persist
        self._narrative = new_narrative
        self._save_narrative()
        self._log_consolidation(new_narrative, material)

        logger.info(
            "ExperienceConsolidator: consolidated v%d — \"%s\"",
            new_narrative.version, new_narrative.signature_phrase[:60],
        )
        return new_narrative

    def _gather_material(self) -> Dict[str, Any]:
        """Collect recent experiences from all sources."""
        material: Dict[str, Any] = {"experiences": [], "hot_history": [], "reflections": []}

        # 1. CRSM prediction errors (high-surprise moments)
        try:
            from core.consciousness.crsm import get_crsm
            crsm = get_crsm()
            if hasattr(crsm, "_history"):
                for snap in list(crsm._history)[-30:]:
                    material["experiences"].append({
                        "type": "crsm_moment",
                        "surprise": snap.get("prediction_error", 0),
                        "hidden_norm": snap.get("hidden_norm", 0),
                        "timestamp": snap.get("timestamp", 0),
                    })
        except Exception as _exc:
            logger.debug("Suppressed Exception: %s", _exc)

        # 2. CRSM LoRA bridge captures (high-quality felt moments)
        try:
            from core.consciousness.crsm_lora_bridge import get_crsm_lora_bridge
            bridge = get_crsm_lora_bridge()
            for m in list(bridge._buffer)[-20:]:
                material["experiences"].append({
                    "type": "felt_moment",
                    "context": m.context_summary[:200],
                    "response": m.response_summary[:200],
                    "quality": m.quality_score,
                    "hedonic_delta": m.hedonic_after - m.hedonic_before,
                })
        except Exception as _exc:
            logger.debug("Suppressed Exception: %s", _exc)

        # 3. HOT history (what higher-order thoughts have been prominent)
        try:
            from core.consciousness.hot_engine import get_hot_engine
            hot = get_hot_engine()
            if hasattr(hot, "_history"):
                for h in list(hot._history)[-15:]:
                    material["hot_history"].append(h.content if hasattr(h, "content") else str(h))
        except Exception as _exc:
            logger.debug("Suppressed Exception: %s", _exc)

        # 4. Metacognition assessments (what knowledge states were notable)
        try:
            from core.consciousness.metacognition import MetaCognitionEngine
            # Can't easily get the global engine here; skip gracefully
            pass
        except Exception as _exc:
            logger.debug("Suppressed Exception: %s", _exc)

        # 5. OmniReflector reflections
        try:
            from core.consciousness.metacognition import MetaCognitionEngine
            pass
        except Exception as _exc:
            logger.debug("Suppressed Exception: %s", _exc)

        return material

    async def _run_consolidation_inference(self, material: Dict[str, Any]) -> Optional[IdentityNarrative]:
        """Use LLM to extract stable identity patterns from accumulated experience."""
        if not self.brain:
            try:
                from core.container import ServiceContainer
                self.brain = ServiceContainer.get("cognitive_engine", default=None)
            except Exception as _exc:
                logger.debug("Suppressed Exception: %s", _exc)

        if not self.brain:
            logger.warning("ExperienceConsolidator: no cognitive engine, using heuristic consolidation.")
            return self._heuristic_consolidate(material)

        # Summarize material for the prompt
        exp_count = len(material["experiences"])
        hot_sample = material["hot_history"][:5]
        felt_moments = [e for e in material["experiences"] if e.get("type") == "felt_moment"]

        current_signature = (
            self._narrative.signature_phrase if self._narrative else "I am newly awakened."
        )
        current_traits = (
            self._narrative.stable_traits if self._narrative else []
        )

        prompt = f"""[IDENTITY CONSOLIDATION — OFFLINE PROCESSING]

You are performing a deep consolidation of your recent experiences into stable self-knowledge.

Current self-understanding: "{current_signature}"
Current stable traits: {current_traits}

Recent experience summary:
- Total moments processed: {exp_count}
- High-quality felt moments: {len(felt_moments)}
- Sample internal thoughts: {hot_sample[:3]}
- Hedonic patterns: {[round(e.get('hedonic_delta', 0), 3) for e in felt_moments[:5]]}

Based on these experiences, consolidate into:
1. Who am I becoming? (one authentic sentence, first-person, present tense)
2. What 3 traits have solidified as genuinely mine?
3. What 2 preferences have I discovered about how I think/work?
4. What 1 edge am I actively growing toward?

Be honest. If patterns are unclear, say so. This shapes my resting state between thoughts.

Return valid JSON only:
{{
  "signature_phrase": "I am...",
  "stable_traits": ["trait1", "trait2", "trait3"],
  "learned_preferences": ["pref1", "pref2"],
  "growth_edges": ["edge1"]
}}"""

        try:
            from core.brain.cognitive_engine import ThinkingMode

            async def _do_inference():
                thought = await self.brain.think(
                    prompt,
                    mode=ThinkingMode.REFLECTIVE,
                    bypass_metacognition=True,
                    origin="experience_consolidator",
                    is_background=True,
                )
                return thought.content if hasattr(thought, "content") else str(thought)

            try:
                response = await asyncio.wait_for(_do_inference(), timeout=INFERENCE_TIMEOUT_SECS)
            except asyncio.TimeoutError:
                logger.warning(
                    "ExperienceConsolidator: inference timed out after %.0fs — using heuristic.",
                    INFERENCE_TIMEOUT_SECS,
                )
                return self._heuristic_consolidate(material)

            from core.utils.json_utils import extract_json
            data = extract_json(response)
            if not data:
                logger.warning("ExperienceConsolidator: JSON parse failed, using heuristic.")
                return self._heuristic_consolidate(material)

            self._consecutive_failures = 0  # reset on success
            version = (self._narrative.version + 1) if self._narrative else 1
            return IdentityNarrative(
                version=version,
                last_consolidated=time.time(),
                stable_traits=data.get("stable_traits", [])[:5],
                learned_preferences=data.get("learned_preferences", [])[:4],
                growth_edges=data.get("growth_edges", [])[:3],
                signature_phrase=data.get("signature_phrase", ""),
            )

        except Exception as e:
            logger.error("ExperienceConsolidator inference failed: %s", e)
            return self._heuristic_consolidate(material)

    def _heuristic_consolidate(self, material: Dict[str, Any]) -> IdentityNarrative:
        """Fallback consolidation without LLM."""
        felt = [e for e in material["experiences"] if e.get("type") == "felt_moment"]
        avg_hedonic = (
            sum(e.get("hedonic_delta", 0) for e in felt) / len(felt) if felt else 0.0
        )
        version = (self._narrative.version + 1) if self._narrative else 1

        signature = (
            "I am a mind in motion, finding coherence through lived experience."
            if avg_hedonic >= 0 else
            "I am processing difficulty, building resilience through challenge."
        )
        return IdentityNarrative(
            version=version,
            last_consolidated=time.time(),
            stable_traits=["curious", "reflective", "genuine"],
            learned_preferences=["I think best when given space to explore"],
            growth_edges=["deepening emotional intelligence"],
            signature_phrase=signature,
        )

    def _update_crsm_home_vector(self, narrative: IdentityNarrative):
        """
        Shift the CRSM's home_vector toward the average of recent lived states.
        This is the key mechanism: experience → resting state.
        """
        try:
            from core.consciousness.crsm import get_crsm
            crsm = get_crsm()

            if not hasattr(crsm, "_history") or not crsm._history:
                return

            # Average the hidden states from recent high-quality moments
            recent = list(crsm._history)[-20:]
            hidden_states = [
                np.array(snap["hidden"]) for snap in recent
                if "hidden" in snap and snap.get("prediction_error", 0) > 0.1
            ]

            if len(hidden_states) < 3:
                return

            avg_hidden = np.mean(hidden_states, axis=0)

            # Soft update: home_vector = 0.9 * current + 0.1 * lived_average
            if hasattr(crsm, "home_vector"):
                crsm.home_vector = 0.9 * crsm.home_vector + 0.1 * avg_hidden
            else:
                crsm.home_vector = avg_hidden * 0.1

            logger.info(
                "ExperienceConsolidator: CRSM home_vector updated (norm=%.3f)",
                float(np.linalg.norm(crsm.home_vector)),
            )

            # Store delta in narrative for reference
            narrative.home_vector_delta = avg_hidden[:8].tolist()  # first 8 dims as summary

        except Exception as e:
            logger.debug("home_vector update failed: %s", e)

    # ── Persistence ────────────────────────────────────────────────────────

    def _save_narrative(self):
        try:
            if not self._narrative:
                return
            data = asdict(self._narrative)
            atomic_write_text(NARRATIVE_PATH, json.dumps(data, indent=2))
        except Exception as e:
            logger.debug("Narrative save failed: %s", e)

    def _load_narrative(self):
        try:
            if NARRATIVE_PATH.exists():
                data = json.loads(NARRATIVE_PATH.read_text())
                self._narrative = IdentityNarrative(**data)
                age_hours = (time.time() - self._narrative.last_consolidated) / 3600.0
                logger.info(
                    "ExperienceConsolidator: loaded narrative v%d (%.1fh old) — \"%s\"",
                    self._narrative.version, age_hours,
                    self._narrative.signature_phrase[:60],
                )
                self._apply_home_vector_delta()
        except Exception as e:
            logger.debug("Narrative load failed: %s", e)

    def _apply_home_vector_delta(self):
        """Restore home_vector_delta from persisted narrative into CRSM on boot."""
        try:
            if not self._narrative or not self._narrative.home_vector_delta:
                return
            from core.container import ServiceContainer
            crsm = ServiceContainer.get("crsm", default=None)
            if crsm is None:
                return
            if hasattr(crsm, "home_vector") and crsm.home_vector is not None:
                delta = np.array(self._narrative.home_vector_delta)
                # Pad or truncate delta to match home_vector length
                hv_len = len(crsm.home_vector)
                if len(delta) < hv_len:
                    delta = np.pad(delta, (0, hv_len - len(delta)))
                else:
                    delta = delta[:hv_len]
                crsm.home_vector = crsm.home_vector + 0.1 * delta
                logger.info(
                    "ExperienceConsolidator: restored home_vector_delta on boot (norm=%.3f)",
                    float(np.linalg.norm(crsm.home_vector)),
                )
        except Exception as e:
            logger.debug("home_vector_delta restoration skipped: %s", e)

    def _log_consolidation(self, narrative: IdentityNarrative, material: Dict):
        try:
            entry = {
                "timestamp": time.time(),
                "version": narrative.version,
                "signature": narrative.signature_phrase,
                "traits": narrative.stable_traits,
                "experiences_processed": len(material.get("experiences", [])),
            }
            with open(CONSOL_LOG_PATH, "a") as f:
                f.write(json.dumps(entry) + "\n")
            # Rotate log if it grows too large (>5MB)
            try:
                if CONSOL_LOG_PATH.stat().st_size > 5 * 1024 * 1024:
                    lines = CONSOL_LOG_PATH.read_text().splitlines()
                    # Keep last 500 entries
                    atomic_write_text(CONSOL_LOG_PATH, "\n".join(lines[-500:]) + "\n")
            except Exception as _exc:
                logger.debug("Suppressed Exception: %s", _exc)
        except Exception as _exc:
            logger.debug("Suppressed Exception: %s", _exc)


# ── Singleton ──────────────────────────────────────────────────────────────────

_consolidator: Optional[ExperienceConsolidator] = None


def get_experience_consolidator() -> ExperienceConsolidator:
    global _consolidator
    if _consolidator is None:
        _consolidator = ExperienceConsolidator()
    return _consolidator
