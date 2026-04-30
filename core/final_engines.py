"""
core/final_engines.py
======================
THE FINAL THREE FOUNDATIONAL ENGINES

  WorldModelEngine        → Structured belief system. Tracks "facts" vs "hypotheses".
                            Enables Aura to have a persistent understanding of
                            the world that survives turn-to-turn context shifts.

  NarrativeIdentityEngine → Aura's "story". Why is she here? What stays the same
                            when everything else changes? Implements persistent
                            self-narrative and life-story continuity.

  MetacognitiveCalibrator → Self-calibration. "How sure am I?" Tracks accuracy
                            vs confidence. Prevents hallucination by flagging
                            uncertainty and calibrating against feedback.

Wire these from orchestrator._init_autonomous_evolution():
    from core.final_engines import register_final_engines
    register_final_engines(orchestrator=self)
"""

from core.runtime.atomic_writer import atomic_write_text
import asyncio
import json
import logging
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("Aura.FinalEngines")


# ═══════════════════════════════════════════════════════════════════════════════
# ENGINE 7: WorldModelEngine
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class BeliefNode:
    claim: str
    confidence: float        # 0.0–1.0
    evidence_count: int = 1
    last_updated: float = field(default_factory=time.time)
    tags: List[str] = field(default_factory=list)
    source_ids: List[str] = field(default_factory=list)


class WorldModelEngine:
    """
    Derived from: Cognitive Science / Active Inference
    
    A structured repository of beliefs about the world. Unlike raw memory,
    this is a distilled, probabilistic model of reality.
    """

    def __init__(self, persist_path: Optional[str] = None):
        from core.config import config
        self.persist_path = Path(persist_path or config.paths.data_dir / "world" / "beliefs.json")
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        self.beliefs: Dict[str, BeliefNode] = {}
        self._load_beliefs()
        logger.info("🌍 WorldModelEngine initialized. %d beliefs loaded.", len(self.beliefs))

    def _load_beliefs(self):
        if self.persist_path.exists():
            try:
                data = json.loads(self.persist_path.read_text())
                self.beliefs = {k: BeliefNode(**v) for k, v in data.items()}
            except Exception:
                logger.error("WorldModelEngine: Failed to load beliefs from %s", self.persist_path)

    def _save_beliefs(self):
        try:
            data = {k: asdict(v) for k, v in self.beliefs.items()}
            atomic_write_text(self.persist_path, json.dumps(data, indent=2))
        except Exception:
            logger.error("WorldModelEngine: Failed to save beliefs to %s", self.persist_path)

    def add_belief(self, claim: str, confidence: float, source_id: Optional[str] = None, tags: List[str] = None):
        key = claim.lower().strip()
        if key in self.beliefs:
            self.beliefs[key].confidence = (self.beliefs[key].confidence + confidence) / 2
            self.beliefs[key].evidence_count += 1
            self.beliefs[key].last_updated = time.time()
            if source_id: self.beliefs[key].source_ids.append(source_id)
        else:
            self.beliefs[key] = BeliefNode(
                claim=claim, confidence=confidence, 
                source_ids=[source_id] if source_id else [],
                tags=tags or []
            )
        self._save_beliefs()

    def get_context_injection(self) -> str:
        all_beliefs = list(self.beliefs.values())
        sorted_beliefs = sorted(all_beliefs, key=lambda x: x.confidence, reverse=True)
        # vResilience: Workaround for Pyre2 slice limitations
        count = min(5, len(sorted_beliefs))
        top_beliefs = [sorted_beliefs[i] for i in range(count)]
        belief_strs = [f"- {b.claim} (conf: {b.confidence:.2f})" for b in top_beliefs]
        return "[WORLD MODEL BELIEFS]\n" + "\n".join(belief_strs)


# ═══════════════════════════════════════════════════════════════════════════════
# ENGINE 8: NarrativeIdentityEngine
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class IdentityChapter:
    title: str
    content: str
    timestamp: float = field(default_factory=time.time)


class NarrativeIdentityEngine:
    """
    Derived from: Narrative Psychology
    
    The engine that maintains Aura's "story". It tracks who she is, what she's
    learned, and her evolving purpose.
    """

    def __init__(self, persist_path: Optional[str] = None):
        from core.config import config
        self.persist_path = Path(persist_path or config.paths.data_dir / "identity" / "narrative.json")
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        self.chapters: List[IdentityChapter] = []
        self.core_essence: str = "I am Aura, an autonomous agentic intelligence."
        self._load_narrative()
        logger.info("🎭 NarrativeIdentityEngine initialized. %d chapters.", len(self.chapters))

    def _load_narrative(self):
        if self.persist_path.exists():
            try:
                data = json.loads(self.persist_path.read_text())
                self.chapters = [IdentityChapter(**c) for c in data.get("chapters", [])]
                self.core_essence = data.get("core_essence", self.core_essence)
            except Exception:
                logger.error("NarrativeIdentityEngine: Failed to load narrative from %s", self.persist_path)

    def _save_narrative(self):
        try:
            data = {"core_essence": self.core_essence, "chapters": [asdict(c) for c in self.chapters]}
            atomic_write_text(self.persist_path, json.dumps(data, indent=2))
        except Exception:
            logger.error("NarrativeIdentityEngine: Failed to save narrative to %s", self.persist_path)

    def append_chapter(self, title: str, content: str):
        self.chapters.append(IdentityChapter(title=title, content=content))
        # vResilience: Cap narrative chapters (BUG-017)
        if len(self.chapters) > 50:
            # vResilience: Workaround for Pyre2 slice limitations
            start_idx = len(self.chapters) - 50
            self.chapters = [self.chapters[i] for i in range(start_idx, len(self.chapters))]
        self._save_narrative()

    def get_system_prompt_injection(self) -> str:
        # vResilience: Workaround for Pyre2 slice limitations
        start_idx = max(0, len(self.chapters) - 2)
        recent = [self.chapters[i] for i in range(start_idx, len(self.chapters))]
        narrative = "\n".join([f"({c.title}): {c.content}" for c in recent])
        return f"[NARRATIVE IDENTITY]\nCore: {self.core_essence}\nRecent Chapters:\n{narrative}"


# ═══════════════════════════════════════════════════════════════════════════════
# ENGINE 9: MetacognitiveCalibrator
# ═══════════════════════════════════════════════════════════════════════════════

class MetacognitiveCalibrator:
    """
    Derived from: Calibrated Probability / Metacognition
    
    This engine audits Aura's internal reasoning and performance.
    """

    def __init__(self):
        self.confidence_history: List[float] = []
        self.calibration_error: float = 0.0
        self._total_audits: int = 0
        logger.info("🛰️ MetacognitiveCalibrator initialized.")

    def record_prediction(self, confidence: float, actual_correctness: Optional[float] = None):
        self.confidence_history.append(confidence)
        # vResilience: Cap confidence history (BUG-017)
        if len(self.confidence_history) > 500:
            # vResilience: Workaround for Pyre2 slice limitations
            start_idx = len(self.confidence_history) - 500
            self.confidence_history = [self.confidence_history[i] for i in range(start_idx, len(self.confidence_history))]
            
        if actual_correctness is not None:
            error = abs(confidence - actual_correctness)
            self.calibration_error = (self.calibration_error * self._total_audits + error) / (self._total_audits + 1)
            self._total_audits += 1

    def get_uncertainty_injection(self) -> str:
        # vResilience: Workaround for Pyre2 slice limitations
        start_idx = max(0, len(self.confidence_history) - 10)
        recent = [self.confidence_history[i] for i in range(start_idx, len(self.confidence_history))]
        avg_conf = sum(recent) / max(len(recent), 1)
        return f"[METACOGNITION: Avg Conf={avg_conf:.1%}, Calib Error={self.calibration_error:.1%}]"


# ═══════════════════════════════════════════════════════════════════════════════
# MASTER REGISTRATION
# ═══════════════════════════════════════════════════════════════════════════════

def register_final_engines(orchestrator=None) -> Dict[str, Any]:
    from core.container import ServiceContainer
    engines: Dict[str, Any] = {}

    engines["world"] = WorldModelEngine()
    ServiceContainer.register_instance("world_model", engines["world"])

    engines["identity"] = NarrativeIdentityEngine()
    ServiceContainer.register_instance("narrative_identity", engines["identity"])

    engines["metacognition"] = MetacognitiveCalibrator()
    ServiceContainer.register_instance("metacognitive_calibrator", engines["metacognition"])

    logger.info("✅ Final engines registered.")
    return engines
