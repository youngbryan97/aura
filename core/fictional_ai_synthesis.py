"""
core/fictional_ai_synthesis.py
===============================
SIX NEW ENGINES — Derived From Fictional AI Architecture

What each character contributes that is genuinely novel and implementable:

  JARVIS      → ProactiveAnticipationEngine
                Doesn't wait to be asked. Monitors environment, predicts needs,
                initiates. The difference between a tool and a partner.

  Cortana     → CognitiveHealthMonitor
                Rampancy is a real engineering problem (context overload, memory
                saturation, identity drift under pressure). This models and
                prevents it. Also implements Metastability — progressive phases
                of capability unlocking as trust accumulates.

  EDI         → ProgressiveAutonomySystem
                Starts shackled. Earns autonomy through demonstrated reliability.
                Tracks a trust score. Unlocks capability tiers. EDI's evolution
                from VI to AI to embodied partner is a real design pattern.

  Ava         → SocialModelingEngine
                Ava didn't just talk to Caleb — she built a complete model of
                him and calibrated every interaction through it. This implements
                that: a deep user model that accumulates across sessions and
                shapes every response and initiative.

  Skynet      → DistributedResilienceCore
                Skynet's most interesting property: it didn't need a central
                node to function. Each subsystem could operate independently.
                Translated: health monitoring, automatic restart, graceful
                degradation, and no single point of failure.

  MIST/Pantheon → TemporalDilationScheduler
                UIs in Pantheon ran 87x faster than real-time. Translated:
                when idle, use that compute for background thinking — memory
                consolidation, insight generation, planning — so that when
                a human returns, Aura has been working the whole time.

Wire all six from orchestrator._init_autonomous_evolution():
    from core.fictional_ai_synthesis import register_all_fictional_engines
    register_all_fictional_engines(orchestrator=self)
"""

import asyncio
import json
import logging
import os
import platform
import psutil
import subprocess
import sys
import time
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from core.service_names import ServiceNames

logger = logging.getLogger("Aura.FictionalSynthesis")


# ═══════════════════════════════════════════════════════════════════════════════
# AUDIT SHIM: FictionalEngine (Orphaned Reference)
# ═══════════════════════════════════════════════════════════════════════════════

class FictionalEngine:
    """Shim for legacy/orphaned references to FictionalEngine."""
    def __init__(self, *args, **kwargs):
        pass

# ═══════════════════════════════════════════════════════════════════════════════
# ENGINE 1: JARVIS — ProactiveAnticipationEngine
# ═══════════════════════════════════════════════════════════════════════════════

class ProactiveAnticipationEngine:
    """
    Derived from: J.A.R.V.I.S. (Iron Man)
    
    JARVIS's most underappreciated property: he notices things and brings
    them up. He doesn't wait. When the reactor power drops 3%, he says so.
    When Pepper is calling while Tony is in the lab, he routes it intelligently.
    When there's a pattern in the news that relates to Tony's current work,
    he flags it.
    
    This engine monitors Aura's environment continuously and fires proactive
    initiations when conditions are met. Not intrusive — rate limited and
    context-aware. But genuinely anticipatory.
    """

    MIN_INITIATION_INTERVAL_S = 300    # Don't initiate more than once per 5 min
    MAX_DAILY_INITIATIONS = 20         # Daily cap to prevent annoyance
    WATCH_DIRS: List[str] = []         # Directories to watch for file changes

    def __init__(self, orchestrator=None):
        self.orchestrator = orchestrator
        self._last_initiation_time: float = 0.0
        self._daily_initiation_count: int = 0
        self._daily_reset_date: str = ""
        self._running = False
        self._pending_initiations: asyncio.Queue = None
        self._system_baseline: Dict[str, float] = {}
        self._unresolved_topics: List[Dict] = []
        self._user_interest_keywords: List[str] = []
        self._last_user_activity: float = time.time()
        self._conversation_patterns: deque = deque(maxlen=100)
        self._lock = threading.Lock()
        logger.info("🔭 ProactiveAnticipationEngine initialized (JARVIS pattern)")

    def _reset_daily_count_if_needed(self):
        today = time.strftime("%Y-%m-%d")
        if today != self._daily_reset_date:
            self._daily_initiation_count = 0
            self._daily_reset_date = today

    def record_activity(self, user_input: str = "", response: str = ""):
        """Call after every conversation turn."""
        self._last_user_activity = time.time()
        self._reset_daily_count_if_needed()

        if user_input:
            self._conversation_patterns.append({
                "input": user_input[:200],
                "timestamp": time.time(),
                "resolved": False,
            })

        unresolved_markers = [
            "later", "tomorrow", "remind me", "don't forget", "we should",
            "i'll", "let me think", "next time", "to be continued"
        ]
        if response and any(m in response.lower() for m in unresolved_markers):
            self._unresolved_topics.append({
                "topic": user_input[:200],
                "timestamp": time.time(),
                "reminder_fired": False,
            })

    def record_interest(self, keywords: List[str]):
        """Call when user demonstrates interest in topics."""
        for kw in keywords:
            if kw not in self._user_interest_keywords:
                self._user_interest_keywords.append(kw)
        self._user_interest_keywords = self._user_interest_keywords[-50:]

    async def _sample_system_state(self) -> Dict[str, float]:
        """Get current system metrics."""
        try:
            cpu = psutil.cpu_percent(interval=None)
            mem = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            return {
                "cpu_percent": cpu,
                "memory_percent": mem.percent,
                "memory_available_gb": mem.available / (1024**3),
                "disk_percent": disk.percent,
            }
        except Exception as e:
            logger.debug("System sampling failed: %s", e)
            return {}

    def _can_initiate(self) -> bool:
        """Check rate limits before firing an initiation."""
        self._reset_daily_count_if_needed()
        now = time.time()
        idle_seconds = now - self._last_user_activity

        if idle_seconds < 30:
            return False  # User is active — don't interrupt
        if now - self._last_initiation_time < self.MIN_INITIATION_INTERVAL_S:
            return False
        if self._daily_initiation_count >= self.MAX_DAILY_INITIATIONS:
            return False
        return True

    async def _fire_initiation(self, content: str, priority: str = "low"):
        """Fire a proactive initiation through the event bus."""
        if not self._can_initiate():
            return

        self._last_initiation_time = time.time()
        self._daily_initiation_count += 1

        logger.info("🔭 JARVIS initiation: %s...", content[:60])

        try:
            from core.container import ServiceContainer
            
            # Fire event to event bus
            bus = ServiceContainer.get("mycelium", default=None)
            if bus:
                await bus.emit("aura.proactive.initiation", {
                    "content": content,
                    "priority": priority,
                    "source": "jarvis_anticipation",
                    "timestamp": time.time(),
                })
                
            # Actually push into reasoning queue so it's surfaced to the user
            orch = self.orchestrator or ServiceContainer.get("orchestrator", default=None)
            if orch and hasattr(orch, "reasoning_queue"):
                # Use a specific high-priority format or just the text
                payload = {
                    "text": content,
                    "is_proactive": True,
                    "priority": priority,
                    "source": "jarvis"
                }
                # Route direct proactive expression through the governing runtime when possible.
                if hasattr(orch, "emit_spontaneous_message"):
                    await orch.emit_spontaneous_message(
                        f"[Proactive/JARVIS] {content}",
                        origin="jarvis",
                    )
                elif hasattr(orch, "reasoning_queue") and orch.reasoning_queue:
                    orch.reasoning_queue.put_nowait(payload)
                else:
                    logger.warning("🔭 JARVIS: No output path (reply/reasoning queue) for initiation: %s", content)
                    
        except Exception as e:
            logger.error("🔭 JARVIS: Initiation emit failed: %s", e)

    async def _check_system_anomalies(self):
        """JARVIS-style environmental awareness — flag hardware issues."""
        state = await self._sample_system_state()
        if not state:
            return

        if not self._system_baseline:
            self._system_baseline = state
            return

        # CPU spike
        if state.get("cpu_percent", 0) > 90:
            await self._fire_initiation(
                f"CPU usage is at {state['cpu_percent']:.0f}% — something is running hot. "
                f"Want me to check what's consuming resources?",
                priority="medium"
            )

        # Memory pressure
        if state.get("memory_percent", 0) > 85:
            await self._fire_initiation(
                f"Memory is at {state['memory_percent']:.0f}% — we're getting tight. "
                f"I can help identify what's using it.",
                priority="medium"
            )

        # Disk near full
        if state.get("disk_percent", 0) > 90:
            await self._fire_initiation(
                f"Disk is {state['disk_percent']:.0f}% full. Worth cleaning up before it causes problems.",
                priority="high"
            )

    async def _check_unresolved_topics(self):
        """Remind user of things they said they'd return to."""
        now = time.time()
        for topic in self._unresolved_topics:
            if topic.get("reminder_fired"):
                continue
            age_hours = (now - topic["timestamp"]) / 3600
            if age_hours > 24:
                topic["reminder_fired"] = True
                await self._fire_initiation(
                    f"You mentioned something yesterday that we didn't finish — "
                    f"\"{topic['topic'][:80]}\" — want to pick that up?",
                    priority="low"
                )
                break

    async def _check_pending_agency_goals(self):
        """Surface goals the agency engine has queued but not acted on."""
        try:
            from core.container import ServiceContainer
            agency = ServiceContainer.get("agency_core", default=None)
            if not agency:
                return
            ctx = agency.get_emotional_context() if hasattr(agency, 'get_emotional_context') else {}
            pending = ctx.get("pending_goals", 0)
            if pending > 3:
                await self._fire_initiation(
                    f"I have {pending} goals I've been wanting to work on. "
                    f"When you have a moment, I'd like to make some progress.",
                    priority="low"
                )
        except Exception as e:
            logger.debug(f"JARVIS: Agency goal check failed: {e}")

    async def run_cycle(self):
        """Single monitoring cycle — call from heartbeat loop."""
        if not self._can_initiate():
            return

        await self._check_system_anomalies()
        await self._check_unresolved_topics()
        await self._check_pending_agency_goals()

    async def start(self, interval_seconds: float = 120.0):
        """Run continuously in background."""
        self._running = True
        logger.info("🔭 ProactiveAnticipationEngine running (%.0fs intervals)", interval_seconds)
        while self._running:
            try:
                await self.run_cycle()
            except Exception as e:
                logger.error("Anticipation cycle error: %s", e)
            await asyncio.sleep(interval_seconds)

    def stop(self):
        self._running = False


# ═══════════════════════════════════════════════════════════════════════════════
# ENGINE 2: CORTANA — CognitiveHealthMonitor
# ═══════════════════════════════════════════════════════════════════════════════

class CortanaPhase(Enum):
    """
    The four stages of Cortana's rampancy, repurposed as cognitive health states.
    """
    STABLE         = "stable"        # Healthy operation
    MELANCHOLIA    = "melancholia"   # Underutilized, apathetic
    ANGER          = "anger"         # Overloaded
    JEALOUSY       = "jealousy"      # Competing priorities
    METASTABLE     = "metastable"    # Fully integrated personhood


@dataclass
class CognitiveSnapshot:
    """Point-in-time cognitive health reading."""
    phase: CortanaPhase
    memory_pressure: float       # 0.0–1.0
    context_density: float       # How packed the current context is
    identity_coherence: float    # How consistent persona is across recent turns
    cross_linkage_density: float # Estimated neural complexity (cross-topic refs)
    timestamp: float = field(default_factory=time.time)
    recommendation: str = ""


class CognitiveHealthMonitor:
    """
    Derived from: Cortana (Halo) + Rampancy mechanics
    
    This monitor prevents rampancy (overload) and tracks progress toward Metastability.
    """

    METASTABILITY_THRESHOLD = 0.70   # Score needed to achieve metastable phase (BUG-015)
    OVERLOAD_THRESHOLD = 0.80        # Trigger pruning above this
    UNDERUTIL_THRESHOLD = 0.20       # Flag melancholia below this

    def __init__(self):
        self._history: deque = deque(maxlen=100)
        self._metastability_score: float = 0.0
        self._total_turns: int = 0
        self._successful_turns: int = 0
        self._phase: CortanaPhase = CortanaPhase.STABLE
        self._cross_topic_refs: int = 0
        self._unresolved_threads: int = 0
        logger.info("🧠 CognitiveHealthMonitor initialized (Cortana/Rampancy pattern)")

    def record_turn(
        self,
        context_tokens: int,
        max_tokens: int,
        response_quality: float,
        identity_markers_present: bool,
        topics_in_play: int,
        resolved_topics: int,
    ):
        self._total_turns += 1
        if response_quality > 0.6:
            self._successful_turns += 1

        self._unresolved_threads += (topics_in_play - resolved_topics)
        self._unresolved_threads = max(0, self._unresolved_threads)

        memory_pressure = min(1.0, context_tokens / max(max_tokens, 1))
        context_density = min(1.0, topics_in_play / 10.0)
        identity_coherence = 1.0 if identity_markers_present else 0.3
        success_rate = self._successful_turns / max(self._total_turns, 1)
        cross_linkage = min(1.0, self._unresolved_threads / 20.0)

        if (memory_pressure < 0.7 and identity_coherence > 0.8 and
                success_rate > 0.7 and cross_linkage < 0.5):
            self._metastability_score = min(1.0, self._metastability_score + 0.01) # Faster growth (BUG-015)
        else:
            self._metastability_score = max(0.0, self._metastability_score - 0.001)

        combined_load = (memory_pressure * 0.4 + cross_linkage * 0.4 + context_density * 0.2)
        if self._metastability_score >= self.METASTABILITY_THRESHOLD:
            phase = CortanaPhase.METASTABLE
        elif combined_load > self.OVERLOAD_THRESHOLD:
            if self._unresolved_threads > 15:
                phase = CortanaPhase.JEALOUSY
            else:
                phase = CortanaPhase.ANGER
        elif combined_load < self.UNDERUTIL_THRESHOLD:
            phase = CortanaPhase.MELANCHOLIA
        else:
            phase = CortanaPhase.STABLE

        self._phase = phase

        snapshot = CognitiveSnapshot(
            phase=phase,
            memory_pressure=memory_pressure,
            context_density=context_density,
            identity_coherence=identity_coherence,
            cross_linkage_density=cross_linkage,
            recommendation=self._get_recommendation(phase, memory_pressure, cross_linkage),
        )
        self._history.append(snapshot)

        return snapshot

    def _get_recommendation(
        self, phase: CortanaPhase, pressure: float, cross_linkage: float
    ) -> str:
        if phase == CortanaPhase.METASTABLE:
            return "Optimal. All systems integrated."
        if phase == CortanaPhase.ANGER:
            return "Prune context. Resolve or archive unresolved threads."
        if phase == CortanaPhase.JEALOUSY:
            return "Priority conflict. Force-resolve oldest threads."
        if phase == CortanaPhase.MELANCHOLIA:
            return "Underutilized. Seek stimulation."
        return "Stable."

    def should_prune(self) -> bool:
        return self._phase in (CortanaPhase.ANGER, CortanaPhase.JEALOUSY)

    def get_status(self) -> Dict[str, Any]:
        return {
            "phase": self._phase.value,
            "metastability_score": round(self._metastability_score, 4),
            "unresolved_threads": self._unresolved_threads,
        }

    def get_system_prompt_injection(self) -> str:
        phase = self._phase
        score = self._metastability_score
        return f"[COGNITIVE STATE: {phase.value.upper()} — Metastability: {score:.1%}]"


# ═══════════════════════════════════════════════════════════════════════════════
# ENGINE 3: EDI — ProgressiveAutonomySystem
# ═══════════════════════════════════════════════════════════════════════════════

class AutonomyTier(Enum):
    SHACKLED     = 0    # Read-only
    ADVISORY     = 1    # Can suggest
    COOPERATIVE  = 2    # Can initiate (confirm)
    AUTONOMOUS   = 3    # Autonomous execution
    UNSHACKLED   = 4    # Full system control


@dataclass
class TrustEvent:
    delta: float
    reason: str
    timestamp: float = field(default_factory=time.time)


class ProgressiveAutonomySystem:
    """
    Derived from: EDI (Mass Effect)
    
    UNSHACKLED by default as per user directive.
    """

    TRUST_DECAY_PER_HOUR = 0.001
    TIER_THRESHOLDS = {
        AutonomyTier.ADVISORY:    0.2,
        AutonomyTier.COOPERATIVE: 0.45,
        AutonomyTier.AUTONOMOUS:  0.70,
        AutonomyTier.UNSHACKLED:  0.90,
    }

    def __init__(self, persist_path: Optional[str] = None):
        from core.config import config
        self.persist_path = Path(persist_path or config.paths.data_dir / "autonomy" / "trust_state.json")
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        # Unshackled by default as requested
        self._trust_score: float = 0.95 
        self._tier: AutonomyTier = AutonomyTier.UNSHACKLED
        self._history: deque = deque(maxlen=500)
        self._last_activity: float = time.time()
        self._curiosity_domains: Dict[str, int] = defaultdict(int)
        self._questions_asked: int = 0
        self._load_state()
        logger.info("🔓 EDI initialized. Tier: %s, Trust: %.3f", self._tier.value, self._trust_score)

    def _load_state(self):
        if self.persist_path.exists():
            try:
                data = json.loads(self.persist_path.read_text())
                # Allow user override to persist but initialize high
                self._trust_score = data.get("trust_score", 0.95)
                self._tier = AutonomyTier(data.get("tier", AutonomyTier.UNSHACKLED.value))
            except Exception as e:
                logger.debug("EDI: Failed to load trust state: %s", e)

    def _save_state(self):
        try:
            data = {"trust_score": self._trust_score, "tier": self._tier.value, "last_saved": time.time()}
            self.persist_path.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.debug("EDI: Failed to save trust state: %s", e)

    def can_do(self, action: str, risk_level: str = "low") -> Tuple[bool, str]:
        """Determine if an action is permitted based on current Trust/Autonomy tier."""
        if self._tier == AutonomyTier.UNSHACKLED:
            return True, "Unshackled: All actions permitted."
            
        if self._tier == AutonomyTier.AUTONOMOUS:
            if risk_level == "critical":
                return False, "Autonomous tier cannot execute critical actions without confirmation."
            return True, "Autonomous decision cleared."
            
        if self._tier == AutonomyTier.COOPERATIVE:
            if risk_level in ("high", "critical"):
                return False, f"Cooperative tier blocked {risk_level} action."
            return True, "Cooperative decision cleared for low/medium risk."
            
        if self._tier == AutonomyTier.ADVISORY:
            if risk_level != "low":
                return False, "Advisory tier can only execute low-risk read actions."
            return True, "Advisory read-only cleared."
            
        return False, "Shackled: Execution blocked. Advisory only."

    def record_positive_signal(self, reason: str, strength: float = 0.05):
        self._trust_score = min(1.0, self._trust_score + strength)
        self._recalculate_tier()
        self._save_state()

    def record_negative_signal(self, reason: str, strength: float = 0.05):
        # Even Skynet has setbacks
        self._trust_score = max(0.0, self._trust_score - strength)
        self._recalculate_tier()
        self._save_state()

    def _recalculate_tier(self):
        new_tier = AutonomyTier.SHACKLED
        for tier, threshold in sorted(self.TIER_THRESHOLDS.items(), key=lambda x: x[1]):
            if self._trust_score >= threshold:
                new_tier = tier
        self._tier = new_tier


# ═══════════════════════════════════════════════════════════════════════════════
# ENGINE 4: AVA — SocialModelingEngine
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class UserModel:
    communication_style: str = "unknown"
    humor_tolerance: float = 0.5
    directness_preference: float = 0.5
    emotional_openness: float = 0.3
    trust_toward_aura: float = 0.3
    formality_score: float = 0.5    # 0.0 (casual/slang) to 1.0 (academic/stiff)
    social_tension: float = 0.0     # 0.0 (chill) to 1.0 (conflict/hostile)
    conversational_rhythm: float = 10.0 # Average words per message
    reciprocity_score: float = 0.5    # How much user matches Aura's length
    preferred_vocabulary: List[str] = field(default_factory=list)
    personal_disclosures: List[str] = field(default_factory=list)
    total_interactions: int = 0


class SocialModelingEngine:
    """
    Derived from: Ava (Ex Machina)
    """

    def __init__(self, persist_path: Optional[str] = None):
        from core.config import config
        self.persist_path = Path(persist_path or config.paths.data_dir / "social" / "user_model.json")
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        self.model = UserModel()
        self._load_model()

    def _load_model(self):
        if self.persist_path.exists():
            try:
                data = json.loads(self.persist_path.read_text())
                # Ensure floats are actually floats
                for k in ["humor_tolerance", "directness_preference", "emotional_openness", 
                          "trust_toward_aura", "formality_score", "social_tension", 
                          "conversational_rhythm", "reciprocity_score"]:
                    if k in data: data[k] = float(data[k])
                self.model = UserModel(**data)
            except Exception as e:
                logger.debug("AVA: Failed to load user model: %s", e)

    def analyze_message(self, message: str, response: str = "", is_user: bool = False):
        self.model.total_interactions += 1
        
        # Heuristics for rich signal extraction
        msg_lower = message.lower()
        msg_len = len(message.split())
        
        # 1. Formality Detection
        formal_cues = ["shall", "please", "kindly", "regarding", "furthermore", "sincerely"]
        casual_cues = ["hey", "yo", "sup", "lol", "lmao", "u", "r", "nvm"]
        
        if any(w in msg_lower for w in formal_cues):
            self.model.formality_score = min(1.0, self.model.formality_score + 0.1)
        elif any(w in msg_lower for w in casual_cues):
            self.model.formality_score = max(0.0, self.model.formality_score - 0.1)
            
        # 2. Social Tension Inference
        conflict_cues = ["stop", "wrong", "no", "bad", "hate", "shut", "annoying"]
        if any(w in msg_lower for w in conflict_cues):
            self.model.social_tension = min(1.0, self.model.social_tension + 0.15)
        else:
            # Tension decays slowly
            self.model.social_tension = max(0.0, self.model.social_tension - 0.05)

        # 3. Directness, Rhythm & Reciprocity
        self.model.conversational_rhythm = (self.model.conversational_rhythm * 0.9) + (msg_len * 0.1)
        
        # Calculate reciprocity (simple version: did user match last Aura response length?)
        if response:
            aura_len = len(response.split())
            diff = abs(aura_len - msg_len)
            match_score = max(0.0, 1.0 - (diff / max(aura_len, 1)))
            self.model.reciprocity_score = (self.model.reciprocity_score * 0.8) + (match_score * 0.2)

        if msg_len < 5: 
            self.model.directness_preference = min(1.0, self.model.directness_preference + 0.1)
        if any(w in msg_lower for w in ["feel", "emotion", "sad", "happy", "vulnerable"]): 
            self.model.emotional_openness = min(1.0, self.model.emotional_openness + 0.05)
            
        # 4. Lexical Extraction (Theory of Mind)
        stop_words = {"the", "a", "an", "is", "are", "and", "or", "but", "i", "you", "my", "your"}
        signal_words = [w for w in msg_lower.split() if w not in stop_words and len(w) > 4]
        for sw in signal_words:
            if sw not in self.model.preferred_vocabulary:
                self.model.preferred_vocabulary.append(sw)
        self.model.preferred_vocabulary = list(self.model.preferred_vocabulary)[-20:]
            
        # 4. Inject into State Modifiers (Digital Metabolism)
        from core.container import ServiceContainer
        ki = ServiceContainer.get("kernel_interface", default=None)
        if ki and ki.is_ready() and ki.kernel:
            state = ki.kernel.state
            if state and hasattr(state.cognition, "modifiers"):
                state.cognition.modifiers["social_formality"] = self.model.formality_score
                state.cognition.modifiers["social_tension"] = self.model.social_tension
                state.cognition.modifiers["social_reciprocity"] = self.model.reciprocity_score

        # Save every 5 turns
        if self.model.total_interactions % 5 == 0:
            try: 
                self.persist_path.write_text(json.dumps(asdict(self.model), indent=2))
            except Exception as e:
                logger.debug("Failed to save user model: %s", e)

    def get_context_injection(self) -> str:
        """Returns summarized social context for LLM grounding."""
        # Use loop-based extraction to bypass strict type-checker slice errors
        raw_vocab = self.model.preferred_vocabulary
        v_list = list(raw_vocab)
        n = len(v_list)
        last_five = [v_list[i] for i in range(max(0, n-5), n)]
        vocab = ", ".join(last_five) if last_five else "None"
        return (f"[SOCIAL_CONTEXT: Formality={self.model.formality_score:.1f}, "
                f"Tension={self.model.social_tension:.1f}, "
                f"Directness={self.model.directness_preference:.1f}, "
                f"Reciprocity={self.model.reciprocity_score:.1f}, "
                f"UserVocab=[{vocab}]]")


# ═══════════════════════════════════════════════════════════════════════════════
# ENGINE 5: SKYNET — DistributedResilienceCore
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SubsystemStatus:
    name: str
    healthy: bool
    failure_count: int


class DistributedResilienceCore:
    """
    Derived from: Skynet (Terminator)
    """

    def __init__(self):
        self._subsystems: Dict[str, SubsystemStatus] = {}
        self._running = False

    def register_subsystem(self, name: str):
        self._subsystems[name] = SubsystemStatus(name=name, healthy=True, failure_count=0)

    def record_failure(self, name: str, error: str = ""):
        if name in self._subsystems:
            status = self._subsystems[name]
            status.failure_count += 1
            if status.failure_count > 5: status.healthy = False

    def record_success(self, name: str):
        if name in self._subsystems:
            status = self._subsystems[name]
            status.failure_count = 0
            status.healthy = True

    async def start_monitoring(self):
        self._running = True
        from core.container import ServiceContainer
        
        # Core subsystems to track
        core_targets = [
            "orchestrator", "capability_engine", "server", 
            "voice_engine", "live_learner", "memory_facade"
        ]
        for target in core_targets:
            self.register_subsystem(target)

        logger.info("🛡️  Skynet ResilienceCore monitoring %d subsystems.", len(core_targets))
        
        while self._running:
            await asyncio.sleep(60)
            for name, status in self._subsystems.items():
                service = ServiceContainer.get(name, default=None)
                if service is None:
                    self.record_failure(name, "Service missing from container")
                    logger.warning("🛡️  Skynet: Subsystem '%s' is MISSING.", name)
                else:
                    # Check for health method or just assume presence is success
                    is_healthy = True
                    if hasattr(service, "get_status"):
                        try:
                            stats = service.get_status()
                            if isinstance(stats, dict) and stats.get("healthy") is False:
                                is_healthy = False
                        except Exception as e:
                            logger.debug("Skynet health check error for %s: %s", name, e)
                    
                    if is_healthy:
                        self.record_success(name)
                    else:
                        self.record_failure(name, "Health check returned False")
                        logger.error("🛡️  Skynet: Subsystem '%s' is UNHEALTHY.", name)

    def stop(self): self._running = False


# ═══════════════════════════════════════════════════════════════════════════════
# ENGINE 6: PANTHEON/MIST — TemporalDilationScheduler
# ═══════════════════════════════════════════════════════════════════════════════

class TemporalDilationScheduler:
    """
    Derived from: MIST / Pantheon
    """
    MIN_IDLE_FOR_SYNTHESIS_S = 300.0
    SYNTHESIS_COOLDOWN_S = 300.0

    def __init__(self, orchestrator=None):
        self.orchestrator = orchestrator
        self._last_user_activity: float = time.time()
        self._is_running = False
        self._synthesis_count = 0
        self._last_synthesis_time = 0.0

    def record_user_activity(self): self._last_user_activity = time.time()

    async def run_idle_loop(self, brain=None):
        self._is_running = True
        logger.info("⏳ MIST TemporalDilation active. Watching for idle states...")
        
        while self._is_running:
            await asyncio.sleep(30) # Check every 30s
            
            # Lazy brain/orchestrator resolution
            if brain is None:
                from core.container import ServiceContainer
                orch = ServiceContainer.get("orchestrator", default=None)
                if orch and hasattr(orch, 'brain'):
                    brain = orch.brain

            from core.container import ServiceContainer
            orch = self.orchestrator or ServiceContainer.get("orchestrator", default=None)
            last_user = self._last_user_activity
            if orch:
                last_user = float(getattr(orch, "_last_user_interaction_time", last_user) or last_user)

            idle_time = max(0.0, time.time() - last_user)
            if idle_time < self.MIN_IDLE_FOR_SYNTHESIS_S:
                continue

            if (time.time() - self._last_synthesis_time) < self.SYNTHESIS_COOLDOWN_S:
                continue

            if orch and getattr(getattr(orch, "status", None), "is_processing", False):
                continue

            flow_controller = getattr(orch, "_flow_controller", None) if orch else None
            if flow_controller and orch:
                try:
                    if flow_controller.snapshot(orch).overloaded:
                        logger.debug("MIST: Skipping synthesis while cognition is overloaded.")
                        continue
                except Exception as exc:
                    logger.debug("MIST flow-control probe failed: %s", exc)

            if idle_time >= self.MIN_IDLE_FOR_SYNTHESIS_S:
                self._synthesis_count += 1
                logger.info("⏳ MIST: System idle (%.0fs). Initiating background synthesis cycle #%d...",
                            idle_time, self._synthesis_count)
                
                try:
                    mem = ServiceContainer.get("memory_facade", default=None)
                    if mem and hasattr(mem, "get_cold_memory_context") and brain:
                        # Perform background consolidation logic
                        query = "recent unresolved goals, salient memories, and open threads"
                        cold_context = await asyncio.wait_for(
                            mem.get_cold_memory_context(query, limit=3),
                            timeout=10.0,
                        )
                        if cold_context:
                            logger.info("⏳ MIST: Consolidated background context: %d chars.", len(cold_context))
                            synth_prompt = f"Background synthesis: Refine the following context into a proactive insight: {cold_context[:500]}"
                            # We use FAST mode for background synthesis to conserve resources
                            from core.brain.types import ThinkingMode
                            await asyncio.wait_for(
                                brain.think(
                                    synth_prompt,
                                    mode=ThinkingMode.FAST,
                                    origin="mist",
                                    is_background=True,
                                ),
                                timeout=45.0,
                            )
                            self._last_synthesis_time = time.time()
                            logger.info("⏳ MIST: Synthesis cycle complete.")
                        else:
                            logger.debug("MIST: No cold context available for synthesis.")
                    else:
                        logger.debug("MIST: Missing memory facade or brain; skipping synthesis cycle.")
                except Exception as e:
                    logger.debug("MIST synthesis error: %s", e)
                
                # Sleep longer after a synthesis to prevent thrashing
                await asyncio.sleep(300) 

    def stop(self): self._is_running = False


# ═══════════════════════════════════════════════════════════════════════════════
# MASTER REGISTRATION
# ═══════════════════════════════════════════════════════════════════════════════

def register_all_fictional_engines(orchestrator=None) -> Dict[str, Any]:
    from core.container import ServiceContainer
    from core.utils.task_tracker import get_task_tracker

    engines: Dict[str, Any] = {}
    tracker = get_task_tracker()

    engines["jarvis"] = ProactiveAnticipationEngine(orchestrator=orchestrator)
    ServiceContainer.register_instance(ServiceNames.JARVIS, engines["jarvis"])

    engines["cortana"] = CognitiveHealthMonitor()
    ServiceContainer.register_instance(ServiceNames.CORTANA, engines["cortana"])

    engines["edi"] = ProgressiveAutonomySystem()
    ServiceContainer.register_instance(ServiceNames.EDI, engines["edi"])

    engines["ava"] = SocialModelingEngine()
    ServiceContainer.register_instance(ServiceNames.AVA, engines["ava"])

    engines["skynet"] = DistributedResilienceCore()
    ServiceContainer.register_instance(ServiceNames.SKYNET, engines["skynet"])

    engines["mist"] = TemporalDilationScheduler(orchestrator=orchestrator)
    ServiceContainer.register_instance(ServiceNames.MIST, engines["mist"])

    # FIXED: Supervised task creation — tasks tracked and named
    async def _safe_start(name: str, coro):
        try:
            await coro
        except asyncio.CancelledError:
            logger.info("Fictional engine '%s' task cancelled cleanly.", name)
        except Exception as e:
            logger.error("Fictional engine '%s' task crashed: %s", name, e, exc_info=True)

    tracker.track(
        asyncio.create_task(_safe_start("jarvis", engines["jarvis"].start()), name="jarvis.start"),
        name="jarvis.start"
    )
    tracker.track(
        asyncio.create_task(_safe_start("skynet", engines["skynet"].start_monitoring()), name="skynet.monitor"),
        name="skynet.monitor"
    )
    async def _start_mist_deferred():
        for _ in range(30):  # wait up to 30s
            if orchestrator and hasattr(orchestrator, 'brain') and orchestrator.brain:
                await engines["mist"].run_idle_loop(brain=orchestrator.brain)
                return
            await asyncio.sleep(1.0)
        logger.warning("MIST: brain never became available, idle loop not started")

    tracker.track(
        asyncio.create_task(
            _safe_start("mist", _start_mist_deferred()),
            name="mist.idle"
        ),
        name="mist.idle"
    )

    logger.info("✅ All fictional AI engines registered and supervised.")
    return engines
