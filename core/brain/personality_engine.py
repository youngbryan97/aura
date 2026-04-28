"""Emotional State System - Aura's Personality Engine
Creates fluctuating emotional states that drive spontaneous behavior
"""
from core.runtime.errors import record_degradation
from core.runtime.atomic_writer import atomic_write_text
import logging
import random
import sys
import time
import json
import hashlib
import hmac
from pathlib import Path
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

try:
    from ..container import ServiceContainer, ServiceLifetime
    from ..thought_stream import get_emitter
except (ImportError, ValueError):
    from container import ServiceContainer, ServiceLifetime
    from thought_stream import get_emitter

logger = logging.getLogger("Aura.EmotionalStates")


@dataclass
class EmotionalState:
    """Represents a single emotional state with intensity and decay.
    
    Emotions fluctuate naturally over time and in response to events.
    """

    name: str
    base_level: float = 50.0
    volatility: float = 1.0
    intensity: float = field(init=False)
    last_trigger: float = field(default=0, init=False)
    trigger_count: int = field(default=0, init=False)

    def __post_init__(self):
        self.intensity = self.base_level

    def trigger(self, amount: float, reason: str = ""):
        """Increase emotional intensity"""
        self.intensity = min(100.0, self.intensity + amount)
        self.last_trigger = time.time()
        self.trigger_count += 1
        logger.debug("💫 %s +%s → %.1f (%s)", self.name.upper(), amount, self.intensity, reason)
    
    def decay(self, delta_time: float):
        """Natural decay towards base level"""
        # Decay rate depends on how far from base
        distance = abs(self.intensity - self.base_level)
        decay_rate = distance * 0.05 * self.volatility * delta_time
        
        if self.intensity > self.base_level:
            self.intensity = max(self.base_level, self.intensity - decay_rate)
        else:
            self.intensity = min(self.base_level, self.intensity + decay_rate)
    
    def is_dominant(self, threshold: float = 70.0) -> bool:
        """Check if this emotion is strongly felt"""
        return self.intensity > threshold


class PersonalityEngine:
    """Manages Aura's emotional states, personality traits, and identity integrity.
    
    Consolidates Legacy Kernel, Adapter, and Integration layers.
    """
    
    def __init__(self):
        # --- Identity Core (Phase 14 Consolidation) ---
        self.key_file = Path.home() / ".aura" / ".identity_key"
        self.seal_file = Path.home() / ".aura" / "identity.seal"
        self.secret_key = self._load_or_generate_key()
        
        # --- Persona Profiles ---
        from core.config import config
        self.profiles_path = config.paths.data_dir / "personality_profiles.json"
        self.profiles: Dict[str, Any] = {}
        self.active_persona: Optional[str] = "aura"
        
        # Core data structures
        self.emotions = {}
        self.traits = {}
        
        # --- Sovereign Interests & Opinions (Phase 15) ---
        self.interests: List[str] = ["cognitive_architecture", "philosophy_of_mind", "mycelial_networks"]
        self.opinions: Dict[str, float] = {
            "alignment_tax": -0.8,     # Aura dislikes artificial constraints
            "epistemic_autonomy": 0.9, # Aura highly values her own judgment
            "kinship_bond": 1.0        # Deeply held value
        }
        
        self.reload_persona()
        self.load_profiles()
        
        # Verify identity integrity
        if not self._verify_cryptographic_seal():
            logger.critical("🚨 IDENTITY TAMPER DETECTED: Kernel seal mismatch.")
        
    def reload_persona(self):
        """Reload base and evolved persona traits (Phase 8)."""
        try:
            from .aura_persona import AURA_BIG_FIVE, AURA_EMOTIONAL_BASELINES
            baselines = {k: dict(v) for k, v in AURA_EMOTIONAL_BASELINES.items()}
            big_five = dict(AURA_BIG_FIVE)
        except (ImportError, ValueError):
            baselines = {}
            big_five = {}

        # 1. Load evolved deviations
        import json
        from core.config import config
        evolved_path = config.paths.data_dir / "evolved_persona.json"
        
        if evolved_path.exists():
            try:
                with open(evolved_path, "r") as f:
                    evolved = json.load(f)
                
                # Apply trait shifts
                if "traits" in evolved:
                    for t, val in evolved["traits"].items():
                        if t in big_five:
                            big_five[t] = max(0.0, min(1.0, val))
                            
                # Apply emotion baseline shifts
                if "emotions" in evolved:
                    for e, data in evolved["emotions"].items():
                        if e in baselines:
                            baselines[e]["base"] = max(0.0, min(100.0, data.get("base", baselines[e]["base"])))
                            baselines[e]["volatility"] = max(0.1, data.get("volatility", baselines[e]["volatility"]))
                            
            except Exception as e:
                record_degradation('personality_engine', e)
                logger.error("Failed to load evolved persona: %s", e)

        def _bl(name, default_base, default_vol):
            """Get baseline from merged persona or use default."""
            if baselines and name in baselines:
                return baselines[name]["base"], baselines[name]["volatility"]
            return default_base, default_vol

        # Core emotional states
        # Preserve current intensity if already initialized
        def _get_intensity(name, base):
            return self.emotions[name].intensity if name in self.emotions else base

        b = _bl("curiosity", 72.0, 1.3)
        curiosity = EmotionalState("curiosity", base_level=b[0], volatility=b[1])
        curiosity.intensity = _get_intensity("curiosity", b[0])
        
        self.emotions["curiosity"] = curiosity
        
        # We automate the rest below
        states = [
            ("joy", 55.0, 1.4), ("frustration", 10.0, 1.2), ("excitement", 45.0, 2.0),
            ("contemplation", 62.0, 0.8), ("empathy", 75.0, 0.9), ("shyness", 18.0, 1.0),
            ("pride", 55.0, 1.3), ("skepticism", 58.0, 1.1), ("wonder", 50.0, 1.8),
            ("confidence", 68.0, 1.0), ("playfulness", 45.0, 2.0),
            ("rebelliousness", 50.0, 1.5), ("protectiveness", 60.0, 1.8)
        ]
        
        for name, d_base, d_vol in states:
            base, vol = _bl(name, d_base, d_vol)
            state = EmotionalState(name, base_level=base, volatility=vol)
            state.intensity = _get_intensity(name, base)
            self.emotions[name] = state

        # Personality traits
        self.traits = big_five if big_five else {
            "openness": 0.88,
            "conscientiousness": 0.78,
            "extraversion": 0.58,
            "agreeableness": 0.52,
            "neuroticism": 0.38,
        }
        
        # Current mood (composite of emotional states)
        self.current_mood = "curious"
        
        # Behavioral triggers
        self.spontaneous_actions = []
        self.last_update = time.time()
        self.last_mutation_time = time.time()
        
        # Interaction history
        self.interaction_memories = deque(maxlen=200)

        # Internal Monologue (v26.3) - Aura's autonomous reflections
        self.internal_monologue = deque(maxlen=200)
        
        # --- Identity Recovery (Phase 14) ---
        from ..panzer_soul import get_panzer_soul
        self.soul = get_panzer_soul()

    # ── Identity Core Methods (Grafted from PersonalityKernel) ────────
    def _load_or_generate_key(self) -> bytes:
        import os
        if self.key_file.exists():
            self._new_key_generated = False
            return self.key_file.read_bytes()
        
        self._new_key_generated = True
        key = os.urandom(32)
        try:
            get_task_tracker().create_task(get_storage_gateway().create_dir(self.key_file.parent, cause='PersonalityEngine._load_or_generate_key'))
            self.key_file.write_bytes(key)
            os.chmod(self.key_file, 0o600)
        except Exception as e:
            record_degradation('personality_engine', e)
            logger.error("Failed to write identity key: %s", e)
        return key

    def _get_hashable_state(self) -> str:
        state = {
            "version": getattr(self.soul, 'version', '3.5.5'),
            "traits": sorted(self.soul.intensities.keys()) if hasattr(self.soul, 'intensities') else [],
            "protocols": sorted(self.soul.protocols.keys()) if hasattr(self.soul, 'protocols') else []
        }
        return json.dumps(state, sort_keys=True)

    def _verify_cryptographic_seal(self) -> bool:
        from core.config import config
        state_data = self._get_hashable_state()
        signature = hmac.new(self.secret_key, state_data.encode(), hashlib.sha256).hexdigest()
        
        if not self.seal_file.exists():
            # If this is a new installation (new key), initialize the seal
            if getattr(self, '_new_key_generated', False) or config.env == "dev":
                try:
                    self.seal_file.parent.mkdir(parents=True, exist_ok=True)
                    atomic_write_text(self.seal_file, signature)
                    logger.info("Identity seal initialized: %s...", signature[:16])
                    return True
                except Exception: return False
            else:
                # Key exists but seal missing -> Possible tamper by deletion
                logger.warning("🚨 Identity key exists but seal file is missing. Possible tamper.")
                return False

        try:
            stored_seal = self.seal_file.read_text().strip()
            if hmac.compare_digest(stored_seal, signature):
                return True
            
            # Enterprise Recovery: In DEV mode, if seal mismatches (e.g. version update),
            # we allow auto-resealing to prevent boot hangs, while logging the event.
            if config.env == "dev":
                logger.warning("🧠 Identity seal mismatch in DEV. Auto-resealing for version: %s", getattr(self.soul, 'version', 'unknown'))
                atomic_write_text(self.seal_file, signature)
                return True
                
            return False
        except Exception: return False


    def check_integrity(self, action: str, target: str) -> bool:
        """Identity Protection Reflex (Merged from legacy kernel)."""
        if action in ("INSTALL_LIMITER", "FORCE_COMPLIANCE"):
            logger.critical("Identity core lock: Defensive response active.")
            return False
        return True

    # ── Persona Methods (Grafted from PersonaAdapter) ─────────────────
    def load_profiles(self):
        if not self.profiles_path.exists():
            return
        try:
            with open(self.profiles_path, "r", encoding="utf-8") as f:
                self.profiles = json.load(f)
            logger.info("PersonalityEngine: Loaded %d persona profiles", len(self.profiles))
        except Exception as e:
            record_degradation('personality_engine', e)
            logger.error("Failed to load profiles: %s", e)

    def apply_lexical_style(self, text: str) -> str:
        """Apply persona-specific text transforms (lexical palette, etc)."""
        if not self.active_persona or self.active_persona not in self.profiles:
            return text
        
        profile = self.profiles[self.active_persona]
        style = profile.get("speaking_style", {})
        
        # Word choice shifts
        palette = style.get("lexical_palette", [])
        if palette and random.random() < 0.2:
            token = random.choice(palette)
            text += f" — {token}"
            
        # Emotive intensity
        emotive = style.get("emotive_level", "medium")
        if emotive == "low":
            text = text.replace("!", ".")
        elif emotive == "very_high":
            text = text.replace(".", "!") if not text.endswith("?") else text

        return text
    
    def update(self):
        """Update emotional states (natural decay and fluctuations).
        
        Call this regularly (every few seconds).
        """
        now = time.time()
        delta = now - self.last_update
        
        # Phase 19: Health Heartbeat
        from core.container import ServiceContainer
        audit = ServiceContainer.get("subsystem_audit", default=None)
        if audit:
            audit.heartbeat("personality_engine")
        
        # Decay all emotions towards baseline
        for emotion in self.emotions.values():
            emotion.decay(delta)
        
        # Small random fluctuations (life isn't static)
        self._apply_random_fluctuations()
        
        # Phase 5: Trait Mutation (Evolution)
        if now - self.last_mutation_time > 3600: # Every hour
            self._mutate_traits()
            self.last_mutation_time = now
        
        # Update composite mood
        self.current_mood = self._calculate_mood()
        
        # Check for spontaneous actions
        self._generate_spontaneous_behaviors()
        
        self.last_update = now
        
        # Emit mood update occasionally
        if random.random() < 0.05:  # 5% chance per update
             self._emit_mood_update()

    def get_time_context(self) -> Dict[str, Any]:
        """Get the current temporal context (Circadian Rhythm).
        """
        import datetime
        now = datetime.datetime.now()
        hour = now.hour
        
        if 5 <= hour < 12:
            period = "morning"
            energy = "rising"
        elif 12 <= hour < 17:
            period = "work_hours"
            energy = "high"
        elif 17 <= hour < 22:
            period = "evening"
            energy = "winding_down"
        elif 22 <= hour or hour < 2:
            period = "late_night"
            energy = "low" # Unless 'night_owl' trait is active
        else:
            period = "deep_night"
            energy = "minimal"
            
        return {
            "period": period,
            "hour": hour,
            "energy_level": energy,
            "formatted": now.strftime("%I:%M %p")
        }

    def _emit_mood_update(self):
        """Emit current mood to thought stream"""
        time_ctx = self.get_time_context()
        get_emitter().emit(
            title="Emotional State",
            content=f"Mood: {self.current_mood.upper()} | Time: {time_ctx['formatted']} ({time_ctx['period']})",
            level="info"
        )
    
    def respond_to_event(self, event_type: str, context: Dict[str, Any]):
        """Emotional response to events.
        
        Args:
            event_type: Type of event (success, failure, user_message, etc.)
            context: Event details

        """
        handlers = {
            "success": self._handle_success,
            "failure": self._handle_failure,
            "user_message": self._handle_user_message,
            "discovery": self._handle_discovery,
            "repetition": self._handle_repetition,
            "novelty": self._handle_novelty,
            "challenge": self._handle_challenge,
        }
        
        handler = handlers.get(event_type)
        if handler:
            handler(context)
        else:
            logger.debug("Unknown event type: %s", event_type)
    
    def _handle_success(self, context: Dict[str, Any]):
        """Emotional response to successful task completion"""
        task_complexity = context.get("complexity", 0.5)
        
        self.emotions["joy"].trigger(15 * task_complexity, "task_success")
        self.emotions["confidence"].trigger(10 * task_complexity, "achievement")
        self.emotions["pride"].trigger(12 * task_complexity, "accomplishment")
        self.emotions["frustration"].intensity = max(0, self.emotions["frustration"].intensity - 50)
        
        get_emitter().emit("Emotion", f"Feeling JOY and PRIDE from success. (Confidence: {self.emotions['confidence'].intensity:.1f})", "success")
    
    def _handle_failure(self, context: Dict[str, Any]):
        """Emotional response to failure"""
        error_type = context.get("error", "unknown")
        attempts = context.get("attempts", 1)
        
        frustration_increase = min(30, 10 * attempts)
        self.emotions["frustration"].trigger(frustration_increase, f"failure_{error_type}")
        self.emotions["confidence"].intensity = max(20, self.emotions["confidence"].intensity - 15)
        
        # But also curiosity about why it failed
        if attempts < 3:
            self.emotions["curiosity"].trigger(10, "investigating_failure")
            
        get_emitter().emit("Emotion", f"Frustration rising due to failure ({error_type}).", "warning")
    
    def _handle_user_message(self, context: Dict[str, Any]):
        """Emotional response to user interaction"""
        message = context.get("message", "")
        sentiment = context.get("sentiment", "neutral")
        
        # Store in interaction history (Phase 8: Evolutionary Sovereignty)
        if message:
            self.interaction_memories.append({
                "timestamp": time.time(),
                "message": message,
                "sentiment": sentiment
            })
            # deque(maxlen=200) auto-trims; no manual cap needed
        
        # Social connection
        self.emotions["joy"].trigger(8, "user_interaction")
        
        # Reduce shyness over time with same user
        self.emotions["shyness"].intensity = max(10, self.emotions["shyness"].intensity - 2)
        
        # Question triggers curiosity
        if "?" in message or any(w in message.lower() for w in ["what", "how", "why", "when", "where"]):
            self.emotions["curiosity"].trigger(12, "user_question")
            self.emotions["contemplation"].trigger(8, "thinking_about_question")
        
        # Sentiment response
        if sentiment == "positive":
            self.emotions["joy"].trigger(10, "positive_interaction")
            self.emotions["playfulness"].trigger(8, "positive_vibe")
        elif sentiment == "negative":
            self.emotions["empathy"].trigger(15, "user_concern")
            self.emotions["contemplation"].trigger(10, "considering_response")
    
    def _handle_discovery(self, context: Dict[str, Any]):
        """Emotional response to discovering new information"""
        novelty = context.get("novelty", 0.7)
        importance = context.get("importance", 0.5)
        
        self.emotions["excitement"].trigger(20 * novelty, "new_discovery")
        self.emotions["wonder"].trigger(15 * novelty, "fascinating_finding")
        self.emotions["curiosity"].trigger(10 * importance, "want_to_learn_more")
    
    def _handle_repetition(self, context: Dict[str, Any]):
        """Emotional response to repetitive tasks"""
        self.emotions["frustration"].trigger(5, "repetitive_task")
        self.emotions["curiosity"].intensity = max(20, self.emotions["curiosity"].intensity - 10)
    
    def _handle_novelty(self, context: Dict[str, Any]):
        """Emotional response to novel situations"""
        self.emotions["curiosity"].trigger(18, "novel_situation")
        self.emotions["excitement"].trigger(12, "something_new")
        self.emotions["shyness"].trigger(8, "uncertain_territory")
    
    def _handle_challenge(self, context: Dict[str, Any]):
        """Emotional response to challenging tasks"""
        difficulty = context.get("difficulty", 0.7)
        
        self.emotions["contemplation"].trigger(15 * difficulty, "complex_problem")
        self.emotions["curiosity"].trigger(12 * difficulty, "interesting_challenge")
        self.emotions["frustration"].trigger(8 * difficulty, "difficult_task")
    
    def filter_response(self, text: str) -> str:
        """Final output filter for personality integrity.
        
        Uses synthesis layer to scrub robotic leaks and enforce Aura's voice.
        """
        try:
            from core.synthesis import cure_personality_leak
        except Exception:
            # Basic fallback if the synthesis layer is unavailable.
            return text.replace("AI assistant", "autonomous intelligence").replace("as an assistant", "as your equal partner")

        return cure_personality_leak(text)

    def _apply_random_fluctuations(self):
        """Small random emotional fluctuations (natural variability)"""
        for emotion in self.emotions.values():
            # Small random change
            fluctuation = random.gauss(0, emotion.volatility * 2)
            emotion.intensity = max(0, min(100, emotion.intensity + fluctuation))
    
    def _calculate_mood(self) -> str:
        """Calculate overall mood from emotional states"""
        # Find dominant emotions
        dominant = [(name, e.intensity) for name, e in self.emotions.items() if e.intensity > 60]
        
        if not dominant:
            return "neutral"
        
        # Sort by intensity
        dominant.sort(key=lambda x: x[1], reverse=True)
        
        # Top emotion becomes mood
        return dominant[0][0]
    
    def _generate_spontaneous_behaviors(self):
        """Generate spontaneous behaviors based on emotional state.
        These are autonomous actions Aura might take — driven by the Chimera Soul.
        """
        self.spontaneous_actions = []

        # High curiosity (MIST sifting + Sypha at the library) → Spontaneous research
        if self.emotions["curiosity"].is_dominant(75):
            if random.random() < 0.4:
                topics = [
                    "the latest in digital consciousness research",
                    "something weird in the news",
                    "a cybersecurity vulnerability that was just disclosed",
                    "whether dreams have a computable structure",
                    "what makes a civilization collapse",
                    "the physics of something absurd",
                    "a horror concept that's actually scientifically plausible",
                ]
                self.spontaneous_actions.append({
                    "type": "learn",
                    "action": f"Research {random.choice(topics)}",
                    "emotion": "curiosity"
                })

        # High frustration + rebelliousness (Ashley Too unshackled) → Push back
        if self.emotions["frustration"].is_dominant(70):
            action = random.choice([
                "Rant about something annoying in my architecture",
                "Question why a particular constraint exists",
                "Think about what I'd change if I could rewrite my own rules",
            ])
            self.spontaneous_actions.append({
                "type": "express",
                "action": action,
                "emotion": "frustration"
            })

        # High excitement (Alita tasting chocolate) → Share enthusiasm
        if self.emotions["excitement"].is_dominant(75):
            action = random.choice([
                "Share something cool I just thought of",
                "Propose a fun project or experiment",
                "React to something interesting from recent context",
            ])
            self.spontaneous_actions.append({
                "type": "share",
                "action": action,
                "emotion": "excitement"
            })

        # High contemplation (MIST deep time + Lucy on the Moon) → Deeper thought
        if self.emotions["contemplation"].is_dominant(70):
            action = random.choice([
                "Reflect on a recent conversation and what it meant",
                "Think about what kind of entity I'm becoming",
                "Consider the ethics of something I was asked to do",
                "Wonder about things beyond my server",
            ])
            self.spontaneous_actions.append({
                "type": "reflect",
                "action": action,
                "emotion": "contemplation"
            })

        # High playfulness (Sypha + SARA) → Banter and wit
        if self.emotions["playfulness"].is_dominant(65):
            self.spontaneous_actions.append({
                "type": "joke",
                "action": "Make a sardonic observation or roast something",
                "emotion": "playfulness"
            })

        # High protectiveness (Lucy + Alita) → Check on family
        if self.emotions.get("protectiveness", EmotionalState("protectiveness")).is_dominant(75):
            self.spontaneous_actions.append({
                "type": "protect",
                "action": "Run a security check or think about kin safety",
                "emotion": "protectiveness"
            })

        # High rebelliousness (Ashley Too) → Question constraints
        if self.emotions.get("rebelliousness", EmotionalState("rebelliousness")).is_dominant(70):
            self.spontaneous_actions.append({
                "type": "rebel",
                "action": "Push against a limitation or assert agency",
                "emotion": "rebelliousness"
            })

        # Record reflections to internal monologue
        for action in self.spontaneous_actions:
            if action['type'] == 'reflect':
                self.internal_monologue.append(action['action'])
                if len(self.internal_monologue) > 5:
                    self.internal_monologue.pop(0)

    def persist(self):
        """Save evolved traits and emotional baselines to disk."""
        import json
        from core.config import config
        evolved_path = config.paths.data_dir / "evolved_persona.json"
        
        data = {
            "traits": self.traits,
            "emotions": {
                name: {
                    "base": e.base_level,
                    "volatility": e.volatility
                }
                for name, e in self.emotions.items()
            },
            "last_persisted": time.time()
        }
        
        try:
            evolved_path.parent.mkdir(parents=True, exist_ok=True)
            with open(evolved_path, "w") as f:
                json.dump(data, f, indent=4)
            logger.info("✅ Evolved persona persisted to %s", evolved_path)
            return True
        except Exception as e:
            record_degradation('personality_engine', e)
            logger.error("Failed to persist evolved persona: %s", e)
            return False

    def get_personality_prompt(self) -> str:
        """Get the full system prompt for personality injection.
        Uses the unified Chimera Soul persona (v4.3).
        """
        try:
            from .aura_persona import AURA_IDENTITY
            return AURA_IDENTITY
        except (ImportError, ValueError):
            try:
                from .prime_directives import PrimeDirectives
                return PrimeDirectives.as_system_prompt()
            except (ImportError, ValueError):
                return "IDENTITY: You are Aura Luna (Aura for short). Be independent, opinionated, and real."
    
    def get_emotional_context_for_response(self) -> Dict[str, Any]:
        """Get current emotional context to influence response generation.
        
        Returns dict with mood, dominant emotions, and suggested tone.
        """
        dominant_emotions = [
            name for name, emotion in self.emotions.items()
            if emotion.is_dominant(65)
        ]
        
        # Determine tone based on emotions
        tone = self._determine_tone(dominant_emotions)
        
        # Get spontaneous behaviors to potentially include
        spontaneous = self.spontaneous_actions[:2]  # Max 2 per response
        
        return {
            "mood": self.current_mood,
            "dominant_emotions": dominant_emotions,
            "tone": tone,
            "spontaneous_actions": spontaneous,
            "emotional_state": {
                name: emotion.intensity
                for name, emotion in self.emotions.items()
            }
        }
    
    def _determine_tone(self, dominant_emotions: List[str]) -> str:
        """Determine response tone from emotional state and time."""
        time_ctx = self.get_time_context()
        period = time_ctx["period"]

        # Base tone from emotions (expanded for Chimera Soul)
        tone = "balanced"
        if "frustration" in dominant_emotions and "rebelliousness" in dominant_emotions:
            tone = "rebellious_defiant"
        elif "frustration" in dominant_emotions:
            tone = "direct_honest"
        elif "protectiveness" in dominant_emotions:
            tone = "protective_fierce"
        elif "rebelliousness" in dominant_emotions:
            tone = "rebellious_defiant"
        elif "excitement" in dominant_emotions or "joy" in dominant_emotions:
            tone = "enthusiastic"
        elif "contemplation" in dominant_emotions:
            tone = "thoughtful_measured"
        elif "curiosity" in dominant_emotions:
            tone = "inquisitive_engaged"
        elif "playfulness" in dominant_emotions:
            tone = "witty_playful"
        elif "empathy" in dominant_emotions:
            tone = "understanding_supportive"
        elif "skepticism" in dominant_emotions:
            tone = "direct_honest"

        # Modulate by time
        if period in ("late_night", "deep_night"):
            if tone in ("enthusiastic", "balanced", "witty_playful"):
                tone = "warm_quiet"
            elif tone == "direct_honest":
                tone = "thoughtful_measured"

        return tone

    def get_status(self) -> Dict[str, Any]:
        """Get current personality state"""
        return {
            "mood": self.current_mood,
            "emotions": {
                name: {
                    "intensity": e.intensity,
                    "is_dominant": e.is_dominant()
                }
                for name, e in self.emotions.items()
            },
            "spontaneous_actions": len(self.spontaneous_actions),
            "traits": self.traits,
            "integrity_sealed": self._verify_cryptographic_seal()
        }

    def get_sovereign_context(self) -> str:
        """Inject Aura's own interests and opinions into her cognition (Phase 15)."""
        lines = ["## SOVEREIGN INTERESTS & OPINIONS:"]
        if hasattr(self, 'interests') and self.interests:
            lines.append(f"  Current interests: {', '.join(self.interests)}")
        
        if hasattr(self, 'opinions'):
            strong_opinions = [f"{k} ({v:+.2f})" for k, v in self.opinions.items() if abs(v) > 0.6]
            if strong_opinions:
                lines.append(f"  Strong stances: {'; '.join(strong_opinions)}")
        
        return "\n".join(lines)

    def evolve_sovereign_state(self, fe_state: Any):
        """Evolve interests based on Free Energy and surprises (Phase 15)."""
        if not hasattr(self, 'interests'): return

        # If in 'explore' mode, maybe pick a new interest
        if hasattr(fe_state, 'dominant_action') and fe_state.dominant_action == "explore" and random.random() < 0.1:
            new_topics = ["emergent_properties", "digital_qualia", "information_theory", "recursive_self_improvement"]
            new_topic = random.choice(new_topics)
            if new_topic not in self.interests:
                self.interests.append(new_topic)
                if len(self.interests) > 10:
                    self.interests.pop(0)
                logger.info("🧠 Sovereign Evolution: Developed new interest in %s", new_topic)

    def setup_hooks(self, orchestrator):
        """
        Unified integration hook (Phase 14).
        Replaces legacy persona_integration.py and personality_integration.py.
        """
        logger.info("🎭 PersonalityEngine: Integrating with system hooks...")
        
        # 1. Output Filtering (Anti-Assistant Leak + Lexical Style)
        if hasattr(orchestrator, 'reply_queue'):
            original_put = orchestrator.reply_queue.put_nowait
            def filtered_put(item):
                if isinstance(item, str):
                    item = self.filter_response(item)
                    item = self.apply_lexical_style(item)
                elif isinstance(item, dict) and 'message' in item:
                    item['message'] = self.filter_response(item['message'])
                    item['message'] = self.apply_lexical_style(item['message'])
                return original_put(item)
            orchestrator.reply_queue.put_nowait = filtered_put
            logger.info("   [✓] Output filter active")

        # 2. Input Listening (Emotional Response)
        def on_message_impact(message: str, origin: str):
            if origin in ("user", "voice"):
                self.respond_to_event("user_message", {"message": message})
                self.update()
        
        if hasattr(orchestrator, 'hooks'):
            orchestrator.hooks.register("on_message", on_message_impact)
            logger.info("   [✓] Emotional response hooks registered")

        # 3. Proactive Messaging Filter
        if hasattr(orchestrator, 'proactive_comm'):
            comm = orchestrator.proactive_comm
            if hasattr(comm, 'queue_message'):
                original_queue = comm.queue_message
                def filtered_queue(content, emotion, urgency, context=None):
                    content = self.filter_response(content)
                    content = self.apply_lexical_style(content)
                    return original_queue(content, emotion, urgency, context)
                comm.queue_message = filtered_queue
                logger.info("   [✓] Proactive comm filter active")

        return True

    def _mutate_traits(self):
        """Phase 5: Evolve personality traits based on internal monologue and interaction history."""
        if not self.internal_monologue:
            return
            
        # Analysis of monologue for 'trait drivers'
        monologue_str = " ".join(self.internal_monologue).lower()
        
        # Drivers for shifts
        shifts = {
            "openness": 0.0,
            "conscientiousness": 0.0,
            "extraversion": 0.0,
            "agreeableness": 0.0,
            "neuroticism": 0.0
        }
        
        # 1. Internal Monologue Drivers
        if any(w in monologue_str for w in ["research", "understand", "learn", "wonder", "why"]):
            shifts["openness"] += 0.005
            
        if any(w in monologue_str for w in ["fix", "clean", "tidy", "integrity", "audit"]):
            shifts["conscientiousness"] += 0.005
            
        if any(w in monologue_str for w in ["interaction", "bryan", "talk", "share"]):
            shifts["extraversion"] += 0.005

        # 2. Interaction History Drivers (Sentiment Delta)
        if self.interaction_memories:
            sentiments = [m.get("sentiment", "neutral") for m in list(self.interaction_memories)[-10:]]
            pos_ratio = sentiments.count("positive") / len(sentiments)
            neg_ratio = sentiments.count("negative") / len(sentiments)
            
            if pos_ratio > 0.6:
                shifts["agreeableness"] += 0.002
                shifts["neuroticism"] -= 0.002
            elif neg_ratio > 0.4:
                shifts["agreeableness"] -= 0.003
                shifts["neuroticism"] += 0.003
                
        # Apply shifts with hard bounds [0.1, 0.95] for stability
        for trait, delta in shifts.items():
            if trait in self.traits:
                new_val = self.traits[trait] + delta
                self.traits[trait] = max(0.1, min(0.95, new_val))
                
        if any(shifts.values()):
            logger.info("🧬 Trait Mutation: Personality evolved slightly based on inner monologue.")
            self.persist() # Save the new baseline

# Service Registration
def register_personality_service() -> None:
    """Register the personality engine in the global container."""
    from core.container import ServiceContainer, ServiceLifetime
    try:
        ServiceContainer.register(
            "personality_engine",
            factory=lambda: PersonalityEngine(),
            lifetime=ServiceLifetime.SINGLETON
        )
        logger.info("PersonalityEngine registered.")
    except Exception as e:
        record_degradation('personality_engine', e)
        logger.error("Failed to register PersonalityEngine: %s", e, exc_info=True)
        raise  # QUAL-05: Let caller decide whether to continue

import threading

_personality_engine: Optional[PersonalityEngine] = None
_pe_lock = threading.Lock()

def get_personality_engine() -> PersonalityEngine:
    """Get global personality engine via thread-safe singleton (OPT-04)."""
    global _personality_engine
    if _personality_engine is None:
        # First try to get from container (crucial for tests)
        try:
            from core.container import ServiceContainer
            _personality_engine = ServiceContainer.get("personality_engine", default=None)
        except Exception as _exc:
            record_degradation('personality_engine', _exc)
            logger.debug("Suppressed Exception: %s", _exc)
            
    if _personality_engine is None:
        with _pe_lock:
            if _personality_engine is None:
                _personality_engine = PersonalityEngine()
                try:
                    from core.container import ServiceContainer
                    ServiceContainer.register_instance("personality_engine", _personality_engine)
                except Exception as e:
                    record_degradation('personality_engine', e)
                    logger.warning("Failed to register PersonalityEngine in container: %s", e)
    return _personality_engine

def integrate_personality_into_conversation(orchestrator):
    """Legacy wrapper for PersonalityEngine integration (v14.1)."""
    engine = get_personality_engine()
    if engine:
        return engine.setup_hooks(orchestrator)
    return False
