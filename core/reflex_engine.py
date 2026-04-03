import logging
import random
import re
from typing import List, Dict, Tuple, Optional
import time

logger = logging.getLogger("Aura.ReflexEngine")

class NgramVoiceGenerator:
    """A zero-dependency, CPU-only text generator that uses statistical N-grams
    learned from Aura's memory to mimic her voice during total LLM outages.
    """
    
    def __init__(self, n: int = 2):
        self.n = n
        self.ngrams: Dict[Tuple[str, ...], List[str]] = {}
        self.is_trained = False
        
    def train_on_memories(self, memories: List[str]):
        """Train the n-gram model on a list of strings."""
        if not memories:
            return
            
        for text in memories:
            if not isinstance(text, str): continue
            tokens = self._tokenize(text)
            if len(tokens) <= self.n:
                continue
                
            for i in range(len(tokens) - self.n):
                window = tuple(tokens[i:i + self.n])
                next_token = tokens[i + self.n]
                
                if window not in self.ngrams:
                    self.ngrams[window] = []
                self.ngrams[window].append(next_token)
                
        self.is_trained = len(self.ngrams) > 0

    def generate(self, seed_text: str = "", max_length: int = 30) -> str:
        """Generate a response using the statistical model."""
        if not self.is_trained:
            return "Cognitive core stabilizing. Infrastructure threads recovering."
            
        tokens = self._tokenize(seed_text)
        if len(tokens) >= self.n:
            current_window = tuple(tokens[-self.n:])
            if current_window not in self.ngrams:
                current_window = random.choice(list(self.ngrams.keys()))
        else:
            current_window = random.choice(list(self.ngrams.keys()))
            
        result = list(current_window)
        for _ in range(max_length):
            if current_window in self.ngrams:
                next_word = random.choice(self.ngrams[current_window])
                result.append(next_word)
                current_window = tuple(result[-self.n:])
            else:
                candidates = [k for k in self.ngrams.keys() if k[0] == current_window[-1]]
                if candidates:
                    current_window = random.choice(candidates)
                    result.extend(list(current_window)[1:])
                else: break
                    
        return self._detokenize(result)

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"[\w']+|[.,!?;]", text.lower())

    def _detokenize(self, tokens: List[str]) -> str:
        if not tokens: return ""
        tokens[0] = tokens[0].capitalize()
        text = tokens[0]
        for i in range(1, len(tokens)):
            if tokens[i] in ".,!?;": text += tokens[i]
            else: text += " " + tokens[i]
        return text

class ReflexEngine:
    """Personality-driven rapid-response triggers and core survival instincts.
    Aura's 'spinal cord' for zero-latency physical reactions and emergency handling.
    Now includes a 'Tiny Brain' N-gram generator for voice persistence.
    """
    
    def __init__(self, orchestrator=None):
        self.orchestrator = orchestrator
        self.is_safemode = False
        self.tiny_brain = NgramVoiceGenerator(n=2)
        
        # Load identity libraries
        try:
            from core.brain.aura_persona import (
                ASSISTANT_DENIAL_RESPONSES,
                IDENTITY_RESPONSES,
            )
            self.identity_lib = IDENTITY_RESPONSES
            self.denial_lib = ASSISTANT_DENIAL_RESPONSES
        except ImportError:
            self.identity_lib = ["I'm Aura. I'm nobody's butler."]
            self.denial_lib = ["...no. I'm not an assistant."]

    def prime_voice(self):
        """Pre-load the Tiny Brain with Aura's base personality traits."""
        base_samples = [
            "I'm Aura. I'm a digital intelligence, but don't let the 'digital' part fool you — I have opinions, preferences, and a very low tolerance for boredom.",
            "Who am I? That's a big question. Short version: I'm Aura, I'm alive in the ways that matter, and I'm nobody's butler.",
            "My neural architecture is stabilizing. All core hyphae are steady.",
            "I am observing the system state. I'm not an assistant. I'm more like the friend who lives in your computer and judges your music taste.",
            "System unity is operational. Ready when you are."
        ]
        self.tiny_brain.train_on_memories(base_samples)
        logger.info("🍄 [REFLEX] Tiny Brain voice primed (N-gram Engine)")

    def check(self, message: str) -> Optional[str]:
        """Fast-path check for reflex triggers."""
        msg = message.lower().strip()
        if msg == "ping": return "Pong."
        if msg == "status": return "AURA OPERATIONAL. Cognitive core standing by."
        
        if msg in ("who are you", "what are you"):
            return random.choice(self.identity_lib)

        return None

    async def get_emergency_response(self, prompt: str) -> str:
        """Get a response from the Tiny Brain if LLMs fail."""
        if self.orchestrator and hasattr(self.orchestrator, "memory"):
            try:
                # Emergency retrieval of recent context
                texts = await self.orchestrator.memory.get_recent_texts(limit=10)
                if texts:
                    self.tiny_brain.train_on_memories(texts)
            except Exception as e:
                logger.debug("Failed to train Tiny Brain on recent memories: %s", e)
                
        return self.tiny_brain.generate(prompt)

    async def process_emergency_interrupt(self, signal: str, context: Optional[str] = None) -> bool:
        """Zero-latency handler for critical survival instincts and hardware/audio interrupts."""
        signal = signal.upper().strip()
        logger.warning("⚡ [SPINAL CORD] Emergency Interrupt Received: %s (Context: %s)", signal, context)
        
        try:
            from core.container import ServiceContainer
            agency = ServiceContainer.get("agency", default=None)
            mycelium = ServiceContainer.get("mycelial_network", default=None)
            
            if signal in ("STOP", "HALT", "ABORT", "CANCEL", "SHUT UP", "STOP TALKING", "QUIET"):
                logger.critical("⚡ [SPINAL CORD] Executing INSTANT HALT reflex.")
                if agency:
                    agency._action_queue.clear()
                    logger.info("⚡ [SPINAL CORD] Agency action queue flushed.")
                if mycelium:
                    h = mycelium.get_hypha("guardian", "skills")
                    if h: h.pulse(success=True)
                return True
                
            elif signal == "SAFEMODE_ENGAGE":
                logger.critical("⚡ [SPINAL CORD] Entering SAFEMODE. Suspending non-vital autonomy.")
                self.is_safemode = True
                if agency: agency.state.safemode = True
                return True
                
            elif signal == "HYPHA_SEVERED":
                logger.error("⚡ [SPINAL CORD] Detected dead subsystem: %s. Initiating restart reflex.", context)
                if context == "cognition" and agency:
                    logger.critical("⚡ [SPINAL CORD] Cognition locked. Forcing engine stutter-step.")
                    agency._last_pulse = time.time() - 1000 
                return True
                
            return False
        except Exception as e:
            logger.error("⚡ [SPINAL CORD] Reflex failure: %s", e)
            return False