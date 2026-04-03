import time
import logging
import hashlib
from dataclasses import dataclass
from typing import Optional, Any
from ..state.aura_state import AuraState
from ..container import ServiceContainer
from core.brain.llm.llm_router import LLMTier

logger = logging.getLogger("Aura.PredictiveEngine")

@dataclass
class Prediction:
    content: str           # What Aura expects to happen/be said
    confidence: float      # 0.0 to 1.0
    generated_at: float
    context_hash: str      # Hash of state at prediction time

@dataclass  
class PredictionError:
    prediction: Prediction
    actual: str
    error_magnitude: float  # How wrong was the prediction
    surprise_signal: float  # Normalized surprise — drives affect update

class PredictiveEngine:
    """
    Before each incoming input is processed, Aura generates a prediction
    of what she expects. After processing, the error between prediction
    and reality becomes a genuine cognitive signal.
    """

    def __init__(self):
        self.router = None

    def _get_router(self):
        if self.router is None:
            self.router = ServiceContainer.get("llm_router", default=None)
        return self.router

    async def predict(self, state: AuraState, **kwargs) -> Prediction:
        """Generate prediction before perceiving next input."""
        router = self._get_router()
        if not router:
            return Prediction("Unknown", 0.5, time.time(), self._hash_context(state))

        prompt = f"""Given your current state and the conversation so far, 
what do you predict the human will say or do next? 
Be specific. This is a genuine prediction, not a hedge.

Your current affect: valence={state.affect.valence:.2f}, 
curiosity={state.affect.curiosity:.2f}
Recent conversation: {self._recent_context(state)}

Prediction:"""
        
        # Use FAST mode for predictions to keep latency low
        try:
            prediction_text = await router.think(
                prompt,
                priority=kwargs.get("priority", 0.5),
                is_background=kwargs.get("is_background", True),
                prefer_tier=kwargs.get("prefer_tier", LLMTier.TERTIARY)
            )
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            prediction_text = "I predict a continuation of the current thread."
        
        # Estimate confidence from affect — high curiosity = high uncertainty
        confidence = 1.0 - (state.affect.curiosity * 0.5)
        
        return Prediction(
            content=(prediction_text or "").strip(),
            confidence=confidence,
            generated_at=time.time(),
            context_hash=self._hash_context(state),
        )

    async def evaluate(
        self, 
        prediction: Prediction, 
        actual: str,
        state: AuraState
    ) -> PredictionError:
        """Compare prediction to reality. Generate error signal."""
        router = self._get_router()
        if not router:
            return PredictionError(prediction, actual, 0.5, 0.25)

        # Ask the LLM to evaluate the prediction error
        # (semantic similarity, not string matching)
        eval_prompt = f"""Compare these two:
Predicted: {prediction.content}
Actual: {actual}

Rate how wrong the prediction was on a scale 0.0 (perfect) to 1.0 (completely wrong).
Respond with only a float."""
        
        try:
            error_text = await router.think(
                eval_prompt,
                priority=0.5,
                is_background=True,
                prefer_tier=LLMTier.TERTIARY
            )
            # Use regex to find float in case of fluff
            import re
            match = re.search(r"(\d+\.\d+)", error_text)
            if match:
                error_magnitude = float(match.group(1))
            else:
                error_magnitude = float(error_text.strip())
        except Exception:
            error_magnitude = 0.5
        
        # Surprise = error weighted by confidence
        # High confidence + high error = maximum surprise
        surprise_signal = error_magnitude * prediction.confidence
        
        # CROSSWIRE-03: Feed surprise signal to AffectStateManager
        affect = ServiceContainer.get("affect_engine", default=None)
        if affect and surprise_signal > 0.0:
            import asyncio
            stimulus = "intrigue" if surprise_signal > 0.5 else "calm"
            intensity = surprise_signal * 10.0
            
            # Audit Fix: Loop-safe task creation
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(affect.apply_stimulus(stimulus, intensity))
            except RuntimeError:
                # Fallback for sync threads or non-running loops
                try:
                    try:
                        loop = asyncio.get_running_loop()
                    except RuntimeError:
                        loop = None
                    if loop and loop.is_running():
                        loop.call_soon_threadsafe(
                            lambda: asyncio.create_task(affect.apply_stimulus(stimulus, intensity))
                        )
                except Exception as _exc:
                    logger.debug("Suppressed Exception: %s", _exc)

        return PredictionError(
            prediction=prediction,
            actual=actual,
            error_magnitude=error_magnitude,
            surprise_signal=surprise_signal,
        )

    def _recent_context(self, state: AuraState) -> str:
        recent = state.cognition.working_memory[-3:]
        return " | ".join(
            f"{m.get('role', 'unknown')}: {m.get('content', '')[:100]}" for m in recent
        )

    def _hash_context(self, state: AuraState) -> str:
        content = str(state.version) + state.identity.current_narrative[:100]
        return hashlib.md5(content.encode()).hexdigest()[:8]
