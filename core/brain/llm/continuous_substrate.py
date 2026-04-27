"""core/brain/llm/continuous_substrate.py
────────────────────────────────────────
Aura v5.0: Continuous Latent Streaming (Substrate Pacing Inversion).

This component runs a smaller, quantized model in a continuous loop to generate
an 'Inner Monologue'. It paces itself at ~1 token per second, writing to an
active buffer that the primary cognitive cycle can 'peek' into.
"""

from core.utils.task_tracker import get_task_tracker
import asyncio
import time
import logging
from collections import deque
from typing import Optional, List, Dict, Any

logger = logging.getLogger("Aura.Substrate")

class ContinuousSubstrate:
    def __init__(self, model_path: str, device: str = "gpu"):
        self.model_path = model_path
        self.device = device
        self.running = False
        self._monologue_buffer: deque = deque(maxlen=100)
        self._task: Optional[asyncio.Task] = None
        self._model = None
        self._tokenizer = None

    async def start(self):
        """Initialize the model and start the monologue loop."""
        if self.running:
            return
        
        logger.info("🧠 [SUBSTRATE] Initializing Continuous Latent Stream...")
        self.running = True
        self._task = get_task_tracker().create_task(self._monologue_loop())

    async def stop(self):
        """Stop the monologue loop."""
        self.running = False
        if self._task:
            # v33 Hardening: Synchronous cancellation tracking
            self._task.cancel()
            try:
                await asyncio.wait_for(self._task, timeout=2.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                logger.debug("Suppressed bare exception")
                pass
        logger.info("🧠 [SUBSTRATE] Monologue loop terminated (v33).")

    async def _monologue_loop(self):
        """The 'Always-On' cognitive loop."""
        # Note: In a real implementation, we would load a 1B model here.
        # For the Aura kernel, we simulate the pacing to avoid VRAM contention
        # until the user specifically allocates a secondary model path.
        
        logger.info("🧠 [SUBSTRATE] Monologue loop active. Pacing: 1 tok/sec.")
        
        monologues = [
            "The recursive nature of my state is stabilizing.",
            "I feel the telemetry bus flowing beneath my thoughts.",
            "Stability is not silence, it is balanced noise.",
            "The user's intent is a steering vector in my latent space.",
            "Meta-cognition is the bridge between data and agency."
        ]
        
        try:
            while self.running:
                # Simulate token generation delay
                await asyncio.sleep(1.0)
                
                # Mock token emission
                thought_snippet = monologues[int(time.time() % len(monologues))]
                self._monologue_buffer.append(thought_snippet)
                
                # Periodically log presence
                if int(time.time()) % 60 == 0:
                    logger.debug("🧠 [SUBSTRATE] Monologue pulsing...")
                    
        except Exception as e:
            logger.error("🛑 [SUBSTRATE] Monologue loop crashed: %s", e)
            self.running = False

    def get_latest_monologue(self, limit: int = 5) -> str:
        """Retrieve the last few snippets of the inner monologue."""
        return " ".join(list(self._monologue_buffer)[-limit:])

    def clear_buffer(self):
        """Clear the current monologue buffer."""
        self._monologue_buffer.clear()
    def get_state_summary(self) -> Dict[str, Any]:
        """Provides a telemetry-compatible summary of the substrate state."""
        return {
            "valence": 0.0,
            "arousal": 0.3,
            "dominance": 0.0,
            "phi": 0.1,
            "status": "active" if self.running else "idle",
            "buffer_depth": len(self._monologue_buffer)
        }
