"""core/pneuma/pneuma.py
PNEUMA — Precision-weighted Neural Epistemic Unified Manifold Architecture.

Integration layer that wires all 5 PNEUMA components into Aura's inference
pipeline. PNEUMA runs as a background async loop that:

  1. Steps the FHN oscillator on every tick
  2. Advances the Neural ODE belief flow
  3. Registers each belief snapshot with the IGTracker
  4. Pushes belief points to TopologicalMemory
  5. Provides the FreeEnergyOracle for response scoring

Public API (used by InferenceGate):
    pneuma.get_context_block()      → inject into system prompt
    pneuma.on_evidence(text)        → push new evidence into belief flow
    pneuma.on_affect_change(v, a)   → push affect perturbation
    pneuma.score_response(text)     → get EFE score for a response
    pneuma.get_llm_temperature()    → precision-weighted temperature
"""

import asyncio
import logging
import time
from typing import Optional

import numpy as np

from .precision_engine import PrecisionEngine, PrecisionConfig
from .neural_ode_flow import NeuralODEFlow
from .information_geometric_tracker import InformationGeometricTracker
from .topological_memory import TopologicalMemoryEngine
from .free_energy_oracle import FreeEnergyOracle

logger = logging.getLogger("PNEUMA")

BELIEF_DIM = 64


class PNEUMA:
    """Unified PNEUMA active inference engine."""

    def __init__(self):
        self.precision = PrecisionEngine(PrecisionConfig(n_heads=32, fhn_dt=0.05))
        self.ode_flow = NeuralODEFlow(dim=BELIEF_DIM)
        self.ig_tracker = InformationGeometricTracker(dim=BELIEF_DIM)
        self.topo_memory = TopologicalMemoryEngine(dim=BELIEF_DIM, window_size=50, update_every=10)
        self.feo = FreeEnergyOracle(epistemic_weight=0.4, pragmatic_weight=0.4, structural_weight=0.2)

        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._tick_interval = 0.5   # seconds between background ticks
        self._last_tick = 0.0
        self._cached_context_block = ""
        self._cached_context_at = 0.0
        self._context_cache_ttl_s = 1.0
        logger.info("PNEUMA online — all 5 layers initialized.")

    async def start(self):
        """Start the PNEUMA background loop."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._loop(), name="PNEUMA.loop")
        logger.info("PNEUMA background loop started.")

    async def stop(self):
        """Stop the PNEUMA background loop."""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
        logger.info("PNEUMA stopped.")

    async def _loop(self):
        """Background tick loop: advance FHN + ODE + IGTracker + Topo."""
        while self._running:
            try:
                await asyncio.sleep(self._tick_interval)
                now = time.time()
                dt = min(2.0, now - self._last_tick) if self._last_tick else 0.5
                self._last_tick = now

                # 1. Advance FHN precision
                self.precision.step()

                # 2. Advance ODE belief flow
                belief_state = await asyncio.to_thread(self.ode_flow.step, dt)

                # 3. Register with IGTracker
                self.ig_tracker.update(belief_state.vector, source="ode_tick")

                # 4. Push to topological memory
                self.topo_memory.push(belief_state.vector)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug("PNEUMA loop error: %s", e)

    # ── Public API ────────────────────────────────────────────────────────────

    def on_evidence(self, text: str, weight: float = 0.25):
        """Inject new textual evidence into the belief flow.

        Text is encoded as a simple bag-of-words hash embedding.
        """
        embedding = self._text_to_embedding(text)
        self.ode_flow.inject_evidence(embedding, weight=weight, source="evidence")
        self.ig_tracker.update(self.ode_flow.current_belief.vector, source="evidence")
        self._cached_context_at = 0.0

    def on_affect_change(self, valence: float, arousal: float):
        """Perturb belief flow with affective state change."""
        self.ode_flow.inject_affect(valence, arousal)
        self.precision.fhn.step(i_ext=0.5 + 0.5 * arousal)
        self._cached_context_at = 0.0

    def score_response(self, text: str) -> float:
        """Return EFE score for a candidate response (lower = better)."""
        return self.feo.quick_score_text(
            text,
            self.ode_flow.current_belief.vector,
            ig_stability=self.ig_tracker.stability,
        )

    def get_llm_temperature(self, base_temp: float = 0.72) -> float:
        """Get precision-weighted LLM temperature."""
        precision_temp = self.precision.get_temperature()
        feo_temp = self.feo.get_preferred_temperature(base_temp)
        # Blend: 60% precision engine, 40% FEO
        return round(0.6 * precision_temp + 0.4 * feo_temp, 3)

    def get_context_block(self) -> str:
        """Format PNEUMA state for injection into LLM system prompt."""
        now = time.time()
        if self._cached_context_block and (now - self._cached_context_at) < self._context_cache_ttl_s:
            return self._cached_context_block
        prec = self.precision.get_state_dict()
        ode = self.ode_flow.get_state_dict()
        igt = self.ig_tracker.get_state_dict()
        topo = self.topo_memory.get_state_dict()

        lines = [
            "## PNEUMA (Active Inference State)",
            f"FHN v={prec['fhn_v']:.3f} w={prec['fhn_w']:.3f} | arousal={prec['arousal']:.3f} | temp={prec['temperature']:.3f}",
            f"Belief norm={ode['belief_norm']:.3f} | confidence={ode['belief_confidence']:.3f}",
            f"IGT stability={igt['stability']:.3f} | drift={igt['is_drifting']}",
            f"Topology: attractors={topo['attractor_count']} complexity={topo['topological_complexity']:.3f}",
        ]
        block = "\n".join(lines)
        self._cached_context_block = block
        self._cached_context_at = now
        return block

    def _text_to_embedding(self, text: str) -> np.ndarray:
        """Encode text as a fixed-dim float32 vector via hash projection."""
        embedding = np.zeros(BELIEF_DIM, dtype=np.float32)
        words = text.lower().split()
        for word in words[:100]:
            h = hash(word) % (2 ** 31)
            idx = h % BELIEF_DIM
            embedding[idx] += 1.0 / max(1, len(words))
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding /= norm
        return embedding

    def get_state_dict(self) -> dict:
        return {
            "precision": self.precision.get_state_dict(),
            "ode_flow": self.ode_flow.get_state_dict(),
            "ig_tracker": self.ig_tracker.get_state_dict(),
            "topo_memory": self.topo_memory.get_state_dict(),
            "feo": self.feo.get_state_dict(),
        }


# ── Singleton ─────────────────────────────────────────────────────────────────

_pneuma: Optional[PNEUMA] = None


def get_pneuma() -> PNEUMA:
    global _pneuma
    if _pneuma is None:
        _pneuma = PNEUMA()
    return _pneuma
