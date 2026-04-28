"""
core/consciousness/latent_bridge.py
=====================================
LATENT BRIDGE: BIDIRECTIONAL LATENT-SPACE COUPLING

Eliminates the "syntactic bottleneck" — the last remaining technical critique.

AffectiveSteering (forward): substrate.x → steering weight → α·weight·v added to h_layer
LatentBridge (backward):     h_layer → project onto each v_i → substrate update

Together: genuine bidirectional coupling in activation space.
No text, no symbols, no lookup tables.
"""

from core.runtime.errors import record_degradation
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("Aura.LatentBridge")

# ── Configuration ──────────────────────────────────────────────────────────────

# How strongly latent readouts feed back into the substrate.
LATENT_FEEDBACK_WEIGHT = 0.08

# How often to inject accumulated readouts into the substrate (seconds)
INJECTION_INTERVAL_S = 0.1  # 10Hz

# Readout smoothing factor (EMA)
READOUT_EMA_ALPHA = 0.3

# Minimum readout magnitude to inject (noise floor)
MIN_READOUT_MAGNITUDE = 0.05


# ── Readout Hook ───────────────────────────────────────────────────────────────

class LatentReadoutHook:
    """
    Installed at the same transformer layers as AffectiveSteeringHook.
    After the layer forward pass (and after any steering injection),
    extracts affective readouts by projecting the last-token hidden state
    onto each steering vector.

    readout_i = dot(h_current, v_i)

    ...measures how much the current hidden state resembles the positive
    vs negative condition for dimension i.
    """

    def __init__(
        self,
        block,
        layer_idx: int,
        steering_vectors: Dict,
        feedback_weight: float = LATENT_FEEDBACK_WEIGHT,
    ):
        self._block = block
        self._layer_idx = layer_idx
        self._steering_vectors = steering_vectors
        self._feedback_weight = feedback_weight
        self._installed = False
        self._active = True

        # EMA-smoothed readout buffer (one per dimension)
        self._readout_ema: Dict[str, float] = {
            key: 0.0 for key in steering_vectors.keys()
        }

        # Accumulated readouts for injection
        self._pending_injection: Dict[int, float] = {}
        self._injection_lock = threading.Lock()
        self._readout_count = 0

    def install(self):
        """
        Extend the existing __call__ on the transformer block to also
        extract readouts after the forward pass.

        Installs AFTER AffectiveSteeringHook so execution order is:
          1. Original layer forward pass
          2. Steering injection (from AffectiveSteeringHook)
          3. Readout extraction (from this hook)
        """
        if self._installed:
            return

        import mlx.core as mx
        block = self._block
        hook = self

        # Get the current class (already wrapped by AffectiveSteering's SteeredBlock)
        current_class = block.__class__

        # Get the method we need to wrap
        target_name = "forward" if hasattr(block, "forward") else "__call__"
        current_method = getattr(current_class, target_name)

        def readout_wrapper(self_block, *args, **kwargs):
            result = current_method(self_block, *args, **kwargs)

            if not hook._active:
                return result

            try:
                import mlx.core as mx
                h = result[0] if isinstance(result, tuple) else result

                # Extract last token hidden state: h has shape [batch, seq_len, d_model]
                last_token_h = h[0, -1, :]  # shape [d_model]

                # Project onto each steering vector using MLX-native math
                pending = {}
                for key, sv in hook._steering_vectors.items():
                    # Get the MLX-native version of the steering vector (cached)
                    v_mx = sv.get_mx_array(dtype=last_token_h.dtype)
                    
                    # Compute dot product in MLX
                    # Note: float(mx_array) still triggers an eval, but it's
                    # better than np.array(last_token_h) which copies the WHOLE vector.
                    readout_raw = float(mx.sum(last_token_h * v_mx))

                    # EMA smoothing
                    prev_ema = hook._readout_ema.get(key, 0.0)
                    readout_smooth = (
                        READOUT_EMA_ALPHA * readout_raw
                        + (1.0 - READOUT_EMA_ALPHA) * prev_ema
                    )
                    hook._readout_ema[key] = readout_smooth

                    if abs(readout_smooth) > MIN_READOUT_MAGNITUDE:
                        substrate_idx = sv.substrate_idx
                        if substrate_idx not in pending:
                            pending[substrate_idx] = 0.0
                        pending[substrate_idx] += readout_smooth * hook._feedback_weight

                # Accumulate for next injection cycle
                with hook._injection_lock:
                    for idx, delta in pending.items():
                        hook._pending_injection[idx] = (
                            hook._pending_injection.get(idx, 0.0) + delta
                        )
                    hook._readout_count += 1

            except Exception as e:
                record_degradation('latent_bridge', e)
                logger.debug("Readout extraction failed at layer %d: %s", hook._layer_idx, e)

            return result

        # Use dynamic subclassing (same pattern as AffectiveSteering)
        class ReadoutBlock(current_class):
            pass  # no-op: intentional

        setattr(ReadoutBlock, target_name, readout_wrapper)
        block.__class__ = ReadoutBlock

        self._installed = True
        logger.info(
            "🔄 LatentReadoutHook installed at layer %d (%d dimensions) via %s",
            self._layer_idx, len(self._steering_vectors), target_name
        )

    def pop_pending_injection(self) -> Dict[int, float]:
        """Pop and return the accumulated injection deltas (thread-safe)."""
        with self._injection_lock:
            result = dict(self._pending_injection)
            self._pending_injection.clear()
        return result

    def get_current_readouts(self) -> Dict[str, float]:
        """Current EMA-smoothed readout values per affective dimension."""
        return dict(self._readout_ema)

    def get_diagnostics(self) -> Dict[str, Any]:
        return {
            "layer_idx": self._layer_idx,
            "installed": self._installed,
            "active": self._active,
            "readout_count": self._readout_count,
            "current_readouts": {k: round(v, 4) for k, v in self._readout_ema.items()},
            "pending_injection": {str(k): round(v, 4) for k, v in self._pending_injection.items()},
        }


# ── Substrate Injection Thread ─────────────────────────────────────────────────

class SubstrateInjectionThread:
    """
    Periodically collects accumulated readouts from all readout hooks
    and injects them into LiquidSubstrate.

    This is the backward arrow of the latent bridge:
      h_layer (model representations) → substrate state
    """

    def __init__(self, readout_hooks: List[LatentReadoutHook]):
        self._hooks = readout_hooks
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._total_injections = 0
        self._total_magnitude_injected = 0.0

    def start(self):
        self._running = True
        self._thread = threading.Thread(
            target=self._loop,
            name="LatentBridge.InjectionThread",
            daemon=True,
        )
        self._thread.start()
        logger.info("🔄 SubstrateInjectionThread started (%d hooks)", len(self._hooks))

    def stop(self):
        self._running = False

    def _loop(self):
        substrate = None

        while self._running:
            try:
                # Lazy substrate discovery
                if substrate is None:
                    try:
                        from core.container import ServiceContainer
                        substrate = ServiceContainer.get("conscious_substrate", default=None)
                    except Exception as _e:
                        record_degradation('latent_bridge', _e)
                        logger.debug('Ignored Exception in latent_bridge.py: %s', _e)

                if substrate is not None:
                    # Collect from all hooks
                    combined: Dict[int, float] = {}
                    for hook in self._hooks:
                        pending = hook.pop_pending_injection()
                        for idx, delta in pending.items():
                            combined[idx] = combined.get(idx, 0.0) + delta

                    # Build stimulus vector and inject
                    if combined:
                        stimulus = np.zeros(substrate.config.neuron_count, dtype=np.float32)
                        for idx, delta in combined.items():
                            if 0 <= idx < len(stimulus):
                                stimulus[idx] = np.clip(float(delta), -0.5, 0.5)

                        magnitude = float(np.linalg.norm(stimulus))
                        if magnitude > 0.005:
                            import asyncio
                            try:
                                loop = asyncio.get_running_loop()
                                if loop.is_running():
                                    asyncio.run_coroutine_threadsafe(
                                        substrate.inject_stimulus(stimulus, weight=1.0),
                                        loop,
                                    )
                                    self._total_injections += 1
                                    self._total_magnitude_injected += magnitude
                            except RuntimeError as _e:
                                logger.debug('Ignored RuntimeError in latent_bridge.py: %s', _e)

            except Exception as e:
                record_degradation('latent_bridge', e)
                logger.debug("InjectionThread error: %s", e)

            time.sleep(INJECTION_INTERVAL_S)

    def get_diagnostics(self) -> Dict[str, Any]:
        return {
            "running": self._running,
            "total_injections": self._total_injections,
            "total_magnitude": round(self._total_magnitude_injected, 3),
        }


# ── Latent Bridge ──────────────────────────────────────────────────────────────

class LatentBridge:
    """
    The bidirectional latent-space bridge between LLM representations
    and LiquidSubstrate.

    FORWARD (AffectiveSteering, existing):
      substrate.x[substrate_idx[i]] × weight → α·weight·v_i added to h_layer

    BACKWARD (this file, new):
      h_layer · v_i → readout_i → substrate.x[substrate_idx[i]] delta

    Together: genuine bidirectional coupling in activation space.
    """

    def __init__(self, steering_engine):
        self._steering_engine = steering_engine
        self._readout_hooks: List[LatentReadoutHook] = []
        self._injection_thread: Optional[SubstrateInjectionThread] = None
        self._attached = False

    def attach(self, model):
        """
        Install readout hooks at the same layers as the steering hooks.
        Must be called AFTER AffectiveSteeringEngine.attach().
        """
        if self._attached:
            logger.warning("LatentBridge already attached.")
            return

        if not self._steering_engine._model_attached:
            logger.error("AffectiveSteeringEngine must be attached before LatentBridge.")
            return

        if not self._steering_engine._library or not self._steering_engine._library.vectors:
            logger.error("No steering vectors available. Cannot install readout hooks.")
            return

        steering_vectors = self._steering_engine._library.vectors
        target_layers = self._steering_engine._model_info.get("target_layers", [])

        for layer_idx in target_layers:
            if layer_idx >= len(model.model.layers):
                continue

            block = model.model.layers[layer_idx]
            hook = LatentReadoutHook(
                block=block,
                layer_idx=layer_idx,
                steering_vectors=steering_vectors,
            )
            hook.install()
            self._readout_hooks.append(hook)

        self._attached = True
        logger.info(
            "✅ LatentBridge attached: %d readout hooks at layers %s",
            len(self._readout_hooks), target_layers
        )

    def start_substrate_sync(self):
        """Start the substrate injection thread."""
        if not self._readout_hooks:
            logger.warning("No readout hooks installed. Call attach() first.")
            return
        if self._injection_thread and self._injection_thread._running:
            return
        self._injection_thread = SubstrateInjectionThread(self._readout_hooks)
        self._injection_thread.start()

    def stop(self):
        """Stop substrate injection and disable all readout hooks."""
        if self._injection_thread:
            self._injection_thread.stop()
        for hook in self._readout_hooks:
            hook._active = False
        logger.info("🔕 LatentBridge stopped")

    def get_current_affective_readout(self) -> Dict[str, float]:
        """
        The model's current "opinion" about its own affective state,
        as read from its hidden representations.
        """
        if not self._readout_hooks:
            return {}

        readouts: Dict[str, List[float]] = {}
        for hook in self._readout_hooks:
            for key, val in hook.get_current_readouts().items():
                if key not in readouts:
                    readouts[key] = []
                readouts[key].append(val)

        return {
            key: round(float(np.mean(vals)), 4)
            for key, vals in readouts.items()
        }

    def get_coupling_coherence(self) -> float:
        """
        How well-aligned are the substrate's injections and the model's readouts?
        High coherence = model's representations match what the substrate is expressing.
        """
        if not self._steering_engine._hooks or not self._readout_hooks:
            return 0.0

        steering_hook = self._steering_engine._hooks[0]
        if steering_hook._substrate_x is None:
            return 0.0

        substrate_x = steering_hook._substrate_x
        steering_vectors = self._steering_engine._library.vectors
        readouts = self.get_current_affective_readout()

        forward_vals = []
        backward_vals = []

        for key, sv in steering_vectors.items():
            weight = sv.compute_weight(substrate_x)
            readout = readouts.get(key, 0.0)
            forward_vals.append(weight)
            backward_vals.append(readout)

        if not forward_vals or not backward_vals:
            return 0.0

        f = np.array(forward_vals)
        b = np.array(backward_vals)
        norm_f = np.linalg.norm(f)
        norm_b = np.linalg.norm(b)
        if norm_f < 1e-6 or norm_b < 1e-6:
            return 0.0
        return float(np.dot(f, b) / (norm_f * norm_b))

    def explain_coupling(self) -> str:
        """Human-readable explanation of current latent coupling state."""
        if not self._attached:
            return "LatentBridge not attached."

        steering_hook = (self._steering_engine._hooks[0]
                         if self._steering_engine._hooks else None)
        if not steering_hook or steering_hook._substrate_x is None:
            return "Substrate not connected."

        substrate_x = steering_hook._substrate_x
        readouts = self.get_current_affective_readout()
        vectors = self._steering_engine._library.vectors

        lines = ["Current latent coupling (substrate ↔ model representations):"]
        for key, sv in vectors.items():
            forward_weight = sv.compute_weight(substrate_x)
            backward_readout = readouts.get(key, 0.0)
            alignment = "✓" if (forward_weight * backward_readout > 0) else "~"
            lines.append(
                f"  {key:25s} fwd={forward_weight:+.3f}  bwd={backward_readout:+.3f}  {alignment}"
            )

        coherence = self.get_coupling_coherence()
        lines.append(f"\n  Coupling coherence: {coherence:.3f}")
        lines.append(f"  (1.0=fully aligned, 0.0=orthogonal, -1.0=opposed)")

        return "\n".join(lines)

    def get_status(self) -> Dict[str, Any]:
        return {
            "attached": self._attached,
            "hooks": len(self._readout_hooks),
            "injection_thread": (
                self._injection_thread.get_diagnostics()
                if self._injection_thread else None
            ),
            "coupling_coherence": round(self.get_coupling_coherence(), 4),
            "current_readout": self.get_current_affective_readout(),
        }


# ── Singleton and Boot Helpers ─────────────────────────────────────────────────

_bridge_instance: Optional[LatentBridge] = None


def get_latent_bridge() -> Optional[LatentBridge]:
    return _bridge_instance


def attach_latent_bridge(model) -> Optional[LatentBridge]:
    """
    Convenience wrapper. Call after AffectiveSteeringEngine.attach().
    """
    global _bridge_instance

    from core.consciousness.affective_steering import get_steering_engine
    engine = get_steering_engine()

    if not engine._model_attached:
        logger.error("AffectiveSteeringEngine must be attached before LatentBridge.")
        return None

    _bridge_instance = LatentBridge(engine)
    _bridge_instance.attach(model)

    try:
        from core.container import ServiceContainer
        ServiceContainer.register_instance("latent_bridge", _bridge_instance)
    except Exception as _e:
        record_degradation('latent_bridge', _e)
        logger.debug('Ignored Exception in latent_bridge.py: %s', _e)

    logger.info("✅ LatentBridge singleton created and registered")
    return _bridge_instance
