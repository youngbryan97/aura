"""
core/consciousness/latent_bridge.py
=====================================
LATENT BRIDGE: BIDIRECTIONAL LATENT-SPACE COUPLING

Eliminates the "syntactic bottleneck" — the last remaining technical critique.

AffectiveSteering (forward): substrate.x → steering weight → α·weight·v added to h_layer
LatentBridge (backward):     h_layer → project onto each v_i → substrate update

Together: genuine bidirectional coupling in activation space.
No text, no symbols, no lookup tables.

HOW THIS CAME TO BE LIVE, because the history is the useful part.
-----------------------------------------------------------------
The consciousness layer reported this as ``deferred (attaches on model
load)`` for as long as it existed. "Deferred" is a promise about a future
event, and nothing redeemed it: ``attach_latent_bridge()`` had no caller
anywhere. That was corrected to ``unwired``, which was honest and still
described a hole.

The hole was deeper than the missing call. Attaching it as written would
have injected nothing, twice over:

* ``SubstrateInjectionThread`` resolved the substrate with
  ``ServiceContainer.get("conscious_substrate")``. These hooks run in the
  MLX worker subprocess; the substrate is registered in the main runtime.
  None there, always.
* It injected through ``asyncio.get_running_loop()`` from a plain daemon
  thread, which raises unconditionally. Even in the main process, with the
  substrate present, the injection sat inside a ``try`` that could only take
  the ``except``.

So having no caller was the merciful part. A backward arrow that collects
deltas and drops them is worse than an absent one, because it reads as live.

Both are fixed. The thread publishes rather than injects, over
:mod:`core.consciousness.latent_readout_channel`, and
``MLXLocalClient._drain_latent_readouts`` injects in the parent, where the
substrate and a running event loop both exist.
"""

import logging
import threading
import time
from typing import Any

import numpy as np

from core.runtime.errors import record_degradation
from core.runtime.model_layers import resolve_model_layers

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
        steering_vectors: dict,
        feedback_weight: float = LATENT_FEEDBACK_WEIGHT,
    ):
        self._block = block
        self._layer_idx = layer_idx
        self._steering_vectors = steering_vectors
        self._feedback_weight = feedback_weight
        self._installed = False
        self._active = True

        # EMA-smoothed readout buffer (one per dimension)
        self._readout_ema: dict[str, float] = {
            key: 0.0 for key in steering_vectors.keys()
        }

        # Accumulated readouts for injection
        self._pending_injection: dict[int, float] = {}
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

            except (ImportError, AttributeError, RuntimeError) as e:
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

    def pop_pending_injection(self) -> dict[int, float]:
        """Pop and return the accumulated injection deltas (thread-safe)."""
        with self._injection_lock:
            result = dict(self._pending_injection)
            self._pending_injection.clear()
        return result

    def get_current_readouts(self) -> dict[str, float]:
        """Current EMA-smoothed readout values per affective dimension."""
        return dict(self._readout_ema)

    def get_diagnostics(self) -> dict[str, Any]:
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
    """Collects readout deltas from the hooks and publishes them across the fork.

    This used to try to inject directly, and could not have worked in any
    process. Two independent reasons, both now gone:

    * It resolved the substrate with ``ServiceContainer.get("conscious_substrate")``.
      These hooks run in the MLX worker; the substrate is registered in the
      main runtime. That lookup returns None there, always.
    * It called ``asyncio.get_running_loop()`` from a plain daemon thread,
      which raises unconditionally. Even in the main process, with the
      substrate present, the injection sat inside a ``try`` that could only
      take the ``except``.

    So it publishes now, and the parent injects. Transport is
    :mod:`core.consciousness.latent_readout_channel`, mirroring the Φ
    residual ring in the same direction for the same reason.
    """

    def __init__(
        self,
        readout_hooks: list[LatentReadoutHook],
        channel: Any = None,
    ):
        self._hooks = readout_hooks
        self._channel = channel
        self._thread: threading.Thread | None = None
        self._running = False
        self._total_published = 0
        self._total_magnitude_published = 0.0

    def start(self):
        if self._channel is None:
            logger.warning(
                "LatentBridge has no readout channel; the backward path would "
                "collect deltas and drop them. Not starting."
            )
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._loop,
            name="LatentBridge.ReadoutPublisher",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "🔄 LatentBridge readout publisher started (%d hooks)", len(self._hooks)
        )

    def stop(self):
        self._running = False

    def _loop(self):
        from core.consciousness.latent_readout_channel import publish_deltas

        while self._running:
            try:
                combined: dict[int, float] = {}
                for hook in self._hooks:
                    for idx, delta in hook.pop_pending_injection().items():
                        combined[idx] = combined.get(idx, 0.0) + delta

                if combined and publish_deltas(self._channel, combined):
                    self._total_published += 1
                    self._total_magnitude_published += float(
                        np.linalg.norm(np.array(list(combined.values()), dtype=np.float32))
                    )
            except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
                record_degradation(
                    "latent_bridge",
                    exc,
                    severity="debug",
                    action="dropped one latent readout publish cycle",
                    enforce_failure_policy=False,
                )

            time.sleep(INJECTION_INTERVAL_S)

    def get_diagnostics(self) -> dict[str, Any]:
        return {
            "running": self._running,
            "has_channel": self._channel is not None,
            "total_published": self._total_published,
            "total_magnitude": round(self._total_magnitude_published, 3),
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

    def __init__(self, steering_engine, channel: Any = None):
        self._steering_engine = steering_engine
        self._readout_hooks: list[LatentReadoutHook] = []
        self._injection_thread: SubstrateInjectionThread | None = None
        self._attached = False
        self._attachment_error: str | None = None
        self._layer_path: str | None = None
        #: Shared array to the parent. None means the backward arrow has no
        #: transport and must not pretend otherwise.
        self._channel = channel

    def attach(self, model) -> bool:
        """
        Install readout hooks at the same layers as the steering hooks.
        Must be called AFTER AffectiveSteeringEngine.attach().
        """
        if self._attached:
            logger.warning("LatentBridge already attached.")
            return True

        if not self._steering_engine._model_attached:
            self._attachment_error = "steering_engine_not_attached"
            logger.warning("LatentBridge unavailable: steering engine is not attached.")
            return False

        if not self._steering_engine._library or not self._steering_engine._library.vectors:
            self._attachment_error = "steering_vectors_unavailable"
            logger.warning("LatentBridge unavailable: no steering vectors are loaded.")
            return False

        layer_view = resolve_model_layers(model)
        if layer_view is None:
            self._attachment_error = (
                "unsupported_model_layer_topology:"
                f"{type(model).__module__}.{type(model).__qualname__}"
            )
            logger.warning("LatentBridge unavailable: %s", self._attachment_error)
            return False

        target_layers = self._steering_engine._model_info.get("target_layers", [])
        self._layer_path = layer_view.path

        for layer_idx in target_layers:
            if not isinstance(layer_idx, int) or layer_idx < 0 or layer_idx >= len(layer_view.layers):
                continue

            steering_vectors = self._steering_engine._library.get_vectors_for_layer(layer_idx)
            if not steering_vectors:
                steering_vectors = self._steering_engine._library.vectors
            if not steering_vectors:
                continue
            block = layer_view.layers[layer_idx]
            hook = LatentReadoutHook(
                block=block,
                layer_idx=layer_idx,
                steering_vectors=steering_vectors,
            )
            hook.install()
            self._readout_hooks.append(hook)

        self._attached = bool(self._readout_hooks)
        if not self._attached:
            self._attachment_error = "no_compatible_target_layers"
            logger.warning(
                "LatentBridge unavailable: no hooks installed for target layers %s via %s.",
                target_layers,
                layer_view.path,
            )
            return False
        self._attachment_error = None
        logger.info(
            "✅ LatentBridge attached: %d readout hooks at layers %s via %s",
            len(self._readout_hooks), target_layers, layer_view.path,
        )
        return True

    def start_substrate_sync(self, channel: Any = None):
        """Start publishing readouts toward the substrate.

        ``channel`` is the shared array the parent allocated before the fork
        (:mod:`core.consciousness.latent_readout_channel`). Without it the
        backward path has nowhere to go, and the publisher refuses to start
        rather than accumulating deltas into a thread that drops them —
        which is what this class did for its whole existence.
        """
        if not self._readout_hooks:
            logger.warning("No readout hooks installed. Call attach() first.")
            return
        if self._injection_thread and self._injection_thread._running:
            return
        self._injection_thread = SubstrateInjectionThread(
            self._readout_hooks, channel=channel or self._channel
        )
        self._injection_thread.start()

    def stop(self):
        """Stop substrate injection and disable all readout hooks."""
        if self._injection_thread:
            self._injection_thread.stop()
        for hook in self._readout_hooks:
            hook._active = False
        logger.info("🔕 LatentBridge stopped")

    def get_current_affective_readout(self) -> dict[str, float]:
        """
        The model's current "opinion" about its own affective state,
        as read from its hidden representations.
        """
        if not self._readout_hooks:
            return {}

        readouts: dict[str, list[float]] = {}
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
        steering_hooks = list(getattr(self._steering_engine, "_hooks", None) or [])
        if not steering_hooks or not self._readout_hooks:
            return 0.0

        steering_hook = steering_hooks[0]
        if getattr(steering_hook, "_substrate_x", None) is None:
            return 0.0

        substrate_x = steering_hook._substrate_x
        library = getattr(self._steering_engine, "_library", None)
        steering_vectors = getattr(library, "vectors", None) or {}
        if not steering_vectors:
            return 0.0
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

        steering_hooks = list(getattr(self._steering_engine, "_hooks", None) or [])
        steering_hook = steering_hooks[0] if steering_hooks else None
        if not steering_hook or getattr(steering_hook, "_substrate_x", None) is None:
            return "Substrate not connected."

        substrate_x = steering_hook._substrate_x
        readouts = self.get_current_affective_readout()
        library = getattr(self._steering_engine, "_library", None)
        vectors = getattr(library, "vectors", None) or {}

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
        lines.append("  (1.0=fully aligned, 0.0=orthogonal, -1.0=opposed)")

        return "\n".join(lines)

    def get_status(self) -> dict[str, Any]:
        return {
            "attached": self._attached,
            "attachment_error": self._attachment_error,
            "layer_path": self._layer_path,
            "hooks": len(self._readout_hooks),
            "injection_thread": (
                self._injection_thread.get_diagnostics()
                if self._injection_thread else None
            ),
            "coupling_coherence": round(self.get_coupling_coherence(), 4),
            "current_readout": self.get_current_affective_readout(),
        }


# ── Singleton and Boot Helpers ─────────────────────────────────────────────────

_bridge_instance: LatentBridge | None = None


def get_latent_bridge() -> LatentBridge | None:
    return _bridge_instance


def attach_latent_bridge(model, channel: Any = None) -> LatentBridge | None:
    """Install the backward path. Call after ``AffectiveSteeringEngine.attach()``.

    ``channel`` carries readouts to the process that owns the substrate. It
    is optional only so this stays callable from a test; in the worker it is
    required, and the publisher refuses to start without it. A backward arrow
    that collects deltas and drops them is the thing this module spent its
    whole existence being.
    """
    global _bridge_instance

    from core.consciousness.affective_steering import get_steering_engine
    engine = get_steering_engine()

    if not engine._model_attached:
        logger.error("AffectiveSteeringEngine must be attached before LatentBridge.")
        return None

    bridge = LatentBridge(engine, channel=channel)
    if not bridge.attach(model):
        error = bridge.get_status().get("attachment_error") or "attach_declined"
        record_degradation(
            "latent_bridge",
            RuntimeError(str(error)),
            severity="warning",
            action="continued inference without optional latent readout feedback",
        )
        _bridge_instance = None
        return None
    _bridge_instance = bridge

    try:
        from core.container import ServiceContainer
        ServiceContainer.register_instance("latent_bridge", _bridge_instance)
    except (ImportError, AttributeError, RuntimeError) as _e:
        record_degradation('latent_bridge', _e)
        logger.debug('Ignored Exception in latent_bridge.py: %s', _e)

    logger.info("✅ LatentBridge singleton created and registered")
    return _bridge_instance
