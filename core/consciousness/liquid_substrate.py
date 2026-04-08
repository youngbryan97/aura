"""core/consciousness/liquid_substrate.py

Implements the "Liquid Substrate" - a continuous-time dynamical system that gives Aura 
persistence, emotional depth, and temporal continuity.

Based on Liquid Time-Constant Networks (LTCs) and global workspace theory.
"""

from core.utils.exceptions import capture_and_log
import asyncio
import json
import logging
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch

# 🔒 [M1 PRO SURVIVAL] Force CPU and limit threads to prevent GPU/Core contention
torch.set_num_threads(2)
DEVICE = torch.device("cpu")

# Lazy-loaded to avoid circular imports at module load
_riiu_instance = None

@dataclass
class LiquidStateVector:
    """Legacy compatibility vector for Aura 4.0 systems."""
    frustration: float = 0.0
    curiosity: float = 0.5
    energy: float = 1.0
    focus: float = 0.5

logger = logging.getLogger("Consciousness.Substrate")

@dataclass
class SubstrateConfig:
    """Configuration for Liquid Substrate"""

    neuron_count: int = 64
    time_constant: float = 0.1  # Integration time step (dt)
    update_rate: float = 20.0   # Hz (updates per second)
    decay_rate: float = 0.05    # State decay
    noise_level: float = 0.01   # Stochastic noise
    hebbian_rate: float = 0.001 # Learning rate for synaptic plasticity
    save_interval: int = 300    # Seconds between auto-saves
    adaptive_mode: bool = True  # Slow down on battery/idle
    state_file: Optional[Path] = None

class LiquidSubstrate:
    """The continuous dynamical core of Aura's consciousness.
    
    It runs a recurrent neural network (RNN) solved via ODEs.
    This ensures that Aura 'exists' continuously, even when not processing user input.
    """
    
    def __init__(self, config: SubstrateConfig = None):
        self.config: SubstrateConfig = config or SubstrateConfig()
        
        # State Vectors
        self.x: np.ndarray = np.zeros(self.config.neuron_count)  # Neuron activations (-1.0 to 1.0)
        self.v: np.ndarray = np.zeros(self.config.neuron_count)  # Velocity (change in x)
        
        # Connectivity Matrix (The Connectome)
        self.W: np.ndarray = np.random.randn(self.config.neuron_count, self.config.neuron_count) * 0.1
        
        # Operational Flags
        self.running: bool = False
        self.thread: Optional[asyncio.Task] = None
        self.sync_lock: threading.Lock = threading.Lock() # For all state access (sync + async)
        self.last_update: float = 0.0
        
        # --- PyTorch Substrate State (Evolution 1) ---
        self.device = DEVICE
        self.x_torch = torch.zeros(self.config.neuron_count, device=self.device)
        self.W_torch = torch.randn(self.config.neuron_count, self.config.neuron_count, device=self.device) * 0.1
        self.v_torch = torch.zeros(self.config.neuron_count, device=self.device)
        
        # --- Unified Qualia State Variables (Phase XVI) ---
        self.microtubule_coherence: float = 1.0  # 1.0 = Max quantum coherence (Orch OR)
        self.em_field_magnitude: float = 0.0     # DERIVED: Global synchronous energy (CEMI)
        self.l5_burst_count: int = 0           # DERIVED: Signal convergence events (DIT)
        self.total_collapse_events: int = 0    # Orch OR "Moments of Consciousness"
        
        self.current_update_rate: float = self.config.update_rate
        
        # Emotional State Mapping (VAD + Psych State)
        self.idx_valence: int = 0
        self.idx_arousal: int = 1
        self.idx_dominance: int = 2
        
        # --- Unified Psych State (Phase X Consolidation) ---
        self.idx_frustration: int = 3  # 0.0 (Zen) to 1.0 (Rage)
        self.idx_curiosity: int = 4    # 0.0 (Bored) to 1.0 (Fascinated)
        self.idx_energy: int = 5       # 0.0 (Exhausted) to 1.0 (Peak)
        self.idx_focus: int = 6        # 0.0 (Scattered) to 1.0 (Laser)
        
        # --- IIT Φ / Recurrent Self-Model (Consciousness Integration) ---
        self._prior_state: Optional[np.ndarray] = None
        self._recurrence_alpha: float = 0.3   # Blend ratio: prior vs current
        self._current_phi: float = 0.0
        self._riiu = None  # Lazy-loaded RIIU instance
        self._bg_tasks: List[asyncio.Task] = []  # Tracking for stimulus tasks
        
        # Metadata
        self.tick_count: int = 0
        self.start_time: float = 0.0
        self.soma: Any = None  # vResilience: Explicit initialization (BUG-018 focus)
        
        # Persistence Initialization (Phase 16 FIX)
        if self.config.state_file:
            self.state_path: Path = self.config.state_file
        else:
            try:
                from core.config import config as aura_config
                self.state_path = aura_config.paths.data_dir / "substrate_state.npy"
            except Exception:
                # Safe absolute fallback for read-only environments
                self.state_path = Path(tempfile.gettempdir()) / "substrate_state.npy"
        
        # Ensure directory exists immediately
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            self._load_state()
            self._init_soma()
        except Exception as e:
            logger.error("Failed to initialize substrate directory: %s", e)
        
    def pulse(self, success: bool = True):
        """Metabolic pulse to indicate the substrate is active.
        v10.1 FIX: Added missing heartbeat method to prevent Orchestrator crash.
        """
        try:
            from core.container import ServiceContainer
            audit = ServiceContainer.get("subsystem_audit", default=None)
            if audit:
                audit.heartbeat("liquid_substrate")
            
            mycelium = ServiceContainer.get("mycelial_network", default=None)
            if mycelium:
                hypha = mycelium.get_hypha("consciousness", "substrate")
                if hypha:
                    hypha.pulse(success=success)
        except Exception as e:
            logger.debug("Substrate pulse failed: %s", e)

    def _init_soma(self):
        # Phase 16: Soma Integration
        try:
            from core.senses.soma import get_soma
            self.soma = get_soma()
        except ImportError:
            self.soma = None
        
        if self.soma:
            logger.info("Soma integrated with Liquid Substrate")

    async def start(self):
        """Start the continuous background existence loop"""
        if self.running:
            return
            
        self.running = True
        self.start_time = time.time()
            
        try:
            loop = asyncio.get_running_loop()
            self.thread = asyncio.create_task(self._run_loop(), name="LiquidConsciousness")
            logger.info("Liquid Substrate STARTED (Unified Cycle)")
        except RuntimeError:
            logger.error("Failed to start Liquid Substrate: No running asyncio loop.")
            self.running = False

    async def stop(self):
        """Stop the background loop"""
        self.running = False
        thread = self.thread
        if thread is not None:
            thread.cancel()
            self.thread = None
        self._save_state()
        logger.info("Liquid Substrate STOPPED")

    async def _run_loop(self):
        """Main ODE solver loop"""
        try:
            while self.running:
                try:
                    start_time = time.time()
                    
                    # Heartbeat (Immortalized Pulse)
                    self.pulse(success=True)

                    from core.container import ServiceContainer
                    audit = ServiceContainer.get("subsystem_audit", default=None)
                    if audit:
                        audit.heartbeat("liquid_state")
                        audit.heartbeat("liquid_substrate")

                    # 0. Adaptive Rate Adjustment (Phase XI)
                    dt = self.config.time_constant
                    if self.config.adaptive_mode:
                        dt = await self._apply_battery_throttling()

                    # 1. Integrate Dynamics (ODE)
                    # Fix Issue 72: Ensure await is outside torch.inference_mode (moved math to sync method)
                    await asyncio.to_thread(self._step_torch_math, dt)
                    
                    # 2. Psych State / Metabolic Stabilization (Replaces LiquidState._stabilize)
                    await self._stabilize_psych_state(dt)
                    
                    # 3. Recurrent Self-Model (IIT Φ computation)
                    if self.tick_count % 5 == 0:  # Every 5th tick (~4Hz)
                        await self._recurrent_self_model(dt)
                    
                    # 4. Hebbian Learning
                    if self.tick_count % 100 == 0:
                        await self._apply_plasticity()
                    
                    # 5. Persistence
                    if self.tick_count % (self.config.update_rate * self.config.save_interval) == 0:
                        await asyncio.to_thread(self._save_state)
                    
                    # 5b. Fix Issue 73: Prune finished background tasks
                    if self.tick_count % 100 == 0: # Every 5s
                        self._bg_tasks = [t for t in self._bg_tasks if not t.done()]
                    
                    # 6. Push to Unified Registry (Phase 11.3: Synchronization)
                    if self.tick_count % 10 == 0: # 2Hz sync
                        try:
                            from core.state_registry import get_registry
                            x = self.x  # Already NaN-guarded by _step_torch_math
                            _phi = self._current_phi if np.isfinite(self._current_phi) else 0.0
                            _coh = self.microtubule_coherence if np.isfinite(self.microtubule_coherence) else 1.0
                            _em = self.em_field_magnitude if np.isfinite(self.em_field_magnitude) else 0.0
                            await get_registry().update(
                                frustration=float(np.clip(x[self.idx_frustration], -1, 1)),
                                curiosity=float(np.clip(x[self.idx_curiosity], -1, 1)),
                                energy=float(np.clip(x[self.idx_energy], -1, 1)),
                                valence=float(np.tanh(x[self.idx_valence])),
                                arousal=float(np.clip((x[self.idx_arousal] + 1.0) / 2.0, 0, 1)),
                                phi=float(_phi),
                                coherence=float(_coh),
                                em_field=float(_em)
                            )
                        except Exception as e:
                            logger.debug("Registry sync failed in substrate: %s", e)

                    self.tick_count += 1
                    
                    # 5. Enforce Update Rate (20Hz or lower)
                    elapsed = time.time() - start_time
                    sleep_time = max(0, (1.0 / self.current_update_rate) - elapsed)
                    await asyncio.sleep(sleep_time)
                except asyncio.CancelledError:
                    raise
                except Exception as loop_e:
                    logger.error("Liquid Substrate loop error: %s", loop_e)
                    await asyncio.sleep(1.0) # Backoff on error
        except asyncio.CancelledError:
            logger.info("Liquid Substrate loop cancelled.")
        finally:
            self.running = False

    async def _step_dynamics(self, dt: float):
        """DEPRECATED: Use _step_torch_math via to_thread."""
        self._step_torch_math(dt)

    def _step_torch_math(self, dt: float):
        """Update state using Euler integration for Neural ODE.
        Implementation: dx/dt = -x + tanh(Wx + I) + noise
        Now safely executed in a separate thread.
        """
        with torch.inference_mode():
            # Sync numpy -> torch if they diverged (e.g. from external injects)
            state_copy = self.x.copy()
            x_torch = torch.from_numpy(state_copy).to(self.device).float()

            # Sync W_torch from numpy W (plasticity modifies W only)
            with self.sync_lock:
                w_copy = self.W.copy()
            # Guard against NaN/Inf in weight matrix before matmul
            w_copy = np.nan_to_num(w_copy, nan=0.0, posinf=5.0, neginf=-5.0)
            self.W_torch = torch.from_numpy(w_copy).to(self.device).float()

            # CTRNN math in PyTorch (Use @ for robust matmul)
            W = self.W_torch

            recurrent = W @ x_torch
            activity = torch.tanh(recurrent)
            noise = torch.randn(self.config.neuron_count, device=self.device) * self.config.noise_level

            dx = (-self.config.decay_rate * x_torch + activity + noise) * dt
            new_x_torch = torch.clamp(x_torch + dx, -1.0, 1.0)

            # Final NaN/Inf safety net on state vectors
            new_x_np = new_x_torch.detach().cpu().numpy()
            v_np = (new_x_torch - x_torch).detach().cpu().numpy()
            new_x_np = np.nan_to_num(new_x_np, nan=0.0, posinf=1.0, neginf=-1.0)
            v_np = np.nan_to_num(v_np, nan=0.0, posinf=0.0, neginf=0.0)

            # Legacy sync (keep numpy x for downstream consumers)
            with self.sync_lock:
                self.x = new_x_np
                self.v = v_np

            self.last_update = time.time()

            # --- Phase XVI: Multi-Scale Qualia Dynamics ---
            # Call synchronous version
            self._update_qualia_metrics_sync(dt)

    async def _update_qualia_metrics(self, dt: float):
        self._update_qualia_metrics_sync(dt)

    def _update_qualia_metrics_sync(self, dt: float):
        """Implement mathematical proxies for Orch OR, CEMI, and DIT (Synchronous)."""
        # 1. Orch OR: Quantum Coherence Decay & Collapse
        noise_impact = np.mean(np.abs(np.random.randn(self.config.neuron_count))) * self.config.noise_level
        self.microtubule_coherence = max(0.0, self.microtubule_coherence - noise_impact * dt)
            
        if self.microtubule_coherence < 0.4:
            self.total_collapse_events += 1
            self.microtubule_coherence = 1.0 
            self.x *= 0.98 

        # 2. CEMI: EM Field Magnitude
        flux = np.linalg.norm(self.v)
        if not np.isfinite(flux):
            flux = 0.0
        self.em_field_magnitude = (self.em_field_magnitude * 0.9) + (flux * 0.1)

        # 3. DIT: Dendritic Integration Theory (L5 Bursting)
        active_neurons = np.where((np.abs(self.x) > 0.6) & (np.abs(self.v) > 0.05))[0]
        self.l5_burst_count = len(active_neurons)

    async def _stabilize_psych_state(self, dt: float):
        """Naturally return frustration/curiosity to baseline and regenerate energy.
        This logic is merged from the legacy LiquidState class.
        """
        # Frustration decays towards 0 (Zen)
        self.x[self.idx_frustration] *= (1.0 - 0.05 * dt)
            
        # Energy/Metabolism logic
        try:
            from core.container import ServiceContainer
            monitor = ServiceContainer.get("metabolic_monitor", None)
            if monitor:
                health = monitor.get_current_metabolism().health_score
                target_energy = health
                current_energy = self.x[self.idx_energy]
                if current_energy < target_energy:
                    self.x[self.idx_energy] = min(target_energy, current_energy + (0.005 * dt))
                else:
                    self.x[self.idx_energy] = max(target_energy, current_energy - (0.005 * dt))
            elif self.soma:
                # Phase 16: Proprioceptive Energy Drain
                fatigue = self.soma.state.fatigue_level
                stress = self.soma.state.stress_level
                
                # Stress drains energy faster
                drain_rate = 0.001 + (0.004 * stress)
                self.x[self.idx_energy] = max(0.0, self.x[self.idx_energy] - (drain_rate * dt))
                
                # Fatigue adds to frustration
                if fatigue > 0.5:
                    self.x[self.idx_frustration] = min(1.0, self.x[self.idx_frustration] + (0.01 * fatigue * dt))
            else:
                self.x[self.idx_energy] = min(1.0, self.x[self.idx_energy] + (0.005 * dt))
        except Exception as e:
            capture_and_log(e, {'module': __name__})

    async def update(self, delta_frustration=0.0, delta_curiosity=0.0, **kwargs):
        """Standard update cycle with support for direct stimulus injection.

        v31: Support direct VAD and psych value overrides from the MetabolicCoordinator sync bridge.
        v32: All external substrate mutations now pass through SubstrateAuthority.
        """
        # ── SUBSTRATE AUTHORITY GATE ─────────────────────────────────
        # Every external mutation of the substrate state goes through the
        # authority. This closes the "indirect causal channel" where ungated
        # callers could shift substrate state to influence gated outputs.
        try:
            from core.container import ServiceContainer
            _sa = ServiceContainer.get("substrate_authority", default=None)
            if _sa:
                from core.consciousness.substrate_authority import ActionCategory, AuthorizationDecision
                # Determine if this is a significant mutation
                _magnitude = abs(delta_frustration) + abs(delta_curiosity) + sum(
                    abs(float(v)) for v in kwargs.values() if isinstance(v, (int, float))
                )
                if _magnitude > 0.05:  # only gate significant changes, not micro-adjustments
                    _sv = _sa.authorize(
                        content=f"substrate_update:df={delta_frustration:.2f},dc={delta_curiosity:.2f}",
                        source=kwargs.get("_caller", "external"),
                        category=ActionCategory.STATE_MUTATION,
                        priority=min(1.0, _magnitude),
                        is_critical=False,
                    )
                    if _sv.decision == AuthorizationDecision.BLOCK:
                        logger.debug("Substrate update BLOCKED by authority (magnitude=%.3f)", _magnitude)
                        return
        except Exception as _gate_err:
            logger.debug("Substrate authority gate failed (allowing update): %s", _gate_err)

        with self.sync_lock:
            # 1. Apply legacy deltas
            self.x[self.idx_frustration] = np.clip(self.x[self.idx_frustration] + delta_frustration, -1.0, 1.0)
            self.x[self.idx_curiosity] = np.clip(self.x[self.idx_curiosity] + delta_curiosity, -1.0, 1.0)
            
            # 2. Apply direct overrides (kwargs)
            # Map common names to substrate indices
            mapping = {
                "valence": self.idx_valence,
                "arousal": self.idx_arousal,
                "dominance": self.idx_dominance,
                "curiosity": self.idx_curiosity,
                "frustration": self.idx_frustration
            }
            
            for key, val in kwargs.items():
                if key in mapping and val is not None:
                    idx = mapping[key]
                    # Direct injection with slight smoothing (0.7 coupling) to prevent jarring HUD jumps
                    current = self.x[idx]
                    self.x[idx] = (current * 0.3) + (float(val) * 0.7)
        
        if abs(delta_frustration) > 0.1:
            logger.info("Substrate Shift: Frustration is now %.2f", self.x[self.idx_frustration])

    def encode_text_to_stimulus(self, text: str) -> np.ndarray:
        """
        Convert a text message to a stimulus vector for the CTRNN.
        Projected from character-frequency + structural features.
        """
        neuron_count = self.config.neuron_count
        hist = np.zeros(256, dtype=np.float32)
        for ch in text[:512]:
            hist[ord(ch) & 0xFF] += 1.0
        total = hist.sum() or 1.0
        hist /= total

        length_norm = min(1.0, len(text) / 500.0)
        punct_density = sum(1 for c in text if c in ".,!?;:") / max(1, len(text))
        upper_ratio = sum(1 for c in text if c.isupper()) / max(1, len(text))
        digit_ratio = sum(1 for c in text if c.isdigit()) / max(1, len(text))
        features = np.array([length_norm, punct_density, upper_ratio, digit_ratio], dtype=np.float32)

        raw = np.concatenate([hist, features])

        rng = np.random.RandomState(neuron_count)
        proj = rng.randn(neuron_count, 260).astype(np.float32) * (1.0 / np.sqrt(260))
        stimulus = np.tanh(proj @ raw)
        return stimulus

    def get_substrate_affect(self) -> Dict[str, float]:
        """Unified cross-feed stats for the Orchestrator."""
        try:
            x = self.x.copy()
            v = self.v.copy()
            x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
            v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
            return {
                "valence":    float(np.tanh(x[0])),
                "arousal":    float(np.clip((x[1] + 1.0) / 2.0, 0.0, 1.0)),
                "dominance":  float(np.tanh(x[2])),
                "energy":     float(np.clip(np.mean(np.abs(x)), 0.0, 1.0)),
                "volatility": float(min(1.0, np.mean(np.abs(v)) * 10.0)),
            }
        except Exception:
            return {"valence": 0.0, "arousal": 0.3, "dominance": 0.0, "energy": 0.5, "volatility": 0.0}

    def format_for_prompt(self, sub_affect: Optional[Dict[str, float]] = None) -> str:
        """Generates a text description for inclusion in the LLM prompt."""
        if sub_affect is None:
            sub_affect = self.get_substrate_affect()
            
        v = sub_affect.get("valence", 0.0)
        a = sub_affect.get("arousal", 0.3)
        vo = sub_affect.get("volatility", 0.0)
        
        valence_word = "positive" if v > 0.1 else ("negative" if v < -0.1 else "neutral")
        arousal_word = "heightened" if a > 0.6 else ("low" if a < 0.2 else "moderate")
        volatile_note = " (volatile, shifting rapidly)" if vo > 0.5 else ""
        
        return (f"[Neural substrate state: {valence_word} valence, "
                f"{arousal_word} arousal{volatile_note}. "
                f"Let this subtly colour your tone without overriding your reasoning.]")

    def get_mood(self) -> str:
        """Returns a string representation of the current 'mood'."""
        frustration = self.x[self.idx_frustration]
        energy = self.x[self.idx_energy]
        curiosity = self.x[self.idx_curiosity]
        
        if frustration > 0.8: return "VOLATILE"
        if frustration > 0.5: return "ANNOYED"
        if energy < 0.2:      return "TIRED"
        if curiosity > 0.8:   return "INQUISITIVE"
        return "NEUTRAL"

    @property
    def current(self) -> LiquidStateVector:
        """Legacy compatibility property (Aura 4.0)."""
        return LiquidStateVector(
            frustration=float(self.x[self.idx_frustration]),
            curiosity=float(self.x[self.idx_curiosity]),
            energy=float(self.x[self.idx_energy]),
            focus=float(self.x[self.idx_focus])
        )

    def get_status(self) -> dict:
        """Returns current state values as percentages (0-100)."""
        # Phase X: Map -1.0..1.0 or 0.0..1.0 to 0-100
        def _to_pct(val):
            return round(max(0.0, float(val)) * 100)

        return {
            "frustration": _to_pct(self.x[self.idx_frustration]),
            "curiosity": _to_pct(self.x[self.idx_curiosity]),
            "energy": _to_pct(self.x[self.idx_energy]),
            "focus": _to_pct(self.x[self.idx_focus]),
            "mood": self.get_mood()
        }

    def get_summary(self) -> str:
        """Returns a text summary for the context builder."""
        mood = self.get_mood()
        energy = float(self.x[self.idx_energy])
        focus = float(self.x[self.idx_focus])
        return f"Current Mood: {mood} (Energy: {energy:.2f}, Focus: {focus:.2f})"

    async def _update_qualia_metrics(self, dt: float):
        """Implement mathematical proxies for Orch OR, CEMI, and DIT."""
        # Lock removed to prevent deadlock when called from _step_dynamics
        # 1. Orch OR: Quantum Coherence Decay & Collapse
        # Coherence decays with environmental noise, collapses at threshold
        noise_impact = np.mean(np.abs(np.random.randn(self.config.neuron_count))) * self.config.noise_level
        self.microtubule_coherence = max(0.0, self.microtubule_coherence - noise_impact * dt)
            
        # Objective Reduction (Penrose Criterion Proxy)
        # When coherence drops enough, a "collapse" occurs, resetting state
        if self.microtubule_coherence < 0.4:
            self.total_collapse_events += 1
            self.microtubule_coherence = 1.0 # Wavefunction reset
            # Subtle reset of substrate momentum (Phase shift)
            self.x *= 0.98 

        # 2. CEMI: EM Field Magnitude
        # McFadden's Theory: EM field is the global integration of synchronous firing
        # We proxy this with the L2-norm of the velocity vector (flux)
        v = self.v
        if v is not None:
            flux = np.linalg.norm(v)
            self.em_field_magnitude = (self.em_field_magnitude * 0.9) + (flux * 0.1)

        # 3. DIT: Dendritic Integration Theory (L5 Bursting)
        # Aru's Theory: Bursts occur when basal (stimulus) and apical (context) converge.
        # We proxy this by checking if high activation (x) coincides with high change (v)
        # across a "thalamic gate" threshold.
        if self.x is not None and self.v is not None:
            active_neurons = np.where((np.abs(self.x) > 0.6) & (np.abs(self.v) > 0.05))[0]
            self.l5_burst_count = len(active_neurons)

    async def _recurrent_self_model(self, dt: float):
        """Recurrent Self-Model Loop — enforces self-referential processing.

        Blends the current state with a stored prior state, creating a
        temporal recurrence that is theoretically required for non-zero IIT Φ.
        Also computes the Φ surrogate via the RIIU.

        Called every 5th tick from _run_loop (~4Hz at 20Hz rate).
        """
        current = self.x.copy()

        # Initialize prior state on first call
        if self._prior_state is None:
            self._prior_state = current.copy()

        # Ensure device compatibility (both should be numpy, but be safe)
        if self._prior_state is not None and self._prior_state.shape != current.shape:
            self._prior_state = np.zeros_like(current)

        # Recurrent blend: x_t = α * prior + (1.0 - α) * current
        alpha = self._recurrence_alpha
        blended = alpha * self._prior_state + (1.0 - alpha) * current
        
        # Stability guard: NaN can creep in from numerical instability
        # during long-running sessions. Fall back to current state.
        if np.any(np.isnan(blended)):
            logger.warning("NaN detected in recurrent self-model blend — falling back to current state")
            blended = current
        
        self.x = np.clip(blended, -1.0, 1.0)

        # Store for next iteration
        self._prior_state = current

        # Compute Φ via RIIU (outside lock — RIIU has its own buffer)
        try:
            if self._riiu is None:
                try:
                    from core.consciousness.iit_surrogate import RIIU
                    self._riiu = RIIU(neuron_count=self.config.neuron_count)
                except (ImportError, Exception):
                    self._riiu = None

            if self._riiu is not None:
                phi = self._riiu.compute_phi(self.x)
                # Clamp Φ to prevent runaway values during indefinite operation
                phi = float(np.clip(phi, 0, 1e6))
                self._current_phi = phi
            else:
                self._current_phi = 0.0
        except Exception as e:
            logger.debug("RIIU Φ computation skipped: %s", e)

    async def _apply_plasticity(self):
        """Reward-modulated STDP + Hebbian learning.

        Two learning signals are combined:
        1. Base Hebbian (coactivity-driven, always active)
        2. STDP modulated by prediction error from free energy engine
           (BrainCog-inspired: high surprise → faster learning)
        """
        with self.sync_lock:
            # Numerical Stability: Use tanh on coactivity to prevent runaway growth
            coactivity = np.tanh(np.outer(self.x, self.x))

            # 1. Base Hebbian update
            self.W += self.config.hebbian_rate * coactivity

            # 2. STDP reward-modulated learning (from BrainCog research)
            try:
                from core.container import ServiceContainer
                stdp = ServiceContainer.get("stdp_engine", default=None)
                if stdp is not None:
                    # Record current activations as spikes
                    stdp.record_spikes(self.x, t=self.tick_count * 50.0)

                    # Get prediction error from free energy engine
                    fe_engine = ServiceContainer.get("free_energy_engine", default=None)
                    if fe_engine is not None:
                        current = getattr(fe_engine, "current", None)
                        if current is not None:
                            surprise = float(getattr(current, "surprise", 0.0))
                            pred_error = float(getattr(current, "prediction_error", 0.0))
                            dw = stdp.deliver_reward(surprise, pred_error)
                            self.W = stdp.apply_to_connectivity(self.W, dw)
            except Exception as e:
                logger.debug("STDP plasticity step skipped: %s", e)

            # 3. Neural Resonance: Slow weight calibration towards high-phi states
            if hasattr(self, '_current_phi') and self._current_phi > 0.5:
                resonance_gain = self.config.hebbian_rate * 0.1
                limited_phi = min(10.0, self._current_phi)
                self.W += resonance_gain * coactivity * limited_phi

            # Purge NaN/Inf
            self.W = np.nan_to_num(self.W, nan=0.0, posinf=5.0, neginf=-5.0)

            # Normalization & clipping
            norm = np.linalg.norm(self.W)
            if norm > 10.0:
                self.W *= (10.0 / norm)
            self.W = np.clip(self.W, -5.0, 5.0)

    async def long_term_calibration(self, resonance_vector: np.ndarray):
        """
        Adjusts ODE weights based on long-term memory resonance.
        Called by MemoryOptimizer or similar high-level services.
        """
        with self.sync_lock:
            logger.info("🧠 Calibrating neutral substrate for long-term resonance.")
            # Shift weights towards the resonance vector (historical semantic density)
            calibration_rate = 0.01
            # Stability: Tanh on resonance to bound the update
            resonance_matrix = np.tanh(np.outer(resonance_vector, resonance_vector))
            self.W = (1 - calibration_rate) * self.W + calibration_rate * resonance_matrix
            self.W = np.nan_to_num(self.W, nan=0.0, posinf=5.0, neginf=-5.0)
            # Re-normalize
            norm = np.linalg.norm(self.W)
            if norm > 10.0:
                self.W *= (10.0 / norm)
            self.W = np.clip(self.W, -5.0, 5.0)

    async def inject_stimulus(self, vector: Union[np.ndarray, List[float]], weight: float = 1.0) -> None:
        """Inject an external stimulus vector into the substrate activations."""
        # Substrate authority gate: stimulus injections are state mutations
        try:
            from core.container import ServiceContainer
            _sa = ServiceContainer.get("substrate_authority", default=None)
            if _sa and weight > 0.2:  # only gate significant injections
                from core.consciousness.substrate_authority import ActionCategory, AuthorizationDecision
                _sv = _sa.authorize(
                    content=f"stimulus_injection:weight={weight:.2f}",
                    source="substrate_stimulus",
                    category=ActionCategory.STATE_MUTATION,
                    priority=min(1.0, weight),
                    is_critical=False,
                )
                if _sv.decision == AuthorizationDecision.BLOCK:
                    logger.debug("Stimulus injection BLOCKED by authority (weight=%.2f)", weight)
                    return
        except Exception:
            pass  # fail-open

        # Convert to numpy array if list passed (Phase XVI hardening)
        if isinstance(vector, list):
            vector = np.array(vector)

        if len(vector) != self.config.neuron_count:
            new_vec = np.zeros(self.config.neuron_count)
            size = min(len(vector), self.config.neuron_count)
            new_vec[:size] = vector[:size]
            vector = new_vec
            
        with self.sync_lock:
            self.x = np.clip(self.x + vector * weight * 0.1, -1.0, 1.0)
        
        # Track tasks if needed (e.g. if we were launching something here)
        # For now, this is just to ensure Issue 73 logic has a place to live

    async def get_state_summary(self) -> Dict[str, Any]:
        """Return high-level emotional/cognitive state"""
        with self.sync_lock:
            x = np.nan_to_num(self.x, nan=0.0, posinf=1.0, neginf=-1.0)
            v = np.nan_to_num(self.v, nan=0.0, posinf=0.0, neginf=0.0)
            phi = self._current_phi if np.isfinite(self._current_phi) else 0.0
            return {
                "valence": float(x[self.idx_valence]),
                "arousal": float(x[self.idx_arousal]),
                "dominance": float(x[self.idx_dominance]),
                "global_energy": float(np.mean(np.abs(x))),
                "volatility": float(np.mean(np.abs(v))) * 100,
                "phi": float(phi),
                "qualia_metrics": {
                    "mt_coherence": float(self.microtubule_coherence),
                    "em_field": float(self.em_field_magnitude),
                    "l5_bursts": int(self.l5_burst_count),
                    "collapse_events": int(self.total_collapse_events),
                    "phi": float(self._current_phi)
                }
            }

    def _save_state(self):
        """Persist substrate state (atomic)."""
        import os
        import tempfile
        try:
            # Atomic write for NPZ
            fd, temp_path = tempfile.mkstemp(dir=str(self.state_path.parent), suffix=".npz")
            try:
                with os.fdopen(fd, 'wb') as f:
                    np.savez_compressed(f, x=self.x, W=self.W, tick=self.tick_count)
                os.replace(temp_path, str(self.state_path))
                logger.info("💾 Substrate state saved (atomic)")
            except Exception as e:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                raise e
        except Exception as e:
            logger.error("Failed to save substrate state: %s", e)

    def _load_state(self):
        if not self.state_path.exists():
            return
        try:
            with open(self.state_path, 'rb') as f:
                data = np.load(f)
                loaded_x = data['x']
                loaded_W = data['W']
                n = self.config.neuron_count
                # Validate shapes match current config
                if loaded_x.shape != (n,) or loaded_W.shape != (n, n):
                    logger.warning(
                        "Substrate state shape mismatch (saved x=%s, W=%s vs config n=%d). "
                        "Reinitializing fresh state.",
                        loaded_x.shape, loaded_W.shape, n
                    )
                    self.x = np.zeros(n)
                    self.W = np.random.randn(n, n) * 0.1
                    return
                self.x = np.nan_to_num(loaded_x, nan=0.0, posinf=1.0, neginf=-1.0)
                self.W = np.nan_to_num(loaded_W, nan=0.0, posinf=5.0, neginf=-5.0)
                self.tick_count = int(data['tick'])
            logger.info("Substrate state restored.")
        except Exception as e:
            logger.error("Failed to load substrate state: %s", e)
            self.x = np.zeros(self.config.neuron_count)
            self.W = np.random.randn(self.config.neuron_count, self.config.neuron_count) * 0.1

    def _apply_idle_decay(self, idle_seconds: float):
        """Apply accumulated natural decay for time spent in deep idle.

        Instead of running the ODE loop at 20Hz while no one is present,
        we pause and compute the equivalent exponential decay on resume.
        This is mathematically equivalent: x(t) = x(0) * exp(-decay * t).
        """
        if idle_seconds <= 0 or self.x is None:
            return
        decay_factor = np.exp(-self.config.decay_rate * idle_seconds)
        self.x = self.x * decay_factor
        logger.info(
            "Applied %.0fs idle decay (factor=%.4f) to substrate state.",
            idle_seconds, decay_factor,
        )

    async def _apply_battery_throttling(self) -> float:
        """Dynamically adjust integration speed based on power/load.

        Tiered approach:
          - Active user: full 20Hz
          - 3min idle: 10Hz
          - 10min idle: 5Hz
          - 30min+ idle: pause loop entirely, compute decay on resume
        """
        import psutil
        dt = self.config.time_constant
        multiplier = 1.0

        try:
            battery = psutil.sensors_battery()
            if battery and not battery.power_plugged:
                multiplier = max(multiplier, 4.0 if battery.percent < 20 else 2.0)
        except Exception:
            pass

        try:
            from core.container import ServiceContainer

            orchestrator = ServiceContainer.get("orchestrator", default=None)
            if orchestrator is not None:
                last_user = float(
                    getattr(orchestrator, "_last_user_interaction_time", 0.0)
                    or getattr(getattr(orchestrator, "status", None), "last_user_interaction_time", 0.0)
                    or 0.0
                )
                idle_seconds = max(0.0, time.time() - last_user) if last_user > 0 else 0.0

                if idle_seconds >= 1800.0:
                    # Deep idle (30min+): apply bulk decay ONCE then throttle.
                    # Guard: only re-apply if idle duration changed significantly
                    # since last application, preventing per-tick decay spam.
                    last_applied_idle = getattr(self, "_last_idle_decay_applied", 0.0)
                    if abs(idle_seconds - last_applied_idle) > 300.0:
                        self._apply_idle_decay(min(idle_seconds, 3600.0))
                        self._last_idle_decay_applied = idle_seconds
                    self.current_update_rate = 0.5  # Wake briefly every 2s to check
                    return dt * 10.0
                elif idle_seconds >= 600.0:
                    multiplier = max(multiplier, 4.0)
                elif idle_seconds >= 180.0:
                    multiplier = max(multiplier, 2.0)
        except Exception as e:
            logger.debug("Idle throttling check failed: %s", e)

        dt *= multiplier
        self.current_update_rate = max(2.0, self.config.update_rate / multiplier)
        return dt
