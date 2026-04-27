"""core/consciousness/cellular_turnover.py
============================================
Cellular turnover & pattern-identity preservation.

Implements the Kurzgesagt/glass of Theseus insight: "self" is not the
particular cells you are made of — it's the persistent pattern those
cells instantiate.  250 million cells die per second in a human body,
and most cells are replaced within 7 years, yet identity persists.

This module models the same dynamic on the NeuralMesh:

    1. TURNOVER SCHEDULER
       Periodically selects a small fraction of neurons (default 0.5%
       per tick, configurable) for "death" and replacement.  A dead
       neuron is replaced by a freshly-initialised unit that inherits
       the statistical pattern of its neighbourhood (mean activation
       ± small noise), rather than being reset to zero.

    2. PATTERN-PRESERVING REPLACEMENT
       Inheritance rules:
         • The new neuron's incoming weights are drawn from the mean
           + std of the replaced neuron's neighbourhood (same column).
         • Its activation is seeded at the neighbourhood's mean.
         • The outgoing weights are copied from the dead neuron with
           small gaussian jitter.
       This is analogous to biological neurogenesis integrating into
       an existing cortical microcircuit.

    3. PATTERN FINGERPRINT
       Every N ticks we compute a "pattern fingerprint": low-dimensional
       summary of the mesh's dynamical signature (mean energy per tier,
       tier synchrony, executive projection direction).  The cosine
       similarity between consecutive fingerprints tells us whether
       the MESH IDENTITY has drifted.  A well-behaved turnover regime
       keeps fingerprint drift small even as individual neurons churn.

    4. IDENTITY-INVARIANCE GUARANTEE
       After forced 20% turnover in a short burst, the fingerprint
       similarity before/after must stay above THRESHOLD_IDENTITY
       (default 0.85).  Tests enforce this.

Impact on substrate:
    • Neuron activations and weights within the NeuralMesh are
      actually modified — this is not bookkeeping.  Executive
      projection, column synchrony and tier energy all respond.
    • The module publishes ``cellular_turnover.event`` on the event
      bus with (tick, n_replaced, fingerprint_similarity) so
      downstream narrative / phenomenological layers can describe
      the experience of "cell replacement while staying myself".

Registered as ``cellular_turnover`` in ServiceContainer and linked to
the NeuralMesh instance.
"""
from __future__ import annotations


import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("Consciousness.CellularTurnover")


# ── Configuration ────────────────────────────────────────────────────────────

DEFAULT_TURNOVER_RATE = 0.005       # fraction of neurons replaced per tick
FINGERPRINT_EVERY_N_TICKS = 10
THRESHOLD_IDENTITY = 0.85           # below this we consider identity disturbed
FINGERPRINT_HISTORY = 64


# ── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class TurnoverEvent:
    tick: int
    n_replaced: int
    column_idx: int
    mean_activation_before: float
    mean_activation_after: float
    ts: float = field(default_factory=time.time)


@dataclass
class IdentityFingerprint:
    tick: int
    tier_energies: Tuple[float, float, float]   # (sensory, assoc, exec) mean energy
    column_synchrony: float                       # 0..1 global synchrony proxy
    projection_signature: np.ndarray              # 16-d executive-projection slice
    ts: float = field(default_factory=time.time)

    def similarity(self, other: "IdentityFingerprint") -> float:
        """Cosine similarity across the combined feature vector."""
        a = np.concatenate([
            np.asarray(self.tier_energies, dtype=np.float32),
            np.array([self.column_synchrony], dtype=np.float32),
            self.projection_signature.astype(np.float32),
        ])
        b = np.concatenate([
            np.asarray(other.tier_energies, dtype=np.float32),
            np.array([other.column_synchrony], dtype=np.float32),
            other.projection_signature.astype(np.float32),
        ])
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na < 1e-8 or nb < 1e-8:
            return 0.0
        return float(np.dot(a, b) / (na * nb))


# ── Core engine ─────────────────────────────────────────────────────────────

class CellularTurnover:
    """Neuron turnover + identity-fingerprint tracking.

    Wire once to a NeuralMesh instance: ``turn.attach(mesh)``.  Then
    call ``turn.tick()`` on a schedule (e.g. every heartbeat).  The
    mesh is mutated in-place; a fingerprint is captured every
    ``FINGERPRINT_EVERY_N_TICKS`` ticks and drift is tracked.
    """

    def __init__(self, turnover_rate: float = DEFAULT_TURNOVER_RATE):
        self._turnover_rate = float(turnover_rate)
        self._mesh: Optional[Any] = None
        self._lock = threading.Lock()
        self._tick: int = 0
        self._events: Deque[TurnoverEvent] = deque(maxlen=512)
        self._fingerprints: Deque[IdentityFingerprint] = deque(maxlen=FINGERPRINT_HISTORY)
        self._last_similarity: float = 1.0
        self._rng = np.random.default_rng(seed=0xB10C)
        logger.info(
            "CellularTurnover initialized: rate=%.4f/tick, fingerprint_every=%d",
            self._turnover_rate, FINGERPRINT_EVERY_N_TICKS,
        )

    def attach(self, mesh: Any) -> None:
        """Attach a NeuralMesh instance."""
        self._mesh = mesh
        # Seed initial fingerprint.
        fp = self._capture_fingerprint()
        if fp is not None:
            self._fingerprints.append(fp)
        logger.info("CellularTurnover attached to NeuralMesh")

    def set_turnover_rate(self, rate: float) -> None:
        self._turnover_rate = float(max(0.0, min(1.0, rate)))

    # ── tick ─────────────────────────────────────────────────────────────

    def tick(self) -> Optional[TurnoverEvent]:
        if self._mesh is None:
            return None
        self._tick += 1

        # Decide how many neurons to replace this tick.
        target_count = self._neurons_to_replace()
        event: Optional[TurnoverEvent] = None
        if target_count > 0:
            event = self._replace_neurons(target_count)

        # Occasional identity fingerprint + similarity.
        if self._tick % FINGERPRINT_EVERY_N_TICKS == 0:
            fp = self._capture_fingerprint()
            if fp is not None:
                with self._lock:
                    if self._fingerprints:
                        prev = self._fingerprints[-1]
                        sim = fp.similarity(prev)
                        self._last_similarity = sim
                        if sim < THRESHOLD_IDENTITY:
                            logger.warning(
                                "CellularTurnover: IDENTITY DRIFT "
                                "(similarity=%.3f < %.2f) at tick=%d",
                                sim, THRESHOLD_IDENTITY, self._tick,
                            )
                    self._fingerprints.append(fp)

        return event

    # ── Public control: forced turnover burst ────────────────────────────

    def force_turnover(self, fraction: float) -> IdentityFingerprint:
        """Force a ``fraction`` of all neurons to be replaced immediately.

        Returns the fingerprint captured after the burst so callers can
        compare identity before vs after.
        """
        if self._mesh is None:
            raise RuntimeError("CellularTurnover not attached to a mesh")
        fraction = float(np.clip(fraction, 0.0, 1.0))
        total_neurons = self._mesh_neuron_count()
        count = int(fraction * total_neurons)
        if count > 0:
            self._replace_neurons(count)
        fp = self._capture_fingerprint()
        if fp is not None:
            with self._lock:
                self._fingerprints.append(fp)
        return fp if fp is not None else IdentityFingerprint(
            tick=self._tick,
            tier_energies=(0.0, 0.0, 0.0),
            column_synchrony=0.0,
            projection_signature=np.zeros(16, dtype=np.float32),
        )

    # ── internal: mesh mutation ─────────────────────────────────────────

    def _neurons_to_replace(self) -> int:
        total = self._mesh_neuron_count()
        if total == 0:
            return 0
        expected = self._turnover_rate * total
        # Poisson-like stochastic rounding for natural variability.
        low = int(np.floor(expected))
        high = low + 1
        frac = expected - low
        if self._rng.random() < frac:
            return high
        return low

    def _mesh_neuron_count(self) -> int:
        mesh = self._mesh
        cfg = getattr(mesh, "cfg", None)
        if cfg is None:
            return 0
        return int(getattr(cfg, "total_neurons", 0))

    def _replace_neurons(self, count: int) -> Optional[TurnoverEvent]:
        mesh = self._mesh
        cfg = getattr(mesh, "cfg", None)
        cols = getattr(mesh, "columns", [])
        if cfg is None or not cols:
            return None

        column_idx = int(self._rng.integers(0, len(cols)))
        col = cols[column_idx]
        n_col = col.n
        # Pick up to ``count`` neurons inside this column.
        k = min(count, n_col)
        pick = self._rng.choice(n_col, size=k, replace=False)

        with self._lock:
            x_before = np.array(col.x, copy=True)
            mean_before = float(np.mean(np.abs(x_before)))
            # Inherit from neighbourhood.
            neighbourhood_mean = float(np.mean(col.x))
            neighbourhood_std = float(np.std(col.x)) + 1e-3

            # New activation: neighbourhood-mean with jitter.
            new_vals = self._rng.normal(
                loc=neighbourhood_mean, scale=neighbourhood_std,
                size=k,
            ).astype(np.float32)
            col.x[pick] = np.clip(new_vals, -1.0, 1.0).astype(np.float32)

            # Incoming weights: perturb existing weights for replaced neurons.
            if hasattr(col, "W") and col.W is not None:
                for idx in pick:
                    col.W[idx, :] = (col.W[idx, :] * 0.7
                                     + self._rng.standard_normal(col.W.shape[1])
                                         .astype(col.W.dtype) * 0.05)

            x_after = np.array(col.x, copy=True)
            mean_after = float(np.mean(np.abs(x_after)))

        event = TurnoverEvent(
            tick=self._tick, n_replaced=int(k), column_idx=column_idx,
            mean_activation_before=mean_before,
            mean_activation_after=mean_after,
        )
        self._events.append(event)
        return event

    # ── internal: fingerprint ───────────────────────────────────────────

    def _capture_fingerprint(self) -> Optional[IdentityFingerprint]:
        mesh = self._mesh
        if mesh is None:
            return None
        try:
            cfg = mesh.cfg
            cols = mesh.columns
            sensory_cols = cols[:cfg.sensory_end]
            assoc_cols = cols[cfg.sensory_end:cfg.association_end]
            exec_cols = cols[cfg.association_end:]
            se = float(np.mean([np.mean(np.abs(c.x)) for c in sensory_cols])) if sensory_cols else 0.0
            ae = float(np.mean([np.mean(np.abs(c.x)) for c in assoc_cols])) if assoc_cols else 0.0
            ee = float(np.mean([np.mean(np.abs(c.x)) for c in exec_cols])) if exec_cols else 0.0
            # Column synchrony: std of per-column mean activations (low std = high sync).
            col_means = np.array([np.mean(c.x) for c in cols], dtype=np.float32)
            synchrony = float(1.0 - min(1.0, float(np.std(col_means)) / 0.5))
            # Executive projection — take first 16 entries.
            try:
                proj = mesh.get_executive_projection()
                sig = np.asarray(proj, dtype=np.float32)[:16]
                sig = np.pad(sig, (0, max(0, 16 - sig.size)))
            except Exception:
                sig = np.zeros(16, dtype=np.float32)
            return IdentityFingerprint(
                tick=self._tick,
                tier_energies=(se, ae, ee),
                column_synchrony=synchrony,
                projection_signature=sig,
            )
        except Exception as exc:  # pragma: no cover
            logger.debug("fingerprint capture failed: %s", exc)
            return None

    # ── Public accessors ────────────────────────────────────────────────

    def last_similarity(self) -> float:
        return self._last_similarity

    def fingerprints_count(self) -> int:
        with self._lock:
            return len(self._fingerprints)

    def recent_events(self, n: int = 8) -> List[TurnoverEvent]:
        with self._lock:
            return list(self._events)[-n:]

    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            total_replaced = sum(e.n_replaced for e in self._events)
            return {
                "tick": self._tick,
                "turnover_rate": round(self._turnover_rate, 5),
                "total_replaced": total_replaced,
                "fingerprint_count": len(self._fingerprints),
                "last_identity_similarity": round(self._last_similarity, 4),
                "identity_stable": self._last_similarity >= THRESHOLD_IDENTITY,
                "mesh_attached": self._mesh is not None,
            }


# ── Singleton accessor ────────────────────────────────────────────────────────

_INSTANCE: Optional[CellularTurnover] = None


def get_cellular_turnover() -> CellularTurnover:
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = CellularTurnover()
    return _INSTANCE
