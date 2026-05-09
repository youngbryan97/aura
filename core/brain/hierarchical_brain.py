"""core/brain/hierarchical_brain.py -- Hierarchical Growing Brain
================================================================
Multi-region growing neural architecture with region birth/pruning
based on capacity pressure.

Architecture:
  - Regions are independent processing units with local connectivity
  - Each region has a capacity (max neurons) and current utilization
  - When a region's utilization exceeds a threshold, it splits (births)
  - When utilization drops below a threshold, it prunes
  - Inter-region connections form a directed graph

This is NOT a replacement for the base LLM — it is a supplementary
neural architecture that processes substrate state, provides
compositional representations, and feeds into the cognitive loop.

Design principles:
  - Deterministic: fixed seed initialization
  - Bounded: maximum total neurons, maximum regions
  - CPU-only: numpy operations
  - Observable: full telemetry on region health
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("Aura.HierarchicalBrain")

_DATA_DIR = Path.home() / ".aura" / "data" / "hierarchical_brain"
_STATE_PATH = _DATA_DIR / "brain_state.npz"


@dataclass
class BrainRegionConfig:
    """Configuration for a single brain region."""
    name: str
    neuron_count: int = 32
    max_neurons: int = 128
    min_neurons: int = 8
    birth_threshold: float = 0.85   # Split when utilization exceeds this
    prune_threshold: float = 0.15   # Prune when utilization drops below this
    seed: int = 0


@dataclass
class RegionMetrics:
    """Runtime metrics for a brain region."""
    utilization: float = 0.0          # Current utilization 0-1
    energy: float = 0.0               # State energy (L2 norm / sqrt(dim))
    mean_activation: float = 0.0      # Mean absolute activation
    max_activation: float = 0.0       # Max absolute activation
    connection_count: int = 0         # Number of inter-region connections


class BrainRegion:
    """A single processing region in the hierarchical brain."""

    def __init__(self, config: BrainRegionConfig) -> None:
        self.config = config
        self.name = config.name
        self._rng = np.random.default_rng(config.seed)

        n = config.neuron_count
        # Recurrent connectivity (sparse, anti-symmetric for oscillation)
        raw = self._rng.standard_normal((n, n)).astype(np.float32) * 0.1
        mask = self._rng.random((n, n)) < 0.3
        self.W = (raw * mask).astype(np.float32)
        np.fill_diagonal(self.W, -0.15)

        # State
        self.state = np.zeros(n, dtype=np.float32)
        self.bias = np.zeros(n, dtype=np.float32)

        # Input projection (will be resized when connections are made)
        self._input_proj: Optional[np.ndarray] = None
        self._output_proj: Optional[np.ndarray] = None

        # Metrics
        self._step_count = 0
        self._activation_history: Deque[float] = deque(maxlen=50)

    @property
    def neuron_count(self) -> int:
        return self.state.size

    def step(self, external_input: Optional[np.ndarray] = None, dt: float = 0.05) -> np.ndarray:
        """Run one integration step."""
        # Compute drive
        drive = self.W @ self.state + self.bias

        # Add external input if provided
        if external_input is not None:
            inp = np.asarray(external_input, dtype=np.float32).ravel()
            if inp.size != self.neuron_count:
                # Project through input projection
                if self._input_proj is not None and self._input_proj.shape[1] == inp.size:
                    inp = self._input_proj @ inp
                else:
                    padded = np.zeros(self.neuron_count, dtype=np.float32)
                    n = min(inp.size, self.neuron_count)
                    padded[:n] = inp[:n]
                    inp = padded
            drive += 0.5 * inp

        # Integrate (Euler)
        dx = (-0.05 * self.state + np.tanh(drive)) * dt
        self.state = self.state + dx

        # Soft saturation
        norm = float(np.linalg.norm(self.state))
        max_norm = math.sqrt(self.neuron_count)
        if norm > max_norm:
            self.state *= max_norm / norm

        self._step_count += 1
        self._activation_history.append(float(np.mean(np.abs(self.state))))

        return self.state.copy()

    def get_metrics(self) -> RegionMetrics:
        """Get current metrics for this region."""
        abs_state = np.abs(self.state)
        mean_act = float(np.mean(abs_state)) if self.state.size > 0 else 0.0
        max_act = float(np.max(abs_state)) if self.state.size > 0 else 0.0

        # Utilization: fraction of neurons with significant activation
        active = np.sum(abs_state > 0.1)
        utilization = float(active / max(1, self.neuron_count))

        energy = float(np.linalg.norm(self.state) / math.sqrt(max(1, self.neuron_count)))

        return RegionMetrics(
            utilization=utilization,
            energy=energy,
            mean_activation=mean_act,
            max_activation=max_act,
        )

    def get_output(self) -> np.ndarray:
        """Get the region's output vector (for inter-region connections)."""
        return np.tanh(self.state).copy()

    def resize(self, new_size: int) -> None:
        """Resize the region (grow or shrink neurons)."""
        new_size = max(self.config.min_neurons, min(self.config.max_neurons, new_size))
        if new_size == self.neuron_count:
            return

        old_n = self.neuron_count
        old_state = self.state
        old_W = self.W

        self.state = np.zeros(new_size, dtype=np.float32)
        self.bias = np.zeros(new_size, dtype=np.float32)
        n = min(old_n, new_size)
        self.state[:n] = old_state[:n]

        self.W = np.zeros((new_size, new_size), dtype=np.float32)
        self.W[:n, :n] = old_W[:n, :n]

        # Initialize new neurons
        if new_size > old_n:
            new_W = self._rng.standard_normal((new_size - old_n, new_size)).astype(np.float32) * 0.05
            self.W[old_n:, :] = new_W
            self.W[:, old_n:] = self._rng.standard_normal((new_size, new_size - old_n)).astype(np.float32) * 0.05
            np.fill_diagonal(self.W, -0.15)

        logger.info("Region '%s' resized: %d → %d neurons", self.name, old_n, new_size)


class HierarchicalBrain:
    """Multi-region growing brain with dynamic topology.

    Usage:
        brain = get_hierarchical_brain()
        output = brain.step(substrate_state)
        metrics = brain.get_status()
    """

    MAX_REGIONS = 16
    MAX_TOTAL_NEURONS = 1024

    # Default regions
    DEFAULT_REGIONS = [
        BrainRegionConfig(name="sensory", neuron_count=32, seed=100),
        BrainRegionConfig(name="association", neuron_count=64, seed=200),
        BrainRegionConfig(name="executive", neuron_count=32, seed=300),
        BrainRegionConfig(name="affective", neuron_count=16, seed=400),
    ]

    # Default connections (from → to)
    DEFAULT_CONNECTIONS = [
        ("sensory", "association"),
        ("association", "executive"),
        ("affective", "association"),
        ("association", "affective"),
        ("executive", "sensory"),  # Top-down feedback
    ]

    def __init__(self) -> None:
        self._regions: Dict[str, BrainRegion] = {}
        self._connections: List[Tuple[str, str, np.ndarray]] = []  # (from, to, weight_matrix)
        self._step_count: int = 0
        self._birth_count: int = 0
        self._prune_count: int = 0
        self._rng = np.random.default_rng(42)

        _DATA_DIR.mkdir(parents=True, exist_ok=True)

        # Initialize default regions
        for rc in self.DEFAULT_REGIONS:
            self._regions[rc.name] = BrainRegion(rc)

        # Initialize default connections
        for from_name, to_name in self.DEFAULT_CONNECTIONS:
            self._add_connection(from_name, to_name)

        self._load()
        logger.info(
            "HierarchicalBrain initialized: %d regions, %d connections, %d total neurons",
            len(self._regions),
            len(self._connections),
            self.total_neurons,
        )

    @property
    def total_neurons(self) -> int:
        return sum(r.neuron_count for r in self._regions.values())

    def _add_connection(self, from_name: str, to_name: str) -> None:
        """Add a connection between two regions."""
        from_region = self._regions.get(from_name)
        to_region = self._regions.get(to_name)
        if from_region is None or to_region is None:
            return

        # Connection weight matrix
        W = self._rng.standard_normal(
            (to_region.neuron_count, from_region.neuron_count)
        ).astype(np.float32) * 0.1

        self._connections.append((from_name, to_name, W))

    # ── Core API ────────────────────────────────────────────────────────

    def step(
        self,
        substrate_state: Optional[np.ndarray] = None,
        dt: float = 0.05,
    ) -> Dict[str, np.ndarray]:
        """Run one step of the hierarchical brain.

        Args:
            substrate_state: External input from the continuous substrate
            dt: Integration timestep

        Returns:
            Dict mapping region name to output vector
        """
        outputs: Dict[str, np.ndarray] = {}

        # First: compute inter-region inputs
        region_inputs: Dict[str, np.ndarray] = {}
        for from_name, to_name, W in self._connections:
            from_region = self._regions.get(from_name)
            to_region = self._regions.get(to_name)
            if from_region is None or to_region is None:
                continue

            # Get output from source region
            source_output = from_region.get_output()

            # Project through connection weights
            # Handle dimension mismatches gracefully
            if W.shape[1] == source_output.size and W.shape[0] == to_region.neuron_count:
                projected = W @ source_output
            else:
                projected = np.zeros(to_region.neuron_count, dtype=np.float32)
                n = min(W.shape[1], source_output.size)
                m = min(W.shape[0], to_region.neuron_count)
                projected[:m] = (W[:m, :n] @ source_output[:n])

            if to_name in region_inputs:
                region_inputs[to_name] += projected
            else:
                region_inputs[to_name] = projected

        # Inject substrate state into sensory region
        if substrate_state is not None and "sensory" in self._regions:
            sensory = self._regions["sensory"]
            sub = np.asarray(substrate_state, dtype=np.float32).ravel()
            if "sensory" in region_inputs:
                n = min(sub.size, sensory.neuron_count)
                region_inputs["sensory"][:n] += sub[:n] * 0.3
            else:
                padded = np.zeros(sensory.neuron_count, dtype=np.float32)
                n = min(sub.size, sensory.neuron_count)
                padded[:n] = sub[:n] * 0.3
                region_inputs["sensory"] = padded

        # Step each region
        for name, region in self._regions.items():
            ext_input = region_inputs.get(name)
            output = region.step(ext_input, dt=dt)
            outputs[name] = output

        self._step_count += 1

        # Check for births/prunes every 100 steps
        if self._step_count % 100 == 0:
            self._check_topology()

        # Auto-persist periodically
        if self._step_count % 1000 == 0:
            self._save()

        return outputs

    def get_composite_output(self) -> np.ndarray:
        """Get a single composite output from all regions."""
        vectors = []
        for region in self._regions.values():
            vectors.append(region.get_output())
        if not vectors:
            return np.zeros(32, dtype=np.float32)
        # Concatenate and project to fixed dimension
        composite = np.concatenate(vectors)
        # Hash-project to 64-dim for consistency
        target_dim = 64
        if composite.size > target_dim:
            # Simple averaging projection
            stride = composite.size // target_dim
            result = np.zeros(target_dim, dtype=np.float32)
            for i in range(target_dim):
                start = i * stride
                end = min(start + stride, composite.size)
                result[i] = float(np.mean(composite[start:end]))
            return result
        else:
            padded = np.zeros(target_dim, dtype=np.float32)
            padded[:composite.size] = composite
            return padded

    def _check_topology(self) -> None:
        """Check if any regions should be split or pruned."""
        births: List[str] = []
        prunes: List[str] = []

        for name, region in self._regions.items():
            metrics = region.get_metrics()

            if (metrics.utilization > region.config.birth_threshold
                    and region.neuron_count < region.config.max_neurons
                    and len(self._regions) < self.MAX_REGIONS
                    and self.total_neurons < self.MAX_TOTAL_NEURONS):
                births.append(name)

            elif (metrics.utilization < region.config.prune_threshold
                  and region.neuron_count > region.config.min_neurons
                  and self._step_count > 1000):
                prunes.append(name)

        for name in births:
            self._birth_region(name)

        for name in prunes:
            self._prune_region(name)

    def _birth_region(self, parent_name: str) -> None:
        """Split a region by growing it."""
        region = self._regions.get(parent_name)
        if region is None:
            return

        new_size = min(region.config.max_neurons, region.neuron_count + 16)
        region.resize(new_size)
        self._birth_count += 1

        logger.info(
            "Region birth: '%s' grew to %d neurons (total=%d)",
            parent_name, new_size, self.total_neurons,
        )

    def _prune_region(self, region_name: str) -> None:
        """Shrink a region by removing inactive neurons."""
        region = self._regions.get(region_name)
        if region is None:
            return

        new_size = max(region.config.min_neurons, region.neuron_count - 8)
        region.resize(new_size)
        self._prune_count += 1

        logger.info(
            "Region pruned: '%s' shrunk to %d neurons (total=%d)",
            region_name, new_size, self.total_neurons,
        )

    # ── Persistence ─────────────────────────────────────────────────────

    def _save(self) -> None:
        """Persist brain state."""
        try:
            save_dict = {
                "step_count": np.array([self._step_count]),
                "birth_count": np.array([self._birth_count]),
                "prune_count": np.array([self._prune_count]),
            }
            for name, region in self._regions.items():
                save_dict[f"region_{name}_state"] = region.state
                save_dict[f"region_{name}_W"] = region.W
                save_dict[f"region_{name}_bias"] = region.bias

            np.savez_compressed(str(_STATE_PATH), **save_dict)
            logger.debug("Brain state saved (step %d)", self._step_count)
        except Exception as exc:
            logger.debug("Brain state save failed: %s", exc)

    def _load(self) -> None:
        """Load persisted brain state."""
        try:
            if not _STATE_PATH.exists():
                return
            data = np.load(str(_STATE_PATH), allow_pickle=False)
            self._step_count = int(data.get("step_count", [0])[0])
            self._birth_count = int(data.get("birth_count", [0])[0])
            self._prune_count = int(data.get("prune_count", [0])[0])

            for name, region in self._regions.items():
                state_key = f"region_{name}_state"
                W_key = f"region_{name}_W"
                bias_key = f"region_{name}_bias"

                if state_key in data and data[state_key].size == region.neuron_count:
                    region.state = data[state_key].astype(np.float32)
                    if W_key in data and data[W_key].shape == region.W.shape:
                        region.W = data[W_key].astype(np.float32)
                    if bias_key in data and data[bias_key].size == region.neuron_count:
                        region.bias = data[bias_key].astype(np.float32)

            logger.info("Brain state restored (step %d)", self._step_count)
        except Exception as exc:
            logger.debug("Brain state load failed: %s", exc)

    # ── Public API ──────────────────────────────────────────────────────

    def get_status(self) -> Dict[str, Any]:
        """Return brain status for observability."""
        region_metrics = {}
        for name, region in self._regions.items():
            m = region.get_metrics()
            region_metrics[name] = {
                "neuron_count": region.neuron_count,
                "utilization": round(m.utilization, 4),
                "energy": round(m.energy, 4),
                "mean_activation": round(m.mean_activation, 4),
            }

        return {
            "step_count": self._step_count,
            "region_count": len(self._regions),
            "connection_count": len(self._connections),
            "total_neurons": self.total_neurons,
            "birth_count": self._birth_count,
            "prune_count": self._prune_count,
            "regions": region_metrics,
        }


# ── Singleton ───────────────────────────────────────────────────────────────

_instance: Optional[HierarchicalBrain] = None


def get_hierarchical_brain() -> HierarchicalBrain:
    """Get or create the singleton HierarchicalBrain."""
    global _instance
    if _instance is None:
        _instance = HierarchicalBrain()
    return _instance
