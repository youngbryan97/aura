"""core/adaptation/dimensional_expansion.py
==========================================
Covariance-Driven Geometric Layer for Open-Ended Feature Discovery.

Replaces the hardcoded 16-dimensional antigen/receptor vector space with a
dynamically expandable basis.  When systemic anomalies cannot be captured by the
existing feature axes, incremental PCA on residual variance detects the deficit
and proposes new dimensions.  Under-used dimensions are contracted to keep the
space tractable.

Integration:
    Called from ``AdaptiveImmuneSystem.present_antigen()`` on every observation.
    The expansion engine's ``current_dim`` governs all receptor/vector sizes
    system-wide.

Lifecycle of a dimension:
    Birth  — eigenvalue exceeds threshold → new axis proposed → governance check
    Growth — usage_count rises, weight evolves through immune fitness feedback
    Death  — contribution_score decays below floor after min-observations → retired
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from core.runtime.atomic_writer import atomic_write_text
from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.DimensionalExpansion")

_EPSILON = 1e-8

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class FeatureAxis:
    """A single discovered feature dimension beyond the canonical 16."""

    axis_id: str
    origin: str  # human-readable description of what triggered discovery
    projection_vector: np.ndarray  # direction in raw telemetry space
    weight: float = 0.5
    created_at: float = field(default_factory=time.time)
    usage_count: int = 0
    contribution_score: float = 0.5  # rolling EMA of explained variance
    total_observations: int = 0  # observations since this axis was born

    def to_dict(self) -> dict[str, Any]:
        return {
            "axis_id": self.axis_id,
            "origin": self.origin,
            "projection_vector": self.projection_vector.astype(float).tolist(),
            "weight": round(float(self.weight), 6),
            "created_at": self.created_at,
            "usage_count": self.usage_count,
            "contribution_score": round(float(self.contribution_score), 6),
            "total_observations": self.total_observations,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FeatureAxis:
        return cls(
            axis_id=str(data["axis_id"]),
            origin=str(data.get("origin", "restored")),
            projection_vector=np.asarray(
                data.get("projection_vector", []), dtype=np.float32
            ),
            weight=float(data.get("weight", 0.5)),
            created_at=float(data.get("created_at", time.time())),
            usage_count=int(data.get("usage_count", 0)),
            contribution_score=float(data.get("contribution_score", 0.5)),
            total_observations=int(data.get("total_observations", 0)),
        )


@dataclass
class ExpansionEvent:
    """Record of a dimension being born."""

    axis: FeatureAxis
    eigenvalue: float
    explained_variance_ratio: float
    residual_buffer_size: int
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "axis_id": self.axis.axis_id,
            "origin": self.axis.origin,
            "eigenvalue": round(float(self.eigenvalue), 6),
            "explained_variance_ratio": round(
                float(self.explained_variance_ratio), 6
            ),
            "residual_buffer_size": self.residual_buffer_size,
            "timestamp": self.timestamp,
        }


@dataclass
class ContractionEvent:
    """Record of a dimension being retired."""

    axis_id: str
    final_contribution_score: float
    total_observations: int
    timestamp: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# DynamicFeatureWeights — replaces the static _FEATURE_WEIGHTS array
# ---------------------------------------------------------------------------


class DynamicFeatureWeights:
    """Manages a variable-length feature weight vector.

    The first ``base_dim`` entries correspond to the canonical features.
    Additional entries are appended/removed as dimensions expand/contract.
    """

    def __init__(self, base_weights: np.ndarray) -> None:
        self._base = base_weights.astype(np.float32).copy()
        self._expanded: list[float] = []

    # -- query -----------------------------------------------------------------

    def get(self) -> np.ndarray:
        """Return the full weight vector (base + expanded)."""
        if not self._expanded:
            return self._base.copy()
        return np.concatenate(
            [self._base, np.asarray(self._expanded, dtype=np.float32)]
        )

    @property
    def dim(self) -> int:
        return len(self._base) + len(self._expanded)

    @property
    def base_dim(self) -> int:
        return len(self._base)

    # -- mutation --------------------------------------------------------------

    def expand(self, weight: float = 0.5) -> None:
        """Append a new dimension weight."""
        self._expanded.append(float(max(0.01, min(2.0, weight))))

    def contract(self, expanded_index: int) -> None:
        """Remove an expanded dimension by its *expanded* index (0-based)."""
        if 0 <= expanded_index < len(self._expanded):
            self._expanded.pop(expanded_index)

    def update_expanded_weight(self, expanded_index: int, weight: float) -> None:
        if 0 <= expanded_index < len(self._expanded):
            self._expanded[expanded_index] = float(max(0.01, min(2.0, weight)))

    # -- persistence -----------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "base": self._base.astype(float).tolist(),
            "expanded": list(self._expanded),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DynamicFeatureWeights:
        base = np.asarray(data.get("base", []), dtype=np.float32)
        inst = cls(base)
        inst._expanded = [float(w) for w in data.get("expanded", [])]
        return inst


# ---------------------------------------------------------------------------
# DimensionalExpansionEngine
# ---------------------------------------------------------------------------


class DimensionalExpansionEngine:
    """Monitors residual variance in the antigen space and dynamically
    expands or contracts the feature basis.

    Thread-safe: all mutations guarded by ``_lock``.
    """

    def __init__(
        self,
        initial_dim: int = 16,
        max_dim: int = 128,
        expansion_check_interval: int = 64,
        expansion_eigenvalue_threshold: float = 0.15,
        contraction_score_floor: float = 0.05,
        contraction_min_observations: int = 500,
        residual_buffer_size: int = 256,
        base_weights: np.ndarray | None = None,
    ) -> None:
        self._initial_dim = initial_dim
        self._max_dim = min(max_dim, 512)  # absolute ceiling
        self._expansion_check_interval = max(8, expansion_check_interval)
        self._expansion_eigenvalue_threshold = max(0.01, expansion_eigenvalue_threshold)
        self._contraction_score_floor = max(0.001, contraction_score_floor)
        self._contraction_min_observations = max(50, contraction_min_observations)

        self._expanded_axes: list[FeatureAxis] = []
        self._expansion_history: deque[ExpansionEvent] = deque(maxlen=256)
        self._contraction_history: deque[ContractionEvent] = deque(maxlen=128)

        # Residual ring buffer — each entry is a 1-D residual vector
        self._residual_buffer: deque[np.ndarray] = deque(
            maxlen=max(32, residual_buffer_size)
        )
        # Raw telemetry dimension (discovered on first observation)
        self._raw_dim: int | None = None

        # Observation counter drives periodic expansion checks
        self._observation_count: int = 0

        # Feature weights
        if base_weights is not None:
            self._feature_weights = DynamicFeatureWeights(base_weights)
        else:
            self._feature_weights = DynamicFeatureWeights(
                np.ones(initial_dim, dtype=np.float32) * 0.5
            )

        self._lock = threading.Lock()

    # -- public properties -----------------------------------------------------

    @property
    def current_dim(self) -> int:
        return self._initial_dim + len(self._expanded_axes)

    @property
    def expanded_count(self) -> int:
        return len(self._expanded_axes)

    @property
    def feature_weights(self) -> DynamicFeatureWeights:
        return self._feature_weights

    # -- main entry point ------------------------------------------------------

    def evaluate_expansion(
        self,
        raw_telemetry: dict[str, Any],
        base_vector: np.ndarray,
    ) -> tuple[np.ndarray, list[ExpansionEvent]]:
        """Evaluate whether the antigen space needs expansion.

        Args:
            raw_telemetry: The original event dict with all numeric fields.
            base_vector: The canonical 16-dim antigen vector already computed.

        Returns:
            (expanded_vector, expansion_events)
            expanded_vector has shape ``(current_dim,)`` — base dims + expanded.
            expansion_events lists any newly born dimensions.
        """
        with self._lock:
            self._observation_count += 1

            # 1. Convert raw telemetry to a numeric vector
            raw_vec = self._telemetry_to_raw_vector(raw_telemetry)

            # 2. Compute projection values for existing expanded axes
            expanded_values = self._project_onto_expanded(raw_vec)

            # 3. Compute residual (what the current basis cannot capture)
            residual = self._compute_residual(raw_vec)
            if residual is not None and np.any(np.isfinite(residual)):
                self._residual_buffer.append(residual)

            # 4. Periodic expansion check
            new_events: list[ExpansionEvent] = []
            if (
                self._observation_count % self._expansion_check_interval == 0
                and len(self._residual_buffer) >= 16
            ):
                new_events = self._check_expansion()

            # 5. Update contribution scores for existing axes
            self._update_contribution_scores(raw_vec)

            # 6. Build the expanded vector
            full_vector = np.zeros(self.current_dim, dtype=np.float32)
            base_len = min(len(base_vector), self._initial_dim)
            full_vector[:base_len] = base_vector[:base_len]
            for i, val in enumerate(expanded_values):
                idx = self._initial_dim + i
                if idx < len(full_vector):
                    full_vector[idx] = float(np.clip(val, 0.0, 1.0))

            return np.clip(full_vector, 0.0, 1.0), new_events

    def evaluate_contraction(self) -> list[str]:
        """Check if any expanded dimensions should be retired.

        Returns:
            List of retired axis IDs.
        """
        with self._lock:
            retired: list[str] = []
            to_remove: list[int] = []

            for i, axis in enumerate(self._expanded_axes):
                if axis.total_observations < self._contraction_min_observations:
                    continue
                if axis.contribution_score < self._contraction_score_floor:
                    retired.append(axis.axis_id)
                    to_remove.append(i)
                    self._contraction_history.append(
                        ContractionEvent(
                            axis_id=axis.axis_id,
                            final_contribution_score=axis.contribution_score,
                            total_observations=axis.total_observations,
                        )
                    )
                    logger.info(
                        "Contracting dimension %s (contribution=%.4f, obs=%d)",
                        axis.axis_id,
                        axis.contribution_score,
                        axis.total_observations,
                    )

            # Remove in reverse order to preserve indices
            for i in reversed(to_remove):
                self._expanded_axes.pop(i)
                self._feature_weights.contract(i)

            return retired

    def resize_vector(
        self, vector: np.ndarray, target_dim: int | None = None
    ) -> np.ndarray:
        """Resize a vector to match ``current_dim`` (or a given target).

        Pads with zeros if shorter, truncates if longer.
        """
        dim = target_dim if target_dim is not None else self.current_dim
        if len(vector) == dim:
            return vector
        result = np.zeros(dim, dtype=np.float32)
        copy_len = min(len(vector), dim)
        result[:copy_len] = vector[:copy_len]
        return result

    def record_fitness_feedback(self, axis_id: str, fitness_delta: float) -> None:
        """Update an expanded axis's weight based on immune fitness feedback."""
        with self._lock:
            for i, axis in enumerate(self._expanded_axes):
                if axis.axis_id == axis_id:
                    # Positive fitness → increase weight, negative → decrease
                    old_w = axis.weight
                    axis.weight = float(
                        np.clip(old_w + 0.05 * fitness_delta, 0.05, 1.5)
                    )
                    self._feature_weights.update_expanded_weight(i, axis.weight)
                    break

    # -- internal: telemetry conversion ----------------------------------------

    def _telemetry_to_raw_vector(self, raw_telemetry: dict[str, Any]) -> np.ndarray:
        """Convert raw telemetry dict to a numeric vector.

        Extracts all numeric values, flattens nested dicts one level deep,
        and normalizes each value to [0, 1] via sigmoid-like squashing.
        """
        values: list[float] = []
        for key, val in sorted(raw_telemetry.items()):
            if isinstance(val, (int, float)):
                fv = float(val)
                if math.isfinite(fv):
                    values.append(fv)
            elif isinstance(val, dict):
                for sub_key, sub_val in sorted(val.items()):
                    if isinstance(sub_val, (int, float)):
                        fv = float(sub_val)
                        if math.isfinite(fv):
                            values.append(fv)
            elif isinstance(val, (list, tuple)):
                for item in val[:16]:  # cap list extraction
                    if isinstance(item, (int, float)):
                        fv = float(item)
                        if math.isfinite(fv):
                            values.append(fv)

        if not values:
            return np.zeros(1, dtype=np.float32)

        vec = np.asarray(values, dtype=np.float32)
        # Sigmoid squash to [0, 1] for normalization
        vec = 1.0 / (1.0 + np.exp(-np.clip(vec, -10.0, 10.0)))
        vec = np.nan_to_num(vec, nan=0.5, posinf=1.0, neginf=0.0)

        # Track raw dimension
        if self._raw_dim is None:
            self._raw_dim = len(vec)
        return vec

    # -- internal: projection & residual ---------------------------------------

    def _project_onto_expanded(self, raw_vec: np.ndarray) -> list[float]:
        """Project raw telemetry onto each expanded axis."""
        projections: list[float] = []
        for axis in self._expanded_axes:
            pv = axis.projection_vector
            if len(pv) == 0:
                projections.append(0.0)
                continue
            # Resize for compatibility
            min_len = min(len(raw_vec), len(pv))
            if min_len == 0:
                projections.append(0.0)
                continue
            dot = float(np.dot(raw_vec[:min_len], pv[:min_len]))
            norm = float(np.linalg.norm(pv[:min_len]))
            if norm > _EPSILON:
                proj = dot / norm
            else:
                proj = 0.0
            # Sigmoid to [0, 1]
            proj = 1.0 / (1.0 + math.exp(-float(np.clip(proj, -10.0, 10.0))))
            projections.append(proj)
            axis.usage_count += 1
        return projections

    def _compute_residual(self, raw_vec: np.ndarray) -> np.ndarray | None:
        """Compute what the current basis cannot explain."""
        if len(raw_vec) < 2:
            return None

        # Build the basis matrix: one row per existing axis
        # (base dimensions are assumed to capture their own variance —
        #  we only measure residual w.r.t. expanded axes)
        if not self._expanded_axes:
            # No expanded axes yet — the full raw vector is residual
            return raw_vec.copy()

        basis_vectors: list[np.ndarray] = []
        for axis in self._expanded_axes:
            pv = axis.projection_vector
            if len(pv) > 0:
                # Resize to match raw_vec
                resized = np.zeros(len(raw_vec), dtype=np.float32)
                copy_len = min(len(pv), len(raw_vec))
                resized[:copy_len] = pv[:copy_len]
                basis_vectors.append(resized)

        if not basis_vectors:
            return raw_vec.copy()

        basis = np.stack(basis_vectors, axis=0)  # shape: (n_axes, raw_dim)
        # Project raw_vec onto the span of basis
        try:
            # Least-squares projection: proj = basis.T @ (basis @ basis.T)^-1 @ basis @ x
            # Simpler: proj = sum of individual projections (axes may not be orthogonal)
            projection = np.zeros_like(raw_vec)
            for bv in basis_vectors:
                norm_sq = float(np.dot(bv, bv))
                if norm_sq > _EPSILON:
                    coeff = float(np.dot(raw_vec, bv)) / norm_sq
                    projection += coeff * bv
            residual = raw_vec - projection
            return np.nan_to_num(residual, nan=0.0, posinf=0.0, neginf=0.0)
        except (np.linalg.LinAlgError, ValueError, FloatingPointError) as exc:
            record_degradation("dimensional_expansion", exc)
            logger.debug("Residual computation failed: %s", exc)
            return raw_vec.copy()

    # -- internal: expansion check (incremental PCA) ---------------------------

    def _check_expansion(self) -> list[ExpansionEvent]:
        """Run incremental PCA on the residual buffer.

        If the top eigenvalue exceeds ``threshold * mean(existing eigenvalues)``,
        a new dimension is born.
        """
        if self.current_dim >= self._max_dim:
            return []

        residuals = list(self._residual_buffer)
        if len(residuals) < 16:
            return []

        # Pad/truncate all residuals to the same length
        max_len = max(len(r) for r in residuals)
        if max_len < 2:
            return []

        padded = np.zeros((len(residuals), max_len), dtype=np.float32)
        for i, r in enumerate(residuals):
            copy_len = min(len(r), max_len)
            padded[i, :copy_len] = r[:copy_len]

        # Remove NaN/Inf
        padded = np.nan_to_num(padded, nan=0.0, posinf=0.0, neginf=0.0)

        # Compute covariance matrix
        try:
            mean_vec = padded.mean(axis=0)
            centered = padded - mean_vec
            cov = (centered.T @ centered) / max(len(centered) - 1, 1)
            cov = np.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)

            # Eigendecomposition (symmetric)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            # eigh returns ascending order; reverse for descending
            eigenvalues = eigenvalues[::-1]
            eigenvectors = eigenvectors[:, ::-1]

            # Filter out negative/tiny eigenvalues
            valid_mask = eigenvalues > _EPSILON
            if not np.any(valid_mask):
                return []

            valid_eigenvalues = eigenvalues[valid_mask]
            mean_eigenvalue = float(np.mean(valid_eigenvalues))
            if mean_eigenvalue < _EPSILON:
                return []

            # Check if top eigenvalue exceeds threshold
            new_events: list[ExpansionEvent] = []
            max_new_dims = min(3, self._max_dim - self.current_dim)  # at most 3 per check

            for k in range(min(max_new_dims, len(valid_eigenvalues))):
                ev = float(valid_eigenvalues[k])
                ratio = ev / max(mean_eigenvalue, _EPSILON)

                if ratio <= (1.0 + self._expansion_eigenvalue_threshold):
                    break  # remaining eigenvalues will be smaller

                # Governance gate
                if not self._governance_approve_expansion(ratio):
                    logger.info(
                        "Dimension expansion denied by governance (ratio=%.3f)", ratio
                    )
                    continue

                # Birth a new dimension
                proj_vec = eigenvectors[:, k].astype(np.float32).copy()
                proj_vec = np.nan_to_num(proj_vec, nan=0.0, posinf=0.0, neginf=0.0)
                norm = float(np.linalg.norm(proj_vec))
                if norm > _EPSILON:
                    proj_vec /= norm

                axis_id = f"expanded_{self._initial_dim + len(self._expanded_axes)}"
                origin = (
                    f"residual_pca_ev={ev:.4f}_ratio={ratio:.3f}_"
                    f"buf={len(residuals)}_obs={self._observation_count}"
                )

                new_axis = FeatureAxis(
                    axis_id=axis_id,
                    origin=origin,
                    projection_vector=proj_vec,
                    weight=min(1.0, 0.3 + 0.2 * ratio),
                )
                self._expanded_axes.append(new_axis)
                self._feature_weights.expand(new_axis.weight)

                event = ExpansionEvent(
                    axis=new_axis,
                    eigenvalue=ev,
                    explained_variance_ratio=ratio,
                    residual_buffer_size=len(residuals),
                )
                new_events.append(event)
                self._expansion_history.append(event)

                logger.info(
                    "🧬 Dimension expanded: %s (eigenvalue=%.4f, ratio=%.3f, dim now=%d)",
                    axis_id,
                    ev,
                    ratio,
                    self.current_dim,
                )

            return new_events

        except (np.linalg.LinAlgError, ValueError, FloatingPointError) as exc:
            record_degradation("dimensional_expansion", exc)
            logger.debug("Expansion PCA failed: %s", exc)
            return []

    def _update_contribution_scores(self, raw_vec: np.ndarray) -> None:
        """Update rolling EMA contribution scores for each expanded axis."""
        for axis in self._expanded_axes:
            axis.total_observations += 1
            pv = axis.projection_vector
            if len(pv) == 0 or len(raw_vec) == 0:
                continue
            min_len = min(len(raw_vec), len(pv))
            dot = float(np.dot(raw_vec[:min_len], pv[:min_len]))
            norm_pv = float(np.linalg.norm(pv[:min_len]))
            norm_rv = float(np.linalg.norm(raw_vec[:min_len]))
            if norm_pv > _EPSILON and norm_rv > _EPSILON:
                # Contribution = abs(cosine similarity)
                contribution = abs(dot / (norm_pv * norm_rv))
            else:
                contribution = 0.0
            # EMA update (alpha=0.02 for slow adaptation)
            axis.contribution_score = 0.98 * axis.contribution_score + 0.02 * contribution

    def _governance_approve_expansion(self, variance_ratio: float) -> bool:
        """Check with the Will/Constitution if expansion is allowed."""
        try:
            from core.will import ActionDomain, get_will

            decision = get_will().decide(
                content=f"Dimensional expansion: variance_ratio={variance_ratio:.3f}",
                source="dimensional_expansion_engine",
                domain=ActionDomain.STATE_MUTATION,
                priority=min(0.6, 0.3 + 0.1 * variance_ratio),
                context={
                    "current_dim": self.current_dim,
                    "max_dim": self._max_dim,
                    "variance_ratio": variance_ratio,
                },
            )
            return bool(decision.is_approved())
        except (ImportError, AttributeError, RuntimeError) as exc:
            record_degradation("dimensional_expansion", exc)
            logger.debug("Governance check unavailable; allowing expansion: %s", exc)
            return True  # fail-open for expansion (bounded by max_dim anyway)

    # -- persistence -----------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "initial_dim": self._initial_dim,
            "max_dim": self._max_dim,
            "expansion_check_interval": self._expansion_check_interval,
            "expansion_eigenvalue_threshold": self._expansion_eigenvalue_threshold,
            "contraction_score_floor": self._contraction_score_floor,
            "contraction_min_observations": self._contraction_min_observations,
            "observation_count": self._observation_count,
            "raw_dim": self._raw_dim,
            "expanded_axes": [a.to_dict() for a in self._expanded_axes],
            "feature_weights": self._feature_weights.to_dict(),
            "expansion_history": [e.to_dict() for e in list(self._expansion_history)[-64:]],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DimensionalExpansionEngine:
        base_weights_data = data.get("feature_weights", {}).get("base")
        base_weights = (
            np.asarray(base_weights_data, dtype=np.float32)
            if base_weights_data
            else None
        )
        engine = cls(
            initial_dim=int(data.get("initial_dim", 16)),
            max_dim=int(data.get("max_dim", 128)),
            expansion_check_interval=int(data.get("expansion_check_interval", 64)),
            expansion_eigenvalue_threshold=float(
                data.get("expansion_eigenvalue_threshold", 0.15)
            ),
            contraction_score_floor=float(data.get("contraction_score_floor", 0.05)),
            contraction_min_observations=int(
                data.get("contraction_min_observations", 500)
            ),
            base_weights=base_weights,
        )
        engine._observation_count = int(data.get("observation_count", 0))
        engine._raw_dim = data.get("raw_dim")
        engine._expanded_axes = [
            FeatureAxis.from_dict(a) for a in data.get("expanded_axes", [])
        ]
        # Restore feature weights from persisted expanded list
        fw_data = data.get("feature_weights")
        if fw_data:
            engine._feature_weights = DynamicFeatureWeights.from_dict(fw_data)
        return engine

    # -- diagnostics -----------------------------------------------------------

    def get_status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "current_dim": self.current_dim,
                "initial_dim": self._initial_dim,
                "max_dim": self._max_dim,
                "expanded_count": len(self._expanded_axes),
                "observation_count": self._observation_count,
                "residual_buffer_size": len(self._residual_buffer),
                "expansion_events_total": len(self._expansion_history),
                "contraction_events_total": len(self._contraction_history),
                "axes": [
                    {
                        "axis_id": a.axis_id,
                        "weight": round(a.weight, 4),
                        "contribution": round(a.contribution_score, 4),
                        "usage": a.usage_count,
                        "observations": a.total_observations,
                    }
                    for a in self._expanded_axes
                ],
            }
