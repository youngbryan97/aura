"""What the field expects to happen next, and how wrong it was.

A prediction the field never checks is a number nobody is accountable for, so
the projection and the surprise that scores it live together. Surprise here is
the residual against the projection, not a distance from a target: a field that
is merely different from what somebody wanted is not surprised.
"""
from __future__ import annotations

import numpy as np


class _PredictsTheNextField:
    """Lifted whole from UnifiedField; see unified_field.py."""

    def _project_prediction(self, name: str, weights: np.ndarray, expected_dim: int) -> np.ndarray:
        # Imported here rather than at module level: the module these
        # came from imports this one to build the class. A call-time
        # import also still sees a test's patch of the original.
        from .unified_field import (
            _RECOVERABLE_FIELD_ERRORS,
            _record_unified_field_degradation,
            logger,
        )

        try:
            field = self.get_field_state()
            weights = self._normalize_matrix(
                weights,
                shape=(self.cfg.dim, expected_dim),
                name=f"W_{name}_projection",
                scale=0.1,
            )
            weight_attrs = {
                "mesh": "W_mesh",
                "neurochemical": "W_chem",
                "binding": "W_bind",
                "interoception": "W_intero",
                "substrate": "W_substrate",
            }
            if name in weight_attrs:
                setattr(self, weight_attrs[name], weights)
                self._sync_input_weight_matrix()
            prediction = np.tanh((weights.T @ field)[:expected_dim]).astype(np.float32)
            return self._safe_reshape(
                prediction,
                expected_dim,
                source=f"{name}_prediction",
                record=True,
                clip_abs=1.0,
            )
        except _RECOVERABLE_FIELD_ERRORS as exc:
            _record_unified_field_degradation(
                exc,
                action=f"returned neutral UnifiedField {name} world-model prediction",
            )
            logger.debug("UnifiedField %s world-model projection failed: %s", name, exc)
            return np.zeros(expected_dim, dtype=np.float32)

    def get_world_model_predictions(self) -> dict[str, np.ndarray]:
        """Generate predictions for what each input subsystem should produce.

        These predictions are the field's world model: its best guess about
        the next state of each input stream. Downstream systems can compare
        their actual output against these predictions to compute local
        prediction errors — making the field the upstream prior for the
        entire cognitive stack.

        Returns a dict mapping input names to predicted state vectors.
        """
        with self._lock:
            # The field's current state, projected back through each input
            # weight matrix (transposed), gives the predicted input.
            # This is the generative model: F → predicted sensory, predicted
            # chemical, predicted binding, etc.
            return {
                "mesh": self._project_prediction("mesh", self.W_mesh, self.cfg.mesh_input_dim),
                "neurochemical": self._project_prediction(
                    "neurochemical",
                    self.W_chem,
                    self.cfg.chem_input_dim,
                ),
                "binding": self._project_prediction(
                    "binding",
                    self.W_bind,
                    self.cfg.binding_input_dim,
                ),
                "interoception": self._project_prediction(
                    "interoception",
                    self.W_intero,
                    self.cfg.intero_input_dim,
                ),
                "substrate": self._project_prediction(
                    "substrate",
                    self.W_substrate,
                    self.cfg.substrate_input_dim,
                ),
            }

    def compute_world_model_surprise(self) -> float:
        """Compute how surprised the field is by its current inputs.

        This is the IWMT-style global surprise: the mismatch between
        what the field predicted and what it actually received. High
        surprise = the world isn't matching the model = act or update.
        """
        from .unified_field import (
            _record_unified_field_degradation,
        )

        predictions = self.get_world_model_predictions()
        total_error = 0.0
        n_active = 0
        actual_attrs = {
            "mesh": ("_mesh_input", self.cfg.mesh_input_dim),
            "neurochemical": ("_chem_input", self.cfg.chem_input_dim),
            "binding": ("_bind_input", self.cfg.binding_input_dim),
            "interoception": ("_intero_input", self.cfg.intero_input_dim),
            "substrate": ("_substrate_input", self.cfg.substrate_input_dim),
        }

        for name, pred in predictions.items():
            actual_attr, expected_dim = actual_attrs[name]
            actual = getattr(self, actual_attr, None)
            if actual is not None:
                actual_vec = self._safe_reshape(
                    actual,
                    expected_dim,
                    source=f"{name}_actual_for_surprise",
                    record=True,
                    clip_abs=1.0,
                )
                pred_vec = self._safe_reshape(
                    pred,
                    expected_dim,
                    source=f"{name}_prediction_for_surprise",
                    record=True,
                    clip_abs=1.0,
                )
                error = float(np.linalg.norm(pred_vec - actual_vec))
                if not np.isfinite(error):
                    _record_unified_field_degradation(
                        FloatingPointError(f"non-finite surprise for {name}"),
                        action="ignored malformed UnifiedField local surprise term",
                        severity="warning",
                    )
                    continue
                total_error += error
                n_active += 1

        surprise = total_error / max(1, n_active)
        return max(0.0, min(10.0, surprise))
