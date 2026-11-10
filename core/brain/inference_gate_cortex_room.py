"""Whether her cortex is loaded, the memory it needs to load, and the fallback ladder below it.

Lifted whole out of `inference_gate`. Every name taken from it is imported at
CALL time: that module imports this one to build the class, and a test that
patches a name on it has to reach the code that reads it.
"""
from __future__ import annotations

import os


class _KnowsWhatTheCortexNeeds:
    """Lifted whole out of InferenceGate; see inference_gate.py."""

    def _primary_lane_ready(self) -> bool:
        """Whether the primary worker is initialized and available now."""
        from .inference_gate import (
            _INFERENCE_RECOVERABLE_ERRORS,
            logger,
        )

        client = getattr(self, "_mlx_client", None)
        if client is None or not hasattr(client, "is_alive"):
            return False
        try:
            return bool(client.is_alive())
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            logger.debug("Client liveness unreadable, reporting the lane not ready: %s", exc)
            return False

    def _memory_blocks_primary_load(self) -> bool:
        """Whether free memory is below what the primary lane needs to load.

        Fails SAFE: an unreadable memory probe returns False, i.e. keep the
        ladder. Shedding the fallback on a guess is the failure this whole
        branch exists to prevent.
        """
        from .inference_gate import (
            _INFERENCE_RECOVERABLE_ERRORS,
            logger,
        )

        if self._primary_lane_ready():
            return False
        try:
            from core.utils.memory_monitor import get_memory_pressure_snapshot

            available_gb = float(get_memory_pressure_snapshot().available_gb)
        except _INFERENCE_RECOVERABLE_ERRORS as exc:
            logger.debug("Primary-load memory probe unavailable: %s", exc)
            return False
        cortex_reserve_gb = self._primary_load_required_gb()
        return available_gb < cortex_reserve_gb

    def _primary_load_required_gb(self) -> float:
        """Return the admission threshold for the configured primary model."""
        from .inference_gate import (
            _INFERENCE_RECOVERABLE_ERRORS,
            logger,
        )

        client = getattr(self, "_mlx_client", None)
        model_path = str(getattr(client, "model_path", "") or "")
        if model_path:
            try:
                from core.brain.llm.mlx_client import _model_load_min_available_gb

                return float(_model_load_min_available_gb(model_path))
            except _INFERENCE_RECOVERABLE_ERRORS as exc:
                logger.debug("Primary model-derived load threshold unavailable: %s", exc)
                return self._env_float(
                    "AURA_MLX_32B_LOAD_MIN_AVAILABLE_GB", 24.0
                )
        return self._env_float("AURA_MLX_32B_LOAD_MIN_AVAILABLE_GB", 24.0)

    def _fallback_ladder_paths(self) -> frozenset[str]:
        """Model paths that form the escalation ladder below the cortex.

        These answer a foreground turn when the primary lane is warming, so a
        foreground-protection shed must leave them alone.
        """
        paths: set[str] = set()
        for attr in (
            "_brainstem_client",
            "_reflex_client",
            "_fallback_client",
            "_secondary_client",
            "_tertiary_client",
        ):
            client = getattr(self, attr, None)
            path = getattr(client, "model_path", None) or getattr(client, "_model_path", None)
            if path:
                paths.add(str(path))
        for env_name in (
            "AURA_MLX_BRAINSTEM_MODEL",
            "AURA_MLX_REFLEX_MODEL",
            "AURA_MLX_FALLBACK_MODEL",
        ):
            value = os.environ.get(env_name)
            if value:
                paths.add(str(value))
        return frozenset(paths)

