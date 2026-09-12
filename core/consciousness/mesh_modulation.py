"""The chemistry the mesh runs under, and who is allowed to change it.

Regional modulation, the global modulatory state, and the criticality
adjustment that keeps the mesh off both floors. Each setter publishes what it
changed while holding the lock, because a modulatory state read between the
write and the publish is a reading of two different meshes.
"""
from __future__ import annotations

import math

import numpy as np


class _CarriesModulation:
    """Lifted whole from NeuralMesh; see neural_mesh.py."""

    def _publish_modulatory_state_locked(self) -> None:
        # Imported here rather than at module level: the module these
        # came from imports this one to build the class. A call-time
        # import also still sees a test's patch of the original.
        from .neural_mesh import (
            CRITICALITY_GAIN_CEILING,
        )

        base_gain, base_plasticity, base_noise = self._base_modulatory_state
        criticality_gain, criticality_noise = self._criticality_modulatory_factors
        # The third ceiling on one quantity, and the one nobody could see. The
        # regulator clamps what it asks for, `set_criticality_adjustment`
        # clamps what it accepts, and this clamps what either of them
        # published -- so the tightest of the three decided, silently. At 3.0
        # the branching ratio reaches 0.961 against a target of 0.98, which is
        # still out of reach, so raising the other two without this one would
        # have changed nothing and looked like a fix.
        effective = (
            max(0.1, min(CRITICALITY_GAIN_CEILING, base_gain * criticality_gain)),
            max(0.0, min(5.0, base_plasticity)),
            max(0.0, min(3.0, base_noise * criticality_noise)),
        )
        self._modulatory_state = effective
        (
            self._modulatory_gain,
            self._modulatory_plasticity,
            self._modulatory_noise,
        ) = effective

    def _tier_vector(self, which: int) -> np.ndarray:
        """One multiplier per column, from the per-tier pair. 0 is gain, 1 is noise."""
        return np.array(
            [
                self._tier_modulation.get(name, (1.0, 1.0))[which]
                for name in self._tier_names
            ],
            dtype=np.float32,
        )

    def set_regional_modulation(
        self, multipliers: dict[str, tuple[float, float]] | None
    ) -> dict[str, tuple[float, float]]:
        """How much each tier scales gain and noise, on top of the global state.

        A receptor field supplies these. Passing None restores uniform, which is
        what the mesh did before it could express the difference at all: one
        scalar for every one of its 4,096 units, so dopamine arriving at the
        sensory tier and dopamine arriving at the executive tier were the same
        event. Cortex is not like that — receptor densities vary by area, and a
        transmitter's effect depends on where it lands and what receptor is
        there — and until a measurement fills these in they stay at one, which
        says the structure is unmeasured rather than guessing at it.
        """
        wanted = {"sensory": (1.0, 1.0), "association": (1.0, 1.0), "executive": (1.0, 1.0)}
        if multipliers:
            for tier, pair in multipliers.items():
                name = str(tier).rsplit(".", 1)[-1].lower()
                if name not in wanted:
                    continue
                try:
                    gain_scale = float(pair[0])
                    noise_scale = float(pair[1])
                except (TypeError, ValueError, IndexError):
                    continue
                if not (math.isfinite(gain_scale) and math.isfinite(noise_scale)):
                    continue
                wanted[name] = (
                    max(0.1, min(4.0, gain_scale)),
                    max(0.0, min(4.0, noise_scale)),
                )
        with self._modulation_lock:
            self._tier_modulation = wanted
        return dict(wanted)

    def regional_modulation(self) -> dict[str, tuple[float, float]]:
        """What each tier is currently scaling gain and noise by."""
        with self._modulation_lock:
            return dict(self._tier_modulation)

    def set_modulatory_state(
        self,
        gain: float = 1.0,
        plasticity: float = 1.0,
        noise: float = 1.0,
    ) -> None:
        """Set the neurochemical base state without erasing other controllers."""
        from .neural_mesh import (
            _clamp_float,
            _finite_float,
            _record_neural_mesh_degradation,
        )

        gain_value, gain_valid = _finite_float(gain, 1.0)
        plasticity_value, plasticity_valid = _finite_float(plasticity, 1.0)
        noise_value, noise_valid = _finite_float(noise, 1.0)
        gain_value, gain_unchanged = _clamp_float(gain_value, lower=0.1, upper=3.0)
        plasticity_value, plasticity_unchanged = _clamp_float(
            plasticity_value,
            lower=0.0,
            upper=5.0,
        )
        noise_value, noise_unchanged = _clamp_float(noise_value, lower=0.0, upper=3.0)
        if not all(
            (
                gain_valid,
                plasticity_valid,
                noise_valid,
                gain_unchanged,
                plasticity_unchanged,
                noise_unchanged,
            )
        ):
            _record_neural_mesh_degradation(
                ValueError("NeuralMesh modulatory state was non-finite or out of bounds"),
                action="normalized NeuralMesh modulatory state before applying it",
                severity="warning",
                extra={
                    "gain": gain_value,
                    "plasticity": plasticity_value,
                    "noise": noise_value,
                },
            )
        with self._modulation_lock:
            self._base_modulatory_state = (
                gain_value,
                plasticity_value,
                noise_value,
            )
            self._publish_modulatory_state_locked()

    def set_criticality_adjustment(
        self,
        *,
        gain: float = 1.0,
        noise: float = 1.0,
    ) -> None:
        """Apply bounded criticality factors over the neurochemical base state."""
        from .neural_mesh import (
            CRITICALITY_GAIN_CEILING,
            _clamp_float,
            _finite_float,
            _record_neural_mesh_degradation,
        )

        gain_value, gain_valid = _finite_float(gain, 1.0)
        noise_value, noise_valid = _finite_float(noise, 1.0)
        # The regulator's own gain clamp, and the same measurement behind it:
        # inside a ceiling of 2.0 this mesh reaches a branching ratio of 0.929
        # against a controller asking for 0.98, so the ceiling made the target
        # unreachable and the controller sat at the rail. See
        # `CriticalityConfig.gain_clamp` for the sweep. The two numbers have to
        # agree or the tighter one decides, silently.
        gain_value, gain_unchanged = _clamp_float(
            gain_value,
            lower=0.5,
            upper=CRITICALITY_GAIN_CEILING,
        )
        noise_value, noise_unchanged = _clamp_float(
            noise_value,
            lower=0.5,
            upper=2.0,
        )
        if not all((gain_valid, noise_valid, gain_unchanged, noise_unchanged)):
            _record_neural_mesh_degradation(
                ValueError("NeuralMesh criticality adjustment was non-finite or out of bounds"),
                action="normalized criticality modulation before applying it",
                severity="warning",
                extra={"gain": gain_value, "noise": noise_value},
            )
        with self._modulation_lock:
            self._criticality_modulatory_factors = (gain_value, noise_value)
            self._publish_modulatory_state_locked()

    def get_modulatory_state(self) -> dict[str, dict[str, float]]:
        """Return one coherent base, criticality, and effective modulation snapshot."""
        with self._modulation_lock:
            base_gain, base_plasticity, base_noise = self._base_modulatory_state
            criticality_gain, criticality_noise = self._criticality_modulatory_factors
            effective_gain, effective_plasticity, effective_noise = self._modulatory_state
        return {
            "base": {
                "gain": base_gain,
                "plasticity": base_plasticity,
                "noise": base_noise,
            },
            "criticality": {
                "gain": criticality_gain,
                "noise": criticality_noise,
            },
            "effective": {
                "gain": effective_gain,
                "plasticity": effective_plasticity,
                "noise": effective_noise,
            },
        }
