"""core/brain/personality_bridge.py — Systemic Influence Bridge
Links LLM-driven PsychState to MuJoCo Physical Dynamics.

Affect is a *model output*; physics constants are a live simulation's inputs.
Everything crossing that boundary is therefore validated, bounded, applied
under whatever synchronization the body offers, and reported — because an
unbounded or NaN affect value became stiffness and damping directly, and a
silent skip was indistinguishable from healthy neutral embodiment.

CP126 89fa9c28 / fe8e6eee / d38c257a / 047a0b0b / 4995f0ab / 288b0d62.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from typing import Any

from core.runtime.errors import record_degradation
from core.runtime.numeric_safety import validated_scalar
from core.runtime.service_registry import get_runtime_service
from core.state.aura_state import AffectVector

logger = logging.getLogger("Aura.PersonalityBridge")

#: Safe operating envelope for every constant this bridge can set. CP126
#: 89fa9c28: negative, huge, NaN or infinite affect propagated straight into
#: stiffness, damping, jitter, tilt and emissive parameters.
PHYSICS_BOUNDS: dict[str, tuple[float, float]] = {
    "stiffness_mult": (0.25, 4.0),
    "damping_mult": (0.10, 3.0),
    "jitter": (0.0, 0.20),
    "tilt_bias": (-0.50, 0.50),
    "emissive_intensity": (0.50, 5.0),
}

#: Joint names searched for the postural (head/neck) degree of freedom.
NECK_JOINT_CANDIDATES = ("neck", "neck_ball", "head", "head_tilt", "cervical")

#: Failures isolated at the bridge boundary rather than raised into the caller.
#: CP126 288b0d62: only ImportError/AttributeError/RuntimeError were caught,
#: while shape mismatches and malformed affect raise TypeError, ValueError,
#: IndexError, FloatingPointError or OSError.
_BRIDGE_ERRORS = (
    ArithmeticError,
    AttributeError,
    ImportError,
    IndexError,
    KeyError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)

#: Guards the read-modify-write when the body offers no barrier of its own.
_FALLBACK_APPLY_LOCK = threading.RLock()


def _finite(value: Any, low: float, high: float, default: float) -> tuple[float, str]:
    """A usable number inside [low, high], plus a fault note if repaired.

    CP126 89fa9c28. Delegates to the shared primitive in
    core/runtime/numeric_safety.py — this is the same defect class as the
    tiered-action risk inputs and the lesion-study probes.
    """
    scalar = validated_scalar(value, name="value", low=low, high=high, default=default)
    return float(scalar), scalar.fault


def _bound(name: str, value: float) -> float:
    low, high = PHYSICS_BOUNDS[name]
    if math.isnan(value) or math.isinf(value):
        return low if name != "tilt_bias" else 0.0
    return max(low, min(high, float(value)))


class PersonalityBridge:
    """
    The 'Ghost in the Machine' bridge.

    It translates abstract psychological states (Valence, Arousal) into
    concrete physical parameters (stiffness, damping, postural bias). Postural
    control is applied only when the body actually exposes a neck/head joint;
    otherwise the receipt says so rather than the docstring implying it
    (CP126 047a0b0b).
    """

    def __init__(self):
        self.last_affect = AffectVector()
        self.last_sync: dict[str, Any] = {}
        self._degradation_stamps: dict[str, float] = {}

    # ── affect → physics ───────────────────────────────────────────
    def derive_physics_modifiers(self, affect: Any) -> dict:
        """Derive bounded multipliers for MuJoCo constants from current mood."""
        valence, valence_fault = _finite(getattr(affect, "valence", 0.0), -1.0, 1.0, 0.0)
        arousal, arousal_fault = _finite(getattr(affect, "arousal", 0.5), 0.0, 1.0, 0.5)
        curiosity, curiosity_fault = _finite(getattr(affect, "curiosity", 0.5), 0.0, 1.0, 0.5)
        faults = tuple(
            f"{name}: {fault}"
            for name, fault in (
                ("valence", valence_fault),
                ("arousal", arousal_fault),
                ("curiosity", curiosity_fault),
            )
            if fault
        )
        if faults:
            self._note("affect_out_of_contract", "; ".join(faults))

        # 1. Stiffness (Rigidity): low valence or high arousal -> stiffer.
        stiffness_mult = 1.0 + (1.0 - valence) * 0.5 + arousal * 0.3
        # 2. Damping (Fluidity): high valence -> shorter oscillations.
        damping_mult = 0.8 + valence * 0.4
        # 3. Micro-vibrations (Restlessness).
        jitter_intensity = max(0.0, arousal - valence) * 0.05
        # 4. Postural bias (Head/Neck) from curiosity.
        tilt_bias = curiosity * 0.2

        return {
            "stiffness_mult": _bound("stiffness_mult", stiffness_mult),
            "damping_mult": _bound("damping_mult", damping_mult),
            "jitter": _bound("jitter", jitter_intensity),
            "tilt_bias": _bound("tilt_bias", tilt_bias),
            "emissive_intensity": _bound("emissive_intensity", arousal * 2.0 + 1.0),
            "input_faults": list(faults),
        }

    # ── embodiment ─────────────────────────────────────────────────
    async def sync_embodiment(self, virtual_body: Any) -> dict:
        """Pull state from the repository and apply it to the physics model.

        Always returns a receipt. CP126 4995f0ab: an absent repository or state
        returned bare ``None`` with no degradation, so an operator could not
        tell healthy neutral embodiment from a disconnected bridge.
        """
        receipt: dict[str, Any] = {
            "applied": False,
            "reason": "",
            "tilt_applied": False,
            "synchronization": "none",
            "timestamp": time.time(),
        }
        try:
            repo = get_runtime_service("state_repository", default=None)
            if not repo:
                return self._incomplete(receipt, "state_repository_unavailable")

            state = await repo.get_current()
            if not state:
                return self._incomplete(receipt, "state_unavailable")

            affect = getattr(state, "affect", None)
            if affect is None:
                return self._incomplete(receipt, "state_has_no_affect")

            mods = self.derive_physics_modifiers(affect)
            receipt.update(mods)
            # CP126 4995f0ab: last_affect was never updated, so the bridge's own
            # notion of "current" affect stayed at construction defaults forever.
            self.last_affect = affect

            model = getattr(virtual_body, "model", None)
            if model is None:
                receipt["reason"] = "no_physics_model"
                self.last_sync = receipt
                return receipt

            self._apply_to_body(virtual_body, model, mods, receipt)
            self.last_sync = receipt
            return receipt
        except _BRIDGE_ERRORS as exc:
            record_degradation(
                "personality_bridge",
                exc,
                action="isolated a physics-embodiment failure from the caller",
            )
            logger.warning("Personality-Body drift: %s", exc)
            receipt["reason"] = f"{type(exc).__name__}: {exc}"
            self.last_sync = receipt
            return receipt

    def _incomplete(self, receipt: dict, reason: str) -> dict:
        """A sync that could not happen, said out loud."""
        self._note(reason, "sync_embodiment")
        receipt["reason"] = reason
        receipt.update(
            {key: None for key in PHYSICS_BOUNDS if key not in receipt}
        )
        self.last_sync = receipt
        return receipt

    def _apply_to_body(
        self, virtual_body: Any, model: Any, mods: dict, receipt: dict
    ) -> None:
        """Write the constants under whatever barrier the body provides."""
        baseline = self._baseline(virtual_body, model, receipt)
        if baseline is None:
            receipt["reason"] = receipt.get("reason") or "baseline_unavailable"
            return

        barrier, kind = self._synchronization(virtual_body)
        receipt["synchronization"] = kind
        if kind == "none":
            # CP126 fe8e6eee: with no lock, pause barrier, command queue or
            # step-boundary handoff, a stepping simulation can observe half of
            # this update. We still apply — that is the only channel the body
            # offers — but the receipt and the ledger say it was unsynchronized.
            self._note(
                "unsynchronized_physics_write",
                f"{type(virtual_body).__name__} exposes no physics lock or command queue",
            )

        with barrier:
            model.jnt_stiffness[:] = baseline["stiffness"] * mods["stiffness_mult"]
            model.dof_damping[:] = baseline["damping"] * mods["damping_mult"]
            receipt["tilt_applied"] = self._apply_gaze(
                virtual_body, model, mods["tilt_bias"], receipt
            )
        receipt["applied"] = True
        receipt["reason"] = receipt.get("reason") or "ok"

    @staticmethod
    def _synchronization(virtual_body: Any):
        """The body's own mutual-exclusion primitive, or a private fallback."""
        for attribute in ("physics_lock", "step_lock", "model_lock", "lock", "_lock"):
            candidate = getattr(virtual_body, attribute, None)
            if candidate is not None and hasattr(candidate, "__enter__"):
                return candidate, attribute
        return _FALLBACK_APPLY_LOCK, "none"

    def _baseline(self, virtual_body: Any, model: Any, receipt: dict) -> dict | None:
        """Versioned baseline constants for this exact model.

        CP126 d38c257a: the first sync snapshotted whatever happened to be in
        the arrays — possibly values another controller had already scaled —
        and a model replacement, reset or legitimate retune never refreshed it.
        The snapshot is now keyed by model identity and array shape.
        """
        try:
            stiffness = model.jnt_stiffness
            damping = model.dof_damping
            fingerprint = (
                id(model),
                tuple(getattr(stiffness, "shape", (len(stiffness),))),
                tuple(getattr(damping, "shape", (len(damping),))),
            )
        except _BRIDGE_ERRORS as exc:
            self._note("model_arrays_unreadable", str(exc))
            return None

        stored = getattr(virtual_body, "_aura_physics_baseline", None)
        if isinstance(stored, dict) and stored.get("fingerprint") == fingerprint:
            receipt["baseline"] = "reused"
            return stored

        baseline = {
            "fingerprint": fingerprint,
            "stiffness": stiffness.copy(),
            "damping": damping.copy(),
            "captured_at": time.time(),
        }
        try:
            virtual_body._aura_physics_baseline = baseline
        except (AttributeError, TypeError) as exc:
            self._note("baseline_not_storable", str(exc))
        receipt["baseline"] = "captured" if stored is None else "recaptured"
        return baseline

    def reset_baseline(self, virtual_body: Any) -> bool:
        """Forget the captured baseline so the next sync re-reads the model."""
        try:
            delattr(virtual_body, "_aura_physics_baseline")
            return True
        # not a failure: no captured baseline is the state this is forgetting it into.
        except (AttributeError, TypeError):
            return False

    def _apply_gaze(
        self, virtual_body: Any, model: Any, tilt_bias: float, receipt: dict
    ) -> bool:
        """Apply the postural bias to the neck/head joint, if the body has one.

        CP126 047a0b0b: this was a bare ``pass``. tilt_bias was computed and
        returned, implying a causal postural effect that never reached the
        body. It is applied where a joint exists, and reported as not applied
        where one does not.
        """
        data = getattr(virtual_body, "data", None)
        qpos = getattr(data, "qpos", None)
        if qpos is None:
            receipt["tilt_reason"] = "no_qpos"
            return False

        index = self._neck_qpos_index(virtual_body, model)
        if index is None:
            receipt["tilt_reason"] = "no_neck_joint"
            return False
        try:
            if index < 0 or index >= len(qpos):
                receipt["tilt_reason"] = f"neck_index_out_of_range:{index}"
                return False
            qpos[index] = _bound("tilt_bias", tilt_bias)
        except _BRIDGE_ERRORS as exc:
            self._note("gaze_write_failed", str(exc))
            receipt["tilt_reason"] = f"{type(exc).__name__}: {exc}"
            return False
        receipt["tilt_reason"] = "applied"
        receipt["tilt_index"] = int(index)
        return True

    @staticmethod
    def _neck_qpos_index(virtual_body: Any, model: Any) -> int | None:
        explicit = getattr(virtual_body, "neck_qpos_index", None)
        if isinstance(explicit, int) and explicit >= 0:
            return explicit
        lookup = getattr(model, "joint", None)
        if not callable(lookup):
            return None
        for name in NECK_JOINT_CANDIDATES:
            try:
                joint = lookup(name)
            except (KeyError, ValueError, TypeError, AttributeError, IndexError):
                continue
            address = getattr(joint, "qposadr", None)
            if address is None:
                continue
            try:
                return int(address[0]) if hasattr(address, "__len__") else int(address)
            except (TypeError, ValueError, IndexError):
                continue
        return None

    # ── observability ──────────────────────────────────────────────
    def _note(self, reason: str, detail: str, *, interval_s: float = 60.0) -> None:
        """Record a bridge degradation, rate-limited per reason."""
        now = time.monotonic()
        if now - self._degradation_stamps.get(reason, 0.0) < interval_s:
            return
        self._degradation_stamps[reason] = now
        try:
            record_degradation(
                "personality_bridge",
                RuntimeError(f"{reason}: {detail}"),
                action="reported incomplete physical embodiment instead of skipping silently",
                severity="warning",
            )
        except _BRIDGE_ERRORS:
            logger.warning("PersonalityBridge %s: %s", reason, detail)

    def status(self) -> dict[str, Any]:
        """Last sync receipt, for health surfaces."""
        return dict(self.last_sync)
