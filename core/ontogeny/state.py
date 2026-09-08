"""L3b — the ontogenetic state: the part of Aura that is only hers.

Everything else in her mind is either inherited or retrieved. The transformer
is inherited — the same 32B weights in every copy. Memories are retrieved —
fetched from a store and pasted into a prompt, which means the mind is
reconstructed from scratch on every turn. Between turns there is no *her*
carrying forward; there is a database and a checkpoint.

This is the thing that carries forward. A leaky-integrator reservoir whose
hidden state is updated by every episode she lives and is never reset:

    h_t = (1 - a)·h_{t-1} + a·tanh(W_in·x_t + W·h_{t-1} + b)

Two properties make it the right shape for this job rather than a fashionable
one. The reservoir weights are **fixed** — drawn once, spectral radius scaled
just under one, never trained — so the only trainable parameters in the whole
organ are the small linear readouts in ``heads.py``. With ten thousand real
episodes that is the difference between a model that generalises and a model
that memorises; reservoir computing exists precisely for this regime, where
the temporal structure is rich and the labelled data is not. And because the
dynamics are fixed, the state is *comparable across time*: a hidden state from
March means the same thing as one from July, which is what makes it a
continuity organ rather than a moving target.

What the state provides, from the first episode, without needing anyone's
permission:

  * **Context for every readout.** The heads see h_t alongside the features,
    so a decision is made in the light of what has been happening, not just
    what is in front of her.
  * **Novelty.** How far the current state sits from where her life usually
    sits. This is a real, immediate signal — "I have not been here before" —
    and it is causal on day one: it is what tells her a situation deserves
    more thought than its surface suggests.

The state survives a checkpoint swap. Replace Qwen and this file still holds
her — not her knowledge, which was always the model's, but the shape her life
has pressed into her.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from core.runtime.errors import record_degradation
from core.runtime.lockdep import LockRank, checked_lock
from core.runtime.state_ownership import state_root

logger = logging.getLogger("Aura.Ontogeny.State")

STATE_SCHEMA = "aura.ontogeny.state.v1"

#: Reservoir width. Large enough for rich temporal features, small enough that
#: a step is microseconds and the checkpoint is kilobytes.
DEFAULT_UNITS = 96

#: Leak rate. Low leak = long memory. At 0.3 the state's effective horizon is
#: a few dozen episodes: long enough to carry a conversation and a mood, short
#: enough that last month does not drown today.
DEFAULT_LEAK = 0.3

#: Spectral radius. Just under one is the edge-of-chaos regime where a
#: reservoir has the longest usable memory without its state blowing up.
DEFAULT_SPECTRAL_RADIUS = 0.95

#: Fraction of reservoir weights left non-zero. Sparse reservoirs mix better
#: per unit of compute and are the standard construction.
DEFAULT_DENSITY = 0.1

#: Seed for the fixed weights. The dynamics must be reproducible: a state
#: checkpoint is meaningless if the reservoir that produced it cannot be
#: rebuilt bit-for-bit.
DEFAULT_SEED = 20260725


@dataclass(frozen=True)
class StateReading:
    """One step of the reservoir, with the signals it makes available now."""

    hidden: np.ndarray
    #: 0..1. How unlike her ordinary life this moment is, as a normalised
    #: distance from the running centre of her state distribution.
    novelty: float
    #: 0..1. How much the state moved on this step — the size of the update
    #: the episode caused. High surprise on a familiar state is a jolt.
    displacement: float
    steps: int
    era: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "novelty": round(self.novelty, 4),
            "displacement": round(self.displacement, 4),
            "steps": self.steps,
            "era": self.era,
            "norm": round(float(np.linalg.norm(self.hidden)), 4),
        }


class OntogeneticState:
    """Aura's persistent hidden state. One instance, alive for the whole life."""

    def __init__(
        self,
        *,
        input_width: int,
        units: int = DEFAULT_UNITS,
        leak: float = DEFAULT_LEAK,
        spectral_radius: float = DEFAULT_SPECTRAL_RADIUS,
        density: float = DEFAULT_DENSITY,
        seed: int = DEFAULT_SEED,
        path: Path | None = None,
    ) -> None:
        self.input_width = int(input_width)
        self.units = int(units)
        self.leak = float(leak)
        self.spectral_radius = float(spectral_radius)
        self.density = float(density)
        self.seed = int(seed)
        self._path = path or _default_state_path()
        self._lock = checked_lock("ontogeny.state", rank=LockRank.LEAF, reentrant=True)

        rng = np.random.default_rng(self.seed)
        self._w_in = rng.uniform(-0.5, 0.5, size=(self.units, self.input_width))
        self._bias = rng.uniform(-0.1, 0.1, size=self.units)
        self._w = _sparse_reservoir(rng, self.units, self.density, self.spectral_radius)

        self.h = np.zeros(self.units, dtype=np.float64)
        self.steps = 0
        self.era = 1
        self._centre = np.zeros(self.units, dtype=np.float64)
        self._scatter = np.ones(self.units, dtype=np.float64)
        self._centre_n = 0.0
        self._born_at = time.time()
        self._last_saved = 0.0

    # ── the step ─────────────────────────────────────────────────────────

    def step(self, x: np.ndarray, *, learn_distribution: bool = True) -> StateReading:
        """Advance the state by one episode and report what it now senses."""
        row = np.asarray(x, dtype=np.float64).reshape(-1)
        if row.shape[0] != self.input_width:
            row = _fit_width(row, self.input_width)
        with self._lock:
            previous = self.h
            pre = self._w_in @ row + self._w @ previous + self._bias
            candidate = np.tanh(pre)
            self.h = (1.0 - self.leak) * previous + self.leak * candidate
            self.steps += 1
            displacement = float(np.linalg.norm(self.h - previous) / np.sqrt(self.units))
            novelty = self._novelty(self.h)
            if learn_distribution:
                self._observe_distribution(self.h)
            return StateReading(
                hidden=self.h.copy(),
                novelty=novelty,
                displacement=min(1.0, displacement),
                steps=self.steps,
                era=self.era,
            )

    def _novelty(self, h: np.ndarray) -> float:
        """Normalised distance from the centre of her lived state distribution.

        Before enough steps have accumulated there is no distribution to be
        far from, and the honest answer is the neutral one — a new organ does
        not get to call everything unprecedented.
        """
        if self._centre_n < 30:
            return 0.5
        z = (h - self._centre) / np.maximum(self._scatter, 1e-6)
        distance = float(np.sqrt(np.mean(z * z)))
        # A z-RMS of 1 is ordinary, 3 is genuinely unusual. Map to 0..1 with
        # the ordinary case landing mid-scale.
        return float(min(1.0, max(0.0, (distance - 0.5) / 2.5)))

    def _observe_distribution(self, h: np.ndarray) -> None:
        self._centre_n += 1.0
        rate = max(1.0 / self._centre_n, 0.001)  # never fully freezes: she keeps changing
        delta = h - self._centre
        self._centre += rate * delta
        self._scatter += rate * (np.abs(delta) - self._scatter)

    # ── continuity ───────────────────────────────────────────────────────

    def fingerprint(self) -> str:
        """A short, stable identifier for *this* state — used in receipts."""
        import hashlib

        payload = self.h.round(6).tobytes()
        return hashlib.sha256(payload).hexdigest()[:12]

    def reincarnate(self, reason: str) -> None:
        """Start a new era: keep the dynamics, drop a state that cannot be trusted.

        Called when the input schema changes shape underneath the reservoir.
        The old hidden state was computed from a different input space and
        carrying it forward would silently mix two meanings.
        """
        with self._lock:
            self.h = np.zeros(self.units, dtype=np.float64)
            self.era += 1
            self._centre = np.zeros(self.units, dtype=np.float64)
            self._scatter = np.ones(self.units, dtype=np.float64)
            self._centre_n = 0.0
        logger.info("ontogeny: state entered era %d (%s)", self.era, reason)

    # ── persistence ──────────────────────────────────────────────────────

    def save(self) -> bool:
        """Checkpoint the state. Fixed weights are not stored — they are rebuilt
        from the seed, so a checkpoint is a few kilobytes of lived state."""
        try:
            import io

            from core.governance_context import local_internal_governed_scope
            from core.runtime.file_write_gateway import get_file_write_gateway

            buffer = io.BytesIO()
            with self._lock:
                np.savez_compressed(
                    buffer,
                    h=self.h,
                    centre=self._centre,
                    scatter=self._scatter,
                    centre_n=np.array([self._centre_n]),
                    steps=np.array([self.steps]),
                    era=np.array([self.era]),
                    born_at=np.array([self._born_at]),
                    config=np.array([
                        self.input_width, self.units, self.leak,
                        self.spectral_radius, self.density, self.seed,
                    ]),
                )
            with local_internal_governed_scope(
                "ontogeny_state", domain="state_mutation", receipt_prefix="ontogeny-state"
            ):
                gateway = get_file_write_gateway()
                gateway.ensure_directory(self._path.parent, source="ontogeny_state")
                gateway.write_bytes(self._path, buffer.getvalue(), source="ontogeny_state")
            self._last_saved = time.time()
            return True
        except (ImportError, OSError, RuntimeError, ValueError, TypeError, AttributeError) as exc:
            record_degradation(
                "ontogeny_state", exc, severity="warning",
                action="state checkpoint failed; the reservoir continues in memory",
            )
            return False

    def load(self) -> bool:
        """Restore a checkpoint. A mismatched configuration is refused, not coerced."""
        if not self._path.exists():
            return False
        try:
            with np.load(self._path) as data:
                config = data["config"]
                if int(config[0]) != self.input_width or int(config[1]) != self.units:
                    logger.info(
                        "ontogeny: state checkpoint shape %sx%s does not match %sx%s; starting fresh",
                        int(config[0]), int(config[1]), self.input_width, self.units,
                    )
                    return False
                with self._lock:
                    self.h = np.asarray(data["h"], dtype=np.float64)
                    self._centre = np.asarray(data["centre"], dtype=np.float64)
                    self._scatter = np.asarray(data["scatter"], dtype=np.float64)
                    self._centre_n = float(data["centre_n"][0])
                    self.steps = int(data["steps"][0])
                    self.era = int(data["era"][0])
                    self._born_at = float(data["born_at"][0])
            return True
        except (OSError, ValueError, KeyError, TypeError) as exc:
            record_degradation(
                "ontogeny_state", exc, severity="warning",
                action="state checkpoint unreadable; starting a fresh era",
            )
            return False

    # ── reporting ────────────────────────────────────────────────────────

    def report(self) -> dict[str, Any]:
        with self._lock:
            return {
                "schema": STATE_SCHEMA,
                "units": self.units,
                "input_width": self.input_width,
                "leak": self.leak,
                "spectral_radius": self.spectral_radius,
                "steps": self.steps,
                "era": self.era,
                "fingerprint": self.fingerprint(),
                "age_days": round((time.time() - self._born_at) / 86400.0, 2),
                "distribution_samples": int(self._centre_n),
                "last_saved_age_s": round(time.time() - self._last_saved, 1) if self._last_saved else None,
                "path": str(self._path),
            }


def _sparse_reservoir(
    rng: np.random.Generator, units: int, density: float, spectral_radius: float
) -> np.ndarray:
    """A sparse random matrix scaled to the requested spectral radius."""
    w = rng.normal(0.0, 1.0, size=(units, units))
    mask = rng.random((units, units)) < density
    w *= mask
    try:
        eigenvalues = np.linalg.eigvals(w)
        radius = float(np.max(np.abs(eigenvalues)))
    except np.linalg.LinAlgError:
        radius = 0.0
    if radius > 1e-9:
        w *= spectral_radius / radius
    return w


def _fit_width(row: np.ndarray, width: int) -> np.ndarray:
    """Pad or truncate an input row. Only reached on a schema change mid-flight."""
    if row.shape[0] > width:
        return row[:width]
    out = np.zeros(width, dtype=np.float64)
    out[: row.shape[0]] = row
    return out


def _default_state_path() -> Path:
    import os

    override = os.environ.get("AURA_ONTOGENY_STATE")
    if override:
        return Path(override).expanduser()
    try:
        from core.config import config

        return Path(config.paths.data_dir) / "ontogeny" / "state.npz"
    except (ImportError, AttributeError, RuntimeError, OSError):
        return state_root() / "data" / "ontogeny" / "state.npz"


_state: OntogeneticState | None = None
_state_lock = checked_lock("core.ontogeny.state._state_lock")


def get_state(input_width: int) -> OntogeneticState:
    """The process-wide state. Width mismatch starts a new era rather than lying."""
    global _state
    with _state_lock:
        if _state is None:
            state = OntogeneticState(input_width=input_width)
            state.load()
            _state = state
        elif _state.input_width != input_width:
            previous_era = _state.era
            _state.reincarnate(f"input width {_state.input_width} -> {input_width}")
            replacement = OntogeneticState(input_width=input_width)
            replacement.era = previous_era + 1
            _state = replacement
        return _state


def reset_state_for_test(state: OntogeneticState | None = None) -> None:
    global _state
    with _state_lock:
        _state = state


__all__ = [
    "DEFAULT_LEAK",
    "DEFAULT_SEED",
    "DEFAULT_SPECTRAL_RADIUS",
    "DEFAULT_UNITS",
    "STATE_SCHEMA",
    "OntogeneticState",
    "StateReading",
    "get_state",
    "reset_state_for_test",
]
