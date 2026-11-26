"""Persistent engram association field — the *learned*-weight half of plasticity.

The transient competition in ``engram_plasticity.py`` uses only the homeostatic
*activity* dynamics. This module activates the other half the whiteboard derives:
the **synaptic weight learning** — voltage-dependent STDP + competition on a real,
persisted weight matrix ``W`` over a pool of concept slots.

Co-recalled engrams drive their slots active and the field runs
``engine.step(learn=True)``, so the Clopath voltage rule potentiates the
connection between things recalled together (Hebbian: *what fires together wires
together*) while homeostasis + the spectral cap keep it bounded. The learned
associations are persisted across sessions and fed back into recall, so a cue can
surface things it has become *associated* with through experience — genuine
associative pattern completion via learned weights, not just shared surface cues.

This makes ``voltage_plasticity_delta`` / ``competition_drive`` / the ``W`` matrix
causal and durable instead of tested-but-dormant.
"""
from __future__ import annotations

import hashlib
import io
import logging
import os
import threading
from pathlib import Path

import numpy as np

from core.config import config
from core.consciousness.voltage_plasticity import (
    VoltageDependentPlasticityEngine,
    VoltagePlasticityConfig,
)
from core.runtime.errors import record_degradation
from core.runtime.file_write_gateway import get_file_write_gateway

logger = logging.getLogger("Aura.Memory.EngramAssociation")

_N_SLOTS = 256
_SAVE_EVERY = 20
_SETTLE = 3


def _enabled() -> bool:
    return os.getenv("AURA_ENGRAM_ASSOCIATION", "1") not in ("0", "false", "False")


class EngramAssociationField:
    """A persisted voltage-STDP weight field over hashed concept slots."""

    def __init__(self, n_slots: int = _N_SLOTS, path: str | None = None) -> None:
        self.n_slots = n_slots
        self.engine = VoltageDependentPlasticityEngine(
            VoltagePlasticityConfig(n_nodes=n_slots, seed=29)
        )
        self._lock = threading.Lock()
        self._learns = 0
        self._path = Path(path) if path else (config.paths.home_dir / "engram_associations.npy")
        self._load()

    # ── slot mapping (stable across sessions, unlike salted hash()) ───────

    def _slot(self, cue: str) -> int:
        h = hashlib.blake2b(str(cue).strip().lower().encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(h, "big") % self.n_slots

    # ── learning ──────────────────────────────────────────────────────────

    def learn(self, cue_groups: list[list[str]]) -> bool:
        """Potentiate associations between co-recalled engrams via voltage-STDP.

        ``cue_groups`` is the cue list for each engram recalled together. Their
        slots are driven active and the weight field learns (``step(learn=True)``),
        so connections among co-active slots strengthen. Returns True if it learned.
        """
        slots = sorted({self._slot(c) for group in cue_groups for c in (group or []) if c})
        if len(slots) < 2:
            return False
        drive = np.zeros(self.n_slots, dtype=np.float64)
        drive[slots] = 1.0
        with self._lock:
            for _ in range(_SETTLE):
                self.engine.step(external_input=drive, learn=True)
            self._learns += 1
            if self._learns % _SAVE_EVERY == 0:
                self._save()
        return True

    # ── readout ───────────────────────────────────────────────────────────

    def association(self, cue_a: str, cue_b: str) -> float:
        """Learned association strength between two cues (symmetric magnitude)."""
        i, j = self._slot(cue_a), self._slot(cue_b)
        if i == j:
            return 0.0
        w = self.engine.W
        return float(abs(w[i, j]) + abs(w[j, i])) * 0.5

    def association_boost(self, query_cues: list[str], item_cues: list[str]) -> float:
        """Total learned association from a query's cues to a candidate's cues.

        Used to bias recall toward engrams this query has become associated with
        through experience (learned pattern completion).
        """
        q = sorted({self._slot(c) for c in (query_cues or []) if c})
        it = sorted({self._slot(c) for c in (item_cues or []) if c})
        if not q or not it:
            return 0.0
        w = self.engine.W
        total = 0.0
        for i in q:
            for j in it:
                if i != j:
                    total += abs(float(w[i, j]))
        return total / max(1, len(q))

    # ── persistence ───────────────────────────────────────────────────────

    def _load(self) -> None:
        try:
            if self._path.exists():
                arr = np.load(self._path)
                if arr.shape == (self.n_slots, self.n_slots):
                    self.engine.W = arr.astype(np.float64)
                    logger.info("[EngramAssociation] loaded learned weights from %s", self._path)
        except (OSError, ValueError, EOFError) as exc:
            record_degradation("engram_association", exc)

    def _save(self) -> None:
        try:
            payload = io.BytesIO()
            np.save(payload, self.engine.W)
            # Persisting learned weights mutates a governed file; it can be driven
            # from background recall/hydration paths that hold no governance
            # token. Open a local maintenance scope so the file_write_gateway has
            # a valid receipt instead of raising GovernanceViolationError.
            from core.governance_context import local_internal_governed_scope

            with local_internal_governed_scope(
                "memory.engram_association.save",
                domain="file_write",
            ):
                get_file_write_gateway().write_bytes(
                    self._path,
                    payload.getvalue(),
                    source="memory:engram_association",
                )
        except (OSError, ValueError, RuntimeError) as exc:
            # Best-effort persistence — a save failure (incl. a
            # GovernanceViolationError, which subclasses RuntimeError) must never
            # cascade into the caller's memory/recall path.
            record_degradation("engram_association", exc)

    def flush(self) -> None:
        with self._lock:
            self._save()

    def status(self) -> dict[str, object]:
        return {
            "n_slots": self.n_slots,
            "learns": self._learns,
            "weight_norm": round(float(np.linalg.norm(self.engine.W)), 4),
            "engine": self.engine.get_status(),
        }


_field: EngramAssociationField | None = None
_field_lock = threading.Lock()


def get_engram_association_field() -> EngramAssociationField:
    global _field
    if _field is None:
        with _field_lock:
            if _field is None:
                _field = EngramAssociationField()
    return _field


def is_engram_association_enabled() -> bool:
    return _enabled()
