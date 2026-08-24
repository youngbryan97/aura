"""ReasonCache-style probe memoization: identical latent states decode once.

Verifier-guided episodes decode short greedy probes to score candidate
states — at branch selection, during latent-opt hill-climbing, and in the
identity/pre-action checks. The probe pipeline (persist slots → bridge →
greedy decode → rewind) is deterministic given the model function and the
exact latent state, so re-probing an unchanged state is pure recomputation:
~(slots + bridge + probe_tokens) × n_layers token-layer applications spent
to learn nothing new.

This cache keys probes by the exact bytes of the branch's seed state and
refined state plus the decode parameters, and returns the memoized tokens on
a hit — charging the budget NOTHING, because nothing ran. Honesty rules:

- the model function is part of the key's validity, so the engine
  invalidates the whole cache the moment fast weights attach, rescale, or
  erase — a memoized probe from a different function is a lie;
- hits/misses/invalidations and the layer-apps actually saved ride the
  episode receipt, so the optimization is measured, not vibed;
- entries are bounded (FIFO) and the cache lives exactly one episode.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any

from core.runtime.tensor_identity import tensor_identity_parts

logger = logging.getLogger("Aura.LatentCortex.ProbeCache")

PROBE_CACHE_SCHEMA = "aura.latent_cortex.probe_cache.v1"

MAX_ENTRIES = 64


def _array_digest(hasher: hashlib._Hash, array: Any) -> None:
    dtype, shape, payload = tensor_identity_parts(array)
    hasher.update(dtype.encode("ascii"))
    hasher.update(str(shape).encode("ascii"))
    hasher.update(payload)


class DecodeProbeCache:
    """Per-episode content-addressed store of decode-probe results."""

    def __init__(self, *, max_entries: int = MAX_ENTRIES) -> None:
        if type(max_entries) is not int or not 1 <= max_entries <= 1024:
            raise ValueError("max_entries must be an integer inside [1, 1024]")
        self.max_entries = max_entries
        self._entries: dict[str, list[int]] = {}
        self._insertion_order: list[str] = []
        self.hits = 0
        self.misses = 0
        self.layer_apps_saved = 0
        self._layer_apps_by_key: dict[str, int] = {}
        self.invalidations: list[str] = []

    # ── Keys ────────────────────────────────────────────────────────────
    def key(
        self,
        seed_z: Any,
        z: Any,
        bridge_tokens: list[int],
        probe_max_tokens: int,
        probe_contract: str = "none",
    ) -> str:
        """Exact-state key for every input that can change probe output."""
        if probe_contract not in {"none", "final_answer_v1"}:
            raise ValueError("probe_contract is invalid")
        hasher = hashlib.sha256()
        _array_digest(hasher, seed_z)
        _array_digest(hasher, z)
        hasher.update(
            ",".join(str(int(token)) for token in bridge_tokens).encode("ascii")
        )
        hasher.update(str(int(probe_max_tokens)).encode("ascii"))
        hasher.update(probe_contract.encode("ascii"))
        return hasher.hexdigest()

    # ── Store ───────────────────────────────────────────────────────────
    def get(self, key: str) -> list[int] | None:
        tokens = self._entries.get(key)
        if tokens is None:
            self.misses += 1
            return None
        self.hits += 1
        self.layer_apps_saved += self._layer_apps_by_key.get(key, 0)
        return list(tokens)

    def put(self, key: str, tokens: list[int], layer_apps_spent: int) -> None:
        if key in self._entries:
            return
        if (
            isinstance(layer_apps_spent, bool)
            or not isinstance(layer_apps_spent, int)
            or layer_apps_spent < 0
        ):
            raise ValueError("layer_apps_spent must be a non-negative integer")
        while len(self._insertion_order) >= self.max_entries:
            evicted = self._insertion_order.pop(0)
            self._entries.pop(evicted, None)
            self._layer_apps_by_key.pop(evicted, None)
        self._entries[key] = list(tokens)
        self._layer_apps_by_key[key] = layer_apps_spent
        self._insertion_order.append(key)

    def invalidate(self, reason: str) -> None:
        """The model function changed — every memoized probe is now stale."""
        if self._entries:
            logger.debug(
                "Probe cache invalidated (%s): %d entries dropped",
                reason,
                len(self._entries),
            )
        self._entries.clear()
        self._layer_apps_by_key.clear()
        self._insertion_order.clear()
        self.invalidations.append(str(reason))

    # ── Receipt ─────────────────────────────────────────────────────────
    def to_receipt(self) -> dict[str, Any]:
        return {
            "schema": PROBE_CACHE_SCHEMA,
            "hits": self.hits,
            "misses": self.misses,
            "entries": len(self._entries),
            "layer_apps_saved": self.layer_apps_saved,
            "invalidations": list(self.invalidations),
        }


__all__ = ["DecodeProbeCache", "MAX_ENTRIES", "PROBE_CACHE_SCHEMA"]
