"""Her one-shot memory kept on disk and in bounds: loading it, persisting it, and which entry goes first when it is full.

Lifted whole out of `nonparametric_memory`. Every name taken from it is imported at
CALL time: that module imports this one to build the class, and a test that
patches a name on it has to reach the code that reads it.
"""
from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any


class _KeepsItsStoreOnDiskAndInBounds:
    """Lifted whole out of NonParametricMemory; see nonparametric_memory.py."""

    def _ensure_capacity(self, needed: int) -> None:
        from .nonparametric_memory import (
            np,
        )

        if needed <= self._capacity:
            return
        new_capacity = min(self._max, max(needed, self._capacity * 2))
        keys = np.empty((new_capacity, self._dim), dtype=np.float32)
        norms = np.empty(new_capacity, dtype=np.float32)
        keys[: self._size] = self._keys[: self._size]
        norms[: self._size] = self._key_norms[: self._size]
        self._keys = keys
        self._key_norms = norms
        self._capacity = new_capacity

    def _evict_one_for_incoming(self) -> None:
        from .nonparametric_memory import (
            np,
        )

        n = self._size
        if n == 0:
            return
        now = time.time()
        gravity = np.array(
            [self._weights[i] * math.exp(-(now - self._ts[i]) / (14 * 24 * 3600.0)) for i in range(n)]
        )
        drop = int(np.argmin(gravity))
        if drop < n - 1:
            self._keys[drop : n - 1] = self._keys[drop + 1 : n]
            self._key_norms[drop : n - 1] = self._key_norms[drop + 1 : n]
        # Every per-row list moves with the rows. The provenance list did not, so after one eviction each later row
        # carried the principal and source of the row before it, which is what decides who may read it; and the
        # dedup index kept pointing one row past every entry after the drop, so a replay rewrote a neighbour.
        for values in (self._token_ids, self._tokens, self._weights, self._ts, self._provenance):
            values.pop(drop)
        self._content_keys = {key: (row - 1 if row > drop else row)
                              for key, row in self._content_keys.items() if row != drop}
        self._size -= 1
        self._stats["evicted"] += 1

    def _load(self) -> None:
        """Read a persisted generation, or start fresh. Never a hybrid.

        Two findings meet here.

        ``b8b9656b`` — keys and metadata were published with two independent
        ``os.replace`` calls, so a crash between them, or two writers
        interleaving, paired vectors from one generation with token metadata
        from another. There was no generation id, no manifest checksum, no
        keys fsync and no directory fsync. Both files now carry the same
        generation id and the metadata carries the keys' digest; a pair that
        does not agree is refused rather than mixed.

        ``848cf532`` — the loader trusted the files after checking rank,
        width and list lengths. Everything else is validated now: schema,
        declared dim, store identity, finite and bounded keys, weights,
        timestamps, token-id range.

        ``1d1d0bfd`` — ``np.load`` materialised the entire persisted matrix
        before the entry count was capped, so a stale or replaced file could
        allocate far past ``max_entries``. It is memory-mapped, sliced, then
        copied.
        """
        from .nonparametric_memory import (
            _provenance_from_mapping,
            _streamed_keys_digest,
            identity_from_mapping,
            np,
            record_degradation,
        )

        keys_p, meta_p = self._path.with_suffix(".keys.npy"), self._path.with_suffix(".meta.json")
        if not (keys_p.exists() and meta_p.exists()):
            return
        keys = None
        try:
            meta = json.loads(meta_p.read_text(encoding="utf-8"))
            if not isinstance(meta, dict):
                raise ValueError("non-parametric memory metadata is not an object")
            if int(meta.get("schema_version", 0) or 0) < 3:
                # A generation written before the manifest existed cannot be
                # paired safely with its keys file. Starting fresh loses a
                # cache; loading it risks pairing across generations.
                record_degradation(
                    "nonparametric_memory_load",
                    ValueError("persisted store predates the generation manifest"),
                    severity="info",
                )
                return
            if int(meta.get("dim", -1)) != self._dim:
                raise ValueError("persisted dim does not match this store")

            saved_identity = identity_from_mapping(meta.get("identity"), dim=self._dim)
            if saved_identity is None:
                raise ValueError("persisted store carries no readable identity")
            compatible, why = self._identity.compatible_with(saved_identity)
            if not compatible:
                # Same width, different model. Reading it would combine one
                # model's vectors with another's token ids (CP126 aba3eb39).
                record_degradation(
                    "nonparametric_memory_load",
                    ValueError(f"refusing incompatible store: {why}"),
                    severity="warning",
                )
                return

            # Memory-mapped, so the cap applies before the allocation.
            keys = np.load(keys_p, mmap_mode="r")
            if keys.ndim != 2 or keys.shape[1] != self._dim:
                return
            digest = str(meta.get("keys_sha256") or "")
            token_ids = list(meta.get("token_ids", []))
            tokens = list(meta.get("tokens", []))
            weights = list(meta.get("weights", []))
            timestamps = list(meta.get("ts", []))
            provenance_rows = list(meta.get("provenance", []))
            lengths = {
                len(keys), len(token_ids), len(tokens), len(weights), len(timestamps)
            }
            if len(lengths) != 1 or not lengths or next(iter(lengths)) <= 0:
                raise ValueError("non-parametric memory persistence metadata is inconsistent")

            total = next(iter(lengths))
            if digest and digest != _streamed_keys_digest(keys):
                # The manifest names a keys file this is not — a torn publish
                # or a replaced file. Refusing is the whole point of writing
                # the digest. Streamed from the memory map so verifying it
                # does not undo the bounded read.
                raise ValueError("persisted keys do not match the manifest digest")
            order = self._retention_order(weights, timestamps, total)
            count = min(total, self._max)
            keep = order[:count]

            capacity = min(max(64, count), self._max)
            new_keys = np.empty((capacity, self._dim), dtype=np.float32)
            new_keys[:count] = np.asarray(keys[keep], dtype=np.float32)
            if not np.all(np.isfinite(new_keys[:count])):
                raise ValueError("non-parametric memory persisted keys are non-finite")
            new_norms = np.empty(capacity, dtype=np.float32)
            norms = np.einsum("ij,ij->i", new_keys[:count], new_keys[:count])
            if not np.all(np.isfinite(norms)) or float(np.max(norms, initial=0.0)) > self.MAX_KEY_NORM:
                raise ValueError("non-parametric memory persisted keys overflow their norms")
            new_norms[:count] = norms

            new_token_ids = [int(token_ids[i]) for i in keep]
            if not all(self._token_id_in_vocabulary(t) for t in new_token_ids):
                raise ValueError("non-parametric memory persisted token ids are out of vocabulary")
            new_tokens = [str(tokens[i]) for i in keep]
            new_weights = [self._clamped_weight(weights[i]) for i in keep]
            new_ts = [float(timestamps[i]) for i in keep]
            if not all(math.isfinite(t) and t > 0.0 for t in new_ts):
                raise ValueError("non-parametric memory persisted timestamps are invalid")
            new_provenance = [
                _provenance_from_mapping(
                    provenance_rows[i] if i < len(provenance_rows) else None
                )
                for i in keep
            ]


            saved_tolerance = meta.get("exact_tolerance")
            try:
                new_tolerance = float(saved_tolerance) if saved_tolerance is not None else None
            except (TypeError, ValueError):
                new_tolerance = None
            if new_tolerance is not None and not (math.isfinite(new_tolerance) and 0.0 <= new_tolerance < 1.0):
                new_tolerance = None
            saved_mu = meta.get("query_mu")
            new_mu = None
            new_mu_n = 0
            if isinstance(saved_mu, list) and len(saved_mu) == self._dim:
                candidate_mu = np.asarray(saved_mu, dtype=np.float32)
                if np.all(np.isfinite(candidate_mu)):
                    new_mu = candidate_mu
                    new_mu_n = max(0, min(10_000, int(meta.get("query_mu_n", 0) or 0)))

            with self._lock:
                self._capacity = capacity
                self._keys = new_keys
                self._key_norms = new_norms
                self._token_ids = new_token_ids
                self._tokens = new_tokens
                self._weights = new_weights
                self._ts = new_ts
                self._provenance = new_provenance
                self._size = count
                if new_mu is not None:
                    self._query_mu = new_mu
                    self._query_mu_n = new_mu_n
                    self._mean_generation += 1
                self._exact_tolerance = new_tolerance
                self._exact_observations = int(meta.get("exact_observations", 0) or 0) if new_tolerance is not None else 0
                self._reindex_content_keys()
                self._content_generation += 1
                self._identity_cache = None
        except (OSError, ValueError, TypeError, KeyError, IndexError, json.JSONDecodeError) as exc:
            record_degradation("nonparametric_memory_load", exc)
        finally:
            if keys is not None and hasattr(keys, "_mmap"):
                try:
                    keys._mmap.close()
                # not a failure: a mapping already closed is the state this is reaching.
                except (AttributeError, OSError, ValueError):
                    pass

    def _retention_order(self, weights: list[Any], timestamps: list[Any], total: int) -> list[int]:
        """Rows to keep, best first, by the SAME gravity eviction uses.

        Loading an oversized store kept the last N records while live
        eviction dropped the lowest weight-times-recency, so retention after
        a restart differed from retention during a run (CP126 ``4d9d6616``).
        """
        now = time.time()
        scored: list[tuple[float, int]] = []
        for i in range(total):
            weight = self._clamped_weight(weights[i] if i < len(weights) else 0.0)
            try:
                age = now - float(timestamps[i])
            except (TypeError, ValueError, IndexError):
                age = float("inf")
            if not math.isfinite(age) or age < 0.0:
                # A future or unreadable timestamp cannot be trusted to rank
                # anything; it goes last rather than winning by accident.
                age = float("inf")
            gravity = weight * math.exp(-min(age, 1e9) / (14 * 24 * 3600.0))
            scored.append((gravity, i))
        scored.sort(key=lambda row: (-row[0], row[1]))
        return [index for _gravity, index in scored]



    def persist(self) -> bool:
        """Publish keys and metadata as ONE generation, or neither.

        The two files were replaced independently, so a crash in between
        left a keys file from one generation beside metadata from another —
        vectors paired with the wrong tokens, with nothing to detect it. The
        metadata now carries the generation id and the keys' digest, both
        files are fsynced, the directory is fsynced, and the keys land first
        so a torn publish leaves the OLD metadata pointing at a digest that
        no longer matches, which the loader refuses.
        """
        from .nonparametric_memory import (
            _fsync_directory,
            _keys_digest,
            np,
            record_degradation,
            tempfile,
        )

        temporary_keys: Path | None = None
        temporary_meta: Path | None = None
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._lock:
                keys = self._keys[: self._size].copy()
                generation = int(self._content_generation)
                meta = {
                    "schema_version": 3,
                    "generation": generation,
                    "dim": self._dim,
                    "identity": self._identity.to_dict(),
                    "saved_at": time.time(),
                    "keys_sha256": _keys_digest(keys),
                    "token_ids": list(self._token_ids),
                    "tokens": list(self._tokens),
                    "weights": list(self._weights),
                    "ts": list(self._ts),
                    "provenance": [record.to_dict() for record in self._provenance],
                    "query_mu": [float(v) for v in self._query_mu],
                    "query_mu_n": int(self._query_mu_n),
                    "exact_tolerance": self._exact_tolerance,
                    "exact_observations": int(self._exact_observations),
                }
            keys_path = self._path.with_suffix(".keys.npy")
            meta_path = self._path.with_suffix(".meta.json")
            with tempfile.NamedTemporaryFile(
                dir=self._path.parent, suffix=".npy", delete=False
            ) as handle:
                np.save(handle, keys)
                handle.flush()
                os.fsync(handle.fileno())
                temporary_keys = Path(handle.name)
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=self._path.parent,
                suffix=".json", delete=False
            ) as handle:
                json.dump(meta, handle, separators=(",", ":"))
                handle.flush()
                os.fsync(handle.fileno())
                temporary_meta = Path(handle.name)
            os.replace(temporary_keys, keys_path)
            temporary_keys = None
            os.replace(temporary_meta, meta_path)
            temporary_meta = None
            _fsync_directory(self._path.parent)
            return True
        except self._ERRORS as exc:
            record_degradation("nonparametric_memory_persist", exc)
            return False
        finally:
            # A failed write left its temporary file behind every time.
            for leftover in (temporary_keys, temporary_meta):
                if leftover is None:
                    continue
                try:
                    # Through the gateway: this is a file this process
                    # created and must remove, and the gateway is where
                    # deletions are governed.
                    from core.runtime.file_write_gateway import get_file_write_gateway

                    get_file_write_gateway().delete_file(
                        leftover, source="brain.nonparametric_memory.persist_cleanup"
                    )
                except (ImportError, OSError, RuntimeError, ValueError) as cleanup_exc:
                    record_degradation(
                        "nonparametric_memory_persist",
                        cleanup_exc,
                        severity="info",
                        action="left a temporary persistence file behind",
                    )


