"""The substrate state, snapshotted, written down and read back.

Three snapshot paths because the caller may or may not hold the lock and may
not be able to wait for it, and taking the wrong one is a deadlock rather than
a slow read. The recovery path is here too: a state that has diverged is
restored from the last snapshot rather than carried forward, because a
substrate running on NaN reports confident numbers about nothing.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch

from core.runtime.errors import record_degradation


class _KeepsItsStateOnDisk:
    """Lifted whole from LiquidSubstrate; see liquid_substrate.py."""

    def _mark_weight_cache_dirty(self) -> None:
        self._weight_cache_dirty = True

    def _refresh_torch_state_from_snapshot(
        self,
        *,
        x_snapshot: np.ndarray,
        v_snapshot: np.ndarray,
        state_revision: int,
    ) -> bool:
        """Build Torch mirrors off-lock and publish only for the same revision."""
        x_source = np.ascontiguousarray(x_snapshot, dtype=np.float32)
        v_source = np.ascontiguousarray(v_snapshot, dtype=np.float32)
        if self.device.type == "cpu":
            x_torch = torch.from_numpy(x_source)
            v_torch = torch.from_numpy(v_source)
        else:
            x_torch = torch.from_numpy(x_source).to(self.device)
            v_torch = torch.from_numpy(v_source).to(self.device)

        with self.sync_lock:
            if self._state_revision != int(state_revision):
                return False
            self.x_torch = x_torch
            self.v_torch = v_torch
            self._torch_state_revision = int(state_revision)
            return True

    def _recover_diverged_state(self, proposed_state: Any, *, source: str) -> np.ndarray:
        """Judge a proposed state, and restore the last sound one if it diverged.

        One place, at the single commit point every transform funnels through,
        so a divergence cannot enter state via a path that forgot to look.

        Fails OPEN on its own error, deliberately: if the recovery layer itself
        is broken, the substrate keeps running on the old coercion rather than
        stopping the mind. That fallback is recorded, never silent.
        """
        # Imported here rather than at module level: the module these
        # came from imports this one to build the class. A call-time
        # import also still sees a test's patch of the original.
        from .liquid_substrate import (
            logger,
        )

        proposed = np.asarray(proposed_state)
        try:
            outcome = self._divergence_recovery.recover(proposed, subsystem="liquid_substrate")
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation("liquid_substrate", exc, severity="warning")
            return np.nan_to_num(proposed, copy=True, nan=0.0, posinf=1.0, neginf=-1.0)

        if outcome.recovered and outcome.state is not None:
            return np.asarray(outcome.state, dtype=float)
        if outcome.state is not None:
            return np.asarray(outcome.state, dtype=float)
        # Diverged with no sound checkpoint ever recorded — recovery already
        # logged this CRITICAL. Coercion is all that is left, and the caller is
        # not silently told it succeeded.
        logger.error(
            "Substrate diverged during %s with no sound checkpoint; falling back to coercion.",
            source,
        )
        return np.nan_to_num(proposed, copy=True, nan=0.0, posinf=1.0, neginf=-1.0)

    def substrate_recovery_metrics(self) -> dict[str, Any]:
        """Divergences, recoveries, escalations and current damping."""
        try:
            return self._divergence_recovery.as_metrics()
        except (AttributeError, TypeError, ValueError) as exc:
            record_degradation("liquid_substrate", exc, severity="warning")
            return {}

    def _commit_worker_state_transform(
        self,
        *,
        source_state: np.ndarray,
        source_revision: int,
        proposed_state: np.ndarray,
        source: str,
        update_velocity: bool,
        set_last_update: bool = False,
    ) -> tuple[np.ndarray, bool]:
        """Commit worker math without erasing concurrent causal mutations.

        Heavy transforms snapshot state, compute without the hot lock, and
        return here. If another subsystem changed state meanwhile, the worker's
        bounded delta is applied to the current state instead of replacing it.
        A value comparison catches legacy writers that forgot to publish a
        revision and exposes them through telemetry.
        """
        source_state = np.nan_to_num(
            np.asarray(source_state),
            copy=True,
            nan=0.0,
            posinf=1.0,
            neginf=-1.0,
        )
        # DEFECT. `np.nan_to_num(..., nan=0.0)` was the ONLY thing standing
        # between a diverged ODE step and live state. It does not detect a
        # divergence and it does not recover from one: it silently substitutes
        # zeros and the run continues. Zeros are not a neutral default here —
        # x[0..6] are valence, arousal, dominance, frustration, curiosity,
        # energy and focus, so a diverged step reset every affective reading to
        # the middle mid-conversation, with nothing recorded. An unlogged
        # divergence is indistinguishable from a calm mind.
        #
        # The proposed state is now judged before it is coerced, and a diverged
        # step is replaced by the last state that was VERIFIED sound — her real
        # previous condition — with the divergence recorded and repeated
        # divergence damping the dynamics that caused it.
        proposed_state = self._recover_diverged_state(proposed_state, source=source)
        if source_state.shape != proposed_state.shape:
            raise ValueError(
                f"substrate transform shape mismatch: {source_state.shape} != {proposed_state.shape}"
            )

        with self.sync_lock:
            current = np.nan_to_num(
                np.asarray(self.x),
                copy=True,
                nan=0.0,
                posinf=1.0,
                neginf=-1.0,
            )
            if current.shape != source_state.shape:
                raise ValueError(
                    f"substrate state changed shape during {source}: "
                    f"{source_state.shape} -> {current.shape}"
                )

            revision_changed = self._state_revision != int(source_revision)
            values_changed = not np.array_equal(current, source_state, equal_nan=True)
            if values_changed and not revision_changed:
                self._untracked_state_mutations += 1
            merged = revision_changed or values_changed
            if merged:
                state_delta = proposed_state - source_state
                committed = np.clip(current + state_delta, -1.0, 1.0)
                self._concurrent_state_merges += 1
                self._state_merges_by_source[source] = (
                    self._state_merges_by_source.get(source, 0) + 1
                )
            else:
                committed = np.clip(proposed_state, -1.0, 1.0)

            if update_velocity:
                self.v = np.nan_to_num(
                    committed - current,
                    copy=False,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
            self.x = committed
            committed_revision = self.mark_state_mutated_locked(source)
            if set_last_update:
                self.last_update = time.time()
            committed_copy = committed.copy()
            velocity_copy = self.v.copy()

        self._refresh_torch_state_from_snapshot(
            x_snapshot=committed_copy,
            v_snapshot=velocity_copy,
            state_revision=committed_revision,
        )
        return committed_copy, merged

    @staticmethod
    def _weight_cache_signature(weights: np.ndarray) -> tuple[Any, ...]:
        arr = np.asarray(weights)
        if arr.size == 0:
            samples: tuple[float, ...] = ()
        else:
            flat = arr.ravel()
            last = flat.size - 1
            indices = sorted({
                0,
                last,
                last // 4,
                last // 2,
                (last * 3) // 4,
            })
            samples = tuple(float(flat[idx]) for idx in indices)
        return (id(weights), tuple(arr.shape), str(arr.dtype), samples)

    def _sync_weight_cache_locked(self) -> None:
        raw = np.asarray(self.W)
        if not np.isfinite(raw).all():
            self.W = np.nan_to_num(raw, copy=True, nan=0.0, posinf=5.0, neginf=-5.0)
            raw = np.asarray(self.W)
        w_torch_source = np.ascontiguousarray(raw, dtype=np.float32)
        if self.device.type == "cpu":
            self.W_torch = torch.from_numpy(w_torch_source)
        else:
            self.W_torch = torch.from_numpy(w_torch_source).to(self.device)
        self._cached_connectivity_norm = float(np.linalg.norm(raw))
        self._cached_connectivity_array_id = id(self.W)
        self._cached_connectivity_signature = self._weight_cache_signature(self.W)
        self._weight_cache_dirty = False

    def _freshness_threshold_s(self) -> float:
        rate = max(0.1, float(self.current_update_rate or self.config.update_rate or 1.0))
        return max(2.0, 3.0 / rate)

    def _state_snapshot(self) -> dict[str, Any]:
        with self.sync_lock:
            snapshot = self._state_snapshot_locked()
        self._last_published_snapshot = snapshot
        return snapshot

    def _state_snapshot_locked(self) -> dict[str, Any]:
        """Build the snapshot dict; caller must hold sync_lock."""
        x = np.nan_to_num(self.x.copy(), nan=0.0, posinf=1.0, neginf=-1.0)
        v = np.nan_to_num(self.v.copy(), nan=0.0, posinf=0.0, neginf=0.0)
        phi = self._current_phi if np.isfinite(self._current_phi) else 0.0
        last_update = float(self.last_update or 0.0)
        update_rate = float(self.current_update_rate or self.config.update_rate or 0.0)
        coherence = (
            self.microtubule_coherence
            if np.isfinite(self.microtubule_coherence)
            else 1.0
        )
        em_field = (
            self.em_field_magnitude
            if np.isfinite(self.em_field_magnitude)
            else 0.0
        )
        return {
            "x": x,
            "v": v,
            "phi": float(phi),
            "last_update": last_update,
            "update_rate_hz": update_rate,
            "snapshot_age_s": max(0.0, time.time() - last_update) if last_update else float("inf"),
            "freshness_threshold_s": self._freshness_threshold_s(),
            "coherence": float(coherence),
            "em_field": float(em_field),
            "l5_bursts": int(self.l5_burst_count),
            "collapse_events": int(self.total_collapse_events),
            "compute_budget_reason": self._last_compute_budget_reason,
            "compute_budget_memory_percent": self._last_compute_budget_memory_percent,
            "state_revision": int(self._state_revision),
            "torch_state_revision": int(self._torch_state_revision),
            "torch_mirror_lag": max(
                0,
                int(self._state_revision - self._torch_state_revision),
            ),
            "concurrent_state_merges": int(self._concurrent_state_merges),
            "untracked_state_mutations": int(self._untracked_state_mutations),
            "state_merges_by_source": dict(self._state_merges_by_source),
            "last_state_mutation_source": self._last_state_mutation_source,
            "last_state_mutation_at": float(self._last_state_mutation_at),
        }

    def _state_snapshot_nowait(self, max_wait_s: float = 0.05) -> dict[str, Any]:
        """Snapshot for telemetry readers — never blocks on a busy substrate.

        Tries the lock briefly; under contention returns the last published
        snapshot (its snapshot_age_s already tells consumers how stale it is)
        instead of stalling the caller — the event loop froze 5.7s live when
        get_status() waited behind a weight-cache rebuild.
        """
        acquired = self.sync_lock.acquire(timeout=max_wait_s)
        if acquired:
            try:
                snapshot = self._state_snapshot_locked()
            finally:
                self.sync_lock.release()
            self._last_published_snapshot = snapshot
            return snapshot
        published = self._last_published_snapshot
        if published is not None:
            stale = dict(published)
            last_update = float(stale.get("last_update") or 0.0)
            stale["snapshot_age_s"] = (
                max(0.0, time.time() - last_update) if last_update else float("inf")
            )
            return stale
        # No published snapshot yet (first read at boot): pay the blocking
        # read once rather than invent numbers.
        return self._state_snapshot()

    def _save_state(self):
        """Persist substrate state (atomic)."""
        import os
        import tempfile

        from .liquid_substrate import (
            logger,
        )

        try:
            # Atomic write for NPZ
            fd, temp_path = tempfile.mkstemp(dir=str(self.state_path.parent), suffix=".npz")
            try:
                with os.fdopen(fd, "wb") as f:
                    np.savez_compressed(f, x=self.x, W=self.W, tick=self.tick_count)
                os.replace(temp_path, str(self.state_path))
                logger.info("💾 Substrate state saved (atomic)")
            except (RuntimeError, AttributeError, TypeError, ValueError) as e:
                record_degradation("liquid_substrate", e)
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                raise e
        except OSError as e:
            record_degradation("liquid_substrate", e)
            logger.error("Failed to save substrate state: %s", e)

    def _load_state(self):
        from .liquid_substrate import (
            logger,
        )

        if not self.state_path.exists():
            return
        try:
            with open(self.state_path, "rb") as f:
                data = np.load(f)
                loaded_x = data["x"]
                loaded_weights = data["W"]
                n = self.config.neuron_count
                # Validate shapes match current config
                if loaded_x.ndim == 1 and loaded_x.shape != (n,):
                    saved_n = int(loaded_x.shape[0])
                    if self._explicit_config and not self._explicit_state_file:
                        logger.warning(
                            "Substrate state x dimension (%d) differs from explicit configured n=%d; "
                            "ignoring persisted state for this isolated substrate.",
                            saved_n,
                            n,
                        )
                        self.x = np.zeros(n)
                        self.W = self._rng.standard_normal((n, n)).astype(np.float32) * (
                            1.0 / np.sqrt(max(n, 1))
                        )
                        self.x_torch = torch.tensor(self.x, dtype=torch.float32, device=self.device)
                        self.v = np.zeros(n)
                        self.v_torch = torch.zeros(n, device=self.device)
                        self._sync_weight_cache_locked()
                        return
                    logger.warning(
                        "Substrate state x dimension (%d) differs from configured n=%d; "
                        "adopting saved dimension for continuity.",
                        saved_n,
                        n,
                    )
                    self.config.neuron_count = saved_n
                    n = saved_n
                    self.v = np.zeros(n)
                    self.x_torch = torch.zeros(n, device=self.device)
                    self.v_torch = torch.zeros(n, device=self.device)
                if loaded_x.shape != (n,):
                    logger.warning(
                        "Substrate state shape mismatch (saved x=%s vs config n=%d). "
                        "Reinitializing fresh state.",
                        loaded_x.shape,
                        n,
                    )
                    self.x = np.zeros(n)
                    self.W = self._rng.standard_normal((n, n)).astype(np.float32) * 0.1
                    self.x_torch = torch.tensor(self.x, dtype=torch.float32, device=self.device)
                    self.v = np.zeros(n)
                    self.v_torch = torch.zeros(n, device=self.device)
                    self._sync_weight_cache_locked()
                    return
                self.x = np.nan_to_num(loaded_x, nan=0.0, posinf=1.0, neginf=-1.0)
                if loaded_weights.shape == (n, n):
                    self.W = np.nan_to_num(loaded_weights, nan=0.0, posinf=5.0, neginf=-5.0)
                else:
                    logger.warning(
                        "Substrate W shape mismatch (saved W=%s vs n=%d); rebuilding W while preserving x.",
                        loaded_weights.shape,
                        n,
                    )
                    self.W = self._rng.standard_normal((n, n)).astype(np.float32) * (
                        1.0 / np.sqrt(max(n, 1))
                    )
                self.x_torch = torch.tensor(self.x, dtype=torch.float32, device=self.device)
                self.v_torch = torch.tensor(self.v, dtype=torch.float32, device=self.device)
                self._sync_weight_cache_locked()
                self.tick_count = int(data["tick"])
            logger.info("Substrate state restored.")
        except (RuntimeError, AttributeError, TypeError, ValueError) as e:
            record_degradation("liquid_substrate", e)
            logger.error("Failed to load substrate state: %s", e)
            self.x = np.zeros(self.config.neuron_count)
            self.W = (
                self._rng.standard_normal(
                    (self.config.neuron_count, self.config.neuron_count)
                ).astype(np.float32)
                * 0.1
            )
            self.x_torch = torch.tensor(self.x, dtype=torch.float32, device=self.device)
            self.v = np.zeros(self.config.neuron_count)
            self.v_torch = torch.zeros(self.config.neuron_count, device=self.device)
            self._sync_weight_cache_locked()
