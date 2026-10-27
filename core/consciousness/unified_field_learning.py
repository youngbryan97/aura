"""How the field learns.

Hebbian plasticity between turns, and what a turn was worth taught to the
connections that were active in it. Lifted whole out of `unified_field`. Every name taken from it is imported at
CALL time: that module imports this one to build the class, and a test that
patches a name on it has to reach the code that reads it.
"""
from __future__ import annotations


class _HowTheFieldLearns:
    """Lifted whole out of UnifiedField; see unified_field.py."""

    def _apply_plasticity(self) -> None:
        """Hebbian learning on the recurrent field connectivity.

        Uses scaled rank-1 update instead of full outer product for efficiency.
        Syncs sparse representation after plasticity.

        Under the lock the rest of this class already uses. This is the one
        path that mutates the weight matrix in place — pruning below a
        threshold, zeroing the diagonal, rescaling — and it was the one path
        that did not take it.
        """
        with self._lock:
            self._apply_plasticity_locked()

    def _apply_plasticity_locked(self) -> None:
        from .unified_field import (
            np,
        )

        self.F = self._safe_reshape(
            self.F,
            self.cfg.dim,
            source="field_state_for_plasticity",
            record=True,
            clip_abs=1.0,
        )
        self.W_field = self._normalize_matrix(
            self.W_field,
            shape=(self.cfg.dim, self.cfg.dim),
            name="W_field",
            scale=0.05,
        )
        # Scaled rank-1 Hebbian update: W += lr * F @ F^T
        # np.outer is fine here since this runs every 10 ticks, not every tick.
        dw = self.cfg.hebbian_rate * np.outer(self.F, self.F)
        self.W_field += dw.astype(np.float32)

        # What each recurrent connection did this step, for the turn's worth to
        # judge: the unit it feeds after the unit it reads. The input weights
        # keep their trace on every tick (`_note_what_the_inputs_did`).
        pre = self._prev_F if self._prev_F is not None else self.F
        self._eligible_field += np.outer(self.F, pre).astype(np.float32)
        self._eligible_steps += 1

        self._settle_field_weights_locked()

    def _note_what_the_inputs_did(self, input_vec: object) -> None:
        """Keep each input weight's trace for this tick: the unit it drives times the input it carried.

        On every tick, at a tenth of a plasticity step's weight each. The
        field takes input on every other tick, since the bridge feeds it at
        half its rate, and kept only on its plasticity ticks the trace saw
        input or never did by the parity of its tick count: a run forked from
        an anchor on an odd count taught its input weights nothing.
        """
        from .unified_field import (
            np,
        )

        share = 1.0 / float(max(1, self.cfg.plasticity_interval))
        with self._lock:
            self._eligible_input += (share * np.outer(self.F, input_vec)).astype(np.float32)

    def _settle_field_weights_locked(self) -> None:
        """Bound the recurrent weights after they change, and resync the sparse copy."""
        from .unified_field import (
            _record_unified_field_degradation,
            np,
        )

        # NaN/Inf guard
        self.W_field = np.nan_to_num(self.W_field, nan=0.0, posinf=3.0, neginf=-3.0)

        # Per-update normalization (prevents long-run drift before sparsity enforcement)
        norm = np.linalg.norm(self.W_field)
        if not np.isfinite(norm):
            _record_unified_field_degradation(
                FloatingPointError("non-finite recurrent field norm"),
                action="reset UnifiedField recurrent weights after plasticity drift",
                severity="critical",
            )
            self.W_field = np.zeros((self.cfg.dim, self.cfg.dim), dtype=np.float32)
            norm = 0.0
        if norm > 4.0:
            self.W_field *= 4.0 / norm

        # Sparsity enforcement: prune weakest connections back to target density
        abs_weights = np.abs(self.W_field)
        target_nonzero = max(
            1,
            int(self.cfg.recurrent_sparsity * self.cfg.dim * self.cfg.dim),
        )
        current_nonzero = np.count_nonzero(self.W_field)
        if current_nonzero > target_nonzero * 1.5:
            threshold = np.sort(abs_weights.ravel())[-target_nonzero]
            self.W_field[abs_weights < threshold] = 0.0

        np.fill_diagonal(self.W_field, 0.0)

        # Sync sparse representation for tick matmul
        self._W_field_sparse = self._to_sparse(self.W_field)

    def teach(self, modulator: float) -> dict[str, object]:
        """Strengthen what was active before a good turn and weaken what was active before a bad one.

        `modulator` is the turn's worth as her chemistry got it: its sign, and
        its size against her recent turns, in [-1, 1]. Each connection moves
        by the modulator times what it did over the turn, at `hebbian_rate`,
        the rate her unconditioned learning already runs at, so a turn's payoff
        can move a connection as far as a turn of co-activity does. The
        connection keeps the trace for the whole turn and is judged when the
        turn ends: a three-factor rule (Fremaux and Gerstner 2016).

        Each unit's input weights are then scaled back to the strength it was
        born with, as synaptic scaling keeps a neuron's total drive (Turrigiano
        2008). A source that was active before payoffs takes a larger share of
        the units it drives from the sources that were not, and one active
        before detriments gives its share up. The traces start again either
        way, a neutral turn included. Returns each source's share of the
        field's input after the lesson.
        """
        from .unified_field import (
            _clamp_float,
            _finite_float,
            np,
        )

        with self._lock:
            modulator, valid = _finite_float(modulator, 0.0)
            modulator, _ = _clamp_float(modulator, lower=-1.0, upper=1.0)
            steps = self._eligible_steps
            if valid and steps and modulator != 0.0:
                rate = self.cfg.hebbian_rate * modulator
                self.W_field += (rate * self._eligible_field).astype(np.float32)
                self._settle_field_weights_locked()
                taught = self._W_input_batched + (rate * self._eligible_input).astype(np.float32)
                taught = np.nan_to_num(taught, nan=0.0, posinf=0.0, neginf=0.0)
                norms = np.linalg.norm(taught, axis=1)
                scale = np.where(norms > 0.0, self._input_row_norms / np.maximum(norms, 1e-12), 1.0)
                taught = (taught * scale[:, None]).astype(np.float32)
                blocks = np.split(taught, np.cumsum(self._input_dims)[:-1], axis=1)
                self.W_mesh, self.W_chem, self.W_bind, self.W_intero, self.W_substrate = blocks
                self._sync_input_weight_matrix()
                self._taught_turns += 1
            self._eligible_field[:] = 0.0
            self._eligible_input[:] = 0.0
            self._eligible_steps = 0
            self._last_teaching = {
                "modulator": round(float(modulator), 6),
                "steps": steps,
                "taught": bool(valid and steps and modulator != 0.0),
                "turns_taught": self._taught_turns,
                "shares": self.input_shares(),
            }
            return dict(self._last_teaching)

    def input_shares(self) -> dict[str, float]:
        """Each source's share of the field's input strength, summed over its units."""
        from .unified_field import (
            np,
        )

        blocks = (self.W_mesh, self.W_chem, self.W_bind, self.W_intero, self.W_substrate)
        strengths = [float(np.sum(np.square(block))) for block in blocks]
        total = sum(strengths)
        if total <= 0.0:
            return {name: 0.0 for name in self._INPUT_SOURCES}
        return {name: round(value / total, 6) for name, value in zip(self._INPUT_SOURCES, strengths, strict=True)}

