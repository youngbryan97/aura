"""The mesh learning by spike timing: STDP over its columns.

Lifted whole out of `neural_mesh`. Every name taken from it is imported at
CALL time: that module imports this one to build the class, and a test that
patches a name on it has to reach the code that reads it.
"""
from __future__ import annotations


class _LearnsBySpikeTiming:
    """Lifted whole out of NeuralMesh; see neural_mesh.py."""

    def _apply_stdp(self, now: float) -> None:
        """Spike-Timing-Dependent Plasticity within each column.

        Pre-before-post → potentiate (causal)
        Post-before-pre → depress   (acausal)
        """
        from .neural_mesh import (
            _clamp_float,
            _finite_float,
            _record_neural_mesh_degradation,
            np,
        )

        if self._foreground_request_active():
            return

        lr = self.cfg.stdp_lr * self._modulatory_state[1]
        inhibitory_lr = lr * self.cfg.inhibitory_stdp_rate_ratio
        inhibitory_window, inhibitory_window_valid = _finite_float(
            self.cfg.inhibitory_stdp_window, 0.02
        )
        inhibitory_window, inhibitory_window_unchanged = _clamp_float(
            inhibitory_window, lower=1e-6, upper=5.0
        )
        if not (inhibitory_window_valid and inhibitory_window_unchanged):
            _record_neural_mesh_degradation(
                ValueError("NeuralMesh inhibitory plasticity window was out of bounds"),
                action="normalized the inhibitory plasticity window",
                severity="warning",
                extra={"window": inhibitory_window},
            )
        inhibitory_alpha, alpha_valid = _finite_float(
            self.cfg.inhibitory_stdp_depression, 0.2
        )
        inhibitory_alpha, alpha_unchanged = _clamp_float(
            inhibitory_alpha, lower=0.0, upper=10.0
        )
        if not (alpha_valid and alpha_unchanged):
            _record_neural_mesh_degradation(
                ValueError("NeuralMesh inhibitory plasticity constant was out of bounds"),
                action="normalized the inhibitory depression constant",
                severity="warning",
                extra={"alpha": inhibitory_alpha},
            )
        lr, lr_valid = _finite_float(lr, self.cfg.stdp_lr)
        lr, lr_unchanged = _clamp_float(lr, lower=0.0, upper=0.05)
        window, window_valid = _finite_float(self.cfg.stdp_window, 0.02)
        window, window_unchanged = _clamp_float(window, lower=1e-6, upper=5.0)
        a_plus, a_plus_valid = _finite_float(self.cfg.stdp_potentiation, 1.0)
        a_minus, a_minus_valid = _finite_float(self.cfg.stdp_depression, 0.5)
        a_plus, a_plus_unchanged = _clamp_float(a_plus, lower=0.0, upper=10.0)
        a_minus, a_minus_unchanged = _clamp_float(a_minus, lower=0.0, upper=10.0)
        if not all(
            (
                lr_valid,
                lr_unchanged,
                window_valid,
                window_unchanged,
                a_plus_valid,
                a_minus_valid,
                a_plus_unchanged,
                a_minus_unchanged,
            )
        ):
            _record_neural_mesh_degradation(
                ValueError("NeuralMesh STDP parameters were non-finite or out of bounds"),
                action="normalized NeuralMesh STDP parameters before plasticity update",
                severity="warning",
                extra={"lr": lr, "window": window, "a_plus": a_plus, "a_minus": a_minus},
            )

        for col in self.columns:
            t = np.nan_to_num(col.last_spike_time, nan=-1.0, posinf=-1.0, neginf=-1.0)
            col.last_spike_time = t
            # Only process neurons that have spiked at least once
            active = t > 0
            if not np.any(active):
                continue

            # `recurrent = W @ x` makes W[i, j] the weight FROM j INTO i, so
            # row i is the postsynaptic cell and column j the presynaptic one.
            # This is the axis the whole rule hangs off, and it was read the
            # other way round: the code potentiated on t_i < t_j, which under
            # this convention is POST before PRE. The mesh was running
            # anti-Hebbian, strengthening exactly the pairs a causal rule
            # weakens.
            delay = t[None, :] - t[:, None]  # pre minus post, so causal is < 0
            both = active[:, None] & active[None, :]
            causal = (delay < 0) & (delay > -window) & both
            acausal = (delay > 0) & (delay < window) & both
            near = np.clip(-np.abs(delay) / window, -20.0, 20.0)
            # The inhibitory window is its own, 20 ms against the excitatory
            # 16.8, and both come from the papers rather than from each other.
            inhibitory_near = np.clip(
                -np.abs(delay) / inhibitory_window, -20.0, 20.0
            )
            inhibitory_paired = (np.abs(delay) < inhibitory_window) & both

            # The excitatory rule, from Bi and Poo: pre before post strengthens,
            # post before pre weakens, and the two halves are not the same size.
            excitatory_dw = lr * (
                np.where(causal, a_plus * np.exp(near), 0.0)
                - np.where(acausal, a_minus * np.exp(near), 0.0)
            ).astype(np.float32)

            # The inhibitory rule, from Vogels et al. One rule for every synapse
            # was the last plasticity assumption left in the mesh, and it is not
            # what was measured: inhibitory synapses learn over a SYMMETRIC
            # window — either order of firing strengthens them — against a
            # standing depression on every presynaptic spike. What that pair of
            # terms does is hold the cell being inhibited near a target rate, so
            # inhibition tracks excitation instead of being learned by a rule
            # derived from excitatory pairs.
            presynaptic_fired = np.broadcast_to(active[None, :], both.shape)
            inhibitory_dw = inhibitory_lr * (
                np.where(inhibitory_paired, np.exp(inhibitory_near), 0.0)
                - np.where(presynaptic_fired, inhibitory_alpha, 0.0)
            ).astype(np.float32)

            # Presynaptic cell class picks the rule, because that is what the
            # synapse belongs to. Columns are presynaptic here.
            dw = np.where(col.inh_mask[None, :], inhibitory_dw, excitatory_dw)
            dw = np.nan_to_num(dw, nan=0.0, posinf=0.0, neginf=0.0)
            # Plasticity changes synapses that exist. It does not grow them.
            #
            # There was no such mask, so any two units that fired inside the
            # window acquired a synapse whether or not one had ever been wired,
            # and a column's density climbed toward full the longer it ran. All
            # the measured densities above describe the mesh at construction and
            # nothing was holding them there.
            dw = dw * (col.W != 0)
            # Soft bounds: a synapse strengthens by less the nearer it is to the
            # bound of one and weakens in proportion to its strength (van
            # Rossum, Bi and Turrigiano 2000; Guetig et al. 2003). With hard
            # bounds alone, potentiation piled weights up at the clip: on 21
            # October the mesh's strongest weight climbed from 0.8 to 1.0 in
            # every run with the conjunctive gains on and sat there, a trend the
            # core read as a clock, and stayed at 0.8 with them off.
            strength = np.abs(col.W)
            dw = np.where(dw > 0.0, dw * (1.0 - strength), dw * strength).astype(np.float32)

            # Both rules change a synapse's STRENGTH; neither may change what it
            # does. An inhibitory synapse is negative, so strengthening it means
            # subtracting.
            col.W = np.nan_to_num(col.W, nan=0.0, posinf=1.0, neginf=-1.0)
            polarity = np.where(col.inh_mask[None, :], -1.0, 1.0).astype(np.float32)
            old_sign = np.sign(col.W)
            col.W = col.W + dw * polarity
            # A synapse that would change sign has been driven to nothing, which
            # is where it stops.
            sign_flipped = (np.sign(col.W) != old_sign) & (old_sign != 0)
            col.W[sign_flipped] = 0.0

            # Normalize to prevent runaway
            norm = np.linalg.norm(col.W)
            if norm > 5.0:
                col.W *= 5.0 / norm
            col.W = np.nan_to_num(
                np.clip(col.W, -1.0, 1.0),
                nan=0.0,
                posinf=1.0,
                neginf=-1.0,
            ).astype(np.float32, copy=False)

