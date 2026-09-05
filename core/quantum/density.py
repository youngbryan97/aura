"""core/quantum/density.py
────────────────────────
Density-matrix simulation with noise channels — the NISQ-realistic half
of the quantum module.

The statevector engine simulates ideal closed-system evolution; real
quantum processors decohere. This module evolves mixed states ρ under
the same gates plus Kraus channels with closed-form physics:

- amplitude damping (T1 relaxation): P(|1⟩) decays as e^{-t/T1}
- phase damping (T2 dephasing): off-diagonals decay, populations hold
- depolarizing: ρ → (1-p)ρ + p·I/2ⁿ, fixed point = maximally mixed

Every channel's analytic law is asserted by tests, not assumed.
Capped at 10 qubits (2²⁰ complex128 = 16 MiB per ρ) — honest exact
simulation, same discipline as the statevector cap.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from core.quantum.statevector import (
    _GATES_1Q,
    QuantumCircuitError,
    Statevector,
)

MAX_DENSITY_QUBITS = 10


class DensityMatrix:
    """Mixed quantum state ρ with gate and Kraus-channel evolution."""

    def __init__(self, num_qubits: int):
        if not isinstance(num_qubits, int) or num_qubits < 1:
            raise QuantumCircuitError("num_qubits must be a positive integer")
        if num_qubits > MAX_DENSITY_QUBITS:
            raise QuantumCircuitError(
                f"{num_qubits} qubits exceeds the density-matrix cap of "
                f"{MAX_DENSITY_QUBITS}")
        self.num_qubits = num_qubits
        dim = 1 << num_qubits
        self.rho = np.zeros((dim, dim), dtype=np.complex128)
        self.rho[0, 0] = 1.0

    @classmethod
    def from_statevector(cls, state: Statevector) -> "DensityMatrix":
        if state.num_qubits > MAX_DENSITY_QUBITS:
            raise QuantumCircuitError("statevector too wide for density form")
        density = cls(state.num_qubits)
        density.rho = np.outer(state.state, np.conj(state.state))
        return density

    # ── operators ──────────────────────────────────────────────

    def _expand_1q(self, matrix: np.ndarray, qubit: int) -> np.ndarray:
        """Embed a single-qubit operator at ``qubit`` (qubit 0 = MSB)."""
        if not 0 <= qubit < self.num_qubits:
            raise QuantumCircuitError(f"qubit {qubit} out of range")
        op = np.array([[1.0]], dtype=np.complex128)
        for index in range(self.num_qubits):
            op = np.kron(op, matrix if index == qubit else np.eye(2))
        return op

    def apply_gate(self, name: str, qubit: int) -> "DensityMatrix":
        gate = _GATES_1Q.get(name.upper())
        if gate is None:
            raise QuantumCircuitError(f"unknown gate '{name}'")
        operator = self._expand_1q(gate, qubit)
        self.rho = operator @ self.rho @ operator.conj().T
        return self

    def apply_unitary_full(self, unitary: np.ndarray) -> "DensityMatrix":
        self.rho = unitary @ self.rho @ unitary.conj().T
        return self

    def apply_kraus(self, operators: Sequence[np.ndarray], qubit: int) -> "DensityMatrix":
        """Apply a single-qubit channel {K_i}: ρ → Σ K ρ K†. Completeness
        Σ K†K = I is verified — malformed channels are refused."""
        completeness = sum(
            (np.asarray(k, dtype=np.complex128).conj().T @ np.asarray(k))
            for k in operators
        )
        if not np.allclose(completeness, np.eye(2), atol=1e-9):
            raise QuantumCircuitError("Kraus operators must satisfy ΣK†K = I")
        expanded = [self._expand_1q(np.asarray(k, dtype=np.complex128), qubit)
                    for k in operators]
        self.rho = sum(k @ self.rho @ k.conj().T for k in expanded)
        return self

    # ── canonical noise channels ───────────────────────────────

    def amplitude_damping(self, gamma: float, qubit: int) -> "DensityMatrix":
        """T1 relaxation: |1⟩ decays to |0⟩ with probability γ."""
        if not 0.0 <= gamma <= 1.0:
            raise QuantumCircuitError("gamma must be in [0, 1]")
        k0 = np.array([[1, 0], [0, math.sqrt(1 - gamma)]])
        k1 = np.array([[0, math.sqrt(gamma)], [0, 0]])
        return self.apply_kraus([k0, k1], qubit)

    def phase_damping(self, lam: float, qubit: int) -> "DensityMatrix":
        """Pure dephasing: coherences decay, populations untouched."""
        if not 0.0 <= lam <= 1.0:
            raise QuantumCircuitError("lambda must be in [0, 1]")
        k0 = np.array([[1, 0], [0, math.sqrt(1 - lam)]])
        k1 = np.array([[0, 0], [0, math.sqrt(lam)]])
        return self.apply_kraus([k0, k1], qubit)

    def depolarizing(self, probability: float, qubit: int) -> "DensityMatrix":
        """ρ → (1-p)ρ + (p/3)(XρX + YρY + ZρZ)."""
        if not 0.0 <= probability <= 1.0:
            raise QuantumCircuitError("probability must be in [0, 1]")
        scale = math.sqrt(1.0 - probability)
        pauli_scale = math.sqrt(probability / 3.0)
        return self.apply_kraus([
            scale * np.eye(2),
            pauli_scale * _GATES_1Q["X"],
            pauli_scale * _GATES_1Q["Y"],
            pauli_scale * _GATES_1Q["Z"],
        ], qubit)

    # ── readout ────────────────────────────────────────────────

    def populations(self) -> np.ndarray:
        return np.real(np.diag(self.rho)).copy()

    def probability_of_one(self, qubit: int) -> float:
        bit = 1 << (self.num_qubits - 1 - qubit)
        populations = self.populations()
        return float(sum(
            p for index, p in enumerate(populations) if index & bit))

    def coherence(self, i: int = 0, j: int = 1) -> complex:
        return complex(self.rho[i, j])

    def purity(self) -> float:
        """Tr(ρ²): 1 for pure states, 1/2ⁿ for maximally mixed."""
        return float(np.real(np.trace(self.rho @ self.rho)))

    def fidelity_to_pure(self, state: np.ndarray) -> float:
        """⟨ψ|ρ|ψ⟩ against a pure reference."""
        vector = np.asarray(state, dtype=np.complex128)
        return float(np.real(np.conj(vector) @ self.rho @ vector))

    def trace(self) -> float:
        return float(np.real(np.trace(self.rho)))
