"""CPU execution adapter for the sealed semantic neural machine.

The CP566 mechanism was measured with MLX tensors. Production chat also keeps
the resident 32B on MLX/Metal, where hundreds of tiny scalar submissions can
queue behind generation. This adapter preserves the learned coefficients,
harmonic decoder, state transitions, receipts, and tissue identity while
executing those bounded scalar operations on the dedicated CPU lane.

Activation binds this source, and runtime qualification compares every state
and transition receipt against the measured MLX implementation before serving.
"""

from __future__ import annotations

import math

from core.learning.semantic_neural_machine import (
    _LEARNED_ADD,
    _LEARNED_MUL,
    _LEARNED_SUB,
    MAX_PROCESS_INTEGER,
    PROCESS_RADIX,
    SemanticNeuralMachine,
)


class SemanticNeuralRuntimeMachine(SemanticNeuralMachine):
    """Numerically equivalent CPU backend for sealed recurrent tissue."""

    def __init__(self) -> None:
        super().__init__()
        self._runtime_coefficients = tuple(
            tuple(float(value) for value in row) for row in self.raw_coefficients.tolist()
        )
        self._runtime_harmonics = tuple(
            float(value) for value in self.tissue.harmonic_weights.tolist()
        )

    def _learned_raw(self, operation: int, left: int, right: int) -> int:
        if (
            operation not in (_LEARNED_ADD, _LEARNED_MUL, _LEARNED_SUB)
            or type(left) is not int
            or type(right) is not int
            or not -MAX_PROCESS_INTEGER <= left <= MAX_PROCESS_INTEGER
            or not -MAX_PROCESS_INTEGER <= right <= MAX_PROCESS_INTEGER
        ):
            raise ValueError("semantic learned arithmetic request is invalid")
        coefficients = self._runtime_coefficients[operation]
        scalar = (
            coefficients[0] * float(left)
            + coefficients[1] * float(right)
            + coefficients[2] * float(left) * float(right)
            + coefficients[3]
        )
        rounded = int(round(scalar))
        if not math.isfinite(scalar) or abs(scalar - rounded) > 1e-3:
            raise RuntimeError("learned arithmetic left the exact integer manifold")
        self._learned_operation_count += 1
        return rounded

    def _learned_radix_residue(self, value: int) -> int:
        def score(candidate: int) -> float:
            phase = (float(value) - float(candidate)) / float(PROCESS_RADIX)
            return sum(
                self._runtime_harmonics[harmonic - 1] * math.cos(2.0 * math.pi * harmonic * phase)
                for harmonic in range(1, self.tissue.config.harmonic_count + 1)
            )

        return max(range(PROCESS_RADIX), key=score)

    def _learned_exact_quotient(self, numerator: int, denominator: int) -> int:
        if not 0 < denominator <= MAX_PROCESS_INTEGER or not 0 <= numerator <= MAX_PROCESS_INTEGER:
            raise ValueError("semantic learned division request is invalid")
        coefficients = self._runtime_coefficients[_LEARNED_MUL]

        def distance(candidate: int) -> float:
            product = (
                coefficients[0] * float(candidate)
                + coefficients[1] * float(denominator)
                + coefficients[2] * float(candidate) * float(denominator)
                + coefficients[3]
            )
            return abs(product - float(numerator))

        quotient = min(range(MAX_PROCESS_INTEGER + 1), key=distance)
        if distance(quotient) > 1e-3:
            raise ValueError("semantic learned division is not exact")
        self._learned_operation_count += 1
        return quotient

    def _learned_floor_quotient(self, numerator: int, denominator: int) -> int:
        if not 0 < denominator <= MAX_PROCESS_INTEGER or not 0 <= numerator <= MAX_PROCESS_INTEGER:
            raise ValueError("semantic learned quotient request is invalid")
        coefficients = self._runtime_coefficients[_LEARNED_MUL]
        quotient = -1
        for candidate in range(MAX_PROCESS_INTEGER + 1):
            product = (
                coefficients[0] * float(candidate)
                + coefficients[1] * float(denominator)
                + coefficients[2] * float(candidate) * float(denominator)
                + coefficients[3]
            )
            if product <= float(numerator) + 1e-3:
                quotient = candidate
        if quotient < 0:
            raise RuntimeError("semantic learned floor quotient has no candidate")
        self._learned_operation_count += 1
        return quotient


__all__ = ["SemanticNeuralRuntimeMachine"]
