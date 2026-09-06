"""core/skills/quantum_lab.py
───────────────────────────
Quantum computation as a governed, pure-compute capability.

Gives Aura's cognition a real quantum circuit simulator: entangled
states, teleportation, Grover search, QFT verification, and free-form
small circuits — every result cross-checked against analytic ground
truth where one exists. Randomness is source-attributed: external quantum
entropy is reported only when the bridge confirms it, OS fallback stays
labelled as fallback, and deterministic algorithms report that entropy was
not used. Unitary evolution remains classical simulation in every mode.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from core.skills.what_every_skill_gives_back import THE_SHARED_RESULT
from core.skills.base_skill import BaseSkill

_QUANTUM_LAB_ERRORS = (ImportError, AttributeError, RuntimeError, TypeError, ValueError)


class _EntropyAudit:
    """Callable entropy source with before/after bridge provenance."""

    def __init__(self, bridge: Any) -> None:
        self._bridge = bridge
        self._stats_error: str | None = None
        self._before = self._read_stats()
        self.calls = 0
        self.failures = 0

    def _read_stats(self) -> dict[str, Any] | None:
        try:
            stats = self._bridge.get_stats()
            if not isinstance(stats, Mapping):
                raise TypeError("entropy bridge statistics must be a mapping")
            return dict(stats)
        except _QUANTUM_LAB_ERRORS as exc:
            self._stats_error = f"{type(exc).__name__}: {exc}"[:300]
            return None

    def __call__(self) -> float:
        self.calls += 1
        try:
            value = float(self._bridge.get_quantum_float())
        except _QUANTUM_LAB_ERRORS:
            self.failures += 1
            raise
        if not math.isfinite(value) or not 0.0 <= value < 1.0:
            self.failures += 1
            raise ValueError("entropy bridge returned a value outside [0, 1)")
        return value

    def report(self) -> dict[str, Any]:
        after = self._read_stats()
        stats_available = self._before is not None and after is not None
        quantum_reads = 0
        fallback_reads = 0
        if stats_available and self._before is not None and after is not None:
            quantum_reads = max(
                0,
                int(after.get("quantum_reads", 0) or 0)
                - int(self._before.get("quantum_reads", 0) or 0),
            )
            fallback_reads = max(
                0,
                int(after.get("fallback_reads", 0) or 0)
                - int(self._before.get("fallback_reads", 0) or 0),
            )
        if self.calls == 0:
            mode = "not_used"
        elif self.failures:
            mode = "entropy_bridge_failed_to_prng"
        elif not stats_available:
            mode = "entropy_bridge_source_unattributed"
        elif quantum_reads and fallback_reads:
            mode = "mixed_external_quantum_and_os_entropy"
        elif quantum_reads:
            mode = "external_quantum_entropy"
        elif fallback_reads:
            mode = "os_entropy_fallback"
        else:
            mode = "entropy_bridge_source_unattributed"
        return {
            "entropy_mode": mode,
            "bridge_draws": self.calls,
            "bridge_failures": self.failures,
            "quantum_reads": quantum_reads,
            "fallback_reads": fallback_reads,
            "stats_available": stats_available,
            "provenance_error": self._stats_error,
        }


def _entropy_source() -> _EntropyAudit | None:
    """Return an attributed entropy bridge, if it can be initialized."""
    try:
        from core.consciousness.quantum_entropy import get_quantum_entropy

        return _EntropyAudit(get_quantum_entropy())
    except _QUANTUM_LAB_ERRORS:
        return None


QuantumAction = Literal["bell", "ghz", "grover", "teleport", "qft_verify", "circuit"]


class QuantumLabInput(BaseModel):  # type: ignore[misc]
    """Typed public contract for the action-oriented quantum skill."""

    model_config = ConfigDict(extra="forbid")

    action: QuantumAction = "bell"
    num_qubits: int | None = Field(default=None, ge=1, le=20)
    marked: int = Field(default=0, ge=0)
    alpha_real: float = Field(default=1.0, allow_inf_nan=False)
    alpha_imag: float = Field(default=0.0, allow_inf_nan=False)
    beta_real: float = Field(default=1.0, allow_inf_nan=False)
    beta_imag: float = Field(default=0.0, allow_inf_nan=False)
    gates: list[list[Any]] = Field(default_factory=list, max_length=512)
    shots: int = Field(default=512, ge=1, le=65_536)
    seed: int | None = Field(default=None, ge=0, le=(1 << 64) - 1)


class QuantumLabSkill(BaseSkill):  # type: ignore[misc]
    #: What a caller gets back. The shared part only: every skill here
    #: returns `ok`, and a schema claiming to be complete would be wrong
    #: for every one that adds a field.
    result_schema = THE_SHARED_RESULT

    name = "quantum_lab"
    description = (
        "Run quantum circuit simulations: Bell/GHZ entanglement, Grover search, "
        "quantum teleportation, QFT, or a custom small circuit. Exact statevector "
        "simulation with analytic cross-checks; honest about being simulation."
    )
    effect_scope = "pure_compute"
    input_model = QuantumLabInput
    output = "Simulation results with analytic verification where defined"
    execution_profile = "cpu"
    memory_mb_estimate = 256
    metabolic_cost = 2

    def match(self, goal: dict[str, Any]) -> bool:
        objective = str(goal.get("objective", "")).lower()
        keywords = (
            "quantum",
            "qubit",
            "entangle",
            "superposition",
            "grover",
            "teleport",
            "bell state",
            "ghz",
            "qft",
            "quantum fourier",
        )
        return any(keyword in objective for keyword in keywords)

    async def execute(self, params: Any, context: dict[str, Any]) -> dict[str, Any]:
        import numpy as np

        from core.quantum import (
            MAX_QUBITS,
            QuantumCircuitError,
            Statevector,
            bell_pair,
            ghz_state,
            grover_search,
            qft_circuit,
            teleport,
        )
        from core.quantum.algorithms import qft_matrix

        try:
            request = (
                params
                if isinstance(params, QuantumLabInput)
                else QuantumLabInput.model_validate(params if isinstance(params, dict) else {})
            )
        except ValidationError as exc:
            return {"ok": False, "error": f"invalid quantum parameters: {exc}"}
        payload = request.model_dump()
        action = request.action
        seed = request.seed
        shots = request.shots
        entropy = None if seed is not None else _entropy_source()

        try:
            if action == "bell":
                if request.num_qubits not in {None, 2}:
                    raise QuantumCircuitError("bell action requires exactly 2 qubits")
                sv = bell_pair(seed=seed, entropy_source=entropy)
                counts = sv.sample_counts(shots)
                entropy_report = self._entropy_report(seed, entropy)
                return self._ok(
                    action,
                    entropy_report,
                    {
                        "counts": counts,
                        "sampling_rng": self._rng_label(entropy_report),
                        "zz_correlation": sv.expectation_pauli("ZZ"),
                        "analytic": {"p_00": 0.5, "p_11": 0.5, "zz_correlation": 1.0},
                        "summary": (
                            f"Bell pair over {shots} shots: {counts}. "
                            "Only 00/11 appear and ⟨ZZ⟩=1 — the qubits are maximally entangled."
                        ),
                    },
                )

            if action == "ghz":
                n = self._bounded_qubits(
                    payload, action=action, default=3, minimum=2, cap=MAX_QUBITS
                )
                sv = ghz_state(n, seed=seed, entropy_source=entropy)
                counts = sv.sample_counts(shots)
                entropy_report = self._entropy_report(seed, entropy)
                return self._ok(
                    action,
                    entropy_report,
                    {
                        "num_qubits": n,
                        "counts": counts,
                        "sampling_rng": self._rng_label(entropy_report),
                        "analytic": {"p_" + "0" * n: 0.5, "p_" + "1" * n: 0.5},
                        "summary": f"{n}-qubit GHZ state sampled {shots} times: {counts}.",
                    },
                )

            if action == "grover":
                n = self._bounded_qubits(payload, action=action, default=4, minimum=1, cap=12)
                marked = request.marked
                result = grover_search(n, marked, seed=seed, entropy_source=entropy)
                deviation = abs(result["success_probability"] - result["analytic_prediction"])
                return self._ok(
                    action,
                    self._entropy_report(seed, entropy, used=False),
                    {
                        "num_candidates": result["num_candidates"],
                        "iterations": result["iterations"],
                        "success_probability": result["success_probability"],
                        "analytic_prediction": result["analytic_prediction"],
                        "matches_theory": deviation < 1e-9,
                        "summary": (
                            f"Grover over {result['num_candidates']} items found the marked "
                            f"item with p={result['success_probability']:.4f} after "
                            f"{result['iterations']} iterations (theory: "
                            f"{result['analytic_prediction']:.4f})."
                        ),
                    },
                )

            if action == "teleport":
                alpha = complex(request.alpha_real, request.alpha_imag)
                beta = complex(request.beta_real, request.beta_imag)
                result = teleport(alpha, beta, seed=seed, entropy_source=entropy)
                entropy_report = self._entropy_report(seed, entropy)
                return self._ok(
                    action,
                    entropy_report,
                    {
                        **result,
                        "measurement_rng": self._rng_label(entropy_report),
                        "summary": (
                            f"Teleported α|0⟩+β|1⟩ with corrections (m0={result['m0']}, "
                            f"m1={result['m1']}); received-state fidelity "
                            f"{result['fidelity']:.6f}."
                        ),
                    },
                )

            if action == "qft_verify":
                n = self._bounded_qubits(payload, action=action, default=4, minimum=1, cap=8)
                max_error = 0.0
                for basis in range(1 << n):
                    sv = Statevector(n, seed=seed or 0)
                    sv.state[:] = 0.0
                    sv.state[basis] = 1.0
                    qft_circuit(sv)
                    expected = qft_matrix(n)[:, basis]
                    max_error = max(max_error, float(np.max(np.abs(sv.state - expected))))
                return self._ok(
                    action,
                    self._entropy_report(seed, entropy, used=False),
                    {
                        "num_qubits": n,
                        "max_amplitude_error": max_error,
                        "verified": max_error < 1e-10,
                        "summary": (
                            f"QFT circuit on {n} qubits reproduces the analytic DFT matrix "
                            f"on all {1 << n} basis states (max error {max_error:.2e})."
                        ),
                    },
                )

            if action == "circuit":
                return self._run_circuit(payload, shots, seed, entropy)

            return {"ok": False, "error": f"Unknown quantum_lab action '{action}'"}
        except (OverflowError, QuantumCircuitError, TypeError, ValueError) as exc:
            return {"ok": False, "error": str(exc)}

    # ── helpers ────────────────────────────────────────────────

    @staticmethod
    def _bounded_qubits(
        params: dict[str, Any],
        *,
        action: str,
        default: int,
        minimum: int,
        cap: int,
    ) -> int:
        raw = params.get("num_qubits")
        n = default if raw is None else int(raw)
        if not minimum <= n <= cap:
            raise ValueError(f"{action} action supports {minimum}..{cap} qubits; received {n}")
        return n

    @staticmethod
    def _rng_label(entropy_report: dict[str, Any]) -> str:
        mode = str(entropy_report.get("entropy_mode") or "")
        if mode == "seeded_prng":
            return "seeded_prng"
        if mode == "os_seeded_prng":
            return "os_seeded_prng"
        if mode == "entropy_bridge_failed_to_prng":
            return "os_seeded_prng_fallback"
        if mode == "entropy_bridge_source_unattributed":
            return "unattributed_entropy_seeded_prng"
        if mode == "not_used":
            return "not_used"
        return "entropy_seeded_prng"

    @staticmethod
    def _entropy_report(
        seed: int | None,
        entropy: _EntropyAudit | None,
        *,
        used: bool = True,
    ) -> dict[str, Any]:
        if not used:
            report: dict[str, Any] = (
                entropy.report()
                if entropy is not None
                else {
                    "bridge_draws": 0,
                    "bridge_failures": 0,
                    "quantum_reads": 0,
                    "fallback_reads": 0,
                    "stats_available": False,
                    "provenance_error": None,
                }
            )
            report["entropy_mode"] = "not_used"
            return report
        if seed is not None:
            return {
                "entropy_mode": "seeded_prng",
                "bridge_draws": 0,
                "bridge_failures": 0,
                "quantum_reads": 0,
                "fallback_reads": 0,
                "stats_available": True,
                "provenance_error": None,
            }
        if entropy is None:
            return {
                "entropy_mode": "os_seeded_prng",
                "bridge_draws": 0,
                "bridge_failures": 0,
                "quantum_reads": 0,
                "fallback_reads": 0,
                "stats_available": False,
                "provenance_error": "entropy bridge unavailable",
            }
        return entropy.report()

    @staticmethod
    def _ok(
        action: str,
        entropy_report: dict[str, Any],
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "ok": True,
            "action": action,
            "entropy_mode": entropy_report["entropy_mode"],
            "entropy_provenance": entropy_report,
            "honest_framing": (
                "Exact classical simulation of quantum dynamics; not quantum hardware."
            ),
            **payload,
        }

    def _run_circuit(
        self,
        params: dict[str, Any],
        shots: int,
        seed: int | None,
        entropy: _EntropyAudit | None,
    ) -> dict[str, Any]:
        from core.quantum import Statevector

        n = self._bounded_qubits(
            params,
            action="circuit",
            default=2,
            minimum=1,
            cap=12,
        )
        sv = Statevector(n, seed=seed, entropy_source=entropy)
        gates = params.get("gates") or []
        if not isinstance(gates, list) or len(gates) > 512:
            return {"ok": False, "error": "gates must be a list of at most 512 steps"}
        single = {"h", "x", "y", "z", "s", "sdg", "t", "tdg"}
        rotations = {"rx", "ry", "rz", "phase"}
        for step in gates:
            if not isinstance(step, (list, tuple)) or not step:
                return {"ok": False, "error": f"malformed gate step: {step!r}"}
            op = str(step[0]).lower()
            args = step[1:]
            if op in single and len(args) == 1:
                getattr(sv, op)(int(args[0]))
            elif op in rotations and len(args) == 2:
                getattr(sv, op)(float(args[1]), int(args[0]))
            elif op in {"cx", "cz", "swap"} and len(args) == 2:
                getattr(sv, op)(int(args[0]), int(args[1]))
            elif op == "ccx" and len(args) == 3:
                sv.ccx(int(args[0]), int(args[1]), int(args[2]))
            else:
                return {"ok": False, "error": f"unsupported gate step: {step!r}"}
        counts = sv.sample_counts(shots)
        norm = float(sum(p for p in sv.probabilities()))
        entropy_report = self._entropy_report(seed, entropy)
        return self._ok(
            "circuit",
            entropy_report,
            {
                "num_qubits": n,
                "gate_count": sv.gate_count,
                "counts": counts,
                "sampling_rng": self._rng_label(entropy_report),
                "norm_preserved": math.isclose(norm, 1.0, abs_tol=1e-9),
                "summary": f"Ran {sv.gate_count} gates on {n} qubits; counts {counts}.",
            },
        )
