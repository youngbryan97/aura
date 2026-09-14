"""Quantum computational module: analytic verification battery.

Every assertion here has a closed-form quantum-mechanical answer; the
simulator must reproduce it to numerical precision, not approximately.
"""

from __future__ import annotations

import numpy as np
import pytest

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

# ── Elementary gate identities ───────────────────────────────────


def test_h_twice_is_identity():
    sv = Statevector(1, seed=7).h(0).h(0)
    assert abs(sv.state[0] - 1.0) < 1e-12
    assert abs(sv.state[1]) < 1e-12


def test_x_flips_basis_state():
    sv = Statevector(2, seed=7).x(0)
    assert sv.probability_of("10") == pytest.approx(1.0)


def test_gates_preserve_norm():
    sv = Statevector(3, seed=7)
    sv.h(0).cx(0, 1).t(2).rz(0.7, 1).ry(1.1, 0).ccx(0, 1, 2).swap(0, 2)
    assert float(np.sum(sv.probabilities())) == pytest.approx(1.0, abs=1e-12)


def test_expectation_values_on_known_states():
    zero = Statevector(1, seed=1)
    assert zero.expectation_pauli("Z") == pytest.approx(1.0)
    plus = Statevector(1, seed=1).h(0)
    assert plus.expectation_pauli("X") == pytest.approx(1.0)
    assert plus.expectation_pauli("Z") == pytest.approx(0.0, abs=1e-12)


# ── Entanglement ─────────────────────────────────────────────────


def test_bell_pair_amplitudes_and_correlations():
    sv = bell_pair(seed=3)
    assert sv.probability_of("00") == pytest.approx(0.5)
    assert sv.probability_of("11") == pytest.approx(0.5)
    assert sv.probability_of("01") == pytest.approx(0.0, abs=1e-12)
    # Perfect correlation in both Z and X bases — the entanglement witness.
    assert sv.expectation_pauli("ZZ") == pytest.approx(1.0)
    assert sv.expectation_pauli("XX") == pytest.approx(1.0)
    # No signaling: each local marginal stays maximally mixed.
    assert sv.expectation_pauli("ZI") == pytest.approx(0.0, abs=1e-12)
    assert sv.expectation_pauli("IZ") == pytest.approx(0.0, abs=1e-12)


def test_bell_measurements_always_agree():
    for seed in range(20):
        sv = bell_pair(seed=seed)
        assert sv.measure(0) == sv.measure(1)


def test_ghz_state_structure():
    sv = ghz_state(4, seed=5)
    assert sv.probability_of("0000") == pytest.approx(0.5)
    assert sv.probability_of("1111") == pytest.approx(0.5)


# ── Measurement semantics ────────────────────────────────────────


def test_measurement_collapse_is_stable():
    sv = Statevector(1, seed=11).h(0)
    first = sv.measure(0)
    for _ in range(5):
        assert sv.measure(0) == first
    assert float(np.sum(sv.probabilities())) == pytest.approx(1.0)


def test_entropy_source_drives_collapse():
    # entropy → 0.99: r < p(1) is false for a fair qubit → outcome 0.
    sv = Statevector(1, entropy_source=lambda: 0.99).h(0)
    assert sv.measure(0) == 0
    # entropy → 0.01 forces outcome 1.
    sv = Statevector(1, entropy_source=lambda: 0.01).h(0)
    assert sv.measure(0) == 1


def test_sampling_is_deterministic_with_seed():
    counts_a = bell_pair(seed=42).sample_counts(1000)
    counts_b = bell_pair(seed=42).sample_counts(1000)
    assert counts_a == counts_b
    assert set(counts_a) <= {"00", "11"}
    assert sum(counts_a.values()) == 1000


def test_entropy_source_seeds_sampling_instead_of_being_ignored():
    calls = []

    def entropy():
        calls.append(True)
        return 0.125

    counts_a = bell_pair(entropy_source=entropy).sample_counts(500)
    counts_b = bell_pair(entropy_source=entropy).sample_counts(500)

    assert len(calls) == 2
    assert counts_a == counts_b
    assert set(counts_a) <= {"00", "11"}


# ── Canonical algorithms ─────────────────────────────────────────


@pytest.mark.parametrize(
    "alpha,beta",
    [
        (1.0, 0.0),
        (0.0, 1.0),
        (1.0, 1.0),
        (0.6, 0.8j),
        (0.5 + 0.5j, 0.5 - 0.5j),
    ],
)
def test_teleportation_has_unit_fidelity(alpha, beta):
    for seed in range(8):
        result = teleport(alpha, beta, seed=seed)
        assert result["fidelity"] == pytest.approx(1.0, abs=1e-10)


def test_teleportation_exercises_all_correction_branches():
    branches = {
        (teleport(0.6, 0.8, seed=seed)["m0"], teleport(0.6, 0.8, seed=seed)["m1"])
        for seed in range(64)
    }
    assert branches == {(0, 0), (0, 1), (1, 0), (1, 1)}


@pytest.mark.parametrize("num_qubits", [2, 3, 4, 6, 8])
def test_grover_matches_analytic_success_probability(num_qubits):
    marked = (1 << num_qubits) - 2
    result = grover_search(num_qubits, marked, seed=0)
    assert result["success_probability"] == pytest.approx(result["analytic_prediction"], abs=1e-9)
    if num_qubits >= 3:
        assert result["success_probability"] > 0.9


@pytest.mark.parametrize("num_qubits", [1, 2, 3, 4, 5])
def test_qft_circuit_reproduces_analytic_matrix(num_qubits):
    analytic = qft_matrix(num_qubits)
    for basis in range(1 << num_qubits):
        sv = Statevector(num_qubits, seed=0)
        sv.state[:] = 0.0
        sv.state[basis] = 1.0
        qft_circuit(sv)
        np.testing.assert_allclose(sv.state, analytic[:, basis], atol=1e-10)


# ── Bounds and error handling ────────────────────────────────────


def test_qubit_cap_enforced():
    with pytest.raises(QuantumCircuitError):
        Statevector(MAX_QUBITS + 1)


def test_invalid_operations_raise():
    sv = Statevector(2, seed=0)
    with pytest.raises(QuantumCircuitError):
        sv.h(5)
    with pytest.raises(QuantumCircuitError):
        sv.cx(1, 1)
    with pytest.raises(QuantumCircuitError):
        sv.probability_of("0")
    with pytest.raises(QuantumCircuitError):
        sv.expectation_pauli("QQ")
    with pytest.raises(QuantumCircuitError, match="not unitary"):
        sv.apply_unitary(np.array([[1.0, 0.0], [0.0, 2.0]]), 0)
    with pytest.raises(QuantumCircuitError, match="unique"):
        Statevector(3).apply_unitary(np.eye(2), 2, controls=(0, 0))
    with pytest.raises(QuantumCircuitError, match="integer"):
        Statevector(2).apply_unitary(np.eye(2), 1, controls=([0],))
    with pytest.raises(QuantumCircuitError, match="positive integer"):
        Statevector(1).sample_counts(True)
    with pytest.raises(QuantumCircuitError, match="positive integer"):
        Statevector(1).sample_counts(1.5)  # type: ignore[arg-type]
    with pytest.raises(QuantumCircuitError, match="positive integer"):
        Statevector(True)  # type: ignore[arg-type]


def test_fidelity_rejects_mismatched_or_unnormalized_states():
    state = Statevector(1)
    with pytest.raises(QuantumCircuitError, match="matching state shapes"):
        state.fidelity(Statevector(2))
    with pytest.raises(QuantumCircuitError, match="normalized"):
        state.fidelity(np.array([1.0, 1.0], dtype=np.complex128))
    with pytest.raises(QuantumCircuitError, match="finite"):
        state.fidelity(np.array([np.nan, 0.0], dtype=np.complex128))


# ── Skill facade (the causal wiring into cognition) ─────────────


async def test_quantum_lab_skill_actions():
    from core.skills.quantum_lab import QuantumLabSkill

    skill = QuantumLabSkill()
    bell = await skill.execute({"action": "bell", "seed": 1, "shots": 200}, {})
    assert bell["ok"] and set(bell["counts"]) <= {"00", "11"}
    assert bell["entropy_mode"] == "seeded_prng"

    grover = await skill.execute({"action": "grover", "num_qubits": 5, "marked": 17, "seed": 1}, {})
    assert grover["ok"] and grover["matches_theory"]

    tp = await skill.execute(
        {"action": "teleport", "alpha_real": 0.6, "beta_real": 0.8, "seed": 2}, {}
    )
    assert tp["ok"] and tp["fidelity"] == pytest.approx(1.0, abs=1e-10)

    qft = await skill.execute({"action": "qft_verify", "num_qubits": 3}, {})
    assert qft["ok"] and qft["verified"]

    circuit = await skill.execute(
        {
            "action": "circuit",
            "num_qubits": 3,
            "seed": 3,
            "gates": [["h", 0], ["cx", 0, 1], ["ccx", 0, 1, 2]],
        },
        {},
    )
    assert circuit["ok"] and circuit["norm_preserved"]
    assert set(circuit["counts"]) <= {"000", "111"}


async def test_quantum_lab_reports_actual_entropy_source(monkeypatch):
    from core.skills import quantum_lab

    class _FallbackBridge:
        def __init__(self):
            self.fallback_reads = 0

        def get_stats(self):
            return {
                "quantum_reads": 0,
                "fallback_reads": self.fallback_reads,
            }

        def get_quantum_float(self):
            self.fallback_reads += 1
            return 0.25

    bridge = _FallbackBridge()
    monkeypatch.setattr(
        quantum_lab,
        "_entropy_source",
        lambda: quantum_lab._EntropyAudit(bridge),
    )
    skill = quantum_lab.QuantumLabSkill()

    sampled = await skill.execute({"action": "bell", "shots": 64}, {})
    deterministic = await skill.execute({"action": "grover", "num_qubits": 3}, {})

    assert sampled["entropy_mode"] == "os_entropy_fallback"
    assert sampled["entropy_provenance"]["bridge_draws"] == 1
    assert sampled["entropy_provenance"]["fallback_reads"] == 1
    assert deterministic["entropy_mode"] == "not_used"
    assert deterministic["entropy_provenance"]["bridge_draws"] == 0

    monkeypatch.setattr(quantum_lab, "_entropy_source", lambda: None)
    unavailable_bridge = await skill.execute(
        {"action": "qft_verify", "num_qubits": 2},
        {},
    )
    assert unavailable_bridge["entropy_mode"] == "not_used"


async def test_quantum_lab_entropy_failures_are_truthfully_attributed(monkeypatch):
    from core.skills import quantum_lab

    class _InvalidBridge:
        def get_stats(self):
            raise RuntimeError("statistics offline")

        def get_quantum_float(self):
            return float("nan")

    monkeypatch.setattr(
        quantum_lab,
        "_entropy_source",
        lambda: quantum_lab._EntropyAudit(_InvalidBridge()),
    )

    result = await quantum_lab.QuantumLabSkill().execute(
        {"action": "bell", "shots": 64},
        {},
    )

    assert result["ok"] is True
    assert result["entropy_mode"] == "entropy_bridge_failed_to_prng"
    assert result["sampling_rng"] == "os_seeded_prng_fallback"
    assert result["entropy_provenance"]["stats_available"] is False
    assert "statistics offline" in result["entropy_provenance"]["provenance_error"]


async def test_quantum_lab_invalid_inputs_fail_closed():
    from core.skills.quantum_lab import QuantumLabSkill

    skill = QuantumLabSkill()
    malformed = await skill.execute({"action": "bell", "shots": "many"}, {})
    nonfinite = await skill.execute(
        {"action": "teleport", "alpha_real": "nan", "beta_real": 1.0, "seed": 1},
        {},
    )
    duplicate_controls = await skill.execute(
        {
            "action": "circuit",
            "num_qubits": 3,
            "seed": 1,
            "gates": [["ccx", 0, 0, 2]],
        },
        {},
    )
    silently_clamped = await skill.execute(
        {"action": "qft_verify", "num_qubits": 12},
        {},
    )
    unknown_parameter = await skill.execute(
        {"action": "bell", "undocumented": True},
        {},
    )

    assert malformed["ok"] is False
    assert nonfinite["ok"] is False
    assert duplicate_controls["ok"] is False
    assert silently_clamped["ok"] is False
    assert "supports 1..8 qubits" in silently_clamped["error"]
    assert unknown_parameter["ok"] is False


def test_quantum_lab_skill_matches_quantum_goals():
    from core.skills.quantum_lab import QuantumLabSkill

    skill = QuantumLabSkill()
    assert skill.match({"objective": "simulate a quantum teleportation protocol"})
    assert skill.match({"objective": "show me Grover search"})
    assert not skill.match({"objective": "what time is it"})


def test_quantum_lab_is_discovered_by_the_skill_catalog():
    from core.skills.discovery import build_skill_catalog

    catalog = build_skill_catalog(try_rust=False)
    names = {declaration.name for declaration in catalog.accepted}
    assert "quantum_lab" in names
    excluded = {declaration.name for declaration in catalog.excluded}
    assert "quantum_lab" not in excluded


async def test_quantum_lab_executes_through_canonical_capability_engine(monkeypatch):
    from core.capability_engine import CapabilityEngine
    from core.runtime import CoreRuntime

    def _pre_runtime_constitution(*_args, **_kwargs):
        raise RuntimeError("constitutional runtime intentionally absent in isolated proof")

    monkeypatch.setattr(
        "core.constitution.get_constitutional_core",
        _pre_runtime_constitution,
    )
    monkeypatch.setattr(
        CoreRuntime,
        "get_sync",
        classmethod(lambda _cls: (_ for _ in ()).throw(RuntimeError("proof runtime absent"))),
    )
    monkeypatch.setattr(
        "core.capability_engine.ServiceContainer.has",
        staticmethod(lambda *_args, **_kwargs: False),
    )
    monkeypatch.setattr(
        "core.capability_engine.resolve_metabolic_monitor", lambda default=None: None
    )
    monkeypatch.setattr("core.capability_engine.resolve_edi", lambda default=None: None)

    engine = CapabilityEngine()
    metadata = engine.skills["quantum_lab"]
    schema = metadata.schema_def

    assert schema["properties"]["shots"]["type"] == "integer"
    assert schema["properties"]["gates"]["type"] == "array"
    assert not schema.get("required")

    result = await engine.execute(
        "quantum_lab",
        {"action": "bell", "seed": 11, "shots": 128},
        {"origin": "test", "objective": "simulate a Bell state"},
    )

    assert result["ok"] is True
    assert result["skill"] == "quantum_lab"
    assert result["retries"] == 0
    assert result["entropy_mode"] == "seeded_prng"
    assert set(result["counts"]) <= {"00", "11"}
    assert engine.instances["quantum_lab"].__class__.__name__ == "QuantumLabSkill"


def test_a_failing_entropy_source_is_counted_rather_than_silently_replaced():
    """A collapse meant to use real entropy that quietly became reproducible.

    The seeded generator is the right fallback. What was missing is any way to
    tell afterwards that it was used, so an experiment reported as
    entropy-driven could have been deterministic from the first draw.
    """
    from core.quantum.statevector import Statevector as Simulator

    def broken() -> float:
        raise RuntimeError("the entropy bridge is down")

    sim = Simulator(1, seed=3, entropy_source=broken)
    assert sim.entropy_fell_back == 0
    sim._random_unit()
    sim._random_unit()
    assert sim.entropy_fell_back == 2

    out_of_range = Simulator(1, seed=3, entropy_source=lambda: 1.5)
    out_of_range._random_unit()
    assert out_of_range.entropy_fell_back == 1

    healthy = Simulator(1, seed=3, entropy_source=lambda: 0.25)
    assert healthy._random_unit() == 0.25
    assert healthy.entropy_fell_back == 0
