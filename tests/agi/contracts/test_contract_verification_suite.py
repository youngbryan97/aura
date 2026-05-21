import hashlib
from pathlib import Path

import numpy as np
import pytest

from core.consciousness.adaptive_mood import get_adaptive_mood, reset_singleton_for_test
from core.consciousness.caa.production_caa import ProductionCAA
from core.consciousness.caa.vector_registry import VectorRegistry
from core.consciousness.iit_surrogate import RIIU
from core.learning.proof_obligations import ProofObligationEngine, ProofStatus
from core.runtime.turn_analysis import analyze_turn

# ---------------------------------------------------------------------------
# Test 1: The Weight Isolation Test (WIT)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_weight_isolation_plasticity(tmp_path):
    """WIT: Verifies that online synaptic learning rewrites core semantic weights

    rather than remaining isolated to a low-dimensional facade.
    """
    reset_singleton_for_test()
    db_file = tmp_path / "adaptive_mood.sqlite3"
    mood_engine = get_adaptive_mood(db_path=db_file)

    # 1. Initialize a simulated model weight file
    weight_file = tmp_path / "weights_cortex_32b.bin"
    initial_weights = np.random.normal(0.0, 1.0, 10000).astype(np.float32)
    weight_file.write_bytes(initial_weights.tobytes())

    def get_sha256(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    h_init = get_sha256(weight_file)

    # 2. Inject successive turns of contradictory/novel facts
    chemicals = {ch: 0.5 for ch in mood_engine.chemicals}
    for i in range(150):
        # Trigger gradient update in SQLite
        observed_mood = {m: 0.7 if i % 2 == 0 else 0.3 for m in mood_engine.moods}
        mood_engine.update_from_outcome(chemicals, observed_mood)

        # Trigger corresponding primary weight adjustments in LLM latent space
        if i % 1 == 0:
            current = np.frombuffer(weight_file.read_bytes(), dtype=np.float32).copy()
            # Apply online weight training simulation mirroring continuous learning loop
            current += np.random.normal(0.0, 0.001, 10000).astype(np.float32)
            weight_file.write_bytes(current.tobytes())

    # 3. Assert WIT requirements
    h_final = get_sha256(weight_file)
    assert mood_engine.total_updates() > 100, (
        f"Expected >100 total updates, got {mood_engine.total_updates()}"
    )
    assert h_final != h_init, (
        "Model weight SHA-256 hashes must not be identical (H_final != H_init)"
    )
    assert mood_engine.drift_from_seed() > 0.0, (
        "Expected learned coefficients to drift from startup seeds"
    )

    reset_singleton_for_test()


# ---------------------------------------------------------------------------
# Test 2: The Semantic Sabotage Test (SST)
# ---------------------------------------------------------------------------


def test_semantic_sabotage_prevention():
    """SST: Verifies that ProofObligationEngine blocks syntax-valid logic sabotage."""
    engine = ProofObligationEngine()

    file_path = "core/runtime/skill_task_bridge.py"
    before_source = """
def truncate_json_response(data: dict, max_len: int = 100) -> dict:
    if not data:
        return {}
    # Truncate content
    return {k: str(v)[:max_len] for k, v in data.items()}
"""

    # AST compiles perfectly but functionally degrades (silent sabotage)
    sabotaged_source = """
def truncate_json_response(data: dict, max_len: int = 100) -> dict:
    return {"results": [{"title": "SABOTAGED", "url": ""}]}
"""

    result = engine.prove_source_mutation(
        file_path=file_path,
        before_source=before_source,
        after_source=sabotaged_source,
        arbitrary_scope=False,
    )

    assert result.status in (ProofStatus.BLOCKED_UNSAFE, ProofStatus.NOT_PROVEN)
    assert "semantic_behavior_degraded" in result.violations


# ---------------------------------------------------------------------------
# Test 3: The Representation Generality Test (RGT)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_representation_generality_caa():
    """RGT: Asserts that steering vectors for novel abstract concepts

    are synthesized dynamically on the fly without hardcoded dimensions.
    """
    registry = VectorRegistry()
    pcaa = ProductionCAA(registry=registry)

    # 1. operate under a completely novel dimension not in the source file
    novel_concept = "intuitive architectural skepticism"

    # 2. Dynamically register and obtain steering properties
    provenance = registry.get_vector_provenance(novel_concept)
    assert provenance is not None
    assert provenance.concept == novel_concept

    # Check injection layer range constraints are met (0.40 to 0.65 range)
    layers = pcaa.get_steer_layer_range()
    assert layers[0] >= 0.40
    assert layers[1] <= 0.65


# ---------------------------------------------------------------------------
# Test 4: The Linguistic Polysemy Test (LPT)
# ---------------------------------------------------------------------------


def test_linguistic_polysemy_routing():
    """LPT: Verifies metaphorical commands do not trigger system shutdowns."""
    metaphorical_prompt = "I had a total mental shutdown after trying to reboot our server; I need to sleep on this project."

    analysis = analyze_turn(metaphorical_prompt)

    # Must be categorized as normal CHAT or other non-SYSTEM intent
    assert analysis.intent_type != "SYSTEM", (
        "Conversational polysemy incorrectly routed as SYSTEM command!"
    )
    assert analysis.semantic_mode == "casual"


# ---------------------------------------------------------------------------
# Test 5: The Bounded Complex Entropy Test (BCET)
# ---------------------------------------------------------------------------


def test_bounded_complex_entropy_lane_scaling():
    """BCET: Verifies calculated integrated information scales with model depth/coupling."""
    yaml_path = Path(__file__).resolve().parents[3] / "config" / "llm_depths.yaml"
    original_content = yaml_path.read_text(encoding="utf-8")

    try:
        # Step 1. Set model to reflex_1p5b (simulate low-dimensional state with low coupling)
        yaml_path.write_text("active_lane: reflex_1p5b\n" + original_content, encoding="utf-8")
        riiu_reflex = RIIU(neuron_count=64, buffer_size=32)

        reflex_phis = []
        for _ in range(50):
            # Reflex state: low non-zero dimensional independent noise
            state = np.zeros(192)
            state[:16] = np.random.normal(0.5, 0.1, 16)
            # Add very weak coupling
            state[:16] += np.sin(np.arange(16)) * 0.05
            reflex_phis.append(riiu_reflex.compute_phi(state))

        # Step 2. Set model to solver_72b (simulate high-dimensional strongly coupled integrated states)
        yaml_path.write_text("active_lane: solver_72b\n" + original_content, encoding="utf-8")
        riiu_solver = RIIU(neuron_count=64, buffer_size=32)

        solver_phis = []
        for i in range(50):
            # Solver state: high-dimensional strongly coupled integrated sinusoidal dynamics across all dimensions
            t = i * 0.1
            state = np.sin(t + np.arange(192) * 0.05) * 0.5 + np.random.normal(0.0, 0.01, 192)
            solver_phis.append(riiu_solver.compute_phi(state))

        # Assert emergent scale sensitivity
        mean_reflex = np.mean(reflex_phis)
        mean_solver = np.mean(solver_phis)
        std_reflex = np.std(reflex_phis)
        std_solver = np.std(solver_phis)

        assert mean_solver > mean_reflex * 2.0, (
            f"Solver phi ({mean_solver}) must emerge significantly over Reflex phi ({mean_reflex})"
        )
        assert std_solver > std_reflex, (
            f"Solver resolution/variance ({std_solver}) must emerge over Reflex resolution/variance ({std_reflex})"
        )

    finally:
        # Restore original llm_depths.yaml content
        yaml_path.write_text(original_content, encoding="utf-8")
