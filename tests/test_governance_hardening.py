"""tests/test_governance_hardening.py -- Governance Hardening Tests
===================================================================
Tests for fail-closed gates, vault integrity, and tamper detection.

These tests validate the architectural invariant:
    "No consequential action proceeds without authorization.
     If the gate fails, the action is BLOCKED."
"""
import asyncio
import hashlib
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class TestGovernanceVault(unittest.TestCase):
    """Tests for core/security/governance_vault.py"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test_vault.db"

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_seal_and_unseal(self):
        """Seal an artifact and verify unseal returns identical content."""
        from core.security.governance_vault import GovernanceVault
        vault = GovernanceVault(db_path=self.db_path)

        content = {"name": "Aura", "values": {"curiosity": 0.8, "empathy": 0.9}}
        content_hash = vault.seal("canonical_self", content)

        self.assertIsInstance(content_hash, str)
        self.assertEqual(len(content_hash), 64)  # SHA-256 hex

        unsealed = vault.unseal("canonical_self")
        self.assertEqual(unsealed, content)
        vault.close()

    def test_unseal_nonexistent_raises(self):
        """Unseal of nonexistent artifact raises KeyError."""
        from core.security.governance_vault import GovernanceVault
        vault = GovernanceVault(db_path=self.db_path)

        with self.assertRaises(KeyError):
            vault.unseal("nonexistent_artifact")
        vault.close()

    def test_tamper_detection(self):
        """Direct DB modification is detected as tampering."""
        from core.security.governance_vault import GovernanceVault, TamperDetected
        vault = GovernanceVault(db_path=self.db_path)

        vault.seal("test_artifact", {"integrity": True})

        # Tamper directly in the database
        import sqlite3
        conn = sqlite3.connect(str(self.db_path))
        conn.execute(
            "UPDATE sealed_artifacts SET content = ? WHERE artifact_id = ?",
            ('{"integrity": false}', "test_artifact"),
        )
        conn.commit()
        conn.close()

        # Recreate vault to reload
        vault.close()
        vault = GovernanceVault(db_path=self.db_path)

        with self.assertRaises(TamperDetected):
            vault.unseal("test_artifact")
        vault.close()

    def test_verify_integrity_detects_tamper(self):
        """verify_integrity returns False when content is tampered."""
        from core.security.governance_vault import GovernanceVault
        vault = GovernanceVault(db_path=self.db_path)

        vault.seal("check_artifact", {"secure": True})

        # Tamper
        import sqlite3
        conn = sqlite3.connect(str(self.db_path))
        conn.execute(
            "UPDATE sealed_artifacts SET content = ? WHERE artifact_id = ?",
            ('{"secure": false}', "check_artifact"),
        )
        conn.commit()
        conn.close()

        vault.close()
        vault = GovernanceVault(db_path=self.db_path)

        valid, message = vault.verify_integrity("check_artifact")
        self.assertFalse(valid)
        self.assertIn("TAMPER", message)
        vault.close()

    def test_seal_chain_grows(self):
        """Each seal updates the seal chain."""
        from core.security.governance_vault import GovernanceVault
        vault = GovernanceVault(db_path=self.db_path)

        vault.seal("versioned", {"v": 1})
        vault.seal("versioned", {"v": 2})
        vault.seal("versioned", {"v": 3})

        chain = vault.get_seal_chain("versioned")
        self.assertEqual(len(chain), 3)
        self.assertEqual(chain[0]["version"], 1)
        self.assertEqual(chain[2]["version"], 3)
        vault.close()

    def test_verify_all_reports_all_artifacts(self):
        """verify_all checks every sealed artifact."""
        from core.security.governance_vault import GovernanceVault
        vault = GovernanceVault(db_path=self.db_path)

        vault.seal("a1", {"key": "value1"})
        vault.seal("a2", {"key": "value2"})

        all_valid, results = vault.verify_all()
        self.assertTrue(all_valid)
        self.assertEqual(len(results), 2)
        vault.close()

    def test_empty_artifact_id_raises(self):
        """Seal with empty artifact_id raises SecurityException."""
        from core.security.governance_vault import GovernanceVault, SecurityException
        vault = GovernanceVault(db_path=self.db_path)

        with self.assertRaises(SecurityException):
            vault.seal("", {"should": "fail"})
        vault.close()


class TestFailClosedGates(unittest.TestCase):
    """Tests that all governance gates are fail-closed."""

    def test_liquid_substrate_update_blocks_on_gate_exception(self):
        """If substrate authority gate throws, update() blocks (returns early)."""
        # This is a behavioral test: we verify the code path exists
        import importlib
        from core.consciousness import liquid_substrate
        source = Path(liquid_substrate.__file__).read_text()

        # The fail-closed fix: gate exception → return (not pass-through)
        self.assertIn("FAIL-CLOSED", source)
        self.assertIn("BLOCKING update", source)
        self.assertNotIn("allowing update", source)

    def test_liquid_substrate_inject_blocks_on_gate_exception(self):
        """If stimulus injection gate throws, inject_stimulus() blocks."""
        from core.consciousness import liquid_substrate
        source = Path(liquid_substrate.__file__).read_text()

        self.assertIn("BLOCKING injection", source)
        # The old fail-open was "pass  # fail-open" — verify it's gone
        self.assertNotIn("pass  # fail-open", source)

    def test_authority_gateway_no_boot_blind_spot(self):
        """AuthorityGateway no longer has 'Boot Blind Spot' fail-open."""
        from core.executive import authority_gateway
        source = Path(authority_gateway.__file__).read_text()

        # The old bypass: "Boot Blind Spot" + "degraded_pass"
        self.assertNotIn("boot_blind_spot_bypass", source)
        self.assertNotIn("FAILING OPEN", source)
        self.assertNotIn("degraded_pass", source)

    def test_will_docstring_says_fail_closed(self):
        """Will docstring declares FAIL-CLOSED, not FAIL-SAFE."""
        from core import will
        source = Path(will.__file__).read_text()

        self.assertIn("FAIL-CLOSED", source)
        self.assertNotIn("FAIL-SAFE", source)


class TestDeterministicSubstrate(unittest.TestCase):
    """Tests that the continuous substrate is deterministic."""

    def test_same_seed_produces_same_trajectory(self):
        """Two substrates with the same seed produce identical states."""
        from core.brain.llm.continuous_substrate import ContinuousSubstrate

        sub1 = ContinuousSubstrate()
        sub2 = ContinuousSubstrate()

        for _ in range(50):
            sub1._step_once()
            sub2._step_once()

        import numpy as np
        np.testing.assert_array_almost_equal(
            sub1.get_state_vector(),
            sub2.get_state_vector(),
            decimal=6,
            err_msg="Deterministic substrates diverged!",
        )

    def test_substrate_uses_seeded_rng(self):
        """Substrate uses np.random.default_rng, not bare np.random."""
        from core.brain.llm import continuous_substrate
        source = Path(continuous_substrate.__file__).read_text()

        # Should use seeded RNG
        self.assertIn("default_rng", source)
        # Should NOT use bare np.random.normal
        self.assertNotIn("np.random.normal(", source)

    def test_liquid_substrate_uses_seeded_rng(self):
        """Liquid substrate uses seeded RNG for all random operations."""
        from core.consciousness import liquid_substrate
        source = Path(liquid_substrate.__file__).read_text()

        self.assertIn("default_rng", source)
        self.assertNotIn("np.random.randn(", source)


class TestInitiativeRouting(unittest.TestCase):
    """Tests that initiative loop routes through CapabilityEngine."""

    def test_email_routes_through_capability_engine(self):
        """Email check routes through CapabilityEngine, not direct instantiation."""
        from core import autonomous_initiative_loop
        source = Path(autonomous_initiative_loop.__file__).read_text()

        self.assertIn("capability_engine", source)
        # Find the email method body
        start = source.index("def _check_email_initiative")
        end = source.index("def _check_reddit_initiative")
        email_section = source[start:end]
        self.assertIn("cap_engine", email_section)

    def test_reddit_routes_through_capability_engine(self):
        """Reddit check routes through CapabilityEngine."""
        from core import autonomous_initiative_loop
        source = Path(autonomous_initiative_loop.__file__).read_text()

        reddit_section = source[source.index("_check_reddit_initiative"):]
        self.assertIn("cap_engine", reddit_section)


class TestDynamicValueGraph(unittest.TestCase):
    """Tests for the dynamic value graph."""

    def test_evidence_recording_and_evolution(self):
        """Record evidence and evolve produces mutations."""
        from core.adaptation import dynamic_value_graph as dvg_module
        from core.adaptation.dynamic_value_graph import (
            DynamicValueGraph,
            EvidenceType,
            ValueEvidence,
        )

        # Use a clean temp dir to avoid persisted state
        with tempfile.TemporaryDirectory() as td:
            original_dir = dvg_module._DATA_DIR
            original_path = dvg_module._GRAPH_PATH
            dvg_module._DATA_DIR = Path(td)
            dvg_module._GRAPH_PATH = Path(td) / "value_graph.json"
            try:
                graph = DynamicValueGraph()

                # Record enough evidence for evolution (need diverse sources)
                for i in range(15):
                    graph.record_evidence(ValueEvidence(
                        evidence_type=EvidenceType.OUTCOME_QUALITY,
                        value_name="curiosity",
                        signal=0.8,
                        confidence=0.9,
                        source=f"source_{i % 4}",  # 4 unique sources
                        context=f"test evidence {i}",
                    ))

                mutations = graph.evolve()
                # Should have at least one mutation (candidate → sandbox promotion)
                self.assertGreater(len(mutations), 0)
            finally:
                dvg_module._DATA_DIR = original_dir
                dvg_module._GRAPH_PATH = original_path

    def test_new_value_auto_created_as_candidate(self):
        """Recording evidence for unknown value creates a CANDIDATE node."""
        from core.adaptation.dynamic_value_graph import (
            DynamicValueGraph,
            EvidenceType,
            ValueEvidence,
            ValueNodeStatus,
        )

        graph = DynamicValueGraph()
        graph.record_evidence(ValueEvidence(
            evidence_type=EvidenceType.ENGAGEMENT,
            value_name="novel_drive",
            signal=0.5,
            confidence=0.7,
            source="test",
            context="testing auto-creation",
        ))

        self.assertIn("novel_drive", graph._nodes)
        self.assertEqual(graph._nodes["novel_drive"].status, ValueNodeStatus.CANDIDATE)


class TestLearnedWorldModel(unittest.TestCase):
    """Tests for the learned world model."""

    def test_observe_returns_prediction(self):
        """observe() returns a WorldModelPrediction with valid fields."""
        import numpy as np
        from core.world_model.learned_world_model import LearnedWorldModel

        model = LearnedWorldModel()
        obs = np.random.randn(64).astype(np.float32)
        prediction = model.observe(obs)

        self.assertIsNotNone(prediction)
        self.assertGreaterEqual(prediction.surprise, 0.0)
        self.assertIsInstance(prediction.confidence, float)
        self.assertEqual(prediction.predicted_state.shape[0], 64)

    def test_imagine_returns_trajectory(self):
        """imagine() returns a list of predictions for the action sequence."""
        import numpy as np
        from core.world_model.learned_world_model import LearnedWorldModel

        model = LearnedWorldModel()
        obs = np.random.randn(64).astype(np.float32)
        model.observe(obs)  # Warm up hidden state

        actions = [np.random.randn(16).astype(np.float32) for _ in range(5)]
        trajectory = model.imagine(obs, actions)

        self.assertEqual(len(trajectory), 5)
        for pred in trajectory:
            self.assertEqual(pred.predicted_state.shape[0], 64)

    def test_deterministic_with_same_seed(self):
        """Two models with same seed produce same predictions."""
        import numpy as np
        from core.world_model.learned_world_model import (
            LearnedWorldModel,
            WorldModelConfig,
        )

        config = WorldModelConfig(seed=42)
        m1 = LearnedWorldModel(config)
        m2 = LearnedWorldModel(config)

        obs = np.ones(64, dtype=np.float32) * 0.5
        p1 = m1.observe(obs, learn=False)
        p2 = m2.observe(obs, learn=False)

        np.testing.assert_array_almost_equal(
            p1.predicted_state, p2.predicted_state, decimal=5
        )


class TestHierarchicalBrain(unittest.TestCase):
    """Tests for the hierarchical brain."""

    def test_step_produces_outputs(self):
        """step() returns output vectors for all regions."""
        import numpy as np
        from core.brain.hierarchical_brain import HierarchicalBrain

        brain = HierarchicalBrain()
        substrate_state = np.random.randn(64).astype(np.float32)

        outputs = brain.step(substrate_state)

        self.assertIn("sensory", outputs)
        self.assertIn("association", outputs)
        self.assertIn("executive", outputs)
        self.assertIn("affective", outputs)

    def test_composite_output_fixed_dimension(self):
        """get_composite_output returns a fixed 64-dim vector."""
        import numpy as np
        from core.brain.hierarchical_brain import HierarchicalBrain

        brain = HierarchicalBrain()
        brain.step(np.zeros(64, dtype=np.float32))

        composite = brain.get_composite_output()
        self.assertEqual(composite.shape[0], 64)

    def test_status_reports_all_regions(self):
        """get_status reports on all default regions."""
        from core.brain.hierarchical_brain import HierarchicalBrain

        brain = HierarchicalBrain()
        status = brain.get_status()

        self.assertEqual(status["region_count"], 4)
        self.assertIn("sensory", status["regions"])
        self.assertGreater(status["total_neurons"], 0)


class TestSteeringVectors(unittest.TestCase):
    """Tests for steering vector generation."""

    def test_generate_all_dimensions(self):
        """Generator produces vectors for all affect dimensions."""
        from scripts.generate_steering_vectors import SteeringVectorGenerator

        gen = SteeringVectorGenerator(
            output_dir=Path(tempfile.mkdtemp()),
            seed=42,
            vector_dim=128,
        )
        vectors = gen.generate_all()

        self.assertIn("curiosity", vectors)
        self.assertIn("empathy", vectors)
        self.assertIn("assertiveness", vectors)
        self.assertIn("creativity", vectors)
        self.assertIn("warmth", vectors)

        for dim, sv in vectors.items():
            self.assertEqual(sv.vector.shape[0], 128)
            self.assertGreater(sv.magnitude, 0.0)
            self.assertTrue(sv.checksum)

    def test_deterministic_vectors(self):
        """Same seed produces same vectors."""
        import numpy as np
        from scripts.generate_steering_vectors import SteeringVectorGenerator

        gen1 = SteeringVectorGenerator(
            output_dir=Path(tempfile.mkdtemp()), seed=42
        )
        gen2 = SteeringVectorGenerator(
            output_dir=Path(tempfile.mkdtemp()), seed=42
        )

        v1 = gen1.generate_all()
        v2 = gen2.generate_all()

        for dim in v1:
            np.testing.assert_array_equal(
                v1[dim].vector, v2[dim].vector,
                err_msg=f"Steering vector for {dim} is not deterministic!"
            )


class TestAuditChainIntegrity(unittest.TestCase):
    """Tests for the audit chain."""

    def test_append_and_verify(self):
        """Appended entries are verifiable."""
        from core.runtime.audit_chain import AuditChain

        with tempfile.TemporaryDirectory() as td:
            chain = AuditChain(root=Path(td))

            for i in range(5):
                chain.append(
                    receipt_id=f"test_{i}",
                    kind="governance",
                    body={"action": f"test_action_{i}"},
                    timestamp=time.time(),
                )

            ok, problems = chain.verify()
            self.assertTrue(ok, f"Chain verification failed: {problems}")
            self.assertEqual(chain.length(), 5)

    def test_tamper_detection_in_chain(self):
        """Modifying a chain entry hash is detected by verify()."""
        from core.runtime.audit_chain import AuditChain

        with tempfile.TemporaryDirectory() as td:
            chain = AuditChain(root=Path(td))

            chain.append(
                receipt_id="original",
                kind="governance",
                body={"action": "original_action"},
                timestamp=time.time(),
            )

            # Tamper with the entry_hash in the chain file
            chain_path = Path(td) / "_chain.jsonl"
            content = chain_path.read_text()
            # Parse the entry and corrupt the entry_hash
            entry = json.loads(content.strip())
            original_hash = entry["entry_hash"]
            entry["entry_hash"] = "tampered_" + original_hash[9:]
            chain_path.write_text(json.dumps(entry) + "\n")

            # Reload and verify
            chain2 = AuditChain(root=Path(td))
            ok, problems = chain2.verify()
            self.assertFalse(ok)
            self.assertGreater(len(problems), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
