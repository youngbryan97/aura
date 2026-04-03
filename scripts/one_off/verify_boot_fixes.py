
import asyncio
import logging
import sys
import unittest
from unittest.mock import MagicMock, AsyncMock

# Add project root to path
sys.path.insert(0, ".")

from core.container import ServiceContainer
from core.orchestrator.main import RobustOrchestrator
from core.consciousness.affective_steering import AffectiveSteeringEngine, SteeringVectorLibrary

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Test.BootVerify")

class TestAuraBoot(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        # Clear container for clean test
        ServiceContainer._services = {}
        ServiceContainer._instances = {}

    async def test_qualia_registration_race(self):
        """Verifies that QualiaSynthesizer can be registered without subsystem_audit error."""
        from core.service_registration import register_all_services
        register_all_services()
        from core.orchestrator.mixins.boot.boot_cognitive import BootCognitiveMixin
        
        # Create a mock orchestrator that inherits from the mixin
        class MockOrchestrator(BootCognitiveMixin):
            def __init__(self):
                self.hooks = MagicMock()
                self.cognitive_engine = MagicMock()
                self.user_identity = {"name": "test_user"}
        
        orch = MockOrchestrator()
        
        # Simulate _init_cognitive_architecture
        # This should now work because we moved subsystem_audit registration inside it
        logger.info("Running _init_cognitive_architecture simulation...")
        await orch._init_cognitive_architecture()
        
        # Verify QualiaSynthesizer is registered
        qualia = ServiceContainer.get("qualia_synthesizer")
        self.assertIsNotNone(qualia)
        logger.info("✅ QualiaSynthesizer successfully registered.")
        
        # Verify subsystem_audit is registered
        audit = ServiceContainer.get("subsystem_audit")
        self.assertIsNotNone(audit)
        logger.info("✅ SubsystemAudit successfully registered.")

    async def test_steering_geometry_discovery(self):
        """Verifies that steering engine can discover geometry from diverse model structures."""
        engine = AffectiveSteeringEngine()
        
        # Mock model with model.layers (Qwen/Phi style)
        mock_model_v1 = MagicMock()
        mock_layer = MagicMock()
        mock_model_v1.layers = [mock_layer] * 32
        
        # Mock projection to give d_model
        # Use PropertyMock or just assign attributes directly
        mock_attn = MagicMock()
        mock_proj = MagicMock()
        mock_proj.weight.shape = (4096, 4096)
        mock_attn.q_proj = mock_proj
        mock_layer.attention = mock_attn
        mock_layer.self_attn = None
        mock_layer.attn = None
        mock_layer.mlp = None
        mock_layer.feed_forward = None
        mock_layer.ff = None
        
        n_layers, d_model = engine._discover_model_geometry(mock_model_v1)
        self.assertEqual(n_layers, 32)
        self.assertEqual(d_model, 4096)
        logger.info("✅ Geometry discovery verified for model.layers structure.")
        
        # Mock model with model.model.layers (Standard Llama style)
        mock_model_v2 = MagicMock()
        mock_model_v2.model.layers = [mock_layer] * 32
        del mock_model_v2.layers # ensure it doesn't find it at top level
        
        n_layers, d_model = engine._discover_model_geometry(mock_model_v2)
        self.assertEqual(n_layers, 32)
        self.assertEqual(d_model, 4096)
        logger.info("✅ Geometry discovery verified for model.model.layers structure.")

if __name__ == "__main__":
    unittest.main()
