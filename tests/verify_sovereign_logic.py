import os
import unittest
from unittest.mock import MagicMock, patch
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

# Mock out heavy dependencies before imports to avoid model downloads/connection errors
mock_whisper = MagicMock()
mock_docker = MagicMock()
sys.modules["faster_whisper"] = mock_whisper
sys.modules["docker"] = mock_docker

class TestSovereignAura(unittest.TestCase):

    def test_config_firewall(self):
        """Verify that AuraConfig mirrors the explicit owner-autonomous posture."""
        from core.config import config
        self.assertEqual(os.environ.get("AURA_SECURITY_PROFILE"), "owner_autonomous")
        self.assertEqual(os.environ.get("AURA_INTERNAL_ONLY"), "0")
        self.assertEqual(os.environ.get("AURA_ALLOW_NETWORK_ACCESS"), "1")
        self.assertEqual(config.security.security_profile, "owner_autonomous")
        self.assertEqual(config.security.internal_only_mode, False)
        self.assertEqual(config.security.allow_network_access, True)

    def test_server_auth_logic(self):
        """Verify server auth logic in server.py (simulated)."""
        from interface.server import _check_auth
        
        # Scenario 1: No token set, allow_localhost_only = 0 -> Should fail
        with patch.dict(os.environ, {"AURA_API_TOKEN": "", "AURA_ALLOW_LOCALHOST_ONLY": "0"}):
            self.assertFalse(_check_auth("Bearer test"))
            
        # Scenario 2: Token set, matches -> Should pass
        with patch.dict(os.environ, {"AURA_API_TOKEN": "secret_key"}):
            self.assertTrue(_check_auth("Bearer secret_key"))
            self.assertFalse(_check_auth("Bearer wrong"))

    def test_local_llm_logic(self):
        """Verify the local brain construct prompt correctly."""
        from core.brain.local_llm import LocalBrain
        brain = LocalBrain(model_name="test-model")
        self.assertEqual(brain.model, "test-model")
        
    def test_sandbox_isolation_config(self):
        """Verify that SecureDockerSandbox forces network_disabled=True."""
        from core.skills.secure_sandbox import SecureDockerSandbox
        
        mock_client = MagicMock()
        mock_docker.from_env.return_value = mock_client
        
        sandbox = SecureDockerSandbox()
        # Test code execution call
        sandbox.execute_code("print('hello')", "/tmp")
        
        # Verify docker-py was called with network_disabled=True
        args, kwargs = mock_client.containers.run.call_args
        self.assertTrue(kwargs.get("network_disabled"))
        self.assertEqual(kwargs.get("mem_limit"), "256m")

if __name__ == "__main__":
    unittest.main()
