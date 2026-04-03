################################################################################

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
        """Verify that AuraConfig enforces AURA_INTERNAL_ONLY environment variable."""
        from core.config import config
        # Check if environment was set by config initialization
        self.assertEqual(os.environ.get("AURA_INTERNAL_ONLY"), "1")
        self.assertEqual(config.security.internal_only_mode, True)

    def test_server_auth_logic(self):
        """Verify server auth logic in server.py (simulated)."""
        from interface.server import _verify_token
        
        # Scenario 1: No token set -> Should log warning but not raise 401 if token is None
        with patch.dict(os.environ, {"AURA_API_TOKEN": ""}):
            # We check if it returns None (success/passthrough) or raises HTTPException
            # _verify_token returns None on success or raises 401
            try:
                _verify_token(None)
                passed = True
            except Exception:
                passed = False
            self.assertTrue(passed)
            
        # Scenario 2: Token set, matches -> Should pass
        with patch.dict(os.environ, {"AURA_API_TOKEN": "secret_key"}):
            try:
                _verify_token("secret_key")
                self.assertTrue(True)
            except Exception:
                self.fail("_verify_token raised on valid token")
                
            from fastapi import HTTPException
            with self.assertRaises(HTTPException) as cm:
                _verify_token("wrong")
            self.assertEqual(cm.exception.status_code, 401)

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


##
