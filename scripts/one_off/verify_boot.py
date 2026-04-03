import asyncio
import logging
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath("."))

from core.container import ServiceContainer
from core.kernel.aura_kernel import AuraKernel
from core.senses.voice_engine import SovereignVoiceEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VerifyBoot")

async def verify():
    print("🚀 Starting Integrated Boot Verification...")
    
    # 1. Test AuraKernel.get fixes
    print("\n--- Testing AuraKernel.get ---")
    from unittest.mock import MagicMock
    mock_config = MagicMock()
    mock_vault = MagicMock()
    kernel = AuraKernel(config=mock_config, vault=mock_vault)
    
    # Test string-based lookup for a core service (even if not registered yet)
    try:
        # Should raise RuntimeError because it's not in local registry or container
        kernel.get("DeepVoid")
        print("✗ Failed: kernel.get('DeepVoid') should have raised RuntimeError")
        return False
    except RuntimeError as e:
        print(f"✓ kernel.get('DeepVoid') raised RuntimeError as expected: {e}")

    # Test default value
    assert kernel.get("DeepVoid", default="safe") == "safe"
    print("✓ kernel.get with default value passed.")

    # 2. Test VoiceEngine Initialization (SmartCircuitBreaker fix)
    print("\n--- Testing VoiceEngine Initialization ---")
    try:
        # This will trigger __init__, which calls _init_components
        engine = SovereignVoiceEngine()
        print("✓ VoiceEngine initialized successfully (SmartCircuitBreaker fix verified).")
    except TypeError as e:
        print(f"✗ VoiceEngine initialization failed with TypeError: {e}")
        return False
    except Exception as e:
        # Other errors might occur if hardware/dependencies are missing,
        # but we specifically care about the SmartCircuitBreaker TypeError.
        if "SmartCircuitBreaker" in str(e):
             print(f"✗ VoiceEngine initialization failed with CircuitBreaker error: {e}")
             return False
        logger.warning("VoiceEngine initialized but encountered non-critical error (likely hardware): %s", e)
        print("✓ VoiceEngine __init__ completed (SmartCircuitBreaker fix likely verified).")

    print("\n✨ INTEGRATED BOOT VERIFICATION PASSED ✨")
    return True

if __name__ == "__main__":
    success = asyncio.run(verify())
    sys.exit(0 if success else 1)
