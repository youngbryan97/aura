import asyncio
import sys
import os
from typing import Any

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Mock missing dependencies for testing logic
from unittest.mock import MagicMock

# Mock httpx
sys.modules["httpx"] = MagicMock()
# Mock core components that might fail on import due to env
sys.modules["core.event_bus"] = MagicMock()
sys.modules["core.config"] = MagicMock()
sys.modules["psutil"] = MagicMock()

async def test_router_compatibility():
    print("🔍 Testing HealthAwareLLMRouter compatibility...")
    try:
        from core.brain.llm_health_router import HealthAwareLLMRouter
        router = HealthAwareLLMRouter()
        
        # Mock client that returns a string
        class MockClient:
            async def generate(self, prompt, **kwargs):
                return "Mock Response"
        
        router.register("Mock", "internal", "mock-model", is_local=True, client=MockClient())
        
        print("  - Calling generate()...")
        res = await router.generate("test prompt")
        print(f"  - Result type: {type(res)}")
        assert isinstance(res, str), f"Expected str, got {type(res)}"
        assert res == "Mock Response"
        
        print("  - Calling generate_with_metadata()...")
        res_meta = await router.generate_with_metadata("test prompt")
        print(f"  - Result type: {type(res_meta)}")
        assert isinstance(res_meta, dict), f"Expected dict, got {type(res_meta)}"
        assert res_meta["text"] == "Mock Response"
        
        print("✅ Router compatibility verified.")
    except Exception as e:
        print(f"❌ Router compatibility failed: {e}")
        import traceback
        traceback.print_exc()

async def test_language_center_hardening():
    print("\n🔍 Testing LanguageCenter hardening...")
    try:
        from core.language_center import strip_aura_prefix
        
        print("  - Testing strip_aura_prefix with dict (should fail if not handled, but we hardened express)...")
        # Note: strip_aura_prefix itself still wants a string, but express now casts to str.
        
        from core.language_center import LanguageCenter
        from core.inner_monologue import ThoughtPacket
        
        class MockRouter:
            async def generate(self, *args, **kwargs):
                return {"text": "I am a dict"} # Simulation of old bug
        
        # Minimal mock container
        class Container:
            def get(self, name, default=None):
                if name == "llm_router": return MockRouter()
                return default
        
        lc = LanguageCenter()
        # Manually inject router
        lc._router = MockRouter()
        
        packet = ThoughtPacket(stance="test")
        
        print("  - Calling express() with a router that returns a dict...")
        # This used to crash re.sub in strip_aura_prefix.
        # Now it should cast to str(dict) and proceed.
        response = await lc.express(packet, "hello")
        print(f"  - Response: {response}")
        assert isinstance(response, str)
        assert "{'text':" in response or "I am a dict" in response
        
        print("✅ LanguageCenter hardening verified.")
    except Exception as e:
        print(f"❌ LanguageCenter hardening failed: {e}")
        import traceback
        traceback.print_exc()

async def test_synthesis_scrubbing():
    print("\n🔍 Testing Synthesis scrubbing for '. .'...")
    try:
        from core.synthesis import strip_meta_commentary
        
        test_cases = [
            (". .", ""),
            ("Hello . .", "Hello"),
            (". . .", ""),
            ("Wait . .. . done", "Wait  done"),
        ]
        
        for input_text, expected in test_cases:
            result = strip_meta_commentary(input_text)
            print(f"  - '{input_text}' -> '{result}'")
            assert result.strip() == expected.strip(), f"Expected '{expected}', got '{result}'"
            
        print("✅ Synthesis scrubbing verified.")
    except Exception as e:
        print(f"❌ Synthesis scrubbing failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_router_compatibility())
    asyncio.run(test_language_center_hardening())
    asyncio.run(test_synthesis_scrubbing())
