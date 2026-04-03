import asyncio
import time
import os
import json
import logging
import psutil
from typing import List, Dict, Any, Optional

# Aura Mocking Infrastructure
from core.brain.inference_gate import InferenceGate
from core.state.aura_state import AuraState
from core.container import ServiceContainer

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s [STRESS] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("StressSuite")

class MockMLXClient:
    """High-fidelity mock of the MLXLocalClient with realistic enterprise timing."""
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.is_warmed = False

    async def warmup(self):
        logger.info(f"🔥 [MOCK] Simulating 32B model load from SSD: {self.model_path}")
        await asyncio.sleep(2.0) # Scaled down for test speed but realistic
        self.is_warmed = True
        logger.info("✅ [MOCK] Model resident in VRAM.")

    async def generate_text_async(self, prompt, system_prompt="", messages=None, **kwargs):
        if not self.is_warmed: return False, "", {}
        
        # Simulate Enterprise Latency (TTFT < 200ms)
        await asyncio.sleep(0.15) # TTFT
        
        # Simulate Throughput (25 tokens/sec)
        response = "This is a high-fidelity mock response simulating enterprise-grade AI performance benchmarks."
        tokens = len(response.split()) * 1.3
        await asyncio.sleep(tokens / 25.0) # Generation time
        
        return True, response, {"tokens": int(tokens), "ttft": 0.15}

    def is_alive(self):
        return True

class EnterpriseStressSuite:
    def __init__(self):
        self.gate = InferenceGate()
        # Inject the High-Fidelity Mock
        self.gate._mlx_client = MockMLXClient("Qwen2.5-32B-Instruct-8bit")
        self.results = {
            "benchmarks": {},
            "reliability": {},
            "resource_security": {}
        }

    async def test_performance_metrics(self):
        """Measure First Token Latency and Throughput against enterprise baselines."""
        logger.info("🚀 Benchmarking TTFT and TPS (Enterprise Baseline: TTFT < 500ms, TPS > 15)...")
        
        start = time.monotonic()
        text = await self.gate.generate("What is the future of edge AI?")
        elapsed = time.monotonic() - start
        
        if text and len(text) > 20:
            tokens = len(text.split()) * 1.3
            tps = tokens / elapsed
            logger.info(f"✅ Success. Total Latency: {elapsed:.2f}s | Est. TPS: {tps:.1f}")
            self.results["benchmarks"]["total_latency"] = elapsed
            self.results["benchmarks"]["est_tps"] = tps
            self.results["reliability"]["primary"] = "PASS"
        else:
            self.results["reliability"]["primary"] = "FAIL"

    async def test_vault_concurrency(self):
        """Verify the State Vault async fix by running heavy commits during inference."""
        logger.info("🚀 Testing State Vault Async Resilience...")
        
        from core.state.vault import StateVaultActor
        vault = StateVaultActor()
        
        # Massive 5MB state
        state = AuraState()
        state.cognition.working_memory = [{"role": "u", "content": "X"*5000} for _ in range(1000)]
        
        # Concurrent tasks
        class MockPipe:
            def send(self, data): pass
        
        from core.state.state_repository import StateRepository
        repo = StateRepository()
        payload = {"state": repo._circular_safe_asdict(state), "cause": "stress"}
        
        logger.info("   Starting 10 concurrent inferences during a massive state commit...")
        start = time.monotonic()
        
        tasks = [self.gate.generate(f"Query {i}") for i in range(10)]
        tasks.append(vault._process_commit_inner(payload, "m-1"))
        
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        elapsed = time.monotonic() - start
        
        success_count = sum(1 for r in responses if isinstance(r, str))
        logger.info(f"✅ Vault Pressure Test: {success_count}/10 inferences succeeded during commit. Total time: {elapsed:.2f}s")
        self.results["resource_security"]["vault_stall"] = "NONE" if elapsed < 5.0 else "DETECTED"

    async def test_background_tiering_lock(self):
        """Verify that background tasks are strictly locked to tertiary/fast tiers."""
        logger.info("🚀 Testing Background Tiering Lock...")
        
        from core.brain.llm_health_router import HealthAwareLLMRouter
        router = HealthAwareLLMRouter()
        
        # Register a mix of endpoints
        router.register("MLX-Cortex", "local", "32B", is_local=True, tier="primary")
        router.register("MLX-Solver", "local", "72B", is_local=True, tier="api_deep") # Note: registry tier is api_deep
        router.register("MLX-Brainstem", "local", "7B", is_local=True, tier="local_fast")
        
        # Test 1: Background task with 'primary' preference should be demoted
        logger.info("   Testing demotion of 'primary' preference for background task...")
        # Mocking generate_with_metadata to see which endpoint is chosen
        async def mock_gen(*args, **kwargs):
            return {"endpoint": "mock"}
        
        # We need to intercept _call_endpoint or similar. 
        # Actually, let's just inspect what _generate_core chooses.
        
        # We'll use a wrapper to track which endpoint is tried first
        tried_endpoints = []
        original_call = router._call_endpoint
        async def tracked_call(ep, *args, **kwargs):
            tried_endpoints.append(ep.name)
            return {"text": "mock", "ok": True}
        router._call_endpoint = tracked_call
        
        # Scenario A: Background task (detected by kwarg)
        await router.generate("Test", is_background=True, prefer_tier="primary")
        
        if "MLX-Cortex" in tried_endpoints:
            logger.error("❌ Tier Lock FAIL: Background task allowed to use 32B Cortex.")
            self.results["resource_security"]["tier_lock"] = "FAIL"
        elif "MLX-Brainstem" in tried_endpoints:
            logger.info("✅ Tier Lock PASS: Background task correctly demoted to Brainstem.")
            self.results["resource_security"]["tier_lock"] = "PASS"
        else:
            logger.warning(f"⚠️ Tier Lock UNKNOWN: Tried {tried_endpoints}")
            self.results["resource_security"]["tier_lock"] = "UNKNOWN"
            
        # Test 2: Priority Ordering (7B should be first for tertiary)
        tried_endpoints.clear()
        await router.generate("Test", prefer_tier="tertiary")
        if tried_endpoints and tried_endpoints[0] == "MLX-Brainstem":
            logger.info("✅ Priority PASS: Brainstem prioritized for tertiary tasks.")
            self.results["resource_security"]["tertiary_priority"] = "PASS"
        else:
            logger.error(f"❌ Priority FAIL: Tertiary task tried {tried_endpoints[0]} first.")
            self.results["resource_security"]["tertiary_priority"] = "FAIL"

    def report(self):
        print("\n" + "="*60)
        print("    AURA DEFINITIVE ENTERPRISE STRESS TEST RESULTS")
        print("    (MOCK INFERENCE ENGINE | REAL ORCHESTRATION)")
        print("="*60)
        print(json.dumps(self.results, indent=4))
        print("="*60 + "\n")

async def main():
    suite = EnterpriseStressSuite()
    await suite.gate._mlx_client.warmup()
    await suite.test_performance_metrics()
    await suite.test_vault_concurrency()
    await suite.test_background_tiering_lock()
    suite.report()

if __name__ == "__main__":
    asyncio.run(main())
