################################################################################

"""tests/stress_test_a_plus.py
Comprehensive stress test for A+ Architectural Hardening.
"""
import asyncio
import threading
import time
import logging
import pytest
import os
import resource
from core.world_model.belief_graph import belief_graph
from core.memory.sqlite_storage import SQLiteMemory
from core.utils.resilience import AsyncCircuitBreaker
from core.skill_management.hephaestus import HephaestusEngine
from core.container import ServiceContainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("A+StressTest")

# 1. Thread-Safe BeliefGraph Stress Test
def test_belief_graph_concurrency():
    logger.info("Running BeliefGraph concurrency test...")
    results = []
    
    def worker(worker_id):
        for i in range(100):
            belief_graph.update_belief(
                source=f"agent_{worker_id}",
                relation="observes",
                target=f"entity_{i}",
                confidence_score=0.9
            )
            # Mixed read/write
            _ = belief_graph.get_beliefs()
            
    threads = [threading.Thread(target=worker, args=(j,)) for j in range(10)]
    for t in threads: t.start()
    for t in threads: t.join()
    
    beliefs = belief_graph.get_beliefs()
    logger.info(f"Final Belief Count: {len(beliefs)}")
    assert len(beliefs) >= 100
    # Clean up for next tests
    belief_graph.persist_path = "/tmp/test_beliefs.json"
    logger.info("✓ BeliefGraph concurrency test PASSED.")

# 2. Concurrent SQLite Stress Test
@pytest.mark.asyncio
async def test_sqlite_memory_concurrency():
    logger.info("Running SQLiteMemory concurrency test...")
    db = SQLiteMemory(storage_file="/tmp/stress_test.db")
    await db._get_conn() 
    await db._ensure_schema()
    
    async def worker(worker_id):
        for i in range(50):
            # FIXED: Pass a dictionary instead of kwargs
            await db.log_event_async({
                "event_type": "stress",
                "goal": f"Worker {worker_id}",
                "outcome": f"event {i}",
                "cost": 0.01
            })
            
    tasks = [worker(j) for j in range(10)]
    await asyncio.gather(*tasks)
    
    # Verify count
    async with db._lock: # SQLiteMemory uses _lock
        conn = await db._get_conn()
        cursor = await conn.execute("SELECT COUNT(*) FROM episodic")
        count = (await cursor.fetchone())[0]
        assert count >= 500

    
    logger.info("✓ SQLiteMemory concurrency test PASSED.")

# 3. Circuit Breaker Resilience Test
@pytest.mark.asyncio
async def test_circuit_breaker_resilience():
    logger.info("Running CircuitBreaker resilience test...")
    breaker = AsyncCircuitBreaker(name="TestBreaker", failure_threshold=3, recovery_timeout=0.1)
    
    async def failing_fn():
        raise ValueError("Simulated Failure")
        
    # Trip the breaker
    for i in range(3):
        try:
            await breaker.execute(failing_fn)
        except ValueError:
            logger.debug(f"Failure {i+1} recorded")
            
    from core.utils.resilience import CircuitState
    logger.info(f"Breaker State: {breaker.state}, Failures: {breaker.failures}")
    assert breaker.state == CircuitState.OPEN
    logger.info("  - Breaker correctly opened after 3 failures.")
    
    # Try while open
    from core.utils.resilience import CircuitBreakerOpenError
    with pytest.raises(CircuitBreakerOpenError):
        await breaker.execute(failing_fn)
    
    # Wait for recovery
    await asyncio.sleep(0.2)
    # The state only changes to HALF_OPEN when we ATTEMPT to call it and it probes
    # but the internal check _should_probe returns True.
    # Actually AsyncCircuitBreaker.execute calls _should_probe which transitions it.
    
    # Next call should probe
    async def success_fn(): return "OK"
    await breaker.execute(success_fn)
    # After one success in probe, it's still HALF_OPEN until success_threshold (default 3)
    assert breaker.state == CircuitState.HALF_OPEN
    logger.info("  - Breaker transitioned to HALF_OPEN during probe.")
    
    # Complete success threshold
    await breaker.execute(success_fn)
    await breaker.execute(success_fn)
    assert breaker.state == CircuitState.CLOSED
    logger.info("✓ CircuitBreaker resilience test PASSED.")

# 4. Sandbox Resource Limit Test
@pytest.mark.asyncio
async def test_sandbox_resource_limits():
    logger.info("Running Sandbox Resource Limit test...")
    from unittest.mock import patch, AsyncMock
    
    # 0. Clear and Setup mocks
    ServiceContainer.clear()
    
    class MockRegistry:
        async def discover_skills(self): pass
    
    class SimpleBrain:
        async def think(self, *args, **kwargs):
            return type('Res', (), {'content': 'def execute(params, context=None):\n    return {"ok": True}'})()

    ServiceContainer.register_instance("capability_engine", MockRegistry())
    ServiceContainer.register_instance("cognitive_engine", SimpleBrain())

    engine = HephaestusEngine()
    
    # 1. CPU Timeout Test
    infinite_loop_code = """
import time
def execute(params, context=None):
    while True: pass
"""
    async def mock_draft_loop(name, obj):
        return {"ok": True, "code": infinite_loop_code, "description": "Loop", "logic_description": "Loop"}
    
    with patch.object(HephaestusEngine, '_draft_logic', side_effect=mock_draft_loop):
        logger.info("  - Testing CPU timeout / Infinite loop protection...")
        result = await engine.synthesize_skill("infinite_loop", "test_loop")
        logger.info(f"Infinite loop result: {result}")
        assert result["ok"] is False
        # Check for timeout or any error that indicates sub-process failure
        assert any(x in result["error"].lower() for x in ["timeout", "loop", "failed"])
        logger.info("  - Infinite loop correctly blocked.")

    # 2. OOM Test
    oom_code = 'x = [0] * (512 * 1024 * 1024 // 4) # 512MB'
    async def mock_draft_oom(name, obj):
        return {"ok": True, "code": oom_code, "description": "OOM", "logic_description": "OOM"}
    
    with patch.object(HephaestusEngine, '_draft_logic', side_effect=mock_draft_oom):
        logger.info("  - Testing Memory/OOM protection...")
        result = await engine.synthesize_skill("oom_skill", "test_oom")
        logger.info(f"OOM result: {result}")
        assert result["ok"] is False
    
    logger.info("✓ Sandbox Resource Limit test PASSED.")


if __name__ == "__main__":
    # Manual run if not using pytest
    test_belief_graph_concurrency()
    asyncio.run(test_sqlite_memory_concurrency())
    asyncio.run(test_circuit_breaker_resilience())
    asyncio.run(test_sandbox_resource_limits())


##
