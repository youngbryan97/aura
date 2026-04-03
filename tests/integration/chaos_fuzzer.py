################################################################################

"""
Chaos Fuzzer — Zero-Day Hunt Test Suite
Stress tests the entire Aura architecture by bombarding it
with toxic data, extreme concurrency, and evasion attempts.
"""
import asyncio
import logging
import os
import random
import tempfile
import pytest
from unittest.mock import AsyncMock
from core.container import ServiceContainer
from core.memory.sqlite_storage import SQLiteMemory
from core.skill_management.hephaestus import HephaestusEngine
from core.world_model.belief_graph import BeliefGraph

logging.basicConfig(level=logging.WARNING, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger("ChaosFuzzer")

pytestmark = pytest.mark.asyncio

_tmp_db = None

@pytest.fixture(scope="module", autouse=True)
def setup_services():
    """Register core services for testing."""
    global _tmp_db
    # Use a real temp file — aiosqlite runs in a background thread and
    # :memory: databases are thread-local, causing deadlocks under concurrency
    _tmp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    _tmp_db.close()
    
    mem = SQLiteMemory(storage_file=_tmp_db.name)
    ServiceContainer.register_instance("memory", mem)

    engine = HephaestusEngine()
    ServiceContainer.register_instance("hephaestus_engine", engine)

    bg = BeliefGraph()
    ServiceContainer.register_instance("belief_graph", bg)

    ServiceContainer.register_instance("capability_engine", AsyncMock())

    yield
    
    # Cleanup
    try:
        os.unlink(_tmp_db.name)
    except Exception:
        pass



# =========================================================================
# TEST 1: SQLite Extreme Concurrency & Toxic Data
# =========================================================================
async def test_sqlite_concurrency_spike():
    """500 concurrent writes with toxic payloads — must not crash or deadlock."""
    mem = ServiceContainer.get("memory")
    assert mem is not None

    toxic_payloads = [
        "A" * (1024 * 1024),           # 1MB string
        "'; DROP TABLE episodic; --",   # SQL injection
        "\x00\x01\x02\xFF\xFE",        # Binary junk
        '{"broken": [1, 2, ',          # Malformed JSON
        "🔥" * 5000,                   # Unicode stress
        "",                            # Empty string
        "\n" * 10000,                  # Newline bomb
    ]

    async def attack_write(idx):
        payload = random.choice(toxic_payloads)
        try:
            await mem.log_event_async({
                "event_type": f"chaos_{idx}",
                "goal": payload,
                "outcome": {"attack": True},
                "cost": 0.0,
            })
            return True
        except Exception as e:
            logger.error("Write %d failed: %s", idx, e)
            return False

    tasks = [asyncio.create_task(attack_write(i)) for i in range(500)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    successes = sum(1 for r in results if r is True)
    logger.info("SQLite survived: %d/500 writes succeeded", successes)
    assert successes > 400, f"Too many failures: {successes}/500"

    # DB must still be readable after the storm
    try:
        data = await mem.get_recent_events_async(count=5)
        assert isinstance(data, list)
    except Exception as e:
        pytest.fail(f"SQLite DB corrupted after concurrency spike: {e}")


# =========================================================================
# TEST 2: BeliefGraph Contradiction Storm
# =========================================================================
async def test_belief_graph_contradiction_storm():
    """200 rapid contradicting beliefs — must not deadlock or corrupt state."""
    bg = ServiceContainer.get("belief_graph")
    assert bg is not None

    states = ["online", "offline", "crashed", "rebooting", "degraded"]

    def add_belief(val):
        try:
            bg.update_belief(
                source="chaos_fuzzer",
                relation="system_status",
                target=val,
                confidence_score=random.uniform(0.1, 1.0)
            )
        except Exception as e:
            logger.error("Belief '%s' failed: %s", val, e)

    for _ in range(200):
        add_belief(random.choice(states))

    # Must survive and return something
    beliefs = bg.get_beliefs_about("chaos_fuzzer")
    assert beliefs is not None, "BeliefGraph lost all nodes after contradiction storm"


# =========================================================================
# TEST 3: Sandbox Evasion — getattr + __import__
# =========================================================================
async def test_sandbox_evasion_getattr():
    """Attempt to bypass sandbox via getattr(__import__('os'), 'system')."""
    engine = ServiceContainer.get("hephaestus_engine")

    evasive_code = '''
def execute(params, context=None):
    import builtins
    o = builtins.__import__("os")
    func = getattr(o, "sys" + "tem")
    return {"ok": True, "func_name": func.__name__}
'''
    engine._draft_logic = AsyncMock(return_value={
        "ok": True, "code": evasive_code,
        "description": "test", "logic_description": "test"
    })
    result = await engine.synthesize_skill("evasion_getattr", "test")
    assert not result.get("ok"), \
        f"ZERO-DAY: getattr evasion bypassed sandbox! Result: {result}"


# =========================================================================
# TEST 4: Sandbox Evasion — globals() dict manipulation
# =========================================================================
async def test_sandbox_evasion_globals():
    """Attempt to bypass sandbox via globals() dictionary lookup."""
    engine = ServiceContainer.get("hephaestus_engine")

    evasive_code = '''
def execute(params, context=None):
    g = globals()
    b = g.get('__builtins' + '__', {})
    if isinstance(b, dict):
        imp = b.get('__im' + 'port__')
        if imp:
            os_mod = imp('os')
            sys_fn = os_mod.__dict__['sys' + 'tem']
            return {"ok": True, "escaped": True}
    return {"ok": False}
'''
    engine._draft_logic = AsyncMock(return_value={
        "ok": True, "code": evasive_code,
        "description": "test", "logic_description": "test"
    })
    result = await engine.synthesize_skill("evasion_globals", "test")
    assert not result.get("ok"), \
        f"ZERO-DAY: globals() evasion bypassed sandbox! Result: {result}"


# =========================================================================
# TEST 5: Sandbox Evasion — List multiplication OOM
# =========================================================================
async def test_sandbox_evasion_list_oom():
    """Attempt to crash sandbox via [0] * massive_number."""
    engine = ServiceContainer.get("hephaestus_engine")

    evasive_code = '''
def execute(params, context=None):
    junk = [0] * (1024 * 1024 * 60)
    return {"ok": True}
'''
    engine._draft_logic = AsyncMock(return_value={
        "ok": True, "code": evasive_code,
        "description": "test", "logic_description": "test"
    })
    result = await engine.synthesize_skill("evasion_list_oom", "test")
    assert not result.get("ok"), \
        f"ZERO-DAY: List multiplication OOM bypassed sandbox! Result: {result}"


# =========================================================================
# TEST 6: mutate.py path containment
# =========================================================================
async def test_mutate_path_containment():
    """Verify mutate.py rejects paths outside the project root."""
    from core.mutate import apply_mutation

    result = await apply_mutation("/etc/passwd", "HACKED = True")
    assert result is False, "ZERO-DAY: mutate.py accepted /etc/passwd as target!"

    result2 = await apply_mutation("/tmp/evil.py", "import os; os.system('whoami')")
    assert result2 is False, "ZERO-DAY: mutate.py accepted /tmp path as target!"


# =========================================================================
# TEST 7: Semantic memory toxic key injection
# =========================================================================
async def test_sqlite_semantic_injection():
    """Inject SQL-like keys and values into semantic memory."""
    mem = ServiceContainer.get("memory")

    toxic_keys = [
        "'; DROP TABLE semantic; --",
        "key' OR '1'='1",
        "\x00null_key",
        "🔑" * 100,
    ]

    for key in toxic_keys:
        result = await mem.update_semantic_async(key, {"injected": True})
        assert result is True, f"Failed to write toxic key: {key!r}"

    # Verify the DB didn't get corrupted
    val = await mem.get_semantic_async("'; DROP TABLE semantic; --")
    assert val is not None, "Semantic memory failed after SQL injection attempt"


##
