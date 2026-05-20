"""tests/test_morphogenesis_migration.py — Unit tests for state migrator & functional bridge

Validates VersionRegistry dynamic updates, MorphicStateProxy transparent migrations,
and EphemeralLogicRouter live AST reload/execute behavior.
"""
import sys
import pytest
from core.morphogenesis.state_migrator import VersionRegistry, MorphicStateProxy
from core.ops.stateless_functional_bridge import EphemeralLogicRouter


# ── Mock Classes for Testing ──

class SimpleCellV1:
    def __init__(self, count: int = 0):
        self.count = count

    def increment(self, amount: int = 1) -> int:
        self.count += amount
        return self.count


class SimpleCellV2:
    def __init__(self, count: int = 0, name: str = "Cell"):
        self.count = count
        self.name = name

    def increment(self, amount: int = 1) -> int:
        self.count += amount * 2  # Mutated behavior
        return self.count

    @classmethod
    def __migrate_state__(cls, old_state: dict) -> dict:
        old_state["name"] = "MigratedCell"
        return old_state


def test_version_registry_and_morphic_state_proxy():
    VersionRegistry.clear()
    
    # 1. Register V1 cell version
    VersionRegistry.register_cell_version("simple_cell", 1, SimpleCellV1)
    
    # 2. Instantiate MorphicStateProxy
    proxy = MorphicStateProxy("simple_cell", {"count": 10})
    
    # 3. Assert transparent forwarding to V1
    assert proxy.count == 10
    assert proxy.increment(5) == 15
    assert proxy.count == 15
    
    # 4. Register V2 cell version
    VersionRegistry.register_cell_version("simple_cell", 2, SimpleCellV2)
    
    # 5. Assert transparent migration & execution of mutated logic
    assert proxy.increment(5) == 25  # 15 + (5 * 2) = 25
    assert proxy.count == 25
    assert proxy.name == "MigratedCell"


def test_ephemeral_logic_router(tmp_path):
    # Create dynamic morphic directory
    module_dir = tmp_path / "morphic_dynamic"
    module_dir.mkdir()
    module_file = module_dir / "dynamic_rule.py"
    
    # Add path to sys.path so we can import it
    sys.path.insert(0, str(tmp_path))
    try:
        # Write V1 of logic function
        module_file.write_text(
            "def calculate(state, val):\n"
            "    state['total'] += val\n"
            "    return state, state['total']\n"
        )
        
        state = {"total": 100}
        state, res = EphemeralLogicRouter.execute_mutated_logic("morphic_dynamic.dynamic_rule", "calculate", state, 50)
        assert res == 150
        assert state["total"] == 150
        
        # Write V2 mutated logic function to simulate dynamic code morphing
        module_file.write_text(
            "def calculate(state, val):\n"
            "    state['total'] += val * 10\n"
            "    return state, state['total']\n"
        )
        
        # Router must reload the module and pick up mutated logic on the same state payload!
        state, res2 = EphemeralLogicRouter.execute_mutated_logic("morphic_dynamic.dynamic_rule", "calculate", state, 10)
        assert res2 == 250
        assert state["total"] == 250
        
    finally:
        sys.path.remove(str(tmp_path))
        if "morphic_dynamic.dynamic_rule" in sys.modules:
            del sys.modules["morphic_dynamic.dynamic_rule"]
