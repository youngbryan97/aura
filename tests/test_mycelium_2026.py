################################################################################


import pytest
import re
from core.mycelium import MycelialNetwork, HardwiredPathway, Hypha

@pytest.fixture
def network():
    # Reset singleton state for clean test
    MycelialNetwork._instance = None
    MycelialNetwork._initialized = False
    return MycelialNetwork()

def test_singleton_safety():
    net1 = MycelialNetwork()
    net2 = MycelialNetwork()
    assert net1 is net2
    assert net1._initialized is True

def test_pathway_validation(network):
    network.register_pathway(
        pathway_id="img_gen",
        pattern=r"draw\s+(.+)",
        skill_name="generate_image",
        param_map={"prompt": 1}
    )
    
    assert "img_gen" in network.pathways
    pw = network.pathways["img_gen"]
    assert isinstance(pw, HardwiredPathway)
    assert pw.skill_name == "generate_image"
    
    match = network.match_hardwired("draw a neon cat")
    assert match is not None
    pw, params = match
    assert params["prompt"] == "a neon cat"

def test_hypha_pydantic(network):
    network.establish_connection("A", "B", priority=2.0)
    assert "A->B" in network.hyphae
    h = network.hyphae["A->B"]
    assert isinstance(h, Hypha)
    assert h.priority == 2.0
    
    h.pulse(success=True)
    assert h.strength > 1.0


def test_dormant_hyphae_are_not_monitored_until_they_carry_traffic(network):
    network.establish_connection("Dormant", "Edge", priority=2.0)
    h = network.hyphae["Dormant->Edge"]

    assert h.pulse_count == 0
    assert network._should_monitor_hypha(h) is False

    h.refresh_heartbeat()
    assert h.strength == 1.0
    assert network._should_monitor_hypha(h) is False

    h.pulse(success=True)
    assert network._should_monitor_hypha(h) is True

def test_infrastructure_mapping(network):
    # Mock some mapped files
    network.mapped_files = {"core.logic": {"path": "/path/to/logic.py"}}
    network.infrastructure_mapped = True
    
    # Register a pathway with a source file
    network.register_pathway(
        pathway_id="test_geo",
        pattern=r"where\s+is\s+(.+)",
        skill_name="geo_skill",
    )
    network.pathways["test_geo"].source_file = "/path/to/logic.py"
    
    # Establish a physical hypha
    network.hyphae["phys_1"] = Hypha(name="phys_1", source="core.logic", target="core.utils", is_physical=True)
    
    # Reinforce the pathway and verify physical hypha pulses
    network.reinforce("test_geo", success=True)
    assert network.hyphae["phys_1"].strength > 1.0


##
