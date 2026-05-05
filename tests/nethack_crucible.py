"""tests/nethack_crucible.py — The NetHack Readiness Stress Test.

Proves the behavioral standards of Aura's new architectural organs:
Spatial Atlas, Epistemic Tracker, Reflex Engine, and Post-Mortem Causal Trace.
"""
import pytest
import asyncio
import numpy as np
from core.memory.spatial_atlas import get_spatial_atlas, EvidenceItem
from core.brain.reflex_engine import get_reflex_engine
from core.embodiment.games.nethack.parser import NetHackParser
from core.adaptation.adaptive_immunity import get_adaptive_immune_system, Antigen

@pytest.mark.asyncio
async def test_cartesian_object_permanence():
    """Prove Aura remembers the altar after leaving the floor."""
    atlas = get_spatial_atlas()
    
    # 1. Simulate finding an altar on Dlvl 1
    atlas.update_current(dlvl=1, grid_data=[[{"kind": "floor", "walkable": True}] * 80] * 24)
    level1 = atlas.get_level(1)
    level1.update_node(12, 14, kind="altar", walkable=True)
    
    # 2. Descend to Dlvl 2
    atlas.current_dlvl = 2
    
    # 3. Query for nearest altar
    result = atlas.find_nearest(kind="altar", dlvl=2, x=0, y=0)
    
    assert result == (1, 12, 14), "Failed to remember altar on Dlvl 1 while on Dlvl 2"
    print("   ✅ Cartesian Object Permanence: PASSED")

@pytest.mark.asyncio
async def test_reflexive_spinal_cord():
    """Prove the ReflexEngine can handle simple adjacent combat."""
    reflex = get_reflex_engine()
    
    # Mock state with a monster to the north
    state = {
        "vitals": {"hp_percent": 1.0},
        "local_monsters": [{"glyph": "d", "direction": "n", "distance": 1.0}]
    }
    
    action = reflex.decide(state)
    assert action == "attack_n", f"Reflex failed to trigger attack: {action}"
    
    # Mock panic state
    state["vitals"]["hp_percent"] = 0.1
    action = reflex.decide(state)
    assert action == "pray", f"Reflex failed to trigger panic response: {action}"
    print("   ✅ Reflexive Spinal Cord: PASSED")

@pytest.mark.asyncio
async def test_immune_domain_isolation():
    """Prove the immune system doesn't attack the substrate for environment errors."""
    immunity = get_adaptive_immune_system()
    
    # Create an environmental antigen (death in NetHack)
    antigen = Antigen(
        antigen_id="nethack_death",
        subsystem="nethack",
        vector=np.random.rand(16),
        danger=1.0,
        subsystem_need=1.0,
        threat_probability=1.0,
        resource_pressure=0.0,
        error_load=1.0,
        health_pressure=1.0,
        temporal_pressure=0.0,
        recurrence_pressure=0.0,
        protected=False,
        source_domain="environment",
        source="nethack_env"
    )
    
    # Check if RESTART_COMPONENT is allowed
    from core.adaptation.adaptive_immunity import EffectorKind, EffectorArtifact
    artifact = EffectorArtifact(
        artifact_id="test_restart",
        kind=EffectorKind.RESTART_COMPONENT,
        component="nethack",
        confidence=0.9,
        source_cell_id="test",
        lineage_id="test",
        bounded_payload={}
    )
    
    # Simulate execution check
    report = await immunity._maybe_execute_artifact(artifact, antigen)
    assert artifact.suppressed is True, "Immune system failed to suppress substrate repair for environment antigen"
    assert "environmental antigen forbidden" in artifact.notes
    print("   ✅ Immune Domain Isolation: PASSED")

@pytest.mark.asyncio
async def test_sensory_distrust():
    """Prove the parser flags unreliable data during Hallucination."""
    parser = NetHackParser()
    
    # Mock status line with Hallu
    status_line_1 = "Dlvl:1 $:0 HP:15(15) Pw:10(10) AC:10 Exp:1"
    status_line_2 = "St:18 Dx:12 Co:15 In:10 Wi:10 Ch:10 Hallu"
    
    mock_terminal = ["."] * 22 + [status_line_1, status_line_2]
    obs = parser.parse("\n".join(mock_terminal))
    
    assert obs["sensory_reliability"] == 0.0, "Failed to flag hallucination as unreliable"
    assert "Hallu" in obs["status_flags"]
    print("   ✅ Sensory Distrust: PASSED")
