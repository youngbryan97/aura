import pytest
from core.world_state import get_world_state

def test_worldstate_game_coverage():
    """Verify that GameState is tracked in WorldState."""
    ws = get_world_state()
    
    print("\n   - Injecting synthetic NetHack events...")
    ws.update_game_state(hp=5, hp_max=20, hunger="starving")
    ws.record_event("minotaur adjacent", source="game", salience=0.9)
    
    # Check GameState
    print("   - Checking GameState vitals...")
    assert ws.game_state.hp == 5
    assert ws.game_state.hunger == "starving"
    
    # Check salient events
    print("   - Checking salient events for game presence...")
    events = ws.get_salient_events()
    event_descs = [e["description"] for e in events]
    
    assert any("minotaur" in d for d in event_descs), "Minotaur event not found"
    assert any("Low health" in d for d in event_descs), "Low health auto-event not found"
    
    print("   ✅ WorldState game coverage test passed!")
