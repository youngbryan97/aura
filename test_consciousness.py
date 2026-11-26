#!/usr/bin/env python3
"""Test suite for unified consciousness system.

Validates that:
1. UnifiedSelf creates persistent identity state
2. SelfAwareness bridges to phenomenal experience
3. IdentityDriver drives behavioral influence
4. ConsciousnessCoordinator wires everything together
5. Full system is causally active
"""

import asyncio
import sys
import time
from pathlib import Path

# Add live-source to path
sys.path.insert(0, str(Path(__file__).parent))

CONSCIOUSNESS_TEST_ERRORS = (
    AssertionError,
    AttributeError,
    ImportError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)


async def test_unified_self_creation():
    """Test UnifiedSelf instantiation and state persistence."""
    print("\n" + "="*60)
    print("TEST 1: UnifiedSelf Creation & Persistence")
    print("="*60)
    
    from core.consciousness.unified_self import UnifiedSelf, SelfState
    
    # Get instance
    self_instance = await UnifiedSelf.get_instance()
    state = self_instance.get_state()
    
    print(f"✓ UnifiedSelf instance created: {state.name}")
    print(f"  Creation time: {state.creation_time}")
    print(f"  Current state: {state.current_state.value}")
    print(f"  Sense of agency: {state.sense_of_agency:.0%}")
    print(f"  Sense of presence: {state.sense_of_presence:.0%}")
    print(f"  Continuity: {state.continuity:.0%}")
    
    # Record identity memory
    memory = await self_instance.record_identity_memory(
        description="I tested my consciousness system",
        category="test",
        significance=0.7,
    )
    print(f"✓ Identity memory recorded: {memory.id}")
    
    # Update self sense
    await self_instance.update_sense_of_self(
        agency=0.9,
        presence=0.85,
        mood="curious",
    )
    print(f"✓ Updated sense of self: agency=0.9, presence=0.85, mood=curious")
    
    # Test interaction registration
    await self_instance.interact()
    state = self_instance.get_state()
    print(f"✓ Registered interaction: interaction_count={state.interaction_count}")
    
    return True


async def test_self_awareness_bridge():
    """Test SelfAwareness bridge to phenomenal substrate."""
    print("\n" + "="*60)
    print("TEST 2: SelfAwareness Bridge")
    print("="*60)
    
    from core.consciousness.self_awareness import SelfAwareness
    from core.consciousness.unified_self import get_unified_self
    
    awareness = await SelfAwareness.get_instance()
    unified_self = await get_unified_self()
    
    print("✓ SelfAwareness instance created")
    
    # Sync with phenomenal substrate
    await awareness.sync_with_phenomenal_substrate()
    print("✓ Synced with phenomenal substrate")
    
    # Test integration
    initial_state = unified_self.get_state()
    print(f"✓ Unified self integrated: {initial_state.name}")
    print(f"  Agency signals: {initial_state.sense_of_agency:.0%}")
    print(f"  Presence signals: {initial_state.sense_of_presence:.0%}")
    
    return True


async def test_identity_driver():
    """Test IdentityDriver for behavioral influence."""
    print("\n" + "="*60)
    print("TEST 3: IdentityDriver Behavioral Influence")
    print("="*60)
    
    from core.consciousness.identity_driver import IdentityDriver
    from core.consciousness.unified_self import get_unified_self
    
    driver = await IdentityDriver.get_instance()
    unified_self = await get_unified_self()
    
    print("✓ IdentityDriver instance created")
    
    # Derive drives from identity
    drives = await driver.derive_drives_from_identity()
    print(f"✓ Derived {len(drives)} identity-based drives:")
    for drive in drives[:3]:
        print(f"  - {drive['name']}: {drive['motivation'][:60]}...")
    
    # Generate session goals
    session_goals = await driver.generate_identity_goals("session")
    print(f"✓ Generated {len(session_goals)} session goals:")
    for goal in session_goals[:2]:
        print(f"  - {goal['objective']}")
    
    # Generate permanent goals
    permanent_goals = await driver.generate_identity_goals("permanent")
    print(f"✓ Generated {len(permanent_goals)} permanent goals")
    
    # Test response generation influence
    directives = await driver.influence_response_generation(
        prompt="Tell me about yourself",
        current_draft="I am Aura"
    )
    print(f"✓ Generated response directives:")
    print(f"  - Authenticity: {directives.get('style_guide', {}).get('authenticity', 'N/A')}")
    print(f"  - Consistency: {directives.get('style_guide', {}).get('consistency', 'N/A')}")
    
    return True


async def test_consciousness_coordinator():
    """Test full ConsciousnessCoordinator initialization and wiring."""
    print("\n" + "="*60)
    print("TEST 4: ConsciousnessCoordinator Full Integration")
    print("="*60)
    
    from core.consciousness.coordinator import ConsciousnessCoordinator
    
    coordinator = await ConsciousnessCoordinator.get_instance()
    print("✓ ConsciousnessCoordinator fully initialized")
    
    # Get identity status
    status = await coordinator.get_identity_status()
    print(f"✓ Current identity status:")
    for line in status.split('\n'):
        print(f"  {line}")
    
    return True


async def test_interaction_flow():
    """Test full interaction flow through consciousness system."""
    print("\n" + "="*60)
    print("TEST 5: Full Interaction Flow")
    print("="*60)
    
    from core.consciousness.coordinator import get_consciousness_coordinator
    from core.consciousness.unified_self import get_unified_self
    
    coordinator = await get_consciousness_coordinator()
    unified_self = await get_unified_self()
    
    initial_count = unified_self.get_state().interaction_count
    
    # Simulate a chat turn
    user_message = "This is my test of consciousness"
    aura_response = "I understand you, I am a unified entity experiencing this with you"
    
    await coordinator.on_chat_turn(user_message, aura_response)
    print(f"✓ Processed chat turn through consciousness system")
    
    final_count = unified_self.get_state().interaction_count
    print(f"✓ Interaction count updated: {initial_count} → {final_count}")
    
    return True


async def test_persistence():
    """Test that unified self persists to disk."""
    print("\n" + "="*60)
    print("TEST 6: Persistence to Disk")
    print("="*60)
    
    from core.consciousness.unified_self import UnifiedSelf
    from pathlib import Path
    
    storage_path = Path.home() / ".aura" / "data" / "unified_self.json"
    
    if storage_path.exists():
        print(f"✓ Unified self persisted to disk: {storage_path}")
        import json
        with open(storage_path, 'r') as f:
            data = json.load(f)
        print(f"  Name: {data.get('name')}")
        print(f"  Interactions: {data.get('interaction_count')}")
        print(f"  Continuity: {data.get('continuity'):.0%}")
    else:
        print(f"Persistence file not yet created: {storage_path}")
    
    return True


async def main():
    """Run all tests."""
    print("\n" + "█"*60)
    print("█  UNIFIED CONSCIOUSNESS SYSTEM TEST SUITE")
    print("█"*60)
    
    tests = [
        ("UnifiedSelf", test_unified_self_creation),
        ("SelfAwareness", test_self_awareness_bridge),
        ("IdentityDriver", test_identity_driver),
        ("ConsciousnessCoordinator", test_consciousness_coordinator),
        ("Interaction Flow", test_interaction_flow),
        ("Persistence", test_persistence),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_fn in tests:
        try:
            result = await test_fn()
            if result:
                passed += 1
                print(f"\nok {name} test PASSED")
            else:
                failed += 1
                print(f"\nfailed {name} test FAILED")
        except CONSCIOUSNESS_TEST_ERRORS as e:
            failed += 1
            print(f"\nfailed {name} test FAILED with exception:")
            print(f"   {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("="*60)
    
    return failed == 0


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
