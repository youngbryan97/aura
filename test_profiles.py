#!/usr/bin/env python3
"""Quick test of semantic fact extraction and profile systems."""

import asyncio
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SCRIPT_RECOVERABLE_ERRORS = (
    ImportError,
    AttributeError,
    RuntimeError,
    TypeError,
    ValueError,
    OSError,
    asyncio.TimeoutError,
)

from core.memory.profile_manager import ProfileManager  # noqa: E402
from core.memory.semantic_fact_extractor import SemanticFactExtractor  # noqa: E402


async def test_fact_extraction():
    """Test fact extraction from sample conversations."""
    print("\n=== Testing Semantic Fact Extraction ===\n")
    
    extractor = SemanticFactExtractor()
    
    # Sample conversation
    user_msg = "I prefer concise responses with bullet points. I'm a backend developer who specializes in Python and Rust."
    aura_resp = "I noticed you like clear, organized information. I've learned that you value efficiency and code quality."
    
    facts = extractor.extract_facts(user_msg, aura_resp)
    
    print(f"Extracted {len(facts)} facts:")
    for fact in facts:
        print(f"  • {fact.to_natural_language()}")
        print(f"    Type: {fact.fact_type.value}, Confidence: {fact.confidence:.0%}")
    
    return len(facts) > 0


async def test_profile_learning():
    """Test profile learning from facts."""
    print("\n=== Testing Profile Learning ===\n")
    
    manager = await ProfileManager.get_instance()
    
    # Simulate a conversation
    user_msg = "I prefer bullet points. I specialize in Python and Kubernetes."
    aura_resp = "I notice you appreciate clarity and technical depth. You work with cloud infrastructure."
    
    user_learned, aura_learned = await manager.learn_from_turn(
        user_id="bryan",
        user_message=user_msg,
        aura_response=aura_resp,
        session_id="test"
    )
    
    print(f"Learning results: {user_learned} user facts, {aura_learned} self facts")
    
    # Check profiles
    user_profile = manager.get_user_profile()
    if user_profile:
        print("\nUser Profile:")
        print(user_profile.summary("bryan"))
    
    aura_profile = manager.get_aura_profile()
    if aura_profile:
        print("\nAura Self-Profile:")
        print(aura_profile.summary())
    
    return user_learned > 0


async def test_context_injection():
    """Test context generation for LLM injection."""
    print("\n=== Testing Context Injection ===\n")
    
    manager = await ProfileManager.get_instance()
    
    context = await manager.get_context_injection("bryan")
    if context:
        print("Generated context block:")
        print(context[:500] + ("..." if len(context) > 500 else ""))
        return True
    else:
        print("No context generated (profiles may be empty)")
        return False


async def main():
    """Run all tests."""
    print("╔════════════════════════════════════════════════════════════╗")
    print("║     Semantic Fact Extraction & Profile System Test         ║")
    print("╚════════════════════════════════════════════════════════════╝")
    
    results = []
    
    try:
        # Test 1: Fact extraction
        test1 = await test_fact_extraction()
        results.append(("Fact Extraction", test1))
        
        # Test 2: Profile learning
        test2 = await test_profile_learning()
        results.append(("Profile Learning", test2))
        
        # Test 3: Context injection
        test3 = await test_context_injection()
        results.append(("Context Injection", test3))
        
    except SCRIPT_RECOVERABLE_ERRORS as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Summary
    print("\n╔════════════════════════════════════════════════════════════╗")
    print("║                        Test Summary                        ║")
    print("╠════════════════════════════════════════════════════════════╣")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"║ {test_name:<40} {status:>15} ║")
    
    print("╠════════════════════════════════════════════════════════════╣")
    print(f"║ Total: {passed}/{total} tests passed                                 ║")
    print("╚════════════════════════════════════════════════════════════╝")
    
    return passed == total


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
