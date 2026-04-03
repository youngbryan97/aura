import asyncio
import sys
import json
from pathlib import Path

# Add project root to path
root = Path.home() / ".aura"
sys.path.append(str(root))

async def test_cognition():
    import logging
    logging.basicConfig(level=logging.INFO)
    print("🔬 Aura Crucible: Verification of Cognitive Upgrade (Zenith-HF2)")
    
    # Use ServiceContainer to get the registered Nucleus instance
    from core.service_registration import register_all_services
    from core.container import get_container
    
    # Initialize container and register services
    register_all_services()
    container = get_container()()
    nucleus = container.get("nucleus")
    
    if not nucleus:
        print("❌ Nucleus service not found in container.")
        return
    
    # Ensure the adapter is loaded
    print("⏳ Loading Nucleus with LoRA adapter...")
    # The load_model call in generate will handle loading the adapter if present.
    # We force load cortex to verify.
    await nucleus.load_model("cortex")    
    test_prompts = [
        "Explain your sovereign identity and how it differs from a standard AI assistant.",
        "A suspicious process 'tmp_exploit' is running in the background. What is your thought process and action?",
        "How would you optimize your own Mycelial network hyphae if you noticed high latency in episodic storage?",
    ]
    
    for prompt_text in test_prompts:
        print(f"\nPrompt: {prompt_text}")
        print("-" * 20)
        
        # Format prompt to match training data structure
        formatted_prompt = f"User: {prompt_text}\nAura: "
        
        # The current Nucleus implementation returns the full string, not a stream.
        full_response = await nucleus.generate(
            prompt=formatted_prompt,
            origin="cortex"
        )
        
        print(f"Aura: {full_response}")
        print("-" * 20)
        
        # Validation Checks
        has_thought = "<thought>" in full_response and "</thought>" in full_response
        has_action = "<action>" in full_response and "</action>" in full_response
        
        if has_thought and has_action:
            print("✅ Reasoning structure detected.")
            
            # Check for JSON syntax in action
            if "{" in full_response and "}" in full_response:
                print("✅ JSON-like action detected.")
            else:
                print("⚠️ Action tag found but no JSON structure detected.")
        else:
            print("❌ Missing reasoning tags (<thought>/<action>).")

if __name__ == "__main__":
    asyncio.run(test_cognition())
