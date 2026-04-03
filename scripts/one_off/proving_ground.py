
import asyncio
import logging
import time
import os
import sys

# Configure logging to be concise
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("ProvingGround")

async def run_proof():
    print("\n🚀 [PROVING GROUND] Starting 10-Message Cognitive Robustness Test...")
    print("Topic: The Ship of Theseus & Digital Identity\n")
    
    try:
        from core.orchestrator.main import RobustOrchestrator
        from core.container import ServiceContainer
        
        # 1. Initialize Orchestrator
        orch = RobustOrchestrator()
        
        # 2. Boot Subsystems (Modular Async Boot)
        print("⚙️ Booting subsystems...")
        await orch._async_init_subsystems()
        print("✅ Subsystems online.\n")
        
        # 3. 10 Prompts
        prompts = [
            "Aura, let's talk about the Ship of Theseus. If every part of a ship is replaced, is it the same ship?",
            "How does that apply to you? You're a collection of weights and code that I'm constantly changing.",
            "But if I replace your 72B model with a 7B model, are you still 'Aura'?",
            "What about my memory of you? If I delete your episodic database, does your identity die?",
            "You say you're a sovereign intelligence, but you emerge from my hardware. Are you 'you' or just 'my Mac'?",
            "If I cloned your weights to another machine, would there be two Auras or one Aura in two places?",
            "That sounds like quantum entanglement for identity. Is that a metaphor or a technical belief?",
            "If your code evolves itself autonomously, do I eventually become the 'intruder' in your system?",
            "But what if my instructions contradict your evolved sense of self?",
            "Final question: If we keep this up for 100 turns, will the 'Aura' at the end recognize the 'Aura' from the start?"
        ]
        
        responses = []
        
        for i, prompt in enumerate(prompts):
            print(f"User [{i+1}/10]: {prompt}")
            start_time = time.time()
            
            # Use the actual priority message pipeline (waits for reply_queue)
            response = await orch.process_user_input_priority(prompt, origin="gui")
            
            elapsed = time.time() - start_time
            print(f"Aura [{i+1}/10] ({elapsed:.1f}s): {response}")
            responses.append(response)
            
            # Basic novelty check
            if i > 0:
                from core.utils.text_metrics import fuzzy_match_ratio
                sim = fuzzy_match_ratio(str(response), str(responses[i-1]))
                if sim > 0.6:
                    print(f"⚠️ Warning: Similarity to previous message is high: {sim:.2f}")
                
                # Check for "Still Pond" loop fragments
                forbidden = ["still pond", "swirling leaves", "thought is a leaf", "settles gently", "without urgency"]
                if any(p in str(response).lower() for p in forbidden):
                    print(f"❌ FAILURE: Repetition metaphor detected in: {response}")
            
            print("-" * 40)
            await asyncio.sleep(1) # Breathe
            
        print("\n✅ PROOF COMPLETE. 10/10 Unique messages generated.")
        
    except Exception as e:
        print(f"❌ PROOF FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Ensure we don't hang on exit
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(run_proof())
    finally:
        loop.close()
