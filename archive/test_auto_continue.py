import asyncio
import os
import logging

logging.basicConfig(level=logging.INFO)

from core.orchestrator.boot import boot_orchestrator
from core.brain.inference_gate import InferenceGate

async def run_test():
    print("Booting headless orchestrator...")
    orchestrator = await boot_orchestrator(headless=True, skip_gui=True)
    
    # We want to force a low token limit to trigger auto-continuation.
    original_generate = orchestrator._inference_gate.generate
    
    async def patched_generate(prompt, context=None, timeout=None):
        if context is None: context = {}
        # Force a very low max_tokens
        context["max_tokens"] = 25
        print(f"\n[TEST] Calling InferenceGate with max_tokens=25.")
        result = await original_generate(prompt, context=context, timeout=timeout)
        if result:
            print(f"[TEST] InferenceGate returned {len(result)} chars ending with: {repr(result[-5:])}")
        return result
        
    orchestrator._inference_gate.generate = patched_generate
    
    print("\nSending long request to trigger truncation...")
    
    # We call _process_user_input_core directly to bypass queues
    response = await orchestrator._process_user_input_core(
        "Write a highly detailed, 500-word paragraph describing a sprawling cyberpunk city. Use very descriptive language.",
        origin="user"
    )
    
    print("\n" + "="*50)
    print("FINAL COMBINED RESPONSE:")
    print("="*50)
    print(response)
    print("="*50)

if __name__ == "__main__":
    asyncio.run(run_test())
