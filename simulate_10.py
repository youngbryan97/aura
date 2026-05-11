#!/usr/bin/env python3
"""
Simulate 10 real conversation turns through the full Aura pipeline.
Uses the same boot path as `python aura_main.py --cli`.
"""
import asyncio
import sys
import os
import time

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PROMPTS = [
    "Hello Aura! How are you doing today?",
    "What is the capital of France?",
    "Can you write a short poem about the ocean?",
    "What is 15 * 12?",
    "If I have 3 apples and eat 2, how many are left?",
    "Who wrote the play Hamlet?",
    "Can you name three programming languages?",
    "What color is the sky on a clear day?",
    "Translate 'Good morning' to Spanish.",
    "What is the square root of 64?"
]


async def main():
    # Use the same boot path as the CLI
    from aura_main import _boot_runtime_orchestrator

    print("=" * 60, flush=True)
    print("  SIMULATION: Booting Aura via canonical CLI path...", flush=True)
    print("=" * 60, flush=True)

    orchestrator = await _boot_runtime_orchestrator(ready_label="Simulate10")

    # Start the orchestrator's main loop in the background
    from core.utils.task_tracker import get_task_tracker
    get_task_tracker().create_task(orchestrator.run(), name="OrchestratorMainLoop")

    # Give background systems a moment to stabilize
    print("\n⏳ Waiting 5s for background systems to settle...", flush=True)
    await asyncio.sleep(5)

    print("\n" + "=" * 60, flush=True)
    print("  SENDING 10 MESSAGES", flush=True)
    print("=" * 60, flush=True)

    successes = 0
    failures = 0

    for i, prompt in enumerate(PROMPTS):
        print(f"\n{'─' * 50}", flush=True)
        print(f"  [{i+1}/10] User: {prompt}", flush=True)
        print(f"{'─' * 50}", flush=True)

        t0 = time.time()
        try:
            response_dict = await asyncio.wait_for(
                orchestrator._process_message(prompt),
                timeout=180.0,
            )

            elapsed = time.time() - t0

            # Extract response text
            response_text = None
            if isinstance(response_dict, dict):
                response_text = response_dict.get("response") or response_dict.get("text") or response_dict.get("reply")
            elif isinstance(response_dict, str):
                response_text = response_dict
            else:
                response_text = str(response_dict) if response_dict else None

            if response_text and len(response_text.strip()) > 0:
                # Truncate for display
                display = response_text.strip()[:500]
                print(f"  Aura: {display}", flush=True)
                print(f"  ✅ [{elapsed:.1f}s]", flush=True)
                successes += 1
            else:
                print(f"  ❌ EMPTY RESPONSE (got: {response_dict})", flush=True)
                failures += 1

        except asyncio.TimeoutError:
            print(f"  ❌ TIMEOUT after 180s", flush=True)
            failures += 1
        except Exception as exc:
            print(f"  ❌ ERROR: {type(exc).__name__}: {exc}", flush=True)
            failures += 1

        # Small pause between messages
        await asyncio.sleep(2)

    print(f"\n{'=' * 60}", flush=True)
    print(f"  RESULTS: {successes}/10 successes, {failures}/10 failures", flush=True)
    print(f"{'=' * 60}", flush=True)

    if successes == 10:
        print("  🎉 ALL 10 REPLIES SUCCESSFUL!", flush=True)
    else:
        print(f"  ⚠️  {failures} failures detected.", flush=True)

    # Clean exit
    sys.exit(0)


if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    asyncio.run(main())
