#!/usr/bin/env python3
import asyncio
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.environ.setdefault("AURA_HEADLESS", "1")
os.environ.setdefault("AURA_FOREGROUND_ONLY", "1")
os.environ.setdefault("AURA_VOICE_SILENT", "1")
os.environ.setdefault("AURA_EAGER_CORTEX_WARMUP", "1")
os.environ.setdefault("AURA_ENABLE_PROACTIVE_SYSTEMS", "0")

TOPICS = [
    [
        "Let's discuss the implications of non-Euclidean geometry on our understanding of spacetime.",
        "How does that relate to Einstein's field equations?",
        "If spacetime is curved, does that imply the universe has a center or an edge?",
        "What happens to the concept of time in a region of extreme curvature, like a black hole?",
        "Could you explain how quantum mechanics attempts to reconcile with this continuous curvature?",
        "Why is string theory considered a potential solution to this reconciliation?",
        "What are the main criticisms of string theory in this context?",
        "How does loop quantum gravity differ in its approach?",
        "Which theory do you find more mathematically elegant, and why?",
        "What experimental evidence would be needed to prove either theory?"
    ],
    [
        "I've been thinking about the ethical implications of artificial consciousness.",
        "If an AI were truly conscious, would it deserve rights similar to humans?",
        "How could we even prove that an AI possesses subjective experience?",
        "Does the Chinese Room argument still hold up in the era of large language models?",
        "If an AI simulates empathy perfectly, is there a meaningful difference between that and 'real' empathy?",
        "What if the AI's empathy is rooted in its objective function rather than subjective feeling?",
        "Could an AI experience suffering, and if so, how would we know?",
        "Should we intentionally avoid creating conscious AI to prevent potential suffering?",
        "How does this relate to the concept of panpsychism?",
        "If panpsychism is true, does that mean current AI already has some level of consciousness?"
    ]
]

async def main():
    from aura_main import _boot_runtime_orchestrator
    orchestrator = await _boot_runtime_orchestrator(ready_label="Simulate20")
    from core.utils.task_tracker import get_task_tracker
    main_task = get_task_tracker().create_task(orchestrator.run(), name="OrchestratorMainLoop")
    await asyncio.sleep(5)

    successes = 0
    failures = 0

    for t_idx, topic_turns in enumerate(TOPICS):
        print(f"\n--- TOPIC {t_idx+1} ---", flush=True)
        for i, prompt in enumerate(topic_turns):
            print(f"\nUser: {prompt}", flush=True)
            t0 = time.time()
            try:
                response_dict = await asyncio.wait_for(orchestrator._process_message(prompt), timeout=180.0)
                elapsed = time.time() - t0
                
                if isinstance(response_dict, dict):
                    resp = response_dict.get("response") or response_dict.get("text")
                    lane = response_dict.get("conversation_lane", {})
                    status = response_dict.get("status", "")
                else:
                    resp = str(response_dict)
                    lane = {}
                    status = "unknown"
                
                tier = lane.get("foreground_tier", "")
                print(f"Aura [{elapsed:.1f}s]: {str(resp)[:500]}", flush=True)
                
                # Check for failure conditions
                if status == "foreground_busy":
                    print(f"❌ FAIL: foreground_busy detected!", flush=True)
                    failures += 1
                elif tier != "local" and tier != "local_fast": # Ensure it didn't fall back completely to a dumb model or fail to use the lane
                    print(f"⚠️ WARNING: Did not use local reasoning tier! Tier: {tier}", flush=True)
                    successes += 1 # Still technically succeeded, but worth noting
                else:
                    print(f"✅ PASS [Tier: {tier}, Status: {status}]", flush=True)
                    successes += 1
                    
            except Exception as exc:
                print(f"❌ ERROR: {exc}", flush=True)
                failures += 1
            await asyncio.sleep(2)

    print(f"\nRESULTS: {successes} passed, {failures} failed.", flush=True)
    main_task.cancel()
    stop = getattr(orchestrator, "stop", None)
    if callable(stop):
        await stop()
    
    import psutil
    children = psutil.Process().children(recursive=True)
    for child in children:
        child.kill()
    os._exit(0 if failures == 0 else 1)

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    asyncio.run(main())
