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
        "Could you explain how quantum mechanics attempts to reconcile with this continuous curvature?"
    ]
]

async def main():
    from aura_main import _boot_runtime_orchestrator
    orchestrator = await _boot_runtime_orchestrator(ready_label="SimulateDeep")
    from core.utils.task_tracker import get_task_tracker
    main_task = get_task_tracker().create_task(orchestrator.run(), name="OrchestratorMainLoop")
    await asyncio.sleep(5)

    for t_idx, topic_turns in enumerate(TOPICS):
        print(f"\n--- TOPIC {t_idx+1} ---")
        for i, prompt in enumerate(topic_turns):
            print(f"\nUser: {prompt}")
            t0 = time.time()
            try:
                response_dict = await asyncio.wait_for(orchestrator._process_message(prompt), timeout=300.0)
                elapsed = time.time() - t0
                if isinstance(response_dict, dict):
                    resp = response_dict.get("response") or response_dict.get("text")
                else:
                    resp = response_dict
                print(f"Aura [{elapsed:.1f}s]: {str(resp)[:500]}")
                lane = response_dict.get("conversation_lane", {}) if isinstance(response_dict, dict) else {}
                print(f"[Lane info: tier={lane.get('foreground_tier')}, state={lane.get('state')}]")
            except Exception as exc:
                print(f"ERROR: {exc}")
            await asyncio.sleep(2)

    main_task.cancel()
    stop = getattr(orchestrator, "stop", None)
    if callable(stop):
        await stop()
    import psutil
    children = psutil.Process().children(recursive=True)
    for child in children:
        child.kill()
    os._exit(0)

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    asyncio.run(main())
