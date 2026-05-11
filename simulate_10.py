#!/usr/bin/env python3
"""
Simulate 10 real conversation turns through the full Aura pipeline.
Uses the same boot path as `python aura_main.py --cli`.
"""
import asyncio
import sys
import os
import time
import re

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Keep the verification run headless and focused on the conversation lane.
os.environ.setdefault("AURA_HEADLESS", "1")
os.environ.setdefault("AURA_FOREGROUND_ONLY", "1")
os.environ.setdefault("AURA_VOICE_SILENT", "1")
os.environ.setdefault("AURA_ENABLE_PROACTIVE_VISION", "0")
os.environ.setdefault("AURA_ENABLE_PERMANENT_SWARM", "0")
os.environ.setdefault("AURA_ENABLE_MORPHOGENESIS", "0")
os.environ.setdefault("AURA_ENABLE_SELF_HEALING", "0")
os.environ.setdefault("AURA_ENABLE_DEEP_REPAIR", "0")
os.environ.setdefault("AURA_ENABLE_AUTONOMY_CONDUCTOR", "0")
os.environ.setdefault("AURA_ENABLE_AUTONOMOUS_SELF_MODIFICATION", "0")
os.environ.setdefault("AURA_ENABLE_SELF_MODIFICATION_ENGINE", "0")
os.environ.setdefault("AURA_ENABLE_EVOLUTION_ORCHESTRATOR", "0")
os.environ.setdefault("AURA_ENABLE_PROACTIVE_SYSTEMS", "0")
os.environ.setdefault("AURA_ENABLE_RESEARCH_CYCLE", "0")
os.environ.setdefault("AURA_ENABLE_SENSORIMOTOR_GROUNDING", "0")
os.environ.setdefault("AURA_ENABLE_SINGULARITY_LOOPS", "0")
os.environ.setdefault("AURA_ENABLE_ACTIVATION_AUDIT", "0")
os.environ.setdefault("AURA_ENABLE_STEM_CELL_CAPTURE", "0")
os.environ.setdefault("AURA_ENABLE_VIABILITY", "0")
os.environ.setdefault("AURA_ENABLE_PERFORMANCE_GUARD", "0")
os.environ.setdefault("AURA_SEED_AUTONOMY_GOALS", "0")
os.environ.setdefault("AURA_REGISTER_REIMPLEMENTATION_LAB", "0")
os.environ.setdefault("AURA_REGISTER_ARCHITECTURE_GOVERNOR", "0")
os.environ.setdefault("AURA_RECURRENT_LOOPS", "0")
os.environ.setdefault("AURA_EAGER_CORTEX_WARMUP", "1")

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

RECOVERY_RE = re.compile(
    r"(try (?:me|it|that|your message) again|send (?:it|your message) again|"
    r"reasoning engine hit|thinking engine hit|cortex is catching up|"
    r"deeper processing is taking longer|couldn'?t respond properly|"
    r"dropped the heavy reasoning lane|deeper lane recovers|lighter mode|"
    r"keeping (?:this|the turn) active|my local cortex hit|"
    r"under load right now|holding (?:it|this|the thread) while i recover|"
    r"let me regroup|my deeper processing)",
    re.IGNORECASE,
)
ROLE_LEAK_RE = re.compile(r"(^|\s)(User|Assistant|Human)\s*:|_user\b|(^|\s)User\s+(?=What|Can|If|Who|Translate|Yes|No)", re.IGNORECASE)

EXPECTED_MARKERS = [
    (),
    ("paris",),
    ("ocean", "wave"),
    ("180",),
    ("1",),
    ("shakespeare",),
    ("python",),
    ("blue",),
    ("buenos",),
    ("8",),
]


async def _shutdown_runtime(orchestrator, main_task=None):
    if main_task is not None:
        main_task.cancel()
        await asyncio.sleep(0.25)

    try:
        stop = getattr(orchestrator, "stop", None)
        if callable(stop):
            await asyncio.wait_for(stop(), timeout=12.0)
    except Exception:
        pass

    try:
        import psutil

        current = psutil.Process()
        children = current.children(recursive=True)
        for child in children:
            try:
                child.terminate()
            except psutil.Error:
                pass
        gone, alive = psutil.wait_procs(children, timeout=4.0)
        for child in alive:
            try:
                child.kill()
            except psutil.Error:
                pass
    except Exception:
        pass


def _extract_response_text(response_dict):
    if isinstance(response_dict, dict):
        candidate = response_dict.get("response") or response_dict.get("text") or response_dict.get("reply")
        if isinstance(candidate, dict):
            for key in ("content", "response", "message", "error"):
                value = candidate.get(key)
                if isinstance(value, str):
                    return value
            return str(candidate)
        return candidate
    if isinstance(response_dict, str):
        return response_dict
    return str(response_dict) if response_dict else None


def _is_coherent(index: int, text: str) -> tuple[bool, str]:
    body = str(text or "").strip()
    if not body:
        return False, "empty"
    if ROLE_LEAK_RE.search(body):
        return False, "role_leak"
    if RECOVERY_RE.search(body):
        return False, "recovery_boilerplate"
    try:
        from core.phases.dialogue_policy import contains_corrupted_language

        if contains_corrupted_language(body):
            return False, "corrupted_language"
    except Exception:
        pass
    markers = EXPECTED_MARKERS[index]
    lower = body.lower()
    if not all(marker in lower for marker in markers):
        return False, f"missing_expected_markers:{markers}"
    return True, "ok"


async def main():
    # Use the same boot path as the CLI
    from aura_main import _boot_runtime_orchestrator

    print("=" * 60, flush=True)
    print("  SIMULATION: Booting Aura via canonical CLI path...", flush=True)
    print("=" * 60, flush=True)

    orchestrator = await _boot_runtime_orchestrator(ready_label="Simulate10")

    # Start the orchestrator's main loop in the background
    from core.utils.task_tracker import get_task_tracker
    main_task = get_task_tracker().create_task(orchestrator.run(), name="OrchestratorMainLoop")

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

            response_text = _extract_response_text(response_dict)

            ok, reason = _is_coherent(i, response_text or "")
            if ok:
                # Truncate for display
                display = response_text.strip()[:500]
                print(f"  Aura: {display}", flush=True)
                print(f"  ✅ [{elapsed:.1f}s]", flush=True)
                successes += 1
            else:
                print(f"  ❌ FAIL ({reason}) got: {response_dict}", flush=True)
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

    exit_code = 0 if successes == 10 else 1
    await _shutdown_runtime(orchestrator, main_task)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)


if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    asyncio.run(main())
