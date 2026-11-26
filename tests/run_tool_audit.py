################################################################################


import asyncio
import logging
import sys
import os

sys.path.append(os.getcwd())

from core.capability_engine import CapabilityEngine
from core.audits.tool_auditor import ToolAuditor
from core.brain.cognitive_engine import CognitiveEngine
from core.event_bus import get_event_bus
from core.utils.task_tracker import get_task_tracker

logging.basicConfig(level=logging.INFO)


_HARNESS_SHUTDOWN_ERRORS = (
    AttributeError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
)


async def _shutdown_harness_runtime() -> list[str]:
    errors: list[str] = []
    try:
        await get_event_bus().shutdown()
    except _HARNESS_SHUTDOWN_ERRORS as exc:
        errors.append(f"event_bus_shutdown:{type(exc).__name__}:{exc}")

    try:
        await get_task_tracker().shutdown(timeout=1.0)
    except _HARNESS_SHUTDOWN_ERRORS as exc:
        errors.append(f"task_tracker_shutdown:{type(exc).__name__}:{exc}")
    return errors


async def main():
    exit_code = 1
    try:
        print("Initializing Aura Tool Audit...")

        # Setup
        brain = CognitiveEngine()
        capability_engine = CapabilityEngine()
        print("✓ CognitiveEngine + CapabilityEngine initialized")

        auditor = ToolAuditor(brain, capability_engine=capability_engine)
    
        print("\nRunning Tool Selection Suite...")
        # Add suite run to ToolAuditor
        results = await auditor.run_suite()
    
        print("\nAudit Results:")
        print(f"Score: {results['score']}/{results['total']}")
        for r in results['details']:
            status = "PASS" if r['success'] else "FAIL"
            print(f"{status} | Q: {r['query'][:30]}... -> Tool: {r['selected_tool']} (Exp: {r['expected']})")
        
        if results['score'] == results['total']:
            print("\nALL TESTS PASSED.")
            exit_code = 0
        else:
            print("\nSOME TESTS FAILED.")
            exit_code = 1
    finally:
        shutdown_errors = await _shutdown_harness_runtime()
        if shutdown_errors:
            print("\nHARNESS SHUTDOWN ERRORS:")
            for err in shutdown_errors:
                print(f"  - {err}")
            if exit_code == 0:
                exit_code = 3
    return exit_code

if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))


##
