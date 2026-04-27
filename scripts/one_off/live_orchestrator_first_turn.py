#!/usr/bin/env python3
"""Manual harness for a single skeletal orchestrator user turn."""

from core.utils.task_tracker import get_task_tracker
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env", override=False)
except Exception:
    pass


def _dump(label: str, payload: object) -> None:
    print(f"\n=== {label} ===")
    print(json.dumps(payload, indent=2, default=str))


async def main() -> int:
    from core.config import config
    from core.container import ServiceContainer
    from core.orchestrator import create_orchestrator

    prompt = "Reply with exactly OK_KERNEL and nothing else."
    config.skeletal_mode = True

    orchestrator = create_orchestrator()
    await orchestrator.start()
    run_task = get_task_tracker().create_task(orchestrator.run(), name="live_orchestrator_first_turn")
    try:
        result = await orchestrator._process_message(prompt)
        gate = ServiceContainer.get("inference_gate", default=None)
        router = ServiceContainer.get("llm_router", default=None)
        _dump("result", result)
        if gate and hasattr(gate, "get_conversation_status"):
            _dump("gate", gate.get_conversation_status())
        if router and hasattr(router, "get_health_report"):
            report = router.get_health_report()
            slim = {
                "foreground_endpoint": report.get("foreground_endpoint"),
                "foreground_tier": report.get("foreground_tier"),
                "last_user_error": report.get("last_user_error"),
                "lane_audit_ok": report.get("lane_audit_ok"),
                "lane_audit_issues": report.get("lane_audit_issues"),
            }
            _dump("router", slim)
    finally:
        run_task.cancel()
        try:
            await run_task
        except asyncio.CancelledError:
            pass
        await orchestrator.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
