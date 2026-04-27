from __future__ import annotations
#!/usr/bin/env python3
"""Capture the real UnitaryResponse Cortex payload and replay it directly."""

from core.utils.task_tracker import get_task_tracker

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
    from core.utils.deadlines import get_deadline

    prompt = "Reply with exactly OK_KERNEL and nothing else."
    config.skeletal_mode = True

    orchestrator = create_orchestrator()
    await orchestrator.start()
    run_task = get_task_tracker().create_task(
        orchestrator.run(),
        name="live_unitary_local_probe",
    )
    captured: dict[str, object] = {}
    try:
        router = None
        for _ in range(200):
            router = ServiceContainer.get("llm_router", default=None)
            if router is not None:
                break
            await asyncio.sleep(0.1)
        if router is None:
            raise RuntimeError("llm_router unavailable")

        original_call_endpoint = router._call_endpoint

        async def traced_call_endpoint(ep, prompt, system_prompt, timeout, schema=None, **kwargs):
            if getattr(ep, "name", "") == "Cortex" and getattr(ep, "is_local", False):
                captured.update(
                    {
                        "prompt": prompt,
                        "system_prompt": system_prompt,
                        "timeout": timeout,
                        "schema": schema,
                        "kwargs": dict(kwargs),
                        "endpoint_model": getattr(ep, "model", ""),
                        "client_type": ep.client.__class__.__name__ if getattr(ep, "client", None) else "",
                        "client": ep.client,
                    }
                )
                messages = kwargs.get("messages")
                _dump(
                    "captured_cortex_request",
                    {
                        "prompt_len": len(str(prompt or "")),
                        "system_prompt_len": len(str(system_prompt or "")),
                        "timeout": timeout,
                        "prefer_tier": kwargs.get("prefer_tier"),
                        "origin": kwargs.get("origin"),
                        "purpose": kwargs.get("purpose"),
                        "is_background": kwargs.get("is_background"),
                        "allow_cloud_fallback": kwargs.get("allow_cloud_fallback"),
                        "message_count": len(messages) if isinstance(messages, list) else None,
                        "message_chars": sum(
                            len(str(msg.get("content", "") or ""))
                            for msg in messages
                            if isinstance(msg, dict)
                        )
                        if isinstance(messages, list)
                        else None,
                        "first_roles": [msg.get("role") for msg in messages[:6]] if isinstance(messages, list) else None,
                        "first_system_head": (
                            str(messages[0].get("content", "") or "")[:800]
                            if isinstance(messages, list) and messages
                            else ""
                        ),
                        "last_user_tail": (
                            str(messages[-1].get("content", "") or "")[-300:]
                            if isinstance(messages, list) and messages
                            else ""
                        ),
                    },
                )
            return await original_call_endpoint(ep, prompt, system_prompt, timeout, schema=schema, **kwargs)

        router._call_endpoint = traced_call_endpoint

        result = await orchestrator._process_message(prompt)
        _dump("orchestrator_result", result)

        client = captured.get("client")
        if client is not None:
            call_kwargs = dict(captured.get("kwargs", {}) or {})
            replay_messages = call_kwargs.get("messages")
            replay = await client.generate_text_async(
                str(captured.get("prompt") or ""),
                system_prompt=str(captured.get("system_prompt") or ""),
                messages=replay_messages,
                foreground_request=True,
                owner_label="live_unitary_replay",
                deadline=get_deadline(120.0),
                max_tokens=call_kwargs.get("max_tokens", 64),
                temperature=call_kwargs.get("temperature"),
                temp=call_kwargs.get("temp"),
                top_p=call_kwargs.get("top_p"),
                schema=captured.get("schema"),
            )
            lane_status = client.get_lane_status() if hasattr(client, "get_lane_status") else {}
            _dump(
                "direct_local_replay",
                {
                    "result": replay,
                    "lane_status": lane_status,
                },
            )
        else:
            _dump("direct_local_replay", {"error": "no captured cortex client"})
    finally:
        try:
            if "router" in locals() and router is not None and "original_call_endpoint" in locals():
                router._call_endpoint = original_call_endpoint
        except Exception:
            pass
        run_task.cancel()
        try:
            await run_task
        except asyncio.CancelledError:
            pass
        await orchestrator.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
