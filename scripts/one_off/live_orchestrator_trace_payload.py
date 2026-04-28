from __future__ import annotations
#!/usr/bin/env python3
"""Trace the real unitary-response payload for a single orchestrator turn."""

from core.utils.task_tracker import get_task_tracker
from core.runtime.atomic_writer import atomic_write_text

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


TRACE_PATH = Path("/tmp/aura_live_trace.jsonl")


def _message_chars(messages: object) -> int | None:
    if not isinstance(messages, list):
        return None
    return sum(len(str(msg.get("content", "") or "")) for msg in messages if isinstance(msg, dict))


def _append_trace(event: str, payload: dict) -> None:
    record = {"event": event, **payload}
    get_task_tracker().create_task(get_storage_gateway().create_dir(TRACE_PATH.parent, cause='_append_trace'))
    with TRACE_PATH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, default=str) + "\n")


async def main() -> int:
    from core.config import config
    from core.container import ServiceContainer
    from core.brain.inference_gate import InferenceGate
    from core.brain.llm.mlx_client import MLXLocalClient
    from core.brain.llm_health_router import HealthAwareLLMRouter
    from core.orchestrator import create_orchestrator

    prompt = "Reply with exactly OK_KERNEL and nothing else."
    config.skeletal_mode = True
    atomic_write_text(TRACE_PATH, "")

    original_router_think = HealthAwareLLMRouter.think
    original_gate_think = InferenceGate.think
    original_gate_generate = InferenceGate._generate_with_client
    original_client_generate = MLXLocalClient.generate_text_async

    async def traced_router_think(self, prompt=None, system_prompt=None, **kwargs):
        messages = kwargs.get("messages")
        if kwargs.get("purpose") in {"reply", "expression"} or kwargs.get("origin") in {"user", "voice"}:
            _append_trace(
                "router_think",
                {
                    "prompt_len": len(str(prompt or "")),
                    "system_prompt_len": len(str(system_prompt or "")),
                    "message_count": len(messages) if isinstance(messages, list) else None,
                    "message_chars": _message_chars(messages),
                    "skip_runtime_payload": kwargs.get("skip_runtime_payload"),
                    "prefer_tier": kwargs.get("prefer_tier"),
                    "origin": kwargs.get("origin"),
                    "purpose": kwargs.get("purpose"),
                    "is_background": kwargs.get("is_background"),
                    "first_roles": [msg.get("role") for msg in messages[:6]] if isinstance(messages, list) else None,
                    "first_system_head": (
                        str(messages[0].get("content", "") or "")[:600]
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
        return await original_router_think(self, prompt, system_prompt=system_prompt, **kwargs)

    async def traced_gate_think(self, prompt, system_prompt="", **kwargs):
        messages = kwargs.get("messages")
        if kwargs.get("origin") in {"user", "voice"} or kwargs.get("foreground_request"):
            _append_trace(
                "gate_think",
                {
                    "prompt_len": len(str(prompt or "")),
                    "system_prompt_len": len(str(system_prompt or "")),
                    "message_count": len(messages) if isinstance(messages, list) else None,
                    "message_chars": _message_chars(messages),
                    "origin": kwargs.get("origin"),
                    "prefer_tier": kwargs.get("prefer_tier"),
                    "foreground_request": kwargs.get("foreground_request"),
                },
            )
        return await original_gate_think(self, prompt, system_prompt=system_prompt, **kwargs)

    async def traced_generate_with_client(
        self,
        client,
        prompt,
        system_prompt,
        history,
        deadline,
        label,
        **kwargs,
    ):
        messages = kwargs.get("messages")
        if kwargs.get("foreground_request"):
            _append_trace(
                "gate_generate_with_client",
                {
                    "label": label,
                    "prompt_len": len(str(prompt or "")),
                    "system_prompt_len": len(str(system_prompt or "")),
                    "history_len": len(history or []),
                    "message_count": len(messages) if isinstance(messages, list) else None,
                    "message_chars": _message_chars(messages),
                    "max_tokens": kwargs.get("max_tokens"),
                    "temperature": kwargs.get("temperature"),
                    "foreground_request": kwargs.get("foreground_request"),
                    "roles": [msg.get("role") for msg in messages[:8]] if isinstance(messages, list) else None,
                },
            )
        result = await original_gate_generate(
            self,
            client,
            prompt,
            system_prompt,
            history,
            deadline,
            label,
            **kwargs,
        )
        if kwargs.get("foreground_request"):
            _append_trace(
                "gate_generate_result",
                {
                    "label": label,
                    "result": result,
                },
            )
        return result

    async def traced_client_generate(self, prompt, **kwargs):
        messages = kwargs.get("messages")
        if kwargs.get("foreground_request"):
            _append_trace(
                "mlx_generate_text_async",
                {
                    "prompt_len": len(str(prompt or "")),
                    "message_count": len(messages) if isinstance(messages, list) else None,
                    "message_chars": _message_chars(messages),
                    "max_tokens": kwargs.get("max_tokens"),
                    "owner_label": kwargs.get("owner_label"),
                },
            )
        result = await original_client_generate(self, prompt, **kwargs)
        if kwargs.get("foreground_request"):
            _append_trace(
                "mlx_generate_result",
                {
                    "owner_label": kwargs.get("owner_label"),
                    "result": result,
                    "lane_status": self.get_lane_status(),
                },
            )
        return result

    HealthAwareLLMRouter.think = traced_router_think
    InferenceGate.think = traced_gate_think
    InferenceGate._generate_with_client = traced_generate_with_client
    MLXLocalClient.generate_text_async = traced_client_generate

    orchestrator = create_orchestrator()
    await orchestrator.start()
    run_task = get_task_tracker().create_task(
        orchestrator.run(),
        name="live_orchestrator_trace_payload",
    )
    try:
        result = await orchestrator._process_message(prompt)
        router = ServiceContainer.get("llm_router", default=None)
        gate = ServiceContainer.get("inference_gate", default=None)
        _dump("result", result)
        if gate and hasattr(gate, "get_conversation_status"):
            _dump("gate", gate.get_conversation_status())
        if router and hasattr(router, "get_health_report"):
            report = router.get_health_report()
            _dump(
                "router",
                {
                    "foreground_endpoint": report.get("foreground_endpoint"),
                    "foreground_tier": report.get("foreground_tier"),
                    "last_user_error": report.get("last_user_error"),
                    "lane_audit_ok": report.get("lane_audit_ok"),
                    "lane_audit_issues": report.get("lane_audit_issues"),
                },
            )
        if TRACE_PATH.exists():
            trace_lines = [json.loads(line) for line in TRACE_PATH.read_text().splitlines() if line.strip()]
            _dump("trace", trace_lines[-20:])
    finally:
        run_task.cancel()
        try:
            await run_task
        except asyncio.CancelledError:
            pass
        await orchestrator.stop()
        HealthAwareLLMRouter.think = original_router_think
        InferenceGate.think = original_gate_think
        InferenceGate._generate_with_client = original_gate_generate
        MLXLocalClient.generate_text_async = original_client_generate
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
