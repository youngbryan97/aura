#!/usr/bin/env python3
"""Manual harness for the embedded MLX first-turn failure.

Runs a few real foreground probes in a fresh process so we can compare:
1. direct raw-prompt generation
2. direct message-based generation
3. a gate-shaped first-turn payload

This is intentionally a manual/live tool, not a unit test.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
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


async def _run_probe(
    client,
    *,
    label: str,
    prompt: str,
    messages: list[dict[str, str]] | None = None,
    max_tokens: int = 48,
) -> None:
    from core.utils.deadlines import get_deadline

    started = time.perf_counter()
    result = await client.generate_text_async(
        prompt,
        messages=messages,
        max_tokens=max_tokens,
        foreground_request=True,
        owner_label=f"live_probe:{label}",
        deadline=get_deadline(90.0),
    )
    elapsed = time.perf_counter() - started
    _dump(
        label,
        {
            "elapsed_s": round(elapsed, 2),
            "result": result,
            "lane_status": client.get_lane_status(),
        },
    )


async def main() -> int:
    from core.brain.inference_gate import InferenceGate
    from core.brain.llm.mlx_client import get_mlx_client
    from core.brain.llm.model_registry import ACTIVE_MODEL, get_local_backend, get_runtime_model_path

    model_path = str(get_runtime_model_path(ACTIVE_MODEL))
    client = get_mlx_client(model_path=model_path)

    _dump(
        "runtime",
        {
            "backend": get_local_backend(),
            "active_model": ACTIVE_MODEL,
            "model_path": model_path,
            "lane_status": client.get_lane_status(),
        },
    )

    await client.warmup()
    _dump("post_warmup", client.get_lane_status())

    await _run_probe(
        client,
        label="raw_prompt",
        prompt="Reply with exactly OK_RAW and nothing else.",
        max_tokens=24,
    )

    await _run_probe(
        client,
        label="simple_messages",
        prompt="",
        messages=[
            {"role": "system", "content": "Reply with exactly OK_MESSAGES and nothing else."},
            {"role": "user", "content": "Reply with exactly OK_MESSAGES."},
        ],
        max_tokens=24,
    )

    gate = InferenceGate()
    system_prompt = gate._build_compact_system_prompt("Reply with exactly OK_GATE.")
    gate_messages = gate._build_compact_messages(
        "Reply with exactly OK_GATE.",
        system_prompt,
        history=[],
    )
    _dump(
        "gate_payload",
        {
            "message_count": len(gate_messages),
            "roles": [msg.get("role", "") for msg in gate_messages],
            "system_prompt_len": len(system_prompt),
        },
    )
    await _run_probe(
        client,
        label="gate_messages",
        prompt="",
        messages=gate_messages,
        max_tokens=48,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
