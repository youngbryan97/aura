#!/usr/bin/env python3
"""Live Aura runtime probe.

This is intentionally not a unit test. It attaches to a running Aura server
and checks that live surfaces do real work:

* HTTP health/readiness responds.
* WebSocket telemetry/neural/action events arrive while probes run.
* `/api/skill/execute` drives governed skills instead of dead buttons.
* Chat can trigger Aura's own coding/file skills to create a runnable artifact.
* Chat maintains continuity on a novel topic without reset boilerplate.
* Computer-use can perform a safe local app action through Aura's skill body.

Exit code 0 means the live runtime met the bar. Any degraded, canned,
blank, reset, or no-effect response is a failure.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import websockets


BANNED_REPLY_RE = re.compile(
    r"(say that again|try (?:again|me again|that again)|ask me again|"
    r"give me a moment|i'?m with you|could you repeat|repeat your question|"
    r"send your message again|lost my (?:thread|train of thought)|"
    r"hit a bump|one moment|how can i help|as an ai|i am an ai)",
    re.IGNORECASE,
)


@dataclass
class ProbeResult:
    name: str
    ok: bool
    detail: str
    elapsed_s: float = 0.0
    data: dict[str, Any] = field(default_factory=dict)


class LiveRuntimeProbe:
    def __init__(self, base_url: str, *, timeout_s: float = 420.0, probe_timeout_s: float | None = None):
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.probe_timeout_s = float(probe_timeout_s or min(timeout_s, 180.0))
        self.events: list[dict[str, Any]] = []
        self.results: list[ProbeResult] = []
        self.headers: dict[str, str] = {}
        token = os.environ.get("AURA_API_TOKEN", "").strip()
        if token:
            self.headers["X-Api-Token"] = token

    async def run(self) -> int:
        async with httpx.AsyncClient(timeout=self.timeout_s, headers=self.headers) as client:
            self.client = client
            ws_task = asyncio.create_task(self._collect_ws_events(), name="live-probe-ws")
            try:
                await self._probe("health", self._health)
                await self._probe("skill_button_file_write", self._skill_button_file_write)
                await self._probe("chat_coding_snake", self._chat_coding_snake)
                await self._probe("novel_topic_continuity", self._novel_topic_continuity)
                await self._probe("computer_use_local_app", self._computer_use_local_app)
                await self._probe("telemetry_neural_stream", self._telemetry_neural_stream)
            finally:
                ws_task.cancel()
                try:
                    await ws_task
                except asyncio.CancelledError:
                    pass

        self._print_summary()
        return 0 if all(r.ok for r in self.results) else 1

    async def _probe(self, name: str, fn) -> None:
        start = time.monotonic()
        try:
            detail, data = await asyncio.wait_for(fn(), timeout=self.probe_timeout_s)
            self.results.append(ProbeResult(name, True, detail, time.monotonic() - start, data or {}))
        except asyncio.TimeoutError:
            self.results.append(ProbeResult(name, False, f"TimeoutError: exceeded {self.probe_timeout_s:.0f}s", time.monotonic() - start))
        except Exception as exc:
            self.results.append(ProbeResult(name, False, f"{type(exc).__name__}: {exc}", time.monotonic() - start))

    async def _get(self, path: str) -> dict[str, Any]:
        response = await self.client.get(f"{self.base_url}{path}")
        response.raise_for_status()
        return response.json()

    async def _post(self, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        response = await self.client.post(f"{self.base_url}{path}", json=payload or {})
        response.raise_for_status()
        return response.json()

    async def _chat(self, message: str) -> dict[str, Any]:
        response = await self._post("/api/chat", {"message": message})
        reply = str(response.get("response") or "").strip()
        if not reply:
            raise AssertionError(f"blank chat reply for: {message[:80]}")
        if BANNED_REPLY_RE.search(reply):
            raise AssertionError(f"reset/canned reply detected: {reply[:240]}")
        if str(response.get("status", "")).lower() in {"timeout", "error", "conversation_unavailable"}:
            raise AssertionError(f"degraded chat status={response.get('status')} reply={reply[:240]}")
        if str(response.get("response_confidence", "")).lower() == "degraded":
            raise AssertionError(f"degraded chat confidence reply={reply[:240]}")
        return response

    async def _skill(self, skill_name: str, params: dict[str, Any]) -> dict[str, Any]:
        return await self._post(f"/api/skill/execute?skill_name={skill_name}", params)

    async def _collect_ws_events(self) -> None:
        ws_url = self.base_url.replace("http://", "ws://").replace("https://", "wss://") + "/ws"
        while True:
            try:
                async with websockets.connect(ws_url, ping_interval=10, ping_timeout=20) as ws:
                    await ws.send(json.dumps({"type": "ping"}))
                    async for raw in ws:
                        try:
                            event = json.loads(raw)
                        except Exception:
                            event = {"type": "raw", "content": str(raw)[:500]}
                        self.events.append(event)
                        if len(self.events) > 1000:
                            self.events = self.events[-1000:]
            except asyncio.CancelledError:
                raise
            except Exception:
                await asyncio.sleep(1.0)

    async def _health(self) -> tuple[str, dict[str, Any]]:
        boot = await self._get("/api/health/boot")
        health = await self._get("/api/health")
        bootstrap = await self._get("/api/ui/bootstrap")
        if not isinstance(boot, dict) or not isinstance(health, dict):
            raise AssertionError("health endpoints did not return JSON objects")
        if "conversation" not in bootstrap:
            raise AssertionError("ui bootstrap missing conversation payload")
        return "health, boot, and UI bootstrap responded", {"boot": boot, "health": health}

    async def _skill_button_file_write(self) -> tuple[str, dict[str, Any]]:
        marker = f"live button probe {int(time.time())}"
        path = "artifacts/live_runtime/button_probe.txt"
        write = await self._skill(
            "file_operation",
            {"action": "write", "path": path, "content": marker},
        )
        if not write.get("ok"):
            raise AssertionError(f"file_operation write failed: {write}")
        read = await self._skill("file_operation", {"action": "read", "path": path})
        if not read.get("ok") or marker not in str(read.get("content", "")):
            raise AssertionError(f"file_operation read did not verify write: {read}")
        return "skill button path wrote and read a real file", {"path": path, "write": write}

    async def _chat_coding_snake(self) -> tuple[str, dict[str, Any]]:
        path = "artifacts/live_runtime/generated/live_snake.html"
        target = Path(path)
        if target.exists():
            target.unlink()
        response = await self._chat(
            "Create a simple game of Snake and save it as "
            f"{path}. Use your own live coding and file tools; don't just describe it."
        )
        if not target.exists():
            raise AssertionError(f"chat did not create {path}; reply={response.get('response')[:300]}")
        content = target.read_text(encoding="utf-8", errors="replace")
        required = ("<canvas", "function tick", "addEventListener", "Score")
        missing = [needle for needle in required if needle not in content]
        if missing:
            raise AssertionError(f"snake artifact missing {missing}")
        return "chat-created Snake artifact exists and is runnable HTML", {
            "path": str(target.resolve()),
            "bytes": len(content.encode("utf-8")),
            "reply": response.get("response"),
        }

    async def _novel_topic_continuity(self) -> tuple[str, dict[str, Any]]:
        first = await self._chat(
            "Novel-topic check: invent a tiny discipline called glass arithmetic. "
            "Give it two rules and one example, naturally."
        )
        second = await self._chat(
            "Stay with glass arithmetic. Add one limitation and connect it to the example you just gave."
        )
        r1 = str(first.get("response") or "")
        r2 = str(second.get("response") or "")
        if "glass" not in r1.lower() or "glass" not in r2.lower():
            raise AssertionError(f"conversation lost the novel topic: {r1[:160]} / {r2[:160]}")
        if len(set(r1.lower().split()) & set(r2.lower().split())) < 4:
            raise AssertionError("follow-up had weak continuity with prior answer")
        return "novel topic remained coherent across chained turns", {"first": r1, "second": r2}

    async def _computer_use_local_app(self) -> tuple[str, dict[str, Any]]:
        opened = await self._skill("computer_use", {"action": "open_app", "target": "Calculator"})
        if not opened.get("ok"):
            raise AssertionError(f"computer_use open_app failed: {opened}")
        clock = await self._skill("computer_use", {"action": "read_menu_clock", "target": ""})
        if not clock.get("ok"):
            raise AssertionError(f"computer_use read_menu_clock failed: {clock}")
        return "computer_use opened a local app and read a live macOS surface", {
            "opened": opened,
            "clock": clock,
        }

    async def _telemetry_neural_stream(self) -> tuple[str, dict[str, Any]]:
        await asyncio.sleep(2.0)
        event_types = [str(e.get("type") or "") for e in self.events]
        interesting = [
            e
            for e in self.events
            if str(e.get("type") or "") in {
                "neural_event",
                "telemetry",
                "action_result",
                "tool_execution",
                "activity",
                "thought",
                "chat_stream_chunk",
                "aura_message",
            }
        ]
        if not interesting:
            raise AssertionError(f"no relevant websocket telemetry received; event_types={event_types[-20:]}")
        decoded = json.dumps(self.events[-80:], ensure_ascii=False)
        if "Thought Decoded" in decoded:
            raise AssertionError("legacy Thought Decoded telemetry appeared in live stream")
        return "websocket emitted live telemetry/action/neural events", {
            "event_count": len(self.events),
            "recent_types": event_types[-40:],
        }

    def _print_summary(self) -> None:
        print("\nLIVE RUNTIME PROBE SUMMARY")
        print("=" * 72)
        for result in self.results:
            mark = "PASS" if result.ok else "FAIL"
            print(f"[{mark}] {result.name} ({result.elapsed_s:.1f}s): {result.detail}")
        print(f"events_collected={len(self.events)}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--timeout", type=float, default=420.0)
    parser.add_argument("--probe-timeout", type=float, default=None)
    args = parser.parse_args()
    return asyncio.run(LiveRuntimeProbe(args.base_url, timeout_s=args.timeout, probe_timeout_s=args.probe_timeout).run())


if __name__ == "__main__":
    raise SystemExit(main())
