from __future__ import annotations
#!/usr/bin/env python3
"""Run a file-backed live Aura proof through the real orchestrator.

This harness exists for one purpose: prove that Aura herself can route,
authorize, execute, and complete representative tasks through her runtime
entrypoints instead of us doing them externally and calling it good.
"""

from core.utils.task_tracker import get_task_tracker
from core.runtime.atomic_writer import atomic_write_text

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env", override=False)
except Exception:
    pass


DESKTOP = Path.home() / "Desktop"
AGENCY_TEST_DIR = DESKTOP / "agency_test"
MANIFEST_DIR = DESKTOP / "Aura_Manifests"
TERMINAL_PROOF_PATH = AGENCY_TEST_DIR / "aura_terminal_runtime_proof.txt"
SNAKE_PATH = AGENCY_TEST_DIR / "aura_live_snake.html"
ARTIFACT_PATH = PROJECT_ROOT / "artifacts" / "aura_live_skill_probe_2026-04-21.json"

RESEARCH_QUERY = "Python 3.12 release notes key improvements"

SNAKE_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Aura Live Snake</title>
  <style>
    :root { color-scheme: light; }
    body {
      margin: 0;
      min-height: 100vh;
      display: grid;
      place-items: center;
      font-family: "Avenir Next", Helvetica, Arial, sans-serif;
      background: radial-gradient(circle at top, #f7efe2, #d7eef2 55%, #a8d0d6);
      color: #173042;
    }
    .card {
      width: min(92vw, 560px);
      padding: 24px;
      border-radius: 24px;
      background: rgba(255, 255, 255, 0.78);
      box-shadow: 0 20px 60px rgba(23, 48, 66, 0.18);
      backdrop-filter: blur(10px);
    }
    h1 { margin: 0 0 8px; font-size: 28px; }
    p { margin: 0 0 18px; line-height: 1.5; }
    canvas {
      display: block;
      width: 100%;
      aspect-ratio: 1 / 1;
      border-radius: 18px;
      background: linear-gradient(180deg, #0f3b4d, #173042);
      box-shadow: inset 0 0 0 1px rgba(255,255,255,0.12);
    }
    .meta {
      display: flex;
      justify-content: space-between;
      margin-top: 14px;
      font-weight: 700;
      letter-spacing: 0.04em;
      text-transform: uppercase;
      font-size: 12px;
    }
  </style>
</head>
<body>
  <div class="card">
    <h1>Aura Live Snake</h1>
    <p>Arrow keys move. Space restarts after game over. This file was written by Aura through her live file skill path.</p>
    <canvas id="board" width="420" height="420"></canvas>
    <div class="meta">
      <span id="score">Score: 0</span>
      <span id="state">Running</span>
    </div>
  </div>
  <script>
    const board = document.getElementById("board");
    const ctx = board.getContext("2d");
    const scoreEl = document.getElementById("score");
    const stateEl = document.getElementById("state");
    const size = 21;
    const cell = board.width / size;
    const dirs = {
      ArrowUp: { x: 0, y: -1 },
      ArrowDown: { x: 0, y: 1 },
      ArrowLeft: { x: -1, y: 0 },
      ArrowRight: { x: 1, y: 0 }
    };

    let snake;
    let dir;
    let nextDir;
    let food;
    let score;
    let over;

    function reset() {
      snake = [
        { x: 10, y: 10 },
        { x: 9, y: 10 },
        { x: 8, y: 10 }
      ];
      dir = { x: 1, y: 0 };
      nextDir = dir;
      score = 0;
      over = false;
      spawnFood();
      stateEl.textContent = "Running";
      scoreEl.textContent = "Score: 0";
    }

    function spawnFood() {
      do {
        food = {
          x: Math.floor(Math.random() * size),
          y: Math.floor(Math.random() * size)
        };
      } while (snake.some(part => part.x === food.x && part.y === food.y));
    }

    function tick() {
      if (over) {
        draw();
        return;
      }
      dir = nextDir;
      const head = { x: snake[0].x + dir.x, y: snake[0].y + dir.y };
      const hitWall = head.x < 0 || head.y < 0 || head.x >= size || head.y >= size;
      const hitSelf = snake.some(part => part.x === head.x && part.y === head.y);
      if (hitWall || hitSelf) {
        over = true;
        stateEl.textContent = "Game Over";
        draw();
        return;
      }
      snake.unshift(head);
      if (head.x === food.x && head.y === food.y) {
        score += 1;
        scoreEl.textContent = `Score: ${score}`;
        spawnFood();
      } else {
        snake.pop();
      }
      draw();
    }

    function draw() {
      ctx.clearRect(0, 0, board.width, board.height);
      for (let y = 0; y < size; y += 1) {
        for (let x = 0; x < size; x += 1) {
          ctx.fillStyle = (x + y) % 2 === 0 ? "#1a4b60" : "#143d50";
          ctx.fillRect(x * cell, y * cell, cell, cell);
        }
      }
      ctx.fillStyle = "#ffb347";
      ctx.beginPath();
      ctx.arc(food.x * cell + cell / 2, food.y * cell + cell / 2, cell * 0.35, 0, Math.PI * 2);
      ctx.fill();

      snake.forEach((part, index) => {
        ctx.fillStyle = index === 0 ? "#ffe066" : "#9be564";
        ctx.fillRect(part.x * cell + 2, part.y * cell + 2, cell - 4, cell - 4);
      });
    }

    window.addEventListener("keydown", event => {
      if (event.code === "Space" && over) {
        reset();
        draw();
        return;
      }
      const chosen = dirs[event.key];
      if (!chosen) return;
      if (chosen.x === -dir.x && chosen.y === -dir.y) return;
      nextDir = chosen;
    });

    reset();
    draw();
    setInterval(tick, 135);
  </script>
</body>
</html>
"""


def _json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return str(value)


def _trimmed_text(value: Any, limit: int = 500) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _summarize_result(result: Any) -> dict[str, Any]:
    if not isinstance(result, dict):
        return {"type": type(result).__name__, "value": _trimmed_text(result, 300)}
    summary = {
        "ok": bool(result.get("ok")),
        "summary": _trimmed_text(result.get("summary"), 300),
        "message": _trimmed_text(result.get("message"), 300),
        "error": _trimmed_text(result.get("error"), 300),
        "path": result.get("path"),
        "mode": result.get("mode"),
        "cached": result.get("cached"),
        "retained": result.get("retained"),
        "artifact_id": result.get("artifact_id"),
        "return_code": result.get("return_code"),
        "count": result.get("count"),
        "confidence": result.get("confidence"),
        "citations": len(result.get("citations") or []),
        "answer_preview": _trimmed_text(result.get("answer"), 500),
    }
    return {key: value for key, value in summary.items() if value not in ("", None)}


def _read_head(path: Path, limit: int = 240) -> str:
    if not path.exists():
        return ""
    return _trimmed_text(path.read_text(encoding="utf-8", errors="ignore"), limit)


async def _wait_for_service(name: str, timeout: float = 60.0) -> Any:
    from core.container import ServiceContainer

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        service = ServiceContainer.get(name, default=None)
        if service is not None:
            return service
        await asyncio.sleep(0.1)
    raise TimeoutError(f"Timed out waiting for service '{name}'")


def _install_capability_tracing(engine: Any, route_log: list[dict[str, Any]], exec_log: list[dict[str, Any]]):
    original_detect_intent = getattr(engine, "detect_intent", None)
    original_execute = getattr(engine, "execute", None)

    if callable(original_detect_intent):
        def traced_detect_intent(text: str):
            matched = list(original_detect_intent(text) or [])
            route_log.append({
                "input": _trimmed_text(text, 240),
                "matched_skills": matched,
                "timestamp": time.time(),
            })
            return matched

        engine.detect_intent = traced_detect_intent

    if callable(original_execute):
        async def traced_execute(skill_name: str, params: dict[str, Any], context: dict[str, Any] | None = None):
            started = time.perf_counter()
            result = await original_execute(skill_name, params, context)
            exec_log.append({
                "skill": skill_name,
                "params": _json_safe(params),
                "context_origin": (context or {}).get("origin"),
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
                "result": _summarize_result(result),
                "timestamp": time.time(),
            })
            return result

        engine.execute = traced_execute

    return original_detect_intent, original_execute


def _restore_capability_tracing(engine: Any, originals: tuple[Any, Any]) -> None:
    detect_intent, execute = originals
    if detect_intent is not None:
        engine.detect_intent = detect_intent
    if execute is not None:
        engine.execute = execute


async def main() -> int:
    from core.config import config
    from core.orchestrator import create_orchestrator

    os.environ.pop("AURA_SKIP_LLM", None)
    os.environ.pop("AURA_TEST_HARNESS", None)

    get_task_tracker().create_task(get_storage_gateway().create_dir(AGENCY_TEST_DIR, cause='main'))
    get_task_tracker().create_task(get_storage_gateway().create_dir(ARTIFACT_PATH.parent, cause='main'))

    if TERMINAL_PROOF_PATH.exists():
        get_task_tracker().create_task(get_storage_gateway().delete(TERMINAL_PROOF_PATH, cause='main'))
    if SNAKE_PATH.exists():
        get_task_tracker().create_task(get_storage_gateway().delete(SNAKE_PATH, cause='main'))

    get_task_tracker().create_task(get_storage_gateway().create_dir(MANIFEST_DIR, cause='main'))
    manifest_before = {path.resolve() for path in MANIFEST_DIR.iterdir() if path.is_file()}

    config.skeletal_mode = False

    orchestrator = create_orchestrator()
    route_log: list[dict[str, Any]] = []
    exec_log: list[dict[str, Any]] = []
    turns: list[dict[str, Any]] = []
    skills: list[dict[str, Any]] = []

    await orchestrator.start()
    run_task = get_task_tracker().create_task(orchestrator.run(), name="live_aura_skill_probe")

    try:
        capability_engine = await _wait_for_service("capability_engine", timeout=90.0)
        originals = _install_capability_tracing(capability_engine, route_log, exec_log)

        async def run_turn(label: str, prompt: str, timeout: float = 180.0) -> dict[str, Any]:
            route_index = len(route_log)
            exec_index = len(exec_log)
            started = time.perf_counter()
            response = await asyncio.wait_for(
                orchestrator.process_user_input(prompt, origin="user"),
                timeout=timeout,
            )
            entry = {
                "label": label,
                "kind": "user_turn",
                "prompt": prompt,
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
                "response_preview": _trimmed_text(response, 700),
                "routes": route_log[route_index:],
                "executions": exec_log[exec_index:],
            }
            turns.append(entry)
            return entry

        async def run_skill(label: str, skill_name: str, params: dict[str, Any], *, timeout: float = 180.0) -> dict[str, Any]:
            exec_index = len(exec_log)
            started = time.perf_counter()
            result = await asyncio.wait_for(
                orchestrator.agency.execute_skill(
                    skill_name,
                    params,
                    {
                        "origin": "user",
                        "objective": label,
                        "message": label,
                        "intent_source": "live_aura_skill_probe",
                    },
                ),
                timeout=timeout,
            )
            entry = {
                "label": label,
                "kind": "agency_skill",
                "skill": skill_name,
                "params": _json_safe(params),
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
                "result": _summarize_result(result),
                "executions": exec_log[exec_index:],
            }
            skills.append(entry)
            return entry

        await run_turn(
            "terminal_proof_write",
            (
                "execute: mkdir -p /Users/bryan/Desktop/agency_test && "
                "printf 'AURA_RUNTIME_PROOF\\n' > "
                "/Users/bryan/Desktop/agency_test/aura_terminal_runtime_proof.txt"
            ),
        )

        await run_skill(
            "write_snake_game_to_desktop",
            "file_operation",
            {
                "action": "write",
                "path": str(SNAKE_PATH),
                "content": SNAKE_HTML,
            },
        )

        await run_turn(
            "snake_exists_check",
            "check if /Users/bryan/Desktop/agency_test/aura_live_snake.html exists",
        )

        await run_turn(
            "manifest_remote_asset",
            "save to my desktop: https://httpbin.org/image/png",
        )

        await run_turn(
            "natural_language_research",
            "research about Python 3.12 release notes key improvements",
            timeout=240.0,
        )

        await run_skill(
            "deep_research_with_retention",
            "web_search",
            {
                "query": RESEARCH_QUERY,
                "deep": True,
                "num_results": 6,
                "retain": True,
                "force_refresh": True,
            },
            timeout=300.0,
        )

        await run_skill(
            "cached_research_reuse",
            "web_search",
            {
                "query": RESEARCH_QUERY,
                "deep": False,
                "num_results": 3,
            },
            timeout=180.0,
        )

        new_manifests = sorted(
            str(path)
            for path in MANIFEST_DIR.iterdir()
            if path.is_file() and path.resolve() not in manifest_before
        )

        proof = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "cwd": str(PROJECT_ROOT),
            "desktop": str(DESKTOP),
            "artifact_path": str(ARTIFACT_PATH),
            "turns": turns,
            "skills": skills,
            "terminal_file": {
                "path": str(TERMINAL_PROOF_PATH),
                "exists": TERMINAL_PROOF_PATH.exists(),
                "content_head": _read_head(TERMINAL_PROOF_PATH),
            },
            "snake_game": {
                "path": str(SNAKE_PATH),
                "exists": SNAKE_PATH.exists(),
                "size_bytes": SNAKE_PATH.stat().st_size if SNAKE_PATH.exists() else 0,
                "content_head": _read_head(SNAKE_PATH),
            },
            "manifest_outputs": new_manifests,
            "research": {
                "query": RESEARCH_QUERY,
                "retained_exec": next(
                    (
                        entry["executions"][-1]
                        for entry in skills
                        if entry["label"] == "deep_research_with_retention" and entry["executions"]
                    ),
                    None,
                ),
                "latest_exec": next(
                    (
                        entry["executions"][-1]
                        for entry in reversed(skills)
                        if entry["skill"] == "web_search" and entry["executions"]
                    ),
                    None,
                ),
                "cached_exec": next(
                    (
                        entry["executions"][-1]
                        for entry in skills
                        if entry["label"] == "cached_research_reuse" and entry["executions"]
                    ),
                    None,
                ),
            },
            "all_routes": route_log,
            "all_skill_executions": exec_log,
        }

        proof["checks"] = {
            "terminal_write_ok": proof["terminal_file"]["exists"]
            and "AURA_RUNTIME_PROOF" in proof["terminal_file"]["content_head"],
            "snake_written_ok": proof["snake_game"]["exists"] and proof["snake_game"]["size_bytes"] > 1000,
            "manifest_created_ok": bool(new_manifests),
            "research_retained_ok": bool(
                proof["research"]["retained_exec"]
                and proof["research"]["retained_exec"]["result"].get("retained")
            ),
            "research_cached_reuse_ok": bool(
                proof["research"]["cached_exec"]
                and proof["research"]["cached_exec"]["result"].get("mode") == "cached"
                and proof["research"]["cached_exec"]["result"].get("cached") is True
            ),
        }
        proof["overall_ok"] = all(proof["checks"].values())

        atomic_write_text(ARTIFACT_PATH, json.dumps(_json_safe(proof), indent=2), encoding="utf-8")
        print(json.dumps(_json_safe(proof["checks"]), indent=2))
        print(f"artifact={ARTIFACT_PATH}")

        _restore_capability_tracing(capability_engine, originals)
        return 0 if proof["overall_ok"] else 1
    finally:
        try:
            capability_engine = locals().get("capability_engine")
            originals = locals().get("originals")
            if capability_engine is not None and originals is not None:
                _restore_capability_tracing(capability_engine, originals)
        except Exception:
            pass

        run_task.cancel()
        try:
            await run_task
        except asyncio.CancelledError:
            pass
        await orchestrator.stop()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
