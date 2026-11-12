"""Lose her microphone in the middle of a question, get it back, ask again, and see what she heard.

    python tools/voice/a_device_change.py "Hey Aura, what is the capital of" "What's two plus two?"

Her window reports each capture change with a generation number (interface/static/voice_mode.js, a removed input
or a dead track reopens capture). This says what it says: the first words stream as her microphone would, then
"recovering" mid-question, a second of nothing, "active" under the next generation, and the second words. Printed:
her receipts for each change, what she heard, and what she said. Passing means she heard only the second words.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "voice"))

from a_spoken_turn import CHUNK_S, RATE_IN, spoken  # noqa: E402


async def changed(args: argparse.Namespace) -> int:
    import websockets

    lost, asked = spoken(args.lost, args.voice), spoken(args.asked, args.voice)
    step = int(RATE_IN * CHUNK_S)
    silence = np.zeros(step, dtype=np.int16)
    url = f"ws://{args.host}/ws/voice?mode=focused"
    events: list[tuple[float, dict]] = []
    async with websockets.connect(url, origin=f"http://{args.host}", max_size=None) as ws:
        while json.loads(await asyncio.wait_for(ws.recv(), timeout=args.ready_s)).get("type") != "voice.ready":
            pass
        await ws.send(json.dumps({"command": "set_floor", "open": True}))

        async def device(state: str, generation: int, reason: str) -> None:
            await ws.send(json.dumps({"command": "device_state", "state": state, "generation": generation,
                                      "reason": reason}))

        async def stream(samples: np.ndarray) -> None:
            for start in range(0, len(samples), step):
                await ws.send(samples[start:start + step].tobytes())
                await asyncio.sleep(CHUNK_S)

        async def microphone() -> None:
            await device("active", 1, "opened")
            await stream(lost)
            await device("recovering", 2, "active_input_removed")
            await asyncio.sleep(1.0)
            await device("active", 3, "recovered:active_input_removed")
            await stream(silence.repeat(25))
            await stream(asked)
            while True:
                await stream(silence)

        mic = asyncio.create_task(microphone())
        started = time.monotonic()
        try:
            while time.monotonic() - started < args.wait_s:
                try:
                    message = await asyncio.wait_for(ws.recv(), timeout=0.5)
                except TimeoutError:
                    continue
                if isinstance(message, str):
                    data = json.loads(message)
                    events.append((time.monotonic() - started, data))
                    if data.get("type") == "voice.metrics":
                        break
        finally:
            mic.cancel()
    receipts = [d for _, d in events if d.get("type") == "voice.device"]
    for receipt in receipts:
        print(f"receipt: generation {receipt.get('generation')} {receipt.get('state')} "
              f"({receipt.get('reason')}), capture available {receipt.get('capture_available')}")
    heard = [d.get("text") for _, d in events if d.get("type") == "voice.final"]
    said = [d.get("text") for _, d in events if d.get("type") == "voice.reply"]
    print(f"she heard: {heard}")
    print(f"she said: {said[-1][:300] if said else 'nothing'}")
    def words(text: object) -> set[str]:
        return {word.strip(".,?!'\"").lower() for word in str(text).split()}

    lost_words = {word for word in words(args.lost) if len(word) >= 5} - words(args.asked)  # its distinct words
    clean = bool(heard) and not any(lost_words & words(text) for text in heard)
    print("she heard only the words after the change" if clean else "the lost words reached her, or nothing did")
    return 0 if clean and said and len(receipts) == 3 else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("lost")
    parser.add_argument("asked")
    parser.add_argument("--voice", default="Samantha")
    parser.add_argument("--host", default="127.0.0.1:8000")
    parser.add_argument("--ready-s", type=float, default=240.0)
    parser.add_argument("--wait-s", type=float, default=180.0)
    return asyncio.run(changed(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
