"""Kill one of her voice's worker processes while she answers, then ask again in the same session.

    python tools/voice/a_worker_dies.py original_speech_worker.py "Tell me about your morning." "What's two plus two?"

``--after-s`` seconds after her first sound, every process whose command line holds the given name gets SIGKILL,
as the kernel's memory killer would do it. When her first answer has ended, the second words are said into the same
session. Printed: what each turn heard and said, and how soon after the second words her voice was heard. Passing
means the second turn was heard and answered aloud with no reload of her voice mode.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "voice"))

from a_spoken_turn import CHUNK_S, RATE_IN, spoken  # noqa: E402

from core.voice.duplex.protocol import AudioOpcode, decode_audio  # noqa: E402


def kill(name: str) -> list[int]:
    found = subprocess.run(["pgrep", "-f", "--", name], capture_output=True, text=True).stdout.split()
    pids = [int(pid) for pid in found if int(pid) != os.getpid()]
    for pid in pids:
        os.kill(pid, signal.SIGKILL)
    return pids


async def dies(args: argparse.Namespace) -> int:
    import websockets

    first, second = spoken(args.first, args.voice), spoken(args.second, args.voice)
    step = int(RATE_IN * CHUNK_S)
    silence = np.zeros(step, dtype=np.int16)
    url = f"ws://{args.host}/ws/voice?mode=focused"
    marks: dict[str, float] = {}
    events: list[tuple[float, dict]] = []
    heard_s: dict[int, float] = {}
    first_at: dict[int, float] = {}
    rate_out = 24000
    async with websockets.connect(url, origin=f"http://{args.host}", max_size=None) as ws:
        while True:
            message = json.loads(await asyncio.wait_for(ws.recv(), timeout=args.ready_s))
            if message.get("type") == "voice.ready":
                rate_out = int(message.get("sample_rate_out") or rate_out)
                break
        await ws.send(json.dumps({"command": "set_floor", "open": True}))
        ask_again = asyncio.Event()

        async def microphone() -> None:
            async def say(words: np.ndarray, mark: str) -> None:
                for start in range(0, len(words), step):
                    await ws.send(words[start:start + step].tobytes())
                    await asyncio.sleep(CHUNK_S)
                marks[mark] = time.monotonic()

            await say(first, "first words ended")
            while not ask_again.is_set():
                await ws.send(silence.tobytes())
                await asyncio.sleep(CHUNK_S)
            await say(second, "second words ended")
            while True:
                await ws.send(silence.tobytes())
                await asyncio.sleep(CHUNK_S)

        mic = asyncio.create_task(microphone())
        metrics, utterance, last_report, killed = 0, 0, 0.0, []
        try:
            deadline = time.monotonic() + args.wait_s
            while time.monotonic() < deadline:
                try:
                    message = await asyncio.wait_for(ws.recv(), timeout=0.05)
                except TimeoutError:
                    message = None
                now = time.monotonic()
                if isinstance(message, bytes):
                    opcode, _last, _seq, utterance_id, payload = decode_audio(message)
                    if opcode is AudioOpcode.SPEECH and payload:
                        first_at.setdefault(utterance_id, now)
                        heard_s[utterance_id] = heard_s.get(utterance_id, 0.0) + len(payload) / 2 / rate_out
                        utterance = utterance_id
                        if "second words ended" in marks:
                            marks.setdefault("second answer heard", now)
                elif isinstance(message, str):
                    data = json.loads(message)
                    events.append((now, data))
                    if data.get("type") == "voice.metrics":
                        metrics += 1
                        if metrics == 1:
                            ask_again.set()
                        else:
                            break
                if first_at and not killed and now - min(first_at.values()) >= args.after_s:
                    killed = kill(args.worker) or [0]
                    marks["killed"] = now
                if utterance and now - last_report >= 0.1:
                    played = min(heard_s.get(utterance, 0.0), now - first_at[utterance])
                    await ws.send(json.dumps({"command": "playback", "played_ms": played * 1000.0,
                                              "utterance_id": utterance}))
                    last_report = now
        finally:
            mic.cancel()
    print(f"killed {args.worker}: {[pid for pid in killed if pid] or 'no such process'}")
    finals = [d.get("text") for _, d in events if d.get("type") == "voice.final"]
    replies = [d.get("text") for _, d in events if d.get("type") == "voice.reply"]
    print(f"she heard: {finals}")
    for reply in replies:
        print(f"she said: {str(reply)[:200]}")
    ended, heard = marks.get("second words ended"), marks.get("second answer heard")
    print(f"her second answer was heard {heard - ended:.2f} s after the words ended" if ended and heard
          else "her second answer was not heard")
    return 0 if killed and heard and len(finals) >= 2 else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("worker")
    parser.add_argument("first")
    parser.add_argument("second")
    parser.add_argument("--after-s", type=float, default=3.0)
    parser.add_argument("--voice", default="Samantha")
    parser.add_argument("--host", default="127.0.0.1:8000")
    parser.add_argument("--ready-s", type=float, default=240.0)
    parser.add_argument("--wait-s", type=float, default=300.0)
    return asyncio.run(dies(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
