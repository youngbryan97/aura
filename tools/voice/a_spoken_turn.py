"""Speak one turn to her voice mode the way her window does, and time what comes back.

    python tools/voice/a_spoken_turn.py "What's two plus two?" --out /tmp/her_reply.wav

The words are said by macOS's `say` into 16 kHz PCM and streamed to /ws/voice in real time, 20 ms at a time, from her
own window's origin on this machine, followed by silence so her endpointing can close the turn. Her speech is played
back on a clock (playback reports, then "drained") as her window would, and saved. Her own latency split
(`voice.metrics`) is printed with what this client measured from the end of the words.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from core.voice.duplex.protocol import AudioOpcode, decode_audio  # noqa: E402

RATE_IN = 16000
CHUNK_S = 0.02


def spoken(text: str, voice: str) -> np.ndarray:
    """The words as 16 kHz mono PCM16, said by macOS."""
    with tempfile.TemporaryDirectory(prefix="aura-spoken-turn-") as scratch:
        aiff, wav = Path(scratch) / "said.aiff", Path(scratch) / "said.wav"
        subprocess.run(["say", "-v", voice, "-o", str(aiff), text], check=True)
        subprocess.run(["afconvert", "-f", "WAVE", "-d", f"LEI16@{RATE_IN}", "-c", "1", str(aiff), str(wav)], check=True)
        import soundfile as sf

        audio, _rate = sf.read(wav, dtype="int16")
    return np.asarray(audio, dtype=np.int16)


async def one_turn(args: argparse.Namespace) -> int:
    import websockets

    words = spoken(args.text, args.voice)
    url = f"ws://{args.host}/ws/voice?mode=focused"
    heard: list[np.ndarray] = []
    rate_out = 24000
    events: list[tuple[float, str, dict]] = []
    async with websockets.connect(url, origin=f"http://{args.host}", max_size=None) as ws:
        began = time.monotonic()
        while True:  # her models warm before she says she is ready
            message = json.loads(await asyncio.wait_for(ws.recv(), timeout=args.ready_s))
            if message.get("type") == "voice.ready":
                rate_out = int(message.get("sample_rate_out") or rate_out)
                print(f"ready after {time.monotonic() - began:.1f} s: echo cancellation {message.get('echo_cancellation')}, "
                      f"speech {message.get('tts_engine')}")
                break
            if message.get("type") == "voice.error":
                print("voice error before ready:", message)
                return 1
        await ws.send(json.dumps({"command": "set_floor", "open": True}))

        step = int(RATE_IN * CHUNK_S)
        silence = np.zeros(step, dtype=np.int16)
        ended_at = 0.0
        played_s, utterance, drained_sent = 0.0, 0, False
        last_report = 0.0
        # Playback as her window does it: the position moves only while audio is buffered. A stall that more audio
        # follows is an underrun, and how long she went silent mid-reply is what a listener hears as a stumble.
        last_tick, starved_s, underruns = 0.0, 0.0, []

        async def stream_words() -> None:
            nonlocal ended_at
            for start in range(0, len(words), step):
                await ws.send(words[start:start + step].tobytes())
                await asyncio.sleep(CHUNK_S)
            ended_at = time.monotonic()
            for _ in range(int(args.tail_s / CHUNK_S)):
                await ws.send(silence.tobytes())
                await asyncio.sleep(CHUNK_S)

        sender = asyncio.create_task(stream_words())
        try:
            deadline = time.monotonic() + args.wait_s
            while time.monotonic() < deadline:
                try:
                    message = await asyncio.wait_for(ws.recv(), timeout=0.1)
                except TimeoutError:
                    message = None
                now = time.monotonic()
                if isinstance(message, bytes):
                    opcode, last, _seq, utterance_id, payload = decode_audio(message)
                    if opcode is AudioOpcode.SPEECH and payload:
                        if not heard:
                            events.append((now, "first audio", {}))
                            last_tick = now
                        elif starved_s > 0.0:
                            underruns.append(round(starved_s, 3))
                            starved_s = 0.0
                        utterance = utterance_id
                        heard.append(np.frombuffer(payload, dtype=np.int16))
                elif isinstance(message, str):
                    data = json.loads(message)
                    kind = data.get("type", "")
                    events.append((now, kind, data))
                    if kind == "voice.metrics":
                        break
                # Played on a clock, as her window plays it.
                if heard:
                    total_s = sum(len(h) for h in heard) / rate_out
                    ahead = played_s + (now - last_tick)
                    if ahead > total_s:
                        starved_s += ahead - max(played_s, total_s)
                    played_s, last_tick = min(total_s, ahead), now
                if heard and now - last_report >= 0.1 and not drained_sent:
                    report = {"command": "playback", "played_ms": played_s * 1000.0, "utterance_id": utterance}
                    if played_s >= total_s and any(k == "voice.reply" for _t, k, _d in events):
                        report["drained"], drained_sent = True, True
                    await ws.send(json.dumps(report))
                    last_report = now
        finally:
            sender.cancel()
    for at, kind, data in events:
        since = f"{at - ended_at:+6.2f} s" if ended_at else "   before end"
        shown = {k: v for k, v in data.items() if k not in {"type"}}
        print(f"{since}  {kind}  {json.dumps(shown)[:200] if shown else ''}")
    if heard:
        print(f"underruns: {len(underruns)} (her voice went silent mid-reply for {underruns or 'no'} s)")
    if heard and args.out:
        import soundfile as sf

        sf.write(args.out, np.concatenate(heard), rate_out)
        print(f"her reply ({sum(len(h) for h in heard) / rate_out:.1f} s) saved to {args.out}")
    return 0 if heard else 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("text")
    parser.add_argument("--voice", default="Samantha")
    parser.add_argument("--host", default="127.0.0.1:8000")
    parser.add_argument("--out", default="")
    parser.add_argument("--ready-s", type=float, default=120.0)
    parser.add_argument("--wait-s", type=float, default=90.0)
    parser.add_argument("--tail-s", type=float, default=2.0)
    return asyncio.run(one_turn(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
