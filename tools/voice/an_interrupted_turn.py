"""Interrupt her while she speaks, the way a person in her voice mode would, and time what she does.

    python tools/voice/an_interrupted_turn.py "Tell me about something you find beautiful." \\
        "Wait, stop. Just tell me in one sentence." --after-s 5

The first words are said by macOS's `say` into her voice mode (as tools/voice/a_spoken_turn.py does), and the stream
goes on as her window's microphone would: silence between words, never a gap. ``--after-s`` seconds after her first
sound, the second words are said over her. Her speech is played back on a clock with playback reports, as her window
plays it. Printed: when she ducked and when she was interrupted after the interruption began, when the last audio of
her first answer arrived, what she heard of the interruption, and how soon after it ended her answer to it was heard.
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

from core.voice.duplex.protocol import AudioOpcode, decode_audio  # noqa: E402


async def interrupted(args: argparse.Namespace) -> int:
    import websockets

    first_words, second_words = spoken(args.first, args.voice), spoken(args.second, args.voice)
    url = f"ws://{args.host}/ws/voice?mode=focused"
    step = int(RATE_IN * CHUNK_S)
    silence = np.zeros(step, dtype=np.int16)
    marks: dict[str, float] = {}
    events: list[tuple[float, str, dict]] = []
    audio_at: dict[int, list[float]] = {}
    sent_mic: list[tuple[float, bytes]] = []  # what her microphone sent, and when
    got_speech: list[tuple[float, bytes]] = []  # her speech as it arrived, and when
    rate_out, heard_s = 24000, {}
    async with websockets.connect(url, origin=f"http://{args.host}", max_size=None) as ws:
        while True:
            message = json.loads(await asyncio.wait_for(ws.recv(), timeout=args.ready_s))
            if message.get("type") == "voice.ready":
                rate_out = int(message.get("sample_rate_out") or rate_out)
                break
        await ws.send(json.dumps({"command": "set_floor", "open": True}))
        interrupt_now = asyncio.Event()

        async def microphone() -> None:
            async def say(words: np.ndarray, mark: str) -> None:
                marks[mark + " began"] = time.monotonic()
                for start in range(0, len(words), step):
                    sent_mic.append((time.monotonic(), words[start:start + step].tobytes()))
                    await ws.send(sent_mic[-1][1])
                    await asyncio.sleep(CHUNK_S)
                marks[mark + " ended"] = time.monotonic()

            await say(first_words, "question")
            while not interrupt_now.is_set():
                sent_mic.append((time.monotonic(), silence.tobytes()))
                await ws.send(sent_mic[-1][1])
                await asyncio.sleep(CHUNK_S)
            await say(second_words, "interruption")
            while True:
                sent_mic.append((time.monotonic(), silence.tobytes()))
                await ws.send(sent_mic[-1][1])
                await asyncio.sleep(CHUNK_S)

        mic = asyncio.create_task(microphone())
        last_report, utterance, metrics = 0.0, 0, 0
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
                        got_speech.append((now, bytes(payload)))
                        audio_at.setdefault(utterance_id, []).append(now)
                        heard_s[utterance_id] = heard_s.get(utterance_id, 0.0) + len(payload) / 2 / rate_out
                        if utterance_id != utterance:
                            utterance = utterance_id
                            marks.setdefault("first sound", now)
                            if "interruption ended" in marks and "answer to it" not in marks and len(audio_at) > 1:
                                marks["answer to it"] = now
                        if "first sound" in marks and not interrupt_now.is_set() and \
                                now - marks["first sound"] >= args.after_s:
                            interrupt_now.set()
                elif isinstance(message, str):
                    data = json.loads(message)
                    kind = data.get("type", "")
                    events.append((now, kind, data))
                    if kind == "voice.metrics":
                        metrics += 1
                        if metrics >= 2 or ("interruption ended" in marks and "answer to it" in marks):
                            break
                if utterance and now - last_report >= 0.1:
                    started = audio_at[utterance][0]
                    played = min(heard_s.get(utterance, 0.0), now - started)
                    await ws.send(json.dumps({"command": "playback", "played_ms": played * 1000.0,
                                              "utterance_id": utterance}))
                    last_report = now
        except websockets.ConnectionClosed as closed:
            print(f"her voice mode closed the socket: {closed}")
        finally:
            mic.cancel()
    if args.record:
        np.savez_compressed(args.record, mic_at=np.array([t for t, _ in sent_mic]), rate_out=rate_out,
                            mic=np.array([np.frombuffer(b, np.int16) for _, b in sent_mic], dtype=object),
                            speech_at=np.array([t for t, _ in got_speech]),
                            speech=np.array([np.frombuffer(b, np.int16) for _, b in got_speech], dtype=object),
                            began=marks.get("interruption began", 0.0))
    began = marks.get("interruption began")
    if began is None:
        print("she never spoke long enough to be interrupted")
        return 2

    def after(at: float | None) -> str:
        return f"{at - began:+.2f} s" if at is not None else "never"

    first = [u for u in audio_at if audio_at[u][0] < began]
    last_of_first = max((t for u in first for t in audio_at[u]), default=None)
    ducked = next((t for t, k, _ in events if k == "voice.duck" and t >= began), None)
    stopped = next((t for t, k, _ in events if k == "voice.interrupted" and t >= began), None)
    finals = [(t, d.get("text")) for t, k, d in events if k == "voice.final" and t >= began]
    replies = [(t, d.get("text")) for t, k, d in events if k == "voice.reply" and t >= began]
    print(f"interruption said {began - marks['first sound']:.1f} s into her answer, for "
          f"{marks.get('interruption ended', began) - began:.1f} s")
    print(f"ducked {after(ducked)}, interrupted {after(stopped)}, last audio of her first answer {after(last_of_first)}")
    print(f"she heard: {finals[0][1] if finals else 'nothing'}")
    answer = marks.get("answer to it")
    ended = marks.get("interruption ended")
    print(f"her answer to it was heard {answer - ended:.2f} s after the interruption ended" if answer and ended
          else "she did not answer the interruption")
    print(f"she said: {replies[-1][1][:300] if replies else 'nothing'}")
    return 0 if stopped is not None and answer else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("first")
    parser.add_argument("second")
    parser.add_argument("--after-s", type=float, default=5.0)
    parser.add_argument("--voice", default="Samantha")
    parser.add_argument("--host", default="127.0.0.1:8000")
    parser.add_argument("--ready-s", type=float, default=240.0)
    parser.add_argument("--wait-s", type=float, default=180.0)
    parser.add_argument("--record", default="", help="save what was sent and what arrived, with times (.npz)")
    return asyncio.run(interrupted(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
