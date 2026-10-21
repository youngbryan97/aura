#!/usr/bin/env python3
"""Her real MLX worker, spawned through her own client as her app spawns it, timed with nothing else of her app running.

LIVE 20 October 2026: in her app her worker decoded at 7 to 9 tokens a second, 128.9 s of 130 waiting on the model,
where her model with every channel on and her head drafting decodes at 21.5 by itself (tools/her_decode_with_each_channel.py
builds those channels itself, not through her worker). Four stalls inside a pass were found and removed and the live
number did not move. This takes her app out of it: the worker is the app's own, its setup is the app's own, and the only
other process is the one asked for.

    her_worker_alone.py --output FILE --worker-log FILE [--with-brainstem idle|busy] [--tokens 600] 2>> FILE

Run with her app down and the Mac on mains power, its standard error going to the file given as --worker-log: her
worker writes its own lines there ("N tokens: Xs waiting on the model", "generation performance"), and each turn's are
read back from it.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PERSONA = "/Users/bryan/.aura/models/Aura-Qwen3.8-27B-persona-crsm-7f6a2e83f73f5eef9d15"
BRAINSTEM = "/Users/bryan/.aura/live-source/models/Ternary-Bonsai-2-27B-mlx-2bit"
QUESTION = "Tell me in a short paragraph how the bicycle changed ordinary people's lives in the 1890s."


def _worker_lines(log: Path, offset: int) -> tuple[list[str], int]:
    """The lines her worker wrote to ``log`` since ``offset``, and where the file ends now."""
    try:
        data = log.read_bytes()
    except OSError:
        return [], offset
    return data[offset:].decode("utf-8", "replace").splitlines(), len(data)


async def _turn(client, log: Path, tokens: int) -> dict:
    began = time.perf_counter()
    _seen, offset = _worker_lines(log, 0)
    text = await client.generate(QUESTION, max_tokens=tokens, temperature=0.0, foreground_request=True,
                                 origin="desktop_quick_user", timeout=600)
    await asyncio.sleep(1.0)  # the worker's last lines reach the file
    rows, _end = _worker_lines(log, offset)
    performance = next((line for line in reversed(rows) if "generation performance" in line), "")
    waited = next((line for line in reversed(rows) if "waiting on the model" in line), "")
    return {"seconds": round(time.perf_counter() - began, 1), "performance": performance[:220], "waited": waited[:160],
            "words": len(str(text or "").split())}


async def main_async(args: argparse.Namespace) -> list[dict]:
    from core.brain.llm.mlx_client import get_mlx_client

    logging.basicConfig(level=logging.INFO)
    log = args.worker_log
    rows = []
    her = get_mlx_client(PERSONA, max_tokens=2048)
    rows.append({"with": "her worker alone, first turn (warm-up)", **await _turn(her, log, 64)})
    rows.append({"with": "her worker alone", **await _turn(her, log, args.tokens)})
    if args.with_brainstem:
        brainstem = get_mlx_client(BRAINSTEM, max_tokens=256)
        await brainstem.generate("Say hello in five words.", max_tokens=16, temperature=0.0, timeout=600)
        rows.append({"with": "her worker, the brainstem's worker loaded and idle", **await _turn(her, log, args.tokens)})
        if args.with_brainstem == "busy":
            busy = asyncio.ensure_future(brainstem.generate("Write a long essay about rivers.", max_tokens=600,
                                                            temperature=0.7, is_background=True, timeout=900))
            await asyncio.sleep(5)
            rows.append({"with": "her worker, the brainstem generating at the same time", **await _turn(her, log, args.tokens)})
            busy.cancel()
    for row in rows:
        print(json.dumps(row), flush=True)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--with-brainstem", choices=("idle", "busy"))
    parser.add_argument("--tokens", type=int, default=600)
    parser.add_argument("--worker-log", type=Path, required=True, help="the file this run's standard error goes to")
    args = parser.parse_args()
    rows = asyncio.run(main_async(args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"question": QUESTION, "rows": rows}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
