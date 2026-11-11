"""Her voice mode's speech recognizer in a process of its own: Parakeet on MLX, one decode at a time over stdin/stdout.

LIVE 11 November 2026: her runtime died of a segmentation fault twice in one day (about 01:12 and at 09:49), each time
with a Parakeet decode inside ``mx.eval`` on an executor thread of her main process, beside the other code there that
uses MLX. Here MLX runs on this process's one thread and nothing else (core/voice/duplex/parakeet_process.py).

    python -u tools/voice/parakeet_worker.py <repo> [<repo> ...]

Each request line is {"audio": base64 little-endian float32 at 16 kHz, "repo": ...}; each answer is
{"text": ..., "tokens": [[text, start_s, end_s], ...]} or {"error": ...}. The repos named on the command line are
loaded and run once on a second of silence before {"ready": true}, so weights and kernels are never paid inside a turn.
"""
from __future__ import annotations

import base64
import contextlib
import json
import sys


def main() -> None:
    output = sys.stdout

    def send(message: dict) -> None:
        output.write(json.dumps(message) + "\n")
        output.flush()

    try:
        with contextlib.redirect_stdout(sys.stderr):
            import mlx.core as mx
            import numpy as np
            from parakeet_mlx import from_pretrained
            from parakeet_mlx.audio import get_logmel

            models: dict[str, object] = {}

            def decode(audio: np.ndarray, repo: str) -> tuple[str, list[list]]:
                model = models.get(repo)
                if model is None:
                    model = models[repo] = from_pretrained(repo)
                mel = get_logmel(mx.array(np.asarray(audio, dtype=np.float32)), model.preprocessor_config)
                results = model.generate(mel)
                result = results[0] if results else None
                if result is None:
                    return "", []
                tokens = [[str(getattr(token, "text", "") or ""), float(getattr(token, "start", 0.0) or 0.0),
                           float(getattr(token, "end", 0.0) or 0.0)] for token in (getattr(result, "tokens", ()) or ())]
                return str(getattr(result, "text", "") or ""), tokens

            for repo in sys.argv[1:]:
                decode(np.zeros(16000, dtype=np.float32), repo)
        send({"ready": True, "repos": sys.argv[1:]})
        for line in sys.stdin:
            try:
                if len(line) > 16 * 1024 * 1024:
                    raise ValueError("decode request too large")
                request = json.loads(line)
                audio = np.frombuffer(base64.b64decode(request["audio"], validate=True), dtype="<f4")
                if not np.isfinite(audio).all():
                    raise ValueError("audio is not finite")
                with contextlib.redirect_stdout(sys.stderr):
                    text, tokens = decode(audio, str(request["repo"]))
                send({"text": text, "tokens": tokens})
            except (ValueError, KeyError, TypeError, RuntimeError, OSError) as exc:
                send({"error": f"{type(exc).__name__}: {exc}"})
    except Exception as exc:  # noqa: BLE001 - the parent learns why its recognizer could not start, then this exits
        send({"error": f"{type(exc).__name__}: {exc}"})
        raise


if __name__ == "__main__":
    main()
