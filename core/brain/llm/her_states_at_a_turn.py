"""Her model's own states at the place where her turn opens, for a transcript.

What a person wants of her is read from here (core/understanding/what_is_asked.py). The transcript is the exchange as it
stood, in the bare chat markup with no system message, read once and never written to: no sampling, no token chosen, no
question put to the model. The last position is where she would begin to answer, and by the middle of the model that
position already holds what kind of answer is in order.

Measured 23 November 2026 on 135 sentences kept out of the fitting, with a 9B of her cortex's family: the states at
three eighths, a half and nineteen thirty-seconds of its depth, each read by a linear classifier and averaged, told the
thirteen kinds apart on 94.8% of them and whether anything was asked at all on 97.8%. Its sentence encoder, read the
same way, came to 73% and 87%; the last layers, which are already turning towards the next token, to less than the middle.

The states leave the worker as a count sketch (Charikar, Chen and Farach-Colton 2002): every coordinate is added, with a
sign of its own, into one of ``SKETCH`` buckets, which keeps inner products in expectation and needs no matrix held. At
2,048 buckets nothing measured above was lost.
"""
from __future__ import annotations

import base64
import contextlib
import json
from collections.abc import Sequence
from typing import Any

from core.runtime.errors import record_degradation

__all__ = ["DEPTHS", "SKETCH", "as_a_job_of_the_worker", "states_from_text", "states_where_she_answers"]

#: How far through the model the states are read, as shares of its depth.
DEPTHS: tuple[float, ...] = (12 / 32, 16 / 32, 19 / 32)
#: How many numbers a state leaves the worker as.
SKETCH = 2048
#: The most transcripts one job reads, so a person's turn never waits behind a long one.
MOST_TRANSCRIPTS = 8
#: The most tokens of a transcript that are read: its end, which holds the turn.
MOST_TOKENS = 1024


class _Tap:
    """A layer that keeps what left it at the last position, and is otherwise the layer."""

    def __init__(self, layer: Any, kept: dict[int, Any], at: int) -> None:
        self._layer, self._kept, self._at = layer, kept, at

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        out = self._layer(*args, **kwargs)
        hidden = out[0] if isinstance(out, tuple) else out
        self._kept[self._at] = hidden[0, -1, :]
        return out

    def __getattr__(self, name: str) -> Any:
        return getattr(self._layer, name)


def _layers(model: Any) -> list[Any]:
    """The list the model walks, itself: taps go into it and come out again."""
    from core.brain.llm.decoder_topology import decoder_backbone, resolve_language_model

    language = resolve_language_model(model)
    for holder in (decoder_backbone(model), language, getattr(language, "model", None)):
        layers = getattr(holder, "layers", None)
        if isinstance(layers, list) and layers:
            return layers
    raise ValueError("no list of layers to read from")


_SKETCHES: dict[int, tuple[Any, Any]] = {}


def _sketch(width: int) -> tuple[Any, Any]:
    """The bucket and the sign of each coordinate of a state this wide: the same every time."""
    import numpy as np

    if width not in _SKETCHES:
        rng = np.random.default_rng(width)
        _SKETCHES[width] = (rng.integers(0, SKETCH, size=width), rng.choice([-1.0, 1.0], size=width))
    return _SKETCHES[width]


def _sketched(state: Any) -> str:
    """One state as text: its count sketch, in half precision."""
    import numpy as np

    vector = np.asarray(state, dtype=np.float64)
    buckets, signs = _sketch(vector.shape[0])
    out = np.bincount(buckets, weights=vector * signs, minlength=SKETCH)
    return base64.b64encode(np.clip(out, -6.0e4, 6.0e4).astype(np.float16).tobytes()).decode("ascii")


def states_where_she_answers(model: Any, tokenizer: Any, transcripts: Sequence[str]) -> list[list[str]]:
    """For each transcript, the sketch of the state at each of ``DEPTHS`` at its last position."""
    import mlx.core as mx
    import numpy as np

    layers = _layers(model)
    count = len(layers)
    at = [min(count - 1, max(0, round(depth * count))) for depth in DEPTHS]
    kept: dict[int, Any] = {}
    held = {index: layers[index] for index in set(at)}
    read: list[list[str]] = []
    try:
        for index, layer in held.items():
            layers[index] = _Tap(layer, kept, index)
        for transcript in transcripts[:MOST_TRANSCRIPTS]:
            ids = list(tokenizer.encode(str(transcript), add_special_tokens=False))[-MOST_TOKENS:]
            if not ids:
                read.append([])
                continue
            kept.clear()
            model(mx.array([ids]))
            # Only what is read is computed: the layers above the last tap are never evaluated.
            states = [kept[index].astype(mx.float32) for index in at]
            mx.eval(states)
            read.append([_sketched(np.array(state)) for state in states])
    finally:
        for index, layer in held.items():
            layers[index] = layer
    return read


def as_a_job_of_the_worker(job: Any, model: Any, tokenizer: Any, engine: Any, metal: Any) -> dict[str, Any]:
    """The same pass as a job of the worker that owns the model lane, her steering stood down: what is read is what was
    said to her, and not how she feels about it. The answer is JSON in the place a generation's text goes."""
    wanted = [str(item) for item in (job.get("states_at_turn") or ())]
    response: dict[str, Any] = {
        "id": job.get("id"), "action": "generate", "status": "ok", "text": "[]", "tokens_used": 0,
        "prompt_tokenization": {"chars": 0, "tokens": 0, "generated_tokens": 0}, "generation_performance": {},
    }
    try:
        with contextlib.ExitStack() as held:
            held.enter_context(metal)
            if engine is not None and hasattr(engine, "controlled_measurement"):
                held.enter_context(engine.controlled_measurement())
                engine.set_alpha(0.0)
            response["text"] = json.dumps(states_where_she_answers(model, tokenizer, wanted))
    except (AttributeError, IndexError, KeyError, RuntimeError, TypeError, ValueError) as exc:
        response.update({"status": "error", "message": f"{type(exc).__name__}: {exc}"})
    return response


def states_from_text(text: Any, expected: int) -> list[Any]:
    """What the worker answered, as one array of (depths, sketch) for each transcript; nothing where it does not hold."""
    import numpy as np

    try:
        rows = json.loads(str(text or "[]"))
    except ValueError:
        return []
    if not isinstance(rows, list) or len(rows) != expected:
        return []
    read = []
    for row in rows:
        if not isinstance(row, list) or len(row) != len(DEPTHS):
            return []
        try:
            state = np.stack([np.frombuffer(base64.b64decode(str(part)), dtype=np.float16) for part in row])
        except (ValueError, TypeError):
            return []
        if state.shape != (len(DEPTHS), SKETCH) or not np.isfinite(state).all():
            return []
        read.append(state.astype(np.float32))
    return read


class _ReadsHerStatesAtATurn:
    """The client's side of the job: a part of MLXLocalClient, kept here with the pass it asks for."""

    async def states_at_turn(self, transcripts: Sequence[str], *, in_a_turn: bool) -> list[Any]:
        """Her states at the end of each transcript, or nothing. ``in_a_turn`` when a person is waiting on it."""
        wanted = [str(item) for item in transcripts if str(item or "").strip()][:MOST_TRANSCRIPTS]
        if not wanted:
            return []
        try:
            text = await self.generate_text_async(  # type: ignore[attr-defined]
                wanted[-1][-400:], states_at_turn=wanted, max_tokens=1, temperature=0.0,
                origin="what_is_asked_of_her", internal_inference=True, foreground_request=in_a_turn,
                is_background=not in_a_turn,
            )
        except (RuntimeError, TypeError, ValueError, OSError, TimeoutError) as exc:
            record_degradation("what_is_asked", exc, severity="warning",
                               action="a turn was not read for what it asks of her; the worker gave no states")
            return []
        return states_from_text(text, len(wanted))
