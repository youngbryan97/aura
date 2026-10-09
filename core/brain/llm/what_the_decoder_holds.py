"""What a job asks the decoder to hold its answer to: a JSON shape, or one of the choices on offer.

The worker's plain and streamed generation paths each carried the same block for
the JSON shape. Both now ask here, so a new thing the decoder holds is added once.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

__all__ = ["hold_the_answer"]

_SHAPES = {"json": "any", "json_object": "object", "json_array": "array"}


def hold_the_answer(job: Any, tokenizer: Any, native_thinking: Any, logits_processors: list[Any], logger: Any,
                    on_failure: Callable[..., Any]) -> None:
    """Add to ``logits_processors`` what holds the answer to the job's ``output_shape`` or ``choose_from``."""
    shape = str(job.get("output_shape") or "").strip().lower()
    choices = [str(c) for c in (job.get("choose_from") or ()) if str(c or "").strip()]
    if shape not in _SHAPES and not choices:
        return
    try:
        closing = None
        if native_thinking is True:
            from core.brain.llm.a_bounded_private_channel import _the_token_that_closes_it

            closing = _the_token_that_closes_it(tokenizer)
        if choices:
            from core.brain.llm.a_choice_the_decoder_enforces import enforce_one_of

            held, what = enforce_one_of(tokenizer, choices, after_token=closing), f"one of {len(choices)} choice(s)"
        else:
            from core.brain.llm.a_shape_the_decoder_enforces import enforce_json

            held, what = enforce_json(tokenizer, after_token=closing, require=_SHAPES[shape]), shape
        if held is not None:
            logits_processors.append(held)
            logger.info("🧠 [WORKER] Answer shape held by the decoder: %s.", what)
        else:
            logger.warning("🧠 [WORKER] Answer shape %s NOT held; MLX unavailable to the processor.", what)
    except (AttributeError, ImportError, RuntimeError, TypeError, ValueError) as e:
        on_failure(e, action="continued generation without the decoder holding the answer's shape", severity="warning")
