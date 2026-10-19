"""An answer sized to the time the world gives, not a deadline stretched to the answer asked for.

Her answer clock grants a turn the time its asked-for tokens take: a turn asked for 644 tokens with her private
channel open was given 106 seconds on 18 October. A conversation with a person does not wait: in a two-minute chat
the other side expects a line in seconds. When a turn carries the time it has (``answer_within_s``), the request
is fitted to it: as many tokens as her model decodes, at its measured rate, in that time less the prompt's reading
and the delivery margin, and never fewer than a short line. The deadline stays what the world set, and the worker,
which sizes her private channel from the seconds left (``a_bounded_private_channel``), gives it none when there is
no time to think in.
"""
from __future__ import annotations

from typing import Any

#: The fewest tokens a reply is given, however short the time: a short line of conversation.
A_SHORT_LINE = 24


def fitted_to_the_time_given(*, seconds: Any, read_s: float, margin_s: float, max_tokens: int, model: str = "") -> int:
    """How many tokens a reply may take in ``seconds``, at her model's measured rate; ``max_tokens`` at most."""
    from core.brain.llm.thinking_reserve import tokens_decodable_in

    room = float(seconds) - float(read_s) - float(margin_s)
    fits = tokens_decodable_in(room, model) if room > 0 else 0
    return max(A_SHORT_LINE, min(int(max_tokens), int(fits)))
