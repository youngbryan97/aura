"""Shared contracts for bounded token-level model evidence.

The final decoder state is strongly contextual, which is useful for reference
resolution but can erase stable lexical identity.  Semantic consumers may ask
for an equal-energy concatenation of the model's input embedding and final
causal state.  Both channels come from the same immutable model basis and the
same tokenization; no generated text or answer evidence enters the surface.
"""

from __future__ import annotations

from typing import Final

FINAL_HIDDEN_V1: Final = "final_hidden_v1"
LEXICAL_CONTEXTUAL_V1: Final = "lexical_contextual_v1"
HIDDEN_SEQUENCE_REPRESENTATIONS: Final = frozenset({FINAL_HIDDEN_V1, LEXICAL_CONTEXTUAL_V1})


def hidden_sequence_schema(representation: str) -> str:
    """Return the exact receipt schema for one evidence representation."""

    if representation == FINAL_HIDDEN_V1:
        return "aura.hidden_sequence_encoding.v1"
    if representation == LEXICAL_CONTEXTUAL_V1:
        return "aura.hidden_sequence_encoding.v2"
    raise ValueError(f"unsupported hidden sequence representation: {representation}")


def hidden_sequence_channels(representation: str) -> tuple[str, ...]:
    """Describe the ordered channels included in a packed hidden row."""

    if representation == FINAL_HIDDEN_V1:
        return ("final_causal_hidden",)
    if representation == LEXICAL_CONTEXTUAL_V1:
        return ("input_token_embedding", "final_causal_hidden")
    raise ValueError(f"unsupported hidden sequence representation: {representation}")


def hidden_sequence_channel_widths(
    representation: str,
    packed_hidden_size: int,
) -> tuple[int, ...]:
    """Recover each packed channel width from the versioned representation."""

    if type(packed_hidden_size) is not int or packed_hidden_size < 1:
        raise ValueError("hidden sequence packed width is invalid")
    if representation == FINAL_HIDDEN_V1:
        return (packed_hidden_size,)
    if representation == LEXICAL_CONTEXTUAL_V1:
        if packed_hidden_size % 2:
            raise ValueError("lexical-contextual packed width is not divisible by two")
        width = packed_hidden_size // 2
        return (width, width)
    raise ValueError(f"unsupported hidden sequence representation: {representation}")
