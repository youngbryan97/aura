"""A section that changes every turn must not sit in the head every turn shares.

Chat templates require one system message at the front, so the authority
messages are merged into one. Merging a PER-TURN section into it made that
message a different token sequence on every turn, and the front of the prompt
is the only part a KV cache can reuse.

Measured live 2026-09-07 on the resident 27B: after the authority head was made
one stable constant, `## LIVE TONE` — mood and tone, new each turn — was still
merged into the same message. The cache reported `matched 0 (0.0%)` of 1,866
tokens and prefill was 12.2s of a 16s turn.

What governs this turn now travels with this turn, immediately before the
person's message: still read, no longer paid for twice.
"""

from __future__ import annotations

from core.brain.llm.chat_format import system_first
from core.brain.llm.context_budget import VOLATILE_AT_OR_ABOVE, split_on_volatility, volatility_of
from core.utils.injected_blocks import RUNTIME_EVIDENCE_ROLE, is_stamped_grounding

_STABLE = "You are Aura Luna. Speak warmly."
_TURN = (
    "## CONTINUITY SUMMARY\nwe spoke about primes\n\n"
    "## LIVE TONE\nMood: empathy\nTone: warm\n"
)


def _transcript() -> list[dict]:
    return [
        {"role": "system", "content": _STABLE},
        {"role": "system", "content": _TURN},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "what is 2+2?"},
    ]


def test_the_head_keeps_only_what_does_not_change() -> None:
    head, tail = split_on_volatility(_STABLE + "\n\n" + _TURN)
    assert "## LIVE TONE" not in head
    assert "## LIVE TONE" in tail
    assert "## CONTINUITY SUMMARY" in head, "a slow section belongs in the head"
    assert head.startswith(_STABLE)


def test_nothing_is_dropped_by_the_split() -> None:
    body = _STABLE + "\n\n" + _TURN
    head, tail = split_on_volatility(body)
    for line in (line for line in body.splitlines() if line.strip()):
        assert line in head or line in tail, line


def test_the_turn_state_lands_immediately_before_the_person() -> None:
    out = system_first(_transcript())
    roles = [str(m["role"]) for m in out]
    assert roles[0] == "system"
    assert roles[-1] == "user"
    assert roles[-2] == RUNTIME_EVIDENCE_ROLE
    assert "## LIVE TONE" in str(out[-2]["content"])
    assert "## LIVE TONE" not in str(out[0]["content"])


def test_the_turn_state_is_stamped_so_it_cannot_be_reclaimed_as_authority() -> None:
    """An unstamped block would be merged back into the head on the next pass."""
    out = system_first(_transcript())
    assert is_stamped_grounding(dict(out[-2], role="system"))
    assert system_first(out) == out


def test_a_transcript_with_no_volatile_section_is_untouched() -> None:
    plain = [
        {"role": "system", "content": _STABLE},
        {"role": "user", "content": "hi"},
    ]
    assert system_first(plain) == plain


def test_an_unwatched_section_moves_out_rather_than_risking_the_prefix() -> None:
    assert volatility_of("## SOMETHING NOBODY DECLARED\nx") >= VOLATILE_AT_OR_ABOVE
    head, tail = split_on_volatility(
        _STABLE + "\n\n## SOMETHING NOBODY DECLARED\nfresh reading\n"
    )
    assert "SOMETHING NOBODY DECLARED" in tail
    assert head == _STABLE


def test_the_conversation_keeps_its_order() -> None:
    out = system_first(_transcript())
    said = [str(m["content"]) for m in out if m["role"] in {"user", "assistant"}]
    assert said == ["hi", "hello", "what is 2+2?"]
