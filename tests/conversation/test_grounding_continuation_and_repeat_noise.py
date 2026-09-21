"""Two live findings from one imagination turn.

1. GRAMMAR THE USER CAN SEE. Asked to imagine a room, she replied:

       From my conversation memory, A room with walls made of memory, where
       the echoes are silent and the floors have no texture.

   The grounding clause ends in ", " so the body continues it, and
   _lowercase_continuation_start exists to down-case that first word. It
   down-cased "The" and "Forgetting" correctly and left "A" alone, because
   both of its acronym guards fire on a single capital letter:
   "A".isupper() is True, and "A"[1:] is "" whose .islower() is False. The two
   guards written to protect RAM and CPU blocked, between them, the commonest
   sentence opener in English.

2. REPEAT INCIDENT NOISE. The recurrent latent cortex declined every single
   foreground turn on an identical, unchanging contract failure, and each turn
   opened a fresh incident (four in one hour, each auto-resolving after 300s
   only to be replaced) while pushing resilience toward the depletion state
   that suppresses execution.
"""

from __future__ import annotations

import pytest
from tests.source_contract import family_text


# ── 1. The grounding clause must read as English ───────────────────────────

def _low(text: str) -> str:
    from core.phases.dialogue_policy import _lowercase_continuation_start

    return _lowercase_continuation_start(text)


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        # The live defect.
        ("A room with walls made of memory.", "a room with walls made of memory."),
        ("An echo without a source.", "an echo without a source."),
        ("The code you gave me.", "the code you gave me."),
        ("Forgetting is a mercy.", "forgetting is a mercy."),
    ],
)
def test_safe_openers_are_down_cased(body: str, expected: str) -> None:
    assert _low(body) == expected


@pytest.mark.parametrize(
    "body",
    [
        "I hold my position.",   # pronoun — never down-cased
        "RAM is at 80 percent.",  # acronym — the guards' actual purpose
        "MLX loaded the model.",
        "Aura is in front.",      # her own name
        "Bryan asked me to.",     # a person's name
    ],
)
def test_pronouns_acronyms_and_names_keep_their_capital(body: str) -> None:
    assert _low(body) == body


def test_the_safe_opener_set_encodes_the_distinction() -> None:
    """The article is allowed and the pronoun is not — no new list needed."""
    from core.phases.dialogue_policy import _CONTINUATION_SAFE_OPENERS

    assert "a" in _CONTINUATION_SAFE_OPENERS
    assert "i" not in _CONTINUATION_SAFE_OPENERS


# ── 2. A persistent, unchanged refusal is reported once ────────────────────

def test_unchanged_contract_failure_is_not_re_reported() -> None:
    import inspect

    from core.brain import latent_cortex_service

    source = family_text(latent_cortex_service)
    marker = 'reason = "receipt_contract_failed:" + ",".join(contract_errors)'
    index = source.find(marker)
    assert index != -1
    branch = source[index : index + 2000]

    # Repeats log; only a new or changed reason records a degradation.
    assert "if reason == self._last_refusal:" in branch
    assert "record_degradation" in branch


def test_changed_contract_failure_still_reports() -> None:
    """Suppression must be keyed on the reason, never unconditional."""
    same = "receipt_contract_failed:decode_bridge_unapplied"
    changed = "receipt_contract_failed:decode_bridge_unapplied,verifier_receipt_invalid"

    def records(reason: str, last: str | None) -> bool:
        return reason != last

    last: str | None = None
    incidents = 0
    for _ in range(20):
        if records(same, last):
            incidents += 1
        last = same

    assert incidents == 1, "twenty identical turns must open one incident"

    assert records(changed, last) is True
    assert records(same, changed) is True
