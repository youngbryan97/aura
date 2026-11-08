"""A move her browser made, and that moved the page, vouches for her saying she made it.

LIVE 8 November 2026, dress rehearsal: after five rounds that searched, clicked a video and landed on it, her answer had
"Correction: I said I clicked something, and scrolled something, and none of those have a verified receipt in this turn"
put after it, because each round's receipt named the pursuit and not the moves it was made of.
"""

from core.conversation.claimed_effect import unverified_effect_claims
from core.skills.desktop_task import _EFFECT_OF_A_MOVE

SAID = "I clicked it. It's loading now, and I'm curious what machine this is."


def _round(actions, ok=True):
    return {"index": 0, "action": "browse_pursue", "ok": ok, "effect_verified": ok, "actions": actions}


def test_a_round_that_clicked_vouches_for_a_click():
    assert unverified_effect_claims(SAID, [_round(["click"])]) == []


def test_a_round_that_only_typed_does_not():
    assert unverified_effect_claims(SAID, [_round(["type"])]) == ["clicked something"]


def test_a_round_that_did_not_land_vouches_for_nothing():
    assert unverified_effect_claims(SAID, [_round(["click"], ok=False)]) == ["clicked something"]


def test_every_kind_of_move_has_an_effect_the_registry_knows():
    from core.epistemics.effect_registry import EFFECT_REGISTRY

    assert set(_EFFECT_OF_A_MOVE) == {"click", "type", "scroll", "go"}
    assert all(effect in EFFECT_REGISTRY for effect in _EFFECT_OF_A_MOVE.values())
