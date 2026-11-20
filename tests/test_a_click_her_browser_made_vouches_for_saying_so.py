"""A move her browser made, and that moved the page, vouches for her saying she made it.

LIVE 8 November 2026, dress rehearsal: after five rounds that searched, clicked a video and landed on it, her answer had
"Correction: I said I clicked something, and scrolled something, and none of those have a verified receipt in this turn"
put after it, because each round's receipt named the pursuit and not the moves it was made of. Whether a round vouches
for what her reply says is read by meaning now (tests/test_what_her_record_refutes.py, DESKTOP_MEASURED).
"""

from core.skills.desktop_task import _EFFECT_OF_A_MOVE


def test_every_kind_of_move_has_an_effect_the_registry_knows():
    from core.epistemics.effect_registry import EFFECT_REGISTRY

    assert set(_EFFECT_OF_A_MOVE) == {"click", "type", "scroll", "go"}
    assert all(effect in EFFECT_REGISTRY for effect in _EFFECT_OF_A_MOVE.values())


def test_a_decision_to_go_somewhere_is_one_that_can_be_done():
    import json

    from core.skills.sovereign_browser import SovereignBrowserSkill

    observation = {"elements": [{"name": "Search", "selector": "#search", "tag": "input"}], "url": "https://x.test/"}
    go = json.dumps({"actions": [{"type": "go", "value": "https://x.test/@someone/videos"}], "why": "his channel"})
    nowhere = json.dumps({"actions": [{"type": "go", "value": ""}], "why": "somewhere"})
    assert SovereignBrowserSkill._decision_is_usable(go, observation, "go to his channel")
    assert not SovereignBrowserSkill._decision_is_usable(nowhere, observation, "go to his channel")


def test_asked_to_watch_and_nothing_played_is_told_whatever_rounds_landed(monkeypatch):
    from interface.routes import chat_desktop_objective

    monkeypatch.setattr(chat_desktop_objective, "_asks_to_watch", lambda objective: True)
    assert chat_desktop_objective._asked_to_watch_and_nothing_plays({"ok": True}, "watch something of his")
    assert not chat_desktop_objective._asked_to_watch_and_nothing_plays({"ok": True, "watched": True}, "x")
    assert not chat_desktop_objective._asked_to_watch_and_nothing_plays({"ok": True, "beside": True}, "x")
