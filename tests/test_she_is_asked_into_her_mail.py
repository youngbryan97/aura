"""F14: asking her to look in her mail lifts the turn's ceiling to her mail, as naming a site lifts it to her browser.

LIVE 2026-11-10, the first time her mail had credentials: "Can you check your email and tell me what's in your
inbox?" ran under the self-service ceiling ("Tool handoff: no capability derived for this request (ceiling
sandboxed_compute)"), and she said she would check before telling him, and did not.
"""
from __future__ import annotations

import pytest

from core.intent.asks_to_go_online import asks_into_her_mail
from core.phases.response_contract import requested_effect_ceiling

pytestmark = pytest.mark.unit

ASKED = [
    "Can you check your email and tell me what's in your inbox?",
    "check your inbox",
    "What's in your inbox today?",
    "Did you get any email from me?",
    "Have you received any new mail?",
    "Any new emails?",
    "Look through your mail for anything from Laird.",
]

NOT_ASKED = [
    "I checked my email this morning.",
    "What is email, really?",
    "Check your answer again.",
    "Can you write an email to my landlord?",
    "Send me a text when you're done.",
]


@pytest.mark.parametrize("message", ASKED)
def test_a_request_into_her_mail_is_read_as_one(message):
    assert asks_into_her_mail(message)


@pytest.mark.parametrize("message", NOT_ASKED)
def test_talk_about_mail_is_not_a_request_into_hers(message):
    assert not asks_into_her_mail(message)


def test_the_live_request_reaches_her_mail_s_ceiling():
    ceiling, scopes = requested_effect_ceiling("Can you check your email and tell me what's in your inbox?")
    assert ceiling == "external_io" and "external_io" in scopes


def _skill(cls):
    from types import SimpleNamespace

    return SimpleNamespace(description=cls.description, effect_scope="external_io", enabled=True)


def test_her_mail_is_offered_when_asked_into_it_and_reddit_s_inbox_only_when_mail_is_not_named():
    """Both declare an inbox; the request's own word for it decides between them."""
    from core.intent.capability_selection import select_capabilities
    from core.skills.email_adapter import EmailAdapterSkill
    from core.skills.reddit_adapter import RedditAdapterSkill

    skills = {"email_adapter": _skill(EmailAdapterSkill), "reddit_adapter": _skill(RedditAdapterSkill)}

    def chosen(text):
        ceiling, scopes = requested_effect_ceiling(text)
        return select_capabilities(text, skills, ceiling=ceiling, admissible_scopes=scopes)

    assert chosen("Can you check your email and tell me what's in your inbox?") == ["email_adapter"]
    assert chosen("Any new emails?") == ["email_adapter"]
    assert set(chosen("What's in your inbox?")) == {"email_adapter", "reddit_adapter"}
    assert chosen("What is email, really?") == []


@pytest.mark.parametrize("module, skill_class, payload", [
    ("core.skills.email_adapter", "EmailAdapterSkill", {"mode": "check", "limit": 5}),
    ("core.skills.reddit_adapter", "RedditAdapterSkill", {"mode": "browse", "subreddit": "aww"}),
])
def test_her_accounts_ask_authority_under_the_turns_origin_not_the_tool_loop(monkeypatch, module, skill_class, payload):
    """LIVE 2026-11-10: her chat's tool loop names itself the source; her mail asked under it and was refused
    "signed_standing_authority_lease_missing" on a turn the owner's foreground grant covers."""
    import asyncio
    import importlib

    import core.executive.authority_gateway as gateways
    from core.executive.standing_authority import context_has_user_authority

    asked: list[str] = []

    class _Gateway:
        async def authorize_tool_execution(self, tool, args, *, source, **_kwargs):
            asked.append(source)
            from types import SimpleNamespace

            return SimpleNamespace(approved=False, reason="probe", capability_token_id=None)

    monkeypatch.setattr(gateways, "get_authority_gateway", lambda: _Gateway())
    skill = getattr(importlib.import_module(module), skill_class)()
    context = {"source": "think_and_act", "origin": "desktop_quick_user"}
    asyncio.run(skill.execute(payload, context))
    assert asked == ["desktop_quick_user"]
    assert context_has_user_authority(asked[0], {}) and not context_has_user_authority("think_and_act", {})
