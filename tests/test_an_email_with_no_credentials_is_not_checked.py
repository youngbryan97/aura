"""Her inbox is not checked when she has no email credentials, and a skill not set up is not an error.

On both day runs her social loop ran the email skill every fifteen minutes under an executive token, and each run
came back "Email credentials not found in Keychain" as a skill warning: 34 in two days.
"""
from __future__ import annotations

import asyncio
import logging

import pytest

pytestmark = pytest.mark.unit


def test_no_credentials_means_no_skill_call(monkeypatch):
    from core.autonomy import social_initiative
    from core.skills.email_adapter import EmailAdapterSkill

    monkeypatch.setattr(EmailAdapterSkill, "configured", staticmethod(lambda: False))
    calls: list[dict] = []

    class Loop(social_initiative._StartsSomethingSocial):
        async def _execute_email_adapter(self, payload, cap_engine=None):
            calls.append(payload)
            return {"ok": True}

    asyncio.run(Loop()._check_email_initiative())
    assert calls == []


def test_unreadable_keychain_reads_as_not_configured(monkeypatch):
    import core.security.zenith_secrets as secrets
    from core.skills.email_adapter import EmailAdapterSkill

    def broken(*_a, **_k):
        raise OSError("keychain locked")

    monkeypatch.setattr(secrets, "get_credential", broken)
    assert EmailAdapterSkill.configured() is False


def test_a_skill_not_set_up_says_so_at_info(caplog):
    from core.skills.base_skill import BaseSkill
    from pydantic import BaseModel

    class Nothing(BaseModel):
        pass

    class NotSetUp(BaseSkill):
        name = "not_set_up_probe"
        description = "probe"
        input_model = Nothing

        async def execute(self, params, context):
            return {"ok": False, "status": "credentials_missing", "error": "no credentials"}

    with caplog.at_level(logging.INFO, logger="Skills"):
        asyncio.run(NotSetUp().safe_execute(Nothing(), {}))
    mine = [r for r in caplog.records if "not_set_up_probe" in r.getMessage()]
    assert mine and all(r.levelno < logging.WARNING for r in mine)
