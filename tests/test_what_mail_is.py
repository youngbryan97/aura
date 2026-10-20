"""F14: mail from a person to her, told from mail a program sent, by its headers."""
from __future__ import annotations

import pytest

from core.skills.what_mail_is import addressed_to, who_sent_it

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("headers", [
    {"From": "Maya <maya@example.org>", "To": "aura@example.com", "Subject": "Quick question about your music post"},
    {"From": "bryan@example.com", "Auto-Submitted": "no", "Return-Path": "<bryan@example.com>"},
])
def test_a_person_wrote_it(headers):
    assert who_sent_it(headers) == ""


@pytest.mark.parametrize(("headers", "why"), [
    ({"From": "Maya <maya@example.org>", "Auto-Submitted": "auto-replied"}, "Auto-Submitted"),
    ({"From": "Substack <news@substack.com>", "List-Unsubscribe": "<mailto:u@substack.com>"}, "List-Unsubscribe"),
    ({"From": "deals@shop.example", "Precedence": "bulk"}, "Precedence"),
    ({"From": "Google <no-reply@accounts.google.com>"}, "no-reply@accounts.google.com"),
    ({"From": "MAILER-DAEMON@mx.example", "Return-Path": "<>"}, "Return-Path"),
    ({"From": "GitHub <notifications@github.com>"}, "notifications@github.com"),
])
def test_a_program_sent_it(headers, why):
    assert why in who_sent_it(headers)


def test_how_it_reached_her():
    assert addressed_to({"To": "Aura <Aura@Example.com>, b@x.org"}, "aura@example.com") == "to"
    assert addressed_to({"To": "team@x.org", "Cc": "aura@example.com"}, "aura@example.com") == "cc"
    assert addressed_to({"To": "list@x.org"}, "aura@example.com") == ""
