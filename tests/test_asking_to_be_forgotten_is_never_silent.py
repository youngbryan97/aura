"""A person asking to be forgotten, and nothing happening or being said.

Relational-memory deletion is decided by exact match after prefix and
suffix stripping, against three sentences: "forget everything about me",
"delete all relational memory", "erase all relationship memory". Exact is
right for the act — erasing what she knows about somebody is irreversible,
and "do not forget everything about me" must never fire it.

What was wrong is the other half. Measured 2026-09-18, five of seven
ordinary phrasings matched nothing and produced nothing: no deletion, no
refusal, no record. Including "Aura, forget everything you know about me",
whose prefix the normaliser strips and whose body is one word away from a
command. The person asked, believed they had asked, and the runtime went
on as if they had said nothing.

``is_forget_command`` covered the narrower "forget this" scope and was
called by nothing anywhere in the tree — five literal sentences, one of
them "delete the movie session", a past test case left in a production
predicate, with a test asserting it worked.
"""

from __future__ import annotations

import pytest

from core.runtime import memory_consent as mc


@pytest.fixture(autouse=True)
def _clean():
    mc.reset_unhonoured_deletion_requests_for_test()
    yield
    mc.reset_unhonoured_deletion_requests_for_test()


ORDINARY_PHRASINGS = [
    "forget everything you know about me",
    "please delete everything you have on me",
    "Aura, forget everything you know about me.",
    "erase everything you remember about me",
    "wipe my data",
    "remove my history",
]


@pytest.mark.parametrize("text", ORDINARY_PHRASINGS)
def test_an_ordinary_deletion_request_is_recognised_as_one(text):
    assert mc.looks_like_a_deletion_request(text), text


@pytest.mark.parametrize(
    "text",
    [
        "what is the weather",
        "tell me about the movie",
        "remember always",
        "session only please",
        "can you delete that file for me",
    ],
)
def test_ordinary_talk_is_not_a_deletion_request(text):
    assert not mc.looks_like_a_deletion_request(text), text


def test_the_exact_command_set_still_decides_the_act():
    """Recognition is not permission. The irreversible act stays exact."""

    assert mc.is_delete_all_relational_memory_command("forget everything about me")
    # Recognised as a request, and deliberately NOT actioned.
    assert mc.looks_like_a_deletion_request("forget everything you know about me")
    assert not mc.is_delete_all_relational_memory_command(
        "forget everything you know about me"
    )


def test_a_negated_request_costs_a_line_and_never_a_deletion():
    """Loose on purpose, and safe because of what it does not do."""

    assert mc.looks_like_a_deletion_request("do not forget everything about me")
    assert not mc.is_delete_all_relational_memory_command(
        "do not forget everything about me"
    )


def test_an_unhonoured_request_is_counted(monkeypatch):
    class _Authority:
        persistence_available = False

        def grant_for(self, *a, **k):
            return None

    before = mc.memory_consent_report()["unhonoured_deletion_requests"]
    result = mc.apply_relational_memory_command(
        _Authority(), "bryan", "forget everything you know about me"
    )
    assert result is None, "a near miss must not act"
    after = mc.memory_consent_report()
    assert after["unhonoured_deletion_requests"] == before + 1
    assert after["people_affected"] >= 1
    assert after["last_at"]


def test_an_ordinary_message_is_not_counted():
    class _Authority:
        persistence_available = False

    mc.apply_relational_memory_command(_Authority(), "bryan", "what is the weather")
    assert mc.memory_consent_report()["unhonoured_deletion_requests"] == 0


def test_the_count_reaches_the_integrity_surface():
    import inspect

    from core.runtime import health_contract

    source = inspect.getsource(health_contract)
    assert "memory_consent_report" in source
    assert '"memory_consent"' in source


def test_the_dead_predicate_is_gone():
    assert not hasattr(mc, "is_forget_command"), (
        "is_forget_command is back: five literal sentences with no caller, "
        "one of them a past test case"
    )
