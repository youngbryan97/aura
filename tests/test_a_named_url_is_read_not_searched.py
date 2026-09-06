"""When the person names a document, that document is the evidence.

LIVE, 2026-08-20. Asked to read a specific API endpoint, the required-evidence
step searched the web for the endpoint's own address. The results described
somebody else's example for New York, and the reply spent itself arguing with
its own evidence: "the results I have are for New York City ... but you asked
for 64.15, -21.94."
"""

from __future__ import annotations

import pytest

from core.phases.response_generation import _first_named_url


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        (
            "read https://api.open-meteo.com/v1/forecast?latitude=64.15&current=temp and tell me",
            "https://api.open-meteo.com/v1/forecast?latitude=64.15&current=temp",
        ),
        ("see https://example.com/docs.", "https://example.com/docs"),
        ("check http://x.test/a/b) now", "http://x.test/a/b"),
        ("first https://one.test then https://two.test", "https://one.test"),
        ("what is 2+2", ""),
        ("the file /tmp/x is here", ""),
        ("ftp://legacy.test/file", ""),
        ("", ""),
    ],
)
def test_the_url_somebody_typed(message: str, expected: str) -> None:
    assert _first_named_url(message) == expected


def test_sentence_punctuation_is_not_part_of_the_address() -> None:
    for suffix in (".", ",", ";", ":", "!", "?"):
        assert _first_named_url(f"go to https://a.test/b{suffix}") == "https://a.test/b"


def test_the_evidence_step_tries_the_document_before_the_search() -> None:
    """Order matters: a search that has already run has already leaked."""
    from pathlib import Path

    source = Path("core/phases/response_generation.py").read_text(encoding="utf-8")
    body = source[source.index("named_url = _first_named_url(visible_objective)") :]
    assert body.index("_fetch_named_url_evidence") < body.index('skill_name = "web_search"')


def test_reading_a_named_document_does_not_wait_for_a_search_turn() -> None:
    """The gap this closed.

    The chat lane stopped treating a message with an address in it as a
    search turn, correctly. This method returned at the same flag, so nothing
    was fetched at all and she told the person the fetch had failed.
    """
    from pathlib import Path

    source = Path("core/phases/response_generation.py").read_text(encoding="utf-8")
    method = source[source.index("async def _execute_required_search_evidence") :]
    method = method[: method.index("\n    async def ", 10)]
    assert method.index("_fetch_named_url_evidence") < method.index(
        'if not getattr(contract, "requires_search", False):'
    )


def test_injected_evidence_cannot_become_a_user_named_address() -> None:
    """The assembled objective may contain source URLs the user never typed."""
    from pathlib import Path

    source = Path("core/phases/response_generation.py").read_text(encoding="utf-8")
    method = source[source.index("async def _execute_required_search_evidence") :]
    method = method[: method.index("\n    async def ", 10)]
    assert "named_url = _first_named_url(visible_objective)" in method
    assert 'runtime_context.get("visible_user_message")' in method


def test_the_fetch_falls_back_rather_than_leaving_the_turn_empty() -> None:
    from pathlib import Path

    source = Path("core/phases/response_generation.py").read_text(encoding="utf-8")
    body = source[source.index("async def _fetch_named_url_evidence") :]
    body = body[: body.index("\n    @staticmethod")]
    assert "return False" in body
    assert 'if not bool(payload.get("ok")):' in body


def test_the_desktop_lane_does_not_search_for_an_address() -> None:
    """The same rule the local-file case already had: the bytes are at the
    address, so no search result can be better evidence than the document."""
    from chat_lane_support import chat_lane_source

    source = chat_lane_source()
    gate = source[source.index("def _chat_requires_search") :] if "def _chat_requires_search" in source else source
    gate = source[source.index("# A file on this disk is not a live-search question.") :]
    gate = gate[: gate.index("contract = _resolve_chat_response_contract(user_message)")]
    assert "first_named_url" in gate


def test_the_reader_is_shared_by_both_lanes() -> None:
    """One binding: the chat lane and the evidence phase read the same one."""
    from core.intent.opaque_spans import first_named_url
    from core.phases.response_generation import _first_named_url

    assert first_named_url is _first_named_url
