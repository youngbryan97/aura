"""A skill result reaches the prompt with its result in it.

LIVE, 2026-08-20. http_request returned {ok, url, status, text, ...}. The
grounding sanitizer kept `url` and dropped `text`, and the renderer showed
what it had been given, so the evidence block read

    [SKILL RESULT: http_request] ✅ Url: https://api.open-meteo.com/…

and nothing else. She was handed proof that a fetch had happened with no
trace of what it said, and answered 10.5 in the tool loop and 12.4 in the
reply against a real 11.9. Nothing was hallucinating; there was nothing to
read.
"""

from __future__ import annotations

from core.phases.response_generation import (
    _EVIDENCE_RENDERED_KEYS,
    _EVIDENCE_SCALAR_KEYS,
    ResponseGenerationPhase,
)

FETCH = {
    "ok": True,
    "url": "https://api.open-meteo.com/v1/forecast?latitude=64.15",
    "status": 200,
    "text": '{"current":{"time":"2026-08-20T12:15","temperature_2m":11.9}}',
    "bytes": 325,
}


def test_the_number_survives_the_sanitizer() -> None:
    compact = ResponseGenerationPhase._sanitize_grounding_payload(FETCH)
    assert "11.9" in str(compact.get("text") or "")


def test_the_number_survives_the_renderer() -> None:
    compact = ResponseGenerationPhase._sanitize_grounding_payload(FETCH)
    block = ResponseGenerationPhase._render_skill_result_block(
        skill_name="http_request", payload=compact
    )
    assert "11.9" in block
    assert "Url:" in block


def test_an_address_alone_is_never_the_whole_block() -> None:
    """The failure shape: one recognised key suppressing the result."""
    compact = ResponseGenerationPhase._sanitize_grounding_payload(FETCH)
    block = ResponseGenerationPhase._render_skill_result_block(
        skill_name="http_request", payload=compact
    )
    body = block.split("Url:", 1)[1]
    assert len(body.strip().splitlines()) > 1


def test_both_readers_share_one_vocabulary() -> None:
    """Written twice, they drifted: only the renderer learned about "text"."""
    for key in ("text", "body"):
        assert key in _EVIDENCE_SCALAR_KEYS
        assert key in _EVIDENCE_RENDERED_KEYS


def test_a_search_result_still_renders_as_before() -> None:
    payload = {
        "ok": True,
        "query": "who wrote Solaris",
        "results": [{"title": "Solaris", "url": "https://x", "snippet": "Stanisław Lem"}],
    }
    compact = ResponseGenerationPhase._sanitize_grounding_payload(payload)
    block = ResponseGenerationPhase._render_skill_result_block(
        skill_name="web_search", payload=compact
    )
    assert "Stanisław Lem" in block
    assert "who wrote Solaris" in block
