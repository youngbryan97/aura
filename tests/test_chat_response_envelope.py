"""A real answer must not be destroyed by its envelope.

Measured live on the desktop path. Aura was asked to open Notes and write a
three-sentence note about orcas. She DID it — the note is on disk:

    Orcas are apex predators known for their intelligence and complex social
    structures. They communicate through a variety of clicks, whistles, and
    pulsed calls. Each pod has its own distinct dialect.

The salvage path then returned the reply as a bare `str`, the delivery boundary
checked `isinstance(response, JSONResponse)`, and the person received:

    The chat route returned an unsupported response format.

Worst possible polarity: the REFUSAL path in the same function was correctly
formed, so a turn that failed reported cleanly while a turn that SUCCEEDED
reported a transport error.
"""

from __future__ import annotations

from tests.chat_lane_support import chat_lane_source


def test_the_salvage_path_returns_a_json_response():
    """`_servable_draft_or_none` returns a str; the handler must not pass it through."""

    source = chat_lane_source()
    start = source.index("async def _fail_closed_degraded_desktop_reply")
    # The salvage function ends at the next sibling definition.
    end = source.index("async def _finalize_fastpath", start)
    body = source[start:end]

    assert "return salvaged" not in body, (
        "the salvaged draft is a bare str; returning it raw makes the delivery "
        "boundary replace a real answer with a 500"
    )
    assert "served_repairable_draft" in body
    # Every exit carries an envelope.
    returns = [
        line.strip()
        for line in body.splitlines()
        if line.strip().startswith("return ")
    ]
    assert returns, "the salvage function must still return something"
    for statement in returns:
        assert (
            statement.startswith("return JSONResponse")
            or statement == "return served"  # already a JSONResponse | None
            # The end of a turn with no reply of hers, typed -> JSONResponse.
            or statement.startswith("return await _chat_no_reply.what_a_turn_with_no_reply_returns(")
        ), f"non-JSONResponse exit from the salvage path: {statement!r}"


def test_the_delivery_boundary_delivers_text_instead_of_erasing_it():
    """The class-level guard: an envelope defect must not cost the answer."""

    source = chat_lane_source()
    marker = "if not isinstance(response, JSONResponse):"
    assert marker in source
    guard = source[source.index(marker) : source.index(marker) + 2600]

    assert "chat_response_envelope_coerced" in guard, (
        "a handler that returned reply TEXT has produced an answer; the boundary "
        "must deliver it rather than ship a 500"
    )
    assert "chat.response_envelope" in guard, (
        "the envelope defect must still be recorded, not silently tolerated"
    )
    # A genuinely unusable return still fails closed.
    assert "chat_response_format_rejected" in guard


def test_speaking_unprompted_does_not_raise_a_name_error():
    """The initiative path crashed the moment she used it.

    `_record_output_delivery` read `content` as a free name inside the
    visible-presence branch — a NameError on exactly the path the method exists
    to record. Both call sites already had the text in scope and never passed
    it, so every unprompted visible message would have died recording itself.
    """
    from core.autonomy.proactive_presence import _INITIATIVE_LOG, ProactivePresence

    presence = ProactivePresence.__new__(ProactivePresence)
    presence._last_output_time = 0.0
    presence.orchestrator = None
    presence._outputs_this_hour = 0
    presence._consecutive_unprompted = 0
    presence._last_visible_output_time = 0.0

    before = int(_INITIATIVE_LOG.get("count", 0))
    presence._record_output_delivery(
        visible_presence=True, content="I was thinking about orcas."
    )
    assert int(_INITIATIVE_LOG.get("count", 0)) == before + 1
    assert _INITIATIVE_LOG.get("last_text") == "I was thinking about orcas."
    assert presence._consecutive_unprompted == 1

    # The non-visible path must stay callable too.
    presence._record_output_delivery(visible_presence=False, content="quiet thought")
