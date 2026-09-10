"""A confirmation prompt for the thing that was just asked for.

LIVE, 2026-08-21. "build me a small web app… Keep it one self-contained
file" reached build_app, which called it, and the permission model refused:

    [think_and_act] turn=1 tool=build_app
    error:Permission denied: Requires user confirmation

The model already asks whether the person pre-approved this class of action —
`context["user_explicitly_authorized"]` — and nothing in the runtime ever
answered it.

The boundary matters more than the fix. Consent carried by a request covers
the effect that request named, and nothing else.
"""

from __future__ import annotations

from pathlib import Path

GATE = Path(__file__).resolve().parents[1] / "core" / "brain" / "inference_gate.py"


def _handoff_body() -> str:
    source = GATE.read_text(encoding="utf-8")
    body = source[source.index("async def _tool_grounded_answer") :]
    return body[: body.index("\n    async def ", 10)]


def test_the_permission_model_is_told_what_was_asked_for() -> None:
    assert '"user_explicitly_authorized"' in _handoff_body()


def test_consent_covers_only_the_effect_that_was_named() -> None:
    """Writing a file was asked for. Sending, deleting and spending were not.

    Asserted on the SET the handoff admits rather than on the spelling of the
    comparison. This matched the literal `ceiling == "read_write_artifacts"`,
    which stopped existing the day the two ceilings moved into names imported
    from the one place that decides them — and a literal assertion that stops
    matching does not fail loudly, it fails on the next full run and looks
    like a regression.
    """
    from core.brain import inference_gate
    from core.phases.response_contract import (
        _REQUESTED_ARTIFACT_CEILING,
        _SELF_SERVICE_CEILING,
    )

    body = _handoff_body()
    assert "user_explicitly_authorized" in body

    admitted = {
        inference_gate._SELF_SERVICE_EFFECT_CEILING,
        inference_gate._REQUESTED_ARTIFACT_EFFECT_CEILING,
    }
    assert admitted == {_SELF_SERVICE_CEILING, _REQUESTED_ARTIFACT_CEILING}
    # The effects a request does not carry consent for, whatever it asked.
    assert "external_io" not in admitted
    assert "privileged_mutation" not in admitted
    # Quoted, because the handoff's own comment names these as the effects it
    # does NOT authorise. What must not appear is a string literal.
    assert '"external_io"' not in body
    assert '"privileged_mutation"' not in body


def test_an_ordinary_turn_carries_no_consent() -> None:
    from core.phases.response_contract import requested_effect_ceiling

    for ordinary in ("how are you today?", "what is 2 + 2", "read /etc/hosts"):
        ceiling, _scopes = requested_effect_ceiling(ordinary)
        assert ceiling != "read_write_artifacts"


def test_a_build_request_carries_consent_to_write_one_file() -> None:
    from core.phases.response_contract import requested_effect_ceiling

    ceiling, _scopes = requested_effect_ceiling(
        "build me a small web app, one self-contained HTML file"
    )
    assert ceiling == "read_write_artifacts"


def test_the_permission_model_still_asks_when_nobody_authorised() -> None:
    """The gate itself is untouched: without consent it still requires it."""
    model = Path("core/capabilities/permission_model.py").read_text(encoding="utf-8")
    assert 'context.get("user_explicitly_authorized", False)' in model
    assert "Requires user confirmation" in model
