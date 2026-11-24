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


def _handed_off(ceiling: str) -> dict:
    """The context the tool-grounded answer hands the tool loop, for one ceiling.

    The handoff runs for real against a client that records what it was
    given; the size sweep cut it out of `_tool_grounded_answer` and a lift
    moved it into `inference_gate_living_context`.
    """
    import asyncio

    from core.brain.inference_gate import InferenceGate

    given: dict = {}

    class _Client:
        model_path = ""

        async def think_and_act(self, **kwargs):
            given.update(kwargs["context"])
            return {"text": "built it"}

    asyncio.run(
        InferenceGate._tool_grounded_answer_part_1(
            ceiling,
            _Client(),
            0,
            [],
            "user",
            ["build_app"],
            "build me a small web app, one self-contained HTML file",
            60.0,
            {"build_app": {}},
        )
    )
    return given


def test_the_permission_model_is_told_what_was_asked_for() -> None:
    given = _handed_off("read_write_artifacts")
    assert given["user_explicitly_authorized"] is True
    assert given["authorised_effect_scope"] == "read_write_artifacts"


def test_consent_covers_only_the_effect_that_was_named() -> None:
    """Writing a file was asked for. Sending, deleting and spending were not.

    Asserted on what the handoff hands over for each ceiling rather than on
    the spelling of the comparison. This once matched the literal
    `ceiling == "read_write_artifacts"`, which stopped existing the day the
    two ceilings moved into names imported from the one place that decides
    them.
    """
    from core.phases.response_contract import (
        _REQUESTED_ARTIFACT_CEILING,
        _SELF_SERVICE_CEILING,
    )

    for ceiling in (_SELF_SERVICE_CEILING, _REQUESTED_ARTIFACT_CEILING):
        assert _handed_off(ceiling)["user_explicitly_authorized"] is True, ceiling
    # The effects a request does not carry consent for, whatever it asked.
    for ceiling in ("external_io", "privileged_mutation"):
        assert _handed_off(ceiling)["user_explicitly_authorized"] is False, ceiling


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


def test_an_order_to_work_a_program_here_carries_consent_for_her_hands_on_it(monkeypatch) -> None:
    """LIVE 24 November 2026: "Open En Croissant and play a game of chess against Stockfish" ran under the
    self-service ceiling; her reading chose desktop_task and the ceiling withheld it, and she answered in words."""
    import core.epistemics.where_a_request_is_done as where
    from core.phases.response_contract import _REQUESTED_DESKTOP_CEILING, requested_effect_ceiling

    monkeypatch.setattr(where, "installed_program_names", lambda: ["en-croissant", "Chess"])
    ceiling, scopes = requested_effect_ceiling(
        "Open En Croissant and play a game of chess against Stockfish. Tell me each move as you go.")
    assert ceiling == _REQUESTED_DESKTOP_CEILING and "foreground_desktop_control" in scopes
    assert _handed_off(_REQUESTED_DESKTOP_CEILING)["user_explicitly_authorized"] is True
    # Naming the program without an order, or the game without the program, asks for no hands on anything.
    for not_an_order in ("What is En Croissant?", "I like a game of chess now and then."):
        assert requested_effect_ceiling(not_an_order)[0] != _REQUESTED_DESKTOP_CEILING, not_an_order
