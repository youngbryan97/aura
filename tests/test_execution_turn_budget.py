"""An execution turn must never be shrunk below the plan it has to emit.

Two independent causes, both measured on the live desktop path, both of which
made the demo fail while looking like flakiness:

1. ARTIFACT SHAPE READ AS REPLY SHAPE.
   "Open the Notes app and write a new note with three sentences about humpback
   whales" parses to sentence_count=3. That was applied to her CHAT REPLY: it
   clamped the budget to max_tokens=288 — far too small for a multi-step desktop
   plan — so she produced conversational filler, nothing executed, and the gate
   then vetoed the filler for not matching the three-sentence shape it had
   itself imposed. Removing the phrase made the identical request plan and
   execute, and the note is on disk. Every demo instruction carries such a
   clause ("3 articles", "a coherent summary", "a short note").

2. MEMORY PRESSURE CLAMPING THE PLAN.
   The unified-memory clamp cuts max_tokens regardless of what the turn needs.
   For a conversational reply that costs words; for an execution turn it costs
   the whole task. This is why the demo degrades exactly when a screen recorder
   is running — the recorder raises pressure, the cap drops, the plan no longer
   fits.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace

from core.brain.llm.mlx_client import _apply_memory_pressure_generation_controls
from tests.source_contract import family_text


def test_pressure_still_clamps_an_ordinary_reply():
    """The clamp must keep doing its job where it costs only words."""
    controlled = _apply_memory_pressure_generation_controls(
        {
            "max_tokens": 2048,
            "clean_user_surface_contract": True,
            "clean_user_surface_recurrent_loops": 2,
        },
        SimpleNamespace(max_token_cap=192),
    )
    assert controlled["max_tokens"] == 192
    assert controlled["clean_user_surface_recurrent_loops"] == 1


def test_question_shaped_completion_floor_survives_high_pressure():
    controlled = _apply_memory_pressure_generation_controls(
        {
            "max_tokens": 1536,
            "clean_user_surface_contract": True,
            "clean_user_surface_recurrent_loops": 2,
            "user_surface_completion_floor": 1024,
        },
        SimpleNamespace(max_token_cap=192),
    )

    assert controlled["max_tokens"] == 1024
    assert controlled["clean_user_surface_recurrent_loops"] == 2


def test_pressure_cannot_shrink_an_execution_turn_below_its_plan():
    controlled = _apply_memory_pressure_generation_controls(
        {"max_tokens": 1024, "desktop_execution_contract": True},
        SimpleNamespace(max_token_cap=192),
    )
    assert controlled["max_tokens"] >= 1024, (
        "a plan that cannot be expressed is a task that cannot be attempted; "
        "slower is recoverable, silent failure is not"
    )


def test_the_plan_floor_is_a_floor_not_a_blank_cheque():
    controlled = _apply_memory_pressure_generation_controls(
        {"max_tokens": 8192, "desktop_execution_contract": True},
        SimpleNamespace(max_token_cap=192),
    )
    assert controlled["max_tokens"] == 1024, "the floor must still bound the turn"


def test_no_pressure_leaves_the_budget_alone():
    controlled = _apply_memory_pressure_generation_controls(
        {"max_tokens": 2048, "desktop_execution_contract": True},
        SimpleNamespace(max_token_cap=None),
    )
    assert controlled["max_tokens"] == 2048


def test_the_gate_forwards_the_execution_flag_to_the_client():
    """The floor is unreachable unless the flag survives the call boundary."""
    from core.brain import inference_gate

    source = family_text(inference_gate)
    assert 'morpho_kwargs["desktop_execution_contract"] = True' in source, (
        "the clamp runs client-side; without the flag it cannot know a plan is "
        "at stake"
    )


def test_artifact_shape_does_not_cap_the_reply_on_an_execution_turn():
    from core.brain import inference_gate

    source = family_text(inference_gate)
    marker = 'if bool(context.get("desktop_execution_contract", False)):'
    assert marker in source

    # STRUCTURAL, NOT A BYTE WINDOW.
    #
    # This read the 1,200 characters after the marker, so the assertion broke
    # the moment the block grew — which it did, when the plan-token floor and
    # its comment landed inside it. The behaviour was correct the whole time;
    # only the ruler was wrong. What actually matters is the ORDER: the flag
    # must be cleared after the execution-turn marker and before the ceiling
    # it would otherwise impose.
    marker_at = source.index(marker)
    cleared_at = source.index("output_contract_is_user_facing = False", marker_at)
    ceiling_at = source.index("output_contract.hard_token_ceiling is not None", marker_at)
    assert marker_at < cleared_at < ceiling_at, (
        "on an execution turn the shape phrase describes the ARTIFACT; the "
        "executor owns it through document_body, so the reply must keep its "
        "full budget"
    )


def test_the_final_veto_does_not_enforce_artifact_shape_on_the_report():
    from interface.routes import chat as chat_routes

    source = inspect.getsource(chat_routes._enforce_final_requested_output_contract)
    assert "desktop_execution_contract" in source
    assert "artifact_shape_not_reply_shape" in source, (
        "a completed task was vetoed for reporting itself in the wrong number "
        "of sentences"
    )
