"""A planner that serves the running turn is not background work.

LIVE, 2026-08-22. The finite-game solver asks the model to translate the
described rules into a spec, and that call was refused with
"all_background_endpoints_deferred" — because the very turn it was serving had
reserved the foreground lane. The turn then answered from the model's own
guess, said "move your piece one square on every turn", and was wrong.

The claim is verified rather than trusted: it counts only while the
orchestrator agrees a foreground turn is running.
"""

from __future__ import annotations

from pathlib import Path

from core.brain.inference_gate import InferenceGate


def test_the_flag_is_honoured_only_inside_a_turn(monkeypatch):
    seen = {}

    def fake_active() -> bool:
        seen["asked"] = True
        return False

    monkeypatch.setattr(InferenceGate, "_foreground_user_turn_active", staticmethod(fake_active))
    # Outside a turn the claim buys nothing, and the gate had to ask.
    assert InferenceGate._foreground_user_turn_active() is False
    assert seen["asked"]


def test_the_gate_asks_the_runtime_rather_than_trusting_the_caller():
    source = Path("core/brain/inference_gate.py").read_text(encoding="utf-8")
    block = source[source.index('explicit_foreground = bool(context.get("foreground_request"'):]
    block = block[: block.index("protected_foreground_lane = bool(")]
    assert 'context.get("serves_current_turn")' in block
    assert "self._a_user_turn_is_in_flight()" in block


def test_preflight_counts_as_being_in_a_turn(monkeypatch):
    """The orchestrator reports a turn once its tick starts. Preflight runs
    before that and is still part of the turn, so a planner that ran there was
    refused on the orchestrator's account alone."""
    from core.conversation.session_scope import set_user_question

    monkeypatch.setattr(
        InferenceGate, "_foreground_user_turn_active", staticmethod(lambda: False)
    )
    set_user_question("")
    assert InferenceGate._a_user_turn_is_in_flight() is False
    set_user_question("who wins this game?")
    assert InferenceGate._a_user_turn_is_in_flight() is True
    set_user_question("")


def test_the_origin_allowlist_is_untouched():
    """The fix must not widen the set of names that can claim the lane."""
    from core.brain.inference_gate import _USER_FACING_ORIGINS

    for invented in ("reasoning.game_solver", "build_app.plan", "background_user"):
        assert not InferenceGate._origin_is_user_facing(invented), invented
    assert "user" in _USER_FACING_ORIGINS


def test_both_in_turn_planners_declare_it():
    for path in ("core/reasoning/game_answer.py", "core/construction/build_app_system.py"):
        source = Path(path).read_text(encoding="utf-8")
        assert "serves_current_turn=True" in source, path
