"""Cognition uses independently executed examples and exact floor arithmetic."""
import pytest

from core.brain.verifiers.code_engine import CodeTruthEngine
from core.brain.verifiers.math_engine import _derive_exact_answer, _final_answer_matches
from core.reasoning.arithmetic_on_the_floor import integer_expression


@pytest.mark.parametrize("expression,expected", [("(3+4)*8-1", 55), ("-17 % 4", 3), ("-17//4", -5),
                                                 ("3**40", 3**40), ("9007199254740993+1", 9007199254740994)])
def test_integer_arithmetic_runs_exactly_on_the_floor(expression, expected):
    assert integer_expression(expression) == expected


@pytest.mark.parametrize("expression", ["__import__('os')", "7/2", "1//0", "2**100000", "True+1", "2**-1"])
def test_unsupported_or_unbounded_arithmetic_remains_unknown(expression):
    assert integer_expression(expression) is None


def test_a_float_collision_cannot_verify_the_wrong_integer():
    assert not _final_answer_matches("Answer: 9007199254740992", "9007199254740993")
    assert _final_answer_matches("Answer: 9.007199254740993e15", "9007199254740993")
    assert not _final_answer_matches("Answer: 1e999999999999", "1")
    assert _final_answer_matches("Answer: 1/8", "0.125")
    assert not _final_answer_matches("Answer: 0.333333", "1/3")
    assert not _final_answer_matches("Answer: 1,2", "12")
    assert not _final_answer_matches("Answer: 1 or 2", "2")
    assert _final_answer_matches("Answer: 12,345", "12345")


def test_a_partial_expression_cannot_supply_an_oracle_for_a_different_question():
    assert _derive_exact_answer("What is 2*3+4?")[1] == "10"
    assert _derive_exact_answer("What is -3 times 4?")[1] == "-12"
    assert _derive_exact_answer("Compare 2*3 with 4*5") is None
    assert _derive_exact_answer("Compute 1,2") is None


@pytest.mark.asyncio
async def test_the_amplifier_code_verifier_executes_external_examples():
    examples = [{"name": "mixed signs", "function": "add", "args": [-3, 7], "expected": 4},
                {"name": "zero", "function": "add", "args": [0, 0], "expected": 0}]
    engine = CodeTruthEngine(run_ruff=False)
    wrong = await engine.verify("def add(a,b): return a-b", context={"function_examples": examples})
    assert wrong.checked and not wrong.ok
    right = await engine.verify("def add(a,b): return a+b", context={"function_examples": examples})
    assert right.checked and right.ok and right.detail["function_checks"]["verified"]
    unknown = await engine.verify("def add(a,b): return a+b", context={"function_examples": []})
    assert not unknown.checked


def test_a_corner_departure_is_attributed_to_the_first_edge_crossed():
    from core.agency.when_motion_breaks_a_rule import departure_edge

    event = {"x": -40, "y": -60, "vx": -200, "vy": -400, "width": 10, "height": 10,
             "last_visible": [1, 40], "last_seen_at": 0, "at": 0.3}
    assert departure_edge(event, (100, 160)) == "left"


def test_a_missing_track_predicted_through_a_wall_is_not_a_witnessed_escape():
    from core.agency.when_motion_breaks_a_rule import MotionChecks
    from types import SimpleNamespace

    moves = SimpleNamespace(shape=(100, 160))
    event = {"what": "gone", "thing": 45, "x": 144, "y": -8, "vx": 77, "vy": -69,
             "width": 3, "height": 3, "last_visible": [124, 10], "last_seen_at": 1, "at": 1.26}
    checks = MotionChecks(frozenset({"top", "bottom"}), "source reflection contract")
    assert not checks.see(moves, [event], 1.26, 1)
    assert checks.unconfirmed[0]["edge"] == "top"
    assert checks.see(moves, [{**event, "last_visible": [124, -1]}], 1.26, 1)[0]["position"] == [124, -1]


def test_visible_clipped_motion_can_reveal_a_breach_before_the_track_is_lost():
    from core.agency.when_motion_breaks_a_rule import MotionChecks
    from types import SimpleNamespace

    body = SimpleNamespace(number=3, seen=1, path=[(1, 60, -2)], vx=20, vy=-50)
    moves = SimpleNamespace(shape=(100, 160), things={3: body})
    checks = MotionChecks(frozenset({"top"}), "source reflection contract")
    assert checks.see(moves, [], 1, 1)[0]["evidence"] == "visible centre outside required boundary"
    assert not MotionChecks().see(moves, [], 1, 1)
    assert not checks.see(moves, [{"what": "new screen"}], 1, 1)


def test_a_runtime_interruption_cannot_be_labelled_a_game_loss():
    from core.skills.sovereign_browser_drawing import _how_the_run_went
    from types import SimpleNamespace

    reflexes = SimpleNamespace(ending_words="You lost", ending_parts=[],
               stretches=[{"pictures": 10, "runtime_checks": {"violations": [{"edge": "top"}]}}],
               what_it_came_to=lambda: "runtime contract violated")
    assert _how_the_run_went(reflexes, {})["ended"] == "interrupted"


@pytest.mark.asyncio
async def test_visible_repair_progress_reaches_the_current_durable_delivery(monkeypatch):
    import asyncio
    import core.runtime.chat_delivery_progress as progress
    import core.skills.screen_pursuit as pursuit
    from core.skills.repairing_a_program import _said

    published = []
    async def report(**event):
        published.append(event)
    monkeypatch.setattr(pursuit, "_tell", lambda line: None)
    monkeypatch.setattr(progress, "current_chat_delivery_identity", lambda: {"turn_id": "owned"})
    monkeypatch.setattr(progress, "report_chat_delivery_progress", report)
    _said("Saved the verified repair.")
    await asyncio.sleep(0)
    assert published == [{"phase": "executing", "message": "Saved the verified repair."}]


@pytest.mark.asyncio
async def test_boolean_integer_alias_is_not_a_passing_function_result():
    from core.self_modification.checking_python import FunctionExample, check_python, examples_in

    checked = await check_python("def identity(x): return True", [FunctionExample(name="integer", function="identity", args=[1], expected=1)])
    assert checked["cases"][0]["verdict"] == "wrong"
    assert examples_in('def f(x):\n    """>>> f(**{\"x\": 1})\n    1\n    """\n    return x') == []
    assert len(examples_in('def f(x):\n    """>>> f(1)\n    1\n    """\n    return (x +')) == 1


@pytest.mark.asyncio
async def test_coding_checks_reach_the_existing_amplifier(monkeypatch):
    from core.brain.reasoning_strategies import ReasoningStrategies, StrategyType
    from core.brain.reasoning_amplifier_v2 import ReasoningAmplifierV2
    from types import SimpleNamespace

    requests = []
    async def amplify(self, request):
        requests.append(request)
        return SimpleNamespace(answer="def f(x): return x", verified=True, confidence=1, calibrated=False,
                               receipt=SimpleNamespace(mode="normal", num_candidates=1, verifiers_run=["code"],
                                                       known_failures=[], to_dict=lambda: {}))
    monkeypatch.setattr(ReasoningAmplifierV2, "amplify", amplify)
    async def generate(prompt, **kwargs):
        return "def f(x): return x"
    checks = [{"name": "one", "function": "f", "args": [1], "expected": 1}]
    result = await ReasoningStrategies(generate).execute("Write f(x)", strategy=StrategyType.CONSISTENCY,
                purpose="coding", verification_context={"function_examples": checks})
    assert requests[0].task_type == "code" and requests[0].context["function_examples"] == checks
    assert result.metadata["amplifier_v2"]


@pytest.mark.asyncio
async def test_saved_partial_repair_has_its_own_verified_file_effect(tmp_path, monkeypatch):
    from core.self_modification.saving_a_verified_repair import save_repair
    from core.runtime.file_write_gateway import get_file_write_gateway
    import core.conversation.surface_disposition as disposition

    path = tmp_path / "program.py"
    path.write_text("before")
    receipts = []
    async def write(p, text, **kwargs):
        p.write_text(text)
    monkeypatch.setattr(get_file_write_gateway(), "write_text_async", write)
    monkeypatch.setattr(disposition, "record_tool_receipt", lambda *a, **kw: receipts.append(kw))
    backup = await save_repair(path, "before", "after")
    assert receipts[0]["ok"] and receipts[0]["effect_observed"]
    assert receipts[0]["object_ref"] == str(path) and "sha256=" in receipts[0]["evidence"]
    assert __import__("pathlib").Path(backup).read_text() == "before"
    with pytest.raises(ValueError, match="newer work"):
        await save_repair(path, "before", "overwrite")


@pytest.mark.asyncio
async def test_a_failed_task_does_not_erase_an_observed_file_receipt(tmp_path, monkeypatch):
    from core.self_modification.saving_a_verified_repair import save_repair
    from core.runtime.file_write_gateway import get_file_write_gateway
    from core.conversation.turn_evidence_custody import bind_turn_evidence_custody
    from core.conversation.surface_disposition import record_tool_receipt, turn_tool_receipts

    path = tmp_path / "program.py"
    path.write_text("before")
    async def write(p, text, **kwargs):
        p.write_text(text)
    monkeypatch.setattr(get_file_write_gateway(), "write_text_async", write)
    with bind_turn_evidence_custody(session_id="repair-proof", turn_id="partial"):
        await save_repair(path, "before", "after")
        assert record_tool_receipt("repair_a_program", ok=False, action="execute",
                                   effect_observed=False, evidence="requested play failed")
        receipts = turn_tool_receipts()
        assert any(r["tool"] == "artifact_repair" and r["ok"] and r["effect_observed"] for r in receipts)
        assert any(r["tool"] == "repair_a_program" and not r["ok"] for r in receipts)
