"""A capability the person asked for by name is not discretionary spending.

LIVE, 2026-08-20. "build me a small web app" reached capability selection with
build_app ranked first — and build_app is a heavy tool, live vitality was
0.683, and the tier below 0.8 caps tool cost at 2. It was dropped without a
word. The turn spent itself reaching for code_repl instead, which governance
then vetoed as needing confirmation, and the person got "I couldn't get to an
answer I'd stand behind."

The metabolic throttle exists so Aura does not CHOOSE expensive work while
tired. It is not a veto on what was asked for. Panic is still absolute.
"""

from __future__ import annotations

from pathlib import Path

ENGINE = Path(__file__).resolve().parents[1] / "core" / "capability_engine.py"
WIRING = Path(__file__).resolve().parents[1] / "core" / "brain" / "llm" / "runtime_wiring.py"


def _fetcher_body() -> str:
    source = ENGINE.read_text(encoding="utf-8")
    body = source[source.index("def _tool_definition_for_skill") :]
    return body[: body.index("\n    def ", 10)]


def test_the_throttle_does_not_veto_what_was_asked_for() -> None:
    body = _fetcher_body()
    assert "and not requested" in body
    assert "requested: bool = False" in body


def test_the_drop_is_no_longer_silent() -> None:
    """It vanished without a word, which is why it took four hours to find."""
    body = _fetcher_body()
    assert "[COST]" in body
    assert "withheld" in body


def test_panic_still_refuses_everything() -> None:
    source = ENGINE.read_text(encoding="utf-8")
    block = source[source.index("if health_score < 0.3:") :]
    block = block[: block.index("elif health_score < 0.6:")]
    assert "allowed_max_cost = 0" in block
    assert "Nothing is exempt here" in block


def test_the_decided_set_is_fetched_by_name() -> None:
    """Ranking a set that was already decided is how a wanted capability
    survived selection and vanished one call later."""
    source = ENGINE.read_text(encoding="utf-8")
    body = source[source.index("def select_tool_definitions") :]
    body = body[: body.index("\n    def ", 10)]
    assert "requested: Sequence[str] | None = None" in body
    assert "asked_for" in body
    assert "requested=name in asked_for" in body


def test_the_tool_map_hands_over_what_it_decided() -> None:
    body = WIRING.read_text(encoding="utf-8")
    body = body[body.index("def build_agentic_tool_map") :]
    body = body[: body.index("\ndef ", 10)]
    assert "requested=sorted(wanted)" in body


def test_an_empty_ranking_is_not_an_empty_answer() -> None:
    """LIVE: "skill=improve_own_code,build_app,internal_sandbox,http_request,
    code_repl offered=NONE (no tool definition)".

    The working set was decided correctly and the selector returned [] before
    it ever reached the requested names, because the relevance ranker found
    nothing.
    """
    source = ENGINE.read_text(encoding="utf-8")
    body = source[source.index("def select_tool_definitions") :]
    body = body[: body.index("\n    def ", 10)]
    assert "if not ordered and not asked_for:" in body
    assert body.index("asked_for = {") < body.index("if not ordered and not asked_for:")


def test_the_planner_asks_for_what_the_goal_needs() -> None:
    """LIVE: "[COST] build_app (cost 3) withheld: this turn allows 2."

    The chat lane exempts a capability the person asked for from the metabolic
    throttle. The engine planning that same request did not, so the tool built
    for the job was withheld from the plan that needed it.
    """
    engine = Path(__file__).resolve().parents[1] / "core" / "agency" / "autonomous_task_engine.py"
    source = engine.read_text(encoding="utf-8")
    body = source[source.index("if hasattr(cap, \"select_tool_definitions\")") :]
    body = body[: body.index("selected_defs = list(cap.get_tool_definitions()")]
    assert "derive_capability_set(goal)" in body
    assert "requested=requested_for_goal" in body
