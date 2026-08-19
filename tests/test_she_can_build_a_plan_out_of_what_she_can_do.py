"""Planning arrived as an injected lambda because a step carries a callable.

A language model cannot write a callable, so nothing in the codebase could
turn an intention into a plan — every pursuit loop took its plan from
somewhere else, or ran without one.

It does not have to. Every registered skill declares a name, a description
and a JSON schema for its arguments, which is a typed vocabulary. The model
names an action and its arguments; the architecture checks the call against
the schema and compiles what survives into steps bound to the same capability
engine tool dispatch uses.

These tests hold that: the vocabulary is what she can really do, a call that
does not check out is reported as the error it is rather than asked again in
other words, and a compiled step really runs the skill it names.
"""
from __future__ import annotations

import pytest

from core.agency.plan_synthesis import (
    ActionSpec,
    ProposedCall,
    action_space,
    check_call,
    compile_step,
    read_calls,
    synthesize_plan,
)


class _Meta:
    def __init__(self, name, description, schema, scope="reasoning", enabled=True):
        self.name = name
        self.description = description
        self._schema = schema
        self.effect_scope = scope
        self.enabled = enabled
        self.requirements = type("R", (), {"requires_approval": False})()

    def schema_def(self):
        return self._schema


class _Engine:
    def __init__(self, skills=None):
        self.skills = skills or {}
        self.ran = []
        self.result = {"ok": True}

    async def execute(self, name, params, context=None):
        self.ran.append((name, dict(params), dict(context or {})))
        return self.result


def _engine():
    return _Engine(
        {
            "web_search": _Meta(
                "web_search",
                "search the web",
                {"type": "object", "properties": {"query": {}}, "required": ["query"]},
            ),
            "read_file": _Meta(
                "read_file",
                "read a file",
                {"type": "object", "properties": {"path": {}}, "required": ["path"]},
            ),
            "delete_everything": _Meta("delete_everything", "no", {}, scope="filesystem_write"),
            "retired": _Meta("retired", "off", {}, enabled=False),
        }
    )


def _thinks(*replies):
    queue = list(replies)
    asked = []

    async def think(objective, evidence):
        asked.append(list(evidence))
        return queue.pop(0) if queue else replies[-1]

    think.asked = asked
    return think


def test_the_vocabulary_is_what_she_can_really_do():
    space = action_space(engine=_engine())
    names = [spec.name for spec in space]
    assert "web_search" in names and "read_file" in names
    assert "retired" not in names, "a disabled skill is not something she can do"
    assert "delete_everything" not in names, "world-changing actions are not offered by default"


def test_world_changing_actions_are_available_when_the_caller_says_so():
    space = action_space(engine=_engine(), allow_world_changing=True)
    assert "delete_everything" in [spec.name for spec in space]


def test_an_action_states_the_arguments_it_takes():
    space = {spec.name: spec for spec in action_space(engine=_engine())}
    assert space["web_search"].arguments() == ["query"]
    assert space["web_search"].required() == ["query"]
    assert "takes: query" in space["web_search"].as_line()


def test_calls_are_read_out_of_a_reply_in_the_shapes_a_model_actually_writes():
    assert read_calls('[{"action": "web_search", "args": {"query": "x"}}]')[0].action == "web_search"
    assert read_calls('{"steps": [{"name": "read_file", "params": {"path": "p"}}]}')[0].args == {"path": "p"}
    assert read_calls("no json here") == []


def test_an_action_she_does_not_have_is_named_as_such():
    space = action_space(engine=_engine())
    spec, problems = check_call(ProposedCall(action="fly", args={}), space)
    assert spec is None
    assert problems == ["'fly' is not something she can do"]


def test_a_missing_argument_is_reported_as_the_error_it_is():
    space = action_space(engine=_engine())
    _spec, problems = check_call(ProposedCall(action="web_search", args={}), space)
    assert problems == ["web_search needs 'query' and it was not given"]


def test_an_argument_that_does_not_exist_is_reported_too():
    space = action_space(engine=_engine())
    _spec, problems = check_call(
        ProposedCall(action="web_search", args={"query": "x", "colour": "blue"}), space
    )
    assert problems == ["web_search takes no argument called 'colour'"]


@pytest.mark.asyncio
async def test_a_compiled_step_really_runs_the_skill_it_names():
    engine = _engine()
    spec = ActionSpec(name="web_search", description="search", schema={})
    step = compile_step(spec, ProposedCall(action="web_search", args={"query": "otters"}), engine=engine)
    assert step.name == "web_search(query)"
    assert step.approach == "plan_synthesis"
    assert await step.action() is True
    assert engine.ran == [("web_search", {"query": "otters"}, {"requested_via": "plan_synthesis"})]


@pytest.mark.asyncio
async def test_a_failing_skill_makes_the_step_fail():
    engine = _engine()
    engine.result = {"ok": False, "error": "no network"}
    step = compile_step(
        ActionSpec(name="web_search", description="", schema={}),
        ProposedCall(action="web_search", args={}),
        engine=engine,
    )
    assert await step.action() is False


@pytest.mark.asyncio
async def test_a_whole_plan_is_built_and_every_step_can_run():
    engine = _engine()
    plan = await synthesize_plan(
        "find out about otters and write it down",
        think=_thinks(
            '[{"action": "web_search", "args": {"query": "otters"}, "because": "learn"},'
            ' {"action": "read_file", "args": {"path": "notes.md"}, "because": "check"}]'
        ),
        engine=engine,
    )
    assert plan.usable
    assert [step.name for step in plan.steps] == ["web_search(query)", "read_file(path)"]
    assert plan.narrate() == "web_search(query) then read_file(path)"
    for step in plan.steps:
        await step.action()
    assert [name for name, _args, _ctx in engine.ran] == ["web_search", "read_file"]


@pytest.mark.asyncio
async def test_a_bad_call_is_answered_with_the_problem_not_with_the_question_again():
    think = _thinks(
        '[{"action": "web_search", "args": {}}]',
        '[{"action": "web_search", "args": {"query": "otters"}}]',
    )
    plan = await synthesize_plan("find otters", think=think, engine=_engine())
    assert plan.usable
    assert plan.attempts == 2
    second = think.asked[1]
    assert any("needs 'query' and it was not given" in line for line in second)


@pytest.mark.asyncio
async def test_a_plan_that_never_checks_out_says_what_was_wrong():
    plan = await synthesize_plan(
        "do the impossible", think=_thinks('[{"action": "fly", "args": {}}]'), engine=_engine(), attempts=2
    )
    assert not plan.usable
    assert plan.rejected == ["'fly' is not something she can do"]
    assert "could not build a plan" in plan.narrate()


@pytest.mark.asyncio
async def test_no_actions_at_all_is_said_plainly():
    plan = await synthesize_plan("anything", think=_thinks("[]"), engine=_Engine({}))
    assert not plan.usable
    assert plan.rejected == ["she has no actions available to plan with"]


@pytest.mark.asyncio
async def test_a_mind_out_of_reach_does_not_produce_a_plan():
    async def unreachable(objective, evidence):
        raise RuntimeError("no model")

    plan = await synthesize_plan("anything", think=unreachable, engine=_engine())
    assert not plan.usable
    assert "could not be reached" in plan.rejected[0]
