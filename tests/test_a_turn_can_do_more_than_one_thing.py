"""A foreground turn can hold a working set of tools, and read without writing.

Two structural limits made every multi-step task impossible, whatever the
model was capable of.

FIRST: a turn was offered exactly one capability. `max_tools=1`, and the tool
map filtered to a single skill name. Reading a file and then running what it
says needs two; checking the result needs three. So "debug this repository",
"analyse this paper", "fix the failing test" could not be attempted — not
badly attempted, not attempted.

SECOND: effect scope is declared per SKILL, and a skill's scope is the worst
thing it can do. `file_operation` is `state_mutation` because it can delete,
so READING a file required permission to destroy one. Every task on a computer
begins by reading something, which put the cheapest safe step behind the most
dangerous grant in the system.

Offering a skill for its safe actions is only defensible if the dangerous ones
are actually refused, so the refusal is tested here beside the offering.
"""

from __future__ import annotations

import pytest

from core.skills.action_scope import (
    action_effect_scope,
    action_within_scope,
    declared_action_scopes,
    resolve_skill_target,
    skill_has_action_within,
)


class _Mixed:
    ACTION_EFFECT_SCOPES = {"read": "read_only", "delete": "state_mutation"}


def test_a_call_is_scoped_by_its_action_not_by_its_skill():
    assert action_effect_scope(_Mixed, "read", "state_mutation") == "read_only"
    assert action_effect_scope(_Mixed, "delete", "state_mutation") == "state_mutation"


def test_an_undescribed_action_keeps_the_skills_own_scope():
    """Silence is not safety."""
    assert action_effect_scope(_Mixed, "chmod", "state_mutation") == "state_mutation"
    assert not action_within_scope(_Mixed, "chmod", "state_mutation", "read_only")


def test_the_safe_half_is_admissible_and_the_dangerous_half_is_not():
    assert action_within_scope(_Mixed, "read", "state_mutation", "read_only")
    assert not action_within_scope(_Mixed, "delete", "state_mutation", "read_only")
    assert skill_has_action_within(_Mixed, "state_mutation", "read_only")


def test_the_real_file_skill_declares_every_action_it_performs():
    """A declaration that misses an action silently grants the skill's scope."""
    from core.skills.file_operation import FILE_ACTIONS, FileOperationSkill

    declared = declared_action_scopes(FileOperationSkill)
    assert set(declared) == set(FILE_ACTIONS)
    assert declared["read"] == "read_only"
    assert declared["delete"] == "state_mutation"


def test_the_action_parameter_is_enumerated():
    """An unenumerated required string makes the caller guess.

    The nine actions were named only inside a description, so the generated
    tool schema carried `type: string, enum: None`.
    """
    from core.skills.file_operation import FILE_ACTIONS, FileOpInput

    schema = FileOpInput.model_json_schema()
    action = schema["properties"]["action"]
    enumerated = action.get("enum") or [
        entry for ref in action.get("allOf", []) for entry in ref.get("enum", [])
    ]
    assert set(enumerated) == set(FILE_ACTIONS)


def test_the_declaration_is_found_before_the_skill_is_instantiated():
    """Registry metadata carries skill_class and instance as None until first use.

    Reading the declaration off either finds nothing, and every action then
    falls back to the skill's worst-case scope — which is the failure this
    module exists to remove.
    """

    class _LazyMeta:
        skill_class = None
        instance = None
        module_path = "core.skills.file_operation"
        class_name = "FileOperationSkill"

    assert declared_action_scopes(resolve_skill_target(_LazyMeta))["read"] == "read_only"


def test_an_out_of_scope_action_is_refused_with_something_to_act_on():
    """The refusal goes back into the loop, not off the end of the turn."""
    from core.brain.llm.mlx_client import _refuse_action_beyond_authority

    class _Engine:
        skills = {
            "file_operation": type(
                "M",
                (),
                {
                    "effect_scope": "state_mutation",
                    "skill_class": None,
                    "instance": None,
                    "module_path": "core.skills.file_operation",
                    "class_name": "FileOperationSkill",
                },
            )
        }

    context = {"authorised_effect_scope": "sandboxed_compute"}
    assert (
        _refuse_action_beyond_authority(
            _Engine, "file_operation", {"action": "read", "path": "x"}, context
        )
        is None
    )
    refused = _refuse_action_beyond_authority(
        _Engine, "file_operation", {"action": "delete", "path": "x"}, context
    )
    assert refused is not None
    assert refused["ok"] is False
    assert "state_mutation" in refused["error"]
    assert "delete" in refused["error"]


def test_a_turn_with_no_declared_authority_is_left_alone():
    """This gate governs the foreground loop, not every dispatch in the runtime."""
    from core.brain.llm.mlx_client import _refuse_action_beyond_authority

    assert _refuse_action_beyond_authority(None, "file_operation", {"action": "delete"}, {}) is None


def test_the_tool_map_carries_a_working_set():
    from core.brain.llm.runtime_wiring import build_agentic_tool_map

    # A sequence and a single name are both accepted; the filter is membership.
    import inspect

    source = inspect.getsource(build_agentic_tool_map)
    assert "wanted" in source
    assert "name != required_skill" not in source


@pytest.mark.parametrize(
    "conversation",
    [
        "how are you feeling today",
        "tell me a story about the sea",
        "what do you think about consciousness",
        "my code doesn't run anymore",
    ],
)
def test_ordinary_conversation_is_offered_no_tools(conversation: str):
    from core.intent.declared_capability import looks_like_a_request

    assert not looks_like_a_request(conversation)


@pytest.mark.parametrize(
    "task",
    [
        "read the ledger repo, run its tests, and fix whatever is failing",
        "read README.md and tell me what it says",
        "run some python and tell me the number",
        "open the config file and check the timeout",
    ],
)
def test_a_task_shaped_turn_is_recognised_whatever_nouns_it_uses(task: str):
    """Nouns are an open class; README.md is in no skill's vocabulary."""
    from core.intent.declared_capability import looks_like_a_request

    assert looks_like_a_request(task)


def test_the_named_set_survives_the_registrys_own_ranking(monkeypatch):
    """The selector takes ONE name; handed a set it matched nothing.

    Live 2026-08-19: the working set was derived correctly —
    "wanted=internal_sandbox,code_repl,web_search,run_code,file_operation" —
    and then thrown away one call later, so the turn was offered no tools at
    all. A capability that was asked for cannot be dropped for ranking below
    something else.
    """
    from core.brain.llm import runtime_wiring

    wanted = ["code_repl", "file_operation"]

    class _Engine:
        def get_tool_definitions(self):
            return [
                {"function": {"name": name}}
                for name in ("code_repl", "file_operation", "web_search")
            ]

        def select_tool_definitions(
            self, *, objective, required_skill, max_tools, requested=None
        ):
            # Ranks by its own idea of relevance and never returns the reader,
            # ignoring `requested` on purpose: the point of the test is that
            # the named set survives a registry that does exactly this. The
            # keyword still has to be ACCEPTED, or the double raises a
            # TypeError, the real call is recorded as a degradation, and the
            # test passes for the wrong reason on a signature that drifted.
            return [{"function": {"name": "web_search"}}]

    from core import container as container_module

    monkeypatch.setattr(
        container_module.ServiceContainer, "get", staticmethod(lambda *a, **k: _Engine())
    )
    offered = runtime_wiring.build_agentic_tool_map(wanted, objective="x", max_tools=2)
    assert offered is not None
    assert set(wanted) <= set(offered)
