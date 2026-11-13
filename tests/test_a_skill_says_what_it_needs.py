"""A skill that validates an input model declares it, so its schema says what a call must carry.

LIVE 13 November 2026: notify_user built NotifyUserInput inside execute and declared no input model, so it advertised
no required argument. Her tool hand-off, which calls a capability on the request alone when its schema needs nothing,
called it with no arguments, and it failed for want of a message.
"""
from __future__ import annotations

import ast
from pathlib import Path

_SKILLS = Path(__file__).resolve().parents[1] / "core" / "skills"


def test_the_notification_needs_its_message():
    from core.brain.llm.mlx_client import _tool_arguments_schema_error
    from core.skills.notify_user import NotifyUserInput, NotifyUserSkill

    assert NotifyUserSkill.input_model is NotifyUserInput
    schema = NotifyUserSkill().get_schema()
    definition = {"function": {"name": "notify_user", "parameters": schema.get("parameters", schema)}}
    assert _tool_arguments_schema_error(definition, {}), "a call with nothing must not pass"
    assert not _tool_arguments_schema_error(definition, {"message": "stretch break"})


def test_no_skill_validates_an_input_model_it_does_not_declare():
    undeclared = []
    for path in sorted(_SKILLS.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        models = {node.name for node in tree.body if isinstance(node, ast.ClassDef)
                  and any(getattr(base, "id", getattr(base, "attr", "")) == "BaseModel" for base in node.bases)}
        for cls in (node for node in tree.body if isinstance(node, ast.ClassDef)):
            declared = any(
                isinstance(stmt, (ast.Assign, ast.AnnAssign))
                and any(getattr(target, "id", "") == "input_model"
                        for target in (stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]))
                for stmt in cls.body
            )
            used = {node.func.id for node in ast.walk(cls)
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in models}
            if used and not declared:
                undeclared.append(f"{path.name}:{cls.name} uses {sorted(used)}")
    assert not undeclared, undeclared
