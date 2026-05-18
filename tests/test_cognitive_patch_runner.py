import ast
import json
from pathlib import Path

from optimizer.cognitive_patch_runner import run_cognitive_patch


def test_cognitive_patch_runner_installs_valid_skill_code(tmp_path: Path) -> None:
    code = '''
from core.skills.base_skill import BaseSkill


class GeneratedFocusSkill(BaseSkill):
    name = "generated_focus"
    description = "Report that the generated skill is reachable."

    async def execute(self, params, context):
        return {"ok": True, "summary": "ready", "params": params, "context": bool(context)}
'''

    result = run_cognitive_patch(
        {
            "ok": True,
            "type": "skill_install",
            "payload": {"name": "generated_focus", "code": code},
        },
        project_root=tmp_path,
    )

    path = tmp_path / "skills" / "generated" / "generated_focus.py"
    assert result["applied"] is True
    assert result["path"] == "skills/generated/generated_focus.py"
    assert len(result["sha256"]) == 64
    ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def test_cognitive_patch_runner_rejects_unsafe_skill_import(tmp_path: Path) -> None:
    code = "import os\n\nVALUE = os.getcwd()\n"

    result = run_cognitive_patch(
        {
            "ok": True,
            "type": "skill_install",
            "payload": {"name": "unsafe_focus", "code": code},
        },
        project_root=tmp_path,
    )

    assert result["applied"] is False
    assert "blocked import" in result["reason"]
    assert not (tmp_path / "skills" / "generated" / "unsafe_focus.py").exists()


def test_cognitive_patch_runner_installs_step_recipe(tmp_path: Path) -> None:
    result = run_cognitive_patch(
        {
            "ok": True,
            "type": "skill_install",
            "payload": {
                "name": "focus_recipe",
                "description": "Small deterministic workflow.",
                "steps": [
                    "Inspect the active context.",
                    {"instruction": "Return a concise status receipt."},
                ],
            },
        },
        project_root=tmp_path,
    )

    path = tmp_path / "skills" / "generated" / "recipes" / "focus_recipe.json"
    recipe = json.loads(path.read_text(encoding="utf-8"))
    assert result["applied"] is True
    assert recipe["name"] == "focus_recipe"
    assert [step["index"] for step in recipe["steps"]] == [1, 2]


def test_cognitive_patch_runner_applies_config_update(tmp_path: Path) -> None:
    config_path = tmp_path / "config" / "runtime.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text('{"runtime": {"limit": 1}, "keep": true}\n', encoding="utf-8")

    result = run_cognitive_patch(
        {
            "ok": True,
            "type": "config_update",
            "payload": {
                "path": "config/runtime.json",
                "updates": {"runtime": {"limit": 2, "enabled": True}},
            },
        },
        project_root=tmp_path,
    )

    updated = json.loads(config_path.read_text(encoding="utf-8"))
    assert result["applied"] is True
    assert updated == {"keep": True, "runtime": {"enabled": True, "limit": 2}}


def test_cognitive_patch_runner_rejects_sensitive_config_keys(tmp_path: Path) -> None:
    result = run_cognitive_patch(
        {
            "ok": True,
            "type": "config_update",
            "payload": {
                "path": "config/runtime.json",
                "create": True,
                "updates": {"auth": {"api_key": "value"}},
            },
        },
        project_root=tmp_path,
    )

    assert result["applied"] is False
    assert "secret store" in result["reason"]
    assert not (tmp_path / "config" / "runtime.json").exists()


def test_cognitive_patch_runner_rejects_config_path_traversal(tmp_path: Path) -> None:
    result = run_cognitive_patch(
        {
            "ok": True,
            "type": "config_update",
            "payload": {
                "path": "../config/runtime.json",
                "updates": {"runtime": {"enabled": True}},
            },
        },
        project_root=tmp_path,
    )

    assert result["applied"] is False
    assert "traverse" in result["reason"]
