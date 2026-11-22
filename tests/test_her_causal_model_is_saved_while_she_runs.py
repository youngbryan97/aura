"""Her causal world model is saved while she runs, and its save does not hold her event loop (F16).

LIVE 21 November 2026, from her own fault record: every save of the graph after a surprise in her mind tick was refused,
"file_write_gateway.write_text:causal_world_model.save called outside governed context", so what she learned from a
surprise never reached her disk, and her loop heard the refusal as a fault some sixty times. The save is her own
maintenance write and now opens its own governed scope. Two of its callers, her mind tick and the environment kernel's
step, called it from a coroutine, writing and syncing the file on her event loop; they call it from a thread now.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from core.brain.causal_world_model import CausalWorldModel

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[1]


def test_a_save_runs_in_a_governed_scope(tmp_path, monkeypatch):
    from core.governance_context import get_active_governance
    from core.runtime import file_write_gateway

    governed: list[bool] = []
    gateway = file_write_gateway.get_file_write_gateway()
    real = gateway.write_text

    def write_text(path, text, *args, **kwargs):
        governed.append(get_active_governance() is not None)
        return real(path, text, *args, **kwargs)

    monkeypatch.setattr(gateway, "write_text", write_text)
    model = CausalWorldModel(data_path=tmp_path / "causal.json")
    assert model.add_observation("a prediction", "what happened", 0.6, reported_by="mind_tick.prediction_surprise")
    assert governed and all(governed), governed
    assert model.last_save_ok and (tmp_path / "causal.json").is_file()


def _awaited_by_name(tree: ast.AST, name: str) -> list[int]:
    """Lines inside a coroutine that call ``.name(...)`` directly, outside any function nested in it."""
    found: list[int] = []

    def visit(node: ast.AST, in_coroutine: bool) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.AsyncFunctionDef):
                visit(child, True)
            elif isinstance(child, (ast.FunctionDef, ast.Lambda)):
                visit(child, False)
            else:
                if (in_coroutine and isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute)
                        and child.func.attr == name):
                    found.append(child.lineno)
                visit(child, in_coroutine)

    visit(tree, False)
    return found


@pytest.mark.parametrize("module", ["core/mind_tick.py", "core/environment/environment_kernel.py"])
def test_no_coroutine_saves_the_graph_on_her_event_loop(module):
    tree = ast.parse((ROOT / module).read_text(encoding="utf-8"))
    assert _awaited_by_name(tree, "add_observation") == []
