from pathlib import Path

from core.environment.command import ActionIntent
from core.environments.terminal_grid import NetHackCommandCompiler, NetHackStateCompiler, NetHackTerminalGridAdapter


def test_terminal_grid_parser_returns_generic_parsed_state():
    text = Path("tests/environments/terminal_grid/fixtures/nethack_start.txt").read_text(encoding="utf-8")
    parsed = NetHackStateCompiler().parse_text(text)
    assert parsed.environment_id == "terminal_grid:nethack"
    assert parsed.resources["health"].normalized == 1.0
    assert any(entity.kind == "self" for entity in parsed.entities)
    assert any(entity.kind == "hostile" for entity in parsed.entities)
    assert any(obj.kind == "transition" for obj in parsed.objects)


def test_terminal_grid_commands_compile_from_generic_action_intents():
    compiler = NetHackCommandCompiler()
    command = compiler.compile(ActionIntent(name="move", parameters={"direction": "west"}))
    assert command.steps[0].value == "h"
    assert command.environment_id == "terminal_grid:nethack"


def test_terminal_grid_adapter_implements_environment_adapter():
    adapter = NetHackTerminalGridAdapter(force_simulated=True)
    assert adapter.environment_id == "terminal_grid:nethack"
    assert adapter.capabilities.can_act
