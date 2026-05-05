import sys
from types import SimpleNamespace

import pytest

from core.environment.command import ActionIntent
from core.environments.terminal_grid import NetHackCommandCompiler, NetHackStateCompiler, NetHackTerminalGridAdapter
from core.environments.terminal_grid.nethack_adapter import EnvironmentMode


@pytest.mark.asyncio
async def test_nethack_adapter_starts_observes_executes_and_closes():
    adapter = NetHackTerminalGridAdapter(force_simulated=True)
    await adapter.start(run_id="nethack-preflight", seed=1)
    obs = await adapter.observe()
    assert obs.text
    parsed = NetHackStateCompiler().compile(obs)
    assert "health" in parsed.resources
    command = NetHackCommandCompiler().compile(ActionIntent(name="move", parameters={"direction": "west"}))
    result = await adapter.execute(command)
    assert result.ok
    await adapter.close()
    assert not adapter.is_alive()


def test_nethack_command_compiler_maps_extended_commands_to_multistep_specs():
    command = NetHackCommandCompiler().compile(ActionIntent(name="pray", expected_effect="prayer_attempted"))
    assert len(command.steps) == 2
    assert command.steps[0].value == "#"
    assert command.steps[1].value == "pray\n"


def test_nethack_state_compiler_detects_return_continuation_prompt():
    lines = ["Warning: status sidecar is unavailable", "", "Hit return to continue:"] + [""] * 21
    parsed = NetHackStateCompiler().parse_text("\n".join(lines[:24]))

    assert parsed.modal_state is not None
    assert parsed.modal_state.safe_default == "\r"


def test_nethack_state_compiler_treats_blank_frame_as_unstable_not_modal():
    parsed = NetHackStateCompiler().parse_text("\n".join([" " * 80 for _ in range(24)]))

    assert parsed.modal_state is None
    assert parsed.uncertainty["blank_observation"] == 1.0


def test_strict_real_startup_prompt_resolution_is_iterative_and_bounded():
    class FakeChild:
        def __init__(self):
            self.sent = []

        def send(self, value):
            self.sent.append(value)

    adapter = NetHackTerminalGridAdapter(force_simulated=True)
    child = FakeChild()
    screens = [
        "Warning: status sidecar unavailable\nHit return to continue:",
        "Cannot open optional state file\nHit space to continue:",
        "Shall I pick a default profile for you? [ynq]",
        "ready",
    ]
    adapter.child = child
    adapter.screen.text = screens[0]

    def update_screen():
        adapter.screen.text = screens[min(len(child.sent), len(screens) - 1)]

    adapter._update_screen = update_screen
    adapter._resolve_startup_prompt()

    assert child.sent == ["\r", " ", "y"]
    assert adapter._last_startup_decision["action"] == "no_prompt"


@pytest.mark.asyncio
async def test_strict_real_adapter_uses_runtime_workspace_for_sidecar_config(monkeypatch, tmp_path):
    class FakeTimeout(Exception):
        pass

    class FakeEof(Exception):
        pass

    class FakeChild:
        def __init__(self):
            self.sent = []

        def setwinsize(self, *_args):
            return None

        def read_nonblocking(self, **_kwargs):
            raise FakeTimeout()

        def send(self, value):
            self.sent.append(value)

        def isalive(self):
            return True

        def terminate(self, force=False):
            return None

    spawned = {}

    def fake_spawn(command, *, env, encoding, timeout):
        spawned["command"] = command
        spawned["env"] = env
        spawned["encoding"] = encoding
        spawned["timeout"] = timeout
        return FakeChild()

    class FakeScreen:
        def __init__(self, width, height):
            self.display = [" " * width for _ in range(height)]

    class FakeStream:
        def __init__(self, screen):
            self.screen = screen

        def feed(self, _text):
            return None

    monkeypatch.setenv("AURA_ENV_RUNTIME_DIR", str(tmp_path / "env-runtime"))
    monkeypatch.setitem(sys.modules, "pexpect", SimpleNamespace(spawn=fake_spawn, TIMEOUT=FakeTimeout, EOF=FakeEof))
    monkeypatch.setitem(sys.modules, "pyte", SimpleNamespace(Screen=FakeScreen, Stream=FakeStream))
    fake_binary = tmp_path / "nethack"
    fake_binary.write_text("#!/bin/sh\n", encoding="utf-8")

    adapter = NetHackTerminalGridAdapter(nethack_path=str(fake_binary), mode=EnvironmentMode.STRICT_REAL)
    await adapter.start(run_id="strict-sidecar", seed=1)

    rc_path = tmp_path / "env-runtime" / "terminal_grid_nethack" / "config" / "nethackrc"
    assert rc_path.exists()
    assert spawned["env"]["NETHACKOPTIONS"] == str(rc_path)
    assert ".nethackrc_aura" not in spawned["env"]["NETHACKOPTIONS"]
    await adapter.close()
