"""NetHack terminal-grid adapter.

The adapter uses the real NetHack binary when explicitly available. If the
binary or optional terminal libraries are unavailable, it runs a deterministic
headless terminal-grid canary so the architecture loop can still be tested
without cheating or environment-specific bypasses.
"""
from __future__ import annotations

import os
import shutil
import time
from enum import Enum
from pathlib import Path

from core.environment.adapter import EnvironmentCapabilities, EnvironmentUnavailableError, ExecutionResult, ensure_command_spec
from core.environment.command import CommandSpec
from core.environment.observation import Observation

from .base import TerminalGridAdapter


class EnvironmentMode(Enum):
    AUTO = "auto"
    SIMULATED = "simulated"
    STRICT_REAL = "strict_real"


class NetHackTerminalGridAdapter(TerminalGridAdapter):
    environment_id = "terminal_grid:nethack"
    capabilities = EnvironmentCapabilities(
        can_observe=True,
        can_act=True,
        supports_dry_run=True,
        supports_replay=True,
        supports_snapshots=False,
        supports_modal_states=True,
        supports_structured_state=True,
        action_latency_ms_target=250,
    )

    def __init__(self, nethack_path: str | None = None, *, force_simulated: bool = False, mode: EnvironmentMode = EnvironmentMode.AUTO) -> None:
        super().__init__()
        self.nethack_path = nethack_path or os.environ.get("AURA_NETHACK_PATH") or shutil.which("nethack") or "/opt/homebrew/bin/nethack"
        
        # Backwards compatibility for force_simulated
        self.mode = EnvironmentMode.SIMULATED if force_simulated else mode
        
        self.child = None
        self._pyte_screen = None
        self._pyte_stream = None
        self._simulated = True

    async def start(self, *, run_id: str, seed: int | None = None) -> None:
        await super().start(run_id=run_id, seed=seed)
        
        if self.mode == EnvironmentMode.SIMULATED:
            self._simulated = True
            return
            
        if not Path(self.nethack_path).exists():
            if self.mode == EnvironmentMode.STRICT_REAL:
                raise EnvironmentUnavailableError(f"NetHack binary not found at {self.nethack_path} but STRICT_REAL mode was requested.")
            self._simulated = True
            return
        try:
            import pexpect  # type: ignore
            import pyte  # type: ignore

            env = os.environ.copy()
            env["TERM"] = "xterm-256color"
            rc_path = Path.home() / ".nethackrc_aura"
            rc_path.write_text(
                "OPTIONS=color,autoquiver,autopickup,hitpointbar,showexp,time,statuslines:2\n"
                "OPTIONS=pettype:none\n"
                "OPTIONS=pickup_types:$\n",
                encoding="utf-8",
            )
            env["NETHACKOPTIONS"] = str(rc_path)
            self._pyte_screen = pyte.Screen(80, 24)
            self._pyte_stream = pyte.Stream(self._pyte_screen)
            self.child = pexpect.spawn(f"{self.nethack_path} -u Aura", env=env, encoding="utf-8", timeout=0.2)
            self.child.setwinsize(24, 80)
            time.sleep(0.5)
            self._update_screen()
            self._simulated = False
        except Exception:
            self.child = None
            self._simulated = True

    def _update_screen(self) -> None:
        if self._simulated or self.child is None:
            return
        try:
            import pexpect  # type: ignore

            out = self.child.read_nonblocking(size=10000, timeout=0.05)
            if out and self._pyte_stream is not None:
                self._pyte_stream.feed(out)
                self.screen.text = "\n".join(self._pyte_screen.display)  # type: ignore[union-attr]
        except Exception:
            pass

    async def observe(self) -> Observation:
        self._update_screen()
        return await super().observe()

    async def execute(self, command: CommandSpec) -> ExecutionResult:
        ensure_command_spec(command)
        if self._simulated or self.child is None:
            return await super().execute(command)
        try:
            for step in command.steps:
                if step.kind == "key":
                    self.child.send(step.value)
                elif step.kind == "text":
                    self.child.send(step.value)
                elif step.kind == "wait":
                    time.sleep(min(0.2, step.timeout_s))
                elif step.kind == "observe":
                    pass
                else:
                    return ExecutionResult(False, command.command_id, None, error=f"unsupported_terminal_step:{step.kind}")
                time.sleep(min(0.2, step.timeout_s))
                self._update_screen()
            observation = await self.observe()
            return ExecutionResult(True, command.command_id, observation, raw_result={"simulated": False})
        except Exception as exc:
            return ExecutionResult(False, command.command_id, None, error=str(exc))

    async def close(self) -> None:
        if self.child is not None:
            try:
                self.child.terminate(force=True)
            except Exception:
                pass
            self.child = None
        await super().close()

    def is_alive(self) -> bool:
        if self._simulated:
            return super().is_alive()
        return bool(self.child is not None and self.child.isalive())


__all__ = ["NetHackTerminalGridAdapter"]
