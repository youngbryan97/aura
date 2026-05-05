"""NetHack terminal-grid parser that emits generic ParsedState."""
from __future__ import annotations

from typing import Any

from core.environment.observation import Observation
from core.environment.parsed_state import ParsedState
from core.environment.state_compiler import StateCompiler
from core.perception.nethack_parser import NetHackParser
from .state_compiler import TerminalGridStateCompiler


class NetHackStateCompiler(StateCompiler):
    def __init__(self) -> None:
        super().__init__()
        self._legacy_parser = NetHackParser()
        self._generic_grid = TerminalGridStateCompiler()

    def compile(self, observation: Observation) -> ParsedState:
        text = observation.text or (observation.raw if isinstance(observation.raw, str) else "")
        text = text.rstrip("\n")
        try:
            legacy = self._legacy_parser.parse(text)
            parsed = self.from_legacy_state(legacy, environment_id="terminal_grid:nethack")
            parsed.raw_observation_ref = observation.stable_hash()
            parsed.sequence_id = observation.sequence_id
            return parsed
        except Exception:
            parsed = self._generic_grid.compile(observation)
            parsed.environment_id = "terminal_grid:nethack"
            parsed.uncertainty["parser_error"] = 1.0
            return parsed

    def parse_text(self, text: str) -> ParsedState:
        observation = Observation(
            environment_id="terminal_grid:nethack",
            run_id="fixture",
            sequence_id=1,
            raw=text,
            text=text,
        )
        return self.compile(observation)


__all__ = ["NetHackStateCompiler"]
