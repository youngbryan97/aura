"""Generic terminal-grid compiler for ASCII/ANSI-like environments."""
from __future__ import annotations

import hashlib
import re

from core.environment.observation import Observation
from core.environment.ontology import EntityState, HazardState, ObjectState, SemanticEvent
from core.environment.parsed_state import ParsedState
from core.environment.state_compiler import StateCompiler


class TerminalGridStateCompiler(StateCompiler):
    """Compile a terminal screen into generic grid, entity, object, and message state."""

    WALKABLE = {".", " ", "_", ":", ";", ",", "'", '"'}
    WALLS = {"|", "-", "#", "+", "X"}
    TRANSITIONS = {"<", ">"}
    HAZARDS = {"^", "!", "~"}
    SELF = {"@", "A"}

    def compile(self, observation: Observation) -> ParsedState:
        parsed = super().compile(observation)
        text = observation.text or (observation.raw if isinstance(observation.raw, str) else "")
        lines = text.splitlines()
        context = parsed.context_id or observation.context_id or "terminal"
        parsed.context_id = str(context)
        parsed.self_state.setdefault("grid_size", (max((len(line) for line in lines), default=0), len(lines)))
        for y, line in enumerate(lines):
            for x, glyph in enumerate(line.rstrip("\n")):
                if glyph in self.SELF:
                    parsed.self_state["local_coordinates"] = (x, y)
                    parsed.observed_ids.add(f"{parsed.environment_id}:self")
                elif glyph in self.TRANSITIONS:
                    oid = f"{parsed.environment_id}:transition:{x}:{y}"
                    parsed.objects.append(ObjectState(
                        object_id=oid,
                        kind="transition",
                        label="grid transition",
                        context_id=str(context),
                        position=(x, y),
                        affordances=["navigate", "use_stairs"],
                        properties={"glyph": glyph},
                        evidence_ref=parsed.raw_observation_ref,
                        last_seen_seq=observation.sequence_id,
                    ))
                    parsed.observed_ids.add(oid)
                elif glyph in self.HAZARDS:
                    hid = f"{parsed.environment_id}:hazard:{x}:{y}"
                    parsed.hazards.append(HazardState(
                        hazard_id=hid,
                        kind="unknown",
                        label="visible grid hazard",
                        context_id=str(context),
                        severity=0.6,
                        properties={"glyph": glyph, "position": (x, y)},
                        evidence_ref=parsed.raw_observation_ref,
                        last_seen_seq=observation.sequence_id,
                    ))
                    parsed.observed_ids.add(hid)
        for idx, message in enumerate(self._extract_messages(lines)):
            parsed.semantic_events.append(SemanticEvent(
                event_id="evt_" + hashlib.sha256(f"{observation.sequence_id}:{idx}:{message}".encode()).hexdigest()[:12],
                kind="message",
                label=message[:160],
                context_id=str(context),
                evidence_ref=parsed.raw_observation_ref,
                confidence=0.7,
                last_seen_seq=observation.sequence_id,
            ))
        if "local_coordinates" not in parsed.self_state:
            parsed.uncertainty["self_position"] = 0.8
        return parsed

    @staticmethod
    def _extract_messages(lines: list[str]) -> list[str]:
        messages: list[str] = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            if re.search(r"[A-Za-z]{3,}", stripped) and not re.fullmatch(r"[-|+.# <>@A~^!,:;_]+", stripped):
                messages.append(stripped)
        return messages[:5]


__all__ = ["TerminalGridStateCompiler"]
