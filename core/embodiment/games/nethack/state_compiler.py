"""NetHack-specific state compiler.

Translates the output of the NetHackParser into the generic ParsedState.
"""
from __future__ import annotations

import hashlib
from typing import Any

from core.environment.observation import Observation
from core.environment.parsed_state import ParsedState
from core.environment.ontology import (
    EntityState, ObjectState, ResourceState, HazardState, SemanticEvent
)
from core.embodiment.games.nethack.parser import NetHackParser


class NetHackStateCompiler:
    """Compiles NetHack parsed grids and vitals into Aura's ontology."""

    def __init__(self):
        self.parser = NetHackParser()

    def compile(self, observation: Observation) -> ParsedState:
        text = observation.text or (observation.raw if isinstance(observation.raw, str) else "")
        context_id = observation.context_id or observation.environment_id or "nethack"
        
        parsed = ParsedState(
            environment_id=observation.environment_id,
            context_id=str(context_id),
            raw_observation_ref=observation.stable_hash(),
            sequence_id=observation.sequence_id,
        )

        nethack_state = self.parser.parse(text)
        
        # Base self_state
        parsed.self_state = {
            "hp": nethack_state["vitals"]["hp"],
            "max_hp": nethack_state["vitals"]["maxhp"],
            "pw": nethack_state["vitals"]["pw"],
            "exp": nethack_state["vitals"]["exp"],
            "ac": nethack_state["vitals"]["ac"],
            "dlvl": nethack_state["vitals"]["dlvl"],
            "local_coordinates": nethack_state["player_pos"],
            "raw_text": text,
            "status_flags": nethack_state["status_flags"],
        }
        
        parsed.observed_ids.add("nethack:self")

        # Resources
        hp = nethack_state["vitals"]["hp"]
        maxhp = nethack_state["vitals"]["maxhp"]
        parsed.resources["health"] = ResourceState(
            name="health",
            value=float(hp),
            max_value=float(maxhp) if maxhp > 0 else 1.0,
            critical_below=0.35,
            kind="health",
            context_id=str(context_id),
            evidence_ref=parsed.raw_observation_ref,
            last_seen_seq=parsed.sequence_id,
        )

        parsed.resources["power"] = ResourceState(
            name="power",
            value=float(nethack_state["vitals"]["pw"]),
            max_value=100.0, # Guess or ignore
            critical_below=0.15,
            kind="power",
            context_id=str(context_id),
            evidence_ref=parsed.raw_observation_ref,
            last_seen_seq=parsed.sequence_id,
        )
        
        # Nutrition from flags
        nutrition_score = 0.8
        for flag in nethack_state["status_flags"]:
            if flag == "Hunger": nutrition_score = 0.45
            elif flag == "Weak": nutrition_score = 0.2
            elif flag == "Fainting": nutrition_score = 0.08
        
        parsed.resources["nutrition"] = ResourceState(
            name="nutrition",
            value=nutrition_score,
            max_value=1.0,
            critical_below=0.25,
            kind="nutrition",
            context_id=str(context_id),
            evidence_ref=parsed.raw_observation_ref,
            last_seen_seq=parsed.sequence_id,
        )

        # Entities
        for idx, m in enumerate(nethack_state["local_monsters"]):
            glyph = m["glyph"]
            pos = m["pos"]
            dist = m["distance"]
            
            # Simple heuristic
            threat = 0.8 if dist <= 2 else 0.5
            if glyph in ("@", "d", "c"): # usually pets like dog/cat
                kind = "ally"
                threat = 0.0
            else:
                kind = "hostile"

            eid = f"nethack:entity:{glyph}:{pos[0]}_{pos[1]}"
            parsed.entities.append(EntityState(
                entity_id=eid,
                kind=kind,
                label=f"monster_{glyph}",
                context_id=str(context_id),
                position=pos,
                threat_score=threat,
                evidence_ref=parsed.raw_observation_ref,
                last_seen_seq=parsed.sequence_id,
            ))
            parsed.observed_ids.add(eid)

        # Map topology (Objects & Hazards)
        grid = nethack_state["raw_grid"]
        for y, row in enumerate(grid):
            for x, char in enumerate(row):
                if char in ("<", ">"):
                    oid = f"nethack:object:stairs_{char}:{x}_{y}"
                    parsed.objects.append(ObjectState(
                        object_id=oid,
                        kind="transition",
                        label="stairs_up" if char == "<" else "stairs_down",
                        context_id=str(context_id),
                        position=(x, y),
                        affordances=["use_stairs", "navigate"],
                        evidence_ref=parsed.raw_observation_ref,
                        last_seen_seq=parsed.sequence_id,
                    ))
                    parsed.observed_ids.add(oid)
                elif char == "+":
                    oid = f"nethack:object:door:{x}_{y}"
                    parsed.objects.append(ObjectState(
                        object_id=oid,
                        kind="transition",
                        label="closed_door",
                        context_id=str(context_id),
                        position=(x, y),
                        affordances=["open_door", "kick_door", "navigate"],
                        evidence_ref=parsed.raw_observation_ref,
                        last_seen_seq=parsed.sequence_id,
                    ))
                    parsed.observed_ids.add(oid)
                elif char in ("$", "%", "*", "!", "?", "/", "=", "\"", "(", "[", ")"):
                    oid = f"nethack:object:item_{char}:{x}_{y}"
                    parsed.objects.append(ObjectState(
                        object_id=oid,
                        kind="item",
                        label=f"item_{char}",
                        context_id=str(context_id),
                        position=(x, y),
                        affordances=["pickup", "inspect"],
                        evidence_ref=parsed.raw_observation_ref,
                        last_seen_seq=parsed.sequence_id,
                    ))
                    parsed.observed_ids.add(oid)
                elif char == "^":
                    hid = f"nethack:hazard:trap:{x}_{y}"
                    parsed.hazards.append(HazardState(
                        hazard_id=hid,
                        kind="damage",
                        label="trap",
                        context_id=str(context_id),
                        severity=0.8,
                        evidence_ref=parsed.raw_observation_ref,
                        last_seen_seq=parsed.sequence_id,
                    ))
                    parsed.observed_ids.add(hid)

        # Sensory reliability
        sensory = nethack_state.get("sensory_reliability", 1.0)
        if sensory < 1.0:
            parsed.uncertainty["sensory"] = 1.0 - sensory

        # Basic message extraction
        lines = text.splitlines()
        msg_line = lines[0] if lines else ""
        if msg_line and not msg_line.startswith(" ") and not "----" in msg_line:
            event_id = "evt_" + hashlib.sha256(msg_line[:240].encode("utf-8")).hexdigest()[:12]
            parsed.semantic_events.append(SemanticEvent(
                event_id=event_id,
                kind="message",
                label=msg_line[:160],
                context_id=str(context_id),
                evidence_ref=parsed.raw_observation_ref,
                last_seen_seq=parsed.sequence_id,
            ))

        return parsed

__all__ = ["NetHackStateCompiler"]
