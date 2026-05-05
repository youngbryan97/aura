"""General raw-observation to ParsedState compiler."""
from __future__ import annotations

import hashlib
from typing import Any

from .modal import ModalState
from .observation import Observation
from .ontology import EntityState, HazardState, ObjectState, ResourceState, SemanticEvent
from .parsed_state import ParsedState


class StateCompiler:
    """Compiles evidence into generic environment state.

    Environment-specific parsers should do richer extraction, but this
    compiler supplies the safe fallback and the adapter from the older
    ``core.perception.EnvironmentState`` surface.
    """

    def compile(self, observation: Observation) -> ParsedState:
        text = observation.text or (observation.raw if isinstance(observation.raw, str) else "")
        context_id = observation.context_id or observation.structured.get("context_id") or "unknown"
        parsed = ParsedState(
            environment_id=observation.environment_id,
            context_id=str(context_id),
            raw_observation_ref=observation.stable_hash(),
            sequence_id=observation.sequence_id,
        )
        if text:
            event_id = "evt_" + hashlib.sha256(text[:240].encode("utf-8")).hexdigest()[:12]
            parsed.semantic_events.append(
                SemanticEvent(
                    event_id=event_id,
                    kind="message",
                    label=text.splitlines()[0][:160],
                    context_id=str(context_id),
                    evidence_ref=parsed.raw_observation_ref,
                    confidence=0.75,
                    last_seen_seq=observation.sequence_id,
                )
            )
        if not text and not observation.structured:
            parsed.uncertainty["empty_observation"] = 1.0
        return parsed

    def from_legacy_state(self, legacy: Any, *, environment_id: str | None = None) -> ParsedState:
        env = environment_id or getattr(legacy, "domain", "generic")
        context_id = getattr(legacy, "context_id", "default")
        raw_ref = getattr(legacy, "observation_id", "") or "legacy"
        parsed = ParsedState(
            environment_id=env,
            context_id=context_id,
            self_state=dict(getattr(legacy, "self_state", {}) or {}),
            raw_observation_ref=raw_ref,
        )
        seq = int(parsed.self_state.get("turn", 0) or 0)
        parsed.sequence_id = seq
        spatial_info = dict(getattr(legacy, "spatial_info", {}) or {})
        player_pos = spatial_info.get("player_pos")
        if isinstance(player_pos, (list, tuple)) and len(player_pos) == 2:
            # Legacy terminal parsers report row/column. The environment OS
            # canonical spatial convention is x/y.
            row, col = int(player_pos[0]), int(player_pos[1])
            parsed.self_state.setdefault("local_coordinates", (col, row))
            parsed.self_state.setdefault("grid_coordinates", (row, col))
        if parsed.self_state:
            parsed.entities.append(
                EntityState(
                    entity_id=f"{env}:self",
                    kind="self",
                    label="self",
                    context_id=str(context_id),
                    properties=dict(parsed.self_state),
                    evidence_ref=raw_ref,
                    last_seen_seq=seq,
                )
            )
            parsed.observed_ids.add(f"{env}:self")

        for name, value, max_name, critical in (
            ("health", parsed.self_state.get("hp"), "max_hp", 0.35),
            ("power", parsed.self_state.get("pw"), "max_pw", 0.15),
        ):
            if value is None:
                continue
            max_value = parsed.self_state.get(max_name)
            parsed.resources[name] = ResourceState(
                name=name,
                value=float(value),
                max_value=float(max_value) if max_value is not None else None,
                critical_below=critical,
                evidence_ref=raw_ref,
                last_seen_seq=seq,
                kind=name,
                context_id=str(context_id),
            )

        hunger = parsed.self_state.get("hunger")
        if hunger is not None:
            nutrition_score = {
                "Satiated": 1.0,
                "Normal": 0.8,
                "Hungry": 0.45,
                "Weak": 0.2,
                "Fainting": 0.08,
                "Fainted": 0.03,
                "Starved": 0.0,
            }.get(str(hunger), 0.5)
            parsed.resources["nutrition"] = ResourceState(
                name="nutrition",
                value=nutrition_score,
                max_value=1.0,
                critical_below=0.25,
                evidence_ref=raw_ref,
                last_seen_seq=seq,
                kind="nutrition",
                context_id=str(context_id),
            )

        # Encumbrance → mobility resource
        encumbrance = parsed.self_state.get("encumbrance", "Normal")
        mobility_score = {
            "Normal": 1.0,
            "Burdened": 0.7,
            "Stressed": 0.4,
            "Strained": 0.2,
            "Overtaxed": 0.05,
        }.get(str(encumbrance), 1.0)
        parsed.resources["mobility"] = ResourceState(
            name="mobility",
            value=mobility_score,
            max_value=1.0,
            critical_below=0.3,
            evidence_ref=raw_ref,
            last_seen_seq=seq,
            kind="mobility",
            context_id=str(context_id),
        )

        # Sensory reliability → uncertainty channel
        sensory = parsed.self_state.get("sensory_reliability")
        if sensory is not None and float(sensory) < 1.0:
            parsed.uncertainty["sensory"] = 1.0 - float(sensory)

        # Inventory items → ObjectState entries
        for item in parsed.self_state.get("inventory_items", []) or []:
            oid = f"{env}:inventory:{item.get('letter', '?')}"
            parsed.objects.append(
                ObjectState(
                    object_id=oid,
                    kind="item",
                    label=str(item.get("description", "unknown")),
                    context_id=str(context_id),
                    affordances=["use", "drop"],
                    risk_tags=["cursed"] if item.get("buc") == "cursed" else [],
                    properties=dict(item),
                    evidence_ref=raw_ref,
                    last_seen_seq=seq,
                )
            )
            parsed.observed_ids.add(oid)

        for idx, ent in enumerate(getattr(legacy, "entities", []) or []):
            ent_type = str(ent.get("type", "unknown"))
            pos = ent.get("pos")
            position = None
            if isinstance(pos, (list, tuple)) and len(pos) == 2:
                row, col = int(pos[0]), int(pos[1])
                position = (col, row)
            if ent_type in {"monster", "large_monster", "hostile"}:
                eid = f"{env}:entity:{ent.get('glyph', ent_type)}:{position or idx}"
                parsed.entities.append(
                    EntityState(
                        entity_id=eid,
                        kind="hostile" if ent.get("hostile", True) else "unknown",
                        label=str(ent.get("label") or ent.get("glyph") or ent_type),
                        context_id=str(context_id),
                        position=position,  # type: ignore[arg-type]
                        threat_score=float(ent.get("threat_score", 0.6)),
                        properties=dict(ent),
                        evidence_ref=raw_ref,
                        last_seen_seq=seq,
                    )
                )
                parsed.observed_ids.add(eid)
            else:
                oid = f"{env}:object:{ent.get('glyph', ent_type)}:{position or idx}"
                kind = "transition" if ent_type == "stairs" else "item"
                affordances = ["navigate"] if ent_type == "stairs" else ["inspect"]
                parsed.objects.append(
                    ObjectState(
                        object_id=oid,
                        kind=kind,
                        label=str(ent.get("label") or ent.get("glyph") or ent_type),
                        context_id=str(context_id),
                        position=position,  # type: ignore[arg-type]
                        affordances=affordances,
                        risk_tags=["unknown"] if ent.get("unknown") else [],
                        properties=dict(ent),
                        evidence_ref=raw_ref,
                        last_seen_seq=seq,
                    )
                )
                parsed.observed_ids.add(oid)
                if ent_type in {"trap", "hazard"}:
                    parsed.hazards.append(
                        HazardState(
                            hazard_id=f"{env}:hazard:{position or idx}",
                            kind="damage",
                            label="visible hazard",
                            context_id=str(context_id),
                            severity=0.7,
                            properties={"position": position} if position is not None else {},
                            evidence_ref=raw_ref,
                            last_seen_seq=seq,
                        )
                    )

        prompts = list(getattr(legacy, "active_prompts", []) or [])
        if prompts:
            text = " ".join(str(p) for p in prompts)
            parsed.modal_state = ModalState.from_prompt_text(text)
            parsed.modal_state.source_evidence = raw_ref
            parsed.uncertainty["modal_state"] = float(getattr(legacy, "uncertainty", {}).get("modal_state", 0.2))

        for msg in getattr(legacy, "messages", []) or []:
            parsed.semantic_events.append(
                SemanticEvent(
                    event_id="evt_" + hashlib.sha256(str(msg).encode("utf-8")).hexdigest()[:12],
                    kind="message",
                    label=str(msg)[:160],
                    context_id=str(context_id),
                    evidence_ref=raw_ref,
                    last_seen_seq=seq,
                )
            )
        parsed.uncertainty.update(dict(getattr(legacy, "uncertainty", {}) or {}))
        return parsed


__all__ = ["StateCompiler"]
