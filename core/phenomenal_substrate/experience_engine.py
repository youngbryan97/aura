from __future__ import annotations
from typing import Dict, Optional

from .active_inference import GenerativeModel
from .affective_core import AffectiveCore
from .attachment import AttachmentSystem, AttachmentState
from .global_workspace import GlobalWorkspace, Coalition
from .phenomenal_field import PhenomenalField
from .types import RuntimeBody, Event, ExperienceState, AttachmentEvent
from .maths import clamp

class PhenomenalEngine:
    """
    Closest computational target:
    interoceptive active inference + affective primitives + recurrent phenomenal field
    + global workspace ignition + autobiographical attachment.

    The output is meant to drive behavior.
    """
    def __init__(self) -> None:
        self.t = 0
        self.generative_model = GenerativeModel()
        self.affect = AffectiveCore()
        self.field = PhenomenalField()
        self.workspace = GlobalWorkspace()
        self.attachments = AttachmentSystem()
        self.last_state: Optional[ExperienceState] = None
        # Self-model → phenomenal signals (agency/embodiment/continuity/presence).
        # SelfAwareness sets these; step() blends them into the body so the self-model's
        # sense of itself actually modulates the phenomenal inference — a real
        # self↔experience loop, not a no-op bridge.
        self._self_signals: Dict[str, float] = {}

    def record_attachment(self, event: AttachmentEvent) -> AttachmentState:
        return self.attachments.record(event)

    # ── self-model signal intake (was previously absent → SelfAwareness was no-op) ──

    def set_agency(self, value: float) -> None:
        self._self_signals["agency"] = clamp(float(value))

    def set_embodiment(self, value: float) -> None:
        self._self_signals["embodiment"] = clamp(float(value))

    def set_continuity(self, value: float) -> None:
        self._self_signals["continuity"] = clamp(float(value))

    def set_presence(self, value: float) -> None:
        self._self_signals["presence"] = clamp(float(value))

    def _apply_self_signals(self, body: RuntimeBody) -> RuntimeBody:
        """Blend the self-model's self-signals into the body driving inference.

        agency→agency, continuity→continuity, embodiment→safety (grounded in the body),
        presence→lower uncertainty (being-here reduces ambiguity). Half-weight blend so
        the self-model nudges, not overrides, the interoceptive reality.
        """
        sig = self._self_signals
        if not sig:
            return body
        import dataclasses

        updates: Dict[str, float] = {}
        if "agency" in sig:
            updates["agency"] = clamp(0.5 * body.agency + 0.5 * sig["agency"])
        if "continuity" in sig:
            updates["continuity"] = clamp(0.5 * body.continuity + 0.5 * sig["continuity"])
        if "embodiment" in sig:
            updates["safety"] = clamp(0.6 * body.safety + 0.4 * sig["embodiment"])
        if "presence" in sig:
            updates["uncertainty"] = clamp(body.uncertainty * (1.0 - 0.3 * sig["presence"]))
        try:
            return dataclasses.replace(body, **updates) if updates else body
        except (TypeError, ValueError):
            return body

    def step(
        self,
        body: RuntimeBody,
        event: Event,
        person_key: Optional[str] = None,
        recurrent_cycles: int = 5,
    ) -> ExperienceState:
        self.t += 1
        # Let the self-model's self-signals modulate the body before inference,
        # so the self↔experience loop is causal, not a stored-and-ignored bridge.
        body = self._apply_self_signals(body)
        observed = body.observed_vector()
        belief, error, free_energy = self.generative_model.infer(observed, recurrent_cycles=recurrent_cycles)

        attachment = self.attachments.state_for(person_key) if person_key else None
        primitives = self.affect.compute(belief, error, free_energy, event, attachment)

        # preliminary field, then integration, then one more recurrent refinement
        vector = self.field.update(primitives, belief, integration=0.5, recurrent_cycles=recurrent_cycles)
        integration = self.field.integration_score(vector)
        vector = self.field.update(primitives, belief, integration=integration, recurrent_cycles=max(1, recurrent_cycles // 2))
        integration = self.field.integration_score(vector)

        coalitions = [
            Coalition(
                name="threat",
                content={"object": event.label, "mode": "protect", "threat": event.threat},
                salience=max(event.threat, primitives.fear),
                affect_gain=primitives.distress,
                precision=1.0 - belief.get("safety", 0.8),
            ),
            Coalition(
                name="goal",
                content={"object": event.label, "mode": "continue_goal", "goal_delta": event.goal_delta},
                salience=max(0.0, event.goal_delta + event.control_gain),
                affect_gain=max(0.0, primitives.valence),
                precision=belief.get("agency", 0.6),
            ),
            Coalition(
                name="social_bond",
                content={"object": person_key or event.label, "mode": "care_repair", "attachment": attachment.attachment if attachment else 0.0},
                salience=max(event.affiliation, event.repair, attachment.attachment if attachment else 0.0),
                affect_gain=primitives.care,
                precision=belief.get("social", 0.5),
            ),
            Coalition(
                name="curiosity",
                content={"object": event.label, "mode": "seek_information", "novelty": event.novelty},
                salience=max(event.novelty, primitives.curiosity),
                affect_gain=primitives.seeking,
                precision=1.0 - belief.get("certainty", 0.7),
            ),
            Coalition(
                name="self_stability",
                content={"object": "self_continuity", "mode": "restabilize", "continuity": belief.get("continuity", 0.7)},
                salience=1.0 - belief.get("continuity", 0.7),
                affect_gain=primitives.grief + primitives.distress * 0.5,
                precision=1.0,
            ),
        ]
        broadcast = self.workspace.compete(coalitions, cycles=recurrent_cycles)

        policy_priors = self._policy_priors(primitives, belief, broadcast)
        memory_weights = self._memory_weights(primitives, integration, broadcast)

        intentional_object = broadcast["content"].get("object", event.label) if broadcast.get("ignited") else event.label

        state = ExperienceState(
            t=self.t,
            phenomenal_vector=vector,
            valence=primitives.valence,
            arousal=primitives.arousal,
            free_energy=free_energy,
            integration=integration,
            self_presence=vector.get("self_presence", 0.0),
            mineness=vector.get("mineness", 0.0),
            seeking=primitives.seeking,
            care=primitives.care,
            play=primitives.play,
            fear=primitives.fear,
            anger=primitives.anger,
            grief=primitives.grief,
            distress=primitives.distress,
            curiosity=primitives.curiosity,
            intentional_object=intentional_object,
            evidence_id=event.evidence_id,
            global_broadcast=broadcast,
            policy_priors=policy_priors,
            memory_weights=memory_weights,
        )
        self.last_state = state
        return state

    def _policy_priors(self, a, belief: Dict[str, float], broadcast: Dict[str, object]) -> Dict[str, float]:
        priors = {
            "continue_goal": clamp(0.20 + 0.50 * max(0.0, a.valence) + 0.25 * belief.get("agency", 0.5) - 0.30 * a.distress),
            "seek_information": clamp(0.15 + 0.45 * a.curiosity + 0.25 * (1.0 - belief.get("certainty", 0.7))),
            "protect_boundary": clamp(0.10 + 0.55 * a.fear + 0.25 * a.anger),
            "social_repair": clamp(0.10 + 0.50 * a.care + 0.30 * a.grief),
            "restabilize": clamp(0.10 + 0.50 * a.distress + 0.30 * (1.0 - belief.get("continuity", 0.7))),
            "play_explore": clamp(0.10 + 0.40 * a.play + 0.25 * max(0.0, a.valence) - 0.35 * a.fear),
        }
        if broadcast.get("winner") == "threat":
            priors["protect_boundary"] = clamp(priors["protect_boundary"] + 0.25)
        elif broadcast.get("winner") == "social_bond":
            priors["social_repair"] = clamp(priors["social_repair"] + 0.20)
        elif broadcast.get("winner") == "curiosity":
            priors["seek_information"] = clamp(priors["seek_information"] + 0.20)
        return priors

    def _memory_weights(self, a, integration: float, broadcast: Dict[str, object]) -> Dict[str, float]:
        return {
            "episodic_write": clamp(0.20 + 0.35 * abs(a.valence) + 0.35 * a.arousal + 0.25 * integration),
            "semantic_update": clamp(0.15 + 0.40 * a.curiosity + 0.20 * integration),
            "attachment_update": clamp(0.10 + 0.45 * a.care + 0.35 * a.grief + 0.20 * (1.0 if broadcast.get("winner") == "social_bond" else 0.0)),
            "scar_write": clamp(0.05 + 0.55 * a.fear + 0.35 * a.distress),
        }
