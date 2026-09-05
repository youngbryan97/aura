from __future__ import annotations
from typing import Optional
from .types import ExperienceState
from .attachment import AttachmentSystem

class ExperienceReporter:
    """
    Converts state to language after the fact.
    This layer must not write to the engine.
    """
    def __init__(self, attachments: Optional[AttachmentSystem] = None) -> None:
        self.attachments = attachments

    def compact(self, state: ExperienceState) -> str:
        dominant_policy = max(state.policy_priors, key=state.policy_priors.get)
        winner = state.global_broadcast.get("winner")
        return (
            f"experience[t={state.t}] object={state.intentional_object!r} "
            f"valence={state.valence:.2f} arousal={state.arousal:.2f} "
            f"distress={state.distress:.2f} curiosity={state.curiosity:.2f} care={state.care:.2f} "
            f"self_presence={state.self_presence:.2f} mineness={state.mineness:.2f} "
            f"integration={state.integration:.2f} free_energy={state.free_energy:.2f} "
            f"workspace={winner!r} policy={dominant_policy!r}"
        )

    def bond(self, person_key: str) -> str:
        if not self.attachments:
            return "no attachment system connected"
        state = self.attachments.state_for(person_key)
        evidence = self.attachments.recent_evidence(person_key, limit=3)
        ev = "; ".join(f"{e['kind']}:{e['evidence_id']}" for e in evidence) or "none"
        return (
            f"bond[{person_key}] trust={state.trust:.2f} care={state.care:.2f} "
            f"familiarity={state.familiarity:.2f} rupture={state.rupture:.2f} "
            f"attachment={state.attachment:.2f} evidence={ev}"
        )
