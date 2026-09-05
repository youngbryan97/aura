"""core/continuity/state_snapshot.py
Compiles LifeState data configurations into serializable snapshots.
"""
from typing import Any

from core.organism.life_state import LifeState


class StateSnapshotSerializer:
    """Serializes LifeState models to JSON dictionary structures."""

    def serialize(self, state: LifeState) -> dict[str, Any]:
        return state.to_dict()

    def deserialize(self, data: dict[str, Any], state: LifeState) -> None:
        """Restores state attributes from snapshot dictionary."""
        state.tick_count = data.get("tick_count", 0)
        
        welfare_data = data.get("welfare", {})
        for k, v in welfare_data.items():
            setattr(state.welfare, k, v)
            
        body_data = data.get("body", {})
        for k, v in body_data.items():
            setattr(state.body, k, v)

        cognition_data = data.get("cognition", {})
        for k, v in cognition_data.items():
            setattr(state.cognition, k, v)

        state.world_model = data.get("world_model", {})
        state.commitments = data.get("commitments", [])
        state.identity = data.get("identity", {})
