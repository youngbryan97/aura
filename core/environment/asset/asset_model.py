"""Asset and tool management for tracking payloads, context, or items."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Asset:
    """Represents a discrete tool, context payload, or physical item the agent possesses."""
    asset_id: str
    category: str
    properties: dict[str, Any] = field(default_factory=dict)
    active: bool = False  # e.g. "wielded", "worn", "in-context"


class AssetModel:
    """Tracks the agent's available tools, assets, and active states."""

    def __init__(self):
        self.assets: dict[str, Asset] = {}

    def update_from_state(self, items: list[dict[str, Any]]) -> None:
        """Sync the internal asset model with a raw list of items from the environment state."""
        # Simple reconciliation: completely replace for now,
        # but in a real system we would diff and track histories.
        new_assets = {}
        for idx, item in enumerate(items):
            asset_id = item.get("id") or item.get("letter") or str(idx)
            asset = Asset(
                asset_id=asset_id,
                category=item.get("category", "general"),
                properties=item,
                active=item.get("equipped", False) or item.get("active", False)
            )
            new_assets[asset_id] = asset
        self.assets = new_assets

    def get_active_assets(self, category: str | None = None) -> list[Asset]:
        """Returns currently active/equipped assets."""
        return [
            a for a in self.assets.values() 
            if a.active and (category is None or a.category == category)
        ]

    def get_assets_by_category(self, category: str) -> list[Asset]:
        """Returns all assets matching a category."""
        return [a for a in self.assets.values() if a.category == category]

__all__ = ["Asset", "AssetModel"]
