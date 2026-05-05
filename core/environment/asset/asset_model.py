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
    confidence: float = 1.0
    last_seen_seq: int = 0
    uncertainty_tags: set[str] = field(default_factory=set)


class AssetModel:
    """Tracks the agent's available tools, assets, and active states."""

    def __init__(self):
        self.assets: dict[str, Asset] = {}
        self.history: list[dict[str, Any]] = []

    def update_from_state(self, items: list[dict[str, Any]], *, sequence_id: int = 0) -> None:
        """Merge observed assets while preserving uncertain unseen state."""
        observed_ids: set[str] = set()
        for idx, item in enumerate(items):
            asset_id = item.get("id") or item.get("letter") or str(idx)
            observed_ids.add(str(asset_id))
            existing = self.assets.get(str(asset_id))
            uncertainty_tags = set(existing.uncertainty_tags if existing else set())
            if item.get("identified") is False or item.get("unknown"):
                uncertainty_tags.add("unknown_identity")
            if item.get("buc") in {"cursed", "unknown"}:
                uncertainty_tags.add(f"buc:{item.get('buc')}")
            asset = Asset(
                asset_id=str(asset_id),
                category=item.get("category", "general"),
                properties={**(existing.properties if existing else {}), **item},
                active=bool(item.get("equipped", False) or item.get("active", False)),
                confidence=min(1.0, max(float(existing.confidence) * 0.95 if existing else 0.0, float(item.get("confidence", 0.85)))),
                last_seen_seq=int(sequence_id or item.get("last_seen_seq", 0) or 0),
                uncertainty_tags=uncertainty_tags,
            )
            self.assets[str(asset_id)] = asset
        for asset_id, asset in list(self.assets.items()):
            if asset_id not in observed_ids:
                asset.confidence = max(0.05, asset.confidence * 0.9)
        self.history.append({"sequence_id": sequence_id, "observed": sorted(observed_ids), "total": len(self.assets)})
        self.history = self.history[-200:]

    def get_active_assets(self, category: str | None = None) -> list[Asset]:
        """Returns currently active/equipped assets."""
        return [
            a for a in self.assets.values() 
            if a.active and (category is None or a.category == category)
        ]

    def get_assets_by_category(self, category: str) -> list[Asset]:
        """Returns all assets matching a category."""
        return [a for a in self.assets.values() if a.category == category]

    def uncertain_assets(self) -> list[Asset]:
        return [asset for asset in self.assets.values() if asset.uncertainty_tags or asset.confidence < 0.6]

    def safe_probe_plan(self, asset_id: str) -> list[str]:
        asset = self.assets.get(asset_id)
        if asset is None:
            return ["observe_inventory"]
        plan = ["inspect_metadata"]
        if "unknown_identity" in asset.uncertainty_tags:
            plan.extend(["compare_known_affordances", "prefer_reversible_test"])
        if any(tag.startswith("buc:") for tag in asset.uncertainty_tags):
            plan.append("avoid_irreversible_equipping_until_verified")
        return plan

__all__ = ["Asset", "AssetModel"]
