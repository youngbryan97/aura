"""core/tools/tool_manifest.py — Tool Manifest and Metadata."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolManifest:
    """Describes the identity, requirements, and security sandbox policies of a tool."""
    name: str
    version: str
    owner: str
    hash_sha256: str
    signature: str
    risk_tier: str  # "low", "medium", "high"
    allowed_domains: list[str] = field(default_factory=list)
    allowed_directories: list[str] = field(default_factory=list)
    permissions: list[str] = field(default_factory=list)
    sandbox_required: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "owner": self.owner,
            "hash_sha256": self.hash_sha256,
            "signature": self.signature,
            "risk_tier": self.risk_tier,
            "allowed_domains": self.allowed_domains,
            "allowed_directories": self.allowed_directories,
            "permissions": self.permissions,
            "sandbox_required": self.sandbox_required,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ToolManifest:
        return cls(
            name=data["name"],
            version=data["version"],
            owner=data["owner"],
            hash_sha256=data["hash_sha256"],
            signature=data["signature"],
            risk_tier=data["risk_tier"],
            allowed_domains=data.get("allowed_domains", []),
            allowed_directories=data.get("allowed_directories", []),
            permissions=data.get("permissions", []),
            sandbox_required=data.get("sandbox_required", True),
        )
