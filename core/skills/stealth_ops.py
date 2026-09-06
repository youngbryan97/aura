from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from core.skills.what_every_skill_gives_back import THE_SHARED_RESULT
from core.skills.base_skill import BaseSkill
from core.utils.privacy_hygiene import MetadataScrubber, get_stealth_mode


class StealthOpsInput(BaseModel):
    model_config = ConfigDict(extra="allow")

    command: str = Field("status", description="status|scrub|enable|disable|rotate")
    text: str = ""
    params: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _unwrap_params(cls, value: Any) -> Any:
        if isinstance(value, dict) and isinstance(value.get("params"), dict):
            merged = dict(value)
            nested = dict(merged.pop("params"))
            nested.update(merged)
            return nested
        return value


class StealthOpsSkill(BaseSkill):
    """Privacy-hygiene surface for Aura.

    The stable `stealth_ops` route is deliberately not an evasion, proxy,
    anti-detection, or anonymity-control interface. It exposes local metadata
    hygiene and status only, and refuses active stealth operations unless a
    future governed gateway provides explicit, auditable authorization.
    """
    #: What a caller gets back. The shared part only: every skill
    #: here returns `ok`, and a schema claiming to be complete
    #: would be wrong for every one that adds a field.
    result_schema = THE_SHARED_RESULT


    name = "stealth_ops"
    description = "Inspect and apply local privacy hygiene without changing network identity or hiding activity."
    input_model = StealthOpsInput
    effect_scope = "read_only"
    metabolic_cost = 1
    timeout_seconds = 5.0

    def _status(self) -> dict[str, Any]:
        mode = get_stealth_mode()
        return {
            "ok": True,
            "status": {
                "privacy_hygiene": "available",
                "metadata_scrubbing": bool(getattr(mode, "stealth_enabled", True)),
                "active_network_stealth": "not_available_from_skill",
                "identity_rotation": "blocked_without_governed_gateway",
            },
            "message": "StealthOps is constrained to privacy hygiene and metadata scrubbing.",
        }

    async def execute(self, params: StealthOpsInput | dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        if isinstance(params, dict):
            params = StealthOpsInput(**params)
        command = params.command.strip().lower()

        if command == "status":
            return self._status()

        if command == "scrub":
            scrubber = MetadataScrubber()
            return {
                "ok": True,
                "status": "scrubbed",
                "message": "Text scrubbed for local metadata and obvious secret patterns.",
                "text": scrubber.scrub_text(params.text),
            }

        if command in {"enable", "disable", "rotate", "proxy", "vpn"}:
            return {
                "ok": False,
                "status": "blocked",
                "error": "Active stealth, proxy, VPN, or identity-rotation operations require explicit authorization and a governed external I/O gateway.",
                "message": "No network identity change was attempted.",
            }

        return {
            "ok": False,
            "status": "invalid_command",
            "error": f"Unsupported StealthOps command: {params.command}",
            "supported_commands": ["status", "scrub"],
        }
