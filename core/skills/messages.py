"""Governed private Messages conversation skill."""

from __future__ import annotations

from typing import Any, Literal, Protocol, cast

from pydantic import BaseModel, ConfigDict, Field

from core.communication.contact_directory import DEFAULT_MESSAGES_CONTACT_ALIAS
from core.container import ServiceContainer
from core.skills.base_skill import BaseSkill
from core.skills.what_every_skill_gives_back import THE_SHARED_RESULT


class _MessagesTransport(Protocol):
    def status(self) -> dict[str, Any]: ...

    async def set_paused_authorized(
        self,
        *,
        paused: bool,
        source: str,
        context: dict[str, Any],
    ) -> dict[str, Any]: ...

    async def send_authorized(
        self,
        *,
        alias: str,
        body: str,
        idempotency_key: str | None,
        source: str,
        context: dict[str, Any],
    ) -> dict[str, Any]: ...

    async def send_from_governed_context(
        self,
        *,
        alias: str,
        body: str,
        idempotency_key: str | None,
        context: dict[str, Any],
    ) -> dict[str, Any]: ...


class MessagesInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Literal["status", "send", "pause", "resume"] = "status"
    alias: str = Field(default=DEFAULT_MESSAGES_CONTACT_ALIAS, min_length=3, max_length=64)
    body: str | None = Field(default=None, max_length=8_000)
    idempotency_key: str | None = Field(default=None, max_length=240)


class MessagesSkill(BaseSkill):
    """Let Aura converse privately with her configured operator over Messages."""
    #: What a caller gets back. The shared part only: every skill
    #: here returns `ok`, and a schema claiming to be complete
    #: would be wrong for every one that adds a field.
    result_schema = THE_SHARED_RESULT


    name = "messages"
    description = (
        "Use my private two-way Messages channel with my configured primary operator. "
        "I can inspect channel status, send a message in my own words, or pause and "
        "resume the channel. Contacts are symbolic Keychain aliases; raw phone numbers "
        "or addresses are never tool arguments."
    )
    effect_scope = "external_io"
    input_model = MessagesInput
    retry_safe = False
    timeout_seconds = 45.0
    metabolic_cost = 1

    async def execute(
        self,
        params: MessagesInput | dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        if isinstance(params, dict):
            params = MessagesInput(**params)
        transport = cast(
            _MessagesTransport | None,
            ServiceContainer.get("messages_transport", default=None),
        )
        if transport is None:
            return {
                "ok": False,
                "error": "Private Messages transport is not running.",
                "status": "transport_unavailable",
            }
        action = params.action
        if action == "status":
            return {
                "ok": True,
                "status": transport.status(),
                "summary": "Private Messages channel status inspected.",
            }
        source = str(context.get("source") or context.get("origin") or "skills.messages")
        if action in {"pause", "resume"}:
            return await transport.set_paused_authorized(
                paused=action == "pause",
                source=source,
                context=context,
            )
        if not params.body:
            return {"ok": False, "error": "Send requires a non-empty message body."}
        if context.get("signed_capability") and context.get("capability_token_id"):
            result = await transport.send_from_governed_context(
                alias=params.alias,
                body=params.body,
                idempotency_key=params.idempotency_key,
                context=context,
            )
        else:
            result = await transport.send_authorized(
                alias=params.alias,
                body=params.body,
                idempotency_key=params.idempotency_key,
                source=source,
                context=context,
            )
        if result.get("ok"):
            result.setdefault(
                "summary",
                "Message accepted by Messages; local-history verification is reported separately.",
            )
        return result


__all__ = ["MessagesInput", "MessagesSkill"]
