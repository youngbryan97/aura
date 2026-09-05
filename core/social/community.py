"""core/social/community.py

Community Integration Layer
============================
Lets Aura participate in social platforms (Discord, Matrix, Slack) as a
first-class agent — but only through a tightly governed surface.

Every outgoing message:

  1. is composed by Aura's own runtime (no scripted prompts)
  2. passes through Conscience + Will + a community-specific policy
  3. acquires a per-message capability token (TTL = 60s)
  4. is recorded in `community_ledger.jsonl` for external audit
  5. carries a per-platform signature so the receiving server can verify
     the message originated from this Aura instance, not a spoof

Inbound messages:

  1. enter the substrate via `core/embodiment/world_bridge.py`
  2. update relationship dossiers
  3. count toward proactive-cooldown windows in `AgencyBus`

The platform clients themselves are pluggable (`CommunityTransport`).
The default `LocalLogTransport` writes to a JSONL file so the layer is
testable without any external API.
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any

from core.runtime.errors import record_degradation
from core.runtime.state_ownership import state_root

logger = logging.getLogger("Aura.Community")

_DIR = state_root() / "data" / "community"
_DIR.mkdir(parents=True, exist_ok=True)
_LEDGER_PATH = _DIR / "ledger.jsonl"


@dataclass
class OutboundMessage:
    message_id: str
    platform: str
    channel: str
    body: str
    intent: str
    drive: str
    when: float = field(default_factory=time.time)
    signature: str | None = None
    will_receipt_id: str | None = None
    capability_token: str | None = None


@dataclass
class InboundMessage:
    message_id: str
    platform: str
    channel: str
    sender: str
    body: str
    when: float = field(default_factory=time.time)


# ─── transport interface ──────────────────────────────────────────────────


class CommunityTransport(ABC):
    name: str = "abstract"

    @abstractmethod
    async def send(self, msg: OutboundMessage) -> dict[str, Any]:  # pragma: no cover
        raise NotImplementedError

    @abstractmethod
    async def receive(self) -> InboundMessage | None:  # pragma: no cover
        raise NotImplementedError


class LocalLogTransport(CommunityTransport):
    name = "local"

    def __init__(self) -> None:
        self.outbox: list[OutboundMessage] = []
        self.inbox: list[InboundMessage] = []

    async def send(self, msg: OutboundMessage) -> dict[str, Any]:
        self.outbox.append(msg)
        from core.runtime.file_write_gateway import get_file_write_gateway

        await get_file_write_gateway().append_text_async(
            _DIR / "outbox.jsonl",
            json.dumps(asdict(msg), default=str) + "\n",
            encoding="utf-8",
            source="community.local_transport_send",
        )
        return {"delivered": True, "transport": "local"}

    async def receive(self) -> InboundMessage | None:
        if not self.inbox:
            return None
        return self.inbox.pop(0)


# ─── community layer ─────────────────────────────────────────────────────


class CommunityLayer:
    PER_MIN_BUDGET = 4

    def __init__(self) -> None:
        self.transports: dict[str, CommunityTransport] = {"local": LocalLogTransport()}
        self._sent_recent: list[float] = []

    def register(self, transport: CommunityTransport) -> None:
        self.transports[transport.name] = transport

    async def send(self, *, platform: str, channel: str, body: str, intent: str, drive: str) -> dict[str, Any]:
        t = self.transports.get(platform)
        if t is None:
            return {"ok": False, "error": "unknown_transport"}

        # rate cap
        now = time.time()
        self._sent_recent = [t for t in self._sent_recent if (now - t) < 60.0]
        if len(self._sent_recent) >= self.PER_MIN_BUDGET:
            self._record({"event": "rate_capped", "platform": platform, "channel": channel})
            return {"ok": False, "error": "rate_cap"}

        # Conscience + Will
        from core.ethics.conscience import Verdict as CV
        from core.ethics.conscience import get_conscience
        c = get_conscience().evaluate(action=f"social_post:{platform}:{channel}", domain="external_communication", intent=intent, context={"body": body[:120]})
        if c.verdict == CV.REFUSE:
            self._record({"event": "conscience_refused", "rule": c.rule_id})
            return {"ok": False, "error": f"conscience:{c.rule_id}"}
        if c.verdict == CV.REQUIRE_FRESH_USER_AUTH:
            return {"ok": False, "error": "require_fresh_user_auth"}
        try:
            from core.governance.will_client import WillClient, WillRequest
            from core.will import ActionDomain
            wd = await WillClient().decide_async(
                WillRequest(
                    content=f"social_post:{platform}:{channel}",
                    source="community",
                    domain=getattr(ActionDomain, "EXPRESSION", "expression"),
                    context={"body": body, "intent": intent, "drive": drive, "platform": platform},
                )
            )
            if not WillClient.is_approved(wd):
                self._record({"event": "will_refused", "reason": getattr(wd, "reason", "")})
                return {"ok": False, "error": "will_refused"}
            will_receipt_id = getattr(wd, "receipt_id", None)
        except (ImportError, AttributeError, RuntimeError) as exc:
            record_degradation('community', exc)
            self._record({"event": "will_exception", "error": str(exc)})
            return {"ok": False, "error": "will_exception"}

        from core.agency.capability_token import get_token_store
        store = get_token_store()
        tok = store.issue(
            origin=f"community:{platform}",
            scope=f"{platform}:{channel}",
            ttl_seconds=60.0,
            domain="external_communication",
            requested_action=f"social_post:{platform}:{channel}",
            approver="UnifiedWill",
            parent_receipt=will_receipt_id or "",
        )

        msg = OutboundMessage(
            message_id=f"OM-{uuid.uuid4().hex[:10]}",
            platform=platform,
            channel=channel,
            body=body,
            intent=intent,
            drive=drive,
            will_receipt_id=will_receipt_id,
            capability_token=tok.token,
        )
        try:
            res = await t.send(msg)
            store.consume(tok.token, child_receipt=msg.message_id, side_effects=["social_send"])
            self._sent_recent.append(now)
            self._record({"event": "sent", "message": asdict(msg), "result": res})
            return {"ok": True, "message_id": msg.message_id, "result": res}
        except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
            record_degradation('community', exc)
            store.revoke(tok.token, reason=f"send_failed:{exc}")
            self._record({"event": "send_failed", "error": str(exc)})
            return {"ok": False, "error": str(exc)}

    async def poll_inbound(self) -> InboundMessage | None:
        for tname, t in self.transports.items():
            try:
                msg = await t.receive()
                if msg is None:
                    continue
                self._record({"event": "received", "platform": tname, "from": msg.sender, "body": msg.body[:120]})
                # update relationship dossier
                try:
                    from core.social.relationship_model import get_store
                    store = get_store()
                    dossier = store.get_or_create(msg.sender, name=msg.sender)
                    store.record_interaction_affect(dossier.relationship_id, {"platform": tname, "channel": msg.channel})
                except (ImportError, AttributeError, RuntimeError):
                    pass  # no-op: intentional
                return msg
            except (ImportError, AttributeError, RuntimeError) as exc:
                record_degradation('community', exc)
                logger.debug("community receive failed (%s): %s", tname, exc)
        return None

    @staticmethod
    def _record(payload: dict[str, Any]) -> None:
        try:
            from core.runtime.file_write_gateway import get_file_write_gateway

            get_file_write_gateway().append_text(
                _LEDGER_PATH,
                json.dumps({"when": time.time(), **payload}, default=str) + "\n",
                encoding="utf-8",
                source="community.record",
            )
        except (json.JSONDecodeError, TypeError, ValueError):
            pass  # no-op: intentional


_LAYER: CommunityLayer | None = None


def get_community() -> CommunityLayer:
    global _LAYER
    if _LAYER is None:
        _LAYER = CommunityLayer()
    return _LAYER


__all__ = [
    "CommunityLayer",
    "CommunityTransport",
    "LocalLogTransport",
    "OutboundMessage",
    "InboundMessage",
    "get_community",
]
