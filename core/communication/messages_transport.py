"""Private two-way Messages transport over Aura's canonical conversation lane.

Incoming owner messages are read from the local Messages history database using
a read-only SQLite connection and handed to the exact `/api/chat` contract used
by the desktop and voice surfaces. Outgoing effects are authorized by the full
AuthorityGateway, capability-bound at the sink, journaled before launch, and
sent through a static JXA program whose destination and body arrive only over
stdin. No contact handle or message body is written to logs or receipts.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import sqlite3
import subprocess
import sys
import time
import uuid
from collections.abc import Awaitable, Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from core.communication.contact_directory import (
    DEFAULT_MESSAGES_CONTACT_ALIAS,
    ContactDirectoryError,
    ContactNotConfiguredError,
    KeychainContactDirectory,
    MessagesContact,
)
from core.communication.messages_history import (
    MAX_BATCH as _MAX_BATCH,
)
from core.communication.messages_history import (
    MAX_MESSAGE_BYTES as _MAX_MESSAGE_BYTES,
)
from core.communication.messages_history import (
    MAX_MESSAGE_CHARS as _MAX_MESSAGE_CHARS,
)
from core.communication.messages_history import (
    HistoryMessage,
    MessagesHistoryReader,
    MessagesHistoryUnavailableError,
    MessagesTransportError,
)
from core.communication.messages_journal import (
    MessagesDeliveryJournal,
    content_digest,
)
from core.executive.execution_policy import canonical_authority_arguments
from core.governance.capability_chain import CapabilityViolation, enforce_capability
from core.governance_context import governed_scope, require_governance
from core.runtime.errors import record_degradation
from core.runtime.lockdep import checked_async_lock
from core.runtime.subprocess_gateway import SubprocessGateway, get_subprocess_gateway
from core.utils.task_tracker import get_task_tracker

logger = logging.getLogger("Aura.MessagesTransport")

_MESSAGES_APP_PATH = Path("/System/Applications/Messages.app")
_HOURLY_SEND_LIMIT = 60
_BURST_SEND_LIMIT = 8
_POLL_INTERVAL_S = 3.0
_CONTACT_RETRY_S = 30.0
_HISTORY_RETRY_S = 60.0
_CHAT_TIMEOUT_S = 180.0


class MessagesSendUnavailableError(MessagesTransportError):
    """Messages.app or its automation surface is unavailable."""


class MessagesSendAmbiguousError(MessagesTransportError):
    """The send process started but terminal effect evidence is unavailable."""


_SEND_JXA = r"""
(() => {
  'use strict';
  ObjC.import('Foundation');
  const inputData = $.NSFileHandle.fileHandleWithStandardInput.readDataToEndOfFile;
  const inputText = ObjC.unwrap(
    $.NSString.alloc.initWithDataEncoding(inputData, $.NSUTF8StringEncoding)
  );
  const payload = JSON.parse(String(inputText));
  if (typeof payload.destination !== 'string' || typeof payload.body !== 'string') {
    throw new Error('messages_payload_invalid');
  }
  const messages = Application('/System/Applications/Messages.app');
  messages.includeStandardAdditions = false;
  const preferred = String(payload.service || 'auto').toLowerCase();
  const accounts = messages.accounts().filter(account => {
    try { return Boolean(account.enabled()); } catch (_) { return false; }
  });
  const ordered = accounts.sort((left, right) => {
    const rank = account => {
      let kind = '';
      try { kind = String(account.serviceType()).toLowerCase(); } catch (_) {}
      if (preferred === 'sms') return kind.includes('sms') ? 0 : 1;
      if (preferred === 'imessage') return kind.includes('imessage') ? 0 : 1;
      return kind.includes('imessage') ? 0 : (kind.includes('sms') ? 1 : 2);
    };
    return rank(left) - rank(right);
  });
  let participant = null;
  for (const account of ordered) {
    let matches = [];
    try {
      matches = account.participants.whose({handle: payload.destination})();
    } catch (_) {}
    if (matches.length > 0) {
      participant = matches[0];
      break;
    }
  }
  if (participant === null) {
    let matches = [];
    try {
      matches = messages.participants.whose({handle: payload.destination})();
    } catch (_) {}
    if (matches.length > 0) participant = matches[0];
  }
  if (participant === null) {
    throw new Error('messages_participant_unavailable');
  }
  messages.send(payload.body, {to: participant});
  return JSON.stringify({ok: true, accepted: true, transport: 'messages'});
})()
""".strip()


class MessagesJXADriver:
    """Effect driver whose argv is static and whose sensitive payload uses stdin."""

    def __init__(self, gateway: SubprocessGateway | None = None) -> None:
        self._gateway = gateway or get_subprocess_gateway()

    @staticmethod
    def preflight() -> None:
        if sys.platform != "darwin" or not _MESSAGES_APP_PATH.is_dir():
            raise MessagesSendUnavailableError("messages_app_unavailable")
        if not Path("/usr/bin/osascript").is_file():
            raise MessagesSendUnavailableError("messages_automation_unavailable")

    async def send(
        self,
        *,
        destination: str,
        body: str,
        service_preference: str,
    ) -> dict[str, Any]:
        self.preflight()
        payload = json.dumps(
            {
                "body": body,
                "destination": destination,
                "service": service_preference,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        completed = await self._gateway.run_async(
            ["/usr/bin/osascript", "-l", "JavaScript", "-e", _SEND_JXA],
            timeout=20.0,
            read_only=False,
            capture_output=True,
            input=payload,
            check=False,
            source="messages_transport.send",
            accelerator_capability="none",
        )
        if int(completed.returncode) != 0:
            raise MessagesSendAmbiguousError("messages_automation_returned_failure")
        try:
            receipt: Any = str(completed.stdout or "").strip()
            for _ in range(2):
                receipt = json.loads(receipt) if isinstance(receipt, str) else receipt
                if not isinstance(receipt, str):
                    break
        except (TypeError, ValueError) as exc:
            raise MessagesSendAmbiguousError("messages_automation_receipt_invalid") from exc
        if not isinstance(receipt, dict) or not receipt.get("accepted"):
            raise MessagesSendAmbiguousError("messages_automation_acceptance_missing")
        return {"accepted": True, "transport": "messages"}

    @property
    def static_argv(self) -> tuple[str, ...]:
        return ("/usr/bin/osascript", "-l", "JavaScript", "-e", _SEND_JXA)


ChatTurn = Callable[..., Awaitable[str | None]]
TaskFactory = Callable[..., asyncio.Task[Any]]


def _validated_body(value: Any) -> str:
    body = str(value or "").strip()
    if not body:
        raise ValueError("Messages body is required")
    if any(character == "\x00" for character in body):
        raise ValueError("Messages body contains an invalid null character")
    encoded = body.encode("utf-8", errors="strict")
    if len(body) > _MAX_MESSAGE_CHARS or len(encoded) > _MAX_MESSAGE_BYTES:
        raise ValueError("Messages body exceeds the bounded private-message limit")
    return body


def _validated_idempotency(value: Any) -> str:
    key = str(value or "").strip()
    if not key:
        key = "messages." + uuid.uuid4().hex
    if len(key) > 240 or not all(
        character.isalnum() or character in "._:-" for character in key
    ):
        raise ValueError("Messages idempotency key is invalid")
    return key


class MessagesTransport:
    """Long-lived private conversation transport with restart-safe effects."""

    def __init__(
        self,
        *,
        chat_turn: ChatTurn,
        directory: KeychainContactDirectory | None = None,
        journal: MessagesDeliveryJournal | None = None,
        history: MessagesHistoryReader | None = None,
        driver: MessagesJXADriver | None = None,
        aliases: Sequence[str] = (DEFAULT_MESSAGES_CONTACT_ALIAS,),
        poll_interval_s: float = _POLL_INTERVAL_S,
        clock: Callable[[], float] = lambda: time.time(),
    ) -> None:
        if not callable(chat_turn):
            raise TypeError("Messages transport requires a canonical chat callback")
        normalized_aliases = tuple(dict.fromkeys(str(item).strip().lower() for item in aliases))
        if not normalized_aliases:
            raise ValueError("Messages transport requires at least one contact alias")
        self._chat_turn = chat_turn
        self._directory = directory or KeychainContactDirectory()
        self._journal = journal or MessagesDeliveryJournal()
        self._history = history or MessagesHistoryReader()
        self._driver = driver or MessagesJXADriver()
        self._aliases = normalized_aliases
        self._poll_interval_s = max(0.25, min(float(poll_interval_s), 60.0))
        self._clock = clock
        self._task: asyncio.Task[Any] | None = None
        self._stop_event = asyncio.Event()
        self._paused = False
        self._contacts: dict[str, MessagesContact] = {}
        self._next_contact_refresh = 0.0
        self._next_history_probe = 0.0
        self._history_state = "unknown"
        self._last_error_code = ""
        self._last_poll_at = 0.0
        self._last_inbound_at = 0.0
        self._last_outbound_at = 0.0
        self._processed_inbound = 0
        self._accepted_outbound = 0
        self._send_lock = checked_async_lock("messages_transport")

    async def start(self, *, task_factory: TaskFactory | None = None) -> None:
        if self._task is not None and not self._task.done():
            return
        self._stop_event.clear()
        coroutine = self._run()
        if task_factory is None:
            self._task = get_task_tracker().create_task(
                coroutine,
                name="messages_transport.poll",
            )
        else:
            self._task = task_factory(coroutine, name="messages_transport.poll")

    async def stop(self) -> None:
        task = self._task
        if task is None:
            return
        owner_loop = task.get_loop()
        current_loop = asyncio.get_running_loop()
        if owner_loop is current_loop:
            await self._stop_owned_task(task)
            return
        if owner_loop.is_closed() or not owner_loop.is_running():
            # There is no loop on which this task can make further progress.
            # Detach the stale handle rather than trying to await a foreign
            # Future, which raises and aborts the rest of ordered shutdown.
            if self._task is task:
                self._task = None
            return

        # The transport is created by the uvicorn lifespan loop but is also a
        # ServiceContainer member. Ordered process shutdown runs on aura_main's
        # loop and can therefore reach this hook before uvicorn's own lifespan
        # cleanup. Cancellation and awaiting must happen on the task's owner.
        try:
            stop_future = asyncio.run_coroutine_threadsafe(
                self._stop_owned_task(task),
                owner_loop,
            )
        except RuntimeError:
            # The owner can finish between the liveness check and submission.
            if task.done() or owner_loop.is_closed() or not owner_loop.is_running():
                if self._task is task:
                    self._task = None
                return
            raise
        await asyncio.wait_for(
            asyncio.wrap_future(stop_future),
            timeout=max(2.0, self._poll_interval_s + 2.0),
        )

    async def _stop_owned_task(self, task: asyncio.Task[Any]) -> None:
        """Cancel one polling generation on its owning event loop."""

        if self._task is not task:
            return
        self._stop_event.set()
        if not task.done():
            task.cancel()
        try:
            await task
        # not a failure: the module is optional here, and its absence is the answer.
        except asyncio.CancelledError:
            pass
        finally:
            if self._task is task:
                self._task = None

    async def on_stop_async(self) -> None:
        await self.stop()

    def status(self) -> dict[str, Any]:
        contacts = {
            alias: contact.public_status()
            for alias, contact in sorted(self._contacts.items())
        }
        return {
            "running": bool(self._task is not None and not self._task.done()),
            "paused": self._paused,
            "configured": bool(contacts),
            "contacts": contacts,
            "history_state": self._history_state,
            "inbound_ready": bool(
                not self._paused
                and self._history_state == "ready"
                and any(contact.allow_inbound for contact in self._contacts.values())
            ),
            "outbound_ready": bool(
                not self._paused
                and any(contact.allow_outbound for contact in self._contacts.values())
            ),
            "last_error_code": self._last_error_code,
            "last_poll_at": self._last_poll_at,
            "last_inbound_at": self._last_inbound_at,
            "last_outbound_at": self._last_outbound_at,
            "processed_inbound": self._processed_inbound,
            "accepted_outbound": self._accepted_outbound,
            "delivery_semantics": {
                "local_history_verification": True,
                "remote_delivery_receipts": False,
                "remote_read_receipts": False,
                "ambiguous_effects_are_never_blindly_retried": True,
            },
        }

    async def _run(self) -> None:
        while not self._stop_event.is_set():
            now = float(self._clock())
            try:
                if now >= self._next_contact_refresh:
                    await self._refresh_contacts(now)
                if not self._paused and self._contacts and now >= self._next_history_probe:
                    await self._poll_inbound(now)
                self._last_poll_at = now
            except asyncio.CancelledError:
                raise
            except (
                ContactDirectoryError,
                MessagesHistoryUnavailableError,
                MessagesTransportError,
                OSError,
                RuntimeError,
                sqlite3.Error,
                TypeError,
                ValueError,
            ) as exc:
                self._note_error(type(exc).__name__)
                self._next_history_probe = now + _HISTORY_RETRY_S
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=self._poll_interval_s,
                )
            # not a failure: the wait ran out, which is what the timeout was set to decide.
            except TimeoutError:
                pass

    async def _refresh_contacts(self, now: float) -> None:
        loaded: dict[str, MessagesContact] = {}
        unavailable = False
        for alias in self._aliases:
            try:
                loaded[alias] = await self._directory.load_async(alias)
            except ContactNotConfiguredError:
                continue
            except ContactDirectoryError:
                unavailable = True
        self._contacts = loaded
        self._next_contact_refresh = now + _CONTACT_RETRY_S
        if unavailable:
            self._note_error("contact_directory_unavailable")
        elif not loaded:
            self._last_error_code = "contact_not_configured"
        elif self._last_error_code in {
            "contact_directory_unavailable",
            "contact_not_configured",
        }:
            self._last_error_code = ""

    async def _poll_inbound(self, now: float) -> None:
        try:
            await self._history.probe()
        except MessagesHistoryUnavailableError:
            self._history_state = "permission_or_history_unavailable"
            self._next_history_probe = now + _HISTORY_RETRY_S
            return
        self._history_state = "ready"
        self._next_history_probe = now
        if self._last_error_code in {
            "MessagesHistoryUnavailableError",
            "messages_history_unavailable",
        }:
            self._last_error_code = ""
        for contact in tuple(self._contacts.values()):
            if not contact.allow_inbound:
                continue
            cursor = await self._journal.cursor(contact.endpoint_ref)
            if cursor is None:
                latest = await self._history.latest_row_id(
                    contact.destination,
                    from_me=False,
                )
                await self._journal.prime_cursor(contact.endpoint_ref, latest)
                continue
            messages = await self._history.messages_after(
                contact.destination,
                from_me=False,
                after_row_id=cursor,
                limit=_MAX_BATCH,
            )
            for message in messages:
                completed = await self._process_inbound(contact, message)
                if not completed:
                    break

    async def _process_inbound(
        self,
        contact: MessagesContact,
        message: HistoryMessage,
    ) -> bool:
        guid_material = message.guid or f"{contact.endpoint_ref}:{message.row_id}"
        guid_sha256 = hashlib.sha256(guid_material.encode("utf-8")).hexdigest()
        body_sha256 = content_digest(message.text)
        state = await self._journal.claim_inbound(
            endpoint_ref=contact.endpoint_ref,
            source_row_id=message.row_id,
            guid_sha256=guid_sha256,
            content_sha256=body_sha256,
        )
        if state == "completed":
            return True
        chat_key = "messages-in-" + guid_sha256[:48]
        surface_context = (
            "[Authenticated private Messages conversation with Aura's configured "
            "primary operator. This is the same continuous relationship and memory "
            "lane as the desktop UI. Respond in Aura's own words, naturally and "
            "concisely for text.]"
        )
        try:
            reply = await self._chat_turn(
                message.text,
                surface="messages",
                surface_context=surface_context,
                session_id=f"messages-{contact.alias}",
                timeout_s=_CHAT_TIMEOUT_S,
                idempotency_key=chat_key,
                client_host="127.0.0.1",
            )
        except (OSError, RuntimeError, TimeoutError, TypeError, ValueError):
            reply = None
        if not reply:
            await self._journal.mark_inbound_retryable(
                guid_sha256,
                "canonical_chat_unavailable",
            )
            return False
        send_result = await self.send_authorized(
            alias=contact.alias,
            body=reply,
            idempotency_key="messages-reply-" + guid_sha256[:48],
            source="messages",
            context={
                "authenticated": True,
                "foreground_request": True,
                "user_explicit_action_request": True,
                "user_explicitly_authorized": True,
                "user_requested_action": True,
                "private_owner_channel": True,
            },
        )
        state = str(send_result.get("state") or "")
        terminal = state in {
            "accepted_unverified",
            "verified_local_history",
            "ambiguous",
        }
        if not terminal:
            await self._journal.mark_inbound_retryable(
                guid_sha256,
                str(send_result.get("error_code") or "reply_send_unavailable"),
            )
            return False
        await self._journal.complete_inbound(
            endpoint_ref=contact.endpoint_ref,
            source_row_id=message.row_id,
            guid_sha256=guid_sha256,
            response_sha256=content_digest(reply),
            outcome_code=state,
        )
        self._last_inbound_at = float(self._clock())
        self._processed_inbound += 1
        return True

    async def send_authorized(
        self,
        *,
        alias: str,
        body: str,
        idempotency_key: str | None,
        source: str,
        context: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        normalized_alias = str(alias or DEFAULT_MESSAGES_CONTACT_ALIAS).strip().lower()
        normalized_body = _validated_body(body)
        normalized_key = _validated_idempotency(idempotency_key)
        arguments = canonical_authority_arguments(
            "messages",
            {
                "action": "send",
                "alias": normalized_alias,
                "body": normalized_body,
                "idempotency_key": normalized_key,
            },
        )
        gateway = None
        authority = None
        result: dict[str, Any] = {
            "ok": False,
            "state": "failed_before_effect",
            "error_code": "authority_not_started",
        }
        try:
            from core.executive.authority_gateway import get_authority_gateway

            gateway = get_authority_gateway()
            authority = await gateway.authorize_tool_execution(
                "messages",
                arguments,
                source=str(source or "unknown"),
                priority=0.65,
                is_critical=False,
                context=dict(context or {}),
            )
            if not authority.approved:
                result = {
                    "ok": False,
                    "state": "failed_before_effect",
                    "error_code": "authority_refused",
                    "authority_reason": str(authority.reason or "refused")[:240],
                }
                return result
            if not gateway.verify_tool_access("messages", authority.capability_token_id):
                result = {
                    "ok": False,
                    "state": "failed_before_effect",
                    "error_code": "capability_token_invalid",
                }
                return result
            if not authority.signed_capability:
                result = {
                    "ok": False,
                    "state": "failed_before_effect",
                    "error_code": "signed_capability_missing",
                }
                return result
            result = await self.send(
                alias=normalized_alias,
                body=normalized_body,
                idempotency_key=normalized_key,
                authority=authority,
                arguments=arguments,
            )
            return result
        except (
            CapabilityViolation,
            ContactDirectoryError,
            MessagesTransportError,
            OSError,
            RuntimeError,
            sqlite3.Error,
            TypeError,
            ValueError,
        ) as exc:
            self._note_error(type(exc).__name__)
            result = {
                "ok": False,
                "state": "failed_before_effect",
                "error_code": type(exc).__name__,
            }
            return result
        finally:
            if gateway is not None and authority is not None:
                try:
                    gateway.finalize_tool_execution(
                        executive_intent_id=authority.executive_intent_id,
                        capability_token_id=authority.capability_token_id,
                        standing_authority_token=authority.standing_authority_token,
                        success=bool(result.get("ok")),
                        result={
                            "state": str(result.get("state") or "unknown"),
                            "accepted": bool(result.get("accepted")),
                        },
                        error=str(result.get("error_code") or "")[:120],
                    )
                except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
                    self._note_error(f"authority_finalize_{type(exc).__name__}")

    async def send(
        self,
        *,
        alias: str,
        body: str,
        idempotency_key: str,
        authority: Any,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        normalized_body = _validated_body(body)
        normalized_key = _validated_idempotency(idempotency_key)
        expected_arguments = canonical_authority_arguments(
            "messages",
            {
                "action": "send",
                "alias": alias,
                "body": normalized_body,
                "idempotency_key": normalized_key,
            },
        )
        if arguments != expected_arguments:
            raise ValueError("Messages authority envelope does not match the execution payload")
        enforce_capability(
            {"signed_capability": getattr(authority, "signed_capability", None)},
            sink="messages_transport.send",
            domain="tool_execution",
            action="messages",
            payload=expected_arguments,
        )
        async with governed_scope(authority, ttl=45.0):
            return await self._send_with_bound_capability(
                alias=alias,
                body=normalized_body,
                idempotency_key=normalized_key,
            )

    async def send_from_governed_context(
        self,
        *,
        alias: str,
        body: str,
        idempotency_key: str | None,
        context: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Consume the authority already issued by CapabilityEngine.

        The canonical skill path is already inside the constitutional governed
        scope. Re-authorizing here would consume a second standing-authority
        lease and could turn one valid request into a false refusal. The sink
        therefore verifies the original capability token and signed envelope,
        then executes under the active scope established by CapabilityEngine.
        """

        normalized_alias = str(alias or DEFAULT_MESSAGES_CONTACT_ALIAS).strip().lower()
        normalized_body = _validated_body(body)
        normalized_key = _validated_idempotency(idempotency_key)
        arguments = canonical_authority_arguments(
            "messages",
            {
                "action": "send",
                "alias": normalized_alias,
                "body": normalized_body,
                "idempotency_key": normalized_key,
            },
        )
        capability_token_id = str(context.get("capability_token_id") or "").strip()
        signed_capability = context.get("signed_capability")
        if not capability_token_id or not signed_capability:
            return {
                "ok": False,
                "state": "failed_before_effect",
                "error_code": "existing_authority_missing",
            }
        from core.executive.authority_gateway import get_authority_gateway

        gateway = get_authority_gateway()
        if not gateway.verify_tool_access("messages", capability_token_id):
            return {
                "ok": False,
                "state": "failed_before_effect",
                "error_code": "capability_token_invalid",
            }
        require_governance(
            "messages_transport.send_from_governed_context",
            strict=True,
            allowed_domains=("tool_execution",),
        )
        enforce_capability(
            {"signed_capability": signed_capability},
            sink="messages_transport.send",
            domain="tool_execution",
            action="messages",
            payload=arguments,
        )
        return await self._send_with_bound_capability(
            alias=normalized_alias,
            body=normalized_body,
            idempotency_key=normalized_key,
        )

    async def _send_with_bound_capability(
        self,
        *,
        alias: str,
        body: str,
        idempotency_key: str,
    ) -> dict[str, Any]:
        async with self._send_lock:
            return await self._send_serialized(
                alias=alias,
                body=body,
                idempotency_key=idempotency_key,
            )

    async def _send_serialized(
        self,
        *,
        alias: str,
        body: str,
        idempotency_key: str,
    ) -> dict[str, Any]:
        if self._paused:
            return {
                "ok": False,
                "state": "failed_before_effect",
                "error_code": "messages_transport_paused",
            }
        contact = await self._directory.load_async(alias)
        self._contacts[contact.alias] = contact
        if not contact.allow_outbound:
            return {
                "ok": False,
                "state": "failed_before_effect",
                "error_code": "contact_outbound_not_permitted",
            }
        body_sha256 = content_digest(body)
        existing = await self._journal.lookup_outbound(
            idempotency_key=idempotency_key,
            endpoint_ref=contact.endpoint_ref,
            content_sha256=body_sha256,
        )
        if existing is not None and not existing.may_execute:
            receipt = existing.public_receipt()
            return {"ok": bool(receipt.get("accepted")), **receipt}

        self._driver.preflight()
        now = float(self._clock())
        burst = await self._journal.recent_outbound_attempts(
            contact.endpoint_ref,
            since=now - 60.0,
        )
        hourly = await self._journal.recent_outbound_attempts(
            contact.endpoint_ref,
            since=now - 3600.0,
        )
        if burst >= _BURST_SEND_LIMIT or hourly >= _HOURLY_SEND_LIMIT:
            return {
                "ok": False,
                "state": "failed_before_effect",
                "error_code": "messages_rate_limited",
                "retryable": True,
            }
        baseline: int | None = None
        try:
            baseline = await self._history.latest_row_id(
                contact.destination,
                from_me=True,
            )
        except MessagesHistoryUnavailableError:
            baseline = None
        admission = await self._journal.admit_outbound(
            idempotency_key=idempotency_key,
            endpoint_ref=contact.endpoint_ref,
            content_sha256=body_sha256,
            baseline_row_id=baseline,
        )
        if not admission.may_execute:
            return {"ok": admission.public_receipt().get("accepted", False), **admission.public_receipt()}
        admission = await self._journal.mark_outbound_sending(idempotency_key)
        if not admission.may_execute:
            receipt = admission.public_receipt()
            return {"ok": bool(receipt.get("accepted")), **receipt}
        try:
            await self._driver.send(
                destination=contact.destination,
                body=body,
                service_preference=contact.service_preference,
            )
        except asyncio.CancelledError:
            await self._journal.mark_outbound_terminal(
                idempotency_key,
                state="ambiguous",
                error_code="send_cancelled_after_effect_admission",
            )
            raise
        except (
            MessagesTransportError,
            OSError,
            RuntimeError,
            subprocess.SubprocessError,
            TimeoutError,
            TypeError,
            ValueError,
        ) as exc:
            terminal = await self._journal.mark_outbound_terminal(
                idempotency_key,
                state="ambiguous",
                error_code=type(exc).__name__,
            )
            return {"ok": False, **terminal.public_receipt()}
        observed_row_id = await self._verify_local_history(
            contact,
            body_sha256=body_sha256,
            baseline_row_id=admission.baseline_row_id,
        )
        state = "verified_local_history" if observed_row_id is not None else "accepted_unverified"
        terminal = await self._journal.mark_outbound_terminal(
            idempotency_key,
            state=state,
            observed_row_id=observed_row_id,
        )
        self._last_outbound_at = float(self._clock())
        self._accepted_outbound += 1
        return {"ok": True, **terminal.public_receipt()}

    async def _verify_local_history(
        self,
        contact: MessagesContact,
        *,
        body_sha256: str,
        baseline_row_id: int | None,
    ) -> int | None:
        if baseline_row_id is None:
            return None
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            try:
                rows = await self._history.messages_after(
                    contact.destination,
                    from_me=True,
                    after_row_id=baseline_row_id,
                    limit=_MAX_BATCH,
                )
            except MessagesHistoryUnavailableError:
                return None
            for row in rows:
                if content_digest(row.text) == body_sha256:
                    return row.row_id
            await asyncio.sleep(0.25)
        return None

    async def set_paused_authorized(
        self,
        *,
        paused: bool,
        source: str,
        context: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        action = "pause" if paused else "resume"
        arguments = canonical_authority_arguments(
            "messages",
            {"action": action, "alias": DEFAULT_MESSAGES_CONTACT_ALIAS},
        )
        from core.executive.authority_gateway import get_authority_gateway

        gateway = get_authority_gateway()
        authority = await gateway.authorize_tool_execution(
            "messages",
            arguments,
            source=str(source or "unknown"),
            priority=0.5,
            context=dict(context or {}),
        )
        result = {"ok": False, "error_code": "authority_refused"}
        try:
            if (
                not authority.approved
                or not gateway.verify_tool_access(
                    "messages", authority.capability_token_id
                )
                or not authority.signed_capability
            ):
                return result
            enforce_capability(
                {"signed_capability": authority.signed_capability},
                sink="messages_transport.control",
                domain="tool_execution",
                action="messages",
                payload=arguments,
            )
            async with governed_scope(authority):
                self._paused = bool(paused)
            result = {"ok": True, "paused": self._paused, "status": self.status()}
            return result
        finally:
            gateway.finalize_tool_execution(
                executive_intent_id=authority.executive_intent_id,
                capability_token_id=authority.capability_token_id,
                standing_authority_token=authority.standing_authority_token,
                success=bool(result.get("ok")),
                result={"paused": self._paused},
                error=str(result.get("error_code") or "")[:120],
            )

    def _note_error(self, code: str) -> None:
        normalized = str(code or "messages_transport_error")[:120]
        changed = normalized != self._last_error_code
        self._last_error_code = normalized
        if changed and normalized not in {
            "contact_not_configured",
            "contact_directory_unavailable",
        }:
            record_degradation(
                "messages_transport",
                MessagesTransportError(normalized),
                severity="warning",
                action="kept private messaging isolated and retryable without duplicating effects",
                enforce_failure_policy=False,
            )


__all__ = [
    "HistoryMessage",
    "MessagesHistoryReader",
    "MessagesHistoryUnavailableError",
    "MessagesJXADriver",
    "MessagesSendAmbiguousError",
    "MessagesSendUnavailableError",
    "MessagesTransport",
    "MessagesTransportError",
]
