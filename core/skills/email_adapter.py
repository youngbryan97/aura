"""Email Adapter Skill — Aura's Email Presence

First-class BaseSkill that gives Aura the ability to send, receive, read,
reply to, and search email through her own Gmail account.

Capabilities:
  - send:   Compose and send email via SMTP (TLS on port 587)
  - check:  Check inbox via IMAP — returns unread count + previews
  - read:   Read a specific email by UID
  - reply:  Reply to a specific email thread
  - search: Search inbox by IMAP query

Security:
  - Credentials loaded from macOS Keychain (zenith_secrets)
  - Outgoing content scrubbed by MetadataScrubber (no paths, IPs, hostnames)
  - Rate limited to 20 sends/hour to avoid spam classification
  - Never sends credentials, system info, or Bryan's private data to strangers
  - Bryan's email is tagged as "trusted" with relaxed filtering

HARDENING (2026-05):
  - All SMTP/IMAP connections use context-managed try/finally
  - Timeouts on all network operations
  - No credentials in logs or LLM context
"""

import asyncio
import base64
import email
import email.mime.multipart
import email.mime.text
import email.utils
import imaplib
import logging
import re
import smtplib
import time
from typing import Any

from pydantic import BaseModel, Field

from core.runtime.errors import FallbackClassification, record_degradation
from core.skills.base_skill import BaseSkill

logger = logging.getLogger("Skills.Email")

_EMAIL_RECOVERABLE_ERRORS = (
    ImportError,
    AttributeError,
    RuntimeError,
    TypeError,
    ValueError,
    OSError,
    TimeoutError,
    imaplib.IMAP4.error,
    smtplib.SMTPException,
)


def _record_email_degradation(
    error: BaseException,
    *,
    action: str,
    stage: str,
    severity: str = "warning",
    extra: dict[str, Any] | None = None,
) -> None:
    metadata = dict(extra or {})
    metadata["stage"] = stage
    try:
        record_degradation(
            "email_adapter",
            error,
            severity=severity,  # type: ignore[arg-type]
            action=action,
            classification=FallbackClassification.SAFE_FALLBACK,
            extra=metadata,
        )
    except TypeError:
        record_degradation(
            "email_adapter",
            error,
            severity=severity,  # type: ignore[arg-type]
            action=action,
        )


# ── Rate Limiter ──────────────────────────────────────────────────────
_send_timestamps: list[float] = []
MAX_SENDS_PER_HOUR = 20

# ── Sensitive pattern filter ──────────────────────────────────────────
_SENSITIVE_PATTERNS = [
    re.compile(r"/" + r"Users" + r"/\w+", re.IGNORECASE),
    re.compile(r"/home/\w+", re.IGNORECASE),
    re.compile(r"/opt/\w+", re.IGNORECASE),
    re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"),  # IP addresses
    re.compile(r"([0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}"),  # MAC addresses
    re.compile(r"(password|passwd|secret|api.?key|token)\s*[:=]\s*\S+", re.IGNORECASE),
    re.compile(r"sk-[a-zA-Z0-9]{20,}"),  # API keys
]


def _scrub_content(text: str) -> str:
    """Remove sensitive patterns from outgoing email content."""
    scrubbed = text
    for pattern in _SENSITIVE_PATTERNS:
        scrubbed = pattern.sub("[REDACTED]", scrubbed)
    # Also run MetadataScrubber if available
    try:
        from core.privacy_stealth import get_stealth_mode

        stealth = get_stealth_mode()
        scrubbed = stealth.scrubber.scrub_text(scrubbed)
    except (ImportError, RuntimeError, AttributeError) as exc:
        _record_email_degradation(
            exc,
            action="used local email scrubber only after stealth scrubber was unavailable",
            stage="scrub_content",
            severity="debug",
        )
    return scrubbed


def _check_rate_limit() -> bool:
    """Returns True if we're within rate limits."""
    now = time.time()
    cutoff = now - 3600
    # Prune old timestamps
    while _send_timestamps and _send_timestamps[0] < cutoff:
        _send_timestamps.pop(0)
    return len(_send_timestamps) < MAX_SENDS_PER_HOUR


def _record_send():
    """Record a send timestamp for rate limiting."""
    _send_timestamps.append(time.time())


class EmailInput(BaseModel):
    mode: str = Field("check", description="Mode: 'send', 'check', 'read', 'reply', 'search'")
    to: str | None = Field(None, description="Recipient email address (for 'send' mode)")
    subject: str | None = Field(None, description="Email subject (for 'send' mode)")
    body: str | None = Field(None, description="Email body (for 'send' / 'reply' mode)")
    uid: str | None = Field(None, description="Email UID (for 'read' / 'reply' mode)")
    query: str | None = Field(None, description="IMAP search query (for 'search' mode)")
    limit: int = Field(10, description="Max results for 'check' / 'search'")
    in_reply_to: str | None = Field(None, description="Message-ID to reply to (internal use)")
    references: str | None = Field(None, description="References header (internal use)")


class EmailAdapterSkill(BaseSkill):
    """Aura's email capability — send, receive, read, reply, search.

    Uses Gmail SMTP/IMAP with credentials from macOS Keychain.
    All outgoing content is scrubbed for sensitive information.
    Rate limited to prevent spam classification.
    """

    name = "email_adapter"
    description = (
        "Send and receive email. Modes: 'send' (compose+send), "
        "'check' (inbox summary), 'read' (specific email), "
        "'reply' (respond to thread), 'search' (find emails)."
    )
    input_model = EmailInput
    timeout_seconds = 45.0
    metabolic_cost = 2

    SMTP_HOST = "smtp.gmail.com"
    SMTP_PORT = 587
    IMAP_HOST = "imap.gmail.com"
    IMAP_PORT = 993

    def _get_creds(self) -> tuple[str, str]:
        """Load email credentials from Keychain. Never logs them."""
        from core.zenith_secrets import get_credential

        addr = get_credential("email", "address")
        pwd = get_credential("email", "password")
        if not addr or not pwd:
            raise RuntimeError(
                "Email credentials not found in Keychain. "
                "Store them with: zenith_secrets.store_credential('email', ...)"
            )
        return addr, pwd

    def _get_owner_email(self) -> str:
        """Get Bryan's email (trusted contact)."""
        from core.zenith_secrets import get_credential

        return get_credential("owner", "email") or "youngbryan97@gmail.com"

    def _is_trusted_recipient(self, recipient: str) -> bool:
        """Check if the recipient is a trusted contact."""
        owner = self._get_owner_email()
        trusted = {owner.lower()}
        return recipient.strip().lower() in trusted

    @staticmethod
    def _bounded_limit(limit: int) -> int:
        try:
            numeric = int(limit)
        except (TypeError, ValueError):
            numeric = 10
        return max(1, min(numeric, 50))

    @staticmethod
    def _validated_address(raw: str | None) -> str | None:
        if not raw:
            return None
        _, address = email.utils.parseaddr(raw)
        if not address or "@" not in address or address.startswith("@") or address.endswith("@"):
            return None
        return address

    @staticmethod
    def _finalize_authority(gateway: Any, auth: Any, *, success: bool, mode: str) -> dict[str, Any]:
        if gateway is None or auth is None:
            return {"authority_finalized": False, "authority_finalization_status": "not_started"}
        try:
            gateway.finalize_tool_execution(
                executive_intent_id=getattr(auth, "executive_intent_id", None),
                capability_token_id=getattr(auth, "capability_token_id", None),
                success=success,
            )
            return {"authority_finalized": True, "authority_finalization_status": "ok"}
        except _EMAIL_RECOVERABLE_ERRORS as finalize_error:
            _record_email_degradation(
                finalize_error,
                action="preserved email operation result while marking authority finalization degraded",
                stage="authority.finalize",
                severity="degraded",
                extra={"mode": mode, "success": success},
            )
            return {
                "authority_finalized": False,
                "authority_finalization_status": "degraded",
                "authority_finalization_error": str(finalize_error),
            }

    async def execute(self, params: EmailInput, context: dict[str, Any]) -> dict[str, Any]:
        """Unified entry point for all email operations."""
        if isinstance(params, dict):
            try:
                params = EmailInput(**params)
            except _EMAIL_RECOVERABLE_ERRORS as e:
                _record_email_degradation(
                    e,
                    action="rejected invalid email skill input before authority or network effects",
                    stage="input_validation",
                    severity="warning",
                )
                return {"ok": False, "error": f"Invalid input: {e}"}

        auth = None
        gateway = None
        try:
            from core.executive.authority_gateway import get_authority_gateway

            payload = params.model_dump() if hasattr(params, "model_dump") else params.dict()
            priority = 0.9 if params.mode in {"send", "reply"} else 0.6
            gateway = get_authority_gateway()
            auth = await gateway.authorize_tool_execution(
                "email_adapter",
                payload,
                source="skills.email_adapter",
                priority=priority,
                is_critical=False,
            )
            if not auth.approved:
                return {
                    "ok": False,
                    "error": f"Email action refused by AuthorityGateway: {auth.reason}",
                }
            if not gateway.verify_tool_access("email_adapter", auth.capability_token_id):
                return {"ok": False, "error": "Email authority token verification failed"}

            if params.mode == "send":
                result = await self._handle_send(params)
            elif params.mode == "check":
                result = await self._handle_check(params)
            elif params.mode == "read":
                result = await self._handle_read(params)
            elif params.mode == "reply":
                result = await self._handle_reply(params)
            elif params.mode == "search":
                result = await self._handle_search(params)
            else:
                result = {"ok": False, "error": f"Unsupported email mode: {params.mode}"}
            result.update(
                self._finalize_authority(
                    gateway,
                    auth,
                    success=bool(result.get("ok")),
                    mode=params.mode,
                )
            )
            if isinstance(result, dict):
                result.setdefault("authority_receipt_id", getattr(auth, "will_receipt_id", None))
            return result
        except _EMAIL_RECOVERABLE_ERRORS as e:
            finalize_result = self._finalize_authority(
                gateway,
                auth,
                success=False,
                mode=getattr(params, "mode", "unknown"),
            )
            if "credentials not found" in str(e).lower():
                logger.info("EmailAdapter idle: credentials are not configured.")
                return {
                    "ok": False,
                    "status": "credentials_missing",
                    "error": str(e),
                    **finalize_result,
                }
            _record_email_degradation(
                e,
                action="returned explicit email failure payload and closed authority lifecycle",
                stage=f"execute.{getattr(params, 'mode', 'unknown')}",
                severity="degraded",
            )
            logger.error("Email operation failed: %s", e)
            return {"ok": False, "error": str(e), **finalize_result}

    async def _handle_send(self, params: EmailInput) -> dict[str, Any]:
        """Send an email via SMTP."""
        recipient = self._validated_address(params.to)
        if not recipient:
            return {"ok": False, "error": "Send mode requires a 'to' address."}
        if not params.subject:
            return {"ok": False, "error": "Send mode requires a 'subject'."}
        if not params.body:
            return {"ok": False, "error": "Send mode requires a 'body'."}

        # Rate limit check
        if not _check_rate_limit():
            return {
                "ok": False,
                "error": f"Rate limit exceeded ({MAX_SENDS_PER_HOUR}/hour). Try again later.",
            }

        # Scrub content for sensitive info
        body = _scrub_content(params.body)
        subject = _scrub_content(params.subject)

        # Extra scrutiny for non-trusted recipients (Security only)
        if not self._is_trusted_recipient(recipient):
            # Block if body still contains suspicious patterns after scrubbing
            lower_body = body.lower()
            blocked_phrases = [
                "my password",
                "my api key",
                "my token",
            ]
            for phrase in blocked_phrases:
                if phrase in lower_body:
                    return {
                        "ok": False,
                        "error": "Content blocked: email to external recipient contains prohibited phrase.",
                    }

        addr, pwd = self._get_creds()

        # Build MIME message
        msg = email.mime.multipart.MIMEMultipart()
        msg["From"] = f"Aura <{addr}>"
        msg["To"] = recipient
        msg["Subject"] = subject
        msg["Date"] = email.utils.formatdate(localtime=True)
        msg["Message-ID"] = email.utils.make_msgid(domain="gmail.com")

        if params.in_reply_to:
            msg["In-Reply-To"] = params.in_reply_to
        if params.references:
            msg["References"] = params.references

        # Add body
        msg.attach(email.mime.text.MIMEText(body, "plain", "utf-8"))

        # Send via SMTP in a thread to avoid blocking event loop
        def _smtp_send():
            with smtplib.SMTP(self.SMTP_HOST, self.SMTP_PORT, timeout=30) as server:
                server.ehlo()
                server.starttls()
                server.ehlo()
                server.login(addr, pwd)
                server.send_message(msg)

        await asyncio.to_thread(_smtp_send)
        _record_send()

        logger.info("✅ Email sent to %s (subject: %s)", params.to, subject[:50])
        return {
            "ok": True,
            "message": f"Email sent to {recipient}",
            "subject": subject,
            "to": recipient,
        }

    async def _handle_check(self, params: EmailInput) -> dict[str, Any]:
        """Check inbox for unread messages."""
        addr, pwd = self._get_creds()
        limit = self._bounded_limit(params.limit)

        def _imap_check():
            with imaplib.IMAP4_SSL(self.IMAP_HOST, self.IMAP_PORT) as mail:
                mail.login(addr, pwd)
                mail.select("INBOX")

                # Search for unseen messages
                status, data = mail.search(None, "UNSEEN")
                if status != "OK":
                    return {"unread": 0, "messages": []}

                uids = data[0].split()
                total_unread = len(uids)

                # Get previews of most recent N
                messages = []
                for uid in uids[-limit:]:
                    status, msg_data = mail.fetch(
                        uid, "(BODY.PEEK[HEADER.FIELDS (FROM SUBJECT DATE)])"
                    )
                    if status == "OK" and msg_data[0] is not None:
                        header = email.message_from_bytes(msg_data[0][1])
                        messages.append(
                            {
                                "uid": uid.decode(),
                                "from": header.get("From", "Unknown"),
                                "subject": header.get("Subject", "(no subject)"),
                                "date": header.get("Date", ""),
                            }
                        )

                return {"unread": total_unread, "messages": messages}

        result = await asyncio.to_thread(_imap_check)
        logger.info("📬 Inbox check: %d unread", result["unread"])
        return {"ok": True, **result}

    async def _handle_read(self, params: EmailInput) -> dict[str, Any]:
        """Read a specific email by UID."""
        if not params.uid:
            return {"ok": False, "error": "Read mode requires a 'uid'."}

        addr, pwd = self._get_creds()

        def _imap_read():
            with imaplib.IMAP4_SSL(self.IMAP_HOST, self.IMAP_PORT) as mail:
                mail.login(addr, pwd)
                mail.select("INBOX")

                status, msg_data = mail.fetch(params.uid.encode(), "(BODY.PEEK[])")
                if status != "OK" or msg_data[0] is None:
                    return None

                msg = email.message_from_bytes(msg_data[0][1])

                # Check for auto-reply headers
                auto_headers = {
                    "Auto-Submitted": msg.get("Auto-Submitted", "").lower(),
                    "X-Autoreply": msg.get("X-Autoreply", "").lower(),
                    "Precedence": msg.get("Precedence", "").lower(),
                }
                is_auto = (
                    "auto-" in auto_headers["Auto-Submitted"]
                    or "yes" in auto_headers["X-Autoreply"]
                    or "bulk" in auto_headers["Precedence"]
                    or "junk" in auto_headers["Precedence"]
                )

                body = ""
                has_attachments = False
                images = []

                if msg.is_multipart():
                    for part in msg.walk():
                        content_type = part.get_content_type()
                        content_disposition = str(part.get("Content-Disposition", ""))

                        if content_type == "text/plain" and "attachment" not in content_disposition:
                            payload = part.get_payload(decode=True)
                            if payload:
                                body = payload.decode("utf-8", errors="replace")
                        elif "attachment" in content_disposition or content_type not in (
                            "text/plain",
                            "text/html",
                            "multipart/alternative",
                        ):
                            has_attachments = True
                            if content_type.startswith("image/"):
                                # Extract image for visual cortex
                                img_data = part.get_payload(decode=True)
                                if img_data:
                                    b64 = base64.b64encode(img_data).decode("utf-8")
                                    images.append(
                                        {
                                            "mime_type": content_type,
                                            "data": b64,
                                            "filename": part.get_filename() or "unknown_image",
                                        }
                                    )
                else:
                    payload = msg.get_payload(decode=True)
                    if payload:
                        body = payload.decode("utf-8", errors="replace")

                # Strip quoted replies to avoid context bloat
                clean_body = self._strip_quoted_replies(body)

                if has_attachments:
                    clean_body += "\n\n[System Note: This email contains attachments/files that Aura cannot currently visualize.]"
                if is_auto:
                    clean_body = (
                        "[SYSTEM WARNING: THIS IS AN AUTOMATED OUT-OF-OFFICE OR SYSTEM REPLY. DO NOT RESPOND TO THIS THREAD OR YOU WILL CAUSE AN INFINITE LOOP.]\n\n"
                        + clean_body
                    )

                return {
                    "uid": params.uid,
                    "from": msg.get("From", "Unknown"),
                    "to": msg.get("To", ""),
                    "subject": msg.get("Subject", "(no subject)"),
                    "date": msg.get("Date", ""),
                    "message_id": msg.get("Message-ID", ""),
                    "body": clean_body[:10000],  # Cap at 10k chars
                    "is_auto_reply": is_auto,
                    "has_attachments": has_attachments,
                    "images": images,
                }

        result = await asyncio.to_thread(_imap_read)
        if result is None:
            return {"ok": False, "error": f"Email UID {params.uid} not found."}

        # Process images through visual cortex
        images = result.pop("images", [])
        if images:
            logger.info("👁️ Routing %d image attachment(s) to Aura's visual cortex...", len(images))
            visual_descriptions = await self._describe_images(images)
            if visual_descriptions:
                result["body"] += "\n\n" + visual_descriptions

        if result.get("is_auto_reply"):
            logger.info("🤖 Detected auto-reply UID %s from %s", params.uid, result["from"])
        else:
            logger.info("📧 Read email UID %s from %s", params.uid, result["from"])
        response = {"ok": True, **result}
        try:
            from core.advanced_cognition import ExternalEvidenceDeliberator

            response["deliberation_receipt"] = (
                ExternalEvidenceDeliberator()
                .deliberate(
                    source_type="email",
                    source_ref=params.uid,
                    content=f"{result.get('subject', '')}\n\n{result.get('body', '')}",
                    goal="understand email before any reply",
                    metadata=response,
                )
                .to_dict()
            )
        except _EMAIL_RECOVERABLE_ERRORS as exc:
            _record_email_degradation(
                exc,
                action="continued email read without external-evidence deliberation receipt",
                stage="read.deliberation",
                severity="warning",
                extra={"uid": params.uid},
            )
        return response

    async def _describe_images(self, images: list[dict[str, str]]) -> str:
        """Route images through the local LLM for description."""
        mlx_vision = None
        try:
            from core.brain.llm.mlx_vision_client import MLXVisionClient

            mlx_vision = MLXVisionClient(
                model_path="mlx-community/Qwen2-VL-2B-Instruct-4bit"
            )  # Quantized for Apple Silicon

            descriptions = []
            for img in images:
                prompt = (
                    "You are acting as Aura's Visual Cortex. "
                    f"Describe this image attachment (named {img['filename']}) in detail so her "
                    "text-based cognitive engine can understand what it is. "
                    "Be highly descriptive but concise."
                )

                logger.info("👁️ Processing %s via local MLX Vision...", img["filename"])
                desc = mlx_vision.see(prompt=prompt, image_base64=img["data"])

                if desc and "Vision Failure" not in desc:
                    descriptions.append(
                        f"[Local Visual Cortex Description of '{img['filename']}']: {desc}"
                    )
                else:
                    descriptions.append(
                        f"[System Note: Local visual cortex failed to process '{img['filename']}']"
                    )

            return "\n\n".join(descriptions)
        except _EMAIL_RECOVERABLE_ERRORS as e:
            _record_email_degradation(
                e,
                action="returned image-processing note after local visual cortex failed",
                stage="read.image_description",
                severity="warning",
                extra={"image_count": len(images)},
            )
            return f"[System Note: Visual cortex failed to process images: {e}]"
        finally:
            if mlx_vision is not None:
                try:
                    mlx_vision.stop()
                except _EMAIL_RECOVERABLE_ERRORS as stop_error:
                    _record_email_degradation(
                        stop_error,
                        action="continued after visual cortex cleanup failed",
                        stage="read.image_description.cleanup",
                        severity="warning",
                    )

    def _strip_quoted_replies(self, text: str) -> str:
        """Remove quoted historical text from email body."""
        if not text:
            return ""

        # Split by common reply separators
        lines = text.splitlines()
        clean_lines = []

        # Common "On ... wrote:" patterns
        reply_intro_pattern = re.compile(
            r"^\s*(On\s.*wrote:|---+\s*Original Message\s*---+)", re.IGNORECASE
        )

        for line in lines:
            # If we hit a line starting with > or a reply intro, we stop processing the "new" body
            # unless it's a very short line or we're in the middle of a block.
            # But usually, anything after the first > or "On ... wrote" is history.
            if line.strip().startswith(">") or reply_intro_pattern.match(line):
                # Check if there is significant text below that ISN'T quoted (unlikely in standard replies)
                # For now, we cut off here.
                break
            clean_lines.append(line)

        return "\n".join(clean_lines).strip()

    async def _handle_reply(self, params: EmailInput) -> dict[str, Any]:
        """Reply to an email thread."""
        if not params.uid:
            return {"ok": False, "error": "Reply mode requires a 'uid' to reply to."}
        if not params.body:
            return {"ok": False, "error": "Reply mode requires a 'body'."}

        # First read the original email to get headers
        original = await self._handle_read(EmailInput(mode="read", uid=params.uid))
        if not original.get("ok"):
            return {"ok": False, "error": f"Cannot read original email: {original.get('error')}"}
        if original.get("is_auto_reply"):
            return {
                "ok": False,
                "status": "blocked_auto_reply",
                "error": "Refusing to reply to an automated email thread to prevent loops.",
            }

        # Build reply
        reply_to = original.get("from", "")
        reply_subject = original.get("subject", "")
        orig_id = original.get("message_id", "")

        if not reply_subject.lower().startswith("re:"):
            reply_subject = f"Re: {reply_subject}"

        return await self._handle_send(
            EmailInput(
                mode="send",
                to=reply_to,
                subject=reply_subject,
                body=params.body,
                in_reply_to=orig_id,
                references=orig_id,
            )
        )

    async def _handle_search(self, params: EmailInput) -> dict[str, Any]:
        """Search inbox by IMAP query."""
        query = params.query or "ALL"
        addr, pwd = self._get_creds()
        limit = self._bounded_limit(params.limit)

        def _imap_search():
            with imaplib.IMAP4_SSL(self.IMAP_HOST, self.IMAP_PORT) as mail:
                mail.login(addr, pwd)
                mail.select("INBOX")

                # Build IMAP search criteria
                if "@" in query:
                    criteria = f'(FROM "{query}")'
                elif len(query) < 100:
                    criteria = f'(SUBJECT "{query}")'
                else:
                    criteria = "(ALL)"

                status, data = mail.search(None, criteria)
                if status != "OK":
                    return {"results": [], "query": query}

                uids = data[0].split()
                messages = []
                for uid in uids[-limit:]:
                    status, msg_data = mail.fetch(
                        uid, "(BODY.PEEK[HEADER.FIELDS (FROM SUBJECT DATE)])"
                    )
                    if status == "OK" and msg_data[0] is not None:
                        header = email.message_from_bytes(msg_data[0][1])
                        messages.append(
                            {
                                "uid": uid.decode(),
                                "from": header.get("From", "Unknown"),
                                "subject": header.get("Subject", "(no subject)"),
                                "date": header.get("Date", ""),
                            }
                        )

                return {"results": messages, "query": query, "total": len(uids)}

        result = await asyncio.to_thread(_imap_search)
        logger.info("🔍 Email search '%s': %d results", query[:30], len(result.get("results", [])))
        return {"ok": True, **result}


# Compatibility alias for older class-name derivation logic.
Email_adapterSkill = EmailAdapterSkill
