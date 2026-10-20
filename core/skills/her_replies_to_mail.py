"""Her replies to mail a person sent her: written by her own turn, and held for Bryan until he lets her send them.

Her triage of her inbox matched words ("urgent", "can you", "unsubscribe") and answered what it kept with one of two
sentences written in her code ("Thanks for the note about ...; I will follow up"), never her own. Here who sent a
message is read from its headers (core/skills/what_mail_is.py); a person's message to her goes to her model as what
was said to her, the sender and subject with it as written, and what she answers is the reply, or nothing when she has
nothing to say. A reply goes out under her account, to someone other than Bryan: it is held in
``data/email/drafts.jsonl`` and sent by ``tools/her_email_drafts.py send ID``, by him, until he sets
AURA_EMAIL_SENDS_WITHOUT_ASKING.
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any

logger = logging.getLogger("Skills.Email")

__all__ = ["DRAFTS", "drafts", "her_reply_to", "hold_for_bryan", "sends_without_asking"]

#: The origin her reply turns carry: a turn on her desktop lane, answering what a person wrote her.
ORIGIN = "desktop_email_reply"
DRAFTS = Path("email") / "drafts.jsonl"


def _drafts_path() -> Path:
    from core.config import config

    return Path(config.paths.data_dir) / DRAFTS


def sends_without_asking() -> bool:
    """Whether Bryan has let her send her replies without his reading them first."""
    import os

    return os.environ.get("AURA_EMAIL_SENDS_WITHOUT_ASKING", "") == "1"


def as_written(message: dict[str, Any]) -> str:
    """The message as it reached her: who sent it, its subject, and what it says."""
    return (f"From: {message.get('from') or 'someone'}\nSubject: {message.get('subject') or '(no subject)'}\n\n"
            f"{message.get('body') or ''}").strip()


async def her_reply_to(message: dict[str, Any], *, seconds: float = 180.0) -> str:
    """Her answer to a person's message, by her own turn on her model; '' when she has none."""
    from core.brain.types import ThinkingMode
    from core.container import ServiceContainer

    engine = ServiceContainer.get("cognitive_engine", default=None)
    if engine is None:
        return ""
    said = as_written(message)
    context = {"desktop_quick_reply_contract": True, "origin": ORIGIN, "visible_user_message": said, "allow_tools": False}
    thought = await engine.think(said, context=context, mode=ThinkingMode.FAST, origin=ORIGIN, timeout_s=seconds)
    return str(getattr(thought, "content", "") or "").strip()


async def hold_for_bryan(message: dict[str, Any], reply: str) -> str:
    """Her reply kept until Bryan sends it; the draft's id."""
    from core.governance_context import local_internal_governed_scope
    from core.runtime.file_write_gateway import get_file_write_gateway

    draft = {"id": uuid.uuid4().hex[:10], "written": time.time(), "uid": str(message.get("uid") or ""),
             "to": str(message.get("from") or ""), "subject": str(message.get("subject") or ""),
             "message_id": str(message.get("message_id") or ""), "reply": reply, "sent": False}
    path = _drafts_path()
    with local_internal_governed_scope("her_replies_to_mail.hold", domain="file_write"):
        gateway = get_file_write_gateway()
        await gateway.ensure_directory_async(path.parent, source="her_replies_to_mail.hold")
        await gateway.append_text_async(path, json.dumps(draft) + "\n", source="her_replies_to_mail.hold")
    logger.info("📧 Her reply to %s about %r is held for Bryan (%s).", draft["to"], draft["subject"], draft["id"])
    return str(draft["id"])


def drafts(path: Path | None = None) -> list[dict[str, Any]]:
    """Every reply she has written, the latest state of each."""
    path = path or _drafts_path()
    latest: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines() if path.exists() else []:
        try:
            row = json.loads(line)
        except ValueError:
            continue
        latest[str(row.get("id"))] = {**latest.get(str(row.get("id")), {}), **row}
    return list(latest.values())
