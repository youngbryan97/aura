"""Whether a message was written by a person to her, or sent by a machine, read from what its headers say of it.

Mail that a program sent says so in its headers, by standards its senders follow: Auto-Submitted (RFC 3834) for
replies and notices a program generates; List-Id and List-Unsubscribe (RFC 2919, RFC 2369) for mailing lists and
newsletters; Precedence bulk, list or junk; Feedback-ID, which bulk senders set for the large mail providers; an empty
Return-Path, which bounces and notices carry; and a sender address a person does not write from (no-reply, mailer-daemon,
notifications). Her triage read words in the body ("unsubscribe", "receipt") and wrote a warning to her model into the
body of a reply it had found automated; neither is needed when the headers are read.
"""
from __future__ import annotations

import re
from collections.abc import Mapping
from email.utils import getaddresses, parseaddr

__all__ = ["HEADERS_READ", "addressed_to", "who_sent_it"]

#: The headers the reading needs, for a fetch of headers alone.
HEADERS_READ = ("FROM", "TO", "CC", "SUBJECT", "DATE", "MESSAGE-ID", "AUTO-SUBMITTED", "PRECEDENCE", "LIST-ID",
                "LIST-UNSUBSCRIBE", "FEEDBACK-ID", "RETURN-PATH", "X-AUTO-RESPONSE-SUPPRESS", "X-AUTOREPLY")

_MACHINE_SENDER = re.compile(
    r"^(?:no-?reply|do-?not-?reply|donotreply|mailer-daemon|postmaster|notifications?|alerts?|bounces?|news(?:letter)?)"
    r"(?:[+._-].*)?@", re.IGNORECASE)


def who_sent_it(headers: Mapping[str, str]) -> str:
    """'' when a person sent it, else what in its headers says a program did."""
    head = {str(k).lower(): str(v or "").strip() for k, v in headers.items()}
    if head.get("auto-submitted", "no").lower() not in ("", "no"):
        return f"Auto-Submitted: {head['auto-submitted']}"
    for name in ("list-id", "list-unsubscribe", "feedback-id"):
        if head.get(name):
            return f"{name.title()} present"
    if head.get("precedence", "").lower() in ("bulk", "list", "junk"):
        return f"Precedence: {head['precedence']}"
    if head.get("x-autoreply", "").lower() == "yes" or head.get("x-auto-response-suppress"):
        return "an auto-reply's own header"
    if "return-path" in head and head["return-path"] in ("<>", ""):
        return "an empty Return-Path"
    sender = parseaddr(head.get("from", ""))[1]
    if _MACHINE_SENDER.search(sender):
        return f"sent from {sender}"
    return ""


def addressed_to(headers: Mapping[str, str], her_address: str) -> str:
    """How the message reached her: 'to' when her address is among its To, 'cc' among its Cc, '' otherwise (a list, a
    blind copy, a forward)."""
    head = {str(k).lower(): str(v or "") for k, v in headers.items()}
    mine = her_address.strip().lower()
    for field in ("to", "cc"):
        if any(address.lower() == mine for _name, address in getaddresses([head.get(field, "")])):
            return field
    return ""
