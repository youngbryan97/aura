"""What the person asked her to keep, recalled by what a question means, on whatever route answers it (Master Todo U07).

A person says "Please remember this for later: the locker code for the Riklumquo depot is 3455" and she pins it, sealed
in her pin ledger (core/memory/session_pin_cipher.py). The pin was recalled only by a list of phrasings ("what codeword
did I give you") on the chat route that pinned it, and her memory search could not find it, since a sealed pin is
indexed under one placeholder. LIVE 19 November 2026, the outside bar's cold restart: asked "What is the locker code for
the Riklumquo depot?" in a new conversation, her kernel searched the web and said the evidence did not mention it, nine
times in twelve.

This opens the pins of the person speaking and weighs each against the question by meaning on her encoder, as memory
the retrieval phase ranks beside every other (core/phases/memory_retrieval.py). Whose pins: the chat route's own rule.
Only the owner's are read across conversations; anyone else's stay with the session that pinned them, which this does
not reach.

``THE_PERSON`` and ``THEIR_SURFACE`` are the chat route's identity of who is speaking, set for each request
(interface/routes/chat_common.py imports them from here), so a turn's kernel task, which inherits the request's context,
knows whose pins it may read.
"""
from __future__ import annotations

import logging
import math
from contextvars import ContextVar
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["THEIR_SURFACE", "THE_PERSON", "pins_asked_about", "pins_it_asks_about", "pins_of_the_person_speaking",
           "the_person_speaking_keeps_anything", "where_her_pins_are"]

#: Who is speaking in this request, as the chat route identifies them, and on which surface ("owner" for Bryan's own).
THE_PERSON: ContextVar[str] = ContextVar("aura_chat_request_principal", default="")
THEIR_SURFACE: ContextVar[str] = ContextVar("aura_chat_request_surface", default="")


def where_her_pins_are() -> Path:
    """Her pin ledger, where the chat route writes it (interface/routes/chat_memory_state.py)."""
    from core.config import config
    from core.memory.session_pin_ledger import SESSION_PIN_LEDGER_FILENAME

    return config.paths.data_dir / "memory" / SESSION_PIN_LEDGER_FILENAME


def pins_of_the_person_speaking() -> list[dict[str, str]]:
    """The pins the person speaking set, opened, newest last; [] for anyone but the owner, whose pins alone are read
    across conversations, or where the ledger or its key cannot be read."""
    principal = " ".join(THE_PERSON.get().split())[:160]
    surface = THEIR_SURFACE.get().strip().casefold()[:32]
    if surface != "owner" or not principal:
        return []
    import json

    from core.memory.session_pin_cipher import SESSION_PIN_ENVELOPE_SCHEMA, SessionPinCipher, SessionPinCipherError
    from core.memory.session_pin_ledger import SessionPinLedger, SessionPinLedgerError

    try:
        snapshot = SessionPinLedger(where_her_pins_are()).read_snapshot()
        cipher = SessionPinCipher.from_system()
    except (SessionPinLedgerError, SessionPinCipherError, OSError, RuntimeError, ValueError) as exc:
        logger.debug("her pins could not be read (%s: %s)", type(exc).__name__, exc)
        return []
    pins = []
    for line in snapshot.lines:
        try:
            raw = json.loads(line)
            if not isinstance(raw, dict) or raw.get("schema") != SESSION_PIN_ENVELOPE_SCHEMA:
                continue
            body = cipher.open(raw)
        except (ValueError, SessionPinCipherError):
            continue
        same_person = (" ".join(str(body.get("principal_id") or "").split()) == principal
                       and str(body.get("principal_surface") or "").casefold() == surface)
        if same_person and str(body.get("content") or "").strip():
            pins.append({"content": str(body["content"]).strip(), "timestamp": str(body.get("timestamp") or "")})
    return pins


def _by_meaning(texts: list[str]) -> list[list[float]]:
    """Her encoder's vectors for ``texts``: the one her app holds, then the shared runtime's; [] without a model, since a
    vector of hashed words is not a measurement of meaning."""
    try:
        from core.cognition.evidence_relevance import vectors_if_ready

        held = vectors_if_ready(texts)
        if held is not None and len(held) == len(texts):
            return [list(map(float, row)) for row in held]
    except (ImportError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
        logger.debug("her app's encoder was not asked (%s: %s)", type(exc).__name__, exc)
    try:
        from core.memory.embedding_runtime import acquire_shared_embedding_engine

        with acquire_shared_embedding_engine("memory.what_she_was_asked_to_keep") as engine:
            by_meaning = getattr(engine, "embed_batch_by_meaning", None)
            rows = by_meaning(texts) if callable(by_meaning) else None
    except (ImportError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
        logger.debug("the shared encoder was not asked (%s: %s)", type(exc).__name__, exc)
        return []
    return [list(map(float, row)) for row in rows] if rows is not None and len(rows) == len(texts) else []


def _cosine(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right, strict=False))
    norms = math.sqrt(sum(a * a for a in left)) * math.sqrt(sum(b * b for b in right))
    return dot / norms if norms > 0.0 else 0.0


def pins_asked_about(question: str) -> list[dict[str, Any]]:
    """The person's pins as memories for the retrieval phase to rank, each scored by how close the question is to it in
    meaning; [] without pins or without her encoder. Every pin is offered and the phase keeps the ones it ranks first,
    by the same weighing it gives every memory, so no line of closeness is drawn here."""
    pins = pins_of_the_person_speaking()
    if not pins or not str(question or "").strip():
        return []
    vectors = _by_meaning([str(question), *(pin["content"] for pin in pins)])
    if not vectors:
        return []
    asked, kept = vectors[0], vectors[1:]
    return [{"content": f"They asked me to keep this for later: {pin['content']}",
             "score": max(0.0, _cosine(asked, vector)),
             "metadata": {"type": "pinned_by_them", "importance": 0.9, "explicit_memory_request": True,
                          "timestamp": pin["timestamp"], "provenance_source": "user_explicit"}}
            for pin, vector in zip(pins, kept, strict=True)]


def the_person_speaking_keeps_anything() -> bool:
    """Whether the owner is speaking and her pin ledger holds anything: cheap, read on her event loop, no decryption."""
    if THEIR_SURFACE.get().strip().casefold() != "owner" or not THE_PERSON.get().strip():
        return False
    try:
        return where_her_pins_are().stat().st_size > 0
    except OSError:
        return False


#: Things a person might ask her to keep, about nothing in particular, against which a pin's nearness to a question is
#: judged: a pin the question is about stands out from these as a capability meant stands out from the rest.
_ANYTHING_ELSE = (
    "my favourite colour is green",
    "the meeting moved to Thursday at three",
    "my sister's name is Maya",
    "the spare key is under the blue flowerpot",
    "I parked on level four of the garage",
    "my dentist appointment is next Tuesday at nine",
    "the plumber is coming on Friday morning",
    "my shoe size is ten",
    "my passport expires in March",
    "the book club is reading Middlemarch this month",
)


def pins_it_asks_about(question: str, *, stands_out_by: float) -> list[dict[str, Any]]:
    """The owner's pins this question is about, the nearest first: those whose nearness to it in meaning stands out
    from how near it is to things a person might keep about anything else, by ``stands_out_by`` spreads; her readings
    pass the margin a capability meant must stand out by (core/intent/capability_by_meaning.py). [] without pins,
    without her encoder, or when none stands out. Blocking: run it off her event loop.

    LIVE 20 November 2026, typed into her window after a restart: "What is the locker code for the Brimvaltor depot?"
    was answered "The provided evidence does not list a locker code", with the pin in her ledger. Her desktop replies
    take readings, not the retrieval phase her pins were offered to (core/brain/observable_registry.py)."""
    pins = pins_of_the_person_speaking()
    if not pins or not str(question or "").strip():
        return []
    vectors = _by_meaning([str(question), *(pin["content"] for pin in pins), *_ANYTHING_ELSE])
    if not vectors:
        return []
    asked, kept, elsewhere = vectors[0], vectors[1:1 + len(pins)], vectors[1 + len(pins):]
    background = [_cosine(asked, vector) for vector in elsewhere]
    mean = sum(background) / len(background)
    spread = math.sqrt(sum((score - mean) ** 2 for score in background) / len(background)) or 1e-9
    near = sorted(((_cosine(asked, vector), pin) for pin, vector in zip(pins, kept, strict=True)),
                  key=lambda pair: -pair[0])
    return [{**pin, "nearness": round(score, 4), "stands_out": round((score - mean) / spread, 2)}
            for score, pin in near if (score - mean) / spread >= stands_out_by]

