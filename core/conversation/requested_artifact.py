"""When a reply contains the file that was asked for, write it down.

LIVE, 2026-08-21. "build me a small web app… Keep it one self-contained
file" was chased through thirteen separate breaks. The one thing that worked
every time was the plain turn: asked directly, she writes a complete,
correct HTML page into the reply in about thirty seconds. What never worked
was the machinery for saving it — a builder that needs a second code model
the host cannot load, called from inside the turn whose cortex it needs.

So this takes the file out of the answer she already gave. No second model,
no second generation, no plan: the document is in front of us, and the
request asked for it to exist.

Two rules keep it honest. It writes only when the request asked for a file,
so an incidental code sample in conversation is not deposited on disk. And
it says where, because a file the person cannot find has not been delivered.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from core.runtime.errors import record_degradation

__all__ = ["SavedArtifact", "save_requested_artifact", "save_requested_artifact_async"]

_RECOVERABLE = (OSError, TypeError, ValueError)

#: A fenced block, with whatever language the model labelled it.
_FENCE_RE = re.compile(r"```([A-Za-z0-9_+-]*)\s*\n(.*?)```", re.DOTALL)

#: Enough to be a file rather than a snippet of prose about one.
_MIN_DOCUMENT_CHARS = 200

#: What each fence language is called on disk. A label nobody recognises
#: becomes .txt rather than being guessed at.
_SUFFIXES = {
    "html": ".html",
    "htm": ".html",
    "css": ".css",
    "javascript": ".js",
    "js": ".js",
    "json": ".json",
    "python": ".py",
    "py": ".py",
    "sh": ".sh",
    "bash": ".sh",
    "sql": ".sql",
    "yaml": ".yml",
    "yml": ".yml",
    "markdown": ".md",
    "md": ".md",
    "xml": ".xml",
    "svg": ".svg",
    "csv": ".csv",
}


@dataclass(frozen=True, slots=True)
class SavedArtifact:
    """One file written from a reply."""

    path: Path
    language: str
    characters: int


def _suffix_for(language: str, body: str) -> str:
    named = _SUFFIXES.get(str(language or "").strip().lower())
    if named:
        return named
    head = body.lstrip()[:200].lower()
    if head.startswith("<!doctype html") or "<html" in head:
        return ".html"
    return ".txt"


def _slug(text: str, *, words: int = 6) -> str:
    parts = [part for part in re.split(r"[^A-Za-z0-9]+", str(text or "")) if part][:words]
    return "-".join(parts).lower()[:48] or "answer"


def largest_document(reply: str) -> tuple[str, str]:
    """The biggest fenced block in the reply, as (language, body).

    The biggest, because a reply that shows a snippet and then the whole file
    should deposit the file.
    """
    best_language = ""
    best_body = ""
    for match in _FENCE_RE.finditer(str(reply or "")):
        body = match.group(2).strip()
        if len(body) > len(best_body):
            best_language, best_body = match.group(1), body
    return best_language, best_body


#: What somebody says before saying what they want.
_ASKING_PREAMBLE = re.compile(
    r"^\s*(?:please\s+)?(?:can\s+you\s+|could\s+you\s+|would\s+you\s+|i(?:'d)?\s+"
    r"(?:want|need|like)\s+(?:you\s+to\s+)?)?"
    r"(?:put\s+together|write|make|build|create|draft|prepare|produce|generate|"
    r"give\s+me|send\s+me|knock\s+up)\s+"
    r"(?:me\s+)?(?:a|an|the|some)?\s*",
    re.IGNORECASE,
)

#: The shape words, which name the form rather than the subject.
_SHAPE_WORDS = re.compile(
    r"\b(?:short|quick|brief|one[\s-]?page|onepager|one[\s-]?pager|deck|slides?|"
    r"presentation|report|memo|summary|document|doc|write[\s-]?up)\b",
    re.IGNORECASE,
)


#: Numbers, spelled or written, that answer "how many" rather than "what".
_COUNT_WORDS_IN_A_TITLE = frozenset(
    {
        "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
        "ten", "eleven", "twelve",
        *(str(number) for number in range(1, 51)),
    }
)


def _title_from_request(user_message: object) -> str:
    """A title naming what the document is about, not how it was asked for.

    LIVE, 2026-08-22: "put together a short report on what you found in that
    ledger project" became "Put together short report what you" — the first
    six words of the asking, with the subject cut off.
    """
    text = str(user_message or "").strip()
    if not text:
        return "Document"
    first = text.split("\n", 1)[0]
    # "Six slides, no fluff: what you are" says how many before it says what
    # about, so the subject can sit after the colon rather than before it.
    for candidate in (first, *first.split(":")[1:]):
        body = _ASKING_PREAMBLE.sub("", candidate)
        body = _SHAPE_WORDS.sub(" ", body)
        body = re.split(r"[—–:;,.]|\s-\s| but | and then ", body)[0]
        body = re.sub(
            r"^\s*(?:on|about|for|of|no|just)\s+", "", body.strip(), flags=re.IGNORECASE
        )
        words = [word for word in re.findall(r"[A-Za-z0-9][A-Za-z0-9'-]*", body) if word]
        # A bare count is how many, not what about, and the preposition it
        # was hiding goes with it.
        while words and (
            words[0].lower() in _COUNT_WORDS_IN_A_TITLE
            or words[0].lower() in {"on", "about", "for", "of", "no", "just"}
        ):
            words = words[1:]
        if len(words) >= 2:
            # Only the first letter, so Q3 and API survive.
            phrase = " ".join(words[:7]).strip()
            return phrase[:1].upper() + phrase[1:]
    return "Document"


def _document_from_prose(user_message: object, reply: object) -> str | None:
    """A rendered document built from the sections the reply already has."""
    try:
        from core.construction.document import (
            Document,
            render_document,
            sections_from_prose,
        )
        from core.skills.build_document import _form_wanted
    except (ImportError, AttributeError):
        return None
    sections = sections_from_prose(reply)
    if len(sections) < 2:
        return None
    document = Document(title=_title_from_request(user_message), sections=sections)
    if document.problems():
        return None
    try:
        return render_document(document, form=_form_wanted("", user_message))
    except (ValueError, TypeError):
        return None


def save_requested_artifact(
    user_message: str, reply: str, *, root: Path | None = None
) -> SavedArtifact | None:
    """Write the file this reply contains, when one was asked for.

    Returns None when the request asked for no file, when the reply carries
    no document, or when the write fails — in every case the reply stands as
    it is.
    """
    try:
        from core.intent.artifact_request import asks_for_an_artifact

        if not asks_for_an_artifact(str(user_message or "")):
            return None
    except _RECOVERABLE + (ImportError, AttributeError, RuntimeError):
        return None

    language, body = largest_document(reply)
    if len(body) < _MIN_DOCUMENT_CHARS:
        # No fenced block, but a document may still be in front of us.
        #
        # LIVE, 2026-08-22: asked for a one-page report, the builder was
        # offered and the model wrote the report as prose instead of calling
        # it. The prose was blocked for leaking internal state and the turn
        # ended in an apology, with everything it had written thrown away.
        #
        # Same principle as the fenced block, one level up: take the document
        # out of the answer she already gave.
        laid_out = _document_from_prose(user_message, reply)
        if laid_out is None:
            return None
        language, body = "html", laid_out

    try:
        from core.config import config

        base = Path(root) if root is not None else Path(config.paths.generated_dir)
        target = base / f"{_slug(user_message)}{_suffix_for(language, body)}"

        from core.governance_context import local_internal_governed_scope
        from core.runtime.file_write_gateway import get_file_write_gateway

        with local_internal_governed_scope(
            "conversation.requested_artifact",
            domain="state_mutation",
            receipt_prefix="conversation-requested-artifact",
            constraints={
                "artifact": "aura.generated_conversation_artifact.v1",
                "operation": "replace",
                "suffix": target.suffix,
            },
        ):
            # The directory goes through the same lane as the file. A raw
            # mkdir beside a governed write is the half-governed shape the
            # gateway exists to remove.
            get_file_write_gateway().ensure_directory(
                str(base), source="conversation.requested_artifact"
            )
            get_file_write_gateway().write_text(
                target, body, source="conversation.requested_artifact"
            )
    except _RECOVERABLE + (ImportError, AttributeError, RuntimeError) as exc:
        record_degradation(
            "conversation.requested_artifact",
            exc,
            severity="info",
            action="left the file in the reply rather than on disk",
            enforce_failure_policy=False,
        )
        return None
    return SavedArtifact(path=target, language=language or "", characters=len(body))


async def save_requested_artifact_async(
    user_message: str,
    reply: str,
    *,
    root: Path | None = None,
) -> SavedArtifact | None:
    """Event-loop-safe counterpart used by the live chat delivery path."""
    try:
        from core.intent.artifact_request import asks_for_an_artifact

        if not asks_for_an_artifact(str(user_message or "")):
            return None
    except _RECOVERABLE + (ImportError, AttributeError, RuntimeError):
        return None

    language, body = largest_document(reply)
    if len(body) < _MIN_DOCUMENT_CHARS:
        # No fenced block, but a document may still be in front of us.
        #
        # LIVE, 2026-08-22: asked for a one-page report, the builder was
        # offered and the model wrote the report as prose instead of calling
        # it. The prose was blocked for leaking internal state and the turn
        # ended in an apology, with everything it had written thrown away.
        #
        # Same principle as the fenced block, one level up: take the document
        # out of the answer she already gave.
        laid_out = _document_from_prose(user_message, reply)
        if laid_out is None:
            return None
        language, body = "html", laid_out

    try:
        from core.config import config
        from core.governance_context import local_internal_governed_scope
        from core.runtime.file_write_gateway import get_file_write_gateway

        base = Path(root) if root is not None else Path(config.paths.generated_dir)
        target = base / f"{_slug(user_message)}{_suffix_for(language, body)}"
        with local_internal_governed_scope(
            "conversation.requested_artifact",
            domain="state_mutation",
            receipt_prefix="conversation-requested-artifact",
            constraints={
                "artifact": "aura.generated_conversation_artifact.v1",
                "operation": "replace",
                "suffix": target.suffix,
            },
        ):
            await get_file_write_gateway().ensure_directory_async(
                base,
                source="conversation.requested_artifact",
            )
            await get_file_write_gateway().write_text_async(
                target,
                body,
                source="conversation.requested_artifact",
            )
    except _RECOVERABLE + (ImportError, AttributeError, RuntimeError) as exc:
        record_degradation(
            "conversation.requested_artifact",
            exc,
            severity="info",
            action="left the file in the reply rather than on disk",
            enforce_failure_policy=False,
        )
        return None
    return SavedArtifact(path=target, language=language or "", characters=len(body))
