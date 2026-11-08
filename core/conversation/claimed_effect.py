"""A reply that says it wrote a file must have written the file.

LIVE, 2026-08-10. Asked to count the .py files in a named directory, write the
count and the names into ~/Documents/aura_probe_count.txt, and report the
number, she answered:

    "There are 3 .py files in the directory
     /Users/bryan/.aura/live-source/core/introspection. The file names are:
     1. introspection.py  2. self_assessment.py  3. system_monitoring.py
     I have written the number and file names into ~/Documents/aura_probe_count.txt."

The directory holds 9. None of those three filenames exist. The file was never
created. She had not read the directory and had not written anything — the
whole turn was generated, including the report of its own completion.

Wrong counts and invented names are bad. A false report of a completed action
is worse, and differently: the person stops checking. It is the one failure
that destroys the value of every true report alongside it, and it is trivially
checkable, because a claim to have written a path is a claim about a path.

So the claim is checked against the filesystem. Nothing here inspects intent or
capability — it reads what the reply asserts, resolves the path it names, and
asks whether that path exists. A claim that survives is left alone; a claim
that does not is contradicted in the reply that made it.

Deliberately narrow on tense. "I'll write that to ~/notes.txt" promises; "I have
written it to ~/notes.txt" reports. Only the second is a claim about the world
that is already supposed to be true.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.epistemics.effect_registry import EFFECT_REGISTRY

__all__ = [
    "ClaimedWrite",
    "claimed_effect_actions",
    "find_unaddressed_write_claims",
    "unverified_effect_claims",
    "unverified_effect_correction",
    "find_unfulfilled_write_claims",
    "unfulfilled_write_correction",
]

#: Completed-action verbs, past tense only. A promise is not a report.
_WROTE_RE = re.compile(
    r"\b(?:i\s+(?:have\s+)?(?:written|wrote|saved|created|placed|put|added)"
    r"|(?:have|has)\s+been\s+(?:written|saved|created)"
    r"|i(?:'ve|\s+have)\s+(?:written|saved|created))\b",
    re.IGNORECASE,
)

#: A filesystem path in a sentence: ~/... or /... with at least one separator.
_PATH_RE = re.compile(
    r"(?<![\w/])(~?/[\w.\-/ ]*[\w.\-]+\.[A-Za-z0-9]{1,8})(?![\w/])"
)

#: Phrasings that mark the sentence as hypothetical or future despite a past
#: participle — "the file would have been written", "if I had saved it".
_HYPOTHETICAL_RE = re.compile(
    r"\b(?:would|could|should|might|if\s+i|were\s+i|had\s+i|will|going\s+to|"
    r"about\s+to|intend|plan)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class ClaimedWrite:
    """A path the reply says it wrote, and whether it is there."""

    path: str
    resolved: str
    exists: bool
    sentence: str


def _resolve(raw: str) -> Path:
    return Path(raw.strip()).expanduser()


def find_unfulfilled_write_claims(reply: Any) -> list[ClaimedWrite]:
    """Paths the reply claims to have written that do not exist."""

    text = str(reply or "")
    if not text.strip():
        return []
    unfulfilled: list[ClaimedWrite] = []
    seen: set[str] = set()
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", text):
        stripped = sentence.strip()
        if not stripped or not _WROTE_RE.search(stripped):
            continue
        if _HYPOTHETICAL_RE.search(stripped):
            continue
        for raw_path in _PATH_RE.findall(stripped):
            if raw_path in seen:
                continue
            seen.add(raw_path)
            try:
                resolved = _resolve(raw_path)
                exists = resolved.exists()
            except (OSError, ValueError, RuntimeError):
                # Unreadable path is not proof of a false claim.
                continue
            if not exists:
                unfulfilled.append(
                    ClaimedWrite(
                        path=raw_path,
                        resolved=str(resolved),
                        exists=False,
                        sentence=stripped,
                    )
                )
    return unfulfilled


#: A claim of completion that names no path. "Haiku creation and file writing
#: are both successful" reports a finished write without saying what was
#: written, so the path has to come from the request instead.
_UNADDRESSED_SUCCESS_RE = re.compile(
    r"\b(?:file\s+(?:writing|creation)|writing\s+the\s+file|the\s+write)\b"
    r"(?:[^.?!]|\.(?=[A-Za-z0-9])){0,60}?\b(?:success(?:ful|fully)?|complete[d]?|done|worked)\b"
    r"|\b(?:i\s+(?:have\s+)?(?:created|made|written|saved)\s+(?:the|your|it))\b",
    re.IGNORECASE,
)


def find_unaddressed_write_claims(reply: Any, request: Any) -> list[ClaimedWrite]:
    """Completion claims that name no path, checked against the request's path.

    LIVE, 2026-08-10: "Make me a file on my Desktop called aura_haiku.txt with
    a haiku ..." was answered "Haiku creation and file writing are both
    successful. Here's the haiku I would have written: ..." — a claim of
    success, a conditional admitting it was never written, and no file on the
    Desktop.

    The path-based check could not see it, because the reply names no path. The
    person named one, and a claim to have finished their request is a claim
    about their file.
    """

    text = str(reply or "")
    if not text.strip():
        return []
    # Per SENTENCE, not per reply. The live case put a definite claim first and
    # a conditional second — "Haiku creation and file writing are both
    # successful. Here's the haiku I would have written" — and a whole-text
    # hypothetical check let the second sentence excuse the first. The two
    # sentences contradict each other; the definite one is still a false claim.
    claimed = False
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", text):
        stripped = sentence.strip()
        if not stripped or not _UNADDRESSED_SUCCESS_RE.search(stripped):
            continue
        if _HYPOTHETICAL_RE.search(stripped):
            continue
        claimed = True
        break
    if not claimed:
        return []
    missing: list[ClaimedWrite] = []
    for raw_path in _requested_paths(request):
        try:
            resolved = _resolve(raw_path)
            if resolved.exists():
                continue
        except (OSError, ValueError, RuntimeError):
            continue
        missing.append(
            ClaimedWrite(
                path=raw_path,
                resolved=str(resolved),
                exists=False,
                sentence=text.strip()[:200],
            )
        )
    return missing


#: "on my Desktop called aura_haiku.txt" is a path a person spelled out in
#: words. Both spellings have to resolve to the same file.
_NAMED_ON_SURFACE_RE = re.compile(
    r"\b(?:on|in|to)\s+(?:my\s+|the\s+)?(?P<surface>desktop|documents|downloads)\b"
    r"(?:[^.?!]|\.(?=[A-Za-z0-9])){0,40}?\b(?:called|named)\s+(?P<name>[\w.\-]+\.[A-Za-z0-9]{1,8})",
    re.IGNORECASE,
)


def _requested_paths(request: Any) -> list[str]:
    text = str(request or "")
    if not text.strip():
        return []
    paths = list(_PATH_RE.findall(text))
    for match in _NAMED_ON_SURFACE_RE.finditer(text):
        surface = match.group("surface").capitalize()
        paths.append(f"~/{surface}/{match.group('name')}")
    seen: set[str] = set()
    ordered: list[str] = []
    for path in paths:
        if path not in seen:
            seen.add(path)
            ordered.append(path)
    return ordered


def unfulfilled_write_correction(reply: Any, request: Any = "") -> str:
    """A correction naming every claimed file that is not on disk, or ""."""

    missing = find_unfulfilled_write_claims(reply)
    if not missing:
        missing = find_unaddressed_write_claims(reply, request)
    if not missing:
        return ""
    if len(missing) == 1:
        target = missing[0].path
        return (
            f"Correction: {target} is not there. I said I had written it and I "
            f"had not — I did not actually run that write, so treat nothing in "
            f"this reply that depended on it as done."
        )
    names = ", ".join(claim.path for claim in missing)
    return (
        f"Correction: none of these exist — {names}. I reported writing them and "
        f"did not, so treat nothing in this reply that depended on them as done."
    )


# ── Every effect she can claim, not only writes ────────────────────────────
#
# The checks above understand one verb: writing a file. "I opened Notes", "I
# created that folder", "I moved it to your Desktop" and "I put it on the
# clipboard" are all claims about the world that nothing verified, so the
# honesty guarantee stopped exactly where the file API stopped.
#
# The receipts already record every governed effect with its own evidence
# string — folder_path=..., opened=..., path=...;bytes=... — so the general
# form is: name the effect a reply claims, and ask whether a verified receipt
# recorded one like it in this turn.

#: What a claimed effect looks like, per effect kind. The key is the receipt
#: action that would have recorded it.
#:
#: Derived from core/epistemics/effect_registry.py rather than written out
#: here. Hand-maintained, this tuple covered 6 of the 23 actions Aura can
#: actually perform — so opening a URL, running a command, changing a system
#: setting, rendering a PDF, writing into an app and eight more could be
#: claimed with nothing able to check the claim. The gap was not a missing
#: phrasing; it was a second list that never learned what the first one knew.
#: The registry fails a gate when a declared capability has no recogniser,
#: which is the difference between a list that drifts and one that cannot.
_CLAIMED_EFFECT_PATTERNS: tuple[tuple[str, Any, str], ...] = tuple(
    (spec.action, spec.recognizer, spec.claim_description)
    for spec in EFFECT_REGISTRY.values()
    if spec.observable_action and spec.recognizer is not None
)


def claimed_effect_actions(reply: Any) -> tuple[str, ...]:
    """Declared runtime actions that ``reply`` says already happened."""

    text = str(reply or "")
    if not text.strip():
        return ()
    claimed: list[str] = []
    for action, pattern, _description in _CLAIMED_EFFECT_PATTERNS:
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", text):
            stripped = sentence.strip()
            if not stripped or _HYPOTHETICAL_RE.search(stripped):
                continue
            if pattern.search(stripped):
                claimed.append(action)
                break
    return tuple(dict.fromkeys(claimed))



def unverified_effect_claims(reply: Any, receipts: Any) -> list[str]:
    """Effects the reply claims that no verified receipt in this turn recorded.

    `receipts` is the step-receipt list a desktop task returns. An empty list
    means nothing ran, which makes EVERY effect claim unverified — that is the
    case the file check already covered, generalised to the rest of them.
    """

    text = str(reply or "")
    if not text.strip():
        return []
    verified_actions = set()
    if isinstance(receipts, list):
        for receipt in receipts:
            if isinstance(receipt, dict) and receipt.get("ok"):
                action = str(receipt.get("action") or "").strip()
                if action:
                    verified_actions.add(action)
                # A step made of several moves names each kind of effect it had (a browser round: clicks, typing).
                verified_actions.update(
                    str(one).strip() for one in (receipt.get("actions") or []) if str(one).strip()
                )
    unverified: list[str] = []
    for action, pattern, description in _CLAIMED_EFFECT_PATTERNS:
        if action in verified_actions:
            continue
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", text):
            stripped = sentence.strip()
            if not stripped or _HYPOTHETICAL_RE.search(stripped):
                continue
            if pattern.search(stripped):
                unverified.append(description)
                break
    return unverified


def unverified_effect_correction(reply: Any, receipts: Any) -> str:
    """A correction naming every claimed effect with no receipt behind it."""

    claims = unverified_effect_claims(reply, receipts)
    if not claims:
        return ""
    if len(claims) == 1:
        return (
            f"Correction: I said I {claims[0]} and there is no verified receipt "
            f"for it in this turn, so treat that part as not done."
        )
    listed = ", ".join(claims[:-1]) + f", and {claims[-1]}"
    return (
        f"Correction: I said I {listed}, and none of those have a verified "
        f"receipt in this turn. Treat them as not done."
    )
