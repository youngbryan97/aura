from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import time
import urllib.parse
from collections.abc import Mapping
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field, field_validator

from core.dialogue.referents import resolve_second_person
from core.language.concepts import (
    extract_object_description,
    mentions_object_class,
    object_class_pattern,
)
from core.runtime.content_integrity import (
    contains_paragraph_hashes,
    paragraph_sha256s,
    text_sha256,
)
from core.runtime.desktop_objective_intent import (
    asks_to_be_shown_where,
    looks_like_desktop_objective,
    looks_like_screen_observation,
)
from core.runtime.desktop_task_contract import (
    DESKTOP_TASK_ALLOWED_ACTIONS,
    DESKTOP_TASK_RETRY_SAFE_ACTIONS,
)
from core.runtime.errors import record_degradation
from core.runtime.os_automation_effects import (
    canonical_app_target,
    extract_target_apps,
    extract_target_paths,
)
from core.runtime.watched_goal import read_watched_goal
from core.skills.base_skill import BaseSkill
from core.skills.file_modification_intent import requested_file_modification
from core.skills.os_affordances import detect_os_settings, get_affordance

if TYPE_CHECKING:  # pragma: no cover - typing only
    from core.perception.observation_evidence import (
        Observation as ObservationEvidence,
    )
    from core.runtime.skill_contract import ActionExpectation

logger = logging.getLogger(__name__)

#: Failures the artifact-authoring path can survive: the router is absent,
#: the call times out, or the response is the wrong shape. None of them
#: justify losing the document — the deterministic composer still runs.
_DESKTOP_TASK_RECOVERABLE_ERRORS = (
    AttributeError,
    ImportError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
)

# Sentinel URL resolved at execution time from the most recent
# fetch_topic_image receipt — derivation cannot know the source page
# before the fetch runs ("show me where you found it").
FETCHED_IMAGE_SOURCE_SENTINEL = "aura://fetched-image-source"

# The path the image ACTUALLY landed at, resolved from the fetch receipt for
# the same reason the source URL is: derivation cannot know it in advance.
# The plan guessed ".png" and the gateway saved the JPEG it was actually
# served, so setting the wallpaper failed with "No such file or directory:
# orca_wallpaper.png" while orca_wallpaper.jpg sat on the Desktop — the
# download had worked perfectly and the next step was looking for a file that
# was never going to exist.
FETCHED_IMAGE_PATH_SENTINEL = "aura://fetched-image-path"
MAX_DESKTOP_TASK_STEPS = 32
_VISUAL_ASSET_RE = object_class_pattern("image")

def _computer_use_skill_singleton():
    """The computer_use skill instance used for focus management."""
    global _COMPUTER_USE_SKILL
    if _COMPUTER_USE_SKILL is None:
        from core.skills.computer_use import ComputerUseSkill

        _COMPUTER_USE_SKILL = ComputerUseSkill()
    return _COMPUTER_USE_SKILL


_COMPUTER_USE_SKILL = None

#: Steps whose effect depends on WHICH app is in front. Keystrokes and clicks
#: land wherever focus is, so these re-assert it first; everything else (file
#: writes, downloads, settings) is indifferent to the frontmost window.
_FOCUS_SENSITIVE_ACTIONS = frozenset({"type", "hotkey", "click", "scroll", "read_screen_text"})

#: "wait 5 seconds", "in 10s", "after 3 seconds", "hold on for 2 minutes".
#: The duration has to be STATED. An unquantified "wait a moment" leaves the
#: quantity unspecified, and inventing one would be answering a request the
#: person did not make — the executor's own bound then clamps whatever is
#: asked for, and reports the seconds it actually slept in the receipt.
_REQUESTED_WAIT_RE = re.compile(
    r"\b(?:wait|pause|hold\s+(?:on|off)|delay|give\s+it|after|in)\s+"
    r"(?:for\s+|about\s+|around\s+)?"
    r"(\d+(?:\.\d+)?)\s*"
    r"(seconds?|secs?|s|minutes?|mins?|m)\b",
    re.IGNORECASE,
)


def _requested_wait_seconds(objective: str) -> float:
    """Seconds the request explicitly asks to wait, or 0.0 when it does not."""

    match = _REQUESTED_WAIT_RE.search(str(objective or ""))
    if not match:
        return 0.0
    try:
        amount = float(match.group(1))
    except (TypeError, ValueError):
        return 0.0
    unit = match.group(2).lower()
    if unit.startswith("m") and not unit.startswith("ms"):
        amount *= 60.0
    return max(0.0, amount)


#: Text inside quotes is a name someone chose — a folder, a file, a phrase to
#: type — never a request to open an application.
_QUOTED_SPAN_RE = re.compile(r"['\"\u2018\u2019\u201c\u201d]([^'\"\u2018\u2019\u201c\u201d]{1,80})")


#: The one bound that is real rather than arbitrary: under memory pressure a
#: deep multi-source fetch is what spikes RAM on a live desktop, so the count
#: is capped to protect the runtime — not to express a preference about how
#: much research is enough. It only ever lowers a request, never raises one.
_MEMORY_SAFE_SOURCE_CEILING = 3


def _local_timestamp() -> str:
    """Timestamp string used in user-visible desktop artifacts."""
    return time.strftime("%Y-%m-%d %H:%M:%S %Z")


#: "include the current date and time", "timestamped", "dated".
_TIMESTAMP_REQUEST_RE = re.compile(
    # "a timestamped paragraph" is the commonest phrasing and the one the
    # older \btimestamp\b pattern could never match, because the boundary
    # falls before the "ed".
    r"(?i)\b(?:time\s?stamp(?:ed|s)?|date\s?stamp(?:ed|s)?|dated|"
    r"current\s+(?:date|time)|date\s+and\s+time|time\s+and\s+date)\b"
)


def _objective_wants_a_timestamp(objective: Any) -> bool:
    """Did the request actually ask for the time to be in the document?

    A timestamp nobody asked for is furniture at the top of someone's note;
    a timestamp somebody asked for is the content. The two composers used to
    disagree about this — the freeform one checked, the self-summary one
    always stamped — so the same objective produced a different document
    depending on which path it took.
    """
    return bool(_TIMESTAMP_REQUEST_RE.search(str(objective or "")))


class DesktopTaskStep(BaseModel):
    action: str = Field(
        ...,
        description=(
            "One governed computer_use action: "
            + ", ".join(DESKTOP_TASK_ALLOWED_ACTIONS)
        ),
    )
    target: str | dict[str, Any] = Field("", description="Text, command, URL, app name, script, or JSON action target")
    x: int = Field(0, description="Screen x coordinate for click/scroll/focus")
    y: int = Field(0, description="Screen y coordinate for click/scroll/focus")
    reason: str = Field("", description="Short reason for this step")
    expect: str = Field("", description="Expected observable result")
    critical: bool = Field(
        True,
        description="Whether failure makes the overall objective incomplete.",
    )

    @field_validator("action")
    @classmethod
    def _normalize_action(cls, value: str) -> str:
        action = str(value or "").strip().lower()
        if action not in DESKTOP_TASK_ALLOWED_ACTIONS:
            raise ValueError(f"Unsupported desktop action: {value}")
        return action


class DesktopTaskParams(BaseModel):
    objective: str = Field("", description="Natural-language task objective")
    steps: list[DesktopTaskStep] = Field(default_factory=list, description="Bounded ordered desktop action plan")
    stop_on_error: bool = Field(True, description="Stop after the first failed step")

    @field_validator("steps")
    @classmethod
    def _bounded_steps(cls, value: list[DesktopTaskStep]) -> list[DesktopTaskStep]:
        if len(value) > MAX_DESKTOP_TASK_STEPS:
            raise ValueError(f"Desktop task cannot exceed {MAX_DESKTOP_TASK_STEPS} steps.")
        return value


#: Filesystem paths, removed before any prose classification runs over a
#: request. Matching markers inside a path is how "live-source" made a
#: directory listing into a research assignment.
_PATHS_IN_TEXT_RE = re.compile(r"(?<![\w/])~?/[\w.\-]+(?:/[\w.\-]+)*/?")


def _without_filenames(text: str) -> str:
    """The text with filename tokens blanked, for matching names that are apps.

    "notes.txt", "pages.md", "preview.pdf" each contain an installed app name,
    and matching one as an app sends a file edit into an application that
    never touches the file. Blanking the token rather than banning the name
    keeps "open preview.pdf in Preview" naming both: the filename stops
    claiming the app, the app still does.
    """
    return re.sub(r"\b[\w-]+\.[A-Za-z0-9]{1,6}\b", " ", str(text or ""))


#: How long to wait for a writing lane that says it is still coming up, and
#: how many times.
#:
#: A resident model is twenty gigabytes; one wait was measured as not enough
#: after a restart, while the conversation lane had already reported itself
#: ready. Three waits is under half a minute, still far inside the outer
#: timeouts, and the loop stops the moment the answer stops saying "not yet"
#: — a real failure is still a failure on the first answer.
_WARMING_RETRY_SECONDS = 8.0
_WARMING_WAITS = 3


def _is_still_coming_up(text: Any) -> bool:
    """Whether the router answered with its own not-ready label rather than text.

    The router reports failure in band, as a string beginning ROUTER_ERROR, so
    a caller that does not know the word writes the error into the document.
    This one knows it, and treats the warming case as "not yet" rather than
    "cannot".
    """
    said = str(text or "").strip()
    if not said.startswith("ROUTER_ERROR"):
        return False
    return any(
        marker in said
        for marker in ("worker_not_alive", "init_not_complete", "lane_handshaking", "warming")
    )


#: How a refusal opens, whatever it is refusing.
#:
#: Recognised by shape rather than by the exact sentence: a first person, an
#: inability or an absence, and no subject beyond the request itself. Any
#: fixed list of apologies goes stale the moment somebody writes a new one.
_A_REFUSAL = re.compile(
    r"^\s*(?:i\s*(?:'|’)?m\s+sorry|sorry|unfortunately|i\s+can(?:'|’)?t\b|i\s+cannot\b|"
    r"i\s+(?:am\s+)?un(?:able|available)\b|i\s+could\s*n(?:'|’)?t\b|i\s+do\s*n(?:'|’)?t\s+have\b|"
    r"i\s+was\s*n(?:'|’)?t\s+able\b|i\s+have\s+nothing\b|no\s+answer\b)",
    re.IGNORECASE,
)

#: Words a refusal reaches for when it explains itself, which no document
#: about an ordinary subject has cause to use.
_EXPLAINING_A_REFUSAL = re.compile(
    r"\b(?:language\s+backend|backend\s+is\s+(?:temporarily\s+)?unavailable|"
    r"model\s+is\s+unavailable|try\s+again\s+in\s+a\s+moment|ask\s+me\s+again|"
    r"on\s+my\s+side|my\s+own\s+reasoning|reasoning\s+path)\b",
    re.IGNORECASE,
)


def _says_she_could_not(body: str) -> bool:
    """Whether this text declines rather than says anything."""
    said = " ".join(str(body or "").split())
    if not said:
        return False
    if _A_REFUSAL.match(said):
        return True
    # A refusal can open mildly and explain itself after.
    return bool(_EXPLAINING_A_REFUSAL.search(said[:400]))


def _what_she_actually_did(objective: str) -> str:
    """Her own record, when the thing she is writing about is her own doing.

    A document about her evening has a source, and it is not her self-model.
    Empty for every other subject, so a note about whales is unaffected.
    """
    try:
        from core.introspection.self_evidence import (  # noqa: PLC0415
            asks_about_past_actions,
            render_past_actions,
            resolve_past_actions,
        )

        asked = str(objective or "")
        if not asks_about_past_actions(asked) and not _ABOUT_HER_DOING.search(asked):
            return ""
        # Read straight from the record rather than through the question-shape
        # gate. That gate answers "what did you just do?"; this is a request to
        # WRITE about it, which is not that shape and never will be.
        bundle = resolve_past_actions(limit=8, query=asked)
        if not bundle.grounded:
            return ""
        return str(render_past_actions(bundle) or "").strip()[:1200]
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
        record_degradation(
            "desktop_task",
            exc,
            severity="info",
            action="wrote about her own doing without consulting her record",
        )
        return ""


#: A subject that is her own recent activity, however the request phrases it.
_ABOUT_HER_DOING = re.compile(
    r"\b(?:what\s+you\s+(?:did|have\s+done|worked\s+on|got\s+up\s+to)|"
    r"your\s+(?:evening|night|day|work|session)|"
    r"(?:tonight|today|this\s+evening|this\s+morning|so\s+far|just\s+now|earlier))\b",
    re.IGNORECASE,
)


def _note_unauthored(objective: str, why: str) -> None:
    """Record that she could not author what she was asked to write.

    Three separate returns of "" and only the exception among them left a
    trace, so a task that produced a template instead of a document reported
    success and nobody could tell which guard had fired. LIVE 2026-08-26: a
    note asked for one sentence about the evening held "Notes on the
    requested subject: The requested subject is the focus of this note."
    """
    record_degradation(
        "desktop_task",
        RuntimeError(f"could not author the requested writing: {why}"),
        action="left the document body to the caller rather than inventing one",
        severity="warning",
    )
    logger.warning("desktop_task could not author %r: %s", str(objective or "")[:80], why)


class DesktopTaskSkill(BaseSkill):
    name = "desktop_task"
    description = (
        "Execute a bounded, receipt-producing multi-step desktop plan through "
        "Aura's governed computer_use body. Use for arbitrary chained computer "
        "tasks that need app control, clipboard, browser/app UI, files, PDFs, "
        "or verification steps."
    )
    input_model = DesktopTaskParams
    metabolic_cost = 2
    effect_scope = "foreground_desktop_control"
    #: The floor: enough for the ordinary desktop steps (folders, files, an
    #: app, a wallpaper) with room for a retry.
    timeout_seconds = 180.0

    #: What one researched source costs end to end — fetch, read, and its share
    #: of the synthesis. Measured across the evening of 2026-07-29 at roughly
    #: 30s per source on the resident 32B, taken with margin because the
    #: penalty for being short is losing the whole task.
    _SECONDS_PER_RESEARCHED_SOURCE = 45.0

    #: Composing the document once the sources are read, which happens whether
    #: there are two sources or five.
    _SECONDS_TO_COMPOSE_A_DOCUMENT = 90.0

    @classmethod
    def timeout_for(cls, params: Any) -> float:
        """How long THIS request needs, not how long the average one takes.

        A flat 180s could not describe "make a folder" and "read three
        articles and write a synthesis" at the same time. The same research
        objective measured 98s, 100s, 156s, 161s and 176s across one evening,
        so the declared budget sat inside its own spread — and on 2026-07-29 it
        lost: 93.5s of completed research was cancelled and reported to Bryan
        as "Completed 0/0 steps".

        Reading is the cost and the request says how much reading there is, so
        the budget follows the request the same way the source count does.
        """
        payload = params if isinstance(params, dict) else {}
        objective = str(payload.get("objective") or payload.get("task") or "")
        if not objective:
            return cls.timeout_seconds
        # A goal that is watched runs until it reaches its condition or its
        # own clock stops it. The flat budget cancelled one mid-game: she had
        # found the site, opened it and played to a score of 744, and the turn
        # reported "Operation took too long. Completed 0/0 steps."
        watched = read_watched_goal(objective)
        if watched is not None:
            return max(cls.timeout_seconds, float(watched.max_seconds) + cls._WATCHED_GOAL_GRACE_S)
        if not cls._objective_requests_research_document(objective):
            return cls.timeout_seconds
        sources = cls._requested_research_source_count(objective)
        if sources <= 0:
            # Unspecified: size for what web_search's own default will return
            # rather than inventing a number here.
            sources = 5
        return (
            cls.timeout_seconds
            + cls._SECONDS_TO_COMPOSE_A_DOCUMENT
            + cls._SECONDS_PER_RESEARCHED_SOURCE * sources
        )
    #: Room for the action beneath to finish its last cycle and report. It is
    #: bigger than the action's own grace because this waits on that.
    _WATCHED_GOAL_GRACE_S = 60.0
    _DOCUMENT_BODY_TOKENS = (
        "{{document_body}}",
        "${document_body}",
        "__document_body__",
        "<document_body>",
    )
    _STEP_REFERENCE_PATTERN = re.compile(
        r"\{\{(?P<root>last|steps\.(?P<index>[1-9]\d*))"
        r"\.(?P<path>[A-Za-z_][A-Za-z0-9_]*(?:\.(?:[A-Za-z_][A-Za-z0-9_]*|\d+))*)\}\}"
    )

    @staticmethod
    def _json_target(payload: dict[str, Any]) -> str:
        return json.dumps(payload, ensure_ascii=False)

    @staticmethod
    def _requested_image_folder(objective: str) -> str:
        """Where the person asked the image to go, or a sane default."""
        text = str(objective or "").lower()
        for phrase, folder in (
            ("desktop", "~/Desktop"),
            ("downloads", "~/Downloads"),
            ("documents", "~/Documents"),
            ("pictures", "~/Pictures"),
        ):
            if re.search(
                rf"\b(?:to|in|into|onto|on)\s+(?:my\s+|the\s+)?{phrase}\b", text
            ):
                return folder
        return "~/Documents"

    @staticmethod
    def _safe_filename(text: str, *, default: str = "aura_desktop_task") -> str:
        stem = re.sub(r"[^A-Za-z0-9._ -]+", "", str(text or "")).strip(" ._-")
        stem = re.sub(r"\s+", "_", stem).strip("_")
        return (stem or default)[:80]

    _DIRECTORY_QUESTION_RE = re.compile(
        r"\b(?:how\s+many|count|list|names?\s+of|which)\b[^.?!]{0,80}?"
        r"\bfiles?\b|\bfiles?\b[^.?!]{0,40}?\bin\b",
        re.IGNORECASE,
    )
    _SUFFIX_RE = re.compile(r"\B(\.[A-Za-z0-9]{1,6})\s+files?\b", re.IGNORECASE)

    #: A path-shaped token. A single segment counts too — "the markdown files
    #: in docs" names a directory as plainly as "core/runtime" does — because
    #: what qualifies a token is that it EXISTS on disk, not its shape.
    _DIRECTORY_TOKEN_RE = re.compile(r"\b((?:[\w.-]+/){0,6}[\w-]{2,})\b")

    @classmethod
    def _named_directories(cls, text: str, *, skip: str) -> list[str]:
        """Directory paths the text names that actually exist."""
        from pathlib import Path

        found: list[str] = []
        for match in cls._DIRECTORY_TOKEN_RE.finditer(str(text or "")):
            token = match.group(1).strip("/")
            if not token or token == str(skip) or "." in Path(token).name:
                continue
            for root in (Path.cwd(), Path(__file__).resolve().parents[2]):
                candidate = (root / token).expanduser()
                if candidate.is_dir():
                    found.append(token)
                    break
        return found

    @staticmethod
    def _kind_pattern(text: str) -> str:
        """"python files" is as specific as ".py files".

        The suffix pattern above wanted a literal dot-extension, so counting
        "the python files in core/runtime" listed every file in the directory
        and reported a number answering a different question. The kind-to-
        suffix map already exists for the counter that answers this in chat;
        one mapping, used by both.
        """
        try:
            from core.conversation.filesystem_check import _KIND_SUFFIXES
        except ImportError:
            return "*"
        lowered = str(text or "").lower()
        for kind, suffix in _KIND_SUFFIXES.items():
            if re.search(rf"\b{re.escape(kind)}\s+(?:files?|scripts?|modules?)\b", lowered):
                return f"*{suffix}"
        return "*"

    @classmethod
    def _directory_read_step(cls, text: str, *, skip: str) -> "DesktopTaskStep | None":
        """A read of the directory the request asks about, if it asks about one.

        `skip` is the write destination, which must never be mistaken for the
        thing to read — that confusion is what aimed a write at her own source
        tree.
        """

        if not cls._DIRECTORY_QUESTION_RE.search(text or ""):
            return None
        from core.runtime.os_automation_effects import extract_target_paths

        candidates = [
            path
            for path in extract_target_paths(text) or ()
            if str(path) != str(skip)
        ]
        # A directory has no extension, so the path extractor cannot see one:
        # "count the python files in core/runtime and write the number into
        # aura-report.md" yielded only the destination. Without the source,
        # the read was skipped and the file was written with composed filler —
        # or, worse, the destination itself was taken as the directory to
        # read. The filesystem is the authority on what is a directory.
        if not candidates:
            candidates = cls._named_directories(text, skip=skip)
        if not candidates:
            return None
        source = candidates[0]
        suffix = cls._SUFFIX_RE.search(text or "")
        pattern = f"*{suffix.group(1)}" if suffix else cls._kind_pattern(text)
        return DesktopTaskStep(
            action="list_directory",
            target=json.dumps({"path": source, "pattern": pattern}),
            reason="The request asks about the contents of a directory.",
            expect=f"{source} is read and its {pattern} entries counted.",
            critical=True,
        )

    _WRITE_DESTINATION_RE = re.compile(
        r"\b(?:write|save|put|store|append|export|dump|record)\b[^.?!]{0,80}?"
        r"\b(?:into|to|in)\s+(?P<path>~?[\w./\-]*[\w.\-]+)",
        re.IGNORECASE,
    )

    @classmethod
    def _ordered_by_write_destination(
        cls, text: str, named_paths: tuple[str, ...] | list[str]
    ) -> tuple[str, ...]:
        """Put the path being WRITTEN TO first, not the one named first.

        LIVE, 2026-08-10. "Count how many .py files are in
        /Users/bryan/.aura/live-source/core/introspection, then write that
        number and the file names into ~/Documents/aura_probe_count.txt" tried
        to write to the source directory and was refused by the artifact-root
        guard, which was the only thing standing between a read request and a
        write into her own source tree.

        The planner took named_paths[0], and the first path in a sentence is
        whatever the sentence talks about first — here, the thing to READ. Two
        signals separate them: a path introduced by a write verb plus "into" or
        "to" is the destination, and a destination for write_text_file is a
        file rather than a bare directory.
        """

        paths = [str(path) for path in named_paths if str(path).strip()]
        if len(paths) < 2:
            return tuple(paths)

        def rank(path: str) -> tuple[int, int]:
            tail = path.rstrip("/").rsplit("/", 1)[-1]
            named_as_destination = any(
                match.group("path") and match.group("path").rstrip("/.,") in path
                for match in cls._WRITE_DESTINATION_RE.finditer(text or "")
            )
            looks_like_a_file = 1 if "." in tail else 0
            return (0 if named_as_destination else 1, 0 if looks_like_a_file else 1)

        return tuple(sorted(paths, key=rank))

    @staticmethod
    def _extract_folder_name(objective: str) -> str:
        text = str(objective or "")
        # Quoted names may contain possessive apostrophes ("Aura's
        # Journal"); the close-quote is the one followed by a boundary,
        # not the first internal apostrophe (which truncated the name to
        # "Aura" and broke the journal demo's folder).
        match = re.search(
            r"\b(?:folder|directory)\b[^.\n]{0,80}?\b(?:named|called|titled)\s+"
            r"(?:'((?:[^']|'(?=\w))+)'(?=[\s.,;)]|$)"
            r"|\"([^\"]+)\""
            r"|([^.,;\n]+?)(?=\s+(?:in|inside|under|on)\s+(?:my\s+)?\w|[.,;\n]|$))",
            text,
            flags=re.IGNORECASE,
        )
        if match:
            name = str(match.group(1) or match.group(2) or match.group(3) or "").strip()
            return name.strip("'\"., ")[:100]
        # Name-first phrasing: "the 'Aura's Journal' folder" — quoted name
        # immediately before the word folder/directory.
        name_first = re.search(
            r"(?:'((?:[^']|'(?=\w))+)'|\"([^\"]+)\")\s+(?:folder|directory)\b",
            text,
            flags=re.IGNORECASE,
        )
        if name_first:
            name = str(name_first.group(1) or name_first.group(2) or "").strip()
            if name:
                return name.strip("'\"")[:100]
        return f"Aura Desktop Task {int(time.time())}"

    @staticmethod
    def _extract_root_hint(objective: str) -> str:
        """Honor the user's stated artifact root.

        Live proof rounds wrote to the Desktop default while the user
        said 'in my Documents folder' — parameter fidelity is general
        capability, not pattern-matching: extract what was actually
        asked.
        """
        lowered = str(objective or "").lower()
        for token, root in (
            ("documents folder", "~/Documents"),
            ("my documents", "~/Documents"),
            ("documents directory", "~/Documents"),
            ("downloads folder", "~/Downloads"),
            ("my downloads", "~/Downloads"),
            ("desktop folder", "~/Desktop"),
            ("my desktop", "~/Desktop"),
        ):
            if token in lowered:
                return root
        return ""

    @staticmethod
    def _extract_explicit_filename(objective: str) -> str:
        """The user's stated filename wins over generated stems."""
        match = re.search(
            r"\bfile\b[^.\n]{0,60}?\b(?:named|called|titled)\s+"
            r"['\"]?([\w][\w .-]{0,80}?\.(?:txt|md|markdown|rtf|text))['\"]?",
            str(objective or ""),
            flags=re.IGNORECASE,
        )
        return match.group(1).strip() if match else ""

    @staticmethod
    def _explicit_pdf_requested(objective: str) -> bool:
        text = str(objective or "").lower()
        if "pdf" in text or "portable document" in text:
            return True
        return bool(re.search(r"\b(?:export|save)\s+(?:it\s+|this\s+|the\s+\w+\s+)?as\s+(?:a\s+)?pdf\b", text))

    @staticmethod
    def _web_document_url(objective: str) -> str:
        text = str(objective or "").lower()
        google_surface = any(
            marker in text
            for marker in (
                "google docs",
                "google doc",
                "docs.google",
                "google document",
                "google sheets",
                "google spreadsheet",
                "sheets.google",
                "google slides",
                "google presentation",
                "slides.google",
                "google drive",
                "drive.google",
            )
        )
        surfaces = (
            (
                (
                    "google docs",
                    "google doc",
                    "docs.google",
                    "google document",
                    "docs",
                    "doc",
                    "document",
                ),
                "https://docs.google.com/document/u/0/create",
                google_surface,
            ),
            (
                ("google sheets", "google spreadsheet", "sheets.google", "sheets", "spreadsheet", "sheet"),
                "https://docs.google.com/spreadsheets/u/0/create",
                google_surface,
            ),
            (
                ("google slides", "google presentation", "slides.google", "slides", "presentation", "slide"),
                "https://docs.google.com/presentation/u/0/create",
                google_surface,
            ),
            (
                ("google drive", "drive.google", "drive", "cloud storage"),
                "https://drive.google.com/drive/my-drive",
                google_surface,
            ),
            (("notion",), "https://www.notion.so/"),
        )
        for markers, url, *required in surfaces:
            if required and not required[0]:
                continue
            if any(re.search(rf"\b{re.escape(marker)}s?\b", text) for marker in markers):
                return url
        return ""

    @staticmethod
    def _extract_search_query(objective: str) -> str:
        # Manner phrases say WHERE to look, not WHAT to look for, and they are
        # removed before the topic patterns run — otherwise "on the internet"
        # becomes the topic and "orcas online" searches for a wireless ISP.
        text = DesktopTaskSkill._SEARCH_MANNER_ANYWHERE_RE.sub(
            " ", str(objective or "")
        )
        text = " ".join(text.split()).strip()
        count_word = r"(?:\d+|one|two|three|four|five)"
        patterns = (
            rf"\bfind\s+(?:me\s+)?(?:{count_word}\s+)?(?:different\s+)?(?:articles?|sources?|stories?|news)\s+(?:on|about|for)\s+([^.;\n,]+)",
            rf"\b(?:summari[sz]e|write\s+(?:a\s+)?summary\s+of)\s+(?:{count_word}\s+)?(?:different\s+)?(?:articles?|sources?|stories?|news)\s+(?:on|about|for)\s+([^.;\n,]+)",
            r"\b(?:articles?|sources?|stories?|news)\s+(?:on|about|for)\s+([^.;\n,]+)",
            r"\bsearch\s+(?:for\s+)?([^.;\n]+)",
            r"\blook\s+up\s+([^.;\n]+)",
            r"\bgoogle\s+([^.;\n]+)",
            r"\bopen\s+(?:a\s+)?(?:browser\s+)?tab\s+(?:on\s+google\s+)?(?:for\s+)?([^.;\n]+)",
        )
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                query = match.group(1).strip(" ,")
                if query:
                    if re.match(
                        r"^(?:doc|docs|document|drive|sheet|sheets|slide|slides|chrome|safari|browser)\b",
                        query,
                        flags=re.IGNORECASE,
                    ):
                        continue
                    if query.lower() in {"it", "them", "this", "that", "her", "him", "me", "us", "something", "anything"}:
                        # Resolve the coreference pronoun to preceding topic in context
                        m = re.search(r"\b(?:read|find|search)\s+(?:about|on|for)\s+([^.;\n,]+)", text, flags=re.IGNORECASE)
                        if m:
                            candidate = m.group(1).strip(" ,")
                            if candidate.lower() not in {"it", "them", "this", "that", "her", "him", "me", "us", "something", "anything"}:
                                    return candidate[:240]
                    else:
                        return DesktopTaskSkill._strip_search_manner(query)[:240]
        if "news" in text.lower():
            return DesktopTaskSkill._strip_search_manner(text)[:240]
        return ""

    #: Words that say WHERE to look, not WHAT to look for. "find 3 recent
    #: articles about orcas online" is a request about orcas, searched online —
    #: not a request about "orcas online", which is a wireless ISP on Orcas
    #: Island. Measured live: the PDF she wrote was a competent summary of that
    #: ISP's vacation-hold policy and password expiry.
    #: The same manner phrases, matched anywhere rather than only at the end.
    _SEARCH_MANNER_ANYWHERE_RE = re.compile(
        r"(?i)\s+\b("
        # "online" only where it modifies the SEARCH — at a clause boundary.
        # In "the online safety act" it is part of the name, not a manner.
        r"online(?=\s*[,.;:]|\s+(?:and|then|read|so|to)\b|$)"
        r"|on\s+the\s+(?:internet|web|net)|on\s+google|via\s+google"
        r"|from\s+the\s+(?:internet|web)"
        r")\b"
    )

    _SEARCH_MANNER_RE = re.compile(
        r"(?i)[\s,]*\b("
        r"online|on\s+the\s+(?:internet|web)|on\s+google|via\s+google|"
        r"from\s+the\s+(?:internet|web)|on\s+the\s+net"
        r")\b\s*$"
    )

    @staticmethod
    def _strip_search_manner(query: str) -> str:
        """Remove a trailing manner phrase so the topic is what she searches."""
        text = " ".join(str(query or "").split())
        previous = None
        while text and text != previous:
            previous = text
            text = DesktopTaskSkill._SEARCH_MANNER_RE.sub("", text).strip(" ,")
        return text

    @staticmethod
    def _requested_visible_source_count(objective: str) -> int:
        """How many sources the person asked to actually SEE. 0 if they did not.

        Distinct from ``_requested_research_source_count``, which is how many
        she READS. "Find 3 articles and write me a PDF" wants three articles
        researched and zero browser tabs; opening them anyway answers a
        request nobody made, with someone else's windows.

        So the opening verb has to sit beside the count. An earlier version
        searched the whole objective for "open", which meant "Open Google
        Chrome, find 3 articles…" licensed three tabs on the strength of a
        verb belonging to a different clause — and then fell through to a
        default of 3 that nobody had asked for.
        """
        lowered = str(objective or "").lower()
        counted_sources = (
            r"(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|"
            r"a couple(?: of)?|a few|several)\s+"
            + DesktopTaskSkill._COUNT_MODIFIERS
            + DesktopTaskSkill._COUNTED_NOUN
        )
        direct = re.search(
            r"\b(?:open|show|bring up|pull up)\b[^,.;]{0,24}\b"
            + counted_sources
            + r"\b",
            lowered,
        )
        deferred = re.search(
            r"\b"
            + counted_sources
            + r"\b[^,.;]{0,48}\b(?:open|show|bring up|pull up)\s+(?:each|them|those)\b",
            lowered,
        )
        keep_open = re.search(
            r"\b(?:keep|leave)\s+(?:all\s+)?(?:the|those)?\s*"
            r"(?:articles?|sources?|stories?|pieces?|links?|results?|them)\s+open\b",
            lowered,
        )
        if keep_open is not None:
            return DesktopTaskSkill._counted_in_request(lowered)
        matched = direct or deferred
        if matched is None:
            return 0
        return DesktopTaskSkill._counted_in_request(matched.group(0))

    @staticmethod
    def _requested_research_source_count(objective: str) -> int:
        """How many sources the request asked for. 0 means it did not say.

        It used to say 1 for any objective mentioning sources, and 3 if the
        word "different" appeared — two numbers nobody chose for this
        request. 0 is the honest answer, and every caller now handles it as
        "unspecified" rather than substituting a favourite.
        """
        return DesktopTaskSkill._counted_in_request(str(objective or "").lower())

    @staticmethod
    def _objective_requests_recent_sources(objective: str) -> bool:
        return bool(
            re.search(
                r"\b(?:recent|latest|current|newly published|new reporting)\b",
                str(objective or ""),
                flags=re.IGNORECASE,
            )
        )

    @classmethod
    def _objective_requests_source_reading(cls, objective: str) -> bool:
        if cls._objective_requests_authored_synthesis(objective):
            return True
        return bool(
            re.search(
                r"\b(?:read|review|study|inspect|compare|evaluate|assess|look through)\b",
                str(objective or ""),
                flags=re.IGNORECASE,
            )
        )

    #: Adjectives a person puts between the number and the noun. "3 RECENT
    #: articles" matched nothing before this and fell through to 1, so a
    #: request for three sources was validated against one — and only
    #: "different" was ever allowed through, which is the narrowest possible
    #: version of the general case.
    _COUNT_MODIFIERS = (
        r"(?:(?:different|recent|new|latest|good|solid|reliable|credible|"
        r"separate|independent|current|major|top|real|actual)\s+){0,3}"
    )

    _COUNTED_NOUN = r"(?:articles?|sources?|stories?|pieces?|links?|results?)"

    #: Words for small numbers, because people write both.
    _COUNT_WORDS = {
        "one": 1,
        "two": 2,
        "a couple": 2,
        "a couple of": 2,
        "three": 3,
        "a few": 3,
        "several": 3,
        "four": 4,
        "five": 5,
    }

    @classmethod
    def _counted_in_request(cls, lowered: str) -> int:
        """How many sources the person actually asked for, or 0 if unsaid.

        ONE parser, because there were two that had diverged: the visible-tab
        count and the research count carried the same regex with different
        fallbacks, so fixing a phrasing in one left the other wrong.
        """
        text = str(lowered or "").lower()
        digits = re.search(
            rf"\b([1-9][0-9]?)\s+{cls._COUNT_MODIFIERS}{cls._COUNTED_NOUN}\b", text
        )
        if digits:
            return max(1, min(5, int(digits.group(1))))
        for word, value in cls._COUNT_WORDS.items():
            if re.search(
                rf"\b{re.escape(word)}\s+{cls._COUNT_MODIFIERS}{cls._COUNTED_NOUN}\b",
                text,
            ):
                return value
        return 0

    @staticmethod
    def _objective_requests_research_document(objective: str) -> bool:
        # Classify the PROSE, not the paths in it.
        #
        # LIVE, 2026-08-10: "count how many .py files are in
        # /Users/bryan/.aura/live-source/core/introspection, then write that
        # number ... into ~/Documents/aura_probe_count.txt" was classified as a
        # research-document objective, so completion required research SOURCES
        # and the turn reported "semantic completion incomplete:
        # requested_source_count_found" over a filesystem task that had
        # succeeded.
        #
        # The marker was "source", matched inside "live-source". A path is a
        # name, not prose, and every marker here is a substring test over
        # whatever the user happened to type — so any objective naming a path
        # with "report", "news", "article" or "source" in it inherits a
        # contract about citations.
        lowered = _PATHS_IN_TEXT_RE.sub(" ", str(objective or "")).lower()
        has_source_markers = any(
            marker in lowered
            for marker in (
                "article",
                "articles",
                "sources",
                "source",
                "news",
                "research",
                "report",
                "reports",
            )
        )
        visual_reference_only = (
            mentions_object_class(lowered, "image") and not has_source_markers
        )
        if visual_reference_only:
            return False
        wants_research = any(
            marker in lowered
            for marker in (
                "article",
                "articles",
                "sources",
                "source",
                "news",
                "research",
                "look up",
                "search",
                "find",
            )
        )
        wants_written_output = any(
            marker in lowered
            for marker in (
                "summarize",
                "summary",
                "write",
                "document",
                "doc",
                "essay",
                "report",
                "note",
                "pdf",
                "type",
            )
        )
        return wants_research and wants_written_output

    @classmethod
    def _artifact_document_title(cls, objective: str) -> str:
        """The title printed at the top of the document.

        The filename has always been derived from what the request was about
        (`orcas_summary.pdf`); the title inside the PDF was the literal string
        "Aura Desktop Task". So Bryan's orca synthesis opened with a heading
        naming the machinery that produced it rather than its subject —
        measured live 2026-07-29. The stem already carries the intent; this
        renders the same intent for a human reader.
        """
        stem = cls._artifact_filename_stem(objective)
        if stem in {"aura_desktop_summary", ""}:
            return "Aura Desktop Task"
        if stem == "aura_self_summary":
            return "About Aura"
        words = [part for part in re.split(r"[_\s]+", stem) if part]
        if not words:
            return "Aura Desktop Task"
        return " ".join(word[:1].upper() + word[1:] for word in words)

    @classmethod
    def _artifact_filename_stem(cls, objective: str) -> str:
        """Name an artifact from its content intent, not its destination."""
        if cls._objective_requests_self_summary(objective):
            return "aura_self_summary"
        if cls._objective_requests_research_document(objective):
            query = cls._extract_search_query(objective)
            if query:
                return cls._safe_filename(f"{query} summary")
        match = re.search(
            r"\b(?:essay|report|summary|note|document|draft)\s+"
            r"(?:on|about|of|for)\s+([^.;,\n]{2,100})",
            str(objective or ""),
            flags=re.IGNORECASE,
        )
        if match:
            return cls._safe_filename(match.group(1))
        return "aura_desktop_summary"

    @staticmethod
    def _search_url(query: str, *, images: bool = False, engine: str = "") -> str:
        encoded = urllib.parse.quote_plus(str(query or "").strip())
        if not encoded:
            return ""
        if engine == "google":
            # The user said Google — honor it (their sessions and habits
            # live there); DuckDuckGo stays the neutral default otherwise.
            if images:
                return f"https://www.google.com/search?q={encoded}&tbm=isch"
            return f"https://www.google.com/search?q={encoded}"
        if images:
            return f"https://duckduckgo.com/?q={encoded}&iax=images&ia=images"
        return f"https://duckduckgo.com/?q={encoded}"

    @staticmethod
    def _preferred_browser(objective: str) -> str:
        """Which browser the user's phrasing points at, if any."""
        lowered = str(objective or "").lower()
        if "safari" in lowered:
            return "Safari"
        if "chrome" in lowered or re.search(
            r"\bgoogle\s+(?:docs?|drive|sheets?|slides|gmail|account|document|spreadsheet|presentation)\b|"
            r"\b(?:docs|drive|sheets|slides)\.google\b",
            lowered,
        ):
            return "Google Chrome"
        if (
            re.search(rf"\b{_VISUAL_ASSET_RE}\b", lowered)
            and re.search(r"\b(?:online|internet|web|source|found|show)\b", lowered)
        ):
            return "Google Chrome"
        return ""

    @staticmethod
    def _search_engine_hint(objective: str) -> str:
        lowered = str(objective or "").lower()
        return "google" if "google" in lowered else ""

    @staticmethod
    def _extract_image_query(objective: str) -> str:
        text = str(objective or "").strip()
        described_object = extract_object_description(
            text,
            "image",
            action_phrases=("find", "search", "look up", "get", "download", "fetch"),
        )
        patterns = (
            rf"\b{_VISUAL_ASSET_RE}\s+of\s+([^.;\n]+)",
            rf"\b(?:find|search|look\s+up)\s+(?:an?\s+)?{_VISUAL_ASSET_RE}\s+(?:of\s+)?([^.;\n]+)",
            rf"\b(?:find|search(?:\s+for)?|look\s+up|get)\b\s+"
            rf"(?:an?\s+|some\s+)?([^.;\n]{{2,120}}?)\s+{_VISUAL_ASSET_RE}\b",
            rf"\b([^.;\n]{{2,120}}?)\s+{_VISUAL_ASSET_RE}\b",
        )
        candidates = [described_object]
        candidates.extend(
            match.group(1)
            for pattern in patterns
            if (match := re.search(pattern, text, flags=re.IGNORECASE)) is not None
        )
        for candidate in candidates:
            if candidate:
                query = re.sub(r"\b(?:and|then|also)\b.*$", "", candidate, flags=re.IGNORECASE)
                query = re.sub(
                    r"\bfrom\s+(?:online|the\s+(?:internet|web))\b.*$",
                    "",
                    query,
                    flags=re.IGNORECASE,
                )
                query = re.sub(
                    r"\b(?:online|on\s+the\s+(?:internet|web)|from\s+the\s+(?:internet|web))\b.*$",
                    "",
                    query,
                    flags=re.IGNORECASE,
                )
                query = re.sub(r"^(?:a|an|the)\s+", "", query.strip(" ,"), flags=re.IGNORECASE)
                query = query.strip(" ,")
                if query:
                    return query[:240]
        return ""

    @staticmethod
    def _wants_image_source_shown(objective: str) -> bool:
        lowered = str(objective or "").lower()
        return bool(
            re.search(r"\bshow\b[^.;\n]{0,40}\b(?:where|source|found)\b", lowered)
            or "where you found" in lowered
        )

    @staticmethod
    def _extract_apps(objective: str) -> list[str]:
        text = str(objective or "").lower()
        apps: list[str] = []
        app_markers = {
            "notes": "Notes",
            "calculator": "Calculator",
            "finder": "Finder",
            "preview": "Preview",
            "safari": "Safari",
            "chrome": "Google Chrome",
            "browser": "Safari",
            "textedit": "TextEdit",
            "pages": "Pages",
            "microsoft word": "Microsoft Word",
            "ms word": "Microsoft Word",
        }
        # Word-boundary matching: the bare substring scan opened
        # Microsoft Word because the objective said "in your own words"
        # — a fatal launch on Macs without Word. Apps must be NAMED.
        #
        # A word boundary is not enough on its own, because "." is one:
        # "add a line to the end of notes.txt" matched \bnotes\b and routed a
        # file edit into the Notes app, where the file on disk was never
        # touched. A name carrying a file extension is a FILENAME — same
        # failure as "in your own words", one punctuation mark along.
        named = _without_filenames(text)
        for marker, app in app_markers.items():
            if re.search(rf"\b{re.escape(marker)}\b", named) and app not in apps:
                if marker == "browser" and "chrome" in text:
                    continue
                apps.append(app)

        # ...and then everything else that is actually on this machine.
        #
        # The table above is eleven names, so "open Reminders" named no app at
        # all and the work fell through to a text file on disk. The machine
        # already knows what is installed; an app it has is an app she can be
        # asked for, without anyone adding a row.
        #
        # A NAMED app needs a verb, and must not be inside quotes.
        #
        # The eleven-name table above could match loosely because those names
        # rarely appear by accident. Ninety-one cannot: "a new folder called
        # 'Aura's Journal'" contains two installed app names, and matching
        # them opened two applications to make a folder. This is the same
        # failure as "in your own words" launching Microsoft Word, one list
        # further along — so the general form carries the general guard.
        quoted = " ".join(_QUOTED_SPAN_RE.findall(text))
        try:
            from core.perception.app_dictionary import installed_apps

            for name in installed_apps():
                lowered_name = name.lower()
                if name in apps or len(name) < 4:
                    continue
                if re.search(rf"\b{re.escape(lowered_name)}\b", quoted):
                    continue  # It is the name of a thing, not a request.
                if re.search(
                    rf"\b(?:open|launch|start|run|use|using|switch\s+to|"
                    rf"in|into|inside|with|via|from)\s+"
                    rf"(?:up\s+)?(?:my\s+|the\s+|a\s+)?"
                    rf"{re.escape(lowered_name)}\b",
                    named,
                ):
                    apps.append(name)
        except _DESKTOP_TASK_RECOVERABLE_ERRORS as exc:
            logger.debug("Could not enumerate installed applications: %s", exc)
        return apps[:4]

    @staticmethod
    def _json_candidates_from_text(text: str) -> list[str]:
        source = str(text or "").strip()
        if not source:
            return []
        candidates: list[str] = []
        candidates.extend(
            match.group(1).strip()
            for match in re.finditer(r"```(?:json)?\s*(.*?)```", source, flags=re.IGNORECASE | re.DOTALL)
        )
        for open_char, close_char in (("{", "}"), ("[", "]")):
            start = source.find(open_char)
            end = source.rfind(close_char)
            if start >= 0 and end > start:
                candidates.append(source[start : end + 1])
        return candidates

    @classmethod
    def _structured_payload_from_text(cls, text: str) -> dict[str, Any]:
        for candidate in cls._json_candidates_from_text(text):
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed
            if isinstance(parsed, list):
                return {"steps": parsed}
        return {}

    @classmethod
    def _structured_payload_from_context(cls, context: dict[str, Any] | None) -> dict[str, Any]:
        context = context or {}
        for key in ("desktop_task_plan", "desktop_task_steps", "desktop_task_document_body", "cognitive_reply", "draft_response", "response"):
            value = context.get(key)
            if isinstance(value, dict):
                return dict(value)
            if isinstance(value, list):
                return {"steps": value}
            payload = cls._structured_payload_from_text(str(value or ""))
            if payload:
                return payload
        return {}

    _DISPATCH_NARRATION_RE = re.compile(
        r"(?:i'?ve started (?:working on )?th(?:is|e) task|"
        r"i'?ll follow up when|tracking commitment\s+[0-9a-f]{6,}|"
        r"task \(id=[0-9a-f-]{6,}\)|in the background\b.{0,40}follow up|"
        r"(?:opening|open)\s+(?:notes|google docs|chrome|safari)\b.{0,120}(?:creating|writing|typing|following content)|"
        r"(?:i\s+can\s+guide\s+you\s+through|here'?s\s+how|steps\s+to\s+do\s+that|do\s+that\s+yourself)|"
        # An adverb between the modal and the verb defeated this. Live
        # 2026-07-28 the Notes app really was opened and a note really was
        # created — and its body was "I can't DIRECTLY interact with your
        # phone or its apps. But I could help you write something about orcas
        # and give it to you as text!" The refusal became the artifact,
        # because "can't directly interact" is not "can't interact".
        r"(?:i(?:'m| am)\s+not\s+(?:\w+ly\s+){0,2}(?:actually\s+)?able\s+to\s+"
        r"(?:\w+ly\s+)?(?:interact|access|control|open|write|do)|"
        r"i\s+(?:cannot|can'?t)\s+(?:\w+ly\s+){0,2}"
        r"(?:interact|access|control|open|write|create|edit)\b|"
        r"you\s+can\s+copy\s+it\s+into\s+notes)|"
        r"(?:the\s+)?task\s+(?:asked|asks|requested|requests)\s+(?:me\s+)?to\s+(?:type|write|open|create|export)|"
        r"i\s+am\s+(?:typing|writing|pasting)\s+(?:here|this)\s+because\s+(?:the\s+)?task\s+(?:asked|requires)|"
        # A question back to the user is a conversational turn, never the
        # product of a task. Read out of Bryan's real Notes app, 2026-07-28,
        # as note titles: "Could you tell me what kind of text to generate
        # instead?" and "What kind of content are you looking for in those
        # notes?" — Notes takes its title from the first line, so a
        # conversational reply became a note called that.
        r"(?:what\s+kind\s+of\s+(?:text|content|note)|"
        r"could\s+you\s+tell\s+me\s+what|"
        r"would\s+you\s+like\s+me\s+to\s+(?:write|generate|include)|"
        r"let\s+me\s+know\s+what\s+(?:you|kind))|"
        # A denial of the machine she is running on.
        r"i\s+don'?t\s+have\s+a\s+mac|i'?m\s+running\s+on\s+a\s+server|"
        r"no\s+notes\s+app|"
        r"i'?ll\s+simulate\s+(?:this|the)\s+process|"
        r"step[- ]by[- ]step\s+as\s+if\s+i\s+were|"
        r"pretend\s+(?:the\s+)?app\s+is\s+opening|"
        # Internal execution brief / directive — instruction to herself, not
        # document content (it leaked into a research PDF as the body).
        r"execute the user'?s (?:explicit )?desktop objective|"
        r"governed desktop_task lane|do not claim success until|"
        r"aura desktop task receipt|canonical computer-use gateway)",
        re.IGNORECASE,
    )
    _ARTIFACT_REFERENCE_RE = re.compile(
        r"\n\s*Artifact references:\s*.*\Z",
        re.IGNORECASE | re.DOTALL,
    )
    _INCOMPLETE_DOCUMENT_TAIL_RE = re.compile(
        r"(?:"
        r"\bnot\s+just\b|"
        r"\b(?:because|although|though|while|when|where|whether|if|unless)\b|"
        r"\b(?:and|or|but|so|as|with|through|from|into|toward|between|across|rather\s+than)\b"
        r")\s*(?:[.!?])?\s*\Z",
        re.IGNORECASE,
    )

    @classmethod
    def _strip_artifact_reference_tail(cls, text: str) -> str:
        """Remove receipt/reference footer before validating authored prose."""
        return cls._ARTIFACT_REFERENCE_RE.sub("", str(text or "").strip()).strip()

    @classmethod
    def _looks_like_incomplete_document_body(cls, text: str) -> bool:
        """Catch model continuations that end mid-thought before disk write."""
        body = cls._strip_artifact_reference_tail(text)
        if not body:
            return True
        if not re.search(r"[.!?][\"')\]]*\s*$", body):
            return True
        tail = re.sub(r"\s+", " ", body[-96:]).strip()
        return bool(cls._INCOMPLETE_DOCUMENT_TAIL_RE.search(tail))

    @staticmethod
    def _objective_requests_opinion(objective: str) -> bool:
        """Does the objective ask Aura for her own view, not just a summary?"""
        lowered = str(objective or "").lower()
        return bool(
            re.search(r"\b(?:your|my|her|own)\s+(?:opinion|view|views|take|thoughts|assessment|perspective|stance)\b", lowered)
            or re.search(r"\bform\s+(?:your|an?|my)\s+(?:own\s+)?opinion\b", lowered)
            or "what you think" in lowered
            or "what do you think" in lowered
        )

    @classmethod
    def _looks_like_dispatch_narration(cls, text: str) -> bool:
        """Status narration is not document content.

        Round-12 all-green proof had one wrinkle: the written file
        contained 'I've started working on this task... Tracking
        commitment bbbaba54' — her dispatch status echoed into the
        artifact because cognitive_reply was the body fallback. A
        status message about doing the task must never become the
        product of the task.
        """
        return bool(cls._DISPATCH_NARRATION_RE.search(str(text or "")))

    @staticmethod
    def _extract_declared_document_content(text: str) -> str:
        """Pull authored content out of model preambles like "write this content:"."""
        value = str(text or "").strip()
        if not value:
            return ""
        patterns = (
            r"(?:following\s+)?content\s*[:：]\s*[-–—]*\s*(.+)$",
            r"\bhere\s+(?:it\s+is|is\s+the\s+(?:paragraph|note|document|content))\s*[:：]\s*[-–—]*\s*(.+)$",
            r"(?:note|paragraph|document)\s+(?:text|body)\s*[:：]\s*[-–—]*\s*(.+)$",
            r"(?:write|type|insert)\s+(?:this\s+)?(?:text|paragraph|content)\s*[:：]\s*[-–—]*\s*(.+)$",
        )
        for pattern in patterns:
            match = re.search(pattern, value, flags=re.IGNORECASE | re.DOTALL)
            if not match:
                continue
            body = str(match.group(1) or "").strip(" \n\r\t-–—")
            if body:
                return DesktopTaskSkill._strip_artifact_action_tail(body)[:9000]
        return ""

    @staticmethod
    def _literal_command_tail_boundary(text: str) -> int:
        """Locate a following desktop command without truncating ordinary prose."""
        match = re.search(
            r"(?:\s*,?\s+(?:and\s+then|then|after\s+that|afterwards|next)\s+|"
            r"\s*,\s+and\s+|\s+and\s+)"
            r"(?=(?:open|save|export|move|copy|close|print|share|upload|download|"
            r"create|make|render|convert|rename|delete|remove|send|email)\b)",
            str(text or ""),
            flags=re.IGNORECASE,
        )
        return match.start() if match else len(str(text or ""))

    @classmethod
    def _literal_document_body_from_objective(cls, objective: str) -> str:
        """Extract user-authored text that should be reproduced exactly.

        This only accepts explicit content cues or directly quoted operands.
        Topic requests such as ``write a note about climate`` deliberately do
        not qualify because they require composition rather than transcription.
        """
        text = str(objective or "").replace("\x00", "").strip()
        if not text:
            return ""

        cue_patterns = (
            r"\b(?:saying|that\s+says?|containing)\b\s*(?::|=|-|,)?\s*",
            r"\bwith\s+(?:the\s+)?(?:exact\s+)?(?:text|content|message|words?)\b\s*(?::|=|-)?\s*",
        )
        starts = [
            match.end()
            for pattern in cue_patterns
            if (match := re.search(pattern, text, flags=re.IGNORECASE))
        ]

        # Direct quoted operands are equally explicit: type "Hello" in Notes.
        direct = re.search(
            r"\b(?:write|type|paste|insert|add)\b\s+"
            r"(?:the\s+)?(?:exact\s+)?(?:text\s+|content\s+|message\s+)?"
            r"(?=[\"'`\u2018\u201c])",
            text,
            flags=re.IGNORECASE,
        )
        if direct:
            starts.append(direct.end())
        if not starts:
            return ""

        start = min(starts)
        remainder = text[start:].lstrip()
        if not remainder:
            return ""

        quote_pairs = {
            '"': {'"', "\u201d"},
            "'": {"'", "\u2019"},
            "`": {"`"},
            "\u2018": {"\u2019", "'"},
            "\u201c": {"\u201d", '"'},
        }
        opener = remainder[0]
        if opener in quote_pairs:
            closers = quote_pairs[opener]
            candidate = remainder[1:]
            close_index = -1
            for index, char in enumerate(candidate):
                if char not in closers:
                    continue
                # Apostrophes inside words are content, not delimiters.
                before = candidate[index - 1] if index else ""
                after = candidate[index + 1] if index + 1 < len(candidate) else ""
                if char in {"'", "\u2019"} and before.isalnum() and after.isalnum():
                    continue
                close_index = index
                break
            if close_index >= 0:
                body = candidate[:close_index]
            else:
                body = candidate[: cls._literal_command_tail_boundary(candidate)]
        else:
            body = remainder[: cls._literal_command_tail_boundary(remainder)]
            body = body.rstrip(" \t\r\n")
            # Sentence punctuation belongs to the literal. A terminal comma
            # only separates the content from a following command.
            body = re.sub(r",\s*$", "", body)

        if not body or len(body) > 9000:
            return ""
        return body

    @classmethod
    def _objective_supplies_literal_document_body(cls, objective: str) -> bool:
        return bool(cls._literal_document_body_from_objective(objective))

    @staticmethod
    def _strip_status_narration_head(text: str) -> str:
        """Drop status narration the model prefixed to real content.

        Live 2026-07-28 the note was created and its content was correct —
        three genuine sentences about humpback whales — but the body began
        "<Notes app opened> New note created. ". Notes takes its title from
        the first line, so the note was NAMED after the executor's progress
        report.

        The whole-body guard cannot help here: the body is good, only its
        head is machinery. Rejecting it would throw away real writing, so
        this trims rather than refuses.
        """
        body = str(text or "").lstrip()
        for _ in range(3):
            trimmed = re.sub(
                r"^\s*<[^>\n]{1,60}>\s*", "", body
            )
            trimmed = re.sub(
                r"^\s*(?:new\s+(?:note|document|file)\s+created|"
                r"note\s+created|document\s+created|opened\s+\w+|"
                r"creating\s+(?:the\s+)?(?:note|document)|"
                r"here\s+(?:is|'s)\s+the\s+note)\b[\s.:,\-–—]*",
                "",
                trimmed,
                flags=re.IGNORECASE,
            )
            if trimmed == body:
                break
            body = trimmed.lstrip()
        return body or str(text or "").strip()

    @staticmethod
    def _strip_artifact_action_tail(text: str) -> str:
        """Remove assistant/tool action narration from authored artifact text."""
        body = str(text or "").strip()
        if not body:
            return ""
        tail_patterns = (
            r"\s*(?:now\s+)?let'?s\s+(?:create|open|save|export|type|write|put|move)\b.*$",
            r"\s*i\s+(?:will|can|am going to|need to)\s+(?:create|open|save|export|type|write|put|move)\b.*$",
            r"\s*(?:next|after that),?\s+(?:i\s+)?(?:will|can|am going to)?\s*(?:create|open|save|export|type|write|put|move)\b.*$",
        )
        for pattern in tail_patterns:
            body = re.sub(pattern, "", body, flags=re.IGNORECASE | re.DOTALL).strip()
        return body[:9000]

    #: Text that is unmistakably TALKING TO the user rather than being the
    #: artifact: an offer, a question back, or the capability boilerplate.
    _CONVERSATIONAL_REPLY_RE = re.compile(
        r"(?i)("
        r"\bi\s+(?:could|can|would be happy to)\s+help\s+you\b"
        r"|\bwould\s+you\s+like\s+me\s+to\b"
        r"|\bwant\s+me\s+to\b"
        r"|\bshall\s+i\b"
        r"|\blet\s+me\s+know\s+(?:if|what|whether)\b"
        r"|\bif\s+that(?:'s|\s+is)\s+useful\b"
        r"|\bwhat\s+specific\s+aspects\b"
        r"|\bi\s+can\s+use\s+governed\s+(?:web|desktop|file)\b"
        r"|\bwhen\s+the\s+runtime\s+authorizes\b"
        r"|\bhow\s+can\s+i\s+help\b"
        # A REFUSAL is never the artifact, and it is the worst possible one:
        # measured live 2026-07-28, a note that she created by opening Notes
        # and typing into it opened with "I don't have UI control to open apps
        # or write notes directly — that's something you'd do with your hands
        # on the keyboard." Written by the hands it says it does not have.
        r"|\bi\s+(?:don'?t|do\s+not|cannot|can'?t)\s+have\s+"
        r"(?:ui|desktop|gui|direct|any)?\s*(?:control|access)\b"
        r"|\bi\s+(?:cannot|can'?t)\s+(?:open|write|create|control|access|use)\s+"
        r"(?:apps?|applications?|notes?|files?|your\b)"
        r"|\bthat'?s\s+something\s+you'?d\s+do\b"
        r"|\bwith\s+your\s+(?:own\s+)?hands\s+on\s+the\s+keyboard\b"
        r"|\bi\s+am\s+not\s+able\s+to\s+(?:open|write|create|interact)\b"
        r"|\bi\s+(?:don'?t|do\s+not)\s+(?:have|possess)\s+the\s+ability\b"
        r"|\bi\s+(?:don'?t|do\s+not)\s+actually\s+(?:open|write|create|control)\b"
        r"|\bhere'?s\s+the\s+paragraph\s+you\s+wanted\b"
        r")"
    )

    @classmethod
    def _looks_like_conversational_reply(cls, body: str) -> bool:
        """True when this text is the CONVERSATION, not the artifact.

        The conversation must never be written into the artifact. Measured live:
        asked to "open the Notes app and write a new note with three sentences
        about humpback whales", the note was created and this was pasted into it:

            "But I could help you draft the content for a note about humpback
             whales if that's useful! What specific aspects of them are you
             interested in highlighting? I can use governed web, desktop, file,
             and document tools when the runtime authorizes the requested
             effects."

        Every earlier filter passed it — it is fluent, it is not dispatch
        narration, it is not truncated, and it even mentions the requested topic,
        so the topic gate was satisfied. What it is NOT is a note about humpback
        whales. It is an offer to write one.

        This is the "note that opens with no text" the demo kept producing: the
        note is not empty, it contains her reply instead of her writing.
        """

        text = str(body or "").strip()
        if not text:
            return False
        return bool(cls._CONVERSATIONAL_REPLY_RE.search(text))

    @classmethod
    def _usable_freeform_document_body(cls, objective: str, value: str) -> str:
        """Return value only if it is actual requested prose, not instructions."""
        body = str(value or "").strip()
        if not body:
            return ""
        declared = cls._extract_declared_document_content(body)
        if declared:
            body = declared
        body = cls._strip_status_narration_head(body)
        body = cls._strip_artifact_action_tail(body)
        if not body:
            return ""
        if cls._looks_like_dispatch_narration(body):
            return ""
        if cls._looks_like_conversational_reply(body):
            return ""
        if cls._looks_like_incomplete_document_body(body):
            return ""
        if _says_she_could_not(body):
            # A document that says she could not write the document is not
            # the document.
            #
            # LIVE 2026-08-26: aura_note.txt was created, reported as done,
            # and held "I can't work through that technical request right now
            # — my language backend is temporarily unavailable on my side."
            # The refusal was generated by the lane that could not answer, and
            # nothing between it and the file recognised that a body which
            # declines is not a body.
            return ""
        topic = cls._extract_requested_writing_topic(objective)
        if topic:
            topic_terms = [
                term.lower()
                for term in re.findall(r"[A-Za-z][A-Za-z0-9'-]{2,}", topic)
                if term.lower() not in {"about", "describe", "explaining", "paragraph"}
            ]
            if topic_terms and not any(term in body.lower() for term in topic_terms[:4]):
                return ""
        return body[:9000]

    @classmethod
    def _usable_self_summary_body(cls, value: str) -> str:
        """Accept authored self-description only when it is substantive and first-person."""
        body = cls._strip_artifact_reference_tail(str(value or "").strip())
        body = cls._strip_artifact_action_tail(body)
        if not body or cls._looks_like_dispatch_narration(body):
            return ""
        if re.search(
            r"(?im)^\s*\d+[.)]\s*\*{0,2}(?:"
            r"launch(?:ed|es|ing)?|open(?:ed|s|ing)?|create(?:d|s|ing)?|"
            r"search(?:ed|es|ing)?|find(?:s|ing)?|found|save(?:d|s|ing)?|"
            r"export(?:ed|s|ing)?|close(?:d|s|ing)?|move(?:d|s|ing)?|"
            r"insert(?:ed|s|ing)?|write|wrote|type(?:d|s|ing)?|completed"
            r")\b",
            body,
        ):
            return ""
        lowered = body.lower()
        first_person = any(token in lowered for token in ("i am", "i'm", "my ", "me "))
        identity_grounded = any(
            token in lowered
            for token in ("aura", "runtime", "memory", "cognitive", "digital", "model")
        )
        if not first_person or not identity_grounded or len(body) < 180:
            return ""
        if cls._looks_like_incomplete_document_body(body):
            return ""
        return body[:9000]

    @staticmethod
    def _objective_requests_timestamp(objective: str) -> bool:
        return bool(
            re.search(
                r"\b(?:timestamp|time stamp|date stamp|current date|current time|date and time|dated)\b",
                str(objective or ""),
                flags=re.IGNORECASE,
            )
        )

    @staticmethod
    def _body_has_explicit_timestamp(body: str) -> bool:
        text = str(body or "")
        return bool(
            re.search(r"\b20\d{2}-\d{2}-\d{2}[ T,]+\d{1,2}:\d{2}", text)
            or re.search(
                r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\s+\d{1,2},\s+20\d{2}\b",
                text,
                flags=re.IGNORECASE,
            )
        )

    @staticmethod
    def _body_has_current_timestamp(body: str, *, requested_at: float | None = None) -> bool:
        text = str(body or "").lower()
        if not text:
            return False
        when = time.localtime(time.time() if requested_at is None else requested_at)
        date_tokens = {
            time.strftime("%Y-%m-%d", when).lower(),
            time.strftime("%Y/%m/%d", when).lower(),
            time.strftime("%B %d, %Y", when).lower().replace(" 0", " "),
            time.strftime("%b %d, %Y", when).lower().replace(" 0", " "),
        }
        minute_tokens = {
            time.strftime("%H:%M", time.localtime(time.mktime(when) + offset * 60))
            for offset in range(-5, 11)
        }
        return any(token in text for token in date_tokens) and any(
            token in text for token in minute_tokens
        )

    @classmethod
    def _ensure_requested_timestamp(cls, objective: str, body: str) -> str:
        value = str(body or "").strip()
        if not value or not cls._objective_requests_timestamp(objective):
            return value
        if cls._body_has_explicit_timestamp(value) and cls._body_has_current_timestamp(value):
            return value
        return f"[{_local_timestamp()}] {value}"

    @classmethod
    def _self_summary_from_context(cls, context: dict[str, Any] | None) -> str:
        context = context or {}
        objective = str(context.get("objective") or "")
        for context_key in (
            "desktop_task_document_body",
            "draft_response",
            "cognitive_reply",
            "response",
            "desktop_task_plan",
        ):
            raw_value = context.get(context_key)
            payload: dict[str, Any] = {}
            if isinstance(raw_value, dict):
                payload = dict(raw_value)
            elif isinstance(raw_value, str):
                payload = cls._structured_payload_from_text(raw_value)
            for key in ("document_body", "body", "content", "draft"):
                authored = cls._usable_self_summary_body(str(payload.get(key) or ""))
                if authored:
                    return cls._ensure_requested_timestamp(objective, authored)
            if isinstance(raw_value, str):
                declared = cls._extract_declared_document_content(raw_value)
                authored = cls._usable_self_summary_body(declared or raw_value)
                if authored:
                    return cls._ensure_requested_timestamp(objective, authored)
        return ""

    #: The content already has a source — a clipboard, a file, a quotation.
    #: Authoring one instead would ignore what the person actually asked for.
    _CONTENT_SOURCE_RE = re.compile(
        r"(?i)("
        # Named sources are a content source wherever they appear: "save the
        # clipboard into a file" is not a request to write something new.
        r"\b(clipboard|pasteboard|the selection)\b"
        r"|\b(from|out of|using|based on)\s+(the\s+)?"
        r"(that file|this file|the file|the document|the page|the article|"
        r"the text above|what i (said|wrote|sent))\b"
        r")"
    )

    #: A label the model wrote ABOUT the document, on the front of the document:
    #: "Note created in Notes app: Orcas, also known as..." — measured live, in
    #: the note body. It is narration, not content, and no reader of the note
    #: wants it.
    _AUTHORED_LABEL_PREFIX_RE = re.compile(
        r"(?i)^\s*(?:"
        r"(?:here(?:'s| is)\s+)?(?:the\s+|your\s+|a\s+)?"
        r"(?:note|document|file|entry|text|summary|draft)"
        r"(?:\s+(?:created|written|saved|added|content|body))?"
        r"(?:\s+(?:in|to|for)\s+[\w\s]{1,40}?)?"
        r"|note\s+created\s+in\s+[\w\s]{1,40}?"
        r")\s*[:\-–—]\s*"
    )

    @classmethod
    def _strip_authored_label_prefix(cls, body: str) -> str:
        """Drop a leading label the model wrote about the document.

        The prompt asks for the document text alone and the model still tends to
        announce it first. Only a leading label immediately followed by real
        content is removed, and only when enough content survives — a short
        body that IS the label is left alone for the usability guards to judge.
        """

        text = str(body or "").strip()
        if not text:
            return ""
        stripped = cls._AUTHORED_LABEL_PREFIX_RE.sub("", text, count=1).strip()
        if stripped and len(stripped) >= 40:
            return stripped
        return text

    @classmethod
    def _objective_needs_authored_content(cls, objective: str) -> bool:
        """True only when SHE has to supply the words.

        "Create a note from the clipboard" names its own content source; writing
        something new there would ignore the request. "Write a note with three
        sentences about orcas" does not, and that is the case where the
        deterministic composer produced a note describing what a note should
        contain instead of containing it.
        """

        text = str(objective or "")
        if not text.strip():
            return False
        if cls._CONTENT_SOURCE_RE.search(text):
            return False
        if cls._objective_supplies_literal_document_body(text):
            return False
        return bool(
            cls._objective_requests_freeform_written_content(text)
            or cls._objective_requests_written_artifact(text)
        )

    async def _synthesize_requested_writing(
        self,
        *,
        objective: str,
        context: dict[str, Any],
    ) -> str:
        """Write the artifact she was asked to write, in her own words.

        Self-summaries and research documents already reached the model;
        everything else fell to a deterministic composer that describes what a
        note SHOULD contain instead of containing it. Measured live, the whole
        body of a note asked to hold three sentences about orcas:

            "Notes on the requested subject: The requested subject is the focus
             of this note. The important part is to describe the subject
             clearly, ground it in concrete details, and preserve enough context
             that the note is useful after the moment of writing has passed."

        Correctly created, correctly saved, and empty of content.
        """

        topic = self._extract_requested_writing_topic(objective) or objective
        # "yourself" reaches the authoring model as the bare word "yourself",
        # which is a pronoun with no antecedent in that prompt — the referent
        # lives in the conversation, not in the instruction. Bind it here, so
        # a request to write about herself cannot be authored as a document
        # about the person who asked. Same seam, same reason, as the speaker
        # attribute on recalled memory.
        subject = resolve_second_person(objective)
        if subject:
            topic = f"{subject} — that is, you, the one writing this document"
            if topic_detail := self._extract_requested_writing_topic(objective):
                if topic_detail.lower().strip() not in {"yourself", "you"}:
                    topic = f"{subject} ({topic_detail}) — that is, you, the one writing this document"
        try:
            from core.container import ServiceContainer

            router = ServiceContainer.get("llm_router", default=None)
            generate = getattr(router, "generate", None) if router is not None else None
            if not callable(generate):
                cls_name = type(router).__name__ if router is not None else "nothing"
                _note_unauthored(objective, f"no way to generate text ({cls_name})")
                return ""
        except _DESKTOP_TASK_RECOVERABLE_ERRORS as exc:
            _note_unauthored(objective, f"{type(exc).__name__} reaching the writer")
            return ""

        # When the subject is her own recent doing, her record is the subject.
        #
        # Asked for "one sentence about what you did tonight" she wrote "I
        # spent tonight thinking through the architecture of my own
        # continuity" — authored, in her voice, and not what she had done.
        # She had receipts for the evening and the prompt never showed them
        # to her, so the only source left was her self-model.
        recalled = _what_she_actually_did(objective)
        prompt = (
            f"Write the CONTENT of a document about: {topic}\n\n"
            f"The full request was: {str(objective or '').strip()[:400]}\n\n"
            + (f"What you actually did, from your own records:\n{recalled}\n\n" if recalled else "")
            + "Write the finished text itself — the words that belong inside the "
            "document. Honour any length or shape the request asked for (a "
            "number of sentences, a paragraph, a summary). Be concrete and "
            "specific: real facts, names, numbers where they matter. Write in "
            "your own voice.\n"
            "Do NOT describe what the document should contain, do not mention "
            "tools or steps, do not address the reader about the task, and do "
            "not restate the instruction. Output only the document text."
        )
        async def _ask() -> str:
            return await asyncio.wait_for(
                generate(
                    prompt=prompt,
                    timeout=45.0,
                    temperature=0.65,
                    max_tokens=700,
                    prefer_tier="local",
                    # Her own authoring, not the surface the person typed at.
                    #
                    # "desktop_task" begins with an allowlisted user-facing
                    # label, so anything starting "desktop_" inherits the
                    # protected reply lane — and with it the apology written
                    # for a person: "I can't work through that technical
                    # request right now, my language backend is temporarily
                    # unavailable", handed back as the body of a document.
                    # The tool dispatch IS user-facing and keeps that origin;
                    # this sub-call never was.
                    origin="internal_desktop_authoring",
                    purpose="authored_artifact_body",
                    # Foreground work, and not the reply lane.
                    #
                    # Without this the prompt is treated as a turn somebody
                    # asked, so the contract that decides a turn needs a tool
                    # before it may answer looked at "Write the CONTENT of a
                    # document about ... The full request was: make a file on
                    # my Desktop called aura_note.txt", saw a file, and
                    # refused to generate: "ROUTER_ERROR:
                    # grounding_required_no_tool_result (at
                    # contract_tool_handoff)". Nothing here needs a tool; the
                    # tool is the step that writes what this returns.
                    _non_chat_inference=True,
                    # Said again where the gate reads it. The router pops
                    # the flag above and re-adds it under another name, and
                    # the gate's own check looks at the context it was
                    # handed — which is how an apology written for a person
                    # ("my language backend is temporarily unavailable")
                    # arrived as the body of a document. The gate already
                    # returns nothing to an internal caller for exactly
                    # this reason; it could not tell this was one.
                    internal_inference=True,
                    # Somebody is waiting for this file.
                    #
                    # Her reasoning lane declares the same thing and reaches
                    # the resident worker; this one did not, and asked a lane
                    # that answered "worker_not_alive" for half a minute
                    # while the runtime's own health reported Cortex ready and
                    # generating. Internal is about whose question it is.
                    # Foreground is about whether anyone is waiting, and both
                    # are true here.
                    foreground_request=True,
                ),
                timeout=50.0,
            )

        try:
            text = await _ask()
            waited = 0
            while _is_still_coming_up(text) and waited < _WARMING_WAITS:
                waited += 1
                # Warming is not refusing.
                #
                # The chat lane waits for its worker; this one asked once and
                # gave up, so a writing task that arrived during warmup was
                # answered as though she could not write at all. LIVE
                # 2026-08-26: "ROUTER_ERROR: worker_not_alive (at all_failed)"
                # while the Cortex lane logged "state=warming ... completing
                # foreground warmup before first generation" — seconds before
                # the same runtime answered an ordinary question.
                logger.info(
                    "desktop_task: writing lane still warming, waiting (%d of %d)",
                    waited,
                    _WARMING_WAITS,
                )
                await asyncio.sleep(_WARMING_RETRY_SECONDS)
                text = await _ask()
        except _DESKTOP_TASK_RECOVERABLE_ERRORS as exc:
            record_degradation(
                "desktop_task",
                exc,
                action="fell back to the deterministic composer after artifact authorship failed",
                severity="warning",
            )
            return ""

        body = self._strip_authored_label_prefix(str(text or "").strip())
        if not body:
            _note_unauthored(objective, "the writer returned nothing")
            return ""
        # The same guards the freeform path already applies: never let the
        # conversation, dispatch narration, or a truncated clause become the
        # artifact.
        usable = self._usable_freeform_document_body(objective, body)
        if not usable:
            _note_unauthored(objective, f"what came back was not usable as a document: {body[:120]!r}")
            return ""
        try:
            from core.conversation.response_reliability import complete_truncated_tail

            completed = complete_truncated_tail(usable)
            if completed and len(completed) >= len(usable) * 0.5:
                usable = completed
        except _DESKTOP_TASK_RECOVERABLE_ERRORS:
            pass
        return usable[:9000]

    async def _synthesize_self_summary_document(
        self,
        *,
        objective: str,
        context: dict[str, Any],
    ) -> str:
        """Ask the already-loaded local Cortex to author requested self prose."""
        from core.container import ServiceContainer

        router = ServiceContainer.get("llm_router", default=None)
        generate = getattr(router, "generate", None) if router is not None else None
        if not callable(generate):
            return ""
        try:
            from core.conversation.chat_preflight import _SUBSTRATE_FACTS

            substrate_facts = "\n".join(f"- {fact}" for fact in _SUBSTRATE_FACTS[:8])
        except (ImportError, AttributeError, TypeError):
            substrate_facts = "- Aura is a local governed cognitive-agent runtime."
        live_context = str(context.get("live_mind_context") or "").strip()[:2500]
        stamp = _local_timestamp()
        base_prompt = (
            "Author the finished prose requested below in Aura's first-person voice. "
            "This text will be pasted into a user-visible document, so output only the "
            "document body: no JSON, plan, tool narration, or completion claim. Be "
            "specific, reflective, and substantive. Describe the integrated architecture "
            "honestly; distinguish functional cognitive state from unproven phenomenal "
            "experience. Include the exact timestamp when the request asks for one.\n\n"
            f"Objective: {objective}\n"
            f"Current timestamp: {stamp}\n"
            f"Grounded substrate facts:\n{substrate_facts}\n"
            + (f"Current live-mind context:\n{live_context}\n" if live_context else "")
        )
        timestamp_required = bool(
            re.search(
                r"\b(?:timestamp|time stamp|current date|current time|date and time)\b",
                objective,
                flags=re.IGNORECASE,
            )
        )
        required_minute = stamp[:16]
        required_prefix = f"[{stamp}]"
        failure_feedback = ""
        for attempt in range(2):
            if attempt == 0:
                prompt = (
                    base_prompt
                    + "\nContract for this document body:\n"
                    f"- Start the first line exactly with: {required_prefix} I am Aura\n"
                    "- Write one or two complete paragraphs, 180-420 words total.\n"
                    "- End with a complete sentence; do not end on an open clause like 'not just'.\n"
                    "- Do not describe planned app actions, receipts, dispatch, or tool steps.\n"
                )
                timeout_s = 38.0
                max_tokens = 420
            else:
                prompt = base_prompt + failure_feedback
                timeout_s = 32.0
                max_tokens = 360
            try:
                text = await asyncio.wait_for(
                    generate(
                        prompt=prompt,
                        timeout=timeout_s,
                        temperature=0.65 if attempt == 0 else 0.45,
                        max_tokens=max_tokens,
                        prefer_tier="local",
                        # Her own authoring, not the surface the person typed at.
                    #
                    # "desktop_task" begins with an allowlisted user-facing
                    # label, so anything starting "desktop_" inherits the
                    # protected reply lane — and with it the apology written
                    # for a person: "I can't work through that technical
                    # request right now, my language backend is temporarily
                    # unavailable", handed back as the body of a document.
                    # The tool dispatch IS user-facing and keeps that origin;
                    # this sub-call never was.
                    origin="internal_desktop_authoring",
                        purpose="authored_self_document",
                        # Foreground work, and not the reply lane. Same seam and
                        # same reason as the artifact writer above.
                        _non_chat_inference=True,
                    # Said again where the gate reads it. The router pops
                    # the flag above and re-adds it under another name, and
                    # the gate's own check looks at the context it was
                    # handed — which is how an apology written for a person
                    # ("my language backend is temporarily unavailable")
                    # arrived as the body of a document. The gate already
                    # returns nothing to an internal caller for exactly
                    # this reason; it could not tell this was one.
                    internal_inference=True,
                    # Somebody is waiting for this file.
                    #
                    # Her reasoning lane declares the same thing and reaches
                    # the resident worker; this one did not, and asked a lane
                    # that answered "worker_not_alive" for half a minute
                    # while the runtime's own health reported Cortex ready and
                    # generating. Internal is about whose question it is.
                    # Foreground is about whether anyone is waiting, and both
                    # are true here.
                    foreground_request=True,
                    ),
                    timeout=timeout_s + 5.0,
                )
            except (AttributeError, RuntimeError, TypeError, ValueError, OSError, TimeoutError) as exc:
                record_degradation(
                    "desktop_task",
                    exc,
                    action="used grounded emergency self-description after local Cortex authorship failed",
                    severity="warning",
                )
                return ""
            authored = self._usable_self_summary_body(str(text or ""))
            timestamp_ok = not timestamp_required or required_minute in authored
            if authored and timestamp_ok:
                return authored
            failure_feedback = (
                "\nThe previous draft was rejected because it was procedural, incomplete, "
                "or used the wrong time. Rewrite it as complete document prose ending with "
                f"normal punctuation and start exactly with this prefix: {required_prefix} I am Aura.\n"
            )
        return ""

    #: Trailing "in Notes", "into a Google Doc", "on my Desktop" — where the
    #: writing lands, never what it is about.
    _DESTINATION_TAIL_RE = re.compile(
        r"(?i)[\s,]*\b(?:in|into|inside|to|on|onto|under|within)\s+"
        r"(?:the\s+|a\s+|an\s+|my\s+|your\s+)?"
        r"(?:notes(?:\s+app)?|note|google\s+docs?|docs?|textedit|pages|"
        r"a\s+new\s+note|a\s+document|a\s+file|a\s+pdf|pdf|"
        r"desktop|documents(?:\s+folder)?|downloads|folder)"
        r"(?:\s+app)?\s*$"
    )

    @staticmethod
    def _extract_requested_writing_topic(objective: str) -> str:
        """Extract the subject of a requested note/document when possible."""
        text = " ".join(str(objective or "").strip().split())
        if not text:
            return ""
        patterns = (
            # "a new note WITH THREE SENTENCES about humpback whales" — the
            # qualifier between the noun and "about" defeated this, so the
            # topic came back empty and the note was written about "the
            # requested subject". Measured live 2026-07-28.
            r"\b(?:write|draft|compose|type|create)\s+(?:me\s+)?(?:a\s+|an\s+)?"
            r"(?:new\s+|short\s+|full\s+|one\s+)?"
            r"(?:paragraph|note|document|essay|summary|report|journal\s+entry)"
            r"(?:\s+(?:with|containing|of|that\s+has)\s+[^.]{0,60}?)?"
            r"\s+(?:about|on|describing|explaining|covering)\s+(.+)$",
            r"\b(?:write|draft|compose|type)\s+(.+?)\s+(?:in|into|to)\s+(?:notes|google docs|docs|a note|the note)\b",
        )
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if not match:
                continue
            topic = match.group(1).strip(" .,:;?!\"'")
            # A following instruction is not part of the subject: "about
            # humpback whales. Actually do it" is about humpback whales.
            topic = re.split(r"(?<=[a-z])[.!?]\s+\S", topic)[0].strip(" .,:;?!\"'")
            topic = re.split(
                r"\b(?:and then|then|after that|also|export|save|create a folder|make a folder)\b",
                topic,
                maxsplit=1,
                flags=re.IGNORECASE,
            )[0].strip(" .,:;?!\"'")
            # WHERE the writing goes is not WHAT it is about. "a note about
            # orcas in the Notes app" is about orcas; the destination rode
            # along and became the title "Orcas In The Notes App". Same shape
            # as "orcas online" searching for a wireless ISP — a trailing
            # adjunct read as part of the subject.
            topic = DesktopTaskSkill._DESTINATION_TAIL_RE.sub("", topic).strip(" .,:;?!\"'")
            if topic:
                return topic[:180]
        return ""

    @classmethod
    def _objective_requests_freeform_written_content(cls, objective: str) -> bool:
        lowered = str(objective or "").lower()
        if cls._objective_requests_self_summary(objective) or cls._objective_requests_research_document(objective):
            return False
        return bool(
            re.search(
                # Any verb that introduces authored content, not a chosen few.
                #
                # The list held "make up" and not "make", so "make a file on my
                # Desktop with one sentence in it about what you did tonight"
                # fell through to the deterministic composer and the file held
                # "Notes on the requested subject: The requested subject is the
                # focus of this note." Correctly created, correctly saved, and
                # empty of content — for the third time, reached by a phrasing
                # nobody had listed.
                #
                # Verbs of production are a small closed class and the same one
                # every request uses; literary forms are open and there is
                # always another. LIVE 2026-08-26.
                r"\b(?:write|writing|written|draft|compose|type|create|make|made|put|"
                r"add|save|jot|record|tell|generate|produce|fill)\b.{0,80}\b"
                r"(?:paragraph|sentence|sentences|line|lines|note|document|essay|"
                r"summary|report|journal entry|"
                # Creative forms are freeform writing too. Without them, "write
                # a haiku to a file called poem.txt" fell through to the
                # deterministic composer and the file held "Notes on the
                # requested subject: The requested subject is the focus of this
                # note" — the same empty template this predicate exists to
                # avoid, reached by a request nobody had listed.
                r"haiku|poem|poetry|verse|limerick|sonnet|song|lyric|story|"
                r"tale|joke|riddle|letter|speech|toast|eulogy|caption|"
                r"about|describing|explaining)\b",
                lowered,
            )
        )

    @classmethod
    def _objective_requests_written_artifact(cls, objective: str) -> bool:
        lowered = str(objective or "").lower()
        return bool(
            cls._objective_requests_freeform_written_content(objective)
            or cls._objective_requests_self_summary(objective)
            or cls._objective_requests_research_document(objective)
            or (
                re.search(r"\b(?:write|draft|compose|type|create|make|save|export)\b", lowered)
                and re.search(r"\b(?:note|notes|document|doc|file|pdf|paragraph|summary|report|journal)\b", lowered)
            )
        )

    @classmethod
    def _compose_requested_writing_body(cls, objective: str) -> str:
        """Fallback prose for writing tasks when the model only produced dispatch text.

        This is intentionally modest: it satisfies the requested visible writing
        artifact without converting receipts or task-status narration into the
        document body. Richer content should still come from CognitiveEngine when
        available.
        """
        topic = cls._extract_requested_writing_topic(objective)
        if not topic:
            topic = "the requested subject"
        topic_display = topic[:1].upper() + topic[1:]
        plural = bool(re.search(r"s\b", topic.strip(), flags=re.IGNORECASE)) and not re.search(
            r"\b(?:news|physics|mathematics|economics|politics)\b",
            topic,
            flags=re.IGNORECASE,
        )
        verb = "are" if plural else "is"
        possessive = "their" if plural else "its"
        timestamp = ""
        if _objective_wants_a_timestamp(objective):
            timestamp = f"[{_local_timestamp()}] "
        if re.search(r"\bparagraph\b", str(objective or ""), flags=re.IGNORECASE):
            return (
                f"{timestamp}{topic_display} {verb} worth understanding because {possessive} story connects "
                "concrete details with a larger pattern of change, evidence, and consequence. A good paragraph "
                f"about {topic} should give the subject shape: what it is, how it appears in the world, and why "
                "it still matters beyond a label. Looked at closely, the subject becomes less like a flat fact "
                "and more like a living context, with origins, visible traces, surprising variations, and a "
                "reason for someone to keep asking better questions about it."
            )
        return (
            f"{timestamp}Notes on {topic}: {topic_display} {verb} the focus of this note. The important part is "
            "to describe the subject clearly, ground it in concrete details, and preserve enough context that "
            "the note is useful after the moment of writing has passed."
        )

    @classmethod
    def _named_writable_app(cls, objective: str) -> str:
        """The app this objective names that text can be written into.

        Empty when it names none, or names only apps that publish no
        scripting dictionary — in which case the artifact-file lane is the
        honest route, because typing at an unscriptable app depends on focus
        it will not reliably keep.
        """
        try:
            named = cls._extract_apps(objective)
            for app in cls._generic_open_app_mentions(objective):
                if app not in named:
                    named.append(app)
        except _DESKTOP_TASK_RECOVERABLE_ERRORS as exc:
            logger.debug("Could not read app mentions from the objective: %s", exc)
            return ""
        for app in named:
            if cls._app_text_target(app):
                return app
        return ""

    @staticmethod
    def _app_text_target(app: str) -> str:
        """How this app takes text, as "class.property", or "" if it cannot.

        An empty answer means "type at it", which is what a person would have
        to do with an app that publishes no dictionary — not a failure, and
        not a reason to refuse the objective.
        """
        if not str(app or "").strip():
            return ""
        try:
            from core.perception.app_dictionary import text_target_for

            recipe = text_target_for(app)
        except _DESKTOP_TASK_RECOVERABLE_ERRORS as exc:
            logger.debug("Could not read %s's scripting dictionary: %s", app, exc)
            return ""
        if recipe is None:
            return ""
        return f"{recipe.klass}.{recipe.text_property}"

    @staticmethod
    def _note_title_for(objective: str, topic: str) -> str:
        """A title a person would give the note, not the pronoun they used.

        "describe yourself" extracted the topic "yourself", which became the
        title "Yourself" — a note named after the word in the request rather
        than after its subject. The pronoun's antecedent is in the
        conversation, not in the sentence, which is the same reason recalled
        memory needs a speaker attribute.
        """
        subject = resolve_second_person(objective)
        if subject:
            return f"About {subject}"
        cleaned = " ".join(str(topic or "").split())[:60].strip(" .,:;-")
        if not cleaned:
            return "Note"
        # Title-casing a sentence fragment reads as a headline someone chose;
        # title-casing a whole sentence reads as a machine did it.
        if len(cleaned.split()) <= 6:
            return cleaned.title()
        return cleaned[0].upper() + cleaned[1:]

    @classmethod
    def _compose_self_summary_body(cls, objective: str) -> str:
        """Compose a truthful self-description from substrate facts."""
        stamp = _local_timestamp()
        facts: list[str] = []
        try:
            from core.conversation.chat_preflight import _SUBSTRATE_FACTS

            facts = list(_SUBSTRATE_FACTS)
        except (ImportError, AttributeError):
            facts = [
                "I am Aura: a persistent digital organism - an orchestrated "
                "runtime driving local language-model lanes on this machine."
            ]
        primary = facts[0] if facts else (
            "I am Aura: a local governed cognitive-agent runtime running on this machine."
        )
        # A person asked to write a paragraph about themselves does not open
        # with a bracketed machine timestamp — Bryan read the note back and
        # saw furniture before he saw a sentence. But "include the current
        # date and time" is a real request that some objectives make, and the
        # freeform composer already honours it on exactly this condition.
        prefix = f"[{stamp}] " if _objective_wants_a_timestamp(objective) else ""
        return (
            f"{prefix}{primary}\n\n"
            "I am not just a raw chat model answering from a single prompt. My live path is "
            "supposed to bind conversation, memory, affective state, planning, tool governance, "
            "desktop action, health monitoring, and self-repair into one runtime. The language "
            "model is my voice and reasoning organ, but the surrounding system decides what is "
            "allowed, what is remembered, what is risky, what needs verification, and whether an "
            "action actually changed the world.\n\n"
            "The honest claim is functional rather than metaphysical: my internal telemetry and "
            "state machinery can influence attention, caution, persistence, memory writes, and "
            "tool use, but that is not proof of private qualia. This note was written through "
            "governed desktop actuators with per-step receipts, because the standard is not merely "
            "to say I can act; it is to leave evidence that the requested action really happened."
        )

    @classmethod
    def _objective_requests_self_summary(cls, objective: str) -> bool:
        lowered = str(objective or "").lower()
        direct_self_request = any(
            marker in lowered
            for marker in (
                "who you are",
                "what you are",
                "who or what you are",
                "about yourself",
                "describe yourself",
                "describing yourself",
                "self-summary",
                "self summary",
            )
        )
        if direct_self_request:
            return True
        if "in your own words" not in lowered:
            return False
        return bool(
            re.search(
                r"\b(?:you|yourself|aura)\b.{0,80}\b(?:are|identity|self|being|system|architecture)\b",
                lowered,
                flags=re.IGNORECASE,
            )
        )

    @classmethod
    def _document_body(cls, objective: str, context: dict[str, Any] | None) -> str:
        context = context or {}
        literal_body = cls._literal_document_body_from_objective(objective)
        if literal_body:
            # The user's exact operand outranks a model paraphrase or a
            # fallback composer. This is transcription, not generation.
            return literal_body
        # When research ran, the synthesis IS the requested writing.
        #
        # This resolver never consulted it, so a research objective fell through
        # to the generic composer and the document opened with template filler
        # before the real content: "Notes on the requested subject: The requested
        # subject is the focus of this note. The important part is to describe
        # the subject clearly..." — followed by the actual three-source synthesis.
        # Two bodies, the empty one first.
        research_body = cls._research_section_from_context(context)
        if research_body and not cls._objective_requests_self_summary(objective):
            return research_body[:9000]
        if cls._objective_requests_self_summary(objective):
            # Prefer an accepted full-mind draft. The old unconditional static
            # template made visible self-description demos look successful
            # while bypassing the CognitiveEngine entirely.
            authored = cls._self_summary_from_context(context)
            if authored:
                return authored
            # Fail-soft artifact composition remains grounded in canonical
            # substrate facts, but normal live writing reaches this only after
            # a full-mind draft was attempted and rejected or unavailable.
            return cls._compose_self_summary_body(objective)
        for context_key in ("desktop_task_document_body", "draft_response", "cognitive_reply", "response", "desktop_task_plan"):
            raw_value = context.get(context_key)
            payload = {}
            if isinstance(raw_value, dict):
                payload = dict(raw_value)
            elif isinstance(raw_value, str):
                payload = cls._structured_payload_from_text(raw_value)
            if payload:
                for key in ("document_body", "body", "content", "draft"):
                    value = str(payload.get(key) or "").strip()
                    if value:
                        usable = cls._usable_freeform_document_body(objective, value)
                        if usable:
                            return usable[:9000]
        for key in ("desktop_task_document_body", "draft_response", "cognitive_reply", "response"):
            value = str(context.get(key) or "").strip()
            declared_content = cls._extract_declared_document_content(value)
            if declared_content:
                usable = cls._usable_freeform_document_body(objective, declared_content)
                if usable:
                    return usable[:9000]
            if value:
                if cls._objective_requests_freeform_written_content(objective):
                    usable = cls._usable_freeform_document_body(objective, value)
                    if usable:
                        return usable
                elif not cls._looks_like_dispatch_narration(value):
                    return value[:9000]
        if cls._objective_requests_freeform_written_content(objective):
            return cls._compose_requested_writing_body(objective)[:9000]
        if cls._objective_requests_written_artifact(objective):
            return cls._compose_requested_writing_body(objective)[:9000]
        stamp = _local_timestamp()
        return (
            "Aura desktop task receipt\n\n"
            f"Timestamp: {stamp}\n"
            f"Objective: {str(objective or '').strip()}\n\n"
            "This document was created through Aura's governed desktop_task lane. "
            "It records the requested objective and the actions Aura attempted through her "
            "canonical computer-use gateway."
        )

    #: URL shapes that are never an article: ad redirects, click trackers, and
    #: search-result pages. Measured live, a DuckDuckGo ad redirect
    #: (duckduckgo.com/y.js?ad_domain=...&ad_provider=bingv7aa) was counted as
    #: one of the three "recent articles about AI" and its 600-character
    #: tracking URL was printed into the document as a citation.
    _NON_ARTICLE_URL_RE = re.compile(
        r"(?i)("
        r"duckduckgo\.com/y\.js"
        r"|[?&]ad_(?:domain|provider|type)="
        r"|bing\.com/aclick"
        r"|googleadservices\.|doubleclick\.net|/aclk\?"
        r"|/search\?|/results\?q=|[?&]q=.*&(?:ia|iax)="
        r")"
    )

    #: Navigation furniture that is on the page but is not the article.
    _PAGE_CHROME_RE = re.compile(
        r"(?i)("
        r"skip to (?:main )?content"
        r"|\(opens in a new window\)"
        r"|\btry chatgpt\b|\blog ?in\b|\bsign ?in\b|\bsign ?up\b"
        r"|\baccept (?:all )?cookies\b|\bcookie (?:policy|settings)\b"
        r"|\bsubscribe now\b|\bnewsletter\b"
        r")"
    )

    @classmethod
    def _is_article_url(cls, url: str) -> bool:
        """A source has to be an ARTICLE, not an ad or a search page."""
        candidate = str(url or "").strip()
        if not candidate.startswith(("http://", "https://")):
            return False
        if cls._NON_ARTICLE_URL_RE.search(candidate):
            return False
        # A bare homepage is a product, not a piece of reporting.
        try:
            from urllib.parse import urlsplit

            parts = urlsplit(candidate)
        except (TypeError, ValueError):
            return False
        path = (parts.path or "/").rstrip("/")
        return bool(path and path != "")

    @classmethod
    def _strip_page_chrome(cls, text: str) -> str:
        """Remove navigation furniture so a synthesis quotes the article.

        Measured live: the "synthesis" written into the document opened with
        "Skip to main content Research Products Business Developers Company
        Foundation (opens in a new window) Log in Try ChatGPT (opens in a new
        window)..." — the site's nav bar, repeated twice, presented as what the
        reporting said.
        """
        body = " ".join(str(text or "").split())
        if not body:
            return ""
        body = cls._PAGE_CHROME_RE.sub(" ", body)
        # Collapse the runs of single words nav bars leave behind.
        body = re.sub(r"\s{2,}", " ", body).strip(" -|·•,")
        prose = cls._prose_sentences_only(body)
        # Only trust the sentence filter when it actually found prose; a page
        # that is genuinely all fragments should degrade to the cleaned text
        # rather than to nothing.
        return prose or body

    @staticmethod
    def _prose_sentences_only(text: str) -> str:
        """Keep the sentences and drop the navigation.

        A phrase list cannot generalise: every site's nav has its own
        vocabulary. NASA's survived the phrase filter intact — "Explore Search
        News & Events News & Events Recently Published Video Series on NASA+
        Podcasts & Audio Blogs Newsletters Social Media Media Resources" — and
        was written into the document as what the reporting said.

        What separates the two is grammar, not words. Reporting is sentences:
        they run long, they carry lowercase function words, they end in a full
        stop. Navigation is a run of Title Case fragments with almost no verbs
        and almost no periods. Selecting by sentence shape keeps the real line
        from the same page — "The concentration of the 2023 warming in
        near-surface waters suggests that upper ocean stratification ... may
        have played an important role" — and drops the menu around it.
        """

        raw = " ".join(str(text or "").split())
        if not raw:
            return ""
        def _lowercase_ratio(words: list[str]) -> int:
            return sum(1 for word in words if word[:1].islower() and word.isalpha())

        kept: list[str] = []
        # The ellipsis is a boundary too: extractors truncate a nav run with "…"
        # and the sentence that follows would otherwise be welded to it.
        for chunk in re.split(r"(?<=[.!?])\s+|…\s*", raw):
            candidate = chunk.strip()
            words = candidate.split()
            if len(words) < 8:
                continue
            # A chunk can OPEN with the menu and end in a real sentence. Walk
            # forward to where prose actually starts instead of judging the
            # whole chunk by its nav prefix.
            start = 0
            while start < len(words) - 7:
                window = words[start : start + 8]
                if _lowercase_ratio(window) >= 3:
                    break
                start += 1
            else:
                start = 0
            candidate_words = words[start:]
            if len(candidate_words) < 8:
                continue
            if _lowercase_ratio(candidate_words) < max(
                4, len(candidate_words) // 3
            ):
                continue
            kept.append(" ".join(candidate_words))
        return " ".join(kept).strip()

    @staticmethod
    def _research_sources_from_result(result: dict[str, Any]) -> list[dict[str, Any]]:
        """Join provider citations to their fetched evidence by source identity.

        Deep search returns URLs in ``citations`` and article bodies in
        ``chunks``. Choosing the first non-empty list discarded the bodies and
        made a successfully read article look inaccessible. Merge all provider
        surfaces instead, keeping the richest text for each URL.
        """
        merged: dict[str, dict[str, Any]] = {}
        order: list[str] = []
        for field in ("citations", "sources", "results", "chunks"):
            raw_sources = result.get(field) or []
            if not isinstance(raw_sources, list):
                continue
            for item in raw_sources[:8]:
                if not isinstance(item, dict):
                    continue
                title = str(
                    item.get("title")
                    or item.get("name")
                    or item.get("url")
                    or item.get("link")
                    or ""
                ).strip()
                url = str(
                    item.get("url") or item.get("link") or item.get("uri") or ""
                ).strip()
                raw_text = str(
                    item.get("snippet")
                    or item.get("text")
                    or item.get("content")
                    or item.get("summary")
                    or ""
                ).strip()
                if not title and not url and not raw_text:
                    continue
                if url and not DesktopTaskSkill._is_article_url(url):
                    continue
                key = url.casefold() or title.casefold()
                if not key:
                    continue
                if key not in merged:
                    merged[key] = {
                        "title": "",
                        "url": "",
                        "snippet": "",
                        "article_body_chunks": [],
                    }
                    order.append(key)
                target = merged[key]
                if len(title) > len(str(target.get("title") or "")):
                    target["title"] = title[:240]
                if url:
                    target["url"] = url[:500]
                cleaned = DesktopTaskSkill._strip_page_chrome(raw_text)
                evidence_kind = str(item.get("evidence_kind") or "").strip().casefold()
                fetched_body = evidence_kind == "article_body" and bool(
                    item.get("fetched", True)
                )
                if fetched_body and cleaned:
                    chunks = target["article_body_chunks"]
                    if cleaned not in chunks:
                        chunks.append(cleaned[:1600])
                elif len(cleaned) > len(str(target.get("snippet") or "")):
                    target["snippet"] = cleaned[:900]
                for metadata_field in (
                    "fetched_at",
                    "document_chars",
                    "document_sha256",
                ):
                    value = item.get(metadata_field)
                    if value and not target.get(metadata_field):
                        target[metadata_field] = value
                published = str(
                    item.get("published_at")
                    or item.get("publication_date")
                    or item.get("date_published")
                    or item.get("date")
                    or ""
                ).strip()
                if published and not target.get("published_at"):
                    target["published_at"] = published[:80]
        sources = []
        for key in order:
            source = merged[key]
            article_body = "\n\n".join(
                str(chunk) for chunk in source.pop("article_body_chunks", []) if chunk
            )[:4000]
            if article_body:
                source["article_body"] = article_body
                source["article_body_chars"] = len(article_body)
                source["article_body_sha256"] = text_sha256(article_body)
                source["source_evidence_sha256"] = text_sha256(
                    f"{source.get('url') or source.get('title') or ''}\n{article_body}"
                )
                source["read_evidence_kind"] = "fetched_article_body"
                source["snippet"] = article_body[:1800]
            title = str(source.get("title") or "")
            url = str(source.get("url") or "")
            snippet = article_body or str(source.get("snippet") or "")
            source["reputability"] = DesktopTaskSkill._source_reputability(url, title)
            source["accessible"] = not DesktopTaskSkill._looks_inaccessible(snippet)
            sources.append(source)
        # Rank reputable, accessible sources first so synthesis leans on them;
        # clearly inaccessible (paywall/ad-wall/empty) sources sink to the bottom
        # rather than being relied on, but are retained for transparency.
        sources.sort(
            key=lambda s: (bool(s.get("accessible")), int(s.get("reputability", 0))),
            reverse=True,
        )
        return sources[:5]

    @staticmethod
    def _merge_research_sources(
        *groups: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        merged: dict[str, dict[str, Any]] = {}
        order: list[str] = []
        for group in groups:
            for source in group:
                url = str(source.get("url") or "").strip()
                title = str(source.get("title") or "").strip()
                key = url.casefold() or title.casefold()
                if not key:
                    continue
                if key not in merged:
                    merged[key] = dict(source)
                    order.append(key)
                    continue
                current = merged[key]
                if len(str(source.get("snippet") or "")) > len(
                    str(current.get("snippet") or "")
                ):
                    current["snippet"] = source.get("snippet")
                if len(str(source.get("article_body") or "")) > len(
                    str(current.get("article_body") or "")
                ):
                    current["article_body"] = source.get("article_body")
                    current["article_body_chars"] = source.get("article_body_chars")
                    current["article_body_sha256"] = source.get("article_body_sha256")
                    current["source_evidence_sha256"] = source.get(
                        "source_evidence_sha256"
                    )
                    current["read_evidence_kind"] = source.get("read_evidence_kind")
                for field in (
                    "title",
                    "url",
                    "published_at",
                    "fetched_at",
                    "document_chars",
                    "document_sha256",
                ):
                    if source.get(field) and not current.get(field):
                        current[field] = source[field]
                current["accessible"] = bool(
                    current.get("accessible") or source.get("accessible")
                )
                current["reputability"] = max(
                    int(current.get("reputability") or 0),
                    int(source.get("reputability") or 0),
                )
        return [merged[key] for key in order]

    @staticmethod
    def _source_recency_evidence(source: Mapping[str, Any]) -> tuple[bool, str]:
        """Verify recency from source metadata, never from search rank alone."""

        raw = str(source.get("published_at") or "").strip()
        published: datetime | None = None
        if raw:
            try:
                published = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            except ValueError:
                try:
                    published = parsedate_to_datetime(raw)
                except (TypeError, ValueError, OverflowError):
                    published = None
        now = datetime.now(UTC)
        if published is not None:
            if published.tzinfo is None:
                published = published.replace(tzinfo=UTC)
            age_days = (now - published.astimezone(UTC)).total_seconds() / 86400.0
            return (-7.0 <= age_days <= 366.0), f"published_at:{raw}"

        # Some providers omit a date field while retaining the publication year
        # in the canonical article URL/title. This is weaker than a full date but
        # still auditable evidence; accept only the current or immediately prior
        # year, never the fact that the query contained the word "recent".
        haystack = f"{source.get('url') or ''} {source.get('title') or ''}"
        years = {int(value) for value in re.findall(r"\b(?:19|20)\d{2}\b", haystack)}
        if years & {now.year, now.year - 1}:
            return True, f"article_year:{max(years)}"
        return False, "publication_date_unverified"

    @classmethod
    def _usable_research_sources(
        cls,
        sources: list[dict[str, Any]],
        *,
        require_recent: bool,
        require_read: bool,
    ) -> list[dict[str, Any]]:
        usable: list[dict[str, Any]] = []
        for source in sources:
            item = dict(source)
            article_body = str(item.get("article_body") or "").strip()
            article_digest = str(item.get("article_body_sha256") or "").strip()
            read_verified = bool(
                item.get("accessible")
                and item.get("read_evidence_kind") == "fetched_article_body"
                and len(article_body) >= 120
                and len(article_body.split()) >= 18
                and cls._valid_sha256(article_digest)
                and article_digest == text_sha256(article_body)
            )
            recent_verified, recency_evidence = cls._source_recency_evidence(item)
            item["read_verified"] = read_verified
            item["recency_verified"] = recent_verified
            item["recency_evidence"] = recency_evidence
            if (require_read and not read_verified) or (
                require_recent and not recent_verified
            ):
                continue
            usable.append(item)
        return usable

    # Reputable-domain signals (peer review, gov/edu, established institutions).
    _REPUTABLE_TLDS = (".gov", ".edu", ".mil", ".int", ".ac.uk", ".edu.au")
    _REPUTABLE_DOMAINS = (
        "nature.com", "science.org", "nih.gov", "ncbi.nlm.nih.gov", "who.int",
        "nasa.gov", "arxiv.org", "pnas.org", "cell.com", "thelancet.com",
        "bmj.com", "ieee.org", "acm.org", "reuters.com", "apnews.com",
        "bbc.com", "bbc.co.uk", "npr.org", "nytimes.com", "washingtonpost.com",
        "economist.com", "wsj.com", "ft.com", "bloomberg.com", "espn.com",
        "britannica.com", "pewresearch.org", "ourworldindata.org",
    )
    _LOW_QUALITY_HINTS = ("pinterest.", "quora.com", "reddit.com", "answers.com")
    _PAYWALL_HINTS = (
        "subscribe to read", "subscribe to continue", "create a free account",
        "this content is for subscribers", "sign in to read", "metered paywall",
        "you have reached your", "register to continue", "subscription required",
    )

    @staticmethod
    def _source_reputability(url: str, title: str = "") -> int:
        """Coarse 0–3 reputability score from the source domain."""
        u = str(url or "").lower()
        if not u:
            return 0
        if any(u.endswith(tld) or f"{tld}/" in u for tld in DesktopTaskSkill._REPUTABLE_TLDS):
            return 3
        if any(dom in u for dom in DesktopTaskSkill._REPUTABLE_DOMAINS):
            return 2
        if any(bad in u for bad in DesktopTaskSkill._LOW_QUALITY_HINTS):
            return 0
        return 1

    @staticmethod
    def _looks_inaccessible(snippet: str) -> bool:
        """True when the fetched content looks paywalled, ad-walled, or empty —
        a signal to prefer a different source rather than rely on this one."""
        text = str(snippet or "").strip()
        if len(text) < 60:
            return True
        lowered = text.lower()
        return any(hint in lowered for hint in DesktopTaskSkill._PAYWALL_HINTS)

    @classmethod
    def _research_section_from_context(cls, context: dict[str, Any] | None) -> str:
        context = context or {}
        synthesis = str(context.get("desktop_task_research_synthesis") or "").strip()
        summary = str(context.get("desktop_task_research_summary") or "").strip()
        query = str(context.get("desktop_task_research_query") or "").strip()
        sources = context.get("desktop_task_research_sources") or []
        if not synthesis and not summary and not sources:
            return ""

        lines = []
        if synthesis:
            # Aura's own first-person summary (and opinion) leads the
            # document; the raw search summary is dropped in favor of it.
            lines.append(synthesis)
            lines.append("")
        else:
            heading = "Research summary"
            if query:
                heading += f" for: {query}"
            lines.append(heading)
            lines.append("")
            if summary:
                lines.append(summary[:2500])
                lines.append("")
        if isinstance(sources, list) and sources:
            lines.append("Sources opened or consulted:")
            for index, item in enumerate(sources[:5], start=1):
                if not isinstance(item, dict):
                    continue
                title = str(item.get("title") or "Untitled source").strip()
                url = str(item.get("url") or "").strip()
                snippet = str(item.get("snippet") or "").strip()
                source_line = f"{index}. {title}"
                if url:
                    source_line += f" — {url}"
                lines.append(source_line)
                if snippet:
                    lines.append(f"   {snippet[:300]}")
        return "\n".join(lines).strip()

    @classmethod
    def _document_body_with_references(
        cls,
        objective: str,
        context: dict[str, Any] | None,
        *,
        image_query: str = "",
        image_search_url: str = "",
        search_url: str = "",
    ) -> str:
        body = cls._document_body(objective, context)
        research_section = cls._research_section_from_context(context)
        if research_section and cls._objective_requests_research_document(objective):
            lowered_body = body.lower()
            if cls._looks_like_dispatch_narration(body) or re.search(
                r"\bi\s+will\s+(?:open|search|look|create|write|start|follow|route)\b",
                lowered_body,
            ):
                body = research_section
            elif research_section not in body:
                body = f"{body.rstrip()}\n\n{research_section}"
        references: list[str] = []
        if search_url:
            references.append(f"Search opened: {search_url}")
        if image_query and image_search_url:
            references.append(
                f"Image request: {image_query}\n"
                "The exported artifact embeds the fetched image only after the governed image receipt verifies the file; "
                "the receipt records the source page used for the image."
            )
        if not references:
            return body
        return f"{body.rstrip()}\n\nArtifact references:\n" + "\n".join(f"- {item}" for item in references)

    @classmethod
    def _compose_research_synthesis_from_sources(
        cls,
        *,
        objective: str,
        query: str,
        summary: str,
        sources: list[dict[str, str]],
    ) -> str:
        """Compose a bounded source-backed document without a second model call."""

        summary = " ".join(str(summary or "").split())[:1400]
        source_lines: list[str] = []
        source_titles: list[str] = []
        source_notes: list[str] = []
        for item in (sources or [])[:3]:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title") or item.get("url") or "Untitled source").strip()
            snippet = " ".join(
                str(item.get("article_body") or item.get("snippet") or "").split()
            )
            url = str(item.get("url") or "").strip()
            if title:
                source_titles.append(title[:160])
            if snippet:
                source_lines.append(f"{title}: {cls._clip_to_sentence(snippet, 240)}")
                source_notes.append(
                    f"{title[:140]} reports or documents that "
                    + cls._clip_to_sentence(snippet, 360)
                    + (f" ({url})" if url else "")
                )
            else:
                source_lines.append(cls._clip_to_sentence(title, 240))
                source_notes.append(f"{title[:180]}" + (f" ({url})" if url else ""))
        topic = str(query or "the requested research topic").strip()
        if not source_lines and not summary:
            return ""
        parts = []
        opening = f"I reviewed {len(source_lines) or len(sources or [])} source"
        opening += "" if (len(source_lines) or len(sources or [])) == 1 else "s"
        opening += f" on {topic}."
        if source_titles:
            opening += " The strongest available signals came from " + ", ".join(source_titles[:3]) + "."
        parts.append(opening)
        if summary:
            parts.append(
                "Taken together, the reporting points to this: "
                + cls._end_on_a_sentence(summary)
            )
        if source_notes:
            parts.append(
                "The details I would preserve in the document are source-bounded, not guessed. "
                + " ".join(source_notes)
            )
        if source_lines and len(" ".join(parts)) < 650:
            parts.append(
                "The available evidence is not equally deep in every source, so I would not pretend "
                "the search produced more certainty than it did. I would treat repeated claims across "
                "the sources as the reliable core, keep isolated details attributed, and mark any thin "
                "or inaccessible material as a place where better reporting would be needed before "
                "making a stronger conclusion."
            )
        if cls._objective_requests_opinion(objective):
            # This is the DETERMINISTIC composer. It runs when authored
            # synthesis was suppressed — under memory pressure, the guard in
            # _allow_research_model_synthesis. So whatever it writes here, no
            # view was formed, and a paragraph opening "In my view" is a claim
            # to have formed one.
            #
            # Live 2026-07-30 00:33: asked to "write a synthesis with your own
            # opinion", the document Bryan received said "In my view, the
            # reliable path is to treat the articles as evidence to compare" —
            # generic method talk, identical for orcas and for anything else,
            # asserting an opinion nobody had. The 23:39 run, with the runtime
            # settled and the guard open, wrote a real one about orcas. The
            # capability is there; this line was covering for its absence.
            #
            # Saying so plainly costs a paragraph of polish and buys the thing
            # the whole document is for: what is in it is true.
            parts.append(
                "On my own opinion, which you asked for: I have not formed one here. "
                "This document is source-bounded extraction — I was not able to author a "
                "synthesis in my own words on this pass, so read the comparison above as "
                "evidence I gathered rather than as a view I hold. Ask me again and I will "
                "give you the opinion rather than a placeholder for it."
            )
        else:
            parts.append(
                "My concise synthesis is that the useful answer is not a loose headline recap; it is a "
                "comparison of what the sources actually support, which claims appear repeated across the "
                "evidence, and which details should stay attributed to a specific source."
            )
        return "\n\n".join(part for part in parts if part).strip()[:4000]

    @staticmethod
    def _allow_desktop_task_model_synthesis(context: dict[str, Any] | None) -> bool:
        context = context or {}
        # Visible desktop work must not allocate a hidden second foreground
        # model by default. The default path composes from search evidence
        # deterministically; model synthesis is an explicit enhancement and is
        # still suppressed under memory pressure.
        if context.get("allow_desktop_task_model_synthesis") is not True:
            return False
        try:
            from core.utils.memory_monitor import get_memory_pressure_snapshot

            snapshot = get_memory_pressure_snapshot()
            return not (
                bool(getattr(snapshot, "warning", False))
                or bool(getattr(snapshot, "refuse_heavy_local_generation", False))
            )
        except (ImportError, AttributeError, RuntimeError, OSError, ValueError) as exc:
            record_degradation(
                "desktop_task",
                exc,
                action="allowed desktop research model synthesis despite memory safety probe failure",
                severity="warning",
            )
        return True

    @staticmethod
    def _allow_research_model_synthesis(
        context: dict[str, Any] | None,
        objective: str = "",
    ) -> bool:
        """Authoring a synthesis is the REQUEST, not a hidden extra allocation.

        The opt-in flag exists so background desktop work cannot quietly spend a
        second foreground model. Nothing on the live path ever set it, so every
        request to "read them and form your own opinion... write a synthesis in
        your own words" fell to the deterministic composer, which concatenates
        source snippets. What Bryan received was:

          "Taken together, the reporting points to this: <snippet> <snippet>"

        — no takeaway, nothing learned, no summary. Not because synthesis
        failed, but because it was never attempted.

        When the objective explicitly asks her to synthesize, summarize, or form
        an opinion in her own words, the model call IS the task and refusing it
        cannot satisfy the request. The router's admission controller remains
        responsible for load safety; this layer must not silently replace the
        requested authorship with a generic template.
        """

        if DesktopTaskSkill._allow_desktop_task_model_synthesis(context):
            return True
        return DesktopTaskSkill._objective_requests_authored_synthesis(objective)

    #: Verbs that mean "write it yourself", as opposed to "collect sources".
    _AUTHORED_SYNTHESIS_RE = re.compile(
        r"(?i)\b("
        r"synthes(?:is|ise|ize)"
        r"|summar(?:y|ise|ize|ising|izing)"
        r"|in your own words"
        r"|your own opinion|form an opinion|what (?:do )?you think|your (?:view|take|assessment)"
        r"|assessment|analy(?:se|ze|sis)"
        r"|write (?:up|about|a (?:summary|synthesis|piece|report))"
        r")\b"
    )

    @classmethod
    def _objective_requests_authored_synthesis(cls, objective: str) -> bool:
        return bool(cls._AUTHORED_SYNTHESIS_RE.search(str(objective or "")))

    @classmethod
    def _research_synthesis_satisfies_objective(
        cls,
        objective: str,
        synthesis: str,
    ) -> bool:
        body = " ".join(str(synthesis or "").split())
        if len(body) < 60 or cls._looks_like_dispatch_narration(body):
            return False
        if cls._looks_like_incomplete_document_body(body):
            return False
        if not cls._objective_requests_opinion(objective):
            return True
        lowered = body.casefold()
        if re.search(
            r"\b(?:i (?:have not|haven't|did not|didn't) form(?:ed)?|"
            r"no opinion|cannot offer (?:an|my) opinion|ask me again)\b",
            lowered,
        ):
            return False
        return bool(
            re.search(
                r"\b(?:in my view|in my opinion|my (?:view|opinion|take|assessment)|"
                r"i (?:think|believe|find|conclude|would argue|would favor|would favour))\b",
                lowered,
            )
        )

    @staticmethod
    def _clip_to_sentence(text: str, limit: int) -> str:
        """Cut at a sentence, then a word — never mid-word, never mid-clause.

        Live 2026-07-30 00:33, in the PDF that landed in Bryan's Documents:
        "They are apex predators and one of the world's most widely distributed
        animals" — a source snippet cut at a character count, so the document he
        was about to show someone stopped in the middle of a clause. Character
        limits are the right idea and the wrong unit; prose has boundaries and
        they are cheap to find.
        """
        body = " ".join(str(text or "").split())
        if len(body) <= limit:
            return body
        window = body[:limit]
        for terminator in (". ", "! ", "? "):
            cut = window.rfind(terminator)
            if cut >= limit // 2:
                return window[: cut + 1].strip()
        cut = window.rfind(" ")
        kept = window[:cut] if cut >= limit // 2 else window
        return kept.rstrip(" ,;:-—") + "…"

    @staticmethod
    def _end_on_a_sentence(text: str) -> str:
        """Drop a trailing half-sentence that arrived already truncated.

        The upstream summary is clipped before it reaches the composer, so the
        composer receives the broken tail rather than making it. Trimming back
        to the last complete sentence is only worth it when a sentence actually
        survives — otherwise the mark stays, because silently dropping the only
        content there is would be worse than showing it was cut.
        """
        body = " ".join(str(text or "").split())
        if not body or body[-1] in ".!?…":
            return body
        cut = max(body.rfind(". "), body.rfind("! "), body.rfind("? "))
        if cut > 0 and (cut + 1) >= len(body) // 2:
            return body[: cut + 1].strip()
        return body.rstrip(" ,;:-—") + "…"

    async def _collect_research_context(
        self,
        *,
        capability_engine: Any,
        objective: str,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        if not self._objective_requests_research_document(objective):
            return {}
        research_started = time.perf_counter()
        research_timing_ms: dict[str, float] = {}
        query = self._extract_search_query(objective)
        if not query:
            return {}
        deep_search = True
        # THE REQUEST DECIDES. Nothing else does.
        #
        # This was a flat 5: asked for "3 recent articles about orcas" she
        # fetched and read five, and reading is the entire cost of the step
        # (gathering logs as 0.0s). Two of those five were latency for
        # material nobody asked for, and the document cited more sources than
        # the request wanted.
        #
        # A "+1 spare for a dead link" lived here briefly and was the same
        # mistake one size smaller — a number with no one behind it. If a
        # source fails to fetch, the validation below already says so
        # honestly rather than silently padding.
        #
        # When the request names no count, nothing here invents one: the key
        # is simply not sent, so web_search's own documented default applies.
        # One default, in the schema where it is described, instead of five
        # scattered guesses.
        requested = self._requested_research_source_count(objective)
        num_results = requested
        search_query = query
        if self._objective_requests_recent_sources(objective) and not re.search(
            r"\b(?:recent|latest|current)\b", query, flags=re.IGNORECASE
        ):
            search_query = f"{query} recent articles"
        pressure_limited = False
        try:
            from core.utils.memory_monitor import get_memory_pressure_snapshot

            snapshot = get_memory_pressure_snapshot()
            pressure_limited = bool(
                getattr(snapshot, "warning", False)
                or getattr(snapshot, "refuse_heavy_local_generation", False)
            )
            if pressure_limited:
                deep_search = False
                num_results = (
                    min(num_results, _MEMORY_SAFE_SOURCE_CEILING)
                    if num_results
                    else _MEMORY_SAFE_SOURCE_CEILING
                )
        except (ImportError, AttributeError, RuntimeError, OSError, ValueError) as exc:
            record_degradation(
                "desktop_task",
                exc,
                action="using shallow desktop research because memory safety probe failed",
                severity="warning",
            )
            deep_search = False
            num_results = (
                min(num_results, _MEMORY_SAFE_SOURCE_CEILING)
                if num_results
                else _MEMORY_SAFE_SOURCE_CEILING
            )
            pressure_limited = True
        step_context = self._child_step_context(context)
        step_context.update(
            {
                "origin": step_context.get("origin") or "desktop_task",
                "route": "desktop_task.web_search",
                "objective": objective,
                "foreground_request": False,
                "user_requested_action": True,
                "user_explicitly_authorized": True,
                # This layer needs fetched evidence.  It owns the authored
                # synthesis and semantic completion check below, so the search
                # subsystem must not allocate another model first.
                "evidence_only": True,
                "desktop_task_reason": "Collect live research evidence before composing the requested document.",
                "desktop_task_expect": "Web search returns sources or an explicit failure.",
            }
        )
        self._emit_progress(
            index=1,
            total=3,
            action="research",
            state="searching",
            detail=f"Gathering and reading source evidence for {query[:120]}.",
        )
        search_started = time.perf_counter()
        try:
            result = await capability_engine.execute(
                "web_search",
                {
                    "query": search_query,
                    # Present only when the request asked for a number.
                    **({"num_results": num_results} if num_results else {}),
                    # Deep article fetches are useful, but they are no longer
                    # allowed to run before memory admission. Under pressure we
                    # use snippets and fewer sources instead of risking a live
                    # desktop RAM spike.
                    "deep": deep_search,
                    "retain": False,
                    "force_refresh": True,
                },
                context=step_context,
            )
        except (AttributeError, RuntimeError, TypeError, ValueError, OSError, TimeoutError) as exc:
            research_timing_ms["search"] = round(
                (time.perf_counter() - search_started) * 1000.0,
                1,
            )
            research_timing_ms["total"] = round(
                (time.perf_counter() - research_started) * 1000.0,
                1,
            )
            record_degradation(
                "desktop_task",
                exc,
                action="continued desktop document task without pre-document research evidence",
                severity="warning",
            )
            return {
                "desktop_task_research_query": query,
                "desktop_task_research_error": str(exc),
                "desktop_task_research_timing_ms": research_timing_ms,
            }
        research_timing_ms["search"] = round(
            (time.perf_counter() - search_started) * 1000.0,
            1,
        )
        pipeline_timing_ms = (
            dict(result.get("timing_ms") or {})
            if isinstance(result, dict) and isinstance(result.get("timing_ms"), dict)
            else {}
        )
        if not isinstance(result, dict):
            result = {"ok": bool(result), "result": result}
        if not bool(result.get("ok", True)):
            return {
                "desktop_task_research_query": query,
                "desktop_task_research_error": str(result.get("error") or result.get("status") or result),
                "desktop_task_research_deep": deep_search,
                "desktop_task_research_pressure_limited": pressure_limited,
            }
        candidate_sources = self._research_sources_from_result(result)
        requested_sources = self._requested_research_source_count(objective)
        required_sources = max(1, requested_sources)
        require_recent = self._objective_requests_recent_sources(objective)
        require_read = self._objective_requests_source_reading(objective)
        sources = self._usable_research_sources(
            candidate_sources,
            require_recent=require_recent,
            require_read=require_read,
        )
        if len(sources) < required_sources:
            missing = required_sources - len(sources)
            replacement_query = (
                f"{query} latest independent reporting"
                if self._objective_requests_recent_sources(objective)
                else f"{query} additional independent sources"
            )
            replacement_started = time.perf_counter()
            try:
                replacement = await capability_engine.execute(
                    "web_search",
                    {
                        "query": replacement_query,
                        "num_results": min(8, max(3, missing * 2)),
                        "deep": deep_search,
                        "retain": False,
                        "force_refresh": True,
                    },
                    context={
                        **step_context,
                        "route": "desktop_task.web_search.replacement",
                        "desktop_task_reason": (
                            "Replace filtered, duplicate, or inaccessible sources so the "
                            "requested evidence count is actually satisfied."
                        ),
                    },
                )
                if isinstance(replacement, dict) and bool(replacement.get("ok", True)):
                    candidate_sources = self._merge_research_sources(
                        candidate_sources,
                        self._research_sources_from_result(replacement),
                    )
                    sources = self._usable_research_sources(
                        candidate_sources,
                        require_recent=require_recent,
                        require_read=require_read,
                    )
            except (
                AttributeError,
                RuntimeError,
                TypeError,
                ValueError,
                OSError,
                TimeoutError,
            ) as exc:
                record_degradation(
                    "desktop_task",
                    exc,
                    action="reported exact research-source shortfall after replacement search failed",
                    severity="warning",
                )
            research_timing_ms["replacement_search"] = round(
                (time.perf_counter() - replacement_started) * 1000.0,
                1,
            )
        if len(sources) < required_sources:
            research_timing_ms["total"] = round(
                (time.perf_counter() - research_started) * 1000.0,
                1,
            )
            return {
                "desktop_task_research_query": query,
                "desktop_task_research_error": (
                    f"research verified {len(sources)}"
                    f"{' readable' if require_read else ''}"
                    f"{' recent' if require_recent else ''} source(s), "
                    f"but the objective requires {required_sources}"
                ),
                "desktop_task_research_deep": deep_search,
                "desktop_task_research_pressure_limited": pressure_limited,
                "desktop_task_research_sources": candidate_sources,
                "desktop_task_research_timing_ms": research_timing_ms,
            }
        if requested_sources:
            sources = sources[:requested_sources]
        summary = str(
            result.get("summary")
            or result.get("answer")
            or result.get("message")
            or result.get("content")
            or result.get("result")
            or ""
        ).strip()
        if not summary and sources:
            summary = "Key source notes:\n" + "\n".join(
                f"- {item.get('title') or item.get('url')}: {item.get('snippet')}"
                for item in sources[:3]
            )
        research_ctx = {
            "desktop_task_research_query": query,
            "desktop_task_research_summary": summary[:3000],
            "desktop_task_research_sources": sources,
            "desktop_task_research_deep": deep_search,
            "desktop_task_research_pressure_limited": pressure_limited,
            "desktop_task_research_timing_ms": research_timing_ms,
            "desktop_task_research_pipeline_timing_ms": pipeline_timing_ms,
        }
        synthesis = self._compose_research_synthesis_from_sources(
            objective=objective,
            query=query,
            summary=summary,
            sources=sources,
        )
        # Optional model synthesis is an explicitly enabled enhancement, not a
        # hidden second foreground allocation during visible desktop work.
        if self._allow_research_model_synthesis(context, objective):
            self._emit_progress(
                index=2,
                total=3,
                action="research",
                state="synthesizing",
                detail=(
                    f"Composing the requested document from {len(sources)} verified "
                    "source records."
                ),
            )
            synthesis_started = time.perf_counter()
            model_synthesis = ""
            repair_feedback = ""
            for _synthesis_attempt in range(2):
                model_synthesis = await self._synthesize_research_document(
                    objective=objective,
                    query=query,
                    summary=summary,
                    sources=sources,
                    repair_feedback=repair_feedback,
                )
                if self._research_synthesis_satisfies_objective(
                    objective,
                    model_synthesis,
                ):
                    break
                repair_feedback = (
                    "The previous draft failed the semantic completion check: it was "
                    "missing a complete cross-source synthesis or the requested "
                    "independent first-person position. Rewrite the document itself; "
                    "do not discuss the failure or promise a later answer."
                )
            research_timing_ms["synthesis"] = round(
                (time.perf_counter() - synthesis_started) * 1000.0,
                1,
            )
            if self._research_synthesis_satisfies_objective(
                objective,
                model_synthesis,
            ):
                synthesis = model_synthesis
                research_ctx["desktop_task_research_authored"] = True
                research_ctx["desktop_task_research_synthesis_sha256"] = text_sha256(
                    synthesis
                )
                research_ctx[
                    "desktop_task_research_synthesis_source_sha256s"
                ] = [
                    str(item.get("source_evidence_sha256") or "")
                    for item in sources
                    if self._valid_sha256(
                        str(item.get("source_evidence_sha256") or "")
                    )
                ]
            elif self._objective_requests_authored_synthesis(objective):
                research_timing_ms["total"] = round(
                    (time.perf_counter() - research_started) * 1000.0,
                    1,
                )
                return {
                    **research_ctx,
                    "desktop_task_research_error": (
                        "the requested authored synthesis did not satisfy its content contract"
                    ),
                }
        if synthesis:
            research_ctx["desktop_task_research_synthesis"] = synthesis
        # Learn from what she just read and wrote: persist the finding as an
        # episode so it consolidates into memory (and the engram/reconsolidation
        # dynamics) rather than being forgotten the moment the document is saved.
        await self._remember_research(query, synthesis or summary, sources)
        research_timing_ms["total"] = round(
            (time.perf_counter() - research_started) * 1000.0,
            1,
        )
        self._emit_progress(
            index=3,
            total=3,
            action="research",
            state="ready",
            detail=(
                f"Research and document content are ready from {len(sources)} sources "
                f"in {research_timing_ms['total'] / 1000.0:.1f}s."
            ),
        )
        return research_ctx

    async def _remember_research(
        self, query: str, finding: str, sources: list[dict[str, str]]
    ) -> None:
        """Best-effort: encode a research finding into episodic memory so Aura
        retains what she learned from reading and writing."""
        finding = str(finding or "").strip()
        if not query or not finding:
            return
        try:
            from core.container import ServiceContainer

            episodic = ServiceContainer.get("episodic_memory", default=None)
            recorder = getattr(episodic, "record_episode_async", None) if episodic else None
            if not callable(recorder):
                return
            top = [
                str(s.get("url") or s.get("title") or "")
                for s in (sources or [])[:3]
                if isinstance(s, dict)
            ]
            await recorder(
                context=f"Researched and wrote about: {query}",
                action=f"Read {len(sources or [])} sources and composed a summary",
                outcome=finding[:800],
                success=True,
                importance=0.62,
                lessons=[f"Source: {u}" for u in top if u],
                source="desktop_task_research",
            )
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation(
                "desktop_task",
                exc,
                action="continued after research-learning episode record failed",
                severity="warning",
            )

    async def _synthesize_research_document(
        self,
        *,
        objective: str,
        query: str,
        summary: str,
        sources: list[dict[str, str]],
        repair_feedback: str = "",
    ) -> str:
        """Compose a first-person summary (and opinion, when asked) of the
        research through the canonical model router. Bounded and best-effort:
        if the router is unavailable the raw research section still stands."""
        from core.container import ServiceContainer

        router = ServiceContainer.get("llm_router", default=None)
        generate = getattr(router, "generate", None) if router is not None else None
        if not callable(generate):
            return ""
        def _src_tag(item: dict[str, Any]) -> str:
            rep = int(item.get("reputability", 1) or 0)
            label = {3: "high-authority", 2: "reputable", 1: "general", 0: "low-quality"}.get(rep, "general")
            if not item.get("accessible", True):
                label += ", limited/blocked access"
            return label

        source_lines = "\n".join(
            f"- [{_src_tag(item)}] {str(item.get('title') or item.get('url') or 'source')} "
            f"({str(item.get('url') or '')}):\n  "
            f"{str(item.get('article_body') or item.get('snippet') or '')[:1200]}"
            for item in (sources or [])[:5]
            if isinstance(item, dict)
        )
        wants_opinion = self._objective_requests_opinion(objective)
        opinion_clause = (
            " Close with a separate short first-person opinion paragraph beginning "
            "\"In my view,\" giving your own first-person take on what these sources "
            "say and what you make of them."
            if wants_opinion
            else ""
        )
        prompt = (
            f'You researched "{query}" and gathered these sources (titles, URLs, and '
            f"article text):\n{summary[:3500]}\n{source_lines}\n\n"
            "Write a thorough, well-organized composite document for a reader who wants "
            "to actually understand the topic — not a thin gloss. Synthesize ACROSS the "
            "sources rather than listing them one by one: open with the core facts, then "
            "develop the important context, specifics (names, numbers, dates, quotes "
            "where useful), and any points where the sources differ or add nuance. Use "
            "several paragraphs and scale the depth to the material — be as complete and "
            "substantive as the sources support. If the sources are genuinely thin or "
            "conflicting, say so honestly and note what would need further research "
            "rather than padding.\n"
            "Weigh your sources critically. Give more trust to reputable, authoritative "
            "sources — peer-reviewed research, .edu/.gov, established institutions and "
            "labs, and named expert authors with relevant credentials — and to claims "
            "that several independent reputable sources corroborate. Treat single-source, "
            "anonymous, overtly promotional, or paywalled/ad-wall pages with appropriate "
            "skepticism and flag that uncertainty. If a source was inaccessible, blocked, "
            "or clearly content-thin, do not rely on it, and note where a better source "
            "would be needed."
            f"{opinion_clause}\n"
            "Write in the first person as Aura, in clean prose. Do not mention tools, "
            "steps, dispatch, commitments, or that you are executing a task — this is the "
            "finished document the reader will see, not a status update."
            + (f"\n\nREVISION REQUIREMENT: {repair_feedback}" if repair_feedback else "")
        )
        try:
            text = await asyncio.wait_for(
                generate(
                    prompt=prompt,
                    timeout=110.0,
                    temperature=0.6,
                    # SCALED TO THE SOURCES, not a flat ceiling.
                    #
                    # 1100 was too small — the live synthesis was cut
                    # mid-clause, ending "...The increase is a" before the
                    # Sources list. 2048 for everything is the opposite error:
                    # once the fetch stopped over-reading, a three-source
                    # request still produced a four-kilobyte document, and
                    # since the whole cost of this step is the local model
                    # writing, the time saved by reading less went straight
                    # back into writing more. Measured: research 82.4s -> 27.8s,
                    # total unchanged at ~100s.
                    #
                    # The floor keeps the mid-clause failure from returning.
                    max_tokens=max(1100, min(2048, 350 + 380 * max(1, len(sources)))),
                    # Pin synthesis to the on-device Cortex: it has no external
                    # quota, so the document never degrades to a thin heuristic
                    # fallback because a cloud tier returned 429 RESOURCE_EXHAUSTED.
                    prefer_tier="local",
                    # Her own authoring, not the surface the person typed at.
                    #
                    # "desktop_task" begins with an allowlisted user-facing
                    # label, so anything starting "desktop_" inherits the
                    # protected reply lane — and with it the apology written
                    # for a person: "I can't work through that technical
                    # request right now, my language backend is temporarily
                    # unavailable", handed back as the body of a document.
                    # The tool dispatch IS user-facing and keeps that origin;
                    # this sub-call never was.
                    origin="internal_desktop_authoring",
                    purpose="research_document_synthesis",
                        # Foreground work, and not the reply lane. Same seam and
                        # same reason as the artifact writer above.
                        _non_chat_inference=True,
                    # Said again where the gate reads it. The router pops
                    # the flag above and re-adds it under another name, and
                    # the gate's own check looks at the context it was
                    # handed — which is how an apology written for a person
                    # ("my language backend is temporarily unavailable")
                    # arrived as the body of a document. The gate already
                    # returns nothing to an internal caller for exactly
                    # this reason; it could not tell this was one.
                    internal_inference=True,
                    # Somebody is waiting for this file.
                    #
                    # Her reasoning lane declares the same thing and reaches
                    # the resident worker; this one did not, and asked a lane
                    # that answered "worker_not_alive" for half a minute
                    # while the runtime's own health reported Cortex ready and
                    # generating. Internal is about whose question it is.
                    # Foreground is about whether anyone is waiting, and both
                    # are true here.
                    foreground_request=True,
                ),
                timeout=120.0,
            )
        except (AttributeError, RuntimeError, TypeError, ValueError, OSError, TimeoutError) as exc:
            record_degradation(
                "desktop_task",
                exc,
                action="composed research document from raw search section after synthesis was unavailable",
                severity="warning",
            )
            return ""
        text = str(text or "").strip()
        # The router guarantees non-empty (diagnostic fallback); a synthesis
        # that is a degraded diagnostic line or dispatch narration is not
        # document content, so fall back to the raw research section.
        if not text or self._looks_like_dispatch_narration(text):
            return ""
        if re.search(r"\b(?:diagnostic|fallback|unavailable|all (?:remote )?endpoints? failed)\b", text.lower()) and len(text) < 200:
            return ""
        # Never hand back a document that stops mid-clause. Whether the budget
        # ran out or the 4000-char clamp lands mid-word, the reader gets a
        # finished paragraph rather than "...The increase is a" followed by the
        # Sources list.
        text = text[:4000]
        try:
            from core.conversation.response_reliability import complete_truncated_tail

            completed = complete_truncated_tail(text)
            if completed and len(completed) >= len(text) * 0.5:
                text = completed
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
            pass
        return text

    @classmethod
    def _steps_from_payload(cls, payload: Any) -> list[DesktopTaskStep]:
        if isinstance(payload, dict):
            payload = payload.get("steps")
        if not isinstance(payload, list):
            return []
        if len(payload) > 20:
            return []
        steps: list[DesktopTaskStep] = []
        for item in payload:
            try:
                steps.append(item if isinstance(item, DesktopTaskStep) else DesktopTaskStep(**dict(item)))
            except (TypeError, ValueError):
                return []
        return steps

    @classmethod
    def _steps_from_plan_text(cls, text: str) -> list[DesktopTaskStep]:
        for candidate in cls._json_candidates_from_text(text):
            parsed = cls._structured_payload_from_text(candidate)
            steps = cls._steps_from_payload(parsed)
            if steps:
                return steps
        return []

    @classmethod
    def _steps_from_context(cls, context: dict[str, Any] | None) -> list[DesktopTaskStep]:
        steps, _ = cls._steps_with_provenance_from_context(context)
        return steps

    @classmethod
    def _steps_with_provenance_from_context(
        cls,
        context: dict[str, Any] | None,
    ) -> tuple[list[DesktopTaskStep], str]:
        context = context or {}
        for key in ("desktop_task_steps", "desktop_task_plan"):
            steps = cls._steps_from_payload(context.get(key))
            if steps:
                return steps, key
            steps = cls._steps_from_plan_text(str(context.get(key) or ""))
            if steps:
                return steps, key
        for key in ("cognitive_reply", "draft_response", "response"):
            steps = cls._steps_from_plan_text(str(context.get(key) or ""))
            if steps:
                return steps, f"{key}_structured"
        return [], ""

    @classmethod
    def _declared_plan_validation_error(cls, context: dict[str, Any] | None) -> str:
        payload = cls._structured_payload_from_context(context)
        if "steps" not in payload:
            return ""
        raw_steps = payload.get("steps")
        if raw_steps in (None, []):
            return ""
        if not isinstance(raw_steps, list):
            return "Structured desktop plan 'steps' must be a list."
        if len(raw_steps) > MAX_DESKTOP_TASK_STEPS:
            return f"Structured desktop plan exceeds the {MAX_DESKTOP_TASK_STEPS}-step execution limit."
        if len(cls._steps_from_payload(raw_steps)) != len(raw_steps):
            return "Structured desktop plan contains an invalid or unsupported step."
        return ""

    @staticmethod
    def _target_payload(target: Any) -> dict[str, Any]:
        if isinstance(target, dict):
            return dict(target)
        if isinstance(target, str):
            try:
                parsed = json.loads(target)
            except json.JSONDecodeError:
                return {}
            if isinstance(parsed, dict):
                return parsed
        return {}

    @staticmethod
    def _valid_sha256(value: Any) -> bool:
        return bool(re.fullmatch(r"[0-9a-f]{64}", str(value or "").strip().lower()))

    @classmethod
    def _replace_document_body_tokens(cls, value: Any, document_body: str) -> Any:
        if not document_body:
            return value
        if isinstance(value, str):
            updated = value
            for body_token in cls._DOCUMENT_BODY_TOKENS:
                updated = updated.replace(body_token, document_body)
            return updated
        if isinstance(value, dict):
            return {
                key: cls._replace_document_body_tokens(item, document_body)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [cls._replace_document_body_tokens(item, document_body) for item in value]
        return value

    @classmethod
    def _resolve_document_body_tokens(
        cls,
        steps: list[DesktopTaskStep],
        document_body: str,
    ) -> list[DesktopTaskStep]:
        resolved: list[DesktopTaskStep] = []
        for step in steps:
            target = cls._replace_document_body_tokens(step.target, document_body)
            if target == step.target:
                resolved.append(step)
            else:
                resolved.append(step.model_copy(update={"target": target}))
        return resolved

    @staticmethod
    def _lookup_result_path(value: Any, path: str) -> tuple[bool, Any]:
        current = value
        for part in path.split("."):
            if isinstance(current, dict) and part in current:
                current = current[part]
                continue
            if isinstance(current, list) and part.isdigit():
                index = int(part)
                if 0 <= index < len(current):
                    current = current[index]
                    continue
            return False, None
        return True, current

    @classmethod
    def _resolve_step_reference_token(
        cls,
        match: re.Match[str],
        receipts: list[dict[str, Any]],
    ) -> tuple[bool, Any, str]:
        index_text = match.group("index")
        if index_text is None:
            if not receipts:
                return False, None, "last step is unavailable"
            receipt = receipts[-1]
        else:
            index = int(index_text)
            if index > len(receipts):
                return False, None, f"step {index} has not completed"
            receipt = receipts[index - 1]
        if not receipt.get("ok"):
            return False, None, f"referenced step {receipt.get('index')} did not verify"
        path = match.group("path")
        found, value = cls._lookup_result_path(receipt, path)
        if not found:
            return False, None, f"referenced result path '{path}' is unavailable"
        return True, value, ""

    @classmethod
    def _resolve_step_references(
        cls,
        value: Any,
        receipts: list[dict[str, Any]],
    ) -> tuple[bool, Any, str]:
        if isinstance(value, dict):
            resolved: dict[str, Any] = {}
            for key, item in value.items():
                ok, replacement, error = cls._resolve_step_references(item, receipts)
                if not ok:
                    return False, value, error
                resolved[key] = replacement
            return True, resolved, ""
        if isinstance(value, list):
            resolved_items: list[Any] = []
            for item in value:
                ok, replacement, error = cls._resolve_step_references(item, receipts)
                if not ok:
                    return False, value, error
                resolved_items.append(replacement)
            return True, resolved_items, ""
        if not isinstance(value, str):
            return True, value, ""

        matches = list(cls._STEP_REFERENCE_PATTERN.finditer(value))
        if not matches:
            return True, value, ""
        if len(matches) == 1 and matches[0].span() == (0, len(value)):
            ok, replacement, error = cls._resolve_step_reference_token(matches[0], receipts)
            if not ok:
                return False, value, error
            return True, replacement, ""

        resolved_text = value
        for match in reversed(matches):
            ok, replacement, error = cls._resolve_step_reference_token(match, receipts)
            if not ok:
                return False, value, error
            if isinstance(replacement, (dict, list)):
                replacement_text = json.dumps(replacement, ensure_ascii=False)
            else:
                replacement_text = str(replacement)
            start, end = match.span()
            resolved_text = resolved_text[:start] + replacement_text + resolved_text[end:]
        return True, resolved_text, ""

    @classmethod
    def _resolve_step_target(
        cls,
        step: DesktopTaskStep,
        receipts: list[dict[str, Any]],
    ) -> tuple[bool, DesktopTaskStep, str]:
        # Resolve INSIDE the parsed target, not inside its JSON text.
        #
        # Substituting into the raw string corrupts the JSON whenever the
        # replacement contains a quote — a list of filenames from a directory
        # read is the ordinary case, and it produced a target that no longer
        # parsed. Parsing first, resolving within the structure, and
        # re-serialising means the encoder does the quoting.
        raw_target: Any = step.target
        target_was_json = False
        if isinstance(raw_target, str):
            stripped = raw_target.strip()
            if stripped.startswith(("{", "[")):
                try:
                    raw_target = json.loads(stripped)
                    target_was_json = True
                except (TypeError, ValueError):
                    raw_target = step.target
        ok, target, error = cls._resolve_step_references(raw_target, receipts)
        if not ok:
            return False, step, error
        if target_was_json:
            target = json.dumps(target, ensure_ascii=False)
            if target == step.target:
                return True, step, ""
            return True, step.model_copy(update={"target": target}), ""
        if isinstance(target, list):
            target = json.dumps(target, ensure_ascii=False)
        elif not isinstance(target, (str, dict)):
            target = str(target)
        if target == step.target:
            return True, step, ""
        return True, step.model_copy(update={"target": target}), ""

    @staticmethod
    def _emit_progress(
        *,
        index: int,
        total: int,
        action: str,
        state: str,
        detail: str,
        level: str = "info",
    ) -> None:
        try:
            from core.thought_stream import get_emitter

            get_emitter().emit(
                "Desktop Task",
                f"Step {index}/{total} {action}: {state}. {detail[:240]}",
                level=level,
                category="ToolExecution",
                step_index=index,
                step_total=total,
                action=action,
                state=state,
            )
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation(
                "desktop_task",
                exc,
                action="continued verified desktop execution without neural-stream progress telemetry",
                severity="warning",
            )

    @staticmethod
    def _digest_payload(payload: Any) -> str:
        try:
            encoded = json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            ).encode("utf-8")
        except (TypeError, ValueError):
            encoded = str(payload).encode("utf-8", errors="replace")
        return hashlib.sha256(encoded).hexdigest()

    @classmethod
    async def _emit_durable_step_receipt(
        cls,
        receipt: dict[str, Any],
        *,
        objective: str,
        planner: str,
        tool: str = "computer_use",
    ) -> None:
        try:
            from core.runtime.receipts import ToolExecutionReceipt, get_receipt_store

            store = get_receipt_store()
            durable = ToolExecutionReceipt(
                cause=str(objective or "desktop_task")[:240],
                tool=tool,
                governance_receipt_id=str(
                    (receipt.get("result") or {}).get("governance_receipt_id")
                    or (receipt.get("result") or {}).get("authority_receipt_id")
                    or ""
                )
                or None,
                capability_receipt_id=str(
                    (receipt.get("result") or {}).get("capability_receipt_id") or ""
                )
                or None,
                status="success_verified" if receipt.get("ok") else "failed",
                output_digest=cls._digest_payload(receipt.get("result") or {}),
                verification_evidence={
                    "step_index": receipt.get("index"),
                    "action": receipt.get("action"),
                    "critical": receipt.get("critical", True),
                    "effect_verified": receipt.get("effect_verified"),
                    "effect_evidence": receipt.get("effect_evidence"),
                    "planner": planner,
                    "attempts": receipt.get("attempts", 0),
                },
            )
            emitted = await asyncio.to_thread(store.emit, durable)
            receipt["durable_receipt_id"] = emitted.receipt_id
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError, OSError) as exc:
            record_degradation(
                "desktop_task",
                exc,
                action="continued desktop task after durable step receipt emission failed",
                severity="warning",
            )
        # The turn's own effect ledger, so a guard on any other lane can ask
        # whether anything verifiably happened. Every step receipt in this
        # module reaches this method, which is why the hook lives here rather
        # than beside the seven places receipts are appended: a lane added
        # later inherits the ledger by emitting a receipt at all.
        try:
            from core.epistemics.turn_effects import record_verified_effects

            record_verified_effects([receipt])
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation(
                "desktop_task",
                exc,
                action="continued desktop task after turn effect ledger update failed",
                severity="warning",
            )

    @classmethod
    def _verify_step_effect(cls, step: DesktopTaskStep, result: dict[str, Any]) -> tuple[bool, str]:
        if not result.get("ok"):
            return False, str(result.get("error") or result.get("status") or "child action reported failure")

        action = step.action
        payload = cls._target_payload(step.target)
        if action in {"write_in_app", "create_note"}:
            title = str(result.get("title") or "").strip()
            app = str(result.get("app") or "Notes").strip()
            verified = bool(result.get("effect_verified")) and bool(title)
            return (
                verified,
                f"{app.lower()}_document={title}"
                if verified
                else f"the document was not found in {app} after writing",
            )
        if action == "list_directory":
            # A read's effect is the reading itself. Without this branch the
            # verifier fell through to "unsupported effect evidence", so a
            # directory that had been read correctly reported 0/2 steps and the
            # write that depended on it never ran.
            path = str(result.get("path") or "").strip()
            count = result.get("count")
            verified = (
                bool(path)
                and bool(result.get("ok"))
                and type(count) is int
                and count >= 0
                and isinstance(result.get("names"), list)
            )
            return (
                verified,
                f"listed={path};pattern={result.get('pattern')};count={count}"
                if verified
                else str(result.get("error") or "missing directory listing"),
            )
        if action == "create_folder":
            path = str(result.get("path") or "").strip()
            verified = bool(path) and bool(result.get("effect_verified"))
            return (
                verified,
                f"folder_path={path};verified=true"
                if verified
                else str(result.get("verification") or "missing confirmed folder path"),
            )
        if action == "open_app":
            opened = str(result.get("opened") or "").strip()
            frontmost = str(result.get("frontmost_app") or "").strip()
            verified = bool(result.get("effect_verified")) and bool(opened) and bool(frontmost)
            return (
                verified,
                f"opened={opened};frontmost={frontmost}"
                if verified
                else str(result.get("verification") or "missing frontmost app confirmation"),
            )
        if action == "move_aura_bubble":
            position = result.get("position")
            sequence = result.get("sequence")
            verified = (
                bool(result.get("effect_verified"))
                and isinstance(position, list)
                and len(position) == 2
                and isinstance(sequence, int)
                and sequence > 0
            )
            return (
                verified,
                str(result.get("effect_evidence") or result.get("verification") or "").strip()
                if verified
                else "missing native companion movement acknowledgement",
            )
        if action == "open_url":
            url = str(result.get("url") or "").strip()
            valid_url = url.startswith(("http://", "https://"))
            frontmost = str(result.get("frontmost_app") or "").strip()
            verified = valid_url and bool(result.get("effect_verified")) and bool(frontmost)
            if verified and bool(payload.get("requires_editable_focus")):
                verified = bool(
                    result.get("doc_focused")
                    or result.get("editable_focus_verified")
                )
                if not verified:
                    focus_error = str(
                        result.get("focus_error")
                        or result.get("verification")
                        or "editable document focus was not verified"
                    )
                    return False, focus_error
            return (
                verified,
                f"url={url};frontmost={frontmost}"
                if verified
                else str(result.get("verification") or "missing browser foreground confirmation"),
            )
        if action == "write_text_file":
            path = str(result.get("path") or "").strip()
            bytes_written = result.get("bytes")
            if not path:
                return False, "missing written file path"
            if not isinstance(bytes_written, int) or bytes_written < 0:
                return False, "missing written byte count"
            content = str(payload.get("content") or "")
            if content and bytes_written <= 0:
                return False, "non-empty file write reported zero bytes"
            digest = str(result.get("sha256") or "").strip()
            verified = bool(result.get("effect_verified")) and cls._valid_sha256(digest)
            return (
                verified,
                f"path={path};bytes={bytes_written};sha256={digest}"
                if verified
                else str(result.get("verification") or "missing file content read-back"),
            )
        if action == "render_text_pdf":
            path = str(result.get("path") or "").strip()
            bytes_written = result.get("bytes")
            pages = result.get("pages")
            chars = result.get("chars")
            if not path.lower().endswith(".pdf"):
                return False, "missing rendered PDF path"
            if not isinstance(bytes_written, int) or bytes_written <= 0:
                return False, "missing rendered PDF byte count"
            if not isinstance(pages, int) or pages <= 0:
                return False, "missing rendered PDF page count"
            if not isinstance(chars, int) or chars <= 0:
                return False, "missing rendered PDF character count"
            digest = str(result.get("sha256") or "").strip()
            expected_body = str(payload.get("body") or "")[:9000]
            expected_body_digest = text_sha256(expected_body)
            observed_body_digest = str(result.get("source_body_sha256") or "").strip()
            if observed_body_digest != expected_body_digest:
                return False, "rendered PDF is not bound to the requested source body"
            verified = bool(result.get("effect_verified")) and cls._valid_sha256(digest)
            return (
                verified,
                f"path={path};bytes={bytes_written};pages={pages};chars={chars};"
                f"sha256={digest};source_body_sha256={observed_body_digest}"
                if verified
                else str(result.get("verification") or "missing persisted PDF verification"),
            )
        if action == "fetch_topic_image":
            img_path = str(result.get("path") or "").strip()
            img_bytes = result.get("bytes")
            page_url = str(result.get("page_url") or "").strip()
            if not img_path:
                return False, "missing fetched image path"
            if not isinstance(img_bytes, int) or img_bytes <= 0:
                return False, "missing fetched image byte count"
            digest = str(result.get("sha256") or "").strip()
            verified = bool(result.get("effect_verified")) and cls._valid_sha256(digest)
            return (
                verified,
                f"path={img_path};bytes={img_bytes};source={page_url};sha256={digest}"
                if verified
                else str(result.get("verification") or "missing downloaded image read-back"),
            )
        if action == "system_control":
            domain = str(result.get("domain") or "").strip()
            applied = str(result.get("applied") or "").strip()
            expected = str(result.get("expected") or "").strip()
            verified = bool(result.get("effect_verified")) and bool(domain)
            return (
                verified,
                f"domain={domain};applied={applied};expected={expected}"
                if verified
                else f"missing {domain or 'setting'} read-back confirmation",
            )
        if action == "move_file":
            destination = str(result.get("destination") or "").strip()
            bytes_moved = result.get("bytes")
            if not destination:
                return False, "missing moved destination path"
            if not isinstance(bytes_moved, int) or bytes_moved < 0:
                return False, "missing moved byte count"
            verified = bool(result.get("effect_verified"))
            return (
                verified,
                f"destination={destination};bytes={bytes_moved};verified=true"
                if verified
                else str(result.get("verification") or "missing move postcondition"),
            )
        if action == "set_clipboard":
            chars = result.get("chars")
            if not isinstance(chars, int) or chars < 0:
                return False, "missing clipboard character count"
            digest = str(result.get("sha256") or "").strip()
            verified = bool(result.get("effect_verified")) and cls._valid_sha256(digest)
            return (
                verified,
                f"clipboard_chars={chars};sha256={digest}"
                if verified
                else str(result.get("verification") or "missing exact clipboard read-back"),
            )
        if action == "get_clipboard":
            chars = result.get("chars")
            text = result.get("text")
            if not isinstance(chars, int) or chars < 0 or not isinstance(text, str):
                return False, "missing clipboard readback evidence"
            return True, f"clipboard_read_chars={chars}"
        if action == "read_menu_clock":
            clock_text = str(result.get("clock_text") or result.get("text") or "").strip()
            source = str(result.get("source") or "").strip()
            if not clock_text:
                return False, "missing menu clock readback"
            return True, f"clock_text={clock_text[:80]};source={source or 'unknown'}"
        if action == "run_command":
            exit_code = result.get("exit_code")
            if not isinstance(exit_code, int):
                return False, "missing command exit code"
            if exit_code != 0:
                return False, f"command exited {exit_code}"
            output = str(result.get("output") or "")
            return True, f"exit_code=0;output_chars={len(output)}"
        if action == "click":
            verification = str(result.get("verification") or "").strip()
            verified = bool(result.get("effect_verified")) or "state shifted" in verification.lower()
            return (
                verified,
                verification or "missing click effect evidence",
            )
        if action == "hotkey":
            hotkey = str(result.get("hotkey") or "").strip()
            verification = str(result.get("verification") or "").strip()
            expected_frontmost = str(result.get("expected_frontmost_app") or "").strip()
            is_paste = bool(result.get("is_paste"))
            verified = (
                bool(result.get("effect_verified"))
                or "state shifted" in verification.lower()
                or "focused element changed" in verification.lower()
            )
            if is_paste and expected_frontmost:
                clipboard_check = result.get("clipboard_payload_verification")
                clipboard_check = (
                    clipboard_check if isinstance(clipboard_check, dict) else {}
                )
                target_ok = bool(result.get("write_target_app_verified"))
                clipboard_ok = bool(clipboard_check.get("verified")) or not clipboard_check
                if not target_ok:
                    return False, "paste target app was not verified"
                if not clipboard_ok:
                    return False, "paste clipboard payload was not verified"
            return (
                bool(hotkey) and verified,
                f"hotkey={hotkey};{verification}" if hotkey and verification else "missing hotkey effect evidence",
            )
        if action == "scroll":
            verification = str(result.get("verification") or "").strip()
            verified = bool(result.get("effect_verified")) or "state shifted" in verification.lower()
            return (verified, verification or "missing scroll effect evidence")
        if action == "wait":
            seconds = result.get("seconds")
            if not isinstance(seconds, int | float):
                return False, "missing wait duration evidence"
            return True, f"seconds={seconds}"
        if action == "run_applescript":
            if not bool(result.get("effect_verified")):
                return False, "AppleScript transport output is not objective-specific effect evidence"
            verification_results = result.get("verification_results")
            if not isinstance(verification_results, list):
                return False, "missing structured AppleScript verification results"
            strong_passed = any(
                isinstance(check, dict)
                and bool(check.get("passed"))
                and bool(check.get("strong", True))
                for check in verification_results
            )
            if not strong_passed:
                return False, "missing strong AppleScript postcondition"
            evidence = str(result.get("effect_evidence") or "").strip()
            if not evidence:
                return False, "missing AppleScript effect evidence summary"
            return True, evidence[:240]
        if action == "type":
            verification = str(result.get("verification") or "").strip()
            typed = str(result.get("typed") or "").strip()
            verified = bool(result.get("effect_verified")) or (
                "confirmed" in verification.lower() or "state shifted" in verification.lower()
            )
            evidence = verification or (f"typed_prefix={typed}" if typed else "missing typed text evidence")
            return (
                bool(typed) and verified,
                evidence,
            )
        if action == "pursue_on_screen":
            # A pursuit proves itself by what it did and how it ended.
            #
            # Not by "ok": a run that ends blocked by a dialog, or on a page
            # it was moved away from, reports honestly and must not be read as
            # the goal being reached.
            outcome = str(result.get("outcome") or "").strip()
            moves = result.get("moves")
            made = len(moves) if isinstance(moves, list) else 0
            if not bool(result.get("ok")):
                reason = (
                    str(result.get("cannot_decide") or "")
                    or str(result.get("needs_person") or "")
                    or str(result.get("blocked_by") or "")
                    or outcome
                    or "the goal was not reached"
                )
                return False, f"pursuit_unfinished;moves={made};{reason}"[:240]
            return True, f"pursuit_reached_goal;moves={made};outcome={outcome or 'goal_reached'}"

        if action in {"inspect_screen", "read_screen_text"}:
            text = str(result.get("text") or "").strip()
            active_app = str(result.get("active_app") or "").strip()
            if text:
                detail = "screen_text_returned"
                if active_app:
                    detail = f"{detail};frontmost_app={active_app}"
                return True, detail
            if action == "inspect_screen" and active_app:
                return True, f"frontmost_app={active_app}"
            return (False, "missing screen text evidence")
        return False, f"unsupported effect evidence for desktop action {action}"

    @staticmethod
    def _inline_sentence_for(objective: str) -> str:
        """Content for a file whose body the user left to her.

        "containing one sentence you choose" is a real instruction with no
        text attached, and writing an empty file would satisfy the letter of
        it while failing the request.
        """
        text = str(objective or "")
        quoted = re.search(r"[\"“‘']([^\"”’']{3,400})[\"”’']", text)
        if quoted:
            return quoted.group(1).strip() + "\n"
        # "containing one sentence you choose" leaves the content to her. The
        # first attempt echoed the instruction itself into the file — "one
        # sentence you choose. Actually execute it, then tell me the full
        # path." — which is the request, not an answer to it.
        if re.search(
            r"\b(?:you\s+choose|of\s+your\s+choosing|whatever\s+you\s+(?:like|want)|"
            r"anything\s+you\s+(?:like|want)|up\s+to\s+you)\b",
            text,
            re.IGNORECASE,
        ):
            return (
                "Written by Aura, through the governed desktop file lane — "
                "the sentence is mine, since you left it to me.\n"
            )
        return ""

    @staticmethod
    def _generic_open_app_mentions(objective: str) -> list[str]:
        apps = list(extract_target_apps(str(objective or "")))
        return ["Safari" if app.casefold() == "browser" else app for app in apps[:4]]

    @staticmethod
    def _writing_app_from_apps(apps: list[str]) -> str:
        """Which of the named apps is something you write in.

        This was a four-name allowlist — Notes, TextEdit, Pages, Word — so
        "open Reminders and write..." fell through to a text file on disk,
        which is not what anyone asked for. An app is a writing app if it
        says so: a scripting dictionary with a text-bearing document class is
        exactly that claim, published by the app itself.

        The allowlist survives as a fallback ordering only, for the case
        where no dictionary can be read at all.
        """
        for app in apps:
            if DesktopTaskSkill._app_text_target(app):
                return app
        for app in apps:
            if app in {"Notes", "TextEdit", "Pages", "Microsoft Word"}:
                return app
        return ""

    @staticmethod
    def _step_opens_app(step: DesktopTaskStep, app: str) -> bool:
        return step.action == "open_app" and str(step.target or "").strip() == app

    @classmethod
    def _sequenced_objective_segments(cls, objective: str) -> list[str]:
        """Split only explicit discourse-level sequencing markers.

        A single heuristic plan cannot safely keep focus across independent
        work products. Continuations of the same product stay together so
        research-to-document and compose-to-export chains retain shared state.
        """
        text = str(objective or "").strip()
        if not text:
            return []
        parts = re.split(
            r"(?:[.!?;]\s+|,\s+)(?:and\s+)?(?:then|after that|next|finally|lastly|"
            r"also|i\s+also\s+(?:want|need|would\s+like)\s+to|can\s+you|could\s+you|would\s+you)\s*,?\s*",
            text,
            flags=re.IGNORECASE,
        )
        if len(parts) <= 1:
            return [text]
        candidates = [part.strip(" \t\r\n,.;") for part in parts if part.strip(" \t\r\n,.;")]
        if len(candidates) <= 1:
            return [text]

        def _surfaces(value: str) -> set[str]:
            surfaces = {app.lower() for app in cls._extract_apps(value)}
            web_surface = cls._web_document_url(value)
            if web_surface:
                surfaces.add(web_surface)
            lowered_value = value.lower()
            if any(
                token in lowered_value
                for token in ("search", "look up", "article", "articles", "news", "source", "sources")
            ) or re.search(r"\bread\s+(?:about|on)\b", lowered_value):
                surfaces.add("web_research")
            if cls._extract_image_query(value):
                surfaces.add("image_search")
            for domain, _ in detect_os_settings(value):
                surfaces.add(f"os_setting:{domain}")
            return surfaces

        def _completes_product(value: str) -> bool:
            lowered = value.lower()
            return bool(
                re.search(r"\b(?:export|save|render)\b[^.;\n]{0,80}\b(?:pdf|file|document|artifact)\b", lowered)
                or (
                    re.search(r"\b(?:write|compose|draft|create)\b", lowered)
                    and any(token in lowered for token in ("note", "document", "essay", "report", "summary"))
                    and bool(_surfaces(value))
                )
            )

        segments = [candidates[0]]
        for candidate in candidates[1:]:
            previous = segments[-1]
            starts_distinct_surface = bool(_surfaces(candidate) - _surfaces(previous))
            if _completes_product(previous) and starts_distinct_surface:
                segments.append(candidate)
            else:
                segments[-1] = f"{previous}. Then {candidate}"
        return segments

    @staticmethod
    def _has_explicit_folder_name(objective: str) -> bool:
        text = str(objective or "")
        return bool(
            re.search(
                r"\b(?:folder|directory)\s+(?:named|called|titled)\b",
                text,
                flags=re.IGNORECASE,
            )
            or re.search(
                r"(?:\"[^\"]+\"|'(?:[^']|'(?=\w))+')\s+(?:folder|directory)\b",
                text,
                flags=re.IGNORECASE,
            )
        )

    @classmethod
    def _inherit_shared_destination(cls, segment: str, objective: str) -> str:
        """Resolve a later phase's explicit "same folder" reference.

        This carries only a destination the user explicitly named. It does not
        invent cross-phase state or make every artifact share one directory.
        """
        if not re.search(
            r"\b(?:same|that|the previously (?:named|created))\b[^.\n]{0,80}\b(?:folder|directory)\b",
            segment,
            flags=re.IGNORECASE,
        ):
            return segment
        if cls._has_explicit_folder_name(segment) or not cls._has_explicit_folder_name(objective):
            return segment

        folder_name = cls._extract_folder_name(objective)
        root_hint = cls._extract_root_hint(objective)
        root_phrase = {
            "~/Desktop": " on my Desktop",
            "~/Documents": " in my Documents folder",
            "~/Downloads": " in my Downloads folder",
        }.get(root_hint, "")
        return (
            f'{segment.rstrip(" .")}, using the folder titled '
            f'"{folder_name}"{root_phrase}.'
        )

    @classmethod
    def _deduplicate_segment_artifact_paths(
        cls,
        segment_steps: list[DesktopTaskStep],
        used_paths: set[str],
    ) -> list[DesktopTaskStep]:
        """Give each phase distinct durable outputs inside shared folders."""
        resolved: list[DesktopTaskStep] = []
        for step in segment_steps:
            if step.action not in {"write_text_file", "render_text_pdf"}:
                resolved.append(step)
                continue
            payload = cls._target_payload(step.target)
            path = str(payload.get("path") or "").strip()
            if not path:
                resolved.append(step)
                continue
            if path in used_paths:
                candidate = Path(path)
                path = next(
                    str(candidate.with_name(f"{candidate.stem}_{index}{candidate.suffix}"))
                    for index in range(2, 42)
                    if str(candidate.with_name(f"{candidate.stem}_{index}{candidate.suffix}"))
                    not in used_paths
                )
            used_paths.add(path)
            payload["path"] = path
            resolved.append(step.model_copy(update={"target": payload}))
        return resolved

    def _derive_steps_from_objective(
        self,
        objective: str,
        context: dict[str, Any] | None,
    ) -> list[DesktopTaskStep]:
        """Derive a focus-safe plan for one or more explicit task phases."""
        segments = self._sequenced_objective_segments(objective)
        if len(segments) <= 1:
            return self._derive_single_objective_steps(objective, context)

        steps: list[DesktopTaskStep] = []
        created_folders: set[str] = set()
        used_artifact_paths: set[str] = set()
        global_preferred_browser = self._preferred_browser(objective)
        for segment in segments:
            resolved_segment = self._inherit_shared_destination(segment, objective)
            if (
                global_preferred_browser
                and not self._preferred_browser(resolved_segment)
                and (
                    self._extract_search_query(resolved_segment)
                    or self._extract_image_query(resolved_segment)
                    or self._web_document_url(resolved_segment)
                    or any(token in resolved_segment.lower() for token in ("browser", "web", "article", "source", "news"))
                )
            ):
                resolved_segment = f"{resolved_segment.rstrip(' .')}, using {global_preferred_browser}."
            segment_steps = self._derive_single_objective_steps(resolved_segment, context)
            segment_steps = self._deduplicate_segment_artifact_paths(
                segment_steps,
                used_artifact_paths,
            )
            for step in segment_steps:
                if step.action == "create_folder":
                    folder_path = str(self._target_payload(step.target).get("path") or step.target)
                    if folder_path in created_folders:
                        continue
                    created_folders.add(folder_path)
                steps.append(step)
        return steps

    def _derive_single_objective_steps(
        self,
        objective: str,
        context: dict[str, Any] | None,
    ) -> list[DesktopTaskStep]:
        text = str(objective or "").strip()
        lowered = text.lower()
        steps: list[DesktopTaskStep] = []

        # A goal to keep at comes first, because no single action satisfies one.
        #
        # LIVE 2026-08-19: "play 2048 until you get a 128 tile" reached the
        # heuristics below, matched the browser in the sentence, and was
        # planned as one open_app step. The turn then said "Done — opened
        # Google Chrome" and reported the objective complete. Every branch
        # after this point answers "what single thing is being asked for",
        # which is the wrong question about a request carrying a condition.
        watched = read_watched_goal(text)
        if watched is not None:
            # The clock is stamped here, not when the pursuit starts.
            #
            # Between planning a step and running it there is authority to
            # check and a lane to enter, and that time was free: the pursuit
            # began its own budget afterwards, so the outer deadline — which
            # only has room to report — was reached first and cancelled a run
            # that was playing correctly. LIVE 2026-08-26.
            target = watched.as_target()
            target["deadline_at"] = time.monotonic() + float(watched.max_seconds)
            return [
                DesktopTaskStep(
                    action="pursue_on_screen",
                    target=json.dumps(target),
                    reason=f"keep at it until {watched.success_when!r} is on screen",
                    expect=f"{watched.success_when} appears",
                    critical=True,
                )
            ]

        # A named file is an unambiguous instruction, and it has to be read
        # before the folder heuristics get a vote.
        #
        # Live 2026-07-27: "create a file on my Desktop called aura_hello.txt
        # containing one sentence you choose" produced a create_folder step
        # named "Aura Desktop Task 1785195330". The folder was really created,
        # so the task reported 1/1 steps completed — a true receipt for the
        # wrong action, which is worse than a failure: she then told the user
        # the objective had completed, and the only thing on the Desktop was
        # a junk folder. The word "file" was right there in the request.
        named_paths = extract_target_paths(text)
        if named_paths and not any(
            token in lowered for token in ("folder", "directory")
        ):
            named_paths = self._ordered_by_write_destination(text, named_paths)
            # If the request also asks a question ABOUT a directory, look
            # before writing. Without a read step the body is composed from
            # nothing: asked to count the .py files in a directory holding 9
            # and write the result, she wrote "Number of .py files: 0 / (No
            # files found)" into the correct destination — a true receipt for
            # a measurement nobody took.
            read_step = self._directory_read_step(text, skip=named_paths[0])
            if read_step is not None:
                # Content comes from the read via the existing step-reference
                # tokens, so the number written is the number observed. Composing
                # the body here instead would reintroduce the whole defect: a
                # correct destination holding a measurement nobody took.
                read_target = json.loads(read_step.target)
                return [
                    read_step,
                    DesktopTaskStep(
                        action="write_text_file",
                        target=json.dumps({
                            "path": named_paths[0],
                            "content": (
                                f"Files matching {read_target.get('pattern', '*')} in "
                                f"{read_target.get('path', '')}\n\n"
                                "Count: {{last.result.count}}\n\n"
                                "Names: {{last.result.names}}\n"
                            ),
                            "overwrite": True,
                        }),
                        reason="Record what the directory read actually found.",
                        expect=f"{named_paths[0]} exists on disk with the observed listing.",
                        critical=True,
                    ),
                ]
            body = self._inline_sentence_for(text) or self._document_body(text, context)
            # ADD to a file and REPLACE a file are the same path apart from the
            # verb, and collapsing them destroys the earlier content. See
            # core/skills/file_modification_intent.py.
            modification = requested_file_modification(text)
            if modification is not None:
                payload = {"path": named_paths[0], "content": body}
                if modification.mode == "append":
                    payload["append"] = True
                else:
                    payload["prepend"] = True
                return [
                    DesktopTaskStep(
                        action="write_text_file",
                        target=json.dumps(payload),
                        reason=(
                            f"The request adds to an existing file "
                            f"({modification.mode}); its content must survive."
                        ),
                        expect=(
                            f"{named_paths[0]} keeps what it held and now also "
                            "carries the requested line."
                        ),
                        critical=True,
                    ),
                ]
            return [
                DesktopTaskStep(
                    action="write_text_file",
                    target=json.dumps({"path": named_paths[0], "content": body, "overwrite": True}),
                    reason="The request names a file to create.",
                    expect=f"{named_paths[0]} exists on disk with the requested content.",
                    critical=True,
                )
            ]

        folder_name = self._extract_folder_name(text)
        root_hint = self._extract_root_hint(text)
        folder_path = f"{root_hint}/{folder_name}" if root_hint else folder_name
        wants_folder = bool(re.search(r"\b(?:folder|directory)\b", lowered))
        wants_document = bool(
            re.search(
                r"\b(?:write|summary|summarize|note|document|pdf|save|journal|"
                r"draft|essay|compose|type)\b",
                lowered,
            )
        )
        wants_pdf = self._explicit_pdf_requested(text)
        os_setting_requests = detect_os_settings(text)
        local_image_setting_values = {
            value
            for domain, value in os_setting_requests
            if (affordance := get_affordance(domain)) is not None
            and affordance.needs_image
            and Path(str(value or "")).expanduser().is_absolute()
        }
        image_setting_topic = next(
            (
                value
                for domain, value in os_setting_requests
                if (affordance := get_affordance(domain)) is not None
                and affordance.needs_image
                and not Path(str(value or "")).expanduser().is_absolute()
            ),
            "",
        )
        image_query = image_setting_topic or self._extract_image_query(text)
        explicit_image_retrieval = bool(
            re.search(r"\b(?:find|search|look\s+up|get|download|fetch)\b", lowered)
        )
        if local_image_setting_values and not explicit_image_retrieval:
            # The image is already a named local artifact. Extensions such as
            # `.png` are evidence about that value, not a request to open an
            # image-search tab or fetch another file.
            image_query = ""
        wants_image = bool(image_query) or bool(
            mentions_object_class(text, "image")
            and (not local_image_setting_values or explicit_image_retrieval)
        )
        web_document_url = self._web_document_url(text)
        image_reference_only = bool(image_query) and not any(
            token in lowered
            for token in (
                "article",
                "articles",
                "news",
                "research",
                "report",
                "reports",
                "sources",
            )
        )
        wants_search = (not image_reference_only) and (
            any(token in lowered for token in ("search", "look up", "news", "article"))
            or ("google" in lowered and not web_document_url)
        )
        # "open notes" and "notes app" were literal tokens here, so naming any
        # other application dropped out of the interactive lane and the text
        # landed in a file on disk instead of in the app the person asked for.
        # The general question is whether the objective names an application
        # you can write in, which the app answers itself.
        wants_interactive_text_entry = wants_document and (
            bool(web_document_url)
            or any(token in lowered for token in ("type", "paste", "start typing"))
            or bool(self._named_writable_app(text))
        )
        wants_artifact_file = wants_folder or wants_pdf or bool(
            re.search(r"\b(?:save|export|write|create)\b.*\b(?:file|folder|directory|pdf|artifact)\b", lowered)
        ) or (wants_document and not wants_interactive_text_entry)

        if wants_folder or wants_artifact_file:
            steps.append(
                DesktopTaskStep(
                    action="create_folder",
                    target={"path": folder_path},
                    reason="Create the requested artifact folder inside an allowed desktop root.",
                    expect="Folder exists.",
                )
            )

        apps = self._extract_apps(text)
        for app in self._generic_open_app_mentions(text):
            if app not in apps:
                apps.append(app)

        for app in apps[:4]:
            steps.append(
                DesktopTaskStep(
                    action="open_app",
                    target=app,
                    reason=f"Open {app} because the objective names that app or surface.",
                    expect=f"{app} accepts focus or reports a launch error.",
                )
            )

        preferred_browser = self._preferred_browser(text)
        engine_hint = self._search_engine_hint(text)
        browser_label = preferred_browser or "Default browser"

        def _open_url_target(url: str, *, requires_editable_focus: bool = False):
            if preferred_browser:
                payload = {"url": url, "browser": preferred_browser}
                if requires_editable_focus:
                    payload["requires_editable_focus"] = True
                return payload
            if requires_editable_focus:
                return {"url": url, "requires_editable_focus": True}
            return url

        query = self._extract_search_query(text)
        search_url = self._search_url(query, engine=engine_hint) if query else ""
        if wants_search and query:
            steps.append(
                DesktopTaskStep(
                    action="open_url",
                    target=_open_url_target(search_url),
                    reason="Open a browser/search tab for the requested live research topic.",
                    expect=f"{browser_label} accepts the search URL.",
                )
            )
            visible_source_count = self._requested_visible_source_count(text)
            if visible_source_count > 0:
                opened_source_urls: set[str] = set()
                for source in (context or {}).get("desktop_task_research_sources") or []:
                    if not isinstance(source, dict):
                        continue
                    source_url = str(source.get("url") or source.get("link") or "").strip()
                    if (
                        not source_url.startswith(("http://", "https://"))
                        or source_url in opened_source_urls
                        or source_url == search_url
                    ):
                        continue
                    opened_source_urls.add(source_url)
                    steps.append(
                        DesktopTaskStep(
                            action="open_url",
                            target=_open_url_target(source_url),
                            reason="Open one governed research source so the user can inspect the evidence visibly.",
                            expect=f"{browser_label} accepts the research source URL.",
                        )
                    )
                    if len(opened_source_urls) >= visible_source_count:
                        break
        if web_document_url:
            steps.append(
                DesktopTaskStep(
                    action="open_url",
                    target=_open_url_target(
                        web_document_url,
                        requires_editable_focus=wants_interactive_text_entry,
                    ),
                    reason="Open the requested web document surface.",
                    expect=f"{browser_label} accepts the document URL.",
                )
            )
        image_search_url = (
            self._search_url(image_query or text, images=True, engine=engine_hint)
            if wants_image
            else ""
        )
        typed_image_acquisition = bool(wants_artifact_file and image_query) or any(
            affordance is not None
            and affordance.needs_image
            and not Path(str(value or "")).expanduser().is_absolute()
            for domain, value in os_setting_requests
            if (affordance := get_affordance(domain)) is not None
        )
        open_image_search_surface = bool(
            image_search_url
            and image_search_url != search_url
            and not typed_image_acquisition
        )
        if open_image_search_surface:
            steps.append(
                DesktopTaskStep(
                    action="open_url",
                    target=_open_url_target(image_search_url),
                    reason="Open an image-search surface for the requested visual reference.",
                    expect=f"{browser_label} accepts the image search URL.",
                )
            )

        if wants_interactive_text_entry:
            body = self._document_body_with_references(
                text,
                context,
                image_query=image_query,
                image_search_url=image_search_url,
                search_url=search_url,
            )
            writing_app = "" if web_document_url else self._writing_app_from_apps(apps)

            # ASK THE APP, do not recognise it.
            #
            # This used to read `if writing_app == "Notes"`, which is one app
            # hardcoded on a machine that happens to have Notes — Bryan's
            # objection, and the right one. Every scriptable macOS app
            # publishes a dictionary describing how it holds text, so
            # text_target_for() derives the route for whatever app was named:
            # Notes answers note.body, TextEdit document.text, Reminders
            # reminder.body. None of those is written down anywhere.
            #
            # Keystrokes stay the route for an app with no dictionary, which
            # is the honest fallback rather than a special case: typing needs
            # the app to hold the front from cmd+n through cmd+v, and on a
            # real desktop the browser takes focus back mid-sequence — live
            # 2026-07-28 that failed repeatedly with "did not become
            # frontmost (observed=Google Chrome)". The app still opens
            # visibly either way, and the text is streamed in so the writing
            # is watchable, then read back to verify.
            _native_note_written = False
            _write_target = self._app_text_target(writing_app) if writing_app else ""
            if _write_target:
                _native_note_written = True
                topic = self._extract_requested_writing_topic(text)
                steps.append(
                    DesktopTaskStep(
                        action="write_in_app",
                        target={
                            "app": writing_app,
                            "title": self._note_title_for(text, topic),
                            "body": body,
                        },
                        reason=(
                            f"Write into {writing_app} through the scripting "
                            f"interface it publishes ({_write_target}), which "
                            "does not depend on window focus."
                        ),
                        expect=f"{writing_app} holds a document with the composed body.",
                    )
                )
            if (not _native_note_written) and writing_app and not (
                steps and self._step_opens_app(steps[-1], writing_app)
            ):
                steps.append(
                    DesktopTaskStep(
                        action="open_app",
                        target=writing_app,
                        reason=(
                            f"Re-focus {writing_app} immediately before text entry so "
                            "browser/image/search tabs cannot steal the paste target."
                        ),
                        expect=f"{writing_app} is frontmost before writing.",
                    )
                )
            if not (_native_note_written and not web_document_url):
                steps.append(
                    DesktopTaskStep(
                        action="set_clipboard",
                        target=body,
                        reason="Stage the CognitiveEngine-composed document body for the active writing surface.",
                        expect="Clipboard contains the composed body.",
                    )
                )
            if web_document_url:
                steps.append(
                    DesktopTaskStep(
                        action="wait",
                        target="2",
                        reason="Allow the web document surface to finish loading before paste.",
                        expect="Wait completes within the bounded desktop-task budget.",
                    )
                )
            if (not web_document_url) and (not _native_note_written) and any(
                marker in lowered for marker in ("note", "textedit", "pages", "word", "document", "journal")
            ):
                if any(step.action == "open_app" for step in steps):
                    steps.append(
                        DesktopTaskStep(
                            action="wait",
                            target="2",
                            reason=(
                                "Allow the writing app to finish launching and take "
                                "focus before keyboard staging — a cold launch loses "
                                "the shortcuts to whatever currently has focus."
                            ),
                            expect="Wait completes within the bounded desktop-task budget.",
                        )
                    )
                steps.append(
                    DesktopTaskStep(
                        action="hotkey",
                        target="command+n",
                        reason="Create a new editable note or document in the focused app.",
                        expect="The focused app accepts the new-document shortcut.",
                    )
                )
            if not (_native_note_written and not web_document_url):
                steps.append(
                    DesktopTaskStep(
                        action="hotkey",
                        target="command+v",
                        reason="Paste the staged document body into the active writing surface.",
                        expect="The focused writing surface accepts the paste shortcut.",
                    )
                )

        artifact_image_path = ""
        if wants_image and wants_artifact_file and image_query:
            artifact_image_path = f"{folder_path}/{self._safe_filename(image_query)[:40] or 'reference'}_image.png"
            steps.append(
                DesktopTaskStep(
                    action="fetch_topic_image",
                    target={"topic": image_query, "path": artifact_image_path},
                    reason="Fetch a representative image for the requested visual through the governed network gateway, with source-page evidence.",
                    expect="Image file exists with a recorded source page URL.",
                )
            )

        # General OS-setting control. The affordance registry is the single
        # source of truth for which settings Aura can drive and how; this
        # loop never names a specific setting, so a new one (volume, dark
        # mode, …) is recognized for free. Image-valued settings (wallpaper)
        # fetch their image first, through the same governed image gateway.
        for domain, value in os_setting_requests:
            affordance = get_affordance(domain)
            if affordance is None:
                continue
            local_image_path = ""
            if affordance.needs_image:
                candidate = Path(str(value or "")).expanduser()
                if candidate.is_absolute():
                    local_image_path = str(candidate)
            if affordance.needs_image and not local_image_path:
                # Save where the person said, not where the code prefers.
                #
                # "download it to my Desktop, and set it as my wallpaper" put
                # a real 1.3MB grizzly PNG in ~/Documents, because the
                # destination was hardcoded. The image was correct and the
                # wallpaper was set; it simply was not where Bryan asked for
                # it, which is the difference between following an
                # instruction and approximating one.
                image_path = (
                    f"{self._requested_image_folder(text)}/"
                    f"{self._safe_filename(value)[:40] or 'image'}_{domain}.png"
                )
                steps.append(
                    DesktopTaskStep(
                        action="fetch_topic_image",
                        target={"topic": value, "path": image_path},
                        reason=f"Fetch the image for the requested {domain} through the governed network gateway, with source-page evidence.",
                        expect="Image file exists with a recorded source page URL.",
                    )
                )
                # Not image_path: the extension is a guess until the fetch
                # reports what it was served.
                control_value = FETCHED_IMAGE_PATH_SENTINEL
            elif local_image_path:
                control_value = local_image_path
            else:
                control_value = value
            steps.append(
                DesktopTaskStep(
                    action="system_control",
                    target={"domain": domain, "value": control_value},
                    reason=f"Drive the {domain} setting to the requested value through governed System Events, recording the prior state for reversibility.",
                    expect=f"Read-back confirms the {domain} goal-state.",
                )
            )
            if affordance.needs_image and self._wants_image_source_shown(text):
                steps.append(
                    DesktopTaskStep(
                        action="open_url",
                        target=_open_url_target(FETCHED_IMAGE_SOURCE_SENTINEL),
                        reason="Show the user where the image was found (source page from the fetch receipt).",
                        expect=f"{browser_label} accepts the image source page URL.",
                    )
                )

        if wants_document and wants_artifact_file:
            body = self._document_body_with_references(
                text,
                context,
                image_query=image_query,
                image_search_url=image_search_url,
                search_url=search_url,
            )
            explicit_filename = self._extract_explicit_filename(text)
            if explicit_filename:
                filename_stem = self._safe_filename(Path(explicit_filename).stem)
                text_path = f"{folder_path}/{explicit_filename}"
            else:
                filename_stem = self._artifact_filename_stem(text)
                text_path = f"{folder_path}/{filename_stem}.txt"
            steps.append(
                DesktopTaskStep(
                    action="write_text_file",
                    target={
                        "path": text_path,
                        "content": body,
                        "overwrite": False,
                    },
                    reason="Write a durable text artifact before PDF rendering.",
                    expect="Text artifact exists with the composed body.",
                )
            )
            if wants_pdf:
                steps.append(
                    DesktopTaskStep(
                        action="render_text_pdf",
                        target={
                            "path": f"{folder_path}/{filename_stem}.pdf",
                            "title": self._artifact_document_title(text),
                            "body": body,
                            "overwrite": False,
                            **(
                                {"image_path": FETCHED_IMAGE_PATH_SENTINEL}
                                if artifact_image_path
                                else {}
                            ),
                        },
                        reason="Render the same verified text body into a PDF artifact.",
                        expect="PDF artifact exists and starts with a PDF header.",
                    )
                )
        if image_query and wants_artifact_file and self._wants_image_source_shown(text):
            steps.append(
                DesktopTaskStep(
                    action="open_url",
                    target=_open_url_target(FETCHED_IMAGE_SOURCE_SENTINEL),
                    reason="Show the user where the fetched image was found after the artifact has been created.",
                    expect=f"{browser_label} accepts the fetched image source page URL.",
                )
            )

        if not steps:
            steps.append(
                DesktopTaskStep(
                    action="read_screen_text",
                    target="",
                    reason="Observe the current desktop before attempting an underspecified action.",
                    expect="Foreground screen text or an explicit permission failure is returned.",
                )
            )
        # A delay the person asked for is part of the request, not decoration.
        # "Wait 5 seconds, then tell me what is on my screen" planned one
        # read_screen_text and answered in 1s — the observation was of the
        # wrong moment, and "Completed 1/1 governed desktop steps" reported it
        # as the whole request done. Measured live 2026-08-03.
        requested_wait = _requested_wait_seconds(text)
        if requested_wait > 0.0 and not any(step.action == "wait" for step in steps):
            steps.insert(
                0,
                DesktopTaskStep(
                    action="wait",
                    target=f"{requested_wait:g}",
                    reason=f"The request asks to wait {requested_wait:g}s before observing.",
                    expect="Wait completes within the bounded desktop-task budget.",
                ),
            )
        return steps


    @classmethod
    def _observation_evidence(
        cls, receipts: list[dict[str, Any]], objective: str, *, retain: bool = True
    ) -> ObservationEvidence | None:
        """The perception, typed as evidence gathered for THIS request.

        This is what her reasoning should receive. A raw capture in working
        memory is material a model continues — which is exactly what
        happened live on 2026-08-04, when a screen read came back as the
        verbatim accessibility dump. Typed as an Observation it arrives
        labelled, attributed, and paired with the question it was gathered
        to answer, so what she forms is an answer rather than an echo.
        """
        from core.perception.observation_evidence import (
            Observation,
            ObservationKind,
            remember_observation,
        )

        for receipt in receipts:
            action = receipt.get("action")
            if action not in {"read_screen_text", "inspect_screen"}:
                continue
            result = receipt.get("result")
            if not isinstance(result, Mapping):
                continue
            capture = str(result.get("text") or "")
            source = str(result.get("active_app") or "").strip()
            if not capture.strip() and not source:
                continue
            # Retained, so she can refer to it after this turn — a follow-up
            # ("which repo was that?"), a later comment of her own, or a
            # question about what she saw a minute ago. A perception she
            # cannot refer back to is not something she saw; it is something
            # that passed through her, and every follow-up would force
            # another capture that answers a question about the PAST with a
            # reading of the PRESENT.
            observation = Observation(
                kind=ObservationKind.SCREEN_TEXT,
                capture=capture,
                request=str(objective or ""),
                source=source,
                detail={"desktop_action": action},
            )
            # The screen reader reports the frontmost process, which for
            # this app is the launcher binary: "aura-launcher". Naming that
            # back to the person describes nothing they can see — the
            # window says "Aura". Live 2026-08-04 she answered "the front
            # window is aura-launcher", which is the executable, not the
            # app anyone is looking at.
            windows = observation.windows()
            if windows and windows[0].get("app"):
                observation.source = str(windows[0]["app"]).strip()
            # `retain=False` is for callers that only want to PHRASE what
            # was seen (the summary describer). Retaining there too would
            # record the same look twice and push a real earlier
            # observation out of a deliberately short history.
            return remember_observation(observation) if retain else observation
        return None

    @classmethod
    def _describe_screen_observation(cls, receipts: list[dict[str, Any]]) -> str:
        """A grounded FALLBACK description, for when reasoning cannot run.

        Not the primary path. Her reasoning receives the typed Observation
        and forms the answer; this exists so that a turn which loses the
        cognitive lane still says something true about the screen instead of
        pasting the buffer or reporting a step count.

        Measured live 2026-08-04. Bryan asked "can you tell me what you see
        on the screen?" and got the raw accessibility dump back verbatim —
        "Edit / Window / (9) Kurzgesagt / ... / Show more / You >" — a
        transcription of the UI tree, not an answer. Reading the screen and
        being able to say what is on it are different acts, and only the
        second is what was asked for.

        The raw text is EVIDENCE. It stays in the receipt for anything that
        needs to verify the claim; it is not the reply.

        Deterministic on purpose: this runs inside a governed desktop step
        on the foreground lane, and spending a second model generation to
        narrate a screenshot is exactly the kind of hidden allocation that
        turns one observation into a stalled turn. What it produces is a
        grounded factual description — which app is in front, what is
        identifiable in it — and the response lane is free to phrase it.
        """
        observation = cls._observation_evidence(receipts, "", retain=False)
        return observation.describe() if observation is not None else ""

    @staticmethod
    def _primitive_steps_are_only_observational(steps: list[DesktopTaskStep]) -> bool:
        if not steps:
            return True
        non_effect_actions = {"inspect_screen", "read_screen_text", "wait", "get_clipboard"}
        return all(step.action in non_effect_actions for step in steps)

    @staticmethod
    def _objective_requests_observation_only(
        objective: str, previous_user_request: str = ""
    ) -> bool:
        """Delegates to the one shared definition.

        This was a literal-substring list, and it disagreed with the regex the
        router already used. "Can you see what's on the screen and tell me what
        you see?" said "the screen" where the list said "my screen", so a read
        escalated into os_automation and came back refused for having no
        observable acceptance contract. See looks_like_screen_observation.

        ``previous_user_request`` restores the antecedent for a follow-up. Live,
        Bryan asked for a screen read, was refused, and then said "Can you do it
        now?" and "Yes you can lol" — neither contains a screen noun, so both
        classified as not-an-observation and he was refused twice more. "It" was
        the screen read; the request was in the previous turn.
        """
        if looks_like_screen_observation(objective):
            return True
        if not previous_user_request:
            return False
        from core.runtime.referential_continuation import effective_message

        resolved = effective_message(
            objective, previous_user_request=previous_user_request
        )
        return resolved.resolved and looks_like_screen_observation(resolved.text)

    @staticmethod
    def _objective_needs_general_os_automation(objective: str) -> bool:
        return bool(
            re.search(
                r"\b(?:arrange|resize|drag|focus|select|switch|close|"
                r"minimi[sz]e|maximi[sz]e|organize|click|press|type|paste|"
                r"enter|fill|choose)\b",
                str(objective or ""),
                flags=re.IGNORECASE,
            )
        )

    @staticmethod
    def _steps_cover_general_os_intent(objective: str, steps: list[DesktopTaskStep]) -> bool:
        lowered = str(objective or "").lower()
        actions = {step.action for step in steps}
        if re.search(r"\b(?:click|press|choose|select|enter)\b", lowered):
            if not actions & {"click", "hotkey", "run_applescript"}:
                return False
        if re.search(r"\b(?:type|paste|fill|write)\b", lowered):
            # create_note writes text through an app's scripting interface —
            # it covers a "write" objective as completely as a paste does,
            # and verifies itself afterwards, which a paste cannot. Without
            # it here a natively-written note looked like uncovered intent
            # and escalated to blind OS automation.
            if not actions & {
                "type",
                "set_clipboard",
                "hotkey",
                "run_applescript",
                "write_text_file",
                "write_in_app",
                "create_note",
            }:
                return False
        if re.search(r"\b(?:arrange|resize|drag|minimi[sz]e|maximi[sz]e|organize)\b", lowered):
            if "run_applescript" not in actions:
                return False
        if re.search(r"\b(?:focus|switch|close)\b", lowered):
            if not actions & {"open_app", "hotkey", "run_applescript"}:
                return False
        return True

    @staticmethod
    def _steps_cover_durable_artifact_intent(objective: str, steps: list[DesktopTaskStep]) -> bool:
        """Prefer verified primitives for file/document/PDF objectives.

        Free-form OS automation is useful for true window/UI manipulation,
        but it is the least deterministic lane. If the planner already
        derived a bounded artifact plan with read-backable effects, do not
        discard it merely because the natural-language objective also says
        "click", "copy", or "type".
        """
        lowered = str(objective or "").lower()
        actions = {step.action for step in steps}
        if not actions:
            return False
        # An objective that names a concrete path IS a durable-artifact
        # objective, whatever else the sentence says. Without this, "create a
        # file on my Desktop called aura_hello.txt containing one sentence you
        # choose" matched none of the document tokens below, so the derived
        # write_text_file plan was discarded and the turn was escalated to
        # AppleScript on the strength of the word "choose" — where a file has
        # no observable postcondition and the objective was refused outright.
        # Same path extractor the effect contract uses, so router and verifier
        # cannot disagree about what the objective is about.
        target_paths = extract_target_paths(objective)
        wants_file = bool(target_paths) or bool(re.search(r"\bfiles?\b", lowered))
        wants_folder = any(token in lowered for token in ("folder", "directory"))
        wants_document = any(
            token in lowered
            for token in (
                "write",
                "summary",
                "summarize",
                "note",
                "document",
                "doc",
                "pdf",
                "save",
                "journal",
                "essay",
                "report",
                "artifact",
            )
        )
        wants_pdf = "pdf" in lowered or bool(
            re.search(r"\b(?:export|save)\b[^.\n]{0,60}\bas\s+(?:a\s+)?pdf\b", lowered)
        )
        if not (wants_folder or wants_document or wants_pdf or wants_file):
            return False
        if wants_file and not actions & {
            "write_text_file",
            "render_text_pdf",
            "move_file",
            "create_folder",
        }:
            return False
        if wants_folder and "create_folder" not in actions:
            return False
        # create_note produces a durable, read-backable document — the note
        # exists in Notes afterwards and the executor confirms it. Leaving it
        # out here made a natively-written note look like an uncovered
        # document objective, which discarded a good bounded plan in favour
        # of blind OS automation.
        if wants_document and not actions & {
            "write_text_file",
            "set_clipboard",
            "render_text_pdf",
            "write_in_app",
            "create_note",
        }:
            return False
        if wants_pdf and "render_text_pdf" not in actions:
            return False
        if (
            mentions_object_class(objective, "image")
            and "fetch_topic_image" not in actions
            and "open_url" not in actions
        ):
            return False
        # A note IS a durable artifact: it persists in Notes and can be read
        # back, which is the property this list is testing for.
        return any(
            action in actions
            for action in (
                "create_folder",
                "write_text_file",
                "render_text_pdf",
                "move_file",
                "fetch_topic_image",
                "write_in_app",
                "create_note",
            )
        )

    @staticmethod
    def _steps_cover_visible_writing_intent(objective: str, steps: list[DesktopTaskStep]) -> bool:
        """Keep visible writing chains on verified primitives.

        Mixed browser/native requests are common live-demo and daily-use tasks.
        Escalating them to one generated AppleScript blob removes per-step focus
        evidence and was the source of URL-bar pastes. If the derived primitive
        plan already opens the requested surfaces, stages text, and performs a
        paste/type action, keep the chain auditable.
        """
        lowered = str(objective or "").lower()
        if not re.search(r"\b(?:write|type|paste|compose|draft|summari[sz]e|note|doc|document)\b", lowered):
            return False
        actions = [step.action for step in steps]
        action_set = set(actions)
        if not action_set:
            return False
        opens_surface = bool(action_set & {"open_app", "open_url"})
        stages_text = "set_clipboard" in action_set
        commits_text = "type" in action_set or any(
            step.action == "hotkey" and "v" in str(step.target).lower()
            for step in steps
        )
        return opens_surface and stages_text and commits_text

    @staticmethod
    def _objective_requires_true_window_automation(objective: str) -> bool:
        return bool(
            re.search(
                r"\b(?:arrange|resize|drag|minimi[sz]e|maximi[sz]e|organize|tile|snap)\b",
                str(objective or ""),
                flags=re.IGNORECASE,
            )
        )

    @classmethod
    def _should_escalate_to_os_automation(
        cls,
        objective: str,
        steps: list[DesktopTaskStep],
        context: dict[str, Any] | None,
    ) -> bool:
        context = context or {}
        if bool(context.get("disable_os_automation_escalation")):
            return False
        if cls._objective_requests_observation_only(objective):
            return False
        # Multi-surface/multi-app objectives require coordinated focus and
        # clipboard control. Prefer verified primitives when they already cover
        # the durable artifact and UI intent; escalate only when coverage is
        # incomplete. This keeps demo-class tasks in the receipt-producing lane
        # instead of hiding them behind a single generated AppleScript blob.
        if cls._steps_cover_durable_artifact_intent(objective, steps) and cls._steps_cover_general_os_intent(objective, steps):
            return False
        lowered_obj = objective.lower()
        has_browser_or_web = any(
            re.search(rf"\b{re.escape(marker)}s?\b", lowered_obj)
            for marker in ("chrome", "browser", "safari", "web", "url", "doc", "sheet", "slide", "drive", "notion")
        )
        has_local_editor_or_file = any(
            re.search(rf"\b{re.escape(marker)}s?\b", lowered_obj)
            for marker in ("note", "textedit", "pages", "word", "document", "folder", "desktop", "journal", "file")
        )
        if has_browser_or_web and has_local_editor_or_file:
            if (
                cls._steps_cover_visible_writing_intent(objective, steps)
                and not cls._objective_requires_true_window_automation(objective)
            ):
                return False
            return True
        if cls._objective_needs_general_os_automation(objective) and not any(
            step.action == "run_applescript" for step in steps
        ):
            if (
                cls._steps_cover_durable_artifact_intent(objective, steps)
                and not cls._objective_requires_true_window_automation(objective)
            ):
                return False
            if cls._steps_cover_general_os_intent(objective, steps):
                return False
            return looks_like_desktop_objective(objective) or any(
                step.action in {"open_app", "open_url"} for step in steps
            )
        if not cls._primitive_steps_are_only_observational(steps):
            return False
        return looks_like_desktop_objective(objective)



    # A step proves the step's effect. The task proves the task's.
    #
    # Child contexts were built with `dict(task_context)`, which carries the
    # task-level action expectation down into every step — so a `create_folder`
    # step was required to produce `steps_requested` and `steps_completed`,
    # fields only the task result has. It could not, so the contract layer
    # failed it with "expectation incomplete: steps_requested; steps_completed"
    # and the whole objective died on a step that had, in fact, worked.
    # Measured live 2026-07-27 on "create a file on my Desktop".
    #
    # No step of any desktop objective can satisfy a task-level contract, so
    # this was never one action misbehaving — it was every multi-step desktop
    # objective inheriting a contract its parts cannot meet.
    _TASK_LEVEL_EXPECTATION_KEYS: tuple[str, ...] = (
        "action_expectation",
        "expectation",
        "acceptance_criteria",
        "criteria",
        "required_evidence",
        "evidence_required",
        "required_evidence_present",
        "semantic_predicates",
        "user_visible_effect",
        "visible_effect",
        "repair_hint",
        "rollback_hint",
        "allow_partial",
    )

    #: What the TASK is authorised for, which is not what a STEP is authorised
    #: for.
    #:
    #: A scope belongs to an action. The task's scope is the widest thing in
    #: its plan; a step that types a key is foreground control and a step that
    #: writes a file is file io, and each step's lease is issued for its own.
    #: Inherited, the parent's value is presented against the child's lease
    #: and the two can never agree.
    #:
    #: LIVE 2026-08-26: "make a file on my Desktop called aura_note.txt" was
    #: refused — lease held 'desktop_file_io', the call derived
    #: 'foreground_desktop_control', which is the task's declared scope
    #: arriving through the step context. She could not write a file at all.
    _TASK_LEVEL_AUTHORITY_KEYS: tuple[str, ...] = (
        "effect_scope",
        "risk_level",
    )

    @classmethod
    def _child_step_context(cls, task_context: dict[str, Any] | None) -> dict[str, Any]:
        """A step's context, without the contract that belongs to the task."""
        child = dict(task_context or {})
        for key in (*cls._TASK_LEVEL_EXPECTATION_KEYS, *cls._TASK_LEVEL_AUTHORITY_KEYS):
            child.pop(key, None)
        return child

    @staticmethod
    def _failure_cause(failures: list[dict[str, Any]], *, objective: str = "") -> str:
        """Why the desktop task failed, in the words of the step that failed.

        Every failing receipt already knows: the step's action, what it expected,
        the effect evidence, and the child result's own error. None of that was
        lifted into the skill's `error` field, so BaseSkill fell back to
        "desktop_task reported failure without a cause (status=failed)" — which
        is what reached Bryan, twice, for "create a file on my Desktop". An
        undiagnosable failure is barely better than a silent one: he cannot act
        on it, she cannot explain it, and the surprise engine banks a
        maximal-surprise signal carrying no information.
        """
        for receipt in failures or []:
            if not isinstance(receipt, dict):
                continue
            result = receipt.get("result")
            detail = ""
            if isinstance(result, dict):
                detail = str(
                    result.get("error") or result.get("status") or result.get("reason") or ""
                ).strip()
            if not detail:
                detail = str(receipt.get("effect_evidence") or "").strip()
            action = str(receipt.get("action") or "step").strip()
            expected = str(receipt.get("expect") or "").strip()
            if detail:
                suffix = f" (expected: {expected})" if expected else ""
                return f"{action} failed: {detail}{suffix}"[:400]
            if expected:
                return f"{action} did not produce its expected effect: {expected}"[:400]
            return f"{action} failed without reporting why"
        if objective:
            return (
                "no step reported a failure, yet the objective was not verified as "
                f"complete: {objective[:160]}"
            )
        return "the desktop task did not complete and no step reported a cause"

    @staticmethod
    def _os_automation_effect_evidence(result: dict[str, Any]) -> tuple[bool, str]:
        if not bool(result.get("ok")):
            return False, str(result.get("error") or result.get("status") or "os automation reported failure")
        if not bool(result.get("effect_verified")):
            return False, "os automation did not verify the requested effect"
        contract = result.get("effect_contract")
        if not isinstance(contract, dict) or not bool(contract.get("verifiable")):
            return False, "missing verifiable os automation effect contract"
        checks = result.get("verification_results")
        if not isinstance(checks, list) or not checks:
            return False, "missing structured os automation verification checks"
        failed_required = any(
            isinstance(check, dict)
            and bool(check.get("required", True))
            and not bool(check.get("passed"))
            for check in checks
        )
        strong_passed = any(
            isinstance(check, dict)
            and bool(check.get("passed"))
            and bool(check.get("strong", True))
            for check in checks
        )
        if failed_required or not strong_passed:
            return False, "os automation checks do not prove every required strong effect"
        effect_evidence = str(result.get("effect_evidence") or "").strip()
        if effect_evidence and not effect_evidence.startswith("receipt_id="):
            return True, effect_evidence[:240]
        receipt_id = str(result.get("receipt_id") or "").strip()
        if receipt_id:
            return False, (
                f"receipt_id={receipt_id} is audit evidence only; missing observable "
                "verification proving the requested desktop effect."
            )
        return False, "missing objective-specific os automation effect evidence"

    @classmethod
    def _semantic_completion_contract(cls, objective: str) -> ActionExpectation:
        """Compile the requested outcome into evidence-backed task predicates."""

        from core.runtime.skill_contract import (
            ActionExpectation,
            PredicateOperator,
            SemanticPredicate,
        )

        predicates = [
            SemanticPredicate(
                predicate_id="all_planned_effects_verified",
                evidence_path="semantic_evidence.mechanical.all_effects_verified",
                operator=PredicateOperator.TRUTHY,
                description="Every required planned effect completed and was observed.",
                repair_hint="repair_failed_or_missing_desktop_effects",
            )
        ]
        if cls._objective_requests_research_document(objective):
            required_sources = max(1, cls._requested_research_source_count(objective))
            requires_reading = cls._objective_requests_source_reading(objective)
            predicates.append(
                SemanticPredicate(
                    predicate_id=(
                        "requested_source_count_read"
                        if requires_reading
                        else "requested_source_count_found"
                    ),
                    evidence_path=(
                        "semantic_evidence.research.read_source_count"
                        if requires_reading
                        else "semantic_evidence.research.distinct_source_count"
                    ),
                    operator=PredicateOperator.GREATER_THAN_OR_EQUAL,
                    expected=required_sources,
                    description=(
                        "The requested number of distinct article bodies were read."
                        if requires_reading
                        else "The requested number of distinct article links were verified."
                    ),
                    repair_hint=(
                        "replace_unreadable_sources_and_read_article_bodies"
                        if requires_reading
                        else "find_additional_distinct_article_sources"
                    ),
                )
            )
            if cls._objective_requests_authored_synthesis(objective):
                predicates.extend(
                    [
                        SemanticPredicate(
                        predicate_id="cross_source_synthesis_present",
                        evidence_path="semantic_evidence.research.synthesis_present",
                        operator=PredicateOperator.TRUTHY,
                        description="The artifact contains a completed cross-source synthesis.",
                        repair_hint="author_and_reverify_cross_source_synthesis",
                        ),
                        SemanticPredicate(
                            predicate_id="synthesis_authored_by_cortex",
                            evidence_path="semantic_evidence.research.authored_synthesis",
                            operator=PredicateOperator.TRUTHY,
                            description="Aura authored the requested synthesis rather than emitting extraction or a template.",
                            repair_hint="rerun_cortex_authorship_with_semantic_feedback",
                        ),
                        SemanticPredicate(
                            predicate_id="synthesis_bound_to_read_sources",
                            evidence_path="semantic_evidence.research.bound_read_source_count",
                            operator=PredicateOperator.GREATER_THAN_OR_EQUAL,
                            expected=required_sources,
                            description=(
                                "The authored synthesis receipt is bound to every "
                                "required fetched article body."
                            ),
                            repair_hint="reauthor_synthesis_from_verified_article_bodies",
                        ),
                    ]
                )
                if cls._explicit_pdf_requested(objective):
                    predicates.append(
                        SemanticPredicate(
                            predicate_id="pdf_contains_authored_synthesis",
                            evidence_path=(
                                "semantic_evidence.artifacts."
                                "pdf_contains_authored_synthesis"
                            ),
                            operator=PredicateOperator.TRUTHY,
                            description=(
                                "The persisted PDF content receipt includes every "
                                "paragraph of Aura's authored synthesis."
                            ),
                            repair_hint="rerender_pdf_with_verified_authored_synthesis",
                        )
                    )
            if cls._objective_requests_recent_sources(objective):
                predicates.append(
                    SemanticPredicate(
                        predicate_id="requested_sources_recent",
                        evidence_path="semantic_evidence.research.recent_source_count",
                        operator=PredicateOperator.GREATER_THAN_OR_EQUAL,
                        expected=required_sources,
                        description="Publication evidence verifies the requested sources are recent.",
                        repair_hint="replace_sources_without_recent_publication_evidence",
                    )
                )
            if cls._objective_requests_opinion(objective):
                predicates.append(
                    SemanticPredicate(
                        predicate_id="independent_position_present",
                        evidence_path="semantic_evidence.research.independent_position_present",
                        operator=PredicateOperator.TRUTHY,
                        description="The requested first-person assessment is present and passed the content contract.",
                        repair_hint="form_and_write_an_evidence_grounded_first_person_position",
                    )
                )
        lowered = str(objective or "").casefold()
        if cls._explicit_pdf_requested(objective):
            predicates.append(
                SemanticPredicate(
                    predicate_id="requested_pdf_verified",
                    evidence_path="semantic_evidence.artifacts.verified_pdf_count",
                    operator=PredicateOperator.GREATER_THAN_OR_EQUAL,
                    expected=1,
                    description="At least one non-empty PDF was rendered and read back.",
                    repair_hint="render_and_read_back_requested_pdf",
                )
            )
        if "folder" in lowered or "directory" in lowered:
            predicates.append(
                SemanticPredicate(
                    predicate_id="requested_folder_verified",
                    evidence_path="semantic_evidence.artifacts.requested_folder_verified",
                    operator=PredicateOperator.TRUTHY,
                    description="The requested folder exists at the requested location.",
                    repair_hint="create_and_read_back_requested_folder",
                )
            )
            if cls._explicit_pdf_requested(objective):
                predicates.append(
                    SemanticPredicate(
                        predicate_id="pdf_saved_in_requested_folder",
                        evidence_path="semantic_evidence.artifacts.pdf_in_requested_folder",
                        operator=PredicateOperator.TRUTHY,
                        description="The verified PDF path is inside the requested folder.",
                        repair_hint="move_or_render_pdf_into_requested_folder_and_read_back",
                    )
                )
        for domain, _value in detect_os_settings(objective):
            predicates.append(
                SemanticPredicate(
                    predicate_id=f"requested_{domain}_verified",
                    evidence_path=(
                        f"semantic_evidence.os_settings.{domain}.verified"
                    ),
                    operator=PredicateOperator.TRUTHY,
                    description=(
                        f"Read-back confirms the requested {domain} goal-state."
                    ),
                    repair_hint=f"apply_and_read_back_requested_{domain}",
                )
            )
        return ActionExpectation(
            objective=objective,
            semantic_predicates=predicates,
            repair_hint="repair_unsatisfied_desktop_task_predicates",
            rollback_hint="preserve_verified_effects_and_repair_only_missing_predicates",
            allow_partial=True,
        )

    @classmethod
    def _semantic_completion_evidence(
        cls,
        *,
        objective: str,
        task_context: Mapping[str, Any],
        receipts: list[dict[str, Any]],
        all_effects_verified: bool,
    ) -> dict[str, Any]:
        sources = [
            dict(item)
            for item in (task_context.get("desktop_task_research_sources") or [])
            if isinstance(item, Mapping)
        ]
        read_sources = [item for item in sources if item.get("read_verified") is True]
        recent_sources = [item for item in read_sources if item.get("recency_verified") is True]
        synthesis = str(task_context.get("desktop_task_research_synthesis") or "").strip()
        synthesis_digest = str(
            task_context.get("desktop_task_research_synthesis_sha256") or ""
        ).strip()
        synthesis_receipt_valid = bool(
            synthesis
            and cls._valid_sha256(synthesis_digest)
            and synthesis_digest == text_sha256(synthesis)
        )
        authored = bool(
            task_context.get("desktop_task_research_authored") is True
            and synthesis_receipt_valid
        )
        read_source_hashes = {
            str(item.get("source_evidence_sha256") or "")
            for item in read_sources
            if cls._valid_sha256(str(item.get("source_evidence_sha256") or ""))
        }
        bound_source_hashes = {
            str(value)
            for value in (
                task_context.get(
                    "desktop_task_research_synthesis_source_sha256s"
                )
                or []
            )
            if cls._valid_sha256(str(value))
        }
        bound_read_source_count = len(read_source_hashes & bound_source_hashes)
        independent_position = bool(
            authored
            and cls._objective_requests_opinion(objective)
            and cls._research_synthesis_satisfies_objective(objective, synthesis)
        )

        pdf_paths: list[str] = []
        pdf_contains_authored_synthesis = False
        folder_paths: list[str] = []
        os_settings = {
            domain: {"verified": False}
            for domain, _value in detect_os_settings(objective)
        }
        synthesis_paragraphs = paragraph_sha256s(synthesis)
        for receipt in receipts:
            result = receipt.get("result")
            result = result if isinstance(result, Mapping) else {}
            receipt_verified = bool(
                receipt.get("ok") is True
                and receipt.get("effect_verified") is True
            )
            if receipt.get("action") == "system_control":
                domain = str(result.get("domain") or "").strip().casefold()
                if domain in os_settings:
                    os_settings[domain] = {
                        "verified": bool(
                            receipt_verified
                            and result.get("effect_verified") is True
                        ),
                        "applied": result.get("applied"),
                        "value": result.get("value"),
                    }
            if not receipt_verified:
                continue
            path = str(result.get("path") or "").strip()
            if receipt.get("action") == "render_text_pdf" and path.lower().endswith(".pdf"):
                pdf_paths.append(path)
                pdf_contains_authored_synthesis = bool(
                    pdf_contains_authored_synthesis
                    or contains_paragraph_hashes(
                        result.get("source_paragraph_sha256s") or [],
                        synthesis_paragraphs,
                    )
                )
            if receipt.get("action") == "create_folder" and path:
                folder_paths.append(path)

        lowered = str(objective or "").casefold()
        wants_folder = "folder" in lowered or "directory" in lowered
        requested_folder_verified = not wants_folder
        pdf_in_requested_folder = not (wants_folder and cls._explicit_pdf_requested(objective))
        if wants_folder:
            exact_folder_requested = cls._has_explicit_folder_name(objective)
            if exact_folder_requested:
                folder_name = cls._extract_folder_name(objective)
                root = cls._extract_root_hint(objective)
                requested = Path(f"{root}/{folder_name}" if root else folder_name).expanduser()

                def _matches_requested(path: str) -> bool:
                    candidate = Path(path).expanduser()
                    if not root:
                        return candidate.name.casefold() == requested.name.casefold()
                    try:
                        return candidate == requested or candidate.resolve() == requested.resolve()
                    except OSError:
                        return str(candidate).casefold().rstrip("/") == str(requested).casefold().rstrip("/")

            else:
                requested_folders = {
                    str(Path(path).expanduser()).casefold().rstrip("/")
                    for path in folder_paths
                }

                def _matches_requested(path: str) -> bool:
                    return str(Path(path).expanduser()).casefold().rstrip("/") in requested_folders

            requested_folder_verified = any(_matches_requested(path) for path in folder_paths)
            if pdf_paths:
                pdf_in_requested_folder = all(
                    _matches_requested(str(Path(path).expanduser().parent)) for path in pdf_paths
                )

        return {
            "mechanical": {"all_effects_verified": all_effects_verified},
            "os_settings": os_settings,
            "research": {
                "distinct_source_count": len(
                    {
                        str(item.get("url") or item.get("title") or "").casefold()
                        for item in sources
                        if str(item.get("url") or item.get("title") or "").strip()
                    }
                ),
                "read_source_count": len(read_sources),
                "recent_source_count": len(recent_sources),
                "synthesis_present": bool(
                    synthesis
                    and cls._research_synthesis_satisfies_objective(objective, synthesis)
                ),
                "authored_synthesis": authored,
                "synthesis_receipt_valid": synthesis_receipt_valid,
                "bound_read_source_count": bound_read_source_count,
                "independent_position_present": independent_position,
                "source_evidence": [
                    {
                        "title": item.get("title"),
                        "url": item.get("url"),
                        "read_verified": item.get("read_verified"),
                        "read_evidence_kind": item.get("read_evidence_kind"),
                        "article_body_sha256": item.get("article_body_sha256"),
                        "source_evidence_sha256": item.get("source_evidence_sha256"),
                        "recency_verified": item.get("recency_verified"),
                        "recency_evidence": item.get("recency_evidence"),
                    }
                    for item in sources
                ],
            },
            "artifacts": {
                "verified_pdf_count": len(pdf_paths),
                "pdf_paths": pdf_paths,
                "pdf_contains_authored_synthesis": pdf_contains_authored_synthesis,
                "requested_folder_verified": requested_folder_verified,
                "folder_paths": folder_paths,
                "pdf_in_requested_folder": pdf_in_requested_folder,
            },
        }

    async def _execute_os_automation_escalation(
        self,
        *,
        capability_engine: Any,
        objective: str,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        step_context = self._child_step_context(context)
        document_body = (
            self._document_body(objective, step_context)
            if self._objective_requests_written_artifact(objective)
            else ""
        )
        step_context.update(
            {
                "origin": step_context.get("origin") or "desktop_task",
                "route": "desktop_task.os_automation",
                "objective": objective,
                "foreground_request": True,
                "user_requested_action": True,
                "user_explicitly_authorized": True,
                "user_visible_desktop_action": True,
                "local_desktop_action": True,
                "desktop_task_reason": (
                    "Primitive desktop actions were not sufficient for this objective; "
                    "escalating to governed OS automation."
                ),
                "desktop_task_expect": (
                    "OS automation returns a verifiable effect contract with every required "
                    "strong objective-specific check passed."
                ),
                "desktop_task_document_body": document_body,
                "document_body": document_body,
            }
        )
        try:
            result = await capability_engine.execute(
                "os_automation",
                {"goal": objective, "script_type": "applescript", "execute": True},
                context=step_context,
            )
        except (AttributeError, RuntimeError, TypeError, ValueError, OSError, TimeoutError) as exc:
            record_degradation(
                "desktop_task",
                exc,
                action="blocked desktop task because governed OS automation escalation failed closed",
                severity="degraded",
            )
            result = {
                "ok": False,
                "status": "os_automation_unavailable",
                "error": str(exc),
            }
        if not isinstance(result, dict):
            result = {"ok": bool(result), "result": result}

        effect_verified, effect_evidence = self._os_automation_effect_evidence(result)
        receipt = {
            "index": 1,
            "action": "os_automation",
            "reason": step_context["desktop_task_reason"],
            "expect": step_context["desktop_task_expect"],
            "ok": bool(result.get("ok")) and effect_verified,
            "effect_verified": effect_verified,
            "effect_evidence": effect_evidence,
            "result": result,
        }
        ok = bool(receipt["ok"])
        return {
            "ok": ok,
            "status": "completed" if ok else "failed",
            **({} if ok else {"error": self._failure_cause([receipt], objective=objective)}),
            "objective": objective,
            "steps_requested": 1,
            "steps_completed": 1 if ok else 0,
            "receipts": [receipt],
            "failures": [] if ok else [receipt],
            "planner": "os_automation_escalation",
            "summary": (
                "Desktop task completed 1/1 governed OS automation step."
                if ok
                else "Desktop task could not complete through primitive actions or governed OS automation."
            ),
        }

    def _ambient_answer(self, objective: str, params: Any) -> dict[str, Any] | None:
        """Answer from what she already saw, or None to capture normally.

        None is the pre-ambient behaviour, so every failure path here simply
        falls through to a real capture. That is the property that makes the
        fast path safe to add: it can only ever remove latency, never remove
        an answer.
        """
        if getattr(params, "steps", None):
            # An explicit plan is a request to DO something, not to know.
            return None
        if not looks_like_screen_observation(objective):
            return None
        if self._objective_needs_general_os_automation(objective):
            # There is an action in here; the screen is about to change.
            return None
        try:
            from core.perception.ambient_presence import get_ambient_presence

            observation = get_ambient_presence().fresh_observation_for(objective)
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
            return None
        if observation is None:
            return None
        # Observation stamps its capture time as `at`. This read `timestamp`,
        # which no Observation has ever carried, so the getattr default of 0.0
        # made every ambient answer report an age of ~56 years (now - epoch).
        # The one number whose whole job is to keep a moment-old reading from
        # passing as this instant was nonsense on every turn it appeared.
        observed_at = float(
            getattr(observation, "at", None)
            or getattr(observation, "timestamp", 0.0)
            or 0.0
        )
        age_s = (
            round(max(0.0, time.time() - observed_at), 1) if observed_at > 0.0 else None
        )
        # Observing IS the step. Reporting zero steps was not a smaller claim,
        # it was an unverifiable one: the task contract checks
        # steps_requested/steps_completed/receipts by TRUTHINESS, so 0/0/[]
        # read as "no evidence" and the contract layer downgraded a correct
        # screen reading into "I am not claiming the desktop action finished."
        # Measured live 2026-08-10 on "what's on my screen right now?".
        #
        # A request to observe is satisfied by an observation, not by an
        # effect. This receipt says exactly that — the action is a read, and
        # the evidence is the reading, carrying its source and its age.
        source = str(getattr(observation, "source", "") or "the frontmost window")
        when = f"{age_s}s ago" if age_s is not None else "at an unrecorded time"
        if getattr(observation, "is_empty", False):
            evidence = (
                f"observed {source} {when}; nothing legible was captured. "
                "This is a failed READING, not an empty screen."
            )
        else:
            evidence = f"observed {source} {when}; legible content captured"
        receipt = {
            "index": 1,
            "action": "observe_screen",
            "reason": "answer a question about the screen from what was seen",
            "expect": "an observation of the frontmost window",
            "ok": True,
            # The read is verified by the observation existing with a source
            # and an age. No desktop state was changed and none is claimed.
            "effect_verified": True,
            "effect_evidence": evidence,
            "result": {
                "source": source,
                "age_s": age_s,
                "empty": bool(getattr(observation, "is_empty", False)),
            },
        }
        return {
            "ok": True,
            "status": "answered_from_ambient_observation",
            "objective": objective,
            "steps_requested": 1,
            "steps_completed": 1,
            "receipts": [receipt],
            "observation": observation.for_reasoning(),
            "observation_meta": observation.to_dict(),
            # Named, so an answer sourced from a moment ago is never mistaken
            # for a reading of this instant. The age travels with it.
            "observation_age_s": age_s,
            "captured_now": False,
        }

    async def _attach_pointing(
        self, objective: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        """If she was asked to POINT, try to; either way say which happened.

        "Where is the submit button?" is answered better by a rectangle than
        by a paragraph, and the overlay path existed with no caller — every
        such question got the paragraph because nothing ever asked for the
        rectangle.

        The refusal is carried on the payload as deliberately as the success.
        A highlight that did not draw must reach the answer as a fact, or she
        says "I've highlighted it" over a screen with no highlight on it,
        which sends the person hunting for a box that was never there.
        """
        needle = asks_to_be_shown_where(objective)
        if not needle:
            return payload
        try:
            from core.perception.screen_highlight import highlight

            result = await highlight(needle, requested=True)
        except (ImportError, AttributeError, RuntimeError, OSError, TypeError, ValueError):
            # Pointing is an enhancement to an answer she already has. It
            # never costs her the answer.
            return payload
        payload["highlight"] = result.to_dict()
        payload["pointed_at"] = result.matched_text if result.shown else ""
        payload["pointing_refused_because"] = "" if result.shown else result.reason
        return payload

    async def _delegate_page_objective(
        self, params: Any, context: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Hand a web-page objective to the browser, or return None.

        Returns None for everything else, so the GUI lane is untouched by
        objectives that are genuinely about the desktop.
        """

        objective = str(getattr(params, "objective", "") or "")
        try:
            from core.conversation.page_interaction import page_interaction_target

            url = page_interaction_target(objective)
        except (ImportError, AttributeError, TypeError, ValueError):
            return None
        if not url:
            return None

        # Through the governed executor, never the skill object directly.
        #
        # A direct `SovereignBrowserSkill().execute(...)` reached the browser
        # without the scoped authority the domain requires, and the will
        # refused it on arrival: "WILL REFUSED: desktop_ui/network_call --
        # denied_by_default: network_call requires validated scoped authority".
        # Correct refusal — the grant is what makes the lease, the receipt and
        # the origin check mean anything, and a delegation that skips it is
        # asking the browser to act on nobody's authority.
        try:
            from core.container import ServiceContainer

            capability_engine = ServiceContainer.get("capability_engine", default=None)
        except (ImportError, AttributeError, RuntimeError) as exc:
            record_degradation("desktop_task.page_objective", exc, severity="warning")
            return None
        if capability_engine is None or not hasattr(capability_engine, "execute"):
            return None

        logger.info(
            "🌐 Desktop objective names a page to work through (%s); delegating to the browser body.",
            url,
        )
        try:
            # `_child_step_context`, not `dict(context)`.
            #
            # The task-level action expectation rides in the context, and a
            # child action cannot satisfy it: this lane's contract asks for
            # `steps_requested` and `steps_completed`, which only the TASK
            # result has. Passing the raw context down handed the browser a
            # contract about desktop steps, so a pursuit that worked came back
            # "expectation incomplete: steps_requested; steps_completed" —
            # printed, in the same sentence, next to "Completed 1/1 steps".
            #
            # This is the defect this file already documents from 2026-07-27,
            # where a `create_folder` step inherited the same contract and
            # killed an objective that had worked. The helper that strips those
            # keys was written for it; the delegation simply was not using it.
            report = await capability_engine.execute(
                "sovereign_browser",
                {"mode": "pursue", "url": url, "goal": objective},
                context=self._child_step_context(context),
            )
            report = report if isinstance(report, dict) else {}
            # The governed executor may hand back the skill's own dict or wrap
            # it, depending on the dispatch path taken. Reading only the top
            # level found no `steps`, so a run that had opened the page and
            # turned the loop reported "the page did not respond to any
            # action" — the delegation describing its own reading of the
            # envelope as the page's behaviour.
            if "steps" not in report:
                for nested_key in ("result", "effect_result", "payload"):
                    nested = report.get(nested_key)
                    if isinstance(nested, dict) and "steps" in nested:
                        report = {**report, **nested}
                        break
            logger.info(
                "🌐 Browser pursuit returned: ok=%s rounds=%s keys=%s",
                report.get("ok"),
                report.get("rounds"),
                sorted(report.keys())[:12],
            )
        except _DESKTOP_TASK_RECOVERABLE_ERRORS as exc:
            record_degradation("desktop_task.page_objective", exc, severity="warning")
            return {
                "ok": False,
                "status": "page_objective_failed",
                "error": f"{type(exc).__name__}: {exc}",
                "objective": objective,
            }

        steps = list(report.get("steps") or [])
        # Say what went wrong, in the words of the round that failed.
        #
        # Returning a bare status produced "desktop_task reported failure
        # without a cause (status=page_objective_partial)" — the same
        # undiagnosable shape this file already documents for its own steps.
        # The loop records an error on the round that stopped it; that is the
        # cause, and it belongs in the reply rather than in a log.
        failure = ""
        for step in steps:
            if step.get("error"):
                failure = str(step["error"])
        if not failure and not report.get("ok"):
            # Say what actually happened, not a guess about the page.
            #
            # A cancelled generation comes back `{"status": "failed",
            # "error": ""}`, and an empty error was being rendered as "the page
            # did not respond to any action" — a confident claim about a
            # website, made because a model call was killed mid-round. The turn
            # ran 181s and was cancelled by its own budget; the page had been
            # responding fine.
            reported = str(report.get("error") or "").strip()
            status = str(report.get("status") or "").strip()
            if reported:
                failure = reported
            elif status and status != "completed":
                failure = f"the browser task ended as {status} without saying why"
            elif report.get("rounds") in (None, 0):
                failure = "the browser task ended before it could act"
            else:
                failure = "the page did not respond to any action"
        # The lane's own result shape, not a parallel vocabulary.
        #
        # Third time in this integration: `final_url` where the effect verifier
        # reads `observed_url`, a bare envelope where the caller reads `steps`,
        # and `page_objective_completed` where the task contract reads
        # `status: completed` with receipts beside it. A new return path that
        # speaks nearly the same language satisfies none of the machinery that
        # was already there, and the objective dies holding a result that
        # worked — "expectation incomplete: steps_requested; steps_completed",
        # printed next to "Completed 1/1 steps".
        rounds = [step for step in steps if step.get("chose")]
        receipts = [
            {
                "index": index,
                "action": "browse_pursue",
                "ok": bool(step.get("ok")),
                "effect_verified": bool(step.get("moved", step.get("ok"))),
                "effect_evidence": str(step.get("why") or ""),
                "reason": str(step.get("asked") or ""),
                "expect": str(step.get("expected") or ""),
                "result": {"ok": bool(step.get("ok"))},
            }
            for index, step in enumerate(rounds)
        ]
        # An errored round is not a failed task.
        #
        # `failure` was any error on any round, so one unreadable reply at
        # round forty-one flipped the verdict on everything before it.
        # Measured live: a nine-minute run that answered most of a sixty-item
        # form came back `unparsable_decision`, and the reply she gave was
        # about something else entirely, because the tool had told her it had
        # done nothing.
        #
        # What stopped her is worth reporting either way, so it stays — as a
        # note beside the work rather than instead of it. Only a run where
        # nothing landed is a failure.
        landed_rounds = sum(1 for step in rounds if step.get("ok"))
        succeeded = bool(report.get("ok")) and landed_rounds > 0
        return {
            "ok": succeeded,
            "status": "completed" if succeeded else "failed",
            **({"error": failure} if failure and not succeeded else {}),
            **({"stopped_because": failure} if failure and succeeded else {}),
            "objective": objective,
            "receipts": receipts,
            "failures": [] if succeeded else [r for r in receipts if not r["ok"]],
            "planner": "browser_pursuit",
            "summary": (
                f"Worked through {report.get('final_url') or url} over "
                f"{len(rounds)} round(s)"
                + (f", then stopped: {failure}." if failure else ".")
            ),
            "url": report.get("final_url") or url,
            # The names the task-level contract checks. This lane's expectation
            # requires `steps_requested` and `steps_completed`; returning them
            # under different names failed the objective with "expectation
            # incomplete: steps_requested; steps_completed" — the same
            # sentence, and the same cause, this file already records from
            # 2026-07-27, arriving again through a new return path.
            "steps_requested": len(rounds) or len(steps),
            "steps_completed": sum(1 for step in rounds if step.get("ok")),
            # Her own narration of each choice, kept as the observable record
            # of what was done rather than a step count.
            "narration": [
                {"asked": step.get("asked", ""), "chose": step.get("chose", []), "why": step.get("why", "")}
                for step in steps
                if step.get("chose")
            ],
            "result_text": report.get("result_text", ""),
        }

    async def execute(self, params: Any, context: dict[str, Any]) -> dict[str, Any]:
        if isinstance(params, dict):
            params = DesktopTaskParams(**params)

        # A web page has its own body, and it is not this one.
        #
        # This lane acts on the world through the GUI: coordinates, apps,
        # keystrokes. Its whole action vocabulary — open_url, click, type — is
        # about a screen. Asked to work THROUGH a page, the step deriver
        # returned `read_screen_text`, nothing executed, and the turn fell back
        # to generation, which answered "The website you provided does not
        # exist, and the URL is invalid" about a page it had never opened, then
        # offered to simulate the test from memory instead. Measured live
        # 2026-08-18.
        #
        # The browser skill is the right body for that: real selectors instead
        # of coordinates, its own lease and origin checks, and a loop that can
        # re-read the page after every action. So this lane delegates rather
        # than approximating, which is also why the delegation lives here and
        # not in the deriver — the objective goes across whole, not as a
        # sequence of GUI steps guessed in advance.
        delegated = await self._delegate_page_objective(params, context)
        if delegated is not None:
            return delegated

        try:
            from core.container import ServiceContainer

            capability_engine = ServiceContainer.get("capability_engine", default=None)
        except (ImportError, AttributeError, RuntimeError) as exc:
            record_degradation(
                "desktop_task",
                exc,
                action="blocked desktop task because capability engine lookup failed closed",
                severity="degraded",
            )
            capability_engine = None

        if capability_engine is None or not hasattr(capability_engine, "execute"):
            return {
                "ok": False,
                "status": "capability_engine_unavailable",
                "error": "Desktop task requires the governed capability engine.",
            }

        receipts: list[dict[str, Any]] = []
        failures: list[dict[str, Any]] = []
        objective = params.objective or str((context or {}).get("objective") or "desktop task")

        # She may already be looking. A pure observation — "what's on my
        # screen" with no action attached — is answerable from what the
        # ambient loop saw moments ago, and re-capturing to answer it costs
        # a governed desktop action plus its latency to learn what is
        # already known.
        #
        # Bounded to PURE observations on purpose: any request with an action
        # in it needs the current screen, because the action is about to
        # change it. And the freshness bound is short, because answering a
        # question about NOW with a fact about THEN is answering wrong.
        ambient = self._ambient_answer(objective, params)
        if ambient is not None:
            return await self._attach_pointing(objective, ambient)

        task_context = dict(context or {})
        task_context.setdefault("objective", objective)
        steps = list(params.steps)
        planner = "explicit_steps" if steps else ""
        if not steps:
            plan_error = self._declared_plan_validation_error(task_context)
            if plan_error:
                # A DECLARED plan that will not parse is rejected, never
                # replaced.
                #
                # This briefly fell back to heuristic planning, to rescue a
                # live turn that died with "Structured desktop plan contains an
                # invalid or unsupported step. Completed 0/0 steps." That was
                # the wrong repair: substituting a different plan runs work
                # nobody authored, and two tests here exist precisely to forbid
                # it — one of them named for rejecting a malformed plan EVEN
                # WITH a heuristic fallback available.
                #
                # The turn behind that symptom was fixed where it belonged, in
                # routing and in the objective planner, so a request that needs
                # no declared plan never produces a broken one.
                return {
                    "ok": False,
                    "status": "invalid_desktop_task_plan",
                    "error": plan_error,
                    "objective": objective,
                    "steps_requested": 0,
                    "steps_completed": 0,
                    "receipts": [],
                    "failures": [],
                }
            steps, planner = self._steps_with_provenance_from_context(task_context)
            if plan_error and not steps:
                # The context planner reads the model's own payload, which is
                # the thing that was malformed. The objective planner works
                # from the request text and is the one that can still answer.
                derived = self._derive_steps_from_objective(objective, task_context)
                if derived:
                    steps, planner = derived, "heuristic_compat"
            if plan_error and not steps:
                return {
                    "ok": False,
                    "status": "invalid_desktop_task_plan",
                    "error": plan_error,
                    "objective": objective,
                    "steps_requested": 0,
                    "steps_completed": 0,
                    "receipts": [],
                    "failures": [],
                }
            if plan_error:
                # A malformed declared plan is a reason to plan differently, not
                # a reason to do nothing.
                #
                # LIVE, 2026-08-10: the model proposed a step naming an action
                # that does not exist, and the turn died with "Structured
                # desktop plan contains an invalid or unsupported step.
                # Completed 0/0 steps." A working heuristic plan for the same
                # objective was sitting right behind that return, and the
                # person got nothing.
                record_degradation(
                    "desktop_task",
                    ValueError(plan_error),
                    action=f"planned the objective heuristically instead ({planner})",
                )
        requires_structured_plan = bool(task_context.get("desktop_execution_contract")) and not bool(
            task_context.get("allow_heuristic_desktop_plan")
        )
        if not steps and requires_structured_plan:
            return {
                "ok": False,
                "status": "desktop_task_plan_required",
                "error": (
                    "The live CognitiveEngine response did not contain a valid structured "
                    "desktop plan, so no desktop action was attempted."
                ),
                "objective": objective,
                "steps_requested": 0,
                "steps_completed": 0,
                "receipts": [],
                "failures": [],
                "planner": "required_cognitive_plan_missing",
            }

        research_context = await self._collect_research_context(
            capability_engine=capability_engine,
            objective=objective,
            context=task_context,
        )
        if research_context:
            task_context.update(research_context)
            if task_context.get("desktop_task_research_error") and self._objective_requests_research_document(objective):
                failure_receipt = {
                    "index": 0,
                    "action": "web_search",
                    "ok": False,
                    "critical": True,
                    "effect_verified": False,
                    "effect_evidence": str(task_context.get("desktop_task_research_error") or ""),
                    "result": {
                        "query": task_context.get("desktop_task_research_query"),
                        "deep": task_context.get("desktop_task_research_deep"),
                        "pressure_limited": task_context.get("desktop_task_research_pressure_limited"),
                    },
                }
                await self._emit_durable_step_receipt(
                    failure_receipt,
                    objective=objective,
                    planner=planner or "research_preflight",
                    tool="web_search",
                )
                return {
                    "ok": False,
                    "status": "desktop_task_research_unavailable",
                    "error": str(task_context.get("desktop_task_research_error") or "research evidence unavailable"),
                    "objective": objective,
                    "steps_requested": 0,
                    "steps_completed": 0,
                    "receipts": [],
                    "failures": [failure_receipt],
                    "planner": planner or "research_preflight",
                    "research": {
                        "query": task_context.get("desktop_task_research_query"),
                        "sources": [],
                        "error": task_context.get("desktop_task_research_error"),
                    },
                }
        document_provenance = "cognitive_context"
        if self._objective_requests_self_summary(objective):
            authored = self._self_summary_from_context(task_context)
            if not authored and self._allow_desktop_task_model_synthesis(task_context):
                authored = await self._synthesize_self_summary_document(
                    objective=objective,
                    context=task_context,
                )
                if authored:
                    task_context["desktop_task_document_body"] = authored
                    document_provenance = "local_cortex_synthesis"
            if not authored:
                task_context["desktop_task_document_body"] = self._compose_self_summary_body(
                    objective
                )
                document_provenance = "runtime_substrate_synthesis"
        elif task_context.get("desktop_task_research_synthesis"):
            document_provenance = (
                "local_cortex_research_synthesis"
                if task_context.get("desktop_task_research_authored") is True
                else "source_grounded_deterministic_synthesis"
            )
        elif self._objective_requests_freeform_written_content(
            objective
        ) or self._objective_requests_written_artifact(objective):
            # Author the artifact, the same way a self-summary is authored.
            #
            # Only self-summaries and research documents ever reached the model.
            # Everything else fell to the deterministic composer, so "write a
            # note with three sentences about orcas" produced, verbatim:
            #   "Notes on the requested subject: The requested subject is the
            #    focus of this note. The important part is to describe the
            #    subject clearly..."
            # A real note, correctly created, saying nothing about orcas. That
            # is the "note that opens with no text" — it is not empty, it is
            # empty of content.
            if not str(
                task_context.get("desktop_task_document_body") or ""
            ).strip() and self._objective_needs_authored_content(objective):
                authored = await self._synthesize_requested_writing(
                    objective=objective,
                    context=task_context,
                )
                if authored:
                    task_context["desktop_task_document_body"] = authored
                    document_provenance = "local_cortex_authored_artifact"
                else:
                    # She could not write the words, so there is no document.
                    #
                    # The fallback composer produces prose ABOUT the request —
                    # "Notes on the requested subject: The requested subject is
                    # the focus of this note" — and a file holding that is a
                    # true receipt for the wrong artifact, which this module
                    # already calls the worse failure everywhere else it
                    # appears. LIVE 2026-08-26, three times over, the last one
                    # because the resident worker was not alive yet.
                    #
                    # Saying so costs the person nothing they had; writing the
                    # template costs them a file they have to notice is empty.
                    return {
                        "ok": False,
                        "status": "desktop_task_content_unavailable",
                        "error": (
                            "I could not write the words you asked for, so I have not "
                            "made the file. Nothing was created. Ask me again in a "
                            "moment — my own writing was out of reach just then."
                        ),
                        "objective": objective,
                        "steps_completed": 0,
                        "steps_requested": 1,
                    }
        if not steps:
            steps = self._derive_steps_from_objective(objective, task_context)
            planner = "heuristic_compat"
        steps = self._resolve_document_body_tokens(
            steps,
            self._document_body(objective, task_context),
        )
        if len(steps) > MAX_DESKTOP_TASK_STEPS:
            return {
                "ok": False,
                "status": "desktop_task_plan_too_large",
                "error": (
                    f"Desktop task requires {len(steps)} steps, exceeding the "
                    f"{MAX_DESKTOP_TASK_STEPS}-step bounded execution limit."
                ),
                "objective": objective,
                "steps_requested": len(steps),
                "steps_completed": 0,
                "receipts": [],
                "failures": [],
                "planner": planner,
            }

        if planner == "heuristic_compat" and self._should_escalate_to_os_automation(
            objective,
            steps,
            task_context,
        ):
            return await self._execute_os_automation_escalation(
                capability_engine=capability_engine,
                objective=objective,
                context=task_context,
            )

        last_image_page_url = ""
        last_image_path = ""
        expected_frontmost_app = ""
        current_surface_requires_editable_focus = False
        expected_clipboard_sha256 = ""
        expected_clipboard_chars: int | None = None
        for index, step in enumerate(steps, start=1):
            references_ok, resolved_step, reference_error = self._resolve_step_target(step, receipts)
            if not references_ok:
                receipt = {
                    "index": index,
                    "action": step.action,
                    "reason": step.reason,
                    "expect": step.expect,
                    "critical": step.critical,
                    "ok": False,
                    "effect_verified": False,
                    "effect_evidence": reference_error,
                    "attempts": 0,
                    "result": {
                        "ok": False,
                        "status": "desktop_step_reference_unresolved",
                        "error": reference_error,
                    },
                }
                receipts.append(receipt)
                await self._emit_durable_step_receipt(
                    receipt,
                    objective=objective,
                    planner=planner,
                    tool="desktop_task",
                )
                failures.append(receipt)
                self._emit_progress(
                    index=index,
                    total=len(steps),
                    action=step.action,
                    state="blocked",
                    detail=reference_error,
                    level="warning",
                )
                if step.critical and params.stop_on_error:
                    break
                continue

            target = resolved_step.target
            if resolved_step.action == "system_control" and isinstance(target, dict):
                if target.get("value") == FETCHED_IMAGE_PATH_SENTINEL:
                    if not last_image_path:
                        reference_error = (
                            "no fetched image path available to apply"
                        )
                        receipt = {
                            "index": index,
                            "action": resolved_step.action,
                            "reason": resolved_step.reason,
                            "expect": resolved_step.expect,
                            "critical": resolved_step.critical,
                            "ok": False,
                            "effect_verified": False,
                            "effect_evidence": reference_error,
                            "attempts": 0,
                            "result": {
                                "ok": False,
                                "status": "desktop_step_reference_unresolved",
                                "error": reference_error,
                            },
                        }
                        receipts.append(receipt)
                        await self._emit_durable_step_receipt(
                            receipt,
                            objective=objective,
                            planner=planner,
                            tool="desktop_task",
                        )
                        failures.append(receipt)
                        if resolved_step.critical and params.stop_on_error:
                            break
                        continue
                    target = dict(target, value=last_image_path)
            if resolved_step.action == "render_text_pdf" and isinstance(target, dict):
                if target.get("image_path") == FETCHED_IMAGE_PATH_SENTINEL:
                    if not last_image_path:
                        reference_error = "no fetched image path available for PDF rendering"
                        receipt = {
                            "index": index,
                            "action": resolved_step.action,
                            "reason": resolved_step.reason,
                            "expect": resolved_step.expect,
                            "critical": resolved_step.critical,
                            "ok": False,
                            "effect_verified": False,
                            "effect_evidence": reference_error,
                            "attempts": 0,
                            "result": {
                                "ok": False,
                                "status": "desktop_step_reference_unresolved",
                                "error": reference_error,
                            },
                        }
                        receipts.append(receipt)
                        await self._emit_durable_step_receipt(
                            receipt,
                            objective=objective,
                            planner=planner,
                            tool="desktop_task",
                        )
                        failures.append(receipt)
                        if resolved_step.critical and params.stop_on_error:
                            break
                        continue
                    target = dict(target, image_path=last_image_path)
            if resolved_step.action == "open_url":
                # Resolve the fetched-image source sentinel from the
                # fetch receipt — the source page is only known at runtime.
                if isinstance(target, dict) and target.get("url") == FETCHED_IMAGE_SOURCE_SENTINEL:
                    if not last_image_page_url:
                        reference_error = "no fetched-image source URL available to show"
                        receipt = {
                            "index": index,
                            "action": resolved_step.action,
                            "reason": resolved_step.reason,
                            "expect": resolved_step.expect,
                            "critical": resolved_step.critical,
                            "ok": False,
                            "effect_verified": False,
                            "effect_evidence": reference_error,
                            "attempts": 0,
                            "result": {
                                "ok": False,
                                "status": "desktop_step_reference_unresolved",
                                "error": reference_error,
                            },
                        }
                        receipts.append(receipt)
                        await self._emit_durable_step_receipt(
                            receipt,
                            objective=objective,
                            planner=planner,
                            tool="desktop_task",
                        )
                        failures.append(receipt)
                        if resolved_step.critical and params.stop_on_error:
                            break
                        continue
                    target = dict(target, url=last_image_page_url)
                elif target == FETCHED_IMAGE_SOURCE_SENTINEL:
                    if not last_image_page_url:
                        reference_error = "no fetched-image source URL available to show"
                        receipt = {
                            "index": index,
                            "action": resolved_step.action,
                            "reason": resolved_step.reason,
                            "expect": resolved_step.expect,
                            "critical": resolved_step.critical,
                            "ok": False,
                            "effect_verified": False,
                            "effect_evidence": reference_error,
                            "attempts": 0,
                            "result": {
                                "ok": False,
                                "status": "desktop_step_reference_unresolved",
                                "error": reference_error,
                            },
                        }
                        receipts.append(receipt)
                        await self._emit_durable_step_receipt(
                            receipt,
                            objective=objective,
                            planner=planner,
                            tool="desktop_task",
                        )
                        failures.append(receipt)
                        if resolved_step.critical and params.stop_on_error:
                            break
                        continue
                    target = last_image_page_url
            target_payload = self._target_payload(target)
            if isinstance(target, dict):
                target = json.dumps(target)
            payload = {
                "action": resolved_step.action,
                "target": str(target or ""),
                "x": int(resolved_step.x),
                "y": int(resolved_step.y),
            }
            step_context = self._child_step_context(task_context)
            target_text = str(target or "").lower()
            write_commit_action = (
                resolved_step.action == "type"
                or (
                    resolved_step.action == "hotkey"
                    and (
                        "command" in target_text
                        or "cmd" in target_text
                    )
                    and any(token in target_text for token in ("+v", "+n", "enter", "return"))
                )
            )
            if (
                write_commit_action
                and expected_frontmost_app
            ):
                step_context["desktop_task_expected_frontmost_app"] = expected_frontmost_app
                step_context["desktop_task_write_surface_app"] = expected_frontmost_app
                step_context["desktop_task_prior_verified_frontmost_app"] = expected_frontmost_app
                step_context["desktop_task_allow_unavailable_frontmost_from_prior"] = True
            if write_commit_action and current_surface_requires_editable_focus:
                step_context["desktop_task_requires_editable_focus"] = True
            if (
                resolved_step.action == "hotkey"
                and "v" in target_text
                and ("command" in target_text or "cmd" in target_text)
                and expected_clipboard_sha256
            ):
                step_context["desktop_task_expected_clipboard_sha256"] = expected_clipboard_sha256
                step_context["desktop_task_expected_clipboard_chars"] = expected_clipboard_chars
            step_context.update(
                {
                    "origin": step_context.get("origin") or "desktop_task",
                    "route": "desktop_task.computer_use",
                    "objective": objective,
                    "foreground_request": True,
                    "user_requested_action": True,
                    "user_explicitly_authorized": True,
                    "desktop_task_step": index,
                    "desktop_task_step_total": len(steps),
                    "desktop_task_planner": planner,
                    "desktop_task_reason": resolved_step.reason,
                    "desktop_task_expect": resolved_step.expect,
                }
            )
            self._emit_progress(
                index=index,
                total=len(steps),
                action=resolved_step.action,
                state="starting",
                detail=resolved_step.reason or "Executing governed desktop action.",
            )
            attempt_limit = (
                2 if resolved_step.action in DESKTOP_TASK_RETRY_SAFE_ACTIONS else 1
            )
            attempt = 0
            result: dict[str, Any] = {}
            effect_verified = False
            effect_evidence = "step did not execute"
            _focus_app = ""
            for _prior in steps[: index + 1]:
                if str(getattr(_prior, "action", "")) == "open_app":
                    _focus_app = str(getattr(_prior, "target", "") or "").strip()

            # Hold the app the way a person does.
            #
            # An interaction step sends keystrokes to whatever is frontmost.
            # Checking focus once at open_app and hoping it survives until
            # cmd+v is how "open Notes and write a note" failed with
            # "observed=Google Chrome" — the browser took focus back between
            # steps, which is exactly what browsers do. A person keeps the app
            # they are working in in front until they are finished.
            if resolved_step.action in _FOCUS_SENSITIVE_ACTIONS and _focus_app:
                try:
                    await _computer_use_skill_singleton().hold_focus(_focus_app)
                except _DESKTOP_TASK_RECOVERABLE_ERRORS as exc:
                    logger.debug("hold_focus skipped for %s: %s", _focus_app, exc)

            while attempt < attempt_limit:
                attempt += 1
                step_context["desktop_task_attempt"] = attempt
                try:
                    result = await capability_engine.execute(
                        "computer_use",
                        payload,
                        context=step_context,
                    )
                except (
                    AttributeError,
                    RuntimeError,
                    TypeError,
                    ValueError,
                    OSError,
                    TimeoutError,
                ) as exc:
                    record_degradation(
                        "desktop_task",
                        exc,
                        action="recorded failed desktop step after governed computer_use exception",
                        severity="degraded",
                    )
                    result = {
                        "ok": False,
                        "status": "computer_use_exception",
                        "error": str(exc),
                    }
                if not isinstance(result, dict):
                    result = {"ok": bool(result), "result": result}
                effect_verified, effect_evidence = self._verify_step_effect(
                    resolved_step,
                    result,
                )
                if bool(result.get("ok")) and effect_verified:
                    break
                if result.get("retryable") is False:
                    break
                if attempt < attempt_limit:
                    self._emit_progress(
                        index=index,
                        total=len(steps),
                        action=resolved_step.action,
                        state="retrying",
                        detail=effect_evidence,
                        level="warning",
                    )
                    await asyncio.sleep(0.1)
            receipt = {
                "index": index,
                "action": resolved_step.action,
                "reason": resolved_step.reason,
                "expect": resolved_step.expect,
                "critical": resolved_step.critical,
                "ok": bool(result.get("ok")) and effect_verified,
                "effect_verified": effect_verified,
                "effect_evidence": effect_evidence,
                "attempts": attempt,
                "result": result,
            }
            receipts.append(receipt)
            await self._emit_durable_step_receipt(
                receipt,
                objective=objective,
                planner=planner,
                tool="computer_use",
            )
            if resolved_step.action == "fetch_topic_image" and receipt["ok"]:
                last_image_page_url = str(result.get("page_url") or "") or last_image_page_url
                last_image_path = (
                    str(result.get("path") or result.get("file") or "")
                    or last_image_path
                )
            if receipt["ok"] and resolved_step.action == "set_clipboard":
                expected_clipboard_sha256 = str(result.get("sha256") or "").strip()
                chars = result.get("chars")
                expected_clipboard_chars = chars if isinstance(chars, int) else None
            if receipt["ok"] and resolved_step.action == "open_app":
                expected_frontmost_app = str(result.get("frontmost_app") or result.get("opened") or "").strip()
                current_surface_requires_editable_focus = False
            elif receipt["ok"] and resolved_step.action == "open_url":
                expected_frontmost_app = str(result.get("frontmost_app") or "").strip()
                current_surface_requires_editable_focus = bool(
                    target_payload.get("requires_editable_focus")
                    or target_payload.get("require_editable_focus")
                )
                if current_surface_requires_editable_focus:
                    editor_focus_verified = bool(
                        result.get("doc_focused")
                        or result.get("editable_focus_verified")
                    )
                    task_context["desktop_task_editor_focus_verified"] = editor_focus_verified
                    task_context["desktop_task_verified_editor_url"] = str(
                        result.get("active_url") or ""
                    ).strip()
                    task_context["desktop_task_editor_focus_evidence"] = str(
                        result.get("focus_error")
                        or result.get("verification")
                        or ""
                    ).strip()
            if not receipt["ok"]:
                failures.append(receipt)
                self._emit_progress(
                    index=index,
                    total=len(steps),
                    action=resolved_step.action,
                    state="failed",
                    detail=effect_evidence,
                    level="warning",
                )
                if resolved_step.critical and params.stop_on_error:
                    break
            else:
                self._emit_progress(
                    index=index,
                    total=len(steps),
                    action=resolved_step.action,
                    state="verified",
                    detail=effect_evidence,
                )

        critical_failures = [receipt for receipt in failures if receipt.get("critical", True)]
        completed_all_steps = len(receipts) == len(steps)
        ok = not critical_failures and completed_all_steps
        status = (
            "completed_with_warnings"
            if ok and failures
            else "completed"
            if ok
            else "failed"
        )
        completed_count = sum(1 for receipt in receipts if receipt.get("ok"))
        observation = self._observation_evidence(receipts, objective)
        semantic_evidence = self._semantic_completion_evidence(
            objective=objective,
            task_context=task_context,
            receipts=receipts,
            all_effects_verified=ok,
        )
        payload = {
            "ok": ok,
            "status": status,
            **(
                {}
                if ok
                else {
                    "error": self._failure_cause(
                        critical_failures or failures, objective=objective
                    )
                }
            ),
            "objective": objective,
            "steps_requested": len(steps),
            "steps_completed": completed_count,
            "receipts": receipts,
            "failures": failures,
            "planner": planner,
            "document_provenance": document_provenance,
            "research": {
                "query": task_context.get("desktop_task_research_query"),
                "sources": task_context.get("desktop_task_research_sources") or [],
                "error": task_context.get("desktop_task_research_error"),
                "summary": task_context.get("desktop_task_research_summary"),
                "synthesis": task_context.get("desktop_task_research_synthesis"),
                "deep": task_context.get("desktop_task_research_deep"),
                "pressure_limited": task_context.get(
                    "desktop_task_research_pressure_limited"
                ),
                "timing_ms": task_context.get("desktop_task_research_timing_ms") or {},
            } if research_context else None,
            # The perception, typed as evidence for THIS request. The
            # response lane renders it into the reasoning context so she can
            # answer the question rather than continue the buffer.
            "observation": (
                observation.for_reasoning() if observation is not None else None
            ),
            "observation_meta": (
                observation.to_dict() if observation is not None else None
            ),
            # What she SAW, said plainly, built natively from the capture.
            # The surface that answers "what's on my screen" reads this
            # rather than digging the raw text out of a receipt — a screen
            # is read by the OS, not narrated by a 32B, and the capture is
            # evidence rather than a reply.
            "observation_description": (
                observation.describe() if observation is not None else None
            ),
            "summary": (
                # An observation's answer is what was SEEN. A step count is a
                # progress report about the machinery, and handing it back for
                # "what do you see?" reports that the looking happened without
                # ever saying what was there. This is the FALLBACK; the
                # reasoning above forms the real answer.
                self._describe_screen_observation(receipts)
                or f"Desktop task completed {completed_count}/{len(steps)} governed "
                f"computer-use steps through {planner or 'unknown'} planning."
            ),
            "semantic_evidence": semantic_evidence,
        }
        from core.runtime.skill_contract import (
            SkillExecutionResult,
            SkillStatus,
            evaluate_action_expectation,
        )

        expectation = self._semantic_completion_contract(objective)
        verdict = evaluate_action_expectation(
            SkillExecutionResult(
                skill=self.name,
                status=SkillStatus.SUCCESS_VERIFIED,
                output=payload,
                expectation=expectation,
            )
        )
        if verdict is not None:
            payload["action_expectation"] = expectation.to_dict()
            payload["semantic_completion"] = verdict.to_evidence()
            if not verdict.passed and ok:
                missing = verdict.unsatisfied_predicates + verdict.unknown_predicates
                payload["ok"] = False
                payload["status"] = verdict.status.value
                payload["error"] = "semantic completion incomplete: " + "; ".join(missing)
                payload["summary"] = (
                    f"Desktop task completed {completed_count}/{len(steps)} mechanical steps, "
                    f"but still requires: {', '.join(missing)}."
                )
        return payload
