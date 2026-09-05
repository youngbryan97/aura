"""What the applications on this machine are, and how to work them.

Bryan's correction, and he is right: a ``create_note`` action is not general
OS control. It is one app, hardcoded, on a machine that happens to have Notes.
She should be able to meet an application she has never seen, find out what it
is, and work it — *especially* put text into it.

macOS already publishes the answer. Every scriptable application ships a
scripting definition (its ``sdef``) describing its own object model: the
classes it owns, which of them can be created, and which properties hold
text. That is not a heuristic about an app, it is the app's own account of
itself, and it is available for every app without anyone writing an
integration.

So the general shape is three questions, each answered from the machine
rather than from a table someone maintained:

    what is installed        scan the application directories
    what is this app         read its dictionary — classes, commands, text
    how do I write into it   find a makeable class with a text property

Notes answers the third with ``note.body``. TextEdit answers it with
``document.text``. Mail answers it with ``outgoing message.content``. Nothing
in this module knows any of those facts in advance; it derives each one, and
an app nobody has thought about answers the same way.

When an app has no dictionary — many do not — the honest answer is "I cannot
write into this one directly", and the caller falls back to typing at it like
a person would. That fallback is worse and it is not a secret: a keystroke
goes wherever focus goes, which is why writing through the dictionary is
preferred whenever the dictionary exists.
"""

from __future__ import annotations

import logging
import os
import plistlib
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

from core.runtime.lockdep import checked_lock
from core.runtime.subprocess_gateway import get_subprocess_gateway

logger = logging.getLogger("Aura.AppDictionary")

__all__ = [
    "AppFacts",
    "TextTarget",
    "describe_app",
    "installed_apps",
    "read_dictionary",
    "resolve_app",
    "text_target_for",
]

#: Where macOS keeps applications. ~/Applications is included because a user's
#: own installs are as real as the system's.
_APP_DIRECTORIES: tuple[str, ...] = (
    "/Applications",
    "/Applications/Utilities",
    "/System/Applications",
    "/System/Applications/Utilities",
    "~/Applications",
)

#: Properties that hold a document's text, best first. `body` before `text`
#: because an app offering both (Notes) means the richer one by `body`.
_TEXT_PROPERTY_PREFERENCE: tuple[str, ...] = (
    "body",
    "text",
    "content",
    "plaintext",
    "source text",
)

#: Classes that mean "a thing a person writes in", best first.
_DOCUMENT_CLASS_PREFERENCE: tuple[str, ...] = (
    "note",
    "document",
    "outgoing message",
    "sheet",
    "presentation",
)

#: Classes that are chrome or plumbing, never a document. A window has a
#: title and an application has a name; neither is a place to put prose.
_NEVER_A_DOCUMENT: frozenset[str] = frozenset(
    {
        "application",
        "window",
        "item",
        "attribute run",
        "character",
        "word",
        "paragraph",
        "attachment",
        "print settings",
        "tab",
        "color",
    }
)

#: Reading an sdef forks a process; it is worth remembering the answer.
_DICTIONARY_TTL_S = 900.0
_APP_LIST_TTL_S = 120.0
_CACHE_LOCK = checked_lock("app_dictionary")
_DICTIONARY_CACHE: dict[str, tuple[float, AppFacts]] = {}
_APP_LIST_CACHE: tuple[float, tuple[str, ...]] | None = None

_SDEF_TIMEOUT_S = 8.0


@dataclass(frozen=True)
class TextTarget:
    """How to put text into one application, derived from its dictionary.

    ``container`` is the collection a new object is made in — "note" is made
    at a folder, "document" is made at the application itself. Empty means
    the application.
    """

    app: str
    klass: str
    text_property: str
    name_property: str = ""
    container: str = ""

    def describe(self) -> str:
        where = f" in a {self.container}" if self.container else ""
        named = f", named through '{self.name_property}'" if self.name_property else ""
        return (
            f"{self.app} can be written to directly: make a new {self.klass}"
            f"{where} and set its '{self.text_property}'{named}."
        )


@dataclass(frozen=True)
class AppFacts:
    """What an application says about itself."""

    name: str
    path: str = ""
    scriptable: bool = False
    #: class name -> its text-bearing property names
    classes: dict[str, tuple[str, ...]] = field(default_factory=dict)
    commands: tuple[str, ...] = ()
    unavailable_reason: str = ""

    @property
    def can_be_written_to(self) -> bool:
        return bool(self.classes)

    def describe(self) -> str:
        if not self.path:
            return f"I don't have {self.name} installed."
        if not self.scriptable:
            return (
                f"{self.name} is installed at {self.path}, but it publishes no "
                "scripting dictionary — I would have to drive it by typing and "
                "clicking, the way a person does."
            )
        parts = [f"{self.name} is installed at {self.path} and is scriptable."]
        if self.classes:
            written = ", ".join(sorted(self.classes))
            parts.append(f"It holds text in: {written}.")
        if self.commands:
            parts.append(f"It answers to: {', '.join(self.commands[:12])}.")
        return " ".join(parts)


def _candidate_directories() -> list[Path]:
    seen: list[Path] = []
    for raw in _APP_DIRECTORIES:
        path = Path(os.path.expanduser(raw))
        if path.is_dir() and path not in seen:
            seen.append(path)
    return seen


def installed_apps(*, fresh: bool = False) -> tuple[str, ...]:
    """Every application on this machine, by name.

    This is the answer to "do I even have that app", which is the question
    that has to come before "how do I use it" — and the reason a hardcoded
    per-app action is the wrong shape: it assumes an answer.
    """
    global _APP_LIST_CACHE
    if not fresh:
        with _CACHE_LOCK:
            cached = _APP_LIST_CACHE
        if cached is not None and (time.time() - cached[0]) < _APP_LIST_TTL_S:
            return cached[1]

    names: list[str] = []
    for directory in _candidate_directories():
        try:
            entries = sorted(directory.iterdir())
        except OSError as exc:
            logger.debug("Could not list %s: %s", directory, exc)
            continue
        for entry in entries:
            if entry.suffix == ".app" and entry.name[:-4] not in names:
                names.append(entry.name[:-4])
    result = tuple(names)
    with _CACHE_LOCK:
        _APP_LIST_CACHE = (time.time(), result)
    return result


def _normalise(name: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name or "").lower())


def resolve_app(name: Any) -> tuple[str, str]:
    """Match a spoken app name to an installed one: ``(name, path)``.

    "notes", "the Notes app", "Notes.app" and "NOTES" all reach the same
    place; an app that is not installed returns ``("", "")`` rather than a
    guess, because acting on a guess is how you type into the wrong window.
    """
    wanted = _normalise(str(name or "").replace(".app", "").replace(" app", ""))
    if not wanted:
        return ("", "")
    candidates = installed_apps()
    exact = [item for item in candidates if _normalise(item) == wanted]
    if not exact:
        # Substring, longest first — "Google Chrome" for "chrome".
        exact = sorted(
            (
                item
                for item in candidates
                if wanted in _normalise(item) or _normalise(item) in wanted
            ),
            key=lambda item: len(item),
        )
    if not exact:
        return ("", "")
    resolved = exact[0]
    for directory in _candidate_directories():
        path = directory / f"{resolved}.app"
        if path.exists():
            return (resolved, str(path))
    return (resolved, "")


def _bundle_declares_scripting(app_path: str) -> bool:
    """Cheap pre-check: the Info.plist says whether there is a dictionary."""
    plist = Path(app_path) / "Contents" / "Info.plist"
    try:
        with plist.open("rb") as handle:
            info = plistlib.load(handle)
    except (OSError, ValueError, plistlib.InvalidFileException):
        return True  # Unknown: let sdef decide rather than refusing early.
    return bool(
        info.get("OSAScriptingDefinition")
        or info.get("NSAppleScriptEnabled")
        or info.get("SBApplicationBundleIdentifier")
        or info.get("NSServices")
    )


def _run_sdef(app_path: str) -> str:
    """The application's own dictionary, however it can be got at.

    ``sdef`` is one way to read it and not the only one. On a machine with
    the command line tools and no Xcode it refuses — "tool 'sdef' requires
    Xcode" — writes nothing to its output, and every application on the
    machine came back as publishing no dictionary at all. So she could never
    write into an app through the interface it publishes, and a request to
    write a note in Notes fell through to typing at whatever had the keyboard
    or to leaving a text file on disk.

    The dictionary is a file inside the bundle either way. The application's
    own Info.plist names it, and reading it needs nothing installed.
    """
    sdef = shutil.which("sdef") or "/usr/bin/sdef"
    try:
        result = get_subprocess_gateway().run(
            [sdef, app_path],
            capture_output=True,
            read_only=True,
            text=True,
            timeout=_SDEF_TIMEOUT_S,
            source="perception.app_dictionary.sdef",
            accelerator_capability="none",
        )
    except subprocess.TimeoutExpired:
        return _sdef_in_the_bundle(app_path)
    except OSError as exc:
        logger.debug("sdef unavailable for %s: %s", app_path, exc)
        return _sdef_in_the_bundle(app_path)
    return (result.stdout or "").strip() or _sdef_in_the_bundle(app_path)


def _sdef_in_the_bundle(app_path: str) -> str:
    """The dictionary file the application ships, read straight off the disk."""
    bundle = Path(str(app_path or ""))
    if not bundle.name:
        return ""
    resources = bundle / "Contents" / "Resources"
    named = _sdef_named_in_the_plist(bundle)
    wanted = [resources / named] if named else []
    try:
        wanted.extend(sorted(resources.glob("*.sdef")))
    except OSError as exc:
        logger.debug("could not list %s: %s", resources, exc)
    for one in wanted:
        try:
            said = one.read_text(encoding="utf-8", errors="replace")
        except (OSError, UnicodeError) as exc:
            logger.debug("could not read %s: %s", one, exc)
            continue
        if said.strip():
            return said
    return ""


def _sdef_named_in_the_plist(bundle: Path) -> str:
    """What the application calls its own dictionary, from its Info.plist."""
    plist = bundle / "Contents" / "Info.plist"
    try:
        with plist.open("rb") as reading:
            said = plistlib.load(reading)
    except (OSError, ValueError, plistlib.InvalidFileException) as exc:
        logger.debug("could not read %s: %s", plist, exc)
        return ""
    named = str((said or {}).get("OSAScriptingDefinition") or "").strip()
    if named and not named.lower().endswith(".sdef"):
        named = f"{named}.sdef"
    return named


def _text_properties_of(element: ElementTree.Element) -> tuple[str, ...]:
    """Which of a class's properties hold the writable text of a document.

    `type="text"` alone is far too loose: a name, a sender, a URL and a file
    path are all typed `text`, so accepting the type made Mail's document body
    come out as `sender` and Chrome's as `window.given name`. A property is
    somewhere to put a document only if it is *named* like one, or if it
    carries a rich-text type, which plain string fields never do.
    """
    found: list[str] = []
    for prop in element.findall("property"):
        prop_name = str(prop.get("name") or "").strip().lower()
        prop_type = str(prop.get("type") or "").strip().lower()
        access = str(prop.get("access") or "").strip().lower()
        if access == "r":
            continue  # Read-only: it is not somewhere to put words.
        if not prop_name:
            continue
        named_like_a_body = prop_name in _TEXT_PROPERTY_PREFERENCE
        rich = prop_type in {"rich text", "text.ctxt"} or prop_type.endswith(".ctxt")
        if (named_like_a_body or rich) and prop_name not in found:
            found.append(prop_name)
    return tuple(found)


def read_dictionary(app: Any) -> AppFacts:
    """Read an application's own account of what it can do.

    Never raises: an app with no dictionary, a broken dictionary, and an app
    that is not installed are three different facts and each is reported as
    itself.
    """
    name, path = resolve_app(app)
    if not name:
        return AppFacts(
            name=str(app or "").strip(),
            unavailable_reason="that application is not installed on this machine",
        )
    cache_key = path or name
    with _CACHE_LOCK:
        cached = _DICTIONARY_CACHE.get(cache_key)
    if cached is not None and (time.time() - cached[0]) < _DICTIONARY_TTL_S:
        return cached[1]

    if not path:
        facts = AppFacts(name=name, unavailable_reason="I could not find its bundle")
    elif not _bundle_declares_scripting(path):
        facts = AppFacts(
            name=name,
            path=path,
            unavailable_reason="its bundle declares no scripting support",
        )
    else:
        raw = _run_sdef(path)
        if not raw.strip():
            facts = AppFacts(
                name=name,
                path=path,
                unavailable_reason="it publishes no scripting dictionary",
            )
        else:
            try:
                root = ElementTree.fromstring(raw)
            except ElementTree.ParseError as exc:
                facts = AppFacts(
                    name=name,
                    path=path,
                    unavailable_reason=f"its dictionary could not be parsed ({exc})",
                )
            else:
                classes: dict[str, tuple[str, ...]] = {}
                commands: list[str] = []
                # class-extension carries the interesting half for apps that
                # extend the Standard Suite's `document` — TextEdit puts its
                # only text property there.
                for tag in ("class", "class-extension"):
                    for element in root.iter(tag):
                        klass = str(
                            element.get("name") or element.get("extends") or ""
                        ).strip().lower()
                        if not klass:
                            continue
                        if klass in _NEVER_A_DOCUMENT:
                            continue
                        text_props = _text_properties_of(element)
                        if not text_props:
                            continue
                        merged = tuple(
                            dict.fromkeys(classes.get(klass, ()) + text_props)
                        )
                        classes[klass] = merged
                for element in root.iter("command"):
                    command = str(element.get("name") or "").strip()
                    if command and command not in commands:
                        commands.append(command)
                facts = AppFacts(
                    name=name,
                    path=path,
                    scriptable=True,
                    classes=classes,
                    commands=tuple(commands),
                )

    with _CACHE_LOCK:
        _DICTIONARY_CACHE[cache_key] = (time.time(), facts)
    return facts


#: Where a new object of a given class is made. Derived from the dictionary
#: where it is stated; this covers the containers whose names the dictionary
#: expresses as elements rather than as a property.
_KNOWN_CONTAINERS: dict[str, str] = {"note": "folder"}


def text_target_for(app: Any) -> TextTarget | None:
    """How to write text into this app, or ``None`` if it cannot be told.

    ``None`` is not a failure — it means "drive this one by typing", which is
    what a person does with an app that has no dictionary.
    """
    facts = read_dictionary(app)
    if not facts.can_be_written_to:
        return None

    def _rank(item: str) -> int:
        try:
            return _DOCUMENT_CLASS_PREFERENCE.index(item)
        except ValueError:
            return len(_DOCUMENT_CLASS_PREFERENCE)

    for klass in sorted(facts.classes, key=_rank):
        properties = facts.classes[klass]
        chosen = ""
        for preferred in _TEXT_PROPERTY_PREFERENCE:
            if preferred in properties:
                chosen = preferred
                break
        if not chosen:
            chosen = properties[0]
        return TextTarget(
            app=facts.name,
            klass=klass,
            text_property=chosen,
            name_property="name" if klass in {"note", "document"} else "",
            container=_KNOWN_CONTAINERS.get(klass, ""),
        )
    return None


def describe_app(app: Any) -> str:
    """What this app is and what she can do with it, in plain words.

    The answer to "do you know what X is" that comes from the machine rather
    than from what a language model happens to remember about a product name.
    """
    facts = read_dictionary(app)
    lines = [facts.describe()]
    target = text_target_for(app) if facts.can_be_written_to else None
    if target is not None:
        lines.append(target.describe())
    elif facts.path:
        lines.append(
            "To put text in it I would have to bring it to the front and type, "
            "which only works while it holds focus."
        )
    if facts.unavailable_reason and not facts.path:
        lines.append(f"({facts.unavailable_reason})")
    return " ".join(lines)
