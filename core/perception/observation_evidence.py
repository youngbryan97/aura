"""A perception is evidence to interpret, not text to repeat.

Measured live 2026-08-04. Bryan asked "can you tell me what you see on the
screen?" The governed desktop lane worked perfectly — the read succeeded,
the receipt verified — and what came back was the raw accessibility dump:

    Edit
    Window
    (9) Kurzgesagt
    ...
    Show more
    You >

The machinery did its job and the answer was still wrong, because nothing
in the system distinguished *what was captured* from *what should be
said*. A screen read entered working memory as an untyped blob of text,
and text in working memory is material a model continues. So it continued
it — verbatim.

The earlier form of the same bug answered with "Completed 1/1 governed
desktop steps", a progress report about the machinery. Both failures share
a shape: a question about the WORLD answered with a fact about the SYSTEM
or with the SYSTEM'S BUFFER.

The missing thing was a type. Aura had dialogue turns and she had skill
result blobs, and a perception is neither: it is evidence, gathered for a
purpose, whose value is in what it lets her say — not in its bytes. This
module is that type.

An ``Observation`` carries three things a raw string cannot:

* the **capture** — kept whole, because verification and follow-up
  questions need the original and because discarding it would replace one
  kind of dishonesty with another;
* the **provenance** — what was looked at, when, by which faculty, so a
  claim about the screen can be checked against the reading that produced
  it;
* the **request it was gathered for** — because "what do you see?", "is
  the build passing?" and "what's the third video called?" want different
  answers from identical pixels. Only the last of those wants raw text at
  all.

What this module deliberately does NOT do is write her reply. It renders
the evidence *with its frame* so her reasoning has the material and knows
what kind of answer the material is for. The sentence stays hers.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from core.runtime.lockdep import checked_lock

__all__ = [
    "ObservationKind",
    "AnswerShape",
    "SCREEN_CHROME",
    "Observation",
    "ObservationMemory",
    "answer_shape_for",
    "get_observation_memory",
    "remember_observation",
    "RETENTION_FRESH_S",
]


class ObservationKind(StrEnum):
    """What faculty produced this, so a claim can name its own source."""

    SCREEN_TEXT = "screen_text"
    SCREEN_IMAGE = "screen_image"
    WINDOW_TREE = "window_tree"
    CLIPBOARD = "clipboard"
    AUDIO = "audio"


class AnswerShape(StrEnum):
    """What kind of answer the REQUEST wants from this evidence.

    The distinction the runtime never had. Identical pixels answer these
    three questions completely differently, and only one of them is served
    by reproducing the capture.
    """

    #: "What do you see?" — describe the whole, as a person would.
    DESCRIBE = "describe"
    #: "Is the build passing?" / "Did it finish?" — find the specific thing
    #: and answer about IT, not about the screen.
    LOCATE = "locate"
    #: "What exactly does that error say?" / "Read me the third line." —
    #: the literal characters ARE the answer. The one case where quoting is
    #: right, and it has to be asked for.
    TRANSCRIBE = "transcribe"


#: Cues that the person wants the literal text back. Deliberately narrow:
#: quoting is the exception, so it must be requested rather than defaulted
#: to. "Exactly", "verbatim", "word for word" and explicit reading requests
#: are the honest signals.
_TRANSCRIBE_CUES = (
    "exact wording",
    "exactly what it says",
    "exactly what it said",
    "verbatim",
    "word for word",
    "word-for-word",
    "copy the text",
    "transcribe",
    "read it out",
    "read out",
    "read me the",
    "quote",
)

#: Cues that a SPECIFIC thing is being asked about. These want a finding,
#: not a tour of the screen.
_LOCATE_CUES = (
    "is there",
    "are there",
    "can you find",
    "do you see a",
    "do you see any",
    "what does it say about",
    "what is the",
    "what's the",
    "which one",
    "how many",
    "did it",
    "does it",
    "is it",
    "has it",
    # A question about something SEEN EARLIER is the same shape: it wants
    # the thing named, not the screen toured. Live 2026-08-04, "what was
    # that repo you saw on my screen?" fell through to DESCRIBE and was
    # answered with the window stack while the repo name sat in the
    # evidence.
    "what was",
    "what were",
    "what did",
    "which ",
    "who ",
    "where ",
    "how much",
    "name the",
    "look for",
    "tell me the",
    "anything about",
    "anything on",
)


#: Chrome that appears on every macOS screen and describes nothing about
#: what the person is actually looking at. Menu bars and nav rails are the
#: bulk of an accessibility capture and none of it is the answer to "what
#: are you looking at".
SCREEN_CHROME = frozenset({
    "edit", "window", "file", "view", "help", "history", "bookmarks",
    "apple", "show more", "show less", "home", "shorts", "you",
    "settings", "search", "menu", "back", "forward", "reload",
    "new", "more", "customize", "premium", "send", "voice", "online",
    "mute", "subscriptions", "playlists", "watch later", "liked videos",
    "your channel", "sign in", "close", "minimize", "maximize",
})


#: Glyphs an accessibility capture carries that belong to the widget, not
#: the words: tab close crosses, separators, disclosure arrows, list
#: bullets, and the icon stubs that come back as "8t" or "<>".
_ELEMENT_EDGE_CHARS = " \t·•−–—-*×✕✖|<>‹›~+#•"

_BADGE_PREFIX = re.compile(r"^\(\s*\d+\s*\)\s*")
_ICON_STUB_PREFIX = re.compile(r"^(?:[0-9]{1,2}[a-z]{1,2}|[a-z]{1,2}[0-9]{1,2})\s+(?=[A-Z])")


#: A browser tab strip comes back as one line holding several tabs, joined
#: by each tab's close button and a divider:
#:   ``Paramount Plus: Stream Mov × | youngbryan97/aura: A cognit x``
#: Read as a single element that is one unreadable run-on; split, it is two
#: titles.
_ADJACENT_ELEMENT_RE = re.compile(r"\s+[×✕✖xX]\s*(?:\||$)|\s+\|\s+")

#: The close button trailing the last tab on a line, with no divider after
#: it to mark where the title ended.
_TRAILING_CLOSE_RE = re.compile(r"\s+[×✕✖xX]\s*$")


def _split_adjacent_elements(lines: Any) -> list[str]:
    """One captured line may hold several distinct things. Separate them."""
    separated: list[str] = []
    for raw in lines:
        text = str(raw or "")
        for part in _ADJACENT_ELEMENT_RE.split(text):
            part = _TRAILING_CLOSE_RE.sub("", str(part or "")).strip()
            if part:
                separated.append(part)
    return separated


#: Menu-bar words. The chrome set above is matched against a WHOLE line,
#: which a menu bar defeats by arriving as one: live 2026-08-04 the reply
#: named "History Bookmarks Protiles lab window Help" as something on the
#: screen. It is the menu bar of every browser ever shipped, it says
#: nothing about what the person is looking at, and no per-line lookup can
#: catch it because as a line it is unique — and misspelled, because the
#: capture is imperfect.
_MENU_WORDS = SCREEN_CHROME | frozenset({
    "profiles", "tab", "tabs", "tools", "develop", "format", "insert",
    "table", "arrange", "go", "favorites", "people", "safari", "chrome",
    "actions", "selection", "run", "terminal", "shell", "finder",
})

#: How much of a line must be menu vocabulary before it is furniture. A
#: capture mangles words ("Protiles", "lab" for "Tab"), so requiring all of
#: them to match would catch nothing; a clear majority is the honest test.
_CHROME_LINE_RATIO = 0.6


def _is_mostly_chrome(line: str) -> bool:
    """Whether this line is a menu bar rather than something on the screen."""
    tokens = [token.strip(".,:;|-—·").casefold() for token in line.split()]
    tokens = [token for token in tokens if token]
    if len(tokens) < 3:
        return False
    hits = sum(1 for token in tokens if token in _MENU_WORDS)
    return hits / len(tokens) >= _CHROME_LINE_RATIO


def _clean_element(line: str) -> str:
    """Strip widget furniture off a captured line.

    Live 2026-08-04 the description named ``× | youngbryan97/aura: A
    cogniti`` — the tab's close cross and separator read as part of the
    title. What the person wants named is the title.
    """
    text = " ".join(str(line or "").split())
    previous = None
    while text and text != previous:
        previous = text
        text = text.strip(_ELEMENT_EDGE_CHARS)
        text = _BADGE_PREFIX.sub("", text)
        text = _ICON_STUB_PREFIX.sub("", text)
    return text


#: One line of the stacking order:
#:   ``  3. Google Chrome "some title" — 1728x1037 at (0, 34), 33% visible``
_WINDOW_LINE_RE = re.compile(
    r"""^\s*(?P<order>\d+)\.\s+
        (?P<app>[^"“]+?)\s*
        (?:["“](?P<title>.*?)["”])?\s*
        (?:[—–-]\s*(?P<tail>.*?))?\s*$""",
    re.VERBOSE,
)

#: A title that says nothing beyond the app name already said.
_EMPTY_TITLE_HINTS = frozenset({"", "untitled", "window"})


def _describe_windows(windows: list[dict[str, Any]]) -> str:
    """Say what is open and what is being looked at, as a person would.

    Front window first, because that is what "what's on my screen" means.
    Then what is behind it, and how much of it can actually be seen — a
    window listed as fully covered is not something she saw, and saying it
    is on screen without that qualifier would overstate the reading.
    """

    def _name(window: dict[str, Any]) -> str:
        app = _clean_element(str(window.get("app") or ""))
        title = _clean_element(str(window.get("title") or ""))
        if title and title.casefold() not in _EMPTY_TITLE_HINTS and title != app:
            return f"{app} (“{title}”)" if app else f"“{title}”"
        return app or "an untitled window"

    front, *behind = windows
    sentences = [f"{_name(front)} is in front."]

    visible_behind = [
        window
        for window in behind
        if "completely hidden" not in str(window.get("visibility") or "").casefold()
    ]
    hidden_behind = [window for window in behind if window not in visible_behind]

    if visible_behind:
        named = [_name(window) for window in visible_behind[:3]]
        listed = (
            named[0]
            if len(named) == 1
            else f"{named[0]} and {named[1]}"
            if len(named) == 2
            else ", ".join(named[:-1]) + f", and {named[-1]}"
        )
        sentences.append(f"Behind it, partly visible: {listed}.")

    if hidden_behind:
        named = [str(window.get("app") or "").strip() for window in hidden_behind[:3]]
        named = [name for name in named if name]
        if named:
            listed = (
                named[0]
                if len(named) == 1
                else f"{named[0]} and {named[1]}"
                if len(named) == 2
                else ", ".join(named[:-1]) + f", and {named[-1]}"
            )
            extra = len(hidden_behind) - len(named)
            verb = "is" if len(named) == 1 and extra <= 0 else "are"
            sentences.append(
                f"{listed} {verb} open but completely hidden"
                + (f", along with {extra} more" if extra > 0 else "")
                + "."
            )

    return " ".join(sentences)


def answer_shape_for(request: Any) -> AnswerShape:
    """What kind of answer this request wants from a perception.

    Order matters: an explicit ask for the literal text wins, because it is
    the narrowest and most deliberate; then a specific question; then the
    default, which is to describe. DESCRIBE is the default because "what do
    you see" is the ordinary case and dumping the buffer is never a
    reasonable reading of it.
    """
    text = str(request or "").strip().lower()
    if not text:
        return AnswerShape.DESCRIBE
    if any(cue in text for cue in _TRANSCRIBE_CUES):
        return AnswerShape.TRANSCRIBE
    if any(cue in text for cue in _LOCATE_CUES):
        return AnswerShape.LOCATE
    return AnswerShape.DESCRIBE


#: How much captured text is worth putting in front of the model. A whole
#: accessibility tree is mostly chrome, and an unbounded paste is how the
#: buffer became the answer in the first place.
_MAX_EVIDENCE_CHARS = 4000


@dataclass
class Observation:
    """One thing Aura looked at, and what it was looked at FOR."""

    kind: ObservationKind
    #: The capture, whole. Evidence, never the reply.
    capture: str
    #: The request this was gathered to answer.
    request: str = ""
    #: Where it came from — frontmost app, window title, device.
    source: str = ""
    at: float = field(default_factory=lambda: time.time())
    #: Anything the faculty knows that the text does not carry.
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def shape(self) -> AnswerShape:
        return answer_shape_for(self.request)

    @property
    def is_empty(self) -> bool:
        return not self.capture.strip()

    def elements(self) -> list[str]:
        """Distinct non-blank lines, in the order they appeared."""
        seen: dict[str, None] = {}
        for line in self.capture.splitlines():
            stripped = line.strip()
            if stripped:
                seen.setdefault(stripped, None)
        return list(seen)

    def describe(self) -> str:
        """What is on the screen, said plainly. Native — no model involved.

        Reading a screen is a sense, not a reasoning problem. The
        accessibility capture already arrives structured, so saying what is
        in front of the person is a matter of dropping the chrome and
        naming what is left. Spending a 32B generation to narrate text the
        OS already handed over is an allocation with nothing to show for
        it, and it is how one observation turns into a stalled turn.

        Returns "" when nothing was legible, so a caller says nothing
        rather than describing a screen it never read.
        """
        windows = self.windows()
        if windows:
            described = _describe_windows(windows)
            # The stacking order says which windows are open; it does not
            # say what is IN the front one. Bryan's original case was a
            # YouTube video — "what do you see" wants the thing being
            # watched, not just the fact that Chrome is open.
            on_page = self.salient(section_only=True)
            fresh = [
                item
                for item in on_page
                if not any(item.casefold() in str(w.get("title") or "").casefold() for w in windows)
            ]
            if fresh:
                shown = fresh[:3]
                listed = (
                    shown[0]
                    if len(shown) == 1
                    else f"{shown[0]} and {shown[1]}"
                    if len(shown) == 2
                    else ", ".join(f"“{item}”" for item in shown[:-1])
                    + f", and “{shown[-1]}”"
                )
                if len(shown) < 3:
                    listed = " and ".join(f"“{item}”" for item in shown)
                described += f" On screen: {listed}."
            return described

        app = self.source.strip()
        content = self.salient()

        if not content:
            if not self.capture.strip():
                # A read happened and came back with nothing. Saying so is
                # the answer; falling through to a step count would report
                # that the looking occurred without ever saying what was
                # found, which is the failure this whole type exists for.
                return (
                    f"{app} is in front, but the screen read came back empty."
                    if app
                    else ""
                )
            return (
                f"{app} is in front, but nothing on it read as content — just "
                "window chrome."
                if app
                else "The screen read came back as window chrome only, with no "
                "readable content."
            )

        # Prose, not a list. A semicolon-joined dump of every string the
        # accessibility tree returned is the same failure as pasting the
        # capture, only shorter: it is still the buffer, still unreadable,
        # and still not an answer. Three or four named things, in a
        # sentence, is what a person says when asked what is on a screen.
        shown = [f"“{item}”" for item in content[:4]]
        if len(shown) == 1:
            listed = shown[0]
        elif len(shown) == 2:
            listed = f"{shown[0]} and {shown[1]}"
        else:
            listed = ", ".join(shown[:-1]) + f", and {shown[-1]}"

        lead = f"{app} is in front" if app else "The screen is showing something"
        rest = len(content) - len(shown)
        tail = f", plus {rest} other bits of text" if rest > 0 else ""
        return f"{lead}, showing {listed}{tail}."

    def _text_section(self) -> list[str]:
        """Just the ``Text on screen:`` part of a structured reading.

        Everything above it is the reading's own scaffolding — field
        labels and the stacking order — and naming that back is how a
        description ends up saying "Screen layout (front to back):".
        """
        lines = self.capture.splitlines()
        for index, line in enumerate(lines):
            if line.strip().casefold().startswith("text on screen"):
                seen: dict[str, None] = {}
                for entry in lines[index + 1:]:
                    stripped = entry.strip()
                    if stripped:
                        seen.setdefault(stripped, None)
                return list(seen)
        return []

    def windows(self) -> list[dict[str, Any]]:
        """The window list the capture already carries, parsed.

        ``read_screen_text`` does not return flat text. It returns a
        structured reading — active app, focused window, then the stacking
        order with titles and how much of each is visible:

            3. Google Chrome "youngbryan97/aura: A cognitive…" — 1728x1037
               at (0, 34), 33% visible, partly behind Aura

        Treating those lines as anonymous strings threw away the best
        material in the capture and described the screen as "Active app:
        aura-launcher"; "Window: Aura Zenith" — the reading's own field
        labels, named back as though they were content. What is on a screen
        is which windows are open and which one you are looking at.
        """
        found: list[dict[str, Any]] = []
        for line in self.capture.splitlines():
            match = _WINDOW_LINE_RE.match(line)
            if not match:
                continue
            # The tail is "1280x852 at (224, 80), fully visible". Splitting
            # it on commas puts the window's Y coordinate in the visibility
            # field — "(80), fully visible" — so the geometry is dropped at
            # its closing bracket instead. Size and position are not what
            # anyone means by what is on their screen; how much of the
            # window can actually be seen is.
            tail = (match.group("tail") or "").strip()
            visibility = tail.rpartition("),")[2].strip() or tail
            found.append(
                {
                    "order": int(match.group("order")),
                    "app": match.group("app").strip(),
                    "title": (match.group("title") or "").strip(),
                    "visibility": visibility.strip().rstrip("."),
                }
            )
        return found

    def salient(self, *, section_only: bool = False) -> list[str]:
        """The lines that are actually CONTENT, best first.

        The accessibility tree hands back everything: menu bars, single
        stray glyphs, clipped tab titles, playback timestamps. Live
        2026-08-04 the reply opened with "E / 0:21 / Claude / File / Edit /
        View / Window / Help / *" — nine lines before anything a person
        would call content. Naming what is on a screen means dropping all
        of that first.
        """
        ranked: list[str] = []
        seen: dict[str, None] = {}
        source_lines = self._text_section() if section_only else self.elements()
        for raw in _split_adjacent_elements(source_lines):
            line = _clean_element(raw)
            if len(line) < 3 or line.casefold() in SCREEN_CHROME:
                continue
            if _is_mostly_chrome(line):
                continue
            # Needs letters to be words. Filters stray glyphs ("*", "×"),
            # playback positions ("0:21"), and counter badges.
            letters = sum(1 for ch in line if ch.isalpha())
            if letters < 3:
                continue
            if line.casefold() in seen:
                continue
            seen[line.casefold()] = None
            ranked.append(line)

        # Prefer SUBSTANTIVE lines over navigation labels. The first cut of
        # this took the first eight distinct strings, which on a YouTube
        # page meant "Subscriptions; RealLifeLore; Nexpo; fern" — the
        # sidebar — while the video titles the person was actually looking
        # at fell past the limit. A description made of nav chrome is a
        # worse answer than a shorter one made of content.
        substantive = [
            line for line in ranked if len(line.split()) >= 2 or len(line) >= 20
        ]
        return substantive if substantive else ranked

    def for_reasoning(self) -> str:
        """The evidence, presented WITH the frame it was gathered under.

        This is the whole point of the type. The model receives material
        that is labelled as something looked at, attributed to a source,
        and paired with what the person actually asked — so the reply it
        forms is an answer rather than a continuation of a buffer.

        It states what the evidence IS and what the question WANTS. It does
        not supply phrasing, a template, or an example answer: the
        reasoning does the reasoning, and the sentence is hers.
        """
        if self.is_empty:
            # "Nothing was read" and "nothing is there" are different facts,
            # and the second one is a claim about the world that this evidence
            # does not support. Told only to "say that plainly", the reply came
            # back as "There are no windows or applications open on it at this
            # time. It is a blank slate." — measured live 2026-08-10, with
            # Gmail, the desktop window and the companion all on screen.
            #
            # So the distinction is stated instead of assumed.
            return (
                f"[OBSERVATION — {self.kind.value}] The capture FAILED"
                + (f" for {self.source}" if self.source else "")
                + ". Nothing legible came back.\n"
                "This tells you that the READING did not succeed. It tells you "
                "NOTHING about what is on the screen — a failed capture and an "
                "empty screen produce the identical result here, so you cannot "
                "distinguish them and must not try.\n"
                "Say that you could not read it. Do NOT say the screen is "
                "empty, blank, clear, or that no windows or applications are "
                "open: you did not observe that, and asserting it would be "
                "inventing a perception."
            )

        capture = self.capture.strip()
        truncated = len(capture) > _MAX_EVIDENCE_CHARS
        if truncated:
            capture = capture[:_MAX_EVIDENCE_CHARS]

        header = [f"[OBSERVATION — {self.kind.value}]"]
        if self.source:
            header.append(f"Captured from: {self.source}.")
        header.append(
            "This is RAW CAPTURED TEXT, not something anyone said and not "
            "your reply. It is the evidence you looked at."
        )

        if self.shape is AnswerShape.TRANSCRIBE:
            intent = (
                "The request asks for the literal text, so quoting the "
                "relevant part IS the answer here. Quote only the part asked "
                "for."
            )
        elif self.shape is AnswerShape.LOCATE:
            intent = (
                "The request asks about one specific thing. Answer about "
                "that thing, using the capture to find it. Do not list the "
                "screen. If it is not there, say so."
            )
        else:
            intent = (
                "The request asks what is there. Describe it as a person "
                "would to another person: what application, what it appears "
                "to be showing, what stands out. Reproducing these lines is "
                "not a description of them."
            )

        parts = [
            " ".join(header),
            f"Elements captured: {len(self.elements())}"
            + (" (capture truncated)" if truncated else "")
            + ".",
            "--- begin capture ---",
            capture,
            "--- end capture ---",
            f"What was asked: {self.request.strip()}" if self.request.strip() else "",
            intent,
        ]
        return "\n".join(part for part in parts if part)

    def to_dict(self) -> dict[str, Any]:
        """Telemetry view. Carries lengths and provenance, not the capture.

        The capture is the person's screen. It belongs in the reasoning
        context for one turn, not in a durable telemetry record.
        """
        return {
            "kind": self.kind.value,
            "source": self.source,
            "at": self.at,
            "answer_shape": self.shape.value,
            "capture_chars": len(self.capture),
            "elements": len(self.elements()),
            "empty": self.is_empty,
        }


# ---------------------------------------------------------------------------
# Retention.
#
# A perception she cannot refer back to is not something she saw, it is
# something that passed through her. "What was the third video called?",
# "you mentioned a repo — which one?", and her own later "I noticed he was
# reading about cancer research" all require the observation to still exist
# after the turn that produced it.
#
# Without this, every follow-up forces another capture: slower, and worse,
# it answers a question about the PAST with a reading of the PRESENT. Asked
# "what was on my screen a minute ago?" a re-read is not an answer, it is a
# different question quietly substituted.
# ---------------------------------------------------------------------------

#: How many recent observations stay referenceable. Small deliberately: this
#: holds screen contents, which are the person's, and a long tail of them is
#: a privacy surface rather than a memory. Recent enough for follow-ups in
#: the same conversation is the whole requirement.
_RETAINED = 8

#: After this, a retained observation is stale for reference purposes. She
#: may still SAY what she saw earlier — that is a memory — but she must not
#: answer "what is on my screen" from it.
RETENTION_FRESH_S = 300.0


class ObservationMemory:
    """Recent observations, referenceable and honest about their age."""

    def __init__(self) -> None:
        self._lock = checked_lock("observation_memory.items", reentrant=True)
        self._items: list[Observation] = []

    def record(self, observation: Observation) -> Observation:
        with self._lock:
            self._items.append(observation)
            while len(self._items) > _RETAINED:
                self._items.pop(0)
        return observation

    def latest(self, kind: ObservationKind | None = None) -> Observation | None:
        with self._lock:
            for item in reversed(self._items):
                if kind is None or item.kind is kind:
                    return item
        return None

    def recent(self, limit: int = _RETAINED) -> list[Observation]:
        with self._lock:
            return list(self._items[-max(1, int(limit)):])

    def age_of_latest(self, kind: ObservationKind | None = None) -> float | None:
        item = self.latest(kind)
        return None if item is None else max(0.0, time.time() - item.at)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()

    def sensory_brief(
        self,
        *,
        max_elements: int = 24,
        max_chars: int = 1600,
        max_age_s: float = RETENTION_FRESH_S,
    ) -> str:
        """What she has actually seen, for the context of any turn.

        Bryan, live 2026-08-04: "she shouldn't be blind from information she
        intakes." A perception that only exists inside the turn that
        captured it makes her blind the moment the next question arrives —
        she reads a screen, says what is on it, and then cannot answer "what
        was that repo called?" a sentence later.

        So this is not the answer to a screen question; it is the standing
        fact that she looked and what she saw, carried into ordinary
        conversation. It names its own provenance and its own age, because
        a perception she cannot source is indistinguishable from something
        she made up, and one she reports as current when it is five minutes
        old is a different claim than the one the evidence supports.

        Returns "" when she has seen nothing, so nothing is asserted.
        """
        item = self.latest()
        if item is None or item.is_empty:
            return ""
        described = item.describe()
        if not described:
            return ""

        age = max(0.0, time.time() - item.at)
        # Her senses, not her archive. A reading from half an hour ago is a
        # memory of a screen rather than a perception of one, and carrying
        # it into every later turn would keep asserting a desk that has
        # since changed. Past the window she can still be ASKED what she
        # saw — that goes through recall, which states its own age.
        if age > max_age_s:
            return ""
        if age <= 90:
            when = "moments ago"
        elif age <= RETENTION_FRESH_S:
            when = f"about {max(1, int(age // 60))} minute(s) ago"
        else:
            when = (
                f"{max(1, int(age // 60))} minute(s) ago — it may well have "
                "changed since, so do not state it as current"
            )

        lines = [
            "[YOUR OWN RECENT PERCEPTION — NOTES, NOT A REPLY]",
            f"You looked at the screen yourself {when}. If asked where this "
            "came from, say you read the screen.",
        ]
        windows = item.windows()
        if windows:
            lines.append("Windows open (front to back):")
            for window in windows[:8]:
                title = str(window.get("title") or "").strip()
                visibility = str(window.get("visibility") or "").strip()
                lines.append(
                    f"- {window.get('app')}"
                    + (f" — “{title}”" if title else "")
                    + (f" ({visibility})" if visibility else "")
                )
        else:
            lines.append(described)
        rest = item.salient(section_only=bool(windows))[:max_elements]
        if rest:
            lines.append("Text read off the screen: " + "; ".join(rest) + ".")
        lines.append(
            "These are your notes on what you saw, not a draft answer. "
            "Answer the question that was actually asked, in your own "
            "words, using only what is relevant here. Do not read this "
            "list back, and do not mention the screen if the question is "
            "not about it."
        )
        lines.append("[END YOUR OWN RECENT PERCEPTION]")
        brief = "\n".join(lines)
        return brief[:max_chars]

    def recall_for(self, request: Any) -> str:
        """What she can honestly say about something she looked at earlier.

        Returns "" when there is nothing to recall, so a caller adds
        nothing rather than asserting an observation that never happened.

        Age is stated, never hidden. An observation five minutes old is a
        memory of a screen, not a reading of one, and answering a
        present-tense question from it would be the same class of error as
        the buffer echo: presenting one thing as another.
        """
        item = self.latest()
        if item is None or item.is_empty:
            return ""
        age = max(0.0, time.time() - item.at)
        freshness = (
            "This is what the screen showed just now."
            if age <= RETENTION_FRESH_S
            else (
                f"This is what the screen showed {int(age // 60)} minute(s) ago; "
                "it may have changed since, and you should say so if it matters."
            )
        )
        referring = Observation(
            kind=item.kind,
            capture=item.capture,
            request=str(request or item.request),
            source=item.source,
            at=item.at,
            detail=dict(item.detail),
        )
        return f"{referring.for_reasoning()}\n{freshness}"


_MEMORY: ObservationMemory | None = None
_MEMORY_LOCK = checked_lock("observation_memory.singleton")


def get_observation_memory() -> ObservationMemory:
    global _MEMORY
    if _MEMORY is None:
        with _MEMORY_LOCK:
            if _MEMORY is None:
                _MEMORY = ObservationMemory()
    return _MEMORY


def remember_observation(observation: Observation) -> Observation:
    """Retain a perception so she can refer to it after the turn that made it."""
    return get_observation_memory().record(observation)


#: Keys under which a capture arrives from the perception skills. computer_use
#: returns ``{"ok": True, "text": <accessibility dump>}``; host_automation and
#: the desktop task lane use their own names for the same thing.
PERCEPTION_KEYS: tuple[str, ...] = (
    "text",
    "screen_text",
    "accessibility_text",
    "observation",
)


def observation_from_result(result: Any, request: str = "") -> Observation | None:
    """Recognise a perception in a tool result, or return None.

    Aura has more than one lane that turns a tool result into prompt text —
    the shortcut injector and the synthesis renderer — and each had its own
    idea of how to render one. The synthesis lane used ``repr(result)``, so
    a screen read reached the model as ``{'ok': True, 'text': '<4000 chars
    of accessibility tree>'}`` and the model reproduced it. Fixing the
    injector alone did not change that, because the live turn went through
    the other lane.

    Recognition belongs in ONE place, next to the type that models it, so a
    third lane cannot quietly invent a fourth rendering.
    """
    if not isinstance(result, dict):
        return None
    for key in PERCEPTION_KEYS:
        capture = result.get(key)
        if isinstance(capture, str) and capture.strip():
            return Observation(
                kind=ObservationKind.SCREEN_TEXT,
                capture=capture,
                request=str(request or ""),
                source=str(result.get("active_app") or result.get("source") or ""),
            )
        # Only the first perception-shaped key decides. A result carrying
        # `text` decides on `text`, even if empty — falling through would
        # let an unrelated key stand in for the capture that failed.
        if capture is not None:
            return None
    return None


def frame_tool_result(result: Any, request: str = "") -> str | None:
    """Render a perception as evidence, retaining it; None if it is not one."""
    observation = observation_from_result(result, request)
    if observation is None:
        return None
    return remember_observation(observation).for_reasoning()
