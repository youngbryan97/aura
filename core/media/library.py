"""core/media/library.py — the music and video that are already on this machine.

Every assistant that can "play music" plays somebody else's music, from
somebody else's servers, when somebody else's servers are up. Ask the current
generation to play a song and the best of them hand you off: a card that
opens a streaming app, a link, a new tab. Nothing plays where you asked for
it, and nothing plays at all without a network.

Aura runs on the machine that has the files. That is a real advantage and it
is worth taking seriously rather than treating local playback as the fallback
for when the internet is down. It is the fastest path (no round trip), the
private one (nothing leaves the host to say what you are listening to), and
the only one that works on a plane.

So this indexes what is here. Three constraints shape it:

**It only ever reads inside roots the user configured.** The index is the
allowlist. Playback resolves an opaque id to a path *through this index*,
never a caller-supplied path, so there is no traversal to defend against —
a path that is not in the index cannot be named.

**It is cheap enough to build on demand.** Scanning is bounded by depth, file
count and time, so a pathological directory tree cannot turn "play something"
into a stall. Results are cached with an mtime check on the roots.

**A miss is a fact, not an error.** "There is no Miles Davis in your library"
is a true and useful thing to say. It is reported as a miss with what *was*
searched, so she can say what she looked through rather than implying the
music does not exist.
"""
from __future__ import annotations

import logging
import os
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.Media.Library")

# Extensions a browser's <audio>/<video> can actually play. Indexing a .flac
# she cannot hand to the page is a promise the player then breaks, so the
# index is limited to what is playable end to end.
AUDIO_EXTENSIONS = frozenset({".mp3", ".m4a", ".aac", ".ogg", ".oga", ".opus", ".wav", ".flac"})
VIDEO_EXTENSIONS = frozenset({".mp4", ".m4v", ".webm", ".ogv", ".mov"})

# Codec support is not extension support: Safari plays .mov and .flac, most
# other engines do not. Reported alongside each item so the surface can say
# "your browser cannot play this file" instead of showing a dead player.
BROADLY_PLAYABLE = frozenset({".mp3", ".m4a", ".aac", ".ogg", ".oga", ".opus", ".wav", ".mp4", ".m4v", ".webm"})

MIME_TYPES = {
    ".mp3": "audio/mpeg",
    ".m4a": "audio/mp4",
    ".aac": "audio/aac",
    ".ogg": "audio/ogg",
    ".oga": "audio/ogg",
    ".opus": "audio/ogg",
    ".wav": "audio/wav",
    ".flac": "audio/flac",
    ".mp4": "video/mp4",
    ".m4v": "video/mp4",
    ".webm": "video/webm",
    ".ogv": "video/ogg",
    ".mov": "video/quicktime",
}

# Bounds. A media library can be enormous and a scan that is not bounded is a
# stall waiting for the right directory. Each of these is a ceiling, not a
# target: a normal library finishes long before any of them bite.
MAX_SCAN_FILES = 20_000
MAX_SCAN_DEPTH = 6
MAX_SCAN_SECONDS = 2.5

# How long an index stays fresh before the roots are re-checked. Short enough
# that a file added a minute ago is findable, long enough that ten requests in
# a row do not each rescan the disk.
INDEX_TTL_S = 120.0


def _media_roots_flag() -> Any:
    """The declared knob for where media lives.

    Declared rather than read raw from the environment: a raw
    ``os.environ.get`` is untyped, undiscoverable and individually parsed,
    which is how this repo accumulated several hundred invisible knobs. The
    declaration puts it in `flag_report()` alongside every other one.
    """
    from core.runtime.flags import FlagKind, declare

    return declare(
        "AURA_MEDIA_ROOTS",
        kind=FlagKind.STRING,
        default="",
        description=(
            "os.pathsep-separated directories to index for playable media; "
            "empty means the platform's usual Music/Movies/Videos/Downloads"
        ),
        owner="core.media.library",
    )


def _default_roots() -> tuple[Path, ...]:
    """Where a person's media actually lives, plus anything they configured.

    ``AURA_MEDIA_ROOTS`` is the explicit answer and wins outright. Without it
    the platform conventions are the honest guess, and a root that does not
    exist is simply skipped rather than reported as an error — most people
    have some of these directories and nobody has all of them.
    """
    configured = str(_media_roots_flag().value() or "").strip()
    if configured:
        roots = [Path(p).expanduser() for p in configured.split(os.pathsep) if p.strip()]
        return tuple(r for r in roots if r.is_dir())

    home = Path.home()
    candidates = [
        home / "Music",
        home / "Movies",
        home / "Videos",
        home / "Downloads",
    ]
    return tuple(c for c in candidates if c.is_dir())


def _normalize(text: str) -> str:
    """Fold case and accents so "bjork" finds "Björk"."""
    decomposed = unicodedata.normalize("NFKD", str(text or ""))
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return " ".join(stripped.casefold().split())


@dataclass(slots=True)
class MediaItem:
    """One playable file, addressed by id rather than by path."""

    item_id: str
    path: Path
    title: str
    kind: str  # "audio" | "video"
    extension: str
    size_bytes: int = 0
    modified_at: float = 0.0

    @property
    def mime_type(self) -> str:
        return MIME_TYPES.get(self.extension, "application/octet-stream")

    @property
    def broadly_playable(self) -> bool:
        """False for files only some browsers can decode (.flac, .mov)."""
        return self.extension in BROADLY_PLAYABLE

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.item_id,
            "title": self.title,
            "kind": self.kind,
            "mime_type": self.mime_type,
            "size_bytes": self.size_bytes,
            "broadly_playable": self.broadly_playable,
            # The directory, never the full path: enough for her to say where
            # she found it, without putting a home directory layout into a
            # payload that ends up in a chat log.
            "folder": self.path.parent.name,
        }


@dataclass(slots=True)
class ScanResult:
    items: tuple[MediaItem, ...] = ()
    roots_scanned: tuple[str, ...] = ()
    files_seen: int = 0
    truncated: bool = False
    elapsed_s: float = 0.0
    built_at: float = field(default_factory=time.time)

    def narrative(self) -> str:
        where = ", ".join(self.roots_scanned) or "no configured media folders"
        note = " (scan hit its bound and stopped early)" if self.truncated else ""
        return f"{len(self.items)} playable files under {where}{note}"


def _item_id(path: Path, modified_at: float, size: int) -> str:
    """A stable, opaque handle for one file.

    Derived from the path so it survives a rescan, and salted with size and
    mtime so that replacing a file invalidates any id already handed out — a
    stale id resolving to different bytes is how a player ends up playing
    something nobody asked for.
    """
    import hashlib

    digest = hashlib.sha256(
        f"{path}\n{modified_at:.0f}\n{size}".encode("utf-8", errors="replace")
    ).hexdigest()
    return digest[:24]


#: Words too common or too short to identify anything on their own.
_UNSEARCHABLE = frozenset(
    {
        "the", "a", "an", "and", "or", "of", "to", "in", "on", "at", "for", "with",
        "it", "its", "this", "that", "these", "those", "them", "they", "me", "my",
        "you", "your", "i", "is", "are", "was", "be", "am", "do", "does", "did",
        "get", "got", "go", "going", "keep", "until", "then", "so", "as", "up",
        "out", "off", "one", "some", "any", "all", "what", "when", "how", "tell",
    }
)


def _searchable_terms(needle: str) -> list[str]:
    """The words in a query that could identify something.

    A two-letter word is a substring of half the filenames on a machine, and
    an ordinary English word is in the other half. Neither can be evidence
    that a particular file is the one being asked for — inside a long query.
    A caller keeps whatever is left when this removes everything.
    """
    return [term for term in needle.split() if len(term) > 2 and term not in _UNSEARCHABLE]


def _hits(term: str, words: set[str]) -> bool:
    """Whether a query word appears as a word, rather than inside one.

    Substring matching is why "it" found "Cognitive": every short word is
    hiding inside some longer one. A prefix match is kept, because "remaster"
    should still find "remastered".
    """
    return any(word == term or word.startswith(term) for word in words)



class MediaLibrary:
    """A bounded, cached index of the playable media on this machine."""

    def __init__(self, roots: tuple[Path, ...] | None = None) -> None:
        self._roots = roots if roots is not None else _default_roots()
        self._scan: ScanResult | None = None
        self._by_id: dict[str, MediaItem] = {}

    @property
    def roots(self) -> tuple[Path, ...]:
        return self._roots

    def index(self, *, force: bool = False) -> ScanResult:
        cached = self._scan
        if (
            not force
            and cached is not None
            and (time.time() - cached.built_at) < INDEX_TTL_S
        ):
            return cached
        scan = self._scan_roots()
        self._scan = scan
        self._by_id = {item.item_id: item for item in scan.items}
        return scan

    def _scan_roots(self) -> ScanResult:
        started = time.perf_counter()
        items: list[MediaItem] = []
        seen = 0
        truncated = False
        scanned: list[str] = []

        for root in self._roots:
            scanned.append(root.name or str(root))
            root_depth = len(root.parts)
            try:
                for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
                    here = Path(dirpath)
                    if len(here.parts) - root_depth >= MAX_SCAN_DEPTH:
                        dirnames[:] = []
                    # Package bundles on macOS are directories full of
                    # resources; walking into an .app or .photoslibrary finds
                    # thousands of files nobody meant to offer for playback.
                    dirnames[:] = [
                        d
                        for d in dirnames
                        if not d.startswith(".")
                        and not d.endswith((".app", ".photoslibrary", ".fcpbundle"))
                    ]
                    for name in filenames:
                        if name.startswith("."):
                            continue
                        seen += 1
                        if seen > MAX_SCAN_FILES:
                            truncated = True
                            break
                        extension = Path(name).suffix.lower()
                        if extension in AUDIO_EXTENSIONS:
                            kind = "audio"
                        elif extension in VIDEO_EXTENSIONS:
                            kind = "video"
                        else:
                            continue
                        path = here / name
                        try:
                            stat = path.stat()
                        except OSError:
                            continue
                        items.append(
                            MediaItem(
                                item_id=_item_id(path, stat.st_mtime, stat.st_size),
                                path=path,
                                title=Path(name).stem,
                                kind=kind,
                                extension=extension,
                                size_bytes=stat.st_size,
                                modified_at=stat.st_mtime,
                            )
                        )
                    if truncated or (time.perf_counter() - started) > MAX_SCAN_SECONDS:
                        truncated = truncated or True
                        break
            except OSError as exc:
                record_degradation(
                    "media.library",
                    exc,
                    action=f"skipped an unreadable media root ({root.name})",
                    severity="debug",
                )
            if truncated:
                break

        elapsed = time.perf_counter() - started
        logger.info(
            "media index: %d items from %d files in %.0f ms%s",
            len(items),
            seen,
            elapsed * 1000.0,
            " (truncated)" if truncated else "",
        )
        return ScanResult(
            items=tuple(items),
            roots_scanned=tuple(scanned),
            files_seen=seen,
            truncated=truncated,
            elapsed_s=elapsed,
        )

    def get(self, item_id: str) -> MediaItem | None:
        """Resolve an opaque id. The only way a path is ever produced."""
        if not self._by_id:
            self.index()
        item = self._by_id.get(str(item_id or ""))
        if item is None:
            return None
        # The index can outlive the file. Returning a MediaItem for something
        # that has been deleted turns a clean miss into a 500 at play time.
        if not item.path.is_file():
            return None
        return item

    def search(self, query: str, *, kind: str = "", limit: int = 8) -> list[MediaItem]:
        """Find media by title, best match first.

        Deliberately simple and deliberately explainable: a whole-phrase hit
        outranks all-words-present, which outranks some-words-present. Nobody
        can debug a relevance score in a chat window, and a wrong first result
        that you can see the reason for is far less annoying than a mysterious
        one.
        """
        needle = _normalize(query)
        if not needle:
            return []
        scan = self.index()
        # Ordinary words are dropped only when something is left without them.
        #
        # "So What" and "Let It Be" are real titles made entirely of words too
        # common to identify anything in a longer query. Emptying the query
        # would make those unfindable, so a query that is nothing but common
        # words is taken at face value: it is short, and somebody typed it.
        terms = _searchable_terms(needle) or [term for term in needle.split() if term]
        if not terms:
            return []

        scored: list[tuple[int, int, MediaItem]] = []
        for item in scan.items:
            if kind and item.kind != kind:
                continue
            hay = _normalize(f"{item.title} {item.path.parent.name}")
            words = set(hay.split())
            if needle in hay:
                rank = 0
            elif all(_hits(term, words) for term in terms):
                rank = 1
            else:
                hits = sum(1 for term in terms if _hits(term, words))
                # Most of what was asked for has to be there.
                #
                # LIVE 2026-08-19: "Play it — keep going until you get a 128
                # tile" was parsed as a seventeen-word query, one word of
                # which appeared inside a filename, and a video started
                # playing in the chat. Matching on any single term means every
                # long sentence containing "play" finds something.
                if hits * 2 < len(terms):
                    continue
                rank = 2 + (len(terms) - hits)
            # Shorter titles that still match are the more exact match:
            # "Blue in Green" beats "Blue in Green (Remastered Live 1997)".
            scored.append((rank, len(item.title), item))

        scored.sort(key=lambda row: (row[0], row[1]))
        return [item for _rank, _len, item in scored[: max(1, int(limit))]]


_LIBRARY: MediaLibrary | None = None


def get_media_library() -> MediaLibrary:
    global _LIBRARY
    if _LIBRARY is None:
        _LIBRARY = MediaLibrary()
    return _LIBRARY


def reset_media_library_for_test() -> None:
    global _LIBRARY
    _LIBRARY = None


__all__ = [
    "AUDIO_EXTENSIONS",
    "BROADLY_PLAYABLE",
    "MIME_TYPES",
    "VIDEO_EXTENSIONS",
    "MediaItem",
    "MediaLibrary",
    "ScanResult",
    "get_media_library",
    "reset_media_library_for_test",
]
