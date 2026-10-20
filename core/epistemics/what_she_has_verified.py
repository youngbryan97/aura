"""What she has verified, kept: facts she checked online, held as verified only when reputable sources agree (L10).

A claim she checks online (core/epistemics/epistemic_reach.py) used to correct one reply and be forgotten. Here each
check is kept as an attestation (the source, its tier, what it said, when it was read), and a claim's standing is
read from all of its attestations together:

- **verified** when at least two independent sources support it, at least one of them a reference, official or
  primary source, and no source of the same standing or higher contradicts it; or when a program computes it;
- **disputed** when sources of that standing disagree, and held that way, never resolved by the newest read;
- **single source** while only one counted source supports it, which is not verified;
- **refuted** when two independent sources contradict it and none of equal standing supports it.

Sources rank by what they are: reference, official and primary above news, and news above blogs, forums and social
media, which are kept but never counted toward the two. Two sources are independent when different publishers
hold them. Only checkable factual claims are kept: never an opinion, an evaluation, a prediction or a preference.
A dated fact (one that names the present: who holds an office now, a current price, a latest record) is checked
again before it is used on a later day than it was read.

Offline first: the store is a file read without the network, and with the network blocked everything here still
answers; only a new check needs a source.
"""
from __future__ import annotations

import json
import logging
import operator
import re
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from urllib.parse import urlparse

logger = logging.getLogger("Aura.Epistemics.Verified")

SUPPORTED, CONTRADICTED = "SUPPORTED", "CONTRADICTED"
VERIFIED, DISPUTED, SINGLE_SOURCE, REFUTED, UNSETTLED = "verified", "disputed", "single_source", "refuted", "unsettled"

#: What a source is, by its publisher. Reference, official and primary sources stand above news; news above the rest.
_REFERENCE = ("wikipedia.org", "britannica.com", "merriam-webster.com", "oxfordreference.com", "dictionary.com",
              "encyclopedia.com", "worldatlas.com", "nationalgeographic.com", "si.edu")
_PRIMARY = ("doi.org", "nature.com", "science.org", "sciencedirect.com", "springer.com", "link.springer.com", "arxiv.org",
            "jstor.org", "plos.org", "cell.com", "nejm.org", "thelancet.com", "ieee.org", "acm.org", "pnas.org",
            "bmj.com", "wiley.com")
_NEWS = ("reuters.com", "apnews.com", "bbc.co.uk", "bbc.com", "nytimes.com", "theguardian.com", "washingtonpost.com",
         "wsj.com", "ft.com", "npr.org", "aljazeera.com", "cnn.com", "bloomberg.com", "economist.com", "pbs.org")
_SOCIAL = ("reddit.com", "quora.com", "medium.com", "substack.com", "x.com", "twitter.com", "facebook.com",
           "instagram.com", "tiktok.com", "youtube.com", "stackexchange.com", "stackoverflow.com", "blogspot.com",
           "wordpress.com", "tumblr.com", "pinterest.com", "fandom.com", "linkedin.com")
_STANDING = {"computed": 4, "reference": 3, "official": 3, "primary": 3, "news": 2, "blog_forum_social": 1, "unranked": 1}
#: Tiers whose sources count toward the two; the rest are kept, and counted for nothing.
_COUNTED = {"computed", "reference", "official", "primary", "news"}
#: The second level of a country's domain that a government holds (gov.uk, gouv.fr, gob.mx, go.jp, gc.ca, admin.ch).
_A_GOVERNMENT = {"gov", "gouv", "gob", "go", "govt", "gc", "admin", "bund", "mil", "gv"}
#: The second level of a country's domain that is not a publisher (bbc.co.uk is the BBC, not "co.uk").
_SECOND_LEVELS = {"co", "ac", "gov", "gouv", "org", "com", "net", "edu", "gob", "go", "or", "ne", "govt", "nic", "mil", "gc"}


def _host(url: str) -> str:
    return (urlparse(str(url or "")).hostname or "").lower().removeprefix("www.")


def publisher(url: str) -> str:
    """Who holds a source: its registrable domain (en.wikipedia.org and de.wikipedia.org are one publisher)."""
    labels = _host(url).split(".")
    if len(labels) >= 3 and labels[-2] in _SECOND_LEVELS and len(labels[-1]) == 2:
        return ".".join(labels[-3:])
    return ".".join(labels[-2:])


def tier_of(url: str) -> str:
    """What a source is: reference, official, primary, news, blog_forum_social, or unranked."""
    host = _host(url)
    if url.startswith("program:"):
        return "computed"

    def under(names: tuple[str, ...]) -> bool:
        return any(host == n or host.endswith("." + n) for n in names)

    labels = host.split(".")
    governs = len(labels) >= 2 and len(labels[-1]) == 2 and labels[-2] in _A_GOVERNMENT
    if host.endswith((".gov", ".mil", ".int")) or governs or host.endswith(("europa.eu", "un.org", "who.int", "canada.ca")):
        return "official"
    if under(_REFERENCE):
        return "reference"
    if under(_PRIMARY) or host.endswith(".edu") or "ac" in labels[-3:-1]:
        return "primary"
    if under(_NEWS):
        return "news"
    if under(_SOCIAL):
        return "blog_forum_social"
    return "unranked"


_NOT_A_FACT = re.compile(
    r"\b(?:i|we|you)\s+(?:think|believe|feel|guess|suppose|reckon|prefer|love|hate|like)\b|\bin my (?:view|opinion)\b|"
    r"\b(?:should|ought|must|best|worst|better|worse|beautiful|ugly|amazing|terrible|awful|boring|overrated|"
    r"underrated|favou?rite|greatest|nicest|wonderful|horrible|delicious|disgusting)\b|"
    r"\b(?:will|going to|likely|unlikely|probably|might|may|could|forecast|expected to|predicted to)\b", re.I)
_NAMES_SOMETHING = re.compile(r"\d|\b[A-Z][a-z]+")
_DATED = re.compile(r"\b(?:current(?:ly)?|now|today|latest|recent(?:ly)?|as of|this year|incumbent|president|"
                    r"prime minister|ceo|chair(?:man|woman|person)?|population|record|champion|price|rate|ranked)\b", re.I)


def checkable(claim: str) -> bool:
    """A factual claim a source can settle: it names something, and it is not an opinion, an evaluation, a
    prediction or a preference, nor a question."""
    text = " ".join(str(claim or "").split())
    if not text or text.endswith("?") or _NOT_A_FACT.search(text):
        return False
    return bool(_NAMES_SOMETHING.search(text[1:] if text[:1].isupper() else text))


def dated(claim: str) -> bool:
    """Whether a claim names the present, so that it may have changed since it was read."""
    return bool(_DATED.search(str(claim or "")))


_ARITHMETIC = re.compile(r"(-?\d+(?:\.\d+)?)\s*([+\-*/×x÷])\s*(-?\d+(?:\.\d+)?)\s*(?:=|is|equals)\s*(-?\d+(?:\.\d+)?)")
_OPERATIONS = {"+": operator.add, "-": operator.sub, "*": operator.mul, "×": operator.mul, "x": operator.mul,
               "/": operator.truediv, "÷": operator.truediv}


def computed(claim: str) -> str:
    """SUPPORTED or CONTRADICTED where the claim states arithmetic a program can do, else ""."""
    found = _ARITHMETIC.search(str(claim or "").replace(",", ""))
    if not found:
        return ""
    left, op, right, stated = found.groups()
    try:
        value = _OPERATIONS[op](float(left), float(right))
    except ZeroDivisionError:
        return CONTRADICTED
    return SUPPORTED if abs(value - float(stated)) <= 1e-9 * max(1.0, abs(value)) else CONTRADICTED


def key_of(claim: str) -> str:
    """The same claim however it is spaced, cased or punctuated."""
    return " ".join(re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", str(claim or "").lower()))


@dataclass
class Attestation:
    """What one source said of a claim, and when it was read."""

    url: str
    verdict: str
    quote: str = ""
    read_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%d"))

    @property
    def tier(self) -> str:
        return tier_of(self.url)

    @property
    def publisher(self) -> str:
        return "program" if self.url.startswith("program:") else publisher(self.url)


@dataclass
class Fact:
    """A claim she checked, with everything that was read of it."""

    claim: str
    attestations: list[Attestation] = field(default_factory=list)

    @property
    def dated(self) -> bool:
        return dated(self.claim)

    @property
    def status(self) -> str:
        return standing(self.attestations)

    @property
    def last_read(self) -> str:
        return max((a.read_at for a in self.attestations), default="")

    def due_again(self, today: str | None = None) -> bool:
        """A dated fact is read again before it is used on a later day than it was last read."""
        return self.dated and self.last_read < (today or time.strftime("%Y-%m-%d"))


def standing(attestations: list[Attestation]) -> str:
    """A claim's standing from everything read of it (the module's rules)."""
    if any(a.tier == "computed" for a in attestations):
        return VERIFIED if next(a for a in attestations if a.tier == "computed").verdict == SUPPORTED else REFUTED
    counted = [a for a in attestations if a.tier in _COUNTED]

    def side(verdict: str) -> tuple[set[str], int]:
        held = [a for a in counted if a.verdict == verdict]
        return {a.publisher for a in held}, max((_STANDING[a.tier] for a in held), default=0)

    for_it, best_for = side(SUPPORTED)
    against, best_against = side(CONTRADICTED)
    strong = _STANDING["reference"]
    if for_it and against and best_against >= best_for:
        return DISPUTED
    if len(for_it) >= 2 and best_for >= strong:
        return VERIFIED
    if len(against) >= 2 and best_against >= strong and not for_it:
        return REFUTED
    if for_it or against:
        return SINGLE_SOURCE if len(for_it | against) == 1 else UNSETTLED
    return UNSETTLED


class VerifiedFacts:
    """Her store of checked claims: a file, read without the network."""

    def __init__(self, path: Path | None = None) -> None:
        if path is None:
            from core.config import config

            path = Path(config.paths.data_dir) / "epistemics" / "verified_facts.json"
        self.path = path
        self._lock = threading.Lock()
        self._facts: dict[str, Fact] = {}
        self._load()

    def _load(self) -> None:
        try:
            rows = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        for row in rows if isinstance(rows, list) else []:
            fact = Fact(claim=str(row.get("claim") or ""), attestations=[Attestation(**a) for a in row.get("attestations") or []])
            if fact.claim:
                self._facts[key_of(fact.claim)] = fact

    def _save(self) -> None:
        from core.runtime.atomic_writer import atomic_write_text

        rows = [{"claim": f.claim, "status": f.status, "attestations": [asdict(a) for a in f.attestations]}
                for f in self._facts.values()]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(self.path, json.dumps(rows, indent=1, ensure_ascii=False))

    def attest(self, claim: str, url: str, verdict: str, quote: str = "") -> Fact | None:
        """Keep what a source said of a checkable claim; a program's own reading is attested with it where it can."""
        if verdict not in (SUPPORTED, CONTRADICTED) or not checkable(claim):
            return None
        with self._lock:
            fact = self._facts.setdefault(key_of(claim), Fact(claim=" ".join(claim.split())))
            fact.attestations.append(Attestation(url=url, verdict=verdict, quote=" ".join(str(quote).split())[:300]))
            reckoned = computed(claim)
            if reckoned and not any(a.tier == "computed" for a in fact.attestations):
                fact.attestations.append(Attestation(url="program:arithmetic", verdict=reckoned))
            self._save()
            return fact

    def get(self, claim: str) -> Fact | None:
        return self._facts.get(key_of(claim))

    def bearing_on(self, text: str, *, at_most: int = 5) -> list[Fact]:
        """The settled facts a text bears on, by the words they share, most shared first: what she knows of it."""
        words = set(key_of(text).split())
        scored = []
        for fact in self._facts.values():
            if fact.status not in (VERIFIED, DISPUTED, REFUTED):
                continue
            mine = set(key_of(fact.claim).split())
            shared = len(words & mine) / max(1, len(mine))
            if shared >= 0.5:
                scored.append((shared, fact))
        return [fact for _shared, fact in sorted(scored, key=lambda pair: -pair[0])[:at_most]]

    def __len__(self) -> int:
        return len(self._facts)


def as_evidence(facts: list[Fact], *, today: str | None = None) -> str:
    """What she has verified that bears on a turn, as data: each claim, its standing, its sources and when read."""
    if not facts:
        return ""
    lines = ["[CHECKED FACTS, KEPT]"]
    for fact in facts:
        sources = "; ".join(f"{a.publisher} ({a.tier}, {a.verdict.lower()}, read {a.read_at})" for a in fact.attestations)
        dated_note = f" (names the present; last read {fact.last_read})" if fact.due_again(today) else ""
        lines.append(f"- {fact.claim} — {fact.status}{dated_note}. Sources: {sources}")
    lines.append("[END CHECKED FACTS, KEPT]")
    return "\n".join(lines)


_STORE: VerifiedFacts | None = None


def her_verified_facts() -> VerifiedFacts:
    """The one store for this process."""
    global _STORE
    if _STORE is None:
        _STORE = VerifiedFacts()
    return _STORE
