"""What a program is, discerned from every witness to it and from what follows from what it does.

A person rebuilding a program they know of does not take one account of it as
the last word. They read what is written about it, listen to what they are
asked for, hear what someone who knows it says, and then think: a program you
write pages in starts with an empty one, so it makes new documents; what is
typed can be mistyped, so it can be undone; what is printed is laid on paper,
so its paper and margins can be set. Each feature here is held with who speaks
for it, and a feature only one doubtful witness speaks for is not taken on its
word.

The witnesses:

- what is written about the program and its kind (her corpus, Wikipedia),
  read for the phrases that mean each part she knows how to make;
- the person's own words;
- her model's reading of the same articles (core/rebuilding/what_a_program_does.py),
  one witness among the others, and the one least sure: what it names is kept
  where it is a part she knows, or where what is written bears it out;
- what follows from what is already in, by what each part needs and implies.

Nothing here decides alone, nothing here is asked of her model, and nothing
here knows which program is being rebuilt. Where the program is not of a kind
she knows how to make (its work is not a page), there is nothing here to
discern by, and her model's reading stands as it is.
"""
from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from core.rebuilding.parts_a_maker_knows import (
    _GENERIC,
    _WRITING_NAMES,
    _WRITTEN_AS,
    PARTS,
    _the_paper_here,
    _words,
    parts_for,
    the_page_for,
    what_is_written_asks_for,
)
from core.rebuilding.what_a_program_does import Feature, Genome

__all__ = ["Discerned", "FOLLOWS", "discerned", "the_parts_for", "what_the_code_does"]

#: What a program that does one thing therefore does: (it has, so it has, because), by the parts she knows.
#: "document page" is the program's work being a page that is written on.
FOLLOWS: tuple[tuple[str, str, str], ...] = (
    ("document page", "new document", "a page is written from empty, so it starts new documents and closes them"),
    ("document page", "history", "what is typed can be mistyped, so it can be undone"),
    ("document page", "clipboard", "words on a page are moved about, so they can be cut, copied and pasted"),
    ("document page", "save as", "what is written is kept, so it is saved as files other programs open"),
    ("printing", "page setup", "what is printed is laid on paper, so its paper and margins can be set"),
    ("lists", "indent", "a list is set in from the margin, and a list within a list further in"),
    ("paragraph styles", "fonts", "a heading is a font and a size, so fonts and sizes can be set"),
)

#: The witnesses, as they are said. Each proposes what the program does; none decides alone.
READ, ONLINE, ASKED, MODEL, MEMORY, FOLLOWED = "what I read", "what is online", "your words", "my model", "what I built before", "what follows"

#: How each witness is said as the one saying something.
_WHO = {READ: "what I read", ONLINE: "what is online", ASKED: "you", MODEL: "my model", MEMORY: "what I built before"}

#: Kinds of program a person names, read from the person's words where nothing written names it.
_A_KIND = re.compile(r"\b(?:an?|the)\s+(?:[\w-]+\s+){0,3}?(word processor|text editor|note[- ]taking app|notes app|email (?:client|composer)|"
                     r"letter writer|document editor|writing app)\b", re.I)

#: How much of what a name means a part's doing must cover for the name to be that part's.
COVERED = 2 / 3


@dataclass(frozen=True)
class _Said:
    text: str


@dataclass
class Discerned:
    """A program as discerned: its genome, who speaks for each feature, and what is not in it and who said it should be."""

    genome: Genome
    kind: str
    witnesses: dict[str, list[str]] = field(default_factory=dict)  # feature name -> who speaks for it
    followed: list[str] = field(default_factory=list)  # why each feature that follows from another does
    by_the_code: dict[str, str] = field(default_factory=dict)  # named, and done already by the code of a part that is in: name -> part
    not_yet: dict[str, list[str]] = field(default_factory=dict)  # named, and no part she knows makes it: name -> who named it

    def said(self, program: str) -> list[str]:
        """What she says of it, a line for each thing worth knowing about how she came to it."""
        named = [n for n in (READ, ONLINE, ASKED, MODEL, MEMORY) if any(n in by for by in self.witnesses.values())]
        listed = _and([_lower(f.name) for f in self.genome.features])
        lines = [f"From {_and(named) or 'what I know'}, {program} is {_a(self.kind)} that does {len(self.genome.features)} things a person uses: "
                 f"{listed}. I am calling mine {self.genome.name}."]
        if self.followed:
            lines.append("What it does says more: " + "; ".join(self.followed) + ".")
        if self.by_the_code:
            done = [f"{name} with the {part}" for name, part in self.by_the_code.items()]
            lines.append(f"Some of it comes with what I make already, because the code does it: {_and(done)}.")
        if self.not_yet:
            # Each set of witnesses once, with everything that set names: "my model and what I read say it also has ...".
            order = (MODEL, READ, ONLINE, MEMORY, ASKED)
            groups: dict[tuple[str, ...], list[str]] = {}
            for name, by in self.not_yet.items():
                groups.setdefault(tuple(w for w in order if w in by), []).append(name)
            whose = [f"what I built before also had {_and(names)}" if who == (MEMORY,) else
                     f"{_and([_WHO.get(w, w) for w in who])} {'say' if len(who) > 1 or who == (ASKED,) else 'says'} it also has {_and(names)}"
                     for who, names in groups.items()]
            many = len(self.not_yet) > 1
            lines.append(_capital("; ".join(whose)) + f". I do not yet know how to make {'those' if many else 'that'}, "
                         f"so this build does not have {'them' if many else 'it'}.")
        return lines


def discerned(program: str, sources: Iterable[Any], asked: str = "", heard: Genome | None = None, *,
              remembered: Iterable[Genome] = ()) -> Discerned | None:
    """The program ``program`` as every witness and what follows say it is, where its work is a page; else None.

    ``sources``: what is written about it (her own copy of Wikipedia, or online). ``asked``: the person's words.
    ``heard``: what her model says of it from its own knowledge. ``remembered``: programs of its kind she built before.
    """
    from core.rebuilding.what_a_program_does import _its_kind
    from core.rebuilding.what_the_frame_gives import _opens, _saves_its_own

    sources = list(sources)
    text = " ".join(str(getattr(source, "text", "") or "") for source in sources)
    said_kind = _A_KIND.search(asked or "")
    kind = (_its_kind(sources[0].text, program) if sources and program else "") or (said_kind.group(1).lower() if said_kind else "") \
        or (_its_kind(heard.what_it_is, program) if heard is not None else "")
    # The kind decides whether its work is a page, as what is written or the person names it; a witness's prose
    # about the work ("a page on a light-grey canvas") is not asked.
    probe = Genome(name="it", what_it_is=f"a {kind}", work=kind, features=[])
    if not kind or the_page_for(probe) is None:
        return None
    by: dict[str, list[str]] = {}

    def witness(part: str, who: str) -> None:
        if who not in by.setdefault(part, []):
            by[part].append(who)

    # What is written, and the person's words, read for the phrases that mean each part.
    for source in sources:
        who = ONLINE if str(getattr(source, "where", "")).startswith("on ") else READ
        for part, _feature in what_is_written_asks_for([source], set()):
            witness(part.name, who)
    for part, _feature in what_is_written_asks_for([_Said(asked)], set()):
        witness(part.name, ASKED)
    # What her model and what she built before name, each name weighed by what she knows how to make.
    claims = [(f, MODEL) for f in (heard.features if heard is not None else [])] + [(f, MEMORY) for g in remembered for f in g.features]
    unplaced: dict[str, list[str]] = {}
    for feature, who in claims:
        parts = the_parts_for(feature.name)
        for part in parts:
            witness(part, who)
        if not (parts or _opens(feature) or _saves_its_own(feature)):
            unplaced.setdefault(_lower(feature.name), [])
            if who not in unplaced[_lower(feature.name)]:
                unplaced[_lower(feature.name)].append(who)
    # What follows from what is in, until nothing more does.
    followed: list[str] = []
    have = {"document page", *by}
    while True:
        more = [(then, because) for given, then, because in FOLLOWS if given in have and then not in have]
        if not more:
            break
        for then, because in more:
            have.add(then)
            witness(then, FOLLOWED)
            followed.append(because)
    # What the code of what is in already does is in by that code; what no part makes is not in, and who named it is said.
    by_the_code: dict[str, str] = {}
    not_yet: dict[str, list[str]] = {}
    for name, who in unplaced.items():
        part = next((p for p in ("document page", *have) if _does(p, name)), None)
        if part is not None:
            by_the_code[name] = "page" if part == "document page" else part
        else:
            not_yet[name] = [*who, *([READ] if READ not in who and _borne_out(name, text) else [])]
    files = [Feature(name="Open", how="Open a document, pick its file", shows="the document on the page", place="File", weight=3),
             Feature(name="Save", how="Save, or press Ctrl+S", shows="the document saved as a file", place="File", weight=3)]
    witnesses = {f.name: [FOLLOWED] for f in files}
    features = list(files)
    for part in PARTS:
        if part.name in by and part.name in _WRITTEN_AS:
            feature = _WRITTEN_AS[part.name][1].model_copy()
            features.append(feature)
            witnesses[feature.name] = by[part.name]
    places = ["File", "Edit", "Format", "Insert", "Layout", "Tools"]
    features.sort(key=lambda f: (places.index(f.place) if f.place in places else len(places), -f.weight))
    ribbon = re.search(r"\bribbon\b", f"{text} {heard.work if heard is not None else ''}", re.I) is not None
    name = heard.name if heard is not None and heard.name and heard.name.lower() not in program.lower() else _a_name(program)
    genome = Genome(name=name, what_it_is=f"A {kind} of its own, rebuilt clean-room from what is known of {program or 'it'}",
                    work=f"A white {_the_paper_here()} page centred on a grey desk, under "
                         + ("a ribbon of tabs (Home, Insert, Layout, Review, View)." if ribbon else "a toolbar of its tools."),
                    accent=heard.accent if heard is not None else "#2b579a", features=features)
    return Discerned(genome, kind, witnesses=witnesses, followed=followed, by_the_code=by_the_code, not_yet=not_yet)


def the_parts_for(name: str) -> list[str]:
    """The parts a feature named ``name`` is: those that do all its name asks, else the one whose doing covers most of what it means.

    "Spell-check and grammar hints" asks for more than the proofing part's words
    say ("hints"), and is three-quarters what proofing does: it is proofing. A
    name of a single meaning word is one part's or none: covering it by part is
    a guess.
    """
    parts = parts_for(Feature(name=name))
    if parts:
        return [p.name for p in parts]
    meant = _meant(name)
    if len(meant) < 2:
        return []
    covers = [(len(meant & p.does) / len(meant), p.name) for p in PARTS if p.name not in ("document page", "tabbed toolbar")]
    share, best = max(covers, default=(0.0, ""))
    return [best] if share >= COVERED else []


#: What a part's code gives a person: the commands, controls and readouts it puts in the frame, by their labels.
_GIVES = re.compile(r"app\.(command|control|status)\(\s*\{\s*label:\s*\"([^\"]+)\"")

#: What each way of plugging into the frame gives, beside its label: a readout is a count or measure shown in the status bar.
_WHERE_IT_GIVES = {"command": "", "control": "", "status": " count shown in the status bar"}


def what_the_code_does(code: str) -> list[str]:
    """What ``code`` gives a person: the labels of the commands, controls and readouts it puts in the frame
    ("Zoom in", "Page", "Words"), and what its opening comment says it does ("counted in pages as it grows, zoomed")."""
    said = []
    for line in (code or "").splitlines():
        if not line.startswith("//"):
            break
        said.append(line.lstrip("/ "))
    return [*(label + _WHERE_IT_GIVES[how] for how, label in _GIVES.findall(code or "")), *([" ".join(said)] if said else [])]


def _does(part: str, name: str) -> bool:
    """Whether the code of ``part`` gives what a feature named ``name`` asks for: each meaning word of the name in one thing it gives."""
    from core.rebuilding.parts_a_maker_knows import _code

    meant = _meant(name)
    return bool(meant) and any(meant <= _words(given) for given in what_the_code_does(_code().get(part, "")))


def _meant(name: str) -> set[str]:
    """The words of a feature's name that say what it is; all of them, where every one is a common word ("Select all")."""
    return (_words(name) - _GENERIC) or _words(name)


def _borne_out(name: str, text: str) -> bool:
    """Whether what is written speaks of a feature: every word of its name that means something is in it."""
    meant = _meant(name)
    written = _words(text)
    return bool(meant) and meant <= written


def _a_name(program: str) -> str:
    name = _WRITING_NAMES[sum(map(ord, program)) % len(_WRITING_NAMES)]
    return name if name.lower() not in program.lower() else _WRITING_NAMES[(_WRITING_NAMES.index(name) + 1) % len(_WRITING_NAMES)]


def _lower(name: str) -> str:
    """A feature's name as it reads inside a sentence: a title-cased "Headers and Footers" is "headers and footers"; a name
    already in a sentence's case keeps its proper nouns ("Save as PDF or Word document"); "PDF" stays "PDF"."""
    words = name.split(" ")
    titled = all(w[:1].isupper() for w in words if len(w) > 3)
    lowered = [w if len(w) > 1 and w.isupper() else w[:1].lower() + w[1:] for w in words]
    return " ".join(lowered) if titled else (lowered[0] + (" " + " ".join(words[1:]) if len(words) > 1 else ""))


def _capital(said: str) -> str:
    return said[:1].upper() + said[1:]


def _a(kind: str) -> str:
    return ("an " if kind[:1].lower() in "aeiou" else "a ") + kind


def _and(items: list[str]) -> str:
    between = "; " if any("," in item for item in items) else ", "
    return items[0] if len(items) == 1 else between.join(items[:-1]) + (";" if between == "; " else "") + " and " + items[-1] if items else ""
