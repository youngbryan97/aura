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

__all__ = ["Discerned", "FOLLOWS", "discerned", "what_the_code_does"]

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

#: The witnesses, as they are said.
READ, ASKED, MODEL, FOLLOWED = "what I read", "your words", "my model", "what follows"

#: Kinds of program a person names, read from the person's words where nothing written names it.
_A_KIND = re.compile(r"\b(?:an?|the)\s+(?:[\w-]+\s+){0,3}?(word processor|text editor|note[- ]taking app|notes app|email (?:client|composer)|"
                     r"letter writer|document editor|writing app)\b", re.I)


@dataclass(frozen=True)
class _Said:
    text: str


@dataclass
class Discerned:
    """A program as discerned: its genome, who speaks for each feature, and what was not taken."""

    genome: Genome
    kind: str
    witnesses: dict[str, list[str]] = field(default_factory=dict)  # feature name -> who speaks for it
    followed: list[str] = field(default_factory=list)  # why each feature that follows from another does
    left_out: list[str] = field(default_factory=list)  # named by her model alone, borne out by nothing
    by_the_code: dict[str, str] = field(default_factory=dict)  # named, and done already by the code of a part that is in: feature -> part
    not_yet: list[str] = field(default_factory=list)  # borne out, and no part she knows makes them

    def said(self, program: str) -> list[str]:
        """What she says of it, a line for each thing worth knowing about how she came to it."""
        named = [n for n in (READ, ASKED, MODEL) if any(n in by for by in self.witnesses.values())]
        listed = _and([_lower(f.name) for f in self.genome.features])
        lines = [f"From {_and(named) or 'what I know'}, {program} is {_a(self.kind)} that does {len(self.genome.features)} things a person uses: "
                 f"{listed}. I am calling mine {self.genome.name}."]
        if self.followed:
            lines.append("What it does says more: " + "; ".join(self.followed) + ".")
        if self.by_the_code:
            done = [f"{name} with the {part}" for name, part in self.by_the_code.items()]
            lines.append(f"Some of it comes with what I make already, because the code does it: {_and(done)}.")
        if self.left_out:
            lines.append(f"My model also named {_and(self.left_out)}, which nothing I read bears out, so I leave "
                         + ("it" if len(self.left_out) == 1 else "them") + " out.")
        if self.not_yet:
            lines.append(f"What I read also speaks of {_and(self.not_yet)}, which I do not yet know how to make, so this build does not have "
                         + ("it." if len(self.not_yet) == 1 else "them."))
        return lines


def discerned(program: str, sources: Iterable[Any], asked: str = "", heard: Genome | None = None) -> Discerned | None:
    """The program ``program`` as every witness and what follows say it is, where its work is a page; else None."""
    from core.rebuilding.what_a_program_does import _its_kind
    from core.rebuilding.what_the_frame_gives import _opens, _saves_its_own

    sources = list(sources)
    text = " ".join(str(getattr(source, "text", "") or "") for source in sources)
    said_kind = _A_KIND.search(asked or "")
    kind = (_its_kind(sources[0].text, program) if sources and program else "") or (said_kind.group(1).lower() if said_kind else "") \
        or (_its_kind(heard.what_it_is, program) if heard is not None else "")
    # The kind decides whether its work is a page, as what is written or the person names it; her model's prose
    # about the work ("a page on a light-grey canvas") is not asked.
    probe = Genome(name="it", what_it_is=f"a {kind}", work=kind, features=[])
    if not kind or the_page_for(probe) is None:
        return None
    by: dict[str, list[str]] = {}

    def witness(part: str, who: str) -> None:
        if who not in by.setdefault(part, []):
            by[part].append(who)

    for part, _feature in what_is_written_asks_for(sources, set()):
        witness(part.name, READ)
    for part, _feature in what_is_written_asks_for([_Said(asked)], set()):
        witness(part.name, ASKED)
    left_out, not_yet, unplaced = [], [], []
    for feature in heard.features if heard is not None else []:
        parts = parts_for(feature)
        for part in parts:
            witness(part.name, MODEL)
        if not (parts or _opens(feature) or _saves_its_own(feature)):
            unplaced.append(feature)
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
    # What the code of what is in already does: a feature named for it is in, by that code, not one she cannot make.
    by_the_code: dict[str, str] = {}
    for feature in unplaced:
        part = next((name for name in ("document page", *have) if _does(name, feature.name)), None)
        if part is not None:
            by_the_code[_lower(feature.name)] = "page" if part == "document page" else part
        else:
            (not_yet if _borne_out(feature.name, text) else left_out).append(_lower(feature.name))
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
    genome = Genome(name=name, what_it_is=f"A {kind} of its own, rebuilt clean-room from what is written about {program or 'it'}",
                    work=f"A white {_the_paper_here()} page centred on a grey desk, under "
                         + ("a ribbon of tabs (Home, Insert, Layout, Review, View)." if ribbon else "a toolbar of its tools."),
                    accent=heard.accent if heard is not None else "#2b579a", features=features)
    return Discerned(genome, kind, witnesses=witnesses, followed=followed, left_out=left_out, by_the_code=by_the_code, not_yet=not_yet)


#: What a part's code gives a person: the commands, controls and readouts it puts in the frame, by their labels.
_GIVES = re.compile(r"app\.(?:command|control|status)\(\s*\{\s*label:\s*\"([^\"]+)\"")


def what_the_code_does(code: str) -> list[str]:
    """What ``code`` gives a person: the labels of the commands, controls and readouts it puts in the frame
    ("Zoom in", "Page", "Words"), and what its opening comment says it does ("counted in pages as it grows, zoomed")."""
    said = []
    for line in (code or "").splitlines():
        if not line.startswith("//"):
            break
        said.append(line.lstrip("/ "))
    return [*_GIVES.findall(code or ""), *([" ".join(said)] if said else [])]


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
    """A feature's name as it reads inside a sentence: "Headers and Footers" is "headers and footers", "PDF" stays "PDF"."""
    return " ".join(word if len(word) > 1 and word.isupper() else word[:1].lower() + word[1:] for word in name.split(" "))


def _a(kind: str) -> str:
    return ("an " if kind[:1].lower() in "aeiou" else "a ") + kind


def _and(items: list[str]) -> str:
    between = "; " if any("," in item for item in items) else ", "
    return items[0] if len(items) == 1 else between.join(items[:-1]) + (";" if between == "; " else "") + " and " + items[-1] if items else ""
