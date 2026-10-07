"""What a program does, learned from what is written about it, never from its code.

A clean-room rebuild starts from a description: what the program is, what a
person does with it, and what they see when it works. That is read here from
what is written about it and its kind of program: the encyclopedia she keeps
(her own corpus, then Wikipedia online when the corpus lacks the page), and
what her model already knows. Her model reads those and writes the program's
features as typed data, each one something a person does and what they then
see. Then, for each feature, the checks a person would make
(checks_a_person_makes.py), written before any code exists.

Her model is a reader and a proposer here, never an authority: a feature is
only what it says, and a check is kept only if it fails on the empty frame.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field

from core.rebuilding.checks_a_person_makes import Check

logger = logging.getLogger("Rebuilding.WhatItDoes")

__all__ = ["Asker", "Feature", "Genome", "Source", "checks_for", "genome_of", "what_is_written_about"]

#: The most of one source shown to her model, in characters.
SOURCE_CHARS = 14_000

#: How many features a program is rebuilt with, at most.
MOST_FEATURES = 30

#: Features whose checks are written in one ask.
FEATURES_AT_ONCE = 3

#: Ask her model for typed data: (prompt, schema class, max tokens) -> instance or None.
Asker = Callable[[str, type[BaseModel], int], Awaitable[BaseModel | None]]


@dataclass
class Source:
    """Something written about the program, and where it came from."""

    title: str
    text: str
    where: str


class Feature(BaseModel):
    name: str = Field(max_length=80)
    how: str = Field(default="", max_length=400, description="what a person does to use it, in a few words")
    shows: str = Field(default="", max_length=400, description="what they see when it worked, in a few words")
    place: str = Field(default="", max_length=40, description="the menu it belongs in, e.g. File, Edit, Insert, Format, View, Tools")
    weight: int = Field(default=2, ge=1, le=3, description="3 = the program is useless without it, 1 = a nicety")


class Genome(BaseModel):
    name: str = Field(max_length=60, description="a name of its own for the rebuilt program, not the original's")
    what_it_is: str = Field(default="", max_length=400)
    work: str = Field(default="", max_length=400, description="what a person works on in the main area, and how it looks")
    accent: str = Field(default="#2b579a", max_length=7, description="the colour of its title bar, as #rrggbb")
    kind: str = Field(default="application", description=(
        '"application" for a program used through a window (documents, tools, games); '
        '"code" for one used by calling its functions or running it in a terminal (a library, a command-line tool)'))
    features: list[Feature] = Field(default_factory=list)


class _Checks(BaseModel):
    checks: list[Check] = Field(default_factory=list)


async def what_is_written_about(program: str, *, corpus: Any = None, online: bool = True) -> list[Source]:
    """The program's own article and its kind's, from her corpus and else from Wikipedia."""
    found: list[Source] = []
    article = await asyncio.to_thread(_from_the_corpus, program, corpus)
    if article is None and online:
        article = await _from_wikipedia(program)
    if article is None:
        return found
    found.append(article)
    kind = _its_kind(article.text, program)
    words = kind.split()
    # "spreadsheet editor" has no article of its own; "Spreadsheet" has.
    for name in dict.fromkeys(n for n in (kind, " ".join(words[:-1]) if len(words) > 2 else words[0] if len(words) == 2 else "") if n):
        written = await asyncio.to_thread(_from_the_corpus, name, corpus)
        if written is None and online:
            written = await _from_wikipedia(name)
        if written is not None and written.title.lower() != article.title.lower():
            found.append(written)
            break
    return found


def what_was_read(sources: list[Source]) -> str:
    """What was read, as a person says it: 'the articles “Microsoft Word” and “Word processor” in my own copy of Wikipedia'."""
    by_where: dict[str, list[str]] = {}
    for source in sources:
        by_where.setdefault(source.where, []).append(f"“{source.title}”")
    return _and([f"the article{'s' if len(titles) > 1 else ''} {_and(titles)} {where}" for where, titles in by_where.items()])


def _and(items: list[str]) -> str:
    """Items as a list is said: "a, b and c"; with semicolons where an item has commas of its own."""
    between = "; " if any("," in item for item in items) else ", "
    return items[0] if len(items) == 1 else between.join(items[:-1]) + (";" if between == "; " else "") + " and " + items[-1] if items else ""


def _its_kind(text: str, program: str) -> str:
    """The kind of program an article opens by saying this one is: 'a word processing program' -> 'word processor'."""
    first = re.split(r"(?<=[.!?])\s", text.strip(), maxsplit=1)[0]
    said = re.search(r"\b(?:is|was) (?:an?|the) ([a-z][a-z \-]{2,60}?)(?: (?:program|application|software|app|tool|suite|package)\b| developed| by| for| that| from| which| and| in| of| with| used| included|[,.;(])", first)
    if not said:
        return ""
    kind = said.group(1).strip()
    words = kind.split()
    while words and words[0] in {"proprietary", "free", "open-source", "commercial", "popular", "freeware", "cross-platform", "online", "desktop", "web-based"}:
        words = words[1:]
    kind = " ".join(words)
    if kind.endswith(" processing"):
        kind = kind[: -len(" processing")] + " processor"
    return kind if kind and kind.lower() not in program.lower() else ""


def _from_the_corpus(title: str, corpus: Any) -> Source | None:
    try:
        if corpus is None:
            from core.knowledge.local_corpus import get_local_corpus_store

            corpus = get_local_corpus_store()
        spellings = (title, title[:1].upper() + title[1:], " ".join(w[:1].upper() + w[1:] for w in title.split()))
        hit = next((h for t in dict.fromkeys(spellings) if (h := corpus.by_title(t))), None)
        if hit is None:
            plain = re.sub(r"\(.*?\)", " ", title)
            words = {w for w in re.findall(r"[a-z0-9]+", plain.lower()) if len(w) > 2}
            hits = [h for h in corpus.search(plain, limit=8) or [] if words and words <= set(re.findall(r"[a-z0-9]+", h.title.lower()))]
            hit = min(hits, key=lambda h: len(h.title), default=None)
        if hit is None:
            return None
        kept = {"wikipedia": "Wikipedia"}.get(str(hit.source or "").lower(), str(hit.source or "").strip() or "what I keep")
        return Source(hit.title, corpus.body(hit.doc_id, max_chars=SOURCE_CHARS * 3), f"in my own copy of {kept}")
    except Exception as why:  # noqa: BLE001 - no corpus here is no source, not a failure
        logger.info("her corpus could not be read for %r: %s", title, why)
        return None


async def _from_wikipedia(title: str) -> Source | None:
    try:
        import httpx

        async with httpx.AsyncClient(timeout=20.0, headers={"User-Agent": "Aura/1.0 (a local assistant reading one article)"}) as client:
            reply = await client.get("https://en.wikipedia.org/w/api.php", params={
                "action": "query", "prop": "extracts", "explaintext": "1", "redirects": "1",
                "titles": title, "format": "json",
            })
            pages = (reply.json().get("query") or {}).get("pages") or {}
            for page in pages.values():
                text = str(page.get("extract") or "")
                if text:
                    return Source(str(page.get("title") or title), text, "on Wikipedia")
    except Exception as why:  # noqa: BLE001 - offline, or the page is not there: one source fewer
        logger.info("Wikipedia could not be read for %r: %s", title, why)
    return None


def _how_soon(section: str) -> int:
    """Where a section of an article goes: what the program does first, its history last."""
    if re.match(r"(features|functions|usage|user interface|file formats?|editing|design|capabilit)", section.lower()):
        return 0
    if re.match(r"(history|release|reception|criticism|versions?|see also|references|external)", section.lower()):
        return 2
    return 1


def _the_useful_part(text: str) -> str:
    """The part of an article that says what a program does: features first, history last."""
    sections = re.split(r"\n(?=[A-Z][^\n]{2,60}\.?\n)", text)
    head, rest = sections[0], sections[1:]
    ordered = [head, *sorted(rest, key=_how_soon)]
    return "\n".join(ordered)[:SOURCE_CHARS]


async def genome_of(program: str, sources: list[Source], ask: Asker, *, asked: str = "") -> Genome | None:
    """Her model reads what is written, and what the person asked for, and says what the program does as features a person uses.

    With a program named, it is rebuilt clean-room from what is written about
    it; with only the person's words, it is built to their specification. The
    person's words come first either way: "rebuild it, with a dark theme" asks
    for the theme too.
    """
    written = "\n\n".join(f"[{s.title}, {s.where}]\n{_the_useful_part(s.text)}" for s in sources)
    wanted = f"What the person asked for, in their words: {asked}\n" if asked else ""
    if program:
        opening = f"The program to rebuild, clean-room, as a web application: {program}.\n{wanted}"
        reading = "From what the person asked for, what is written about it below, and what you know of it and of programs of its kind, "
    else:
        opening = f"A program to build as a web application, to the person's specification.\n{wanted}"
        reading = "From what the person asked for, and what you know of programs of this kind, "
    prompt = (
        opening + reading
        + f"list the features a person uses, at most {MOST_FEATURES}, the ones a person would miss first ahead. "
        "Everything the person asked for is a feature. Each says what a person does and what they then see. "
        "Leave out what a web page cannot do (installing, licensing, cloud accounts, other programs). "
        "Give the program a name of its own.\n\n"
        + (written or ("(nothing written was found; use what you know)" if program else ""))
    )
    genome = await ask(prompt, Genome, 2048)
    if not isinstance(genome, Genome) or not genome.features:
        return None
    seen: set[str] = set()
    genome.features = [f for f in genome.features if not (f.name.lower() in seen or seen.add(f.name.lower()))]
    genome.features.sort(key=lambda f: -f.weight)
    genome.features = genome.features[:MOST_FEATURES]
    return genome


_HOW_CHECKS_ARE_WRITTEN = """\
A check is steps a person takes in the program, then what they see. Steps ("do"):
  type        type value into the main work area (or the field that has focus); "\\n" is Enter
  select_text select the text target in the work area
  select_all  select everything in the work area
  click       click the control (button, menu item) whose visible name or tooltip is target
  press       press keys, e.g. value "Ctrl+B"
  fill        put value in the field labelled target (in a dialog, or the toolbar)
  choose      choose value in the list labelled target
  click_text  put the cursor at the end of the text target
  give_file   when the program next asks for a file to open, give it one named target holding value
  wait        wait value milliseconds
What is seen ("see"):
  text / no_text      the text target is / is not shown
  style               the text target shows CSS property = value (e.g. font-weight bold, font-style italic,
                      text-decoration underline, color red, font-size 18px, text-align center, font-family Georgia)
  element / count     the work area has an element matching CSS selector target (count: exactly value of them)
  download            a file was saved whose name ends with value and that contains target
  dialog              a dialog titled or saying target is open
  value               the field labelled target holds value
  printed             printing was asked for
  text_after_reopen   after the program is closed and opened again, the text target is shown
  name                the document is named value
  moving              the picture (CSS selector target, else the largest canvas) changes on its own: a game runs
  watched             a game is watched as it plays and target is seen right: "controls" (the keys move what the
                      player controls the way they are named), "went through" (things turn back off the player
                      rather than pass through), "credited" (a miss counts for the right side), "escaped" (nothing
                      leaves the play without counting), "idle" (the other side moves and plays)
Controls are named by what a person reads on them: "Bold", "Insert Table", "Save".
Each check starts from a fresh, empty program, so it types what it needs first."""


async def checks_for(genome: Genome, features: list[Feature], ask: Asker, *, sources: list[Source] = (), asked: str = "") -> list[Check]:
    """The checks a person would make of these features: one for each of a feature's rules, the rules found by code.

    A check written from a feature's name alone guesses what it should show,
    and a wrong guess throws away right code. Tests written one for each stated
    rule catch more faults and reject less right code than tests asked to "try
    the edges" (Specification Grounding Drives Test Effectiveness for LLM Code,
    2026). A feature's rules are what it says it does and the sentences written
    about it (`rules_of`); which rules got a check is counted here, and only
    the rules left without one are asked for again.
    """
    named = {f.name.lower(): f.name for f in features}
    unchecked = {f.name: rules_of(f, sources) for f in features}
    kept: list[Check] = []
    for _round in range(2):
        wanting = [f for f in features if unchecked[f.name]]
        if not wanting:
            break
        listed = "\n".join(
            f"- {f.name} (menu: {f.place or 'any'})\n" + "\n".join(f"  rule {n}: {r}" for n, r in enumerate(unchecked[f.name], start=1))
            for f in wanting
        )
        wanted = f"What the person asked for, in their words: {asked}\n" if asked else ""
        prompt = (
            f"{genome.name}: {genome.what_it_is}\nThe work area: {genome.work}\n{wanted}\n"
            f"Write a check for each rule of each feature below, naming the feature exactly and the rule it checks:\n{listed}\n\n"
            + _HOW_CHECKS_ARE_WRITTEN
        )
        got = await ask(prompt, _Checks, CHECKS_TOKENS)
        new: list[Check] = []
        for check in got.checks if isinstance(got, _Checks) else []:
            name = named.get(check.feature.lower()) or next((n for k, n in named.items() if k in check.feature.lower() or check.feature.lower() in k), None)
            if name is not None and check.expect:
                new.append(check.model_copy(update={"feature": name}))
        kept += new
        # Which rules have a check now, counted here; only the rest are asked for again.
        for f in wanting:
            mine = [c for c in new if c.feature == f.name]
            unchecked[f.name] = [r for n, r in enumerate(unchecked[f.name], start=1) if not any(_checks_rule(c, r, n) for c in mine)]
    return kept


#: The longest answer asked for when checks are written, in tokens.
CHECKS_TOKENS = 2048


def rules_of(feature: Feature, sources: list[Source]) -> list[str]:
    """A feature's rules: what it says it does, and what is written about it, each a thing a person can see."""
    said = [f"{feature.how} -> {feature.shows}".strip(" ->")]
    written = what_is_written_of(feature, sources)
    said += [s for s in re.split(r"(?<=[.!?])\s+", written) if s.strip()]
    return [r for r in said if r][:3]


def _checks_rule(check: Check, rule: str, number: int) -> bool:
    """Whether ``check`` says it checks ``rule`` (by its number or most of its words)."""
    said = check.rule.lower()
    if re.search(rf"\brule\s*{number}\b", said) or said.strip() == str(number):
        return True
    words = {w for w in re.findall(r"[a-z]{3,}", rule.lower()) if w not in _PLAIN}
    return bool(words) and len(words & set(re.findall(r"[a-z]{3,}", said))) * 2 >= len(words)


_PLAIN = frozenset("the and for with from that this into when then what which their them they your have will are can its".split())


def what_is_written_of(feature: Feature, sources: list[Source], *, at_most: int = 2) -> str:
    """The sentences of what is written that are about ``feature``: those sharing most of its own words."""
    words = {w for w in re.findall(r"[a-z]{3,}", f"{feature.name} {feature.how} {feature.shows}".lower()) if w not in _PLAIN}
    if not words:
        return ""
    named = {w for w in re.findall(r"[a-z]{3,}", feature.name.lower()) if w not in _PLAIN}
    scored = []
    for source in sources:
        for sentence in re.split(r"(?<=[.!?])\s+", source.text):
            said = {w for w in re.findall(r"[a-z]{3,}", sentence.lower())}
            # About it: one of the words of its name, and more of its words besides.
            if named & said and len(words & said) >= 2 and 30 <= len(sentence) <= 400:
                scored.append((len(words & said) + 2 * len(named & said), sentence.strip()))
    scored.sort(key=lambda pair: -pair[0])
    return " ".join(sentence for _n, sentence in scored[:at_most])


def said_as_json(thing: BaseModel) -> str:
    return json.dumps(thing.model_dump(), indent=1)[:6000]
