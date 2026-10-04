"""Mending a program by what it does: suspicions from its code, verdicts from running it.

Asked to fix a program she did not write, she works the way a careful person
does. She runs it first and says what is wrong with it, in terms of what it
does: the up key moves the paddle down, the ball goes through her paddle, a
miss scores for the one who missed. She reads the code for shapes that are
almost never meant (core/self_modification/code_that_looks_wrong.py). And then
she tries the edits those shapes suggest, one at a time, each in a fresh copy
watched through the same game (core/self_modification/watching_a_program_run.py),
and keeps an edit only when something that was wrong stops being wrong and
nothing new goes wrong. After each kept edit she reads the code again and runs
it again, because a program that works a little better shows faults the
broken one hid: no wall can be tested while every ball goes straight through
the paddle.

What the program should do comes from its own words, and from what she
already knows about the thing it is a version of. That knowledge is looked up
in her own reference corpus and shown beside each finding, as evidence. Nothing
here asks a model what the fix is.
"""
from __future__ import annotations

import asyncio
import logging
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.self_modification.code_that_looks_wrong import Edit, Suspicion, applied, what_looks_wrong

logger = logging.getLogger("SelfModification.RepairingByBehaviour")

__all__ = ["Repair", "repair_by_behaviour", "the_programs_own_words"]

#: How many copies are watched at once while trying edits.
AT_ONCE = 3

#: How long each part of a watch runs, in seconds.
WATCH_S = 10.0


@dataclass
class Repair:
    """What was wrong, what was changed, and what it does now."""

    path: str
    before: dict[str, str] = field(default_factory=dict)
    after: dict[str, str] = field(default_factory=dict)
    kept: list[dict[str, Any]] = field(default_factory=list)
    left: list[str] = field(default_factory=list)
    knowledge: list[str] = field(default_factory=list)
    said: list[str] = field(default_factory=list)
    seconds: float = 0.0
    written_to: str = ""
    backup: str = ""
    #: Checks never seen either way, so nothing is claimed about them.
    unseen: list[str] = field(default_factory=list)


def the_programs_own_words(source: str) -> str:
    """The sentences a program shows its user: its string literals that read as words, and its title."""
    words = []
    title = re.search(r"<title>(.*?)</title>", source, re.IGNORECASE | re.DOTALL)
    if title:
        words.append(" ".join(title.group(1).split()))
    for literal in re.findall(r"\"([^\"\n]{3,200})\"|'([^'\n]{3,200})'", source):
        text = literal[0] or literal[1]
        if " " in text and re.search(r"[A-Za-z]{3}", text) and not re.search(r"[{};=<>]|^\w+\s*\(", text):
            words.append(text)
    return " ".join(words)


def _title(source: str) -> str:
    found = re.search(r"<title>(.*?)</title>", source, re.IGNORECASE | re.DOTALL)
    return " ".join(found.group(1).split()) if found else ""


def what_she_knows_about(title: str, findings: dict[str, str]) -> list[str]:
    """Sentences from her reference corpus about the thing the program is a version of, nearest the findings."""
    if not title:
        return []
    try:
        from core.knowledge.local_corpus import get_local_corpus_store

        store = get_local_corpus_store()
        hit = store.by_title(title)
        body = store.body(hit.doc_id, max_chars=8000) if hit is not None else ""
    except (ImportError, OSError, RuntimeError, ValueError, AttributeError) as exc:
        logger.info("her reference corpus could not be read for %r: %s", title, exc)
        return []
    sentences = re.split(r"(?<=[.!?])\s+", " ".join(body.split()))
    wanted = set(re.findall(r"[a-z]{4,}", " ".join(findings.values()).lower())) | {"paddle", "ball", "point", "points", "player", "return", "score", "opponent", "control", "controls"}
    scored = sorted(
        ((len(wanted & set(re.findall(r"[a-z]{4,}", sentence.lower()))), sentence) for sentence in sentences[:80]),
        key=lambda pair: -pair[0],
    )
    return [sentence for score, sentence in scored[:3] if score >= 2]


async def _watched(browser: Any, html: str, words: str, keys: list[str], folder: Path, seconds: float = WATCH_S) -> Any:
    """One copy, written beside the original so its relative paths still resolve, and watched."""
    from core.runtime.file_write_gateway import get_file_write_gateway
    from core.self_modification.watching_a_program_run import what_it_does

    copy = folder / f".trying-{abs(hash(html)) % 10**10}.html"
    await get_file_write_gateway().write_text_async(copy, html, source="repairing_by_behaviour")
    page = await browser.new_page(viewport={"width": 900, "height": 700})
    try:
        return await what_it_does(page, copy.resolve().as_uri(), words=words, keys=keys, seconds=seconds)
    finally:
        await page.close()
        get_file_write_gateway().delete_file(copy, source="repairing_by_behaviour")


@dataclass
class _Believed:
    """What she believes about each check across the repair, not only in the last watch.

    A check stays wrong until it is seen right: one watch that happened not to
    measure it does not mend it. And one that was seen right stays right until
    it is seen wrong.
    """

    wrong: set[str] = field(default_factory=set)
    right: set[str] = field(default_factory=set)
    findings: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_watch(cls, behaviour: Any) -> _Believed:
        return cls(set(behaviour.wrong), set(behaviour.right), dict(behaviour.findings))

    def after(self, behaviour: Any) -> _Believed:
        wrong = (self.wrong - behaviour.right) | behaviour.wrong
        right = (self.right - behaviour.wrong) | behaviour.right
        findings = {**{k: v for k, v in self.findings.items() if k in wrong}, **behaviour.findings}
        return _Believed(wrong, right, findings)


def _improves(behaviour: Any, now: _Believed) -> bool:
    """Something believed wrong now measures right, or something never measured does, and nothing right went wrong.

    A finding that disappears only counts when its check now says right:
    unmeasured is never fine. Offline 2026-10-03, an edit that left the paddle
    stuck at the top made every check go quiet, and was taken for a complete
    repair. And a fault the edit reveals is not harm the edit did: with the
    controls mended the ball reached the paddle for the first time and went
    through it, a fault the broken controls had hidden.
    """
    mended = now.wrong & behaviour.right
    newly_right = behaviour.right - now.right - now.wrong
    harmed = behaviour.wrong & now.right
    return not harmed and bool(mended or newly_right)


def _worth(behaviour: Any, now: _Believed) -> tuple[int, int, int]:
    """How much better an improving trial is: faults mended, checks newly right, faults seen."""
    return (len(now.wrong & behaviour.right), len(behaviour.right - now.right), -len(behaviour.wrong))


def _candidates(suspicions: list[Suspicion]) -> list[tuple[Suspicion, list[Edit]]]:
    """Each edit alone, and the edits of one pattern in one function together.

    Some faults are one fault written twice: the up key and the down key both
    backwards. Mended one at a time, each half makes the program no better, so
    a search over single edits never finds the repair.
    """
    singles = [(suspicion, [edit]) for suspicion in suspicions for edit in suspicion.edits]
    together: dict[tuple[str, str], list[Suspicion]] = {}
    for suspicion in suspicions:
        if len(suspicion.edits) == 1:
            together.setdefault((suspicion.pattern, suspicion.function), []).append(suspicion)
    pairs = [
        (group[0], [s.edits[0] for s in group])
        for group in together.values()
        if len(group) > 1
    ]
    return pairs + singles


#: The most pairs of edits tried together in one round.
MOST_PAIRS = 10


def _pairs(candidates: list[tuple[Suspicion, list[Edit]]]) -> list[tuple[Suspicion, list[Edit]]]:
    """Candidates from two different places in the code, applied together."""
    pairs = []
    for index, (first, first_edits) in enumerate(candidates):
        for second, second_edits in candidates[index + 1 :]:
            if second.line == first.line or any(a.start < b.end and b.start < a.end for a in first_edits for b in second_edits):
                continue
            together = Suspicion(
                f"{first.pattern}, with {second.pattern}", first.line,
                f"{first.why} (line {first.line}); and {second.why} (line {second.line})",
                first_edits + second_edits,
                " and ".join(dict.fromkeys(f for f in (first.function, second.function) if f)),
            )
            pairs.append((together, first_edits + second_edits))
            if len(pairs) >= MOST_PAIRS:
                return pairs
    return pairs


def _name_of(edits: list[Edit]) -> str:
    return "|".join(f"{e.start}:{e.end}:{e.text}" for e in edits)


async def _confirmed(browser: Any, before: str, after: str, words: str, keys: list[str], folder: Path,
                     now: _Believed, trial: Any) -> tuple[Any, list[str]] | None:
    """The chosen edit watched again beside the program without it, for longer.

    Kept only if no watch shows it doing harm, and only for what it is seen
    to do: what was right with it in the trial and in a side-by-side watch,
    and not right without it in that same pair. One watch that happened to
    see a wall bounce is not the edit's doing: LIVE 2026-10-04 the score line
    of the wrong branch was kept on the strength of a wall seen in its trial,
    and the score was left counting backwards. And one pair in which the
    thing the edit mends never happened (no ball got past her) says nothing
    either way, so up to CONFIRM_PAIRS pairs are watched before it is refused.
    """
    for _pair in range(CONFIRM_PAIRS):
        without, again = await asyncio.gather(
            _watched(browser, before, words, keys, folder, WATCH_S * 2.5),
            _watched(browser, after, words, keys, folder, WATCH_S * 2.5),
        )
        harm = again.wrong & (now.right - trial.wrong)
        if harm:
            # One watch is one game: a ball that clips a corner can read as
            # harm once. A third watch settles it, and harm seen twice in
            # three is harm.
            logger.info("a watch of the chosen edit found harm (%s); watching again", sorted(harm))
            third = await _watched(browser, after, words, keys, folder, WATCH_S * 2.5)
            if third.wrong & harm:
                logger.info("found again: %s", sorted(third.wrong & harm))
                return None
            again = third
        shown = sorted((trial.right & again.right) - without.right)
        if shown:
            return again, shown
        logger.info("beside the program without it, the chosen edit made no difference this time")
    return None


#: Side-by-side watches of a chosen edit before it is refused for making no
#: difference that can be seen.
CONFIRM_PAIRS = 3


async def _try_edits(browser: Any, current: str, words: str, keys: list[str], folder: Path,
                     now: _Believed, seconds: float = WATCH_S, refused: set[str] | None = None,
                     in_pairs: bool = False) -> tuple[Suspicion, list[Edit], Any] | None:
    """Every edit the code suggests, watched in parallel; the one that mends the most and breaks nothing."""
    candidates = [c for c in _candidates(what_looks_wrong(current, ".html")) if _name_of(c[1]) not in (refused or set())]
    if in_pairs:
        candidates = _pairs(candidates)
    best = None
    for start in range(0, len(candidates), AT_ONCE):
        batch = candidates[start : start + AT_ONCE]
        watched = await asyncio.gather(
            *(_watched(browser, applied(current, edits), words, keys, folder, seconds) for _s, edits in batch),
            return_exceptions=True,
        )
        for (suspicion, edit), behaviour in zip(batch, watched, strict=True):
            if isinstance(behaviour, BaseException):
                logger.info("a trial copy could not be watched: %s", behaviour)
                continue
            logger.info(
                "tried %s at line %s (%d edit(s)): wrong %s, right %s",
                suspicion.pattern, suspicion.line, len(edit), sorted(behaviour.wrong), sorted(behaviour.right),
            )
            if _improves(behaviour, now) and (best is None or _worth(behaviour, now) > _worth(best[2], now)):
                best = (suspicion, edit, behaviour)
    return best


async def repair_by_behaviour(path: Path, *, say: Callable[[str], Any] | None = None) -> Repair:
    """Mend the program at ``path`` in place, keeping a copy of how it was."""
    from playwright.async_api import async_playwright

    from core.agency.playing_as_it_happens import controls_named_in
    from core.runtime.file_write_gateway import get_file_write_gateway

    began = time.monotonic()
    source = await asyncio.to_thread(path.read_text)
    words = the_programs_own_words(source)
    keys = [key for key in controls_named_in(words)[0] if key in ("up", "down", "left", "right")]
    repair = Repair(path=str(path))

    def tell(line: str) -> None:
        repair.said.append(line)
        logger.info("repairing: %s", line)
        if say is not None:
            say(line)

    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        try:
            first = await _watched(browser, source, words, keys, path.parent)
            repair.before = dict(first.findings)
            tell(_what_is_wrong(first))
            repair.knowledge = what_she_knows_about(_title(source), first.findings)
            if repair.knowledge:
                tell(f"What I know about {_title(source)}: {repair.knowledge[0]}")
            suspicions = what_looks_wrong(source, ".html")
            tell(_what_looks_wrong(suspicions))
            current, last = source, _Believed.from_watch(first)
            refused: set[str] = set()
            # On while anything helps, not only while something is known to be
            # wrong: a fault that has not happened in a watch yet is still a
            # fault, and an edit that makes a check come right where it had
            # never been seen right is the evidence for it (offline
            # 2026-10-04, the top wall was never seen missing, the loop
            # stopped when nothing was known wrong, and it was left missing).
            for _round in range(MOST_ROUNDS):
                chosen = await _try_edits(browser, current, words, keys, path.parent, last, refused=refused)
                if chosen is None:
                    # Nothing settled it in a short watch. A fault that shows
                    # only now and then needs a longer one before an edit can
                    # be said to have mended it.
                    if last.wrong:
                        tell("Nothing I tried settled it in a short watch, so I am watching each try for longer.")
                    chosen = await _try_edits(browser, current, words, keys, path.parent, last,
                                              seconds=WATCH_S * 2.5, refused=refused)
                if chosen is None and last.wrong:
                    # Two faults can each hide what mending the other would
                    # show: a ball that goes through the paddle never tests the
                    # scoring. Mended together, both come right at once.
                    tell("No one change settles it alone, so I am trying them two at a time.")
                    chosen = await _try_edits(browser, current, words, keys, path.parent, last,
                                              seconds=WATCH_S * 2.5, refused=refused, in_pairs=True)
                if chosen is None:
                    break
                suspicion, edit, behaviour = chosen
                confirmed = await _confirmed(browser, current, applied(current, edit), words, keys, path.parent, last, behaviour)
                if confirmed is None:
                    refused.add(_name_of(edit))
                    continue
                again, shown = confirmed
                repair.kept.append({
                    "where": suspicion.function, "line": suspicion.line, "pattern": suspicion.pattern,
                    "change": ", ".join(e.says(current) for e in edit), "why": suspicion.why,
                    "shown": [_RIGHT_SAID.get(name, name) for name in shown],
                })
                tell(_what_this_change_did(suspicion, edit, current, shown))
                current, last = applied(current, edit), last.after(behaviour).after(again)
            final = await _watched(browser, current, words, keys, path.parent)
            if final.wrong:
                # A fault said to remain is said to the person: seen twice, or not said.
                second = await _watched(browser, current, words, keys, path.parent, WATCH_S * 2.5)
                for name in final.wrong - second.wrong:
                    final.findings.pop(name, None)
                final.right |= second.right - final.wrong
            believed = last.after(final)
            repair.after = dict(believed.findings) if believed.wrong else {}
            repair.unseen = sorted(set(_RIGHT_SAID) - believed.right - believed.wrong)
        finally:
            await browser.close()
    repair.left = [f"{s.function}: {s.why}" for s in what_looks_wrong(current, ".html")]
    if current != source:
        gateway = get_file_write_gateway()
        backup = path.with_name(path.name + ".before-repair")
        await gateway.write_text_async(backup, source, source="repairing_by_behaviour")
        await gateway.write_text_async(path, current, source="repairing_by_behaviour")
        repair.written_to, repair.backup = str(path), str(backup)
    tell(_how_it_ends(repair))
    repair.seconds = round(time.monotonic() - began, 1)
    return repair


def _what_is_wrong(behaviour: Any) -> str:
    if not behaviour.findings:
        return "I ran it and played it, and nothing it did was wrong that I could see."
    return "I ran it and played it. " + "; ".join(f"{finding[0].upper()}{finding[1:]}" for finding in behaviour.findings.values()) + "."


def _what_looks_wrong(suspicions: list[Suspicion]) -> str:
    if not suspicions:
        return "Nothing in the code has a shape I would suspect."
    places = "; ".join(f"line {s.line} in {s.function or 'the script'}: {s.why}" for s in suspicions[:6])
    return f"Reading the code, {len(suspicions)} place(s) look wrong: {places}."


def _what_this_change_did(suspicion: Suspicion, edit: list[Edit], current: str, shown: list[str]) -> str:
    # The line each edit is on, which is not always the line the suspicion
    # names: of two mirrored branches, the edit may be to either.
    lines = sorted({current.count("\n", 0, e.start) + 1 for e in edit})
    where = ", ".join(str(n) for n in lines) or str(suspicion.line)
    said = (f"In {suspicion.function or 'the code'} (line {where}): {suspicion.why}, "
            f"so I changed {', '.join(e.says(current) for e in edit)}.")
    if shown:
        said += " Now " + " and ".join(f"I can see {_RIGHT_SAID.get(name, name)}" for name in shown) + "."
    else:
        said += " It plays no worse for it, and what it should mend did not come up again in the second look."
    return said


def _and_now(mended: list[str], before: Any, after: Any) -> str:
    gone = [f"it is no longer true that {before.findings.get(name, name)}" for name in mended]
    shown = [f"I can see {_RIGHT_SAID.get(name, name)}" for name in sorted(after.right - before.right) if name not in mended]
    return " and ".join(gone + shown) or "it behaves better"


#: The most rounds of trying edits in one repair.
MOST_ROUNDS = 8

#: What a check never seen either way is about, in words.
_UNSEEN_SAID = {
    "controls": "what the keys do",
    "went through": "the ball meet every part of my paddle",
    "escaped": "the ball reach both the top and bottom walls",
    "credited": "a ball get past me and the score change",
    "idle": "the other player's paddle",
}

#: What a check coming right shows, in words.
_RIGHT_SAID = {
    "controls": "each key move my paddle the way it says",
    "went through": "the ball turn back off my paddle",
    "escaped": "the ball bounce off the walls",
    "credited": "a miss count for the other side",
    "idle": "the other paddle moving",
}


def _how_it_ends(repair: Repair) -> str:
    if not repair.kept:
        return "I could not find a change that made it behave better, so I left the file as it was."
    line = f"I made {len(repair.kept)} change(s) and saved the file; the original is beside it as {Path(repair.backup).name}."
    if repair.after:
        line += " Still wrong: " + "; ".join(repair.after.values()) + "."
    elif repair.unseen:
        line += (" Nothing I saw it do was wrong, but I never saw "
                 + " or ".join(_UNSEEN_SAID.get(name, name) for name in repair.unseen)
                 + ", so I cannot vouch for that.")
    else:
        line += " Run again and played, everything I check came right."
    if repair.left:
        line += f" {len(repair.left)} place(s) still look odd in the code but made no difference I could see when changed: " + "; ".join(repair.left[:3]) + "."
    return line
