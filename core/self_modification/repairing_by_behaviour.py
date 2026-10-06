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
in her own reference corpus and shown as background. Model-suggested edits
remain hypotheses until execution checks them.
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
AT_ONCE = 4

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
    #: Behaviour checks actually seen to hold, after the retained edits.
    checked: list[str] = field(default_factory=list)
    code_checks: list[dict[str, Any]] = field(default_factory=list)
    required_edges: list[str] = field(default_factory=list)


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


def what_she_knows_about(title: str, findings: dict[str, str], own_words: str = "") -> list[str]:
    """Sentences from her reference corpus about the thing the program is a version of.

    Nearest what was seen wrong and what the program says of itself (its
    title and the words it shows), and nothing else: which words matter is
    read from this program, not from a list kept for one kind of program.
    """
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
    wanted = set(re.findall(r"[a-z]{4,}", " ".join([*findings.values(), own_words]).lower()))
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
        behaviour = await what_it_does(page, copy.resolve().as_uri(), words=words, keys=keys, seconds=seconds)
        evidence = getattr(behaviour, "evidence", {}) or {}
        logger.info(
            "watched %.0fs: %s pictures (%s a second), hers %r, wrong %s, right %s",
            getattr(behaviour, "seconds", 0) or 0, evidence.get("pictures"), evidence.get("pictures_a_second"),
            evidence.get("hers"), sorted(getattr(behaviour, "wrong", ()) or ()), sorted(getattr(behaviour, "right", ()) or ()),
        )
        return behaviour
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
    #: How many watches have seen each check right, and wrong.
    seen_right: dict[str, int] = field(default_factory=dict)
    seen_wrong: dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_watch(cls, behaviour: Any) -> _Believed:
        return cls().after(behaviour)

    def after(self, behaviour: Any) -> _Believed:
        wrong = (self.wrong - behaviour.right) | behaviour.wrong
        right = (self.right - behaviour.wrong) | behaviour.right
        findings = {**{k: v for k, v in self.findings.items() if k in wrong}, **behaviour.findings}
        seen_right = dict(self.seen_right)
        seen_wrong = dict(self.seen_wrong)
        for name in behaviour.right:
            seen_right[name] = seen_right.get(name, 0) + 1
        for name in behaviour.wrong:
            seen_wrong[name] = seen_wrong.get(name, 0) + 1
        return _Believed(wrong, right, findings, seen_right, seen_wrong)

    @property
    def surely_right(self) -> set[str]:
        """Checks seen right in two watches or more and never wrong: what an edit can be said to break.

        One reading is not enough to veto a repair: offline 2026-10-04 a first
        watch read the broken score as right, and when the mended controls let
        the real fault show, every right edit "broke" the score and was turned
        down.
        """
        return {name for name in self.right if self.seen_right.get(name, 0) >= 2 and not self.seen_wrong.get(name, 0)}


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
    harmed = behaviour.wrong & now.surely_right
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
        harm = again.wrong & (now.surely_right - trial.wrong)
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
        # Or a fault the program without it showed in this same pair, under
        # the same dice, that the program with it did not, here or in its
        # trial. A missing wall is seen missing when the ball leaves through
        # it; seen present only when the ball happens to strike both walls,
        # which a short watch often does not show (LIVE 2026-10-04).
        gone = sorted((without.wrong & now.wrong) - again.wrong - trial.wrong)
        if shown or gone:
            return again, shown + [f"no longer: {name}" for name in gone]
        logger.info("beside the program without it, the chosen edit made no difference this time")
    return None


#: Side-by-side watches of a chosen edit before it is refused for making no
#: difference that can be seen.
CONFIRM_PAIRS = 3


def _alongside(best: tuple[Suspicion, list[Edit], Any], improving: list[Any], now: _Believed) -> list[Any]:
    """The other trials of the same round that each mend something the best does not, in other places of the code.

    One fix a round meant five rounds for five faults, a minute and a half
    each, with the person waiting. Trials that mend different things, by
    edits in different places, are faults of their own: watched together,
    they are kept together.
    """
    def gains(behaviour: Any) -> set[str]:
        return (now.wrong & behaviour.right) | (behaviour.right - now.right - now.wrong)

    def where(edits: list[Edit]) -> list[tuple[int, int]]:
        return [(e.start, e.end) for e in edits]

    taken, covered, places = [], set(gains(best[2])), where(best[1])
    seen = {(best[0].pattern, best[0].line)}
    for suspicion, edit, behaviour in sorted(improving, key=lambda t: _worth(t[2], now), reverse=True):
        mine = gains(behaviour)
        if (suspicion.pattern, suspicion.line) in seen or not mine - covered:
            continue
        if any(a < d and c < b for a, b in where(edit) for c, d in places):
            continue  # edits in the same place are two ways of doing one fix
        taken.append((suspicion, edit, behaviour))
        covered |= mine
        places += where(edit)
        seen.add((suspicion.pattern, suspicion.line))
    return taken


async def _try_edits(browser: Any, current: str, words: str, keys: list[str], folder: Path,
                     now: _Believed, seconds: float = WATCH_S, refused: set[str] | None = None,
                     in_pairs: bool = False, proposed: list[Suspicion] | None = None,
                     tell: Callable[[str], None] | None = None, also: list[Any] | None = None) -> tuple[Suspicion, list[Edit], Any] | None:
    """Every edit the code suggests, watched in parallel; the one that mends the most and breaks nothing.

    ``proposed`` replaces the code's own suggestions with edits from elsewhere
    (edits_her_model_proposes.py), tried the same way. Where she is in it is
    said as she goes, so a person watching knows what the minutes are for.
    """
    suggested = proposed if proposed is not None else what_looks_wrong(current, ".html")
    candidates = [c for c in _candidates(suggested) if _name_of(c[1]) not in (refused or set())]
    if in_pairs:
        candidates = _pairs(candidates)
    if tell is not None and candidates:
        rounds = -(-len(candidates) // AT_ONCE)
        many = len(candidates) != 1
        what = ("pairs of fixes" if many else "pair of fixes") if in_pairs else ("fixes" if many else "fix")
        took = rounds * seconds
        tell(f"Trying {len(candidates)} possible {what} on copies of the game{f', {AT_ONCE} at a time' if len(candidates) > AT_ONCE else ''}, "
             f"watching each copy play for about {seconds:.0f} seconds"
             f"{f' (about {took / 60:.0f} minutes)' if took >= 90 else ''}.")
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
            if _improves(behaviour, now) and also is not None:
                also.append((suspicion, edit, behaviour))
            if _improves(behaviour, now) and (best is None or _worth(behaviour, now) > _worth(best[2], now)):
                best = (suspicion, edit, behaviour)
        if tell is not None and start + AT_ONCE < len(candidates):
            tell(f"{start + len(batch)} of {len(candidates)} tried.")
    if tell is not None and candidates:
        tell(f"The one that helps most: line {best[0].line}, {best[0].why}." if best is not None
             else "None of those made it play better on its own.")
    return best


async def _kept_together(browser: Any, current: str, words: str, keys: list[str], folder: Path, last: _Believed,
                         fixes: list[Any], tell: Callable[[str], None]) -> dict[str, Any] | None:
    """Fixes that each mend something different, confirmed together as one is: kept, each said; or None, and the best is tried alone."""
    from types import SimpleNamespace

    tell(f"{len(fixes)} of those fixes each mend something different, in different places. Watching them together, "
         "beside the game without them, to be sure they help and break nothing.")
    edits = [e for _s, edit, _b in fixes for e in edit]
    trial = SimpleNamespace(right=set().union(*(b.right for _s, _e, b in fixes)),
                            wrong=set.intersection(*(set(b.wrong) for _s, _e, b in fixes)))
    after = applied(current, edits)
    confirmed = await _confirmed(browser, current, after, words, keys, folder, last, trial)
    if confirmed is None:
        tell("Together they did not hold up, so I am checking the best of them alone.")
        return None
    again, shown = confirmed

    def mended_by(fault: str) -> list[Any]:
        # A fault gone when the fixes are watched together is said of the fix
        # whose own trial mended it, not of every fix in the group: LIVE
        # 2026-10-06 the scoring fix was said to have mended the keys.
        return ([b for _s, _e, b in fixes if fault in b.right] or [b for _s, _e, b in fixes if fault not in b.wrong]
                or [b for _s, _e, b in fixes])

    kept = []
    for suspicion, edit, behaviour in fixes:
        mine = [name for name in shown
                if (any(b is behaviour for b in mended_by(name[11:])) if name.startswith("no longer: ") else name in behaviour.right)]
        kept.append(({
            "where": suspicion.function, "line": suspicion.line, "pattern": suspicion.pattern,
            "change": ", ".join(e.says(current) for e in edit), "why": suspicion.why,
            "shown": [f"no longer {_WRONG_SAID.get(n[11:], n[11:])}" if n.startswith("no longer: ") else _RIGHT_SAID.get(n, n) for n in mine],
        }, _what_this_change_did(suspicion, edit, current, mine)))
    believed = last
    for _s, _e, behaviour in fixes:
        believed = believed.after(behaviour)
    return {"kept": kept, "current": after, "last": believed.after(again)}


async def _kept_from_reading(browser: Any, current: str, words: str, keys: list[str], folder: Path, last: _Believed,
                             refused: set[str], repair: Repair, tell: Callable[[str], None]) -> tuple[str, _Believed]:
    """Fixes the code is plainly wrong without, kept when watching could not show them either way and they harm nothing.

    A wall missing from the top of a court is seen only when the ball goes
    there, and in some watches it never does: LIVE 2026-10-06 the top wall's
    fix was tried four times, never made a difference that could be seen, and
    was left out of a repair that had read it right. A person who reads code
    wrong by its shape mends it, and then checks the change breaks nothing.
    One edit for each place still suspected, each watched twice beside the
    game without it; kept if neither watch shows harm, and said to be from
    reading, not from seeing.
    """
    left = [c for c in _candidates(what_looks_wrong(current, ".html")) if _name_of(c[1]) not in refused and len(c[1]) == 1]
    done: set[tuple[str, int]] = set()
    for suspicion, edit in left:
        if (suspicion.pattern, suspicion.line) in done:
            continue
        tell(f"Line {suspicion.line} is wrong as it is written ({suspicion.why}), though the game never showed it while I watched. "
             "Changing it, and checking the change breaks nothing.")
        after = applied(current, edit)
        harmed = False
        for _pair in range(2):
            without, again = await asyncio.gather(
                _watched(browser, current, words, keys, folder, WATCH_S * 2.5),
                _watched(browser, after, words, keys, folder, WATCH_S * 2.5),
            )
            if again.wrong - without.wrong:
                harmed = True
                break
            last = last.after(again)
        if harmed:
            tell(f"Changed, the game did something wrong it did not do before, so line {suspicion.line} stays as it was.")
            refused.add(_name_of(edit))
            continue
        repair.kept.append({
            "where": suspicion.function, "line": suspicion.line, "pattern": suspicion.pattern,
            "change": ", ".join(e.says(current) for e in edit), "why": suspicion.why, "shown": ["from reading the code"],
        })
        tell(f"In {suspicion.function or 'the code'} (line {suspicion.line}): {suspicion.why}, so I "
             f"{', '.join(e.says(current) for e in edit)}. Watched twice, it breaks nothing.")
        current = after
        done.add((suspicion.pattern, suspicion.line))
    return current, last


async def _what_else_could_do_it(browser: Any, current: str, words: str, keys: list[str], folder: Path,
                                 last: _Believed, refused: set[str], tell: Callable[[str], None]) -> Any:
    """Past the shapes she knows: ask her own model what else in the code could do what is seen, and try each.

    Thinking it through before trying anything, privately; what comes back
    is a list of guesses, tried on copies and kept only for what they are
    seen to mend.
    """
    from core.self_modification.edits_her_model_proposes import edits_her_model_proposes

    proposed = await edits_her_model_proposes(current, dict(last.findings))
    if not proposed:
        return None
    tell(f"None of the shapes I know in code settles it, so I thought about what else could do this, "
         f"and I am trying {len(proposed)} idea(s) on copies.")
    return await _try_edits(browser, current, words, keys, folder, last, seconds=WATCH_S * 2.5,
                            refused=refused, proposed=proposed, tell=tell)


async def repair_by_behaviour(path: Path, *, say: Callable[[str], Any] | None = None) -> Repair:
    """Mend the program at ``path`` in place, keeping a copy of how it was."""
    from playwright.async_api import async_playwright

    from core.agency.playing_as_it_happens import controls_named_in
    from core.runtime.file_write_gateway import get_file_write_gateway
    from core.self_modification.checking_code_paths import boundary_checks, check_code_paths

    began = time.monotonic()
    # Copies an earlier repair was trying when it was cut short (a restart) are hers to clear.
    for stale in await asyncio.to_thread(lambda: list(path.parent.glob(".trying-*.html"))):
        get_file_write_gateway().delete_file(stale, source="repairing_by_behaviour")
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
            if not first.findings and not first.right:
                # A look that measured nothing, right or wrong, has not seen
                # the program do anything yet; it is not a finding that it does
                # nothing wrong (LIVE 2026-10-04, a first look on a busy machine).
                first = await _watched(browser, source, words, keys, path.parent, WATCH_S * 2.5)
            repair.before = dict(first.findings)
            tell(_what_is_wrong(first))
            repair.knowledge = what_she_knows_about(_title(source), first.findings, words)
            # Reference context remains in the receipt. Work narration reports
            # experiments and their results, rather than arbitrary article text.
            suspicions = what_looks_wrong(source, ".html")
            tell(_what_looks_wrong(suspicions))
            current, last = source, _Believed.from_watch(first)
            contracts = boundary_checks(source, path.suffix)
            repair.required_edges = sorted({edge for c in contracts for edge in
                                            (("top", "bottom") if c.axis == "y" else ("left", "right"))})
            if contracts:
                initial_paths = await check_code_paths(browser, current, suffix=path.suffix, checks=contracts)
                repair.code_checks.append({"stage": "before", "results": initial_paths})
                for suspicion in suspicions:
                    if suspicion.pattern != "one-sided boundary":
                        continue
                    after = applied(current, suspicion.edits)
                    tested = await check_code_paths(browser, after, suffix=path.suffix, checks=contracts)
                    right_before = {(r["check"]["position"], r["case"]) for r in initial_paths if r["verdict"] == "right"}
                    now_right = { (r["check"]["position"], r["case"]) for r in tested if r["verdict"] == "right" }
                    if not right_before <= now_right or len(now_right) <= len(right_before):
                        continue
                    tell(f"Line {suspicion.line}: testing each boundary directly on a copy confirmed {suspicion.why}. "
                         "The changed function keeps the object inside, turns it back, and changes no counters.")
                    repair.kept.append({"where": suspicion.function, "line": suspicion.line,
                                        "pattern": suspicion.pattern, "change": ", ".join(e.says(current) for e in suspicion.edits),
                                        "why": suspicion.why, "shown": ["directed function checks: both boundaries and interior"]})
                    repair.code_checks.append({"stage": "candidate", "results": tested})
                    current = after
                    last = last.after(await _watched(browser, current, words, keys, path.parent))
                    from types import SimpleNamespace

                    last = last.after(SimpleNamespace(right={"escaped"}, wrong=set(), findings={}))
                    initial_paths = tested
            refused: set[str] = set()
            asked = 0
            # On while anything helps, not only while something is known to be
            # wrong: a fault that has not happened in a watch yet is still a
            # fault, and an edit that makes a check come right where it had
            # never been seen right is the evidence for it (offline
            # 2026-10-04, the top wall was never seen missing, the loop
            # stopped when nothing was known wrong, and it was left missing).
            for _round in range(MOST_ROUNDS):
                improving: list[Any] = []
                chosen = await _try_edits(browser, current, words, keys, path.parent, last, refused=refused, tell=tell, also=improving)
                left_to_try = [c for c in _candidates(what_looks_wrong(current, ".html")) if _name_of(c[1]) not in refused]
                if chosen is None and left_to_try:
                    # Nothing settled it in a short watch. A fault that shows
                    # only now and then needs a longer one before an edit can
                    # be said to have mended it.
                    if last.wrong:
                        tell("Nothing I tried settled it in a short watch, so I am watching each try for longer.")
                    chosen = await _try_edits(browser, current, words, keys, path.parent, last,
                                              seconds=WATCH_S * 2.5, refused=refused, tell=tell)
                if chosen is None and last.wrong and len(left_to_try) > 1:
                    # Two faults can each hide what mending the other would
                    # show: a ball that goes through the paddle never tests the
                    # scoring. Mended together, both come right at once.
                    tell("No one change settles it alone, so I am trying them two at a time.")
                    chosen = await _try_edits(browser, current, words, keys, path.parent, last,
                                              seconds=WATCH_S * 2.5, refused=refused, in_pairs=True, tell=tell)
                if chosen is None and last.wrong and asked < MOST_ASKS:
                    # What is still believed wrong may only be unseen since: a
                    # look at the game as it now is comes before thinking up
                    # changes the code's own shapes did not suggest.
                    looked = await _watched(browser, current, words, keys, path.parent, WATCH_S * 2.5)
                    last = last.after(looked)
                    if not last.wrong:
                        break
                    asked += 1
                    chosen = await _what_else_could_do_it(browser, current, words, keys, path.parent, last, refused, tell)
                if chosen is None:
                    break
                together = _alongside(chosen, improving, last)
                if together:
                    kept_together = await _kept_together(browser, current, words, keys, path.parent, last, [chosen, *together], tell)
                    if kept_together is not None:
                        for entry, said in kept_together["kept"]:
                            repair.kept.append(entry)
                            tell(said)
                        current, last = kept_together["current"], kept_together["last"]
                        continue
                suspicion, edit, behaviour = chosen
                tell("Watching the game with that fix again, beside the game without it, to be sure it helps and breaks nothing.")
                confirmed = await _confirmed(browser, current, applied(current, edit), words, keys, path.parent, last, behaviour)
                if confirmed is None:
                    tell("Watched again, that fix did not hold up, so I left it out and went on.")
                    refused.add(_name_of(edit))
                    continue
                again, shown = confirmed
                repair.kept.append({
                    "where": suspicion.function, "line": suspicion.line, "pattern": suspicion.pattern,
                    "change": ", ".join(e.says(current) for e in edit), "why": suspicion.why,
                    "shown": [
                        f"no longer {_WRONG_SAID.get(name[11:], name[11:])}" if name.startswith("no longer: ")
                        else _RIGHT_SAID.get(name, name) for name in shown
                    ],
                })
                tell(_what_this_change_did(suspicion, edit, current, shown))
                current, last = applied(current, edit), last.after(behaviour).after(again)
            current, last = await _kept_from_reading(browser, current, words, keys, path.parent, last, refused, repair, tell)
            tell(f"{len(repair.kept)} fix(es) kept. Watching the mended game as a whole once more before I write it back.")
            final = await _watched(browser, current, words, keys, path.parent)
            if final.wrong:
                # A fault said to remain is said to the person: seen twice, or not said.
                second = await _watched(browser, current, words, keys, path.parent, WATCH_S * 2.5)
                for name in final.wrong - second.wrong:
                    final.findings.pop(name, None)
                final.right |= second.right - final.wrong
            believed = last.after(final)
            repair.after = dict(believed.findings) if believed.wrong else {}
            repair.checked = sorted(believed.right)
            repair.unseen = sorted((final.checks or set(_RIGHT_SAID)) - believed.right - believed.wrong)
            if contracts:
                tested = await check_code_paths(browser, current, suffix=path.suffix, checks=contracts)
                repair.code_checks.append({"stage": "final", "results": tested})
                if all(r["verdict"] == "right" for r in tested):
                    if "escaped" not in repair.checked:
                        repair.checked.append("escaped")
                    repair.unseen = [name for name in repair.unseen if name != "escaped"]
                    repair.after.pop("escaped", None)
                elif any(r["verdict"] == "wrong" for r in tested):
                    repair.after["escaped"] = "a directed boundary check still fails"
                    repair.checked = [name for name in repair.checked if name != "escaped"]
        finally:
            await browser.close()
    repair.left = [f"{s.function}: {s.why}" for s in what_looks_wrong(current, ".html")]
    if current != source:
        from core.self_modification.saving_a_verified_repair import save_repair

        repair.backup = await save_repair(path, source, current)
        repair.written_to = str(path)
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
            f"so I {', '.join(e.says(current) for e in edit)}.")
    seen = [f"I can see {_RIGHT_SAID.get(name, name)}" for name in shown if not name.startswith("no longer: ")]
    gone = [f"I no longer see {_WRONG_SAID.get(name[11:], name[11:])}" for name in shown if name.startswith("no longer: ")]
    if seen or gone:
        said += " Now " + " and ".join(seen + gone) + "."
    else:
        said += " It plays no worse for it, and what it should mend did not come up again in the second look."
    return said


def _and_now(mended: list[str], before: Any, after: Any) -> str:
    gone = [f"it is no longer true that {before.findings.get(name, name)}" for name in mended]
    shown = [f"I can see {_RIGHT_SAID.get(name, name)}" for name in sorted(after.right - before.right) if name not in mended]
    return " and ".join(gone + shown) or "it behaves better"


#: The most rounds of trying edits in one repair.
MOST_ROUNDS = 8

#: How many times in one repair her model is asked what else could be wrong.
MOST_ASKS = 2

#: What a check never seen either way is about, in words.
_UNSEEN_SAID = {
    "controls": "what the keys do",
    "went through": "something meet every part of what I control",
    "escaped": "something reach both the top and bottom edges",
    "credited": "something get past me and the score change",
    "idle": "the other side's player",
    "errors": "it run without an error",
    "dead": "what each control does",
    "failed": "everything it asks for load",
}

#: What a check seen wrong shows, in words.
_WRONG_SAID = {
    "controls": "a key move what I control the wrong way",
    "went through": "things go straight through what I control",
    "escaped": "things leave through an edge and the game stand still",
    "credited": "my side's score go up when something gets past me",
    "idle": "the other side's player stand still",
    "errors": "it throw an error",
    "dead": "a control that does nothing",
    "failed": "something it asks for fail to load",
}

#: What a check coming right shows, in words.
_RIGHT_SAID = {
    "controls": "each key move what I control the way it says",
    "went through": "things turn back off what I control",
    "escaped": "things turn back at the top and bottom edges",
    "credited": "a miss count for the other side",
    "idle": "the other side's player moving",
    "errors": "it run without an error",
    "dead": "every control do something when used",
    "failed": "everything it asks for load",
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
