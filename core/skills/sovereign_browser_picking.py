"""Items picked from a page's list by a rule the person stated, each then done as the person asked.

"Play three of the games, one after another. To pick them: number the games
on the list from 0, take the current minute..." is a list, a rule, and a task
for each item. Her pursuit used to read all of it as one goal and leave the
picking to a decision each round: LIVE 2026-10-05 she opened the first game on
the list whatever the minute was. Here the list is read off the page (its
repeated links, in reading order), the rule is worked out by code
(core/language/picking_by_a_rule.py) and said with its working, and each item
picked is pursued in turn, for the task said of each one.

Nothing here knows what the list is of.
"""
from __future__ import annotations

import logging
import re
from collections.abc import Mapping
from typing import Any

from core.language.picking_by_a_rule import a_picking_rule

logger = logging.getLogger("Skills.SovereignBrowser.Picking")

__all__ = ["picked_by_the_rule", "pursued", "the_list_on_the_page", "the_task_for_each"]

#: The page's list: its largest set of links that look alike (same tags and classes up to the content), in reading order.
_THE_LIST = r"""
() => {
  const main = document.querySelector("main, [role=main], #content, #main, .content") || document.body;
  const shown = (el) => { const r = el.getBoundingClientRect(); const s = getComputedStyle(el); return r.width > 0 && r.height > 0 && s.visibility !== "hidden"; };
  const outside = (el) => !!el.closest("header, nav, footer, [role=navigation], #wm-ipp-base, #wm-ipp, #donato");
  const shape = (el) => { const path = []; for (let e = el; e && e !== main; e = e.parentElement) {
      const cls = typeof e.className === "string" ? e.className.trim().split(/\s+/).filter((c) => !/^(active|current|selected|hover|is-|js-)/.test(c)).sort().join(".") : "";
      path.push(e.tagName + (cls ? "." + cls : "")); } return path.join("<"); };
  const groups = new Map();
  for (const a of main.querySelectorAll("a[href]")) {
    if (!shown(a) || outside(a) || !(a.innerText || "").trim() || /^(javascript:|#|mailto:)/i.test(a.getAttribute("href") || "")) continue;
    const k = shape(a);
    if (!groups.has(k)) groups.set(k, []);
    groups.get(k).push(a);
  }
  let best = [];
  for (const g of groups.values()) {
    const distinct = [...new Map(g.map((a) => [a.href, a])).values()];
    if (distinct.length > best.length) best = distinct;
  }
  return best.map((a) => {
    const named = a.querySelector("h1, h2, h3, h4, h5, h6, [class*=title], strong, b");
    const text = ((named ? named.innerText : a.innerText) || "").split("\n").map((s) => s.trim()).filter(Boolean)[0] || "";
    return { text, href: a.href };
  });
}
"""


async def the_list_on_the_page(browser: Any) -> list[dict[str, str]]:
    """The items of the list the page shows, each its name and where it leads, as they are listed."""
    page = getattr(browser, "page", None)
    if page is None:
        return []
    try:
        items = await page.evaluate(_THE_LIST)
    except Exception as why:  # noqa: BLE001 - a page that cannot be read has no list to pick from
        logger.info("the page's list could not be read: %s", why)
        return []
    return [i for i in items or [] if isinstance(i, dict) and i.get("href") and i.get("text")]


_COUNTED = re.compile(r"\b(?:all\s+)?(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|some|a\s+few)\s+(?:of\s+)?(?:the\s+)?([a-z]+s)\b", re.I)


def _one(plural: str) -> str:
    if re.search(r"ies$", plural):
        return plural[:-3] + "y"
    if re.search(r"(s|x|ch|sh)es$", plural):
        return plural[:-2]
    return plural[:-1]


def the_task_for_each(words: str) -> str:
    """What the person asked to be done with each item, said of one: "play three of the games ... and win each one" is "Play this game and win it"."""
    text = " ".join(str(words or "").split())
    rule_at = re.search(r"\b(?:to\s+pick\s+them|to\s+choose\s+them|number\s+the|take\s+the\s+(?:current|last)|pick\s+them\s+by)\b", text, re.I)
    asked = text[: rule_at.start()] if rule_at else text
    asked = re.split(r"(?<=[.!?])\s+", asked.strip())[0]
    asked = re.sub(r"^\s*(?:please\s+)?go\s+to\s+\S+\s*(?:and|,|then)?\s*", "", asked, flags=re.I)
    asked = re.sub(r",?\s*one\s+after\s+(?:another|the\s+other),?", "", asked, flags=re.I)
    asked = _COUNTED.sub(lambda m: f"this {_one(m.group(1))}", asked, count=1)
    asked = re.sub(r"\b(?:each\s+one|each\s+of\s+them|every\s+one|all\s+of\s+them|them\s+all|them)\b", "it", asked, flags=re.I)
    asked = re.sub(r"\s+,", ",", asked).strip(" .,")
    return (asked[:1].upper() + asked[1:] + ".") if asked else "Do with it what was asked."


#: Tasks done with the thing itself, run: for these a copy that runs comes before a copy of a page about it.
_RUN_IT = re.compile(r"\b(play|run|launch|watch|listen|hear)\b", re.I)


async def _the_item_itself(skill: Any, browser: Any, url: str, name: str, task: str = "") -> str:
    """The item opened in ``browser``: where the list points; else, where the task is to run it, a copy the archive runs;
    else the archive's copy of that page; else whatever the archive keeps by its name."""
    from core.skills import sovereign_browser_going as going

    if await skill._safe_browse(browser, url):
        return url
    runs = bool(_RUN_IT.search(task))
    tries = [lambda: going.the_same_thing_elsewhere(browser, name, runnable=True)] if runs else []
    tries += [lambda: going.the_archived_copy(skill, browser, url), lambda: going.the_same_thing_elsewhere(browser, name)]
    for n, attempt in enumerate(tries):
        found = await attempt()
        if not found:
            continue
        if found.startswith("https://archive.org/details/"):
            if not await skill._safe_browse(browser, found):
                continue
            skill._say_out_loud(f"“{name}” is not to be had where the list points; the Internet Archive keeps it"
                                + (", and runs it in its page, so I play it there." if runs and n == 0 else ", so I go there."),
                                {"label": "Going to", "said": found})
        return found
    return ""


def _ordinal(n: int) -> str:
    return {1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth"}.get(n, f"number {n}")


async def pursued(skill: Any, browser: Any, url: str | None, goal: str, max_steps: int, *, action_context: Mapping[str, Any] | None = None,
                  said_before: str = "") -> dict[str, Any]:
    """The goal pursued from ``url``: each item its rule picks from the page's list in turn, where it states one; else as one pursuit."""
    if url and a_picking_rule(goal) is not None:
        picked = await picked_by_the_rule(skill, browser, url, goal, max_steps, action_context=action_context, said_before=said_before)
        if picked is not None:
            return picked
    return await skill._handle_pursue(browser, url, goal, max_steps, action_context=action_context, said_before=said_before)


async def picked_by_the_rule(skill: Any, browser: Any, url: str, goal: str, max_steps: int, *, action_context: Mapping[str, Any] | None = None,
                             said_before: str = "") -> dict[str, Any] | None:
    """Each item the goal's rule picks from the list at ``url``, pursued in turn; None where the goal states no rule or the page no list."""
    from core.skills.sovereign_browser_going import the_archived_copy

    rule = a_picking_rule(goal)
    if rule is None:
        return None
    if not await skill._safe_browse(browser, url) and not await the_archived_copy(skill, browser, url):
        return None  # the pursuit says why the page could not be had
    items = await the_list_on_the_page(browser)
    if len(items) < 2:
        skill._say_out_loud("I could not find a list on this page to pick from, so I am going on from what the page shows.")
        return None
    picks = rule.worked_out([i["text"] for i in items])
    noun = (re.search(r"\bnumber\w*\s+the\s+([a-z]+)", goal, re.I) or re.search(r"\bthe\s+([a-z]+s)\s+on\s+the\s+list\b", goal, re.I))
    called = noun.group(1) if noun else "items"
    skill._say_out_loud(f"There are {len(items)} {called} on the list. By your rule: " + ". Then ".join(
        f"{p.working}: number {p.index + rule.counted_from}, “{p.item}”" for p in picks) + ".")
    task = the_task_for_each(goal)
    results: list[dict[str, Any]] = []
    for n, pick in enumerate(picks, start=1):
        item = items[pick.index]
        skill._say_out_loud(f"The {_ordinal(n)} of {len(picks)}: “{pick.item}”.")
        logger.info("picked by the rule: %s -> %s (%s)", pick.working, pick.item, item["href"])
        here = await _the_item_itself(skill, browser, item["href"], pick.item, task)
        if not here:
            skill._say_out_loud(f"“{pick.item}” cannot be had where the list points, nor anywhere I can find it kept, so I go on to the next.")
            results.append({"number": pick.number, "item": pick.item, "url": item["href"], "ok": False, "completed": False,
                            "concluded": "it could not be had anywhere"})
            continue
        done = await skill._handle_pursue(browser, None, f"{task} (It is “{pick.item}”, the {_ordinal(n)} of the {len(picks)} picked.)",
                                          max_steps, action_context=action_context, said_before=said_before)
        done = done if isinstance(done, dict) else {}
        results.append({"number": pick.number, "item": pick.item, "url": item["href"], "ok": bool(done.get("ok")),
                        "completed": bool(done.get("completed")), "concluded": str(done.get("concluded") or "")})
        if n < len(picks):
            skill._say_out_loud(f"That was “{pick.item}”. On to the next one the rule picks.")
    last = done if results else {}
    return {
        **last,
        "ok": any(r["ok"] for r in results),
        "completed": all(r["completed"] for r in results),
        "goal": goal,
        "picked": results,
        "concluded": " ".join(f"{_ordinal(n).capitalize()}, “{r['item']}”: {r['concluded'] or ('done' if r['ok'] else 'not done')}"
                              for n, r in enumerate(results, start=1)),
    }
