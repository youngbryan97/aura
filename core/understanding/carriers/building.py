"""A program asked for: made, put where it can be used, started, and every rule of it told by whether its test holds.

The engine is core/making. Here it is reached for a request in anybody's words, what it made is put where a person keeps
such a thing, and it is started there to see that it runs.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import subprocess
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from core.runtime.wall_clock import monotonic_time
from core.understanding.carrying_out import CameTo, Tell
from core.understanding.her_model_between_turns import ask as _ask
from core.understanding.her_model_between_turns import write as _write
from core.understanding.requests_of_her import Part, Request
from core.understanding.workbench import Work, get_workbench, this_work

logger = logging.getLogger("Aura.Understanding.Building")

__all__ = ["build", "change", "runs_here"]

#: How long an app may take to have a process once it is opened.
STARTS_WITHIN_S = 20.0


def _room_to_think() -> dict[str, Any]:
    """How her model is given room to think where she is asked what she thinks of what she made: in the mode that opens her
    thinking channel, with as much room as her thinking has been measured to take (core/brain/llm/thinking_reserve.py)."""
    from core.brain.llm.thinking_reserve import reserve_tokens
    from core.runtime.service_access import resolve_inference_gate

    client = getattr(resolve_inference_gate(default=None), "_mlx_client", None)
    room = int(reserve_tokens(str(getattr(client, "model_path", "") or "")))
    return {"cognitive_mode": "reflective", **({"private_channel_budget": room} if room else {})}


async def _her_eyes(prompt: str, png: str) -> str:
    """What her eyes see in a screen of what she made."""
    from core.making.her_view import her_eyes

    return await her_eyes(prompt, png)


def _builds() -> Path:
    from core.skills.changing_a_program import where_builds_are

    return where_builds_are()[-1]


def runs_here(app: Path, *, within_s: float = STARTS_WITHIN_S) -> bool:
    """Whether an application, opened, has a process of its own: it did not refuse to start or fall over at once."""
    import psutil

    from core.governance_context import local_internal_governed_scope
    from core.runtime.subprocess_gateway import get_subprocess_gateway

    with local_internal_governed_scope("building.open", domain="tool_execution"):
        get_subprocess_gateway().run(["open", "-g", str(app)], timeout=30, source="building.open",
                                     accelerator_capability="none")
    inside = str(app / "Contents" / "MacOS")
    until = monotonic_time() + within_s
    while monotonic_time() < until:
        if any(str(p.info.get("exe") or "").startswith(inside) for p in psutil.process_iter(["exe"])):
            return True
        time.sleep(0.5)
    return False


def _places_named(words: str) -> list[Path]:
    """The folders of the person's own that their words name, by the names the folders have."""
    said = set(re.findall(r"[\w-]+", words.casefold()))
    return sorted(folder for folder in Path.home().iterdir()
                  if folder.is_dir() and not folder.name.startswith(".") and folder.name.casefold() in said
                  and folder.name.casefold() not in {"applications", "library"})


def _linked_in(folder: Path, thing: Path, name: str) -> Path | None:
    """A link to what was made in a folder, under its name; nothing of anybody else's is replaced."""
    from core.governance_context import local_internal_governed_scope
    from core.runtime.file_write_gateway import get_file_write_gateway

    link = folder / (thing.name if thing.suffix == ".app" else f"{name}{thing.suffix}")
    if link.exists() and not link.is_symlink():
        return None
    with local_internal_governed_scope("building.link", domain="file_write"):
        get_file_write_gateway().replace_symlink(link, thing, source="building.link")
    return link


async def _put_where_it_belongs(made: Any, request: Request) -> tuple[list[Part], dict[str, Any]]:
    """The program where it can be used: an application installed and started, a site opened, a tool given its command;
    linked in every folder of theirs they named. What is so of each."""
    parts: list[Part] = []
    places: dict[str, Any] = {}
    if made.design.kind in {"game", "app"}:
        from core.rebuilding.as_a_mac_app import as_a_mac_app

        try:
            from core.making.pictures import icon_at

            thing = await asyncio.to_thread(as_a_mac_app, made.path, made.design.name, accent=made.design.look.accent,
                                            picture=await asyncio.to_thread(icon_at, made.folder))
        except (OSError, subprocess.SubprocessError, RuntimeError, ValueError) as why:
            return [Part("it is installed as an application", "unmet", f"{type(why).__name__}: {why}"[:300])], places
        places["app"] = str(thing)
        parts.append(Part(f"it is installed, at {thing}", "met" if await asyncio.to_thread(thing.is_dir) else "unmet", "it is there"))
        ran = await asyncio.to_thread(runs_here, thing)
        parts.append(Part("it starts", "met" if ran else "unmet",
                          "opened, and its process is running" if ran else "opened, and no process of it appeared"))
    elif made.design.kind == "tool":
        thing = await asyncio.to_thread(_its_command, made)
        places["command"] = str(thing)
        parts.append(Part(f"it can be run, as {thing}", "met" if await asyncio.to_thread(thing.is_file) else "unmet", "its command is there"))
    else:
        thing = made.path
        parts.append(Part(f"it is on disk, at {thing}", "met" if await asyncio.to_thread(thing.is_file) else "unmet", "it is there"))
    for folder in await asyncio.to_thread(_places_named, request.whole):
        link = await asyncio.to_thread(_linked_in, folder, thing, made.design.name)
        if link is not None:
            places[f"in {folder.name}"] = str(link)
        parts.append(Part(f"it is in {folder.name}", "met" if link is not None and await asyncio.to_thread(link.exists) else "unmet",
                          str(link) if link is not None else "something else of that name is there"))
    work = this_work()
    if work is not None:
        work.made(made.design.name, str(thing))
    return parts, places


def _its_command(made: Any) -> Path:
    """A tool as one command, beside what it runs."""
    from core.governance_context import local_internal_governed_scope
    from core.making.tools import languages
    from core.rebuilding.her_build_writes import governed
    from core.runtime.subprocess_gateway import get_subprocess_gateway

    name = re.sub(r"[^a-z0-9]+", "-", made.design.name.lower()).strip("-") or "tool"
    language = languages().get(made.design.language.lower()) or languages()["python"]
    out = made.folder / "program"
    if language.build:
        with local_internal_governed_scope("building.tool", domain="tool_execution"):
            get_subprocess_gateway().run([part.format(file=str(made.path), out=str(out)) for part in language.build], timeout=180,
                                         check=True, source="building.tool", accelerator_capability="none")
    run = " ".join(f'"{part.format(file=str(made.path), out=str(out))}"' for part in language.run)
    command = made.folder / name
    governed("write_text", command, f"#!/bin/sh\nexec {run} \"$@\"\n")
    with local_internal_governed_scope("building.tool", domain="tool_execution"):
        get_subprocess_gateway().run(["chmod", "755", str(command)], timeout=10, source="building.tool", accelerator_capability="none")
    return command


async def build(request: Request, tell: Tell) -> CameTo:
    """Make what the request describes, or go on with what is missing of it."""
    from core.making import maker
    from core.making.pictures import by_her

    page = Path(str(request.made.get("page") or ""))
    folder = page.parent if request.made.get("page") and await asyncio.to_thread(page.is_file) else None
    async with _a_browser() as browser:
        if folder is None:
            made = await maker.make(request.whole, ask=_ask, write=_write, work=_this(request), where=_builds(), browser=browser,
                                    told=request.told, when_kept=_keeps(request), paints=by_her, eyes=_her_eyes, thinking=_room_to_think())
        elif request.added and not await asyncio.to_thread(_taken_in, folder, request.added[-1]):
            made = await maker.change(folder, "\n".join(request.added), ask=_ask, write=_write, work=_this(request), browser=browser, paints=by_her, eyes=_her_eyes, thinking=_room_to_think())
        else:
            made = await maker.mend(folder, ask=_ask, write=_write, work=_this(request), browser=browser, paints=by_her, eyes=_her_eyes, thinking=_room_to_think())
    if made is None:
        return CameTo(how="read what was asked as a design", came_to="no design of it could be had",
                      parts=[Part("the program exists", "unmet", "no design of it came that held everything asked")])
    return await _as_it_came_out(request, made)


async def change(request: Request, tell: Tell) -> CameTo:
    """Change a program she made to what was asked, everything that held still held to."""
    from core.making import maker
    from core.making.pictures import by_her
    from core.rebuilding.rebuilding_a_program import the_build_meant
    from core.skills.changing_a_program import where_builds_are

    page = Path(str(request.made.get("page") or ""))
    folder = page.parent if request.made.get("page") and await asyncio.to_thread(page.is_file) else await asyncio.to_thread(
        the_build_meant, request.whole, where_builds_are())
    if folder is None or not await asyncio.to_thread((folder / "program.json").is_file):
        return CameTo(how="looked for the program meant", again=False, came_to="no program of hers was found to change",
                      parts=[Part("the program meant was found", "unmet", "she has built none that matches")])
    if not await asyncio.to_thread(_made_here, folder):
        return await _changed_as_rebuilt(request, folder, tell)
    wanted = (request.added or [request.said])[-1]
    taken_in = await asyncio.to_thread(_taken_in, folder, wanted)
    async with _a_browser() as browser:
        if taken_in:
            made = await maker.mend(folder, ask=_ask, write=_write, work=_this(request), browser=browser, paints=by_her, eyes=_her_eyes, thinking=_room_to_think())
        else:
            made = await maker.change(folder, wanted, ask=_ask, write=_write, work=_this(request), browser=browser, paints=by_her, eyes=_her_eyes, thinking=_room_to_think())
    if made is None:
        return CameTo(how="read the change into the design", came_to="it could not be read into the design",
                      parts=[Part("the change asked for", "unmet", "no design with it came")])
    return await _as_it_came_out(request, made)


def _taken_in(folder: Path, words: str) -> bool:
    """Whether a program's design has these words in it already: it is gone on with then, and not changed to them again. LIVE
    25 November 2026 a program being changed was designed again, and its tests written again, at every try, for an hour."""
    try:
        asked = json.loads((folder / "program.json").read_text(encoding="utf-8")).get("asked")
    except (OSError, ValueError):
        return False
    return " ".join(str(words).split()) in " ".join(str(asked or "").split())


def _keeps(request: Request) -> Any:
    """Where the program stands, on the request as soon as there is one to go on with: what cuts the work short leaves it."""
    from core.understanding.requests_of_her import get_requests_of_her
    from core.utils.task_tracker import get_task_tracker

    def kept(page: Path) -> None:
        if request.made.get("page") != str(page):
            request.made["page"] = str(page)
            get_task_tracker().create_task(get_requests_of_her().keep(), name=f"request.kept.{request.id}")

    return kept


def _this(request: Request) -> Work:
    return this_work() or get_workbench().begin(request.id, request.said, request.effect)


@asynccontextmanager
async def _a_browser() -> AsyncIterator[Any]:
    from playwright.async_api import async_playwright

    async with async_playwright() as started:
        browser = await started.chromium.launch(headless=True)
        try:
            yield browser
        finally:
            await browser.close()


def _made_here(folder: Path) -> bool:
    """Whether a program was made by core/making, and not by the engine that rebuilds programs."""
    return "design" in json.loads((folder / "program.json").read_text("utf-8"))


async def _as_it_came_out(request: Request, made: Any) -> CameTo:
    """What is so of a program that was made: each rule by its tests, how it runs, and where it is."""
    from core.making.trying import by_rule

    generic = made.trial.generic
    broken = [r for r in generic if not r.ok]
    parts = [Part("it runs, and does what it does when anything is pressed", "unmet" if broken else "met",
                  "; ".join(f"{r.name}: {r.why}" for r in broken)[:400] if broken else f"{len(generic)} ways of trying it hold")]
    parts += [Part(r.name, "met" if r.ok else "unmet", ("its test holds" + (", as it was read again and corrected" if r.name in made.corrected else ""))
                   if r.ok else r.why[:300]) for r in by_rule(made.trial.rules)]
    where: dict[str, Any] = {"page": str(made.path), "heard": "\n".join(request.added)}
    if made.working():
        placed, places = await _put_where_it_belongs(made, request)
        parts += placed
        where.update(places)
    holds = sum(1 for part in parts if part.stands == "met")
    return CameTo(how="designed, tested, written, tried and mended", parts=parts, made=where,
                  came_to=f"{made.design.name}: {made.design.about} {holds} of {len(parts)} things asked or expected of it are so, "
                          f"in {made.seconds / 60:.0f} minutes.")


async def _changed_as_rebuilt(request: Request, folder: Path, tell: Tell) -> CameTo:
    """A program of the engine that rebuilds programs, changed by it."""
    from core.rebuilding.changing_what_was_built import change_it

    changed = await change_it(folder, request.whole, _ask, tell=tell)
    if not changed.outcomes:
        return CameTo(how="read the change as features", came_to=changed.why_not,
                      parts=[Part("the change asked for", "unmet", changed.why_not or "it was not made")])
    parts = [Part(o.feature.name, "met" if o.kept else "unmet", "seen working" if o.kept else o.why_not[:300]) for o in changed.outcomes]
    return CameTo(how="changed, with every check that held before still holding", parts=parts,
                  made={"page": str(folder / "index.html")}, came_to=changed.summary())
