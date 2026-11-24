"""A program asked for: built, installed, opened, and used, and every feature told by whether it was seen working.

The engine is core/rebuilding, which says what the program does as features, writes the checks a person would make
before any code, and keeps each part only for what doing the checks shows. Here it is reached for a request in anybody's
words, its result is put where a person keeps programs, and it is started there to see that it runs.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import subprocess
import time
from pathlib import Path
from typing import Any

from core.understanding.carrying_out import CameTo, Tell
from core.understanding.requests_of_her import Part, Request

logger = logging.getLogger("Aura.Understanding.Building")

__all__ = ["build", "change", "runs_here"]

#: How long an app may take to have a process once it is opened.
STARTS_WITHIN_S = 20.0


def _builds() -> Path:
    from core.skills.changing_a_program import where_builds_are

    return where_builds_are()[-1]


async def _ask(prompt: str, schema: Any, max_tokens: int) -> Any:
    from core.rebuilding.her_model import ask_her_model

    return await ask_her_model(prompt, schema, max_tokens, origin="work_asked_for")


def _as_kept(page: Path) -> tuple[str, str]:
    """The program's name and its colour, as its build kept them."""
    kept = page.parent / "program.json"
    made = json.loads(kept.read_text("utf-8")) if kept.exists() else {}
    return str(made.get("title") or page.parent.name.replace("-", " ").title()), str(made.get("accent") or "")


def runs_here(app: Path, *, within_s: float = STARTS_WITHIN_S) -> bool:
    """Whether an application, opened, has a process of its own: it did not refuse to start or fall over at once."""
    import psutil

    from core.governance_context import local_internal_governed_scope
    from core.runtime.subprocess_gateway import get_subprocess_gateway

    with local_internal_governed_scope("building.open", domain="tool_execution"):
        get_subprocess_gateway().run(["open", "-g", str(app)], timeout=30, source="building.open",
                                     accelerator_capability="none")
    inside = str(app / "Contents" / "MacOS")
    until = time.monotonic() + within_s
    while time.monotonic() < until:
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


def _linked_in(folder: Path, app: Path) -> Path | None:
    """A link to the application in a folder, under its own name; nothing of anybody else's is replaced."""
    from core.governance_context import local_internal_governed_scope
    from core.runtime.file_write_gateway import get_file_write_gateway

    link = folder / app.name
    if link.exists() and not link.is_symlink():
        return None
    with local_internal_governed_scope("building.link", domain="file_write"):
        get_file_write_gateway().replace_symlink(link, app, source="building.link")
    return link


async def _put_where_it_belongs(page: Path, request: Request) -> tuple[list[Part], dict[str, Any]]:
    """The program as an application, installed, linked where it was asked to be, and started; what is so of each."""
    from core.rebuilding.as_a_mac_app import as_a_mac_app

    parts: list[Part] = []
    made: dict[str, Any] = {"page": str(page)}
    name, accent = await asyncio.to_thread(_as_kept, page)
    try:
        app = await asyncio.to_thread(as_a_mac_app, page, name, accent=accent)
    except (OSError, subprocess.SubprocessError, RuntimeError, ValueError) as why:
        return [Part("it is installed as an application", "unmet", f"{type(why).__name__}: {why}"[:300])], made
    made["app"] = str(app)
    parts.append(Part(f"it is installed, at {app}", "met" if app.is_dir() else "unmet", "it is there"))
    for folder in await asyncio.to_thread(_places_named, request.whole):
        link = await asyncio.to_thread(_linked_in, folder, app)
        if link is not None:
            made[f"in {folder.name}"] = str(link)
        parts.append(Part(f"it is in {folder.name}", "met" if link is not None and link.exists() else "unmet",
                          str(link) if link is not None else "something else of that name is there"))
    ran = await asyncio.to_thread(runs_here, app)
    parts.append(Part("it starts", "met" if ran else "unmet",
                      "opened, and its process is running" if ran else "opened, and no process of it appeared"))
    return parts, made


async def build(request: Request, tell: Tell) -> CameTo:
    """Build what the request describes, or go on with what is missing of it."""
    page = Path(str(request.made.get("page") or ""))
    if request.made.get("page") and page.is_file():
        return await _what_is_missing(request, page, tell)
    from core.rebuilding.rebuilding_a_program import rebuild

    rebuilt = await rebuild("", _ask, _builds(), tell=tell, asked=request.whole)
    if rebuilt.built is None:
        return CameTo(how="built from the words of the request", came_to=rebuilt.why_not or "nothing was built",
                      parts=[Part("the program exists", "unmet", rebuilt.why_not or "nothing was built")])
    return await _as_it_came_out(request, rebuilt.built.path, rebuilt.built.outcomes, rebuilt.summary(),
                                 "built feature by feature, each kept for what using it showed")


async def _as_it_came_out(request: Request, page: Path, outcomes: list[Any], summary: str, how: str) -> CameTo:
    parts = [Part(o.feature.name, "met" if o.kept else "unmet",
                  f"{o.held} of {o.of} checks hold, by doing it" if o.kept else (o.why_not or "no version of it passed its checks")[:300])
             for o in outcomes]
    made: dict[str, Any] = {"page": str(page)}
    if any(part.stands == "met" for part in parts) and page.suffix.lower() in (".html", ".htm"):
        placed, where = await _put_where_it_belongs(page, request)
        parts += placed
        made.update(where)
    elif page.suffix.lower() not in (".html", ".htm"):
        parts.append(Part(f"it is on disk, at {page}", "met" if page.is_file() else "unmet"))
    return CameTo(how=how, parts=parts, made=made, came_to=summary)


async def _what_is_missing(request: Request, page: Path, tell: Tell) -> CameTo:
    """The features that did not work, tried again on the program as it stands; what works is kept as it is."""
    from core.rebuilding.changing_what_was_built import change_it

    missing = [part for part in request.missing() if not part.what.startswith("it ")]
    if not missing and not request.added:
        placed, made = await _put_where_it_belongs(page, request)
        kept = [p for p in request.parts if p.stands == "met" and not p.what.startswith("it ")]
        return CameTo(how="put in its place again", parts=kept + placed, made=made, again=False)
    wanted = [f"{part.what}: this does not work yet. {part.shown_by}" for part in missing] + list(request.added)
    changed = await change_it(page.parent, "Make these work:\n" + "\n".join(wanted), _ask, tell=tell)
    done = {o.feature.name.lower(): o for o in changed.outcomes}
    parts: list[Part] = []
    for part in request.parts:
        if part.what.startswith("it "):
            continue
        outcome = done.pop(part.what.lower(), None)
        if outcome is None:
            parts.append(part)
        else:
            parts.append(Part(part.what, "met" if outcome.kept else "unmet", outcome.why_not[:300] if not outcome.kept else "seen working now"))
    parts += [Part(o.feature.name, "met" if o.kept else "unmet", o.why_not[:300] if not o.kept else "seen working now") for o in done.values()]
    placed, made = await _put_where_it_belongs(page, request)
    return CameTo(how="worked on what was missing, the rest kept as it was", parts=parts + placed, made=made,
                  came_to=changed.summary())


async def change(request: Request, tell: Tell) -> CameTo:
    """Change a program she made to what was asked, everything that worked kept working."""
    from core.rebuilding.changing_what_was_built import change_it
    from core.rebuilding.rebuilding_a_program import the_build_meant
    from core.skills.changing_a_program import where_builds_are

    page = Path(str(request.made.get("page") or ""))
    folder = page.parent if request.made.get("page") and page.is_file() else await asyncio.to_thread(
        the_build_meant, request.whole, where_builds_are())
    if folder is None:
        return CameTo(how="looked for the program meant", again=False, came_to="no program of hers was found to change",
                      parts=[Part("the program meant was found", "unmet", "she has built none that matches")])
    changed = await change_it(folder, request.whole, _ask, tell=tell)
    if not changed.outcomes:
        return CameTo(how="read the change as features", came_to=changed.why_not,
                      parts=[Part("the change asked for", "unmet", changed.why_not or "it was not made")])
    return await _as_it_came_out(request, folder / "index.html", changed.outcomes, changed.summary(),
                                 "changed, with every check that held before still holding")
