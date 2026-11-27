"""Her work, seen as it goes, and a word in while it does: the workbench's routes.

What is shown is what the work keeps (core/understanding/workbench.py): steps, what she says of them, code as she writes
it, what trying it showed, and what she waits on. What a person says to a work that is going reaches it at once; said to
one that has ended, it takes the request up again with those words.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from core.understanding.requests_of_her import get_requests_of_her
from core.understanding.workbench import Work, get_workbench
from interface.auth import _require_internal

logger = logging.getLogger("Aura.Server.Work")
router = APIRouter(tags=["work"])

__all__ = ["router"]

#: How many works are listed.
LISTED = 20
#: How often a stream looks for what has happened, and how long it says nothing where nothing has, in seconds.
LOOKS_S, QUIET_S = 0.06, 2.0
#: What a made program may do in the panel it is shown running in: run, and nothing of hers.
_APART = {"Content-Security-Policy": "sandbox allow-scripts", "Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"}


class Said(BaseModel):
    words: str = Field(min_length=1, max_length=8000)


def _listed(work: Work) -> dict[str, Any]:
    seen = work.seen(after=10**9)
    seen.pop("happened")
    seen["now"] = {k: v for k, v in seen["now"].items() if k != "writing"}
    folder = _folder(work.id)
    seen["icon"] = folder is not None and (folder / "icon.png").is_file()
    request = get_requests_of_her().get(work.id)
    # What she made for herself, with nobody asking for it.
    seen["hers"] = request is not None and request.conversation == "herself"
    return seen


def _folder(work_id: str) -> Path | None:
    """The folder of what a work is making, where there is one."""
    request = get_requests_of_her().get(work_id)
    page = str((request.made.get("page") or "") if request is not None else "")
    return Path(page).parent if page and Path(page).is_file() else None


@router.get("/work")
async def works(_: None = Depends(_require_internal)) -> JSONResponse:
    """The works, the newest first."""
    works = list(reversed(get_workbench().all()[-LISTED:]))
    return JSONResponse({"works": await asyncio.to_thread(lambda: [_listed(work) for work in works])})


@router.get("/work/{work_id}")
async def work_seen(work_id: str, after: int = 0, _: None = Depends(_require_internal)) -> JSONResponse:
    """One work, and what has happened in it after the last that was seen."""
    return JSONResponse(_the(work_id).seen(after))


@router.get("/work/{work_id}/stream")
async def stream(work_id: str, after: int = 0, _: None = Depends(_require_internal)) -> StreamingResponse:
    """A work as it goes: what has happened in it as it happens, and what she is writing as she writes it, a little at a time."""
    return StreamingResponse(_as_it_goes(_the(work_id), after), media_type="text/event-stream",
                             headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})


async def _as_it_goes(work: Work, after: int) -> AsyncIterator[str]:
    sent, stood, said = "", "", 0.0
    while True:
        seen = work.seen(after)
        now = seen["now"]
        writing = now.pop("writing", None)
        now["is_writing"] = writing is not None
        if writing is not None:
            at = len(sent) if writing.startswith(sent) else _alike(sent, writing)
            if len(writing) > at or at < len(sent):
                now["more"] = [at, writing[at:]]
        sent = writing or ""
        stands = json.dumps([work.state, now.get("what"), now.get("waits"), now.get("reading"), writing is not None])
        if seen["happened"] or "more" in now or stands != stood or time.monotonic() - said > QUIET_S:
            after = seen["happened"][-1]["n"] if seen["happened"] else after
            stood, said = stands, time.monotonic()
            yield f"event: seen\ndata: {json.dumps(seen, ensure_ascii=False)}\n\n"
        await asyncio.sleep(LOOKS_S if work.state == "going" else QUIET_S / 2)


def _alike(a: str, b: str) -> int:
    """How far two texts are the same."""
    same, differ = 0, min(len(a), len(b)) + 1
    while differ - same > 1:
        at = (same + differ) // 2
        if a[:at] == b[:at]:
            same = at
        else:
            differ = at
    return same


@router.get("/work/{work_id}/preview")
async def preview(work_id: str, _: None = Depends(_require_internal)) -> HTMLResponse:
    """What a work is making, as it stands, to be shown running: apart from everything of hers."""
    folder = await asyncio.to_thread(_folder, work_id)
    page = folder / "index.html" if folder is not None else None
    if page is None or not await asyncio.to_thread(page.is_file):
        raise HTTPException(status_code=404, detail="nothing to show yet")
    return HTMLResponse(await asyncio.to_thread(page.read_text, "utf-8"), headers=_APART)


@router.get("/work/{work_id}/program")
async def program(work_id: str, _: None = Depends(_require_internal)) -> JSONResponse:
    """What a work is making, as it is written: each of its functions, and its tests."""
    folder = await asyncio.to_thread(_folder, work_id)
    if folder is None:
        return JSONResponse({"parts": {}})

    def read() -> dict[str, Any]:
        made = json.loads((folder / "program.json").read_text(encoding="utf-8"))
        parts, design = made.get("parts") or {}, made.get("design") or {}
        named = dict(parts.get("functions") or ({} if not parts.get("text") else {"program": parts["text"]}))
        if parts.get("tests"):
            named["tests"] = parts["tests"]
        return {"parts": named, "language": "python" if design.get("kind") == "tool" else "html" if design.get("kind") == "site" else "javascript"}

    try:
        return JSONResponse(await asyncio.to_thread(read))
    except (OSError, ValueError):
        return JSONResponse({"parts": {}})


@router.get("/work/{work_id}/picture/{name}")
async def picture(work_id: str, name: str, _: None = Depends(_require_internal)) -> FileResponse:
    """A picture she made for a work."""
    folder = await asyncio.to_thread(_folder, work_id)
    safe = re.sub(r"[^\w-]+", "-", name)[:60]
    path = None if folder is None else folder / "icon.png" if safe == "icon" else folder / "pictures" / f"{safe}.png"
    if path is None or not await asyncio.to_thread(path.is_file):
        raise HTTPException(status_code=404, detail="no such picture")
    return FileResponse(path, media_type="image/png", headers={"Cache-Control": "no-cache"})


@router.post("/work/{work_id}/say")
async def say(work_id: str, said: Said, _: None = Depends(_require_internal)) -> JSONResponse:
    """Say something to a work."""
    from interface.routes.chat_carrying_out import said_to_work

    if get_workbench().get(work_id) is None and get_requests_of_her().get(work_id) is None:
        raise HTTPException(status_code=404, detail="no such work")
    return JSONResponse({"what": await said_to_work(work_id, said.words)})


@router.post("/work/{work_id}/{what}")
async def told(work_id: str, what: str, _: None = Depends(_require_internal)) -> JSONResponse:
    """Hold a work, let it go on, or stop it; or open what it made."""
    work = _the(work_id)
    if what == "hold":
        work.hold()
    elif what == "go_on":
        work.go_on()
    elif what == "stop":
        work.stop()
    elif what == "open":
        return JSONResponse({"opened": await asyncio.to_thread(_open, work_id)})
    else:
        raise HTTPException(status_code=404, detail="nothing a work can be told")
    return JSONResponse({"state": work.state})


@router.get("/workbench")
async def workbench_page() -> HTMLResponse:
    """The workbench as a page of its own."""
    page = Path(__file__).resolve().parents[1] / "static" / "workbench.html"
    return HTMLResponse(await asyncio.to_thread(page.read_text, "utf-8"))


def _the(work_id: str) -> Work:
    work = get_workbench().get(work_id)
    if work is None:
        raise HTTPException(status_code=404, detail="no such work")
    return work


def _open(work_id: str) -> str:
    """Open what a request made, as it was installed."""
    from core.governance_context import local_internal_governed_scope
    from core.runtime.subprocess_gateway import get_subprocess_gateway

    request = get_requests_of_her().get(work_id)
    made = str((request.made.get("app") or request.made.get("page") or "") if request is not None else "")
    if not made or not Path(made).exists():
        return ""
    with local_internal_governed_scope("work.open", domain="tool_execution"):
        get_subprocess_gateway().run(["open", made], timeout=30, source="work.open", accelerator_capability="none")
    return made
