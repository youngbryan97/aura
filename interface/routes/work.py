"""Her work, seen as it goes, and a word in while it does: the workbench's routes.

What is shown is what the work keeps (core/understanding/workbench.py): steps, what she says of them, code as she writes
it, what trying it showed, and what she waits on. What a person says to a work that is going reaches it at once; said to
one that has ended, it takes the request up again with those words.
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from core.understanding.requests_of_her import get_requests_of_her
from core.understanding.workbench import Work, get_workbench
from interface.auth import _require_internal

logger = logging.getLogger("Aura.Server.Work")
router = APIRouter(tags=["work"])

__all__ = ["router"]

#: How many works are listed.
LISTED = 20


class Said(BaseModel):
    words: str = Field(min_length=1, max_length=8000)


def _listed(work: Work) -> dict[str, Any]:
    seen = work.seen(after=10**9)
    seen.pop("happened")
    seen["now"] = {k: v for k, v in seen["now"].items() if k != "writing"}
    return seen


@router.get("/work")
async def works(_: None = Depends(_require_internal)) -> JSONResponse:
    """The works, the newest first."""
    return JSONResponse({"works": [_listed(work) for work in reversed(get_workbench().all()[-LISTED:])]})


@router.get("/work/{work_id}")
async def work_seen(work_id: str, after: int = 0, _: None = Depends(_require_internal)) -> JSONResponse:
    """One work, and what has happened in it after the last that was seen."""
    return JSONResponse(_the(work_id).seen(after))


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
