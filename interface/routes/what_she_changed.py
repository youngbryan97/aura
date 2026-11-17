"""What she changed in her own source, day by day, for a person to read (Master Todo F16, part 5).

Mounted on the inner-state router (interface/routes/inner_state.py), so it is served as /api/self/changes; the page
that reads it is /static/what_she_changed.html.
"""
from __future__ import annotations

import asyncio

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from core.runtime.errors import record_degradation

router = APIRouter()


@router.get("/self/changes")
async def her_changes() -> JSONResponse:
    """Her changes grouped by day (core/agency/what_she_changed.py), read off her event loop."""
    from core.agency.what_she_changed import changes_by_day

    from core.agency.gaps_she_names import kept_gaps

    try:
        days = await asyncio.to_thread(changes_by_day)
        gaps = list(reversed(await asyncio.to_thread(kept_gaps)))
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        record_degradation("what_she_changed", exc, severity="info", action="the page shows no changes this time")
        return JSONResponse({"days": [], "gaps": [], "error": type(exc).__name__}, status_code=503)
    return JSONResponse({"days": days, "gaps": gaps})
