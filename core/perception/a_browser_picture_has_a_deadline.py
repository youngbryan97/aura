"""A stalled browser cannot own the next observation indefinitely."""
from __future__ import annotations

import asyncio
import math
from collections.abc import Awaitable

from core.verify import invariant

PICTURE_REQUEST_S = 1.0


async def picture_request[Result](request: Awaitable[Result]) -> Result:
    """Cancel an unanswered browser request; callers retain their existing unavailable result."""
    return await asyncio.wait_for(request, timeout=PICTURE_REQUEST_S)


@invariant('perception.browser_picture_requests_have_a_finite_budget', scope='perception',
           owner='core/perception/a_browser_picture_has_a_deadline.py', observational=False)
def _picture_request_budget_invariant() -> tuple:
    assert math.isfinite(PICTURE_REQUEST_S) and 0 < PICTURE_REQUEST_S <= 5.
    return ()
