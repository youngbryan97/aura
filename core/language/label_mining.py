"""Labels the runtime already produced, read back as training examples.

Nothing here is annotation. Every pair comes from something that actually
happened: a request arrived, a capability ran, and the intention log recorded
both. That record is the ground truth for the decisions the routing layer
makes by word list — which capability a turn needs, and whether it needs the
screen at all.

Two rules keep the mining honest:

* **Never learn from the answer.** An intention whose text names the tool it
  used ("Use tool 'web_search'") teaches nothing except that the name appears
  twice. Those are dropped.
* **Distinct requests, not rows.** The same message repeated four times is one
  example. Counting rows would report a dataset four times the size of the
  one that exists.
"""

from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from core.runtime.errors import record_degradation

__all__ = [
    "ACTUATION_TOOLS",
    "LabelledRequest",
    "mine_desktop_actuation_labels",
    "mine_request_tool_pairs",
]

_RECOVERABLE = (OSError, sqlite3.Error, TypeError, ValueError, json.JSONDecodeError)

#: Short enough to be a fragment rather than a request.
_MIN_REQUEST_CHARS = 20

#: Drives that carry a person's own words. The autonomous loops carry their
#: own templated objectives — "Autonomous self-development scan" appears 533
#: times — and learning routing from those would learn the background loop's
#: vocabulary rather than what a person asking for something looks like.
_PERSON_DRIVES = ("user", "desktop_ui", "capability_engine")

#: Capabilities that drive the screen. Everything else answers, reads, fetches
#: or builds, and the distinction is the one that misrouted a request to build
#: a web app into the screen-automation lane.
ACTUATION_TOOLS = frozenset(
    {"computer_use", "desktop_task", "os_automation", "sovereign_browser", "screen_capture"}
)


@dataclass(frozen=True, slots=True)
class LabelledRequest:
    """One request, and the capability that actually ran for it."""

    request: str
    tool: str


def _database() -> Path | None:
    try:
        from core.config import config

        path = Path(config.paths.data_dir) / "memory" / "intention_loop.db"
    except _RECOVERABLE:
        return None
    return path if path.is_file() else None


def _first_successful_tool(actions_json: object) -> str:
    """The capability that ran AND worked, or "".

    A receipt records what happened, not what should have. "build me a small
    web app" is in this log against desktop_task, because that is where it was
    misrouted — os_automation refused it and completed 0 of 1 steps. Learning
    from every receipt would teach the routing mistake as the routing rule.

    Success is the difference between a record of an event and a record of an
    event that answered the request.
    """
    try:
        actions = json.loads(str(actions_json or "[]"))
    except (json.JSONDecodeError, TypeError, ValueError):
        return ""
    for action in actions:
        if not isinstance(action, dict):
            continue
        if not bool(action.get("success", False)):
            continue
        name = str(action.get("tool_name") or "").strip()
        if name:
            return name
    return ""


def mine_request_tool_pairs(limit: int = 4000) -> list[LabelledRequest]:
    """Distinct requests a person made, with the capability that ran."""
    path = _database()
    if path is None:
        return []
    try:
        connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=2.0)
    except _RECOVERABLE as exc:
        record_degradation(
            "language.label_mining",
            exc,
            severity="debug",
            action="mined no labels this run",
            enforce_failure_policy=False,
        )
        return []
    try:
        placeholders = ",".join("?" for _ in _PERSON_DRIVES)
        rows = connection.execute(
            "SELECT intention, actions_json FROM intentions "
            f"WHERE completed_at > 0 AND status = 'completed' AND drive IN ({placeholders}) "
            "ORDER BY completed_at DESC LIMIT ?",
            (*_PERSON_DRIVES, max(1, int(limit))),
        ).fetchall()
    except _RECOVERABLE:
        return []
    finally:
        connection.close()

    seen: dict[str, str] = {}
    for intention, actions_json in rows:
        request = " ".join(str(intention or "").split())
        if len(request) < _MIN_REQUEST_CHARS or request in seen:
            continue
        tool = _first_successful_tool(actions_json)
        if not tool:
            continue
        # Never learn from the answer.
        if request.lower().startswith("use tool") or re.search(
            rf"\b{re.escape(tool)}\b", request, re.IGNORECASE
        ):
            continue
        seen[request] = tool
    return [LabelledRequest(request=request, tool=tool) for request, tool in seen.items()]


def mine_desktop_actuation_labels(
    pairs: Iterable[LabelledRequest] | None = None,
) -> tuple[list[str], list[str]]:
    """Requests that needed the screen, and requests that did not.

    The decision `looks_like_desktop_objective` makes with seventeen patterns,
    read instead off what actually ran.
    """
    rows = list(pairs) if pairs is not None else mine_request_tool_pairs()
    positives = [row.request for row in rows if row.tool in ACTUATION_TOOLS]
    negatives = [row.request for row in rows if row.tool not in ACTUATION_TOOLS]
    return positives, negatives
