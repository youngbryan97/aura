"""Build an app from a request: plan, compile, verify, write.

The language model contributes one thing — a plan, as typed data. Everything
after that is the runtime's own work: the plan is repaired where it is thin,
compiled to a single file, run against the Python model of the same
operations, and only then written to disk.

This replaces a path that asked a code model for a finished HTML document.
That path needed a 21.5GB model beside a 25.3GB resident cortex, so on this
host it could not run at all, and when it did run nothing checked the result.
A plan is a few hundred tokens and fits any lane.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.construction.app_compiler import compile_app
from core.construction.app_planner import PlannedApp, plan_from_json, plan_schema
from core.construction.app_verifier import verify_app
from core.runtime.errors import record_degradation

logger = logging.getLogger(__name__)

__all__ = ["BuiltApp", "build_app", "plan_request"]

#: A plan is small. This is a ceiling, not a target.
_PLAN_TOKENS = 900

#: Planning is one call. Two is a retry, not a conversation.
_PLAN_ATTEMPTS = 2


@dataclass(frozen=True, slots=True)
class BuiltApp:
    ok: bool
    request: str = ""
    path: str = ""
    title: str = ""
    checks: tuple[str, ...] = ()
    problems: tuple[str, ...] = ()
    repairs: tuple[str, ...] = ()
    verified_semantics: bool = False
    seconds: float = 0.0

    def summary(self) -> str:
        if not self.ok:
            return f"Could not build it: {'; '.join(self.problems[:3]) or 'no workable plan'}."
        checked = "; ".join(self.checks)
        return f"Built {self.title} at {self.path}. Checked: {checked}."

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "request": self.request,
            "path": self.path,
            "title": self.title,
            "checks": list(self.checks),
            "problems": list(self.problems),
            "repairs": list(self.repairs),
            "verified_semantics": self.verified_semantics,
            "seconds": round(self.seconds, 2),
        }


def _plan_request_text(request: str) -> str:
    """The planning call: a schema and the request, with nothing else in it."""
    schema = json.dumps(plan_schema(), indent=1)
    return (
        f"Request: {request.strip()}\n\n"
        f"Return one JSON object matching this schema and nothing else.\n{schema}\n"
    )


async def _ask_the_model(text: str, *, origin: str) -> str:
    from core.container import ServiceContainer

    gate = ServiceContainer.get("inference_gate", default=None)
    if gate is None or not hasattr(gate, "think"):
        return ""
    return str(
        await gate.think(
            text,
            system_prompt="Return JSON only. No prose, no code, no explanation.",
            max_tokens=_PLAN_TOKENS,
            temperature=0.1,
            origin=origin,
            serves_current_turn=True,
        )
        or ""
    )


async def plan_request(
    request: str, *, propose: Callable[[str], Awaitable[str]] | None = None
) -> PlannedApp | None:
    """A plan for this request, or None when nothing usable came back."""
    ask = propose or (lambda text: _ask_the_model(text, origin="build_app.plan"))
    text = _plan_request_text(request)
    for attempt in range(_PLAN_ATTEMPTS):
        try:
            raw = await ask(text)
        except (RuntimeError, OSError, ValueError, TypeError, AttributeError) as exc:
            logger.warning(
                "build: planning call failed (%s: %s).", type(exc).__name__, str(exc)[:160]
            )
            record_degradation(
                "build.plan",
                exc,
                severity="warning",
                action="the app plan could not be requested",
                enforce_failure_policy=False,
            )
            return None
        planned = plan_from_json(raw, request)
        if planned is not None and planned.spec.actions:
            logger.info(
                "build: plan accepted on attempt %d — %d field(s), %d action(s), %d repair(s).",
                attempt + 1,
                len(planned.spec.fields),
                len(planned.spec.actions),
                len(planned.repairs),
            )
            return planned
        logger.info("build: attempt %d returned no usable plan (%d chars).", attempt + 1, len(raw or ""))
    return None


def _slug(text: str) -> str:
    words = re.findall(r"[a-z0-9]+", str(text or "").lower())
    return "_".join(words[:5]) or "app"


async def build_app(
    request: str,
    *,
    out_dir: str | Path,
    propose: Callable[[str], Awaitable[str]] | None = None,
) -> BuiltApp:
    """Plan it, build it, check it, write it."""
    started = time.perf_counter()
    planned = await plan_request(request, propose=propose)
    if planned is None:
        return BuiltApp(
            ok=False,
            request=request,
            problems=("the planning step returned nothing that named any state or actions",),
            seconds=time.perf_counter() - started,
        )

    spec = planned.spec
    problems = spec.problems()
    if problems:
        # spec_from_plan repairs on the way in, so anything left is a defect
        # in this module rather than in the plan.
        return BuiltApp(
            ok=False,
            request=request,
            title=spec.title,
            problems=problems,
            repairs=planned.repairs,
            seconds=time.perf_counter() - started,
        )

    html = compile_app(spec)
    report = await asyncio.to_thread(verify_app, spec, html)
    if not report.ok:
        return BuiltApp(
            ok=False,
            request=request,
            title=spec.title,
            checks=report.checks,
            problems=report.problems,
            repairs=planned.repairs,
            verified_semantics=report.semantics_checked,
            seconds=time.perf_counter() - started,
        )

    target = Path(out_dir) / f"{_slug(spec.title)}.html"
    from core.runtime.file_write_gateway import get_file_write_gateway

    gateway = get_file_write_gateway()
    await gateway.ensure_directory_async(str(target.parent), source="build_app")
    await gateway.write_text_async(str(target), html, source="build_app")
    logger.info(
        "build: wrote %s (%d bytes) — %s", target, len(html), "; ".join(report.checks)
    )
    return BuiltApp(
        ok=True,
        request=request,
        path=str(target),
        title=spec.title,
        checks=report.checks,
        repairs=planned.repairs,
        verified_semantics=report.semantics_checked,
        seconds=time.perf_counter() - started,
    )
