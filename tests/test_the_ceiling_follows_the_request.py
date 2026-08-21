"""What a turn may do depends on what was asked for.

LIVE, 2026-08-20. "build me a small web app… tell me where you put it" ran
under the self-service ceiling, so the only capabilities offered were ones
that change nothing. The model reached for code_repl — the closest thing
available — and governance vetoed it: running arbitrary code needs
confirmation, correctly.

Meanwhile build_app, whose entire description is building a runnable
single-file web app, sits at read_write_artifacts and was never offered, on a
turn whose whole point was to produce a file.

The ceiling's own comment says it is the most a turn may do WITHOUT the person
having asked for that effect. Asking for a page to exist is asking for it.
"""

from __future__ import annotations

import pytest

from core.phases.response_contract import (
    _SELF_SERVICE_CEILING,
    requested_effect_ceiling,
)


@pytest.mark.parametrize(
    "request_text",
    [
        "build me a small web app: a single HTML page with a timer",
        "write me a python script that renames files by date",
        "create an html page with a countdown",
    ],
)
def test_asking_for_a_file_raises_the_ceiling(request_text: str) -> None:
    ceiling, scopes = requested_effect_ceiling(request_text)
    assert ceiling == "read_write_artifacts"
    assert "read_write_artifacts" in scopes


@pytest.mark.parametrize(
    "request_text",
    [
        "what is the temperature at that endpoint",
        "read /etc/hosts and tell me the first line",
        "how are you today?",
        "",
    ],
)
def test_everything_else_keeps_the_self_service_ceiling(request_text: str) -> None:
    ceiling, scopes = requested_effect_ceiling(request_text)
    assert ceiling == _SELF_SERVICE_CEILING
    assert "read_write_artifacts" not in scopes


def test_the_capability_built_for_this_is_offered() -> None:
    from core.intent.capability_selection import select_capabilities
    from core.skills.discovery import build_skill_catalog

    class _Meta:
        def __init__(self, declaration):
            self.description = declaration.description
            self.effect_scope = declaration.effect_scope
            self.enabled = True
            self.name = declaration.name
            self.module_path = declaration.module_path
            self.class_name = declaration.class_name
            self.skill_class = None
            self.instance = None

    skills = {d.name: _Meta(d) for d in build_skill_catalog().accepted}
    request = "build me a small web app: a single HTML page with a timer"
    ceiling, scopes = requested_effect_ceiling(request)
    offered = select_capabilities(request, skills, ceiling=ceiling, admissible_scopes=scopes)
    assert offered and offered[0] == "build_app"


def test_the_dispatch_authorises_what_selection_offered() -> None:
    """Offering a capability the dispatch then refuses is worse than not
    offering it: the turn spends itself reaching for something it was never
    allowed to use."""
    from pathlib import Path

    gate = Path("core/brain/inference_gate.py").read_text(encoding="utf-8")
    body = gate[gate.index("async def _tool_grounded_answer") :]
    body = body[: body.index("\n    async def ", 10)]
    assert "requested_effect_ceiling(text)" in body
    assert '"authorised_effect_scope": _ceiling' in body
    assert "_SELF_SERVICE_CEILING" not in body
