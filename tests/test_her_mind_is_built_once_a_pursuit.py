"""Her mind is assembled once for a page pursuit, and she says where she is going at once.

LIVE 2026-10-02, a psych run: every page decision rebuilt her mind, each build
drew a fresh nonce for the blocks it fences as data, and the first of those sits
about 1,470 tokens into the prompt. Every decision diverged from the cached
prompt there and read the other 75-82% again, a minute and a half each on her
27B; and for the first of those minutes nothing at all was said, so a request
she had taken up read as one she had not.
"""
from __future__ import annotations

import asyncio
import inspect

import pytest

from core.skills import sovereign_browser
from core.skills.sovereign_browser import SovereignBrowserSkill
from core.skills.sovereign_browser_one_question import HER_MIND_THIS_PURSUIT
from core.skills.sovereign_browser_what_it_did import where_she_is_going

pytestmark = pytest.mark.unit


def _a_skill_that_counts_builds():
    skill = SovereignBrowserSkill.__new__(SovereignBrowserSkill)
    builds: list[int] = []

    async def build_now():
        builds.append(1)
        return f"her mind, build {len(builds)}"

    skill._her_mind_built_now = build_now
    return skill, builds


def test_inside_a_pursuit_her_mind_is_built_once():
    skill, builds = _a_skill_that_counts_builds()

    async def run():
        HER_MIND_THIS_PURSUIT.set({})
        first = await skill._assembled_mind()
        again = await skill._assembled_mind()
        return first, again

    first, again = asyncio.run(run())
    assert first == again == "her mind, build 1"
    assert len(builds) == 1


def test_outside_a_pursuit_every_call_builds_its_own():
    skill, builds = _a_skill_that_counts_builds()

    async def run():
        # Outside one, said here: an async test before this one on a shared
        # loop can leave a pursuit's holder in the context this run copies.
        from core.skills.sovereign_browser_one_question import HER_MIND_THIS_PURSUIT

        HER_MIND_THIS_PURSUIT.set(None)
        return await skill._assembled_mind(), await skill._assembled_mind()

    assert asyncio.run(run()) == ("her mind, build 1", "her mind, build 2")


def test_a_pursuit_starts_with_no_mind_kept_from_the_last_one():
    async def run():
        HER_MIND_THIS_PURSUIT.set({"mind": "an old one"})
        sovereign_browser._saying_it_moves_for({})
        return HER_MIND_THIS_PURSUIT.get()

    assert asyncio.run(run()) == {}


def test_she_says_where_she_is_going_before_her_first_look():
    said = where_she_is_going("https://www.example.test/tests/one", "Take the test on it ")
    assert said == "Opening example.test — take the test on it."
    body = inspect.getsource(SovereignBrowserSkill._handle_pursue)
    assert body.index("where_she_is_going(url, goal)") < body.index("for _round in range(")


def test_the_pursuit_keeps_the_mind_it_builds_first():
    """LIVE 2026-10-02 22:37: built before the holder was set, the first build was
    not kept, and the second call of the run read 77% of its prompt again."""
    body = inspect.getsource(SovereignBrowserSkill._handle_pursue)
    assert body.index("_saying_it_moves_for(action_context)") < body.index("await self._assembled_mind()")
