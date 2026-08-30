"""The machinery for a goal she works out herself existed and could not reach her.

Her initiatives are endogenous at the trigger: boredom, curiosity, social
hunger, runtime load and a phi-derived autonomy scale decide whether one is
proposed, and firing one changes the affect that produced it. But what those
triggers instantiate are authored schemas — consolidate, review the graph, idle
attentively. Endogenous state choosing among written motives.

A second engine synthesises goals from recurring tensions instead. Production
code feeds it: every regretted autonomous action is recorded against it. And
nothing ever asked it what those observations came to. `synthesize()` and
`adopt_into_goal_engine()` had no live caller anywhere — the capability ran in
tests and could not reach her.

This pins the join: the progression from observation to adopted goal happens on
the same loop that runs everything else she does on her own.
"""

from __future__ import annotations

import pytest

from core.runtime.autonomy_conductor import AutonomyConductor


class Emergent:
    """An emergent-goal engine with a given set of ready candidates."""

    def __init__(self, candidates=(), adopts=()):
        self._candidates = list(candidates)
        self._adopts = list(adopts)
        self.synthesized = 0
        self.adopted_into = None

    def synthesize(self):
        self.synthesized += 1
        return list(self._candidates)

    async def adopt_into_goal_engine(self, goal_engine):
        self.adopted_into = goal_engine
        return [{"name": name} for name in self._adopts]


@pytest.fixture
def wired(monkeypatch):
    def install(emergent, goals=object()):
        import core.container as container

        def peek(name, default=None, **_kw):
            if name == "emergent_goal_engine":
                return emergent
            if name == "goal_engine":
                return goals
            return default

        monkeypatch.setattr(container.ServiceContainer, "get", staticmethod(peek), raising=False)
        return AutonomyConductor()

    return install


# ── the progression runs ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_the_tensions_she_recorded_are_asked_what_they_come_to(wired):
    emergent = Emergent(candidates=["a", "b"])
    done = await wired(emergent)._job_emergent_goal_adoption()
    assert emergent.synthesized == 1
    assert done["candidates"] == 2


@pytest.mark.asyncio
async def test_and_the_ready_ones_become_goals_she_is_pursuing(wired):
    emergent = Emergent(candidates=["a"], adopts=["stop losing the thread"])
    done = await wired(emergent)._job_emergent_goal_adoption()
    assert done["adopted"] == ["stop losing the thread"]
    assert emergent.adopted_into is not None


@pytest.mark.asyncio
async def test_nothing_ready_adopts_nothing(wired):
    done = await wired(Emergent(candidates=["a"]))._job_emergent_goal_adoption()
    assert done["adopted"] == []


# ── and it degrades honestly ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_without_the_engine_it_says_so(wired):
    conductor = wired(None)
    done = await conductor._job_emergent_goal_adoption()
    assert done["ran"] is False
    assert "emergent goal engine" in done["why"]


@pytest.mark.asyncio
async def test_without_somewhere_to_adopt_into_it_still_synthesises(wired):
    emergent = Emergent(candidates=["a"])
    done = await wired(emergent, goals=None)._job_emergent_goal_adoption()
    assert done["ran"] is True and done["candidates"] == 1
    assert "goal engine" in done["why"]


# ── on the loop that runs everything else she does on her own ────────────

def _default_jobs():
    conductor = AutonomyConductor()
    conductor.register_defaults()
    return dict(conductor.status().get("jobs") or {})


def test_it_is_a_job_the_conductor_actually_runs():
    assert "emergent_goal_adoption" in _default_jobs()


def test_it_runs_often_enough_to_act_in_the_session_it_happened_in():
    """A recurring tension needs long enough to recur and short enough to matter."""
    assert _default_jobs()["emergent_goal_adoption"]["interval_s"] <= 900.0
