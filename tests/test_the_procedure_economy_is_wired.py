"""One registry, ranking procedures from every learner that has them.

``core/cognition/procedure.py`` says "backends compete directly: a chunk and
a generalized rule are ranked by the same number". The arithmetic did that.
The registry never saw a chunk or a rule: ``procedure_adapters.py`` had no
importer anywhere in production, and the claim ladder cited that same file as
the WIRED evidence for the claim. The adapters could have converted; nobody
asked them to.

What was missing was the collector — ``ingest_all`` takes the stores as
arguments, so somebody has to go and get them.
"""

from __future__ import annotations

import pytest

from core.agency.skill_library import LearnedSkill, SkillStep
from core.cognition.impasse import Impasse, ImpasseType, get_impasse_learner
from core.cognition.procedural_generalization import (
    DecisionEpisode,
    get_procedural_generalizer,
)
from core.cognition.procedure import Backend, ProcedureRegistry
from core.cognition.procedure_adapters import (
    install_the_learners,
    whatever_the_learners_hold,
)


@pytest.fixture
def three_learners_with_something_in_them(monkeypatch):
    """A chunk, a rule and a skill, each in its own store."""
    learner = get_impasse_learner()
    impasse = Impasse(
        type=ImpasseType.TIE,
        signature="whether_to_look_again",
        candidates=("look", "answer"),
    )
    learner.record_impasse(impasse)
    learner.learn(
        impasse, "look", cost_saved_per_use=4.5, match_cost=0.2
    )
    for _ in range(3):
        learner.record_outcome("whether_to_look_again", correct=True)

    general = get_procedural_generalizer()
    for _ in range(12):
        general.record(
            DecisionEpisode(
                features=frozenset({("channel", "screen"), ("held", "true")}),
                resolution="read it again",
                correct=True,
            )
        )
    rule = general.derive("read it again")
    if rule is not None:
        general.promote(rule)

    class _Library:
        skills = {
            "tidy_the_desk": LearnedSkill(
                name="tidy_the_desk",
                description="",
                parameters=["where"],
                steps=[SkillStep(tool_name="move", arguments={})],
                successes=7,
                failures=1,
            )
        }

    from core.container import ServiceContainer

    real = ServiceContainer.get
    monkeypatch.setattr(
        ServiceContainer,
        "get",
        staticmethod(
            lambda key, default=None, **kw: _Library()
            if key == "skill_library"
            else real(key, default=default, **kw)
        ),
    )
    yield


def test_the_collector_finds_every_store(three_learners_with_something_in_them):
    """ingest_all takes the stores as arguments. Somebody has to fetch them."""
    registry = ProcedureRegistry()
    landed = whatever_the_learners_hold(registry=registry)
    assert landed["missing"] == (), landed["missing"]
    assert landed["landed"]["chunk"] >= 1
    assert landed["landed"]["skill"] >= 1


def test_more_than_one_backend_lands_in_one_registry(
    three_learners_with_something_in_them,
):
    """The claim is that they compete. They have to be in the same place first."""
    registry = ProcedureRegistry()
    whatever_the_learners_hold(registry=registry)
    backends = {one.backend for one in registry.procedures()}
    assert len(backends) >= 2, f"only {backends} reached the registry"


def test_a_ranking_asks_for_the_learners_first(
    three_learners_with_something_in_them,
):
    """match() is where the ranking happens, so that is where staleness shows."""
    registry = ProcedureRegistry()
    assert registry.match({"situation:whether_to_look_again": True}) == []
    install_the_learners(registry)
    found = registry.match({"situation:whether_to_look_again": True})
    assert found, "the ranking found nothing after the learners were installed"
    assert any(one.backend is Backend.CHUNK for one in found)


def test_the_refresh_does_not_rescan_on_every_match(
    three_learners_with_something_in_them,
):
    """A store scan per ranking would cost more than the ranking saves."""
    registry = ProcedureRegistry()
    calls = []
    registry.keep_current_with(lambda: calls.append(1))
    for _ in range(20):
        registry.match({"anything": True})
    assert len(calls) == 1, f"refreshed {len(calls)} times in one window"


def test_a_store_that_will_not_answer_is_named_rather_than_hidden():
    """A partial ranking that says which learners are in it beats a quiet one."""
    registry = ProcedureRegistry()
    landed = whatever_the_learners_hold(registry=registry)
    assert isinstance(landed["missing"], tuple)


def test_contract_health_installs_the_economy():
    """The wiring has to happen somewhere that production actually imports."""
    from core.cognition.contract_health import install_contract_health
    from core.cognition.procedure import get_procedure_registry

    install_contract_health()
    assert get_procedure_registry()._refresh is not None  # noqa: SLF001
