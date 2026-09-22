"""A campaign's conversation turn meets a person, as a desktop turn does.

On 22 September the named reading `unknown_person` held 0 for all 10,560
frames of a seed-7 campaign. It is one less her confidence about the partner,
read on a person's turn, and the subject driver never set a partner: its turns
came through the turn door without the step the desktop runtime's incoming path
ran beside it. So her model of the other person never learned anyone, the
interior appraisal of what they said never landed, and nothing keyed to a
person moved in any campaign. That step is in the door now, for both.
"""

from __future__ import annotations

import asyncio
import types
from pathlib import Path

import pytest

from core.kernel import turn_door


class _Other:
    def __init__(self) -> None:
        self.seen: list[tuple[str, str]] = []

    def observe_message(self, person, message, **_kwargs):
        self.seen.append((person, message))

    def cognitive_snapshot(self, person, at):
        return {"person": person, "turns": len(self.seen)}


class _Observer:
    def __init__(self) -> None:
        self.noted: list[str] = []
        self.registered: list[dict] = []

    def observe_agent(self, person, **_kwargs):
        self.noted.append(person)

    def register_interaction(self, person, social):
        self.registered.append(social)


@pytest.mark.unit
def test_the_door_learns_the_person_and_names_the_partner(monkeypatch: pytest.MonkeyPatch) -> None:
    other, observer = _Other(), _Observer()
    services = {"other_agent_model": other, "recursive_tom": observer}
    monkeypatch.setattr(
        "core.runtime.service_access.optional_service",
        lambda name, default=None: services.get(name, default),
    )
    state = types.SimpleNamespace(cognition=types.SimpleNamespace(current_partner=""))
    social = turn_door.observe_the_person(state, "someone", "hello there", observed_at=100.0)
    assert state.cognition.current_partner == "someone"
    assert other.seen == [("someone", "hello there")]
    assert observer.noted == ["someone"]
    assert social is not None and social["evidence_digest"] and social["at"] == 100.0
    assert observer.registered == [social]


@pytest.mark.slow
def test_a_campaign_turn_meets_the_person_and_learns_them(tmp_path: Path) -> None:
    from core.social.other_agent_model import get_other_agent_model
    from core.subject.clock import installed_clock
    from core.subject.driver import CONDITIONS, build_runtime, calibrate_clock, start_organism

    conversation = next(c for c in CONDITIONS if c.name == "conversation")

    async def run() -> tuple[list[int], dict[str, int], str]:
        runtime = build_runtime(tmp_path / "runtime", seed=7)
        await start_organism(runtime)
        await calibrate_clock(runtime, CONDITIONS, turns=1)
        observed = []
        for _ in range(4):
            await runtime.turn_once(conversation)
            record = get_other_agent_model()._models.get(runtime.PARTNER)
            observed.append(0 if record is None else int(record.observations))
        return observed, dict(runtime.failures), str(runtime.state.cognition.current_partner)

    try:
        observed, failures, partner = asyncio.run(run())
    finally:
        clock = installed_clock()
        if clock is not None:
            clock.uninstall()
    assert partner == "the_person"
    assert "turn_door.person" not in failures, failures
    # Every turn reaches her model of the person. How well she knows them then
    # rises with what their messages carry, which a neutral one does not.
    assert observed == sorted(observed) and observed[-1] > observed[0] > 0, observed
