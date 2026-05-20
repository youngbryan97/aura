import asyncio
from types import SimpleNamespace

from core.phases.social_context_phase import SocialContextPhase
from core.runtime.errors import get_degradation_tracker
from core.service_names import ServiceNames
from core.state.aura_state import AuraState


class SocialContainer:
    def __init__(self, *, ava=None, theory_of_mind=None, fail_theory=False):
        self.ava = ava
        self.theory_of_mind = theory_of_mind
        self.fail_theory = fail_theory

    def get(self, name, default=None):
        if name == ServiceNames.AVA:
            return self.ava if self.ava is not None else default
        if name == "theory_of_mind":
            if self.fail_theory:
                raise RuntimeError("theory service unavailable")
            return self.theory_of_mind if self.theory_of_mind is not None else default
        return default


class AvaAnalysisUnavailable:
    def __init__(self):
        self.analysis_calls = 0
        self.context_calls = 0

    async def analyze_message(self, _message):
        self.analysis_calls += 1
        raise RuntimeError("analysis unavailable")

    def get_context_injection(self):
        self.context_calls += 1
        return "steady, warm, concise social context"


class AvaContextUnavailable:
    def __init__(self):
        self.context_calls = 0

    def analyze_message(self, _message):
        return None

    def get_context_injection(self):
        self.context_calls += 1
        raise RuntimeError("context unavailable")


def _user_state() -> AuraState:
    state = AuraState()
    state.cognition.current_origin = "user"
    return state


def test_social_context_keeps_cues_when_ava_analysis_fails():
    async def scenario():
        tracker = get_degradation_tracker()
        tracker.reset()
        ava = AvaAnalysisUnavailable()
        phase = SocialContextPhase(SocialContainer(ava=ava))
        state = _user_state()
        state.cognition.modifiers = None

        result = await phase.execute(
            state,
            objective="I need a careful release plan that keeps the team calm.",
        )

        assert result is state
        assert ava.analysis_calls == 1
        assert ava.context_calls == 1
        assert state.cognition.modifiers["interaction_style"] == "balanced_flow"
        assert state.cognition.modifiers["social_context"] == "steady, warm, concise social context"
        assert "careful" in state.cognition.modifiers["lexical_mirror"]
        assert any(
            "Ava message analysis failed" in record.action
            for record in tracker.recent(subsystem="social_context_phase")
        )
        tracker.reset()

    asyncio.run(scenario())


def test_social_context_keeps_local_cues_when_context_injection_fails():
    async def scenario():
        tracker = get_degradation_tracker()
        tracker.reset()
        ava = AvaContextUnavailable()
        phase = SocialContextPhase(SocialContainer(ava=ava))
        state = _user_state()

        await phase.execute(state, objective="ok")

        assert ava.context_calls == 1
        assert state.cognition.modifiers["interaction_style"] == "proactive_engagement"
        assert state.cognition.modifiers["desired_brevity"] == "extreme"
        assert "social_context" not in state.cognition.modifiers
        assert any(
            "context injection failed" in record.action
            for record in tracker.recent(subsystem="social_context_phase")
        )
        tracker.reset()

    asyncio.run(scenario())


def test_social_context_keeps_cues_when_theory_register_fails():
    async def scenario():
        tracker = get_degradation_tracker()
        tracker.reset()
        ava = SimpleNamespace(get_context_injection=lambda: "available context")
        phase = SocialContextPhase(SocialContainer(ava=ava, fail_theory=True))
        state = _user_state()

        await phase.execute(state, objective="Please keep this careful and grounded.")

        assert state.cognition.modifiers["interaction_style"] == "balanced_flow"
        assert state.cognition.modifiers["social_context"] == "available context"
        assert "relational_register" not in state.cognition.modifiers
        assert any(
            "without theory-of-mind rapport register" in record.action
            for record in tracker.recent(subsystem="social_context_phase")
        )
        tracker.reset()

    asyncio.run(scenario())


def test_social_context_clamps_rapport_and_sets_register():
    async def scenario():
        user_model = SimpleNamespace(rapport=9.5)
        theory = SimpleNamespace(known_selves={"bryan": user_model})
        phase = SocialContextPhase(SocialContainer(theory_of_mind=theory))
        state = _user_state()

        await phase.execute(
            state,
            objective="Let's make this release feel polished and emotionally precise.",
        )

        assert state.cognition.modifiers["rapport_level"] == 1.0
        assert state.cognition.modifiers["relational_register"] == "intimate"
        assert state.cognition.modifiers["lexical_mirror"][:3] == [
            "let's",
            "release",
            "polished",
        ]

    asyncio.run(scenario())
