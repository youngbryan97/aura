from core.brain.llm.context_assembler import ContextAssembler
from core.container import ServiceContainer
from core.runtime.errors import get_degradation_tracker
from core.runtime.principal_context import relational_principal_scope
from core.social.relational_memory import RelationalMemoryAuthority
from core.state.aura_state import AuraState


def test_short_self_inquiry_is_not_treated_as_casual():
    assert ContextAssembler._is_casual_interaction("Do you feel anything?") is False
    assert ContextAssembler._is_casual_interaction("Is Aura conscious?") is False


def test_short_greeting_stays_casual():
    assert ContextAssembler._is_casual_interaction("hey") is True


def test_rendering_a_prompt_does_not_move_attention():
    """A retry, a preview, a gate-side assembly against a payload copy, and a
    generation that died before its first token all called this. None of them
    is an accepted turn, and every one of them moved what Aura was attending
    to."""
    state = AuraState.default()
    state.cognition.attention_focus = "the thing she was already thinking about"

    ContextAssembler.build_messages(state, "Let's debug the retrieval pipeline.")

    assert state.cognition.attention_focus == "the thing she was already thinking about"


def test_the_serving_lane_can_still_record_attention():
    state = AuraState.default()
    state.cognition.attention_focus = None

    ContextAssembler.build_messages(
        state, "Let's debug the retrieval pipeline.", record_attention=True
    )

    assert state.cognition.attention_focus == "Let's debug the retrieval pipeline."


def test_only_the_serving_lane_opts_in():
    """One caller, named, rather than a default that every caller inherits."""
    import re as _re
    from pathlib import Path as _Path

    root = _Path(__file__).resolve().parents[1]
    opted_in = []
    for path in (root / "core").rglob("*.py"):
        text = path.read_text("utf-8", errors="ignore")
        if "record_attention=True" in text:
            opted_in.append(str(path.relative_to(root)))
    assert opted_in == ["core/brain/cognitive_engine.py"], opted_in
    assert not _re.search(
        r"record_attention\s*:\s*bool\s*=\s*True",
        (root / "core" / "brain" / "llm" / "context_assembler.py").read_text("utf-8"),
    ), "the default flipped; every renderer would mutate again"


def test_the_prompt_path_does_not_call_the_stakes_organ():
    """Its block is deliberately not injected. The call remained anyway, with
    the return discarded, so the organ ran on the foreground path for nothing."""
    from pathlib import Path as _Path

    source = (
        _Path(__file__).resolve().parents[1]
        / "core" / "brain" / "llm" / "context_assembler.py"
    ).read_text("utf-8")

    assert "stakes.get_context_block()" not in source
    assert "Existential stakes are deliberately absent" in source, (
        "keep the note explaining the absence, or the call comes back"
    )


class ToolAffordanceRecorder:
    def __init__(self):
        self.calls = 0
        self.kwargs = []

    def build_tool_affordance_block(self, **kwargs):
        self.calls += 1
        self.kwargs.append(kwargs)
        return "## LIVE TOOL OPTIONS\n- clock: Check time and date."


def test_build_system_prompt_uses_compact_turn_specific_tool_affordances(monkeypatch):
    state = AuraState.default()
    state.cognition.current_objective = "What time is it right now?"

    engine = ToolAffordanceRecorder()

    original_get = ServiceContainer.get

    def _get(name, default=None):
        if name == "capability_engine":
            return engine
        return original_get(name, default)

    monkeypatch.setattr(ServiceContainer, "get", staticmethod(_get))

    prompt = ContextAssembler.build_system_prompt(state)

    assert "## LIVE TOOL OPTIONS" in prompt
    assert "action affordances of your current body" in prompt
    assert "no magic phrase is required" in prompt
    assert "hypothetical, quotation, negation, memory" in prompt
    assert "If you need facts, USE web_search/search_web/free_search." not in prompt
    assert engine.calls == 1
    assert engine.kwargs[0]["objective"] == "What time is it right now?"


def test_context_assembler_methods_are_not_replaced_at_import_time():
    assert ContextAssembler._is_casual_interaction.__module__ == "core.brain.llm.context_assembler"
    assert ContextAssembler.build_system_prompt.__module__ == "core.brain.llm.context_assembler"
    assert ContextAssembler.build_messages.__module__ == "core.brain.llm.context_assembler"
    assert not getattr(ContextAssembler, "_patched_v1", False)


def test_context_assembler_uses_exact_active_social_agent_without_intimacy_claims():
    class ExactAgentEstimator:
        active_agent_id = "bryan"

        def __init__(self):
            self.requested_agents = []

        def context_injection(self, agent_id):
            self.requested_agents.append(agent_id)
            return f"SOCIAL_MARKER agent={agent_id} hypothesis_only=true"

    estimator = ExactAgentEstimator()
    ServiceContainer.clear()
    ServiceContainer.register_instance(
        "other_agent_model",
        estimator,
        required=False,
    )
    state = AuraState.default()
    state.cognition.current_origin = "user"
    state.cognition.current_objective = "Continue the architecture review."

    try:
        prompt = ContextAssembler.build_system_prompt(state)
    finally:
        ServiceContainer.clear()

    assert estimator.requested_agents == ["bryan"]
    assert "SOCIAL_MARKER agent=bryan hypothesis_only=true" in prompt
    lowered = prompt.lower()
    assert "## who i'm talking to" not in lowered
    assert "deep bond" not in lowered
    assert "be more personal" not in lowered
    assert "high rapport → lean in" not in lowered
    assert "relational register: intimate" not in lowered


def test_internal_cognition_does_not_inherit_the_last_active_social_agent():
    class AmbientEstimator:
        active_agent_id = "bryan"

        def __init__(self):
            self.requested_agents = []

        def context_injection(self, agent_id):
            self.requested_agents.append(agent_id)
            return f"AMBIENT_SOCIAL_MARKER agent={agent_id}"

    tracker = get_degradation_tracker()
    tracker.reset()
    estimator = AmbientEstimator()
    ServiceContainer.clear()
    ServiceContainer.register_instance("other_agent_model", estimator, required=False)
    state = AuraState.default()
    state.cognition.current_origin = "curriculum_loop"
    state.cognition.current_objective = "Evaluate an internal learning transition."

    try:
        prompt = ContextAssembler.build_system_prompt(state)
        scope_failures = tracker.recent(subsystem="context_assembler.relational_scope")
    finally:
        ServiceContainer.clear()
        tracker.reset()

    assert estimator.requested_agents == []
    assert "AMBIENT_SOCIAL_MARKER" not in prompt
    assert scope_failures == []
    assert state.response_modifiers["relational_scope_receipt"] == {
        "status": "unbound_internal",
        "principal_bound": False,
        "relational_memory_consulted": False,
        "ambient_agent_hint_consulted": False,
        "origin": "curriculum_loop",
    }


def test_context_assembler_excludes_unscoped_legacy_relationship_memory():
    class LegacySocialMemory:
        @staticmethod
        def get_social_context():
            return "PRIVATE_OTHER_USER_RELATIONSHIP"

    ServiceContainer.clear()
    ServiceContainer.register_instance(
        "social_memory",
        LegacySocialMemory(),
        required=False,
    )

    try:
        prompt = ContextAssembler.build_system_prompt(AuraState.default())
    finally:
        ServiceContainer.clear()

    assert "PRIVATE_OTHER_USER_RELATIONSHIP" not in prompt


def test_context_assembler_injects_only_consented_exact_agent_memory(tmp_path):
    authority = RelationalMemoryAuthority(
        tmp_path / "relational.json",
        encryption_key=b"k" * 32,
        legacy_paths=(),
        auto_provision_key=False,
    )
    authority.grant_consent(
        "bryan",
        kinds=["boundary"],
        operations=["persist", "recall", "prompt"],
        receipt_id="grant-1",
    )
    authority.record(
        "bryan",
        kind="boundary",
        content="Keep the project codename private.",
    )
    authority.grant_consent(
        "alice",
        kinds=["boundary"],
        operations=["persist", "recall", "prompt"],
        receipt_id="grant-2",
    )
    authority.record(
        "alice",
        kind="boundary",
        content="ALICE_PRIVATE_BOUNDARY",
    )

    estimator = type(
        "Estimator",
        (),
        {
            "active_agent_id": "bryan",
            "context_injection": lambda self, agent_id: f"agent={agent_id}",
        },
    )()
    ServiceContainer.clear()
    ServiceContainer.register_instance("other_agent_model", estimator, required=False)
    ServiceContainer.register_instance("relational_memory", authority, required=False)

    from core.runtime.principal_context import relational_principal_scope

    try:
        # No bound principal for this request. The estimator's active_agent_id
        # is process-global — whoever it last saw — and used to be the last
        # link in the fallback chain that keyed relational memory, so one
        # interlocutor's stored history could be assembled into another's
        # prompt. A hint is enough to model who she is talking to; it is not
        # enough to hand over what somebody told her.
        unbound = ContextAssembler.build_system_prompt(AuraState.default())
        assert "Keep the project codename private." not in unbound
        assert "ALICE_PRIVATE_BOUNDARY" not in unbound

        with relational_principal_scope("bryan"):
            bound = ContextAssembler.build_system_prompt(AuraState.default())
    finally:
        ServiceContainer.clear()

    assert "Keep the project codename private." in bound
    assert "ALICE_PRIVATE_BOUNDARY" not in bound


def test_request_scoped_principal_overrides_process_global_active_agent(tmp_path):
    authority = RelationalMemoryAuthority(
        tmp_path / "relational.json",
        encryption_key=b"k" * 32,
        legacy_paths=(),
        auto_provision_key=False,
    )
    for agent_id, content in (
        ("bryan", "BRYAN_PRIVATE_BOUNDARY"),
        ("alice", "ALICE_PRIVATE_BOUNDARY"),
    ):
        authority.grant_consent(
            agent_id,
            kinds=["boundary"],
            operations=["persist", "recall", "prompt"],
            receipt_id=f"grant-{agent_id}",
        )
        authority.record(agent_id, kind="boundary", content=content)

    class Estimator:
        active_agent_id = "bryan"

        def __init__(self):
            self.requested_agents = []

        def context_injection(self, agent_id):
            self.requested_agents.append(agent_id)
            return f"agent={agent_id}"

    estimator = Estimator()
    ServiceContainer.clear()
    ServiceContainer.register_instance("other_agent_model", estimator, required=False)
    ServiceContainer.register_instance("relational_memory", authority, required=False)

    try:
        with relational_principal_scope("alice"):
            prompt = ContextAssembler.build_system_prompt(AuraState.default())
    finally:
        ServiceContainer.clear()

    assert estimator.requested_agents == ["alice"]
    assert "agent=alice" in prompt
    assert "ALICE_PRIVATE_BOUNDARY" in prompt
    assert "BRYAN_PRIVATE_BOUNDARY" not in prompt


def test_deep_conversation_keeps_compact_continuity():
    state = AuraState.default()
    state.cognition.current_objective = "Continue the architecture review."
    state.cognition.rolling_summary = "We are preserving the canonical desktop path."
    state.cognition.modifiers["continuity_obligations"] = {
        "identity_mismatch": False,
        "current_objective": "Harden live context assembly",
        "active_commitments": ["keep the live path coherent"],
        "subject_thread": "desktop reliability",
    }
    state.cognition.working_memory = [
        {"role": "user" if index % 2 == 0 else "assistant", "content": f"turn {index}"}
        for index in range(32)
    ]

    prompt = ContextAssembler.build_system_prompt(state)

    assert "## CONTINUITY SUMMARY" in prompt
    assert "canonical desktop path" in prompt
    assert "## TEMPORAL OBLIGATIONS" in prompt
    assert "Harden live context assembly" in prompt


def test_context_assembler_excludes_proof_fixture_from_lived_continuity():
    state = AuraState.default()
    state.cognition.rolling_summary = (
        "Mode=reactive | Commitments=A long-running microservice periodically "
        "crashes with OSError: too many open files. A code review reveals a resource leak"
    )

    prompt = ContextAssembler.build_system_prompt(state)

    assert "long-running microservice" not in prompt
    assert "code review reveals" not in prompt
    assert "## CONTINUITY SUMMARY" not in prompt


def test_context_assembler_does_not_promote_evaluation_objective_to_attention():
    state = AuraState.default()
    state.cognition.attention_focus = "Bryan's current conversation"
    fixture = (
        "A long-running microservice periodically crashes with OSError; "
        "code review reveals a resource leak."
    )

    messages = ContextAssembler.build_messages(state, fixture, max_tokens=2048)

    assert state.cognition.attention_focus == "Bryan's current conversation"
    assert messages[-1]["content"] == fixture


def test_build_messages_preserves_current_input_under_tight_budget(monkeypatch):
    state = AuraState.default()
    state.response_modifiers["black_box_steering"] = True
    objective = "BEGIN-" + ("x" * 9000) + "-END"
    monkeypatch.setattr(
        ContextAssembler,
        "build_system_prompt",
        staticmethod(lambda _state, **_kw: "SYSTEM-HEAD\n" + ("s" * 20000) + "\nSYSTEM-TAIL"),
    )

    messages = ContextAssembler.build_messages(state, objective, max_tokens=2048)

    assert len(messages[0]["content"]) <= 8192
    assert messages[0]["content"].startswith("SYSTEM-HEAD")
    assert messages[0]["content"].endswith("SYSTEM-TAIL")
    assert messages[-1]["role"] == "user"
    assert messages[-1]["content"].startswith("BEGIN-")
    assert messages[-1]["content"].endswith("-END")
    assert sum(len(message["content"]) for message in messages) <= 8192


def test_build_messages_counts_dropped_history_without_negative_slice(monkeypatch):
    state = AuraState.default()
    state.cognition.working_memory = [
        {"role": "user" if index % 2 == 0 else "assistant", "content": "z" * 500}
        for index in range(12)
    ]
    monkeypatch.setattr(
        ContextAssembler,
        "build_system_prompt",
        staticmethod(lambda _state, **_kw: "s" * 6500),
    )

    messages = ContextAssembler.build_messages(state, "current", max_tokens=2048)

    notices = [
        message["content"]
        for message in messages
        if message["role"] == "system" and "older conversational messages were omitted" in message["content"]
    ]
    assert notices
    assert "10 older conversational messages" in notices[0]


def test_the_being_runtime_is_sampled_once_per_assembly(monkeypatch):
    """runtime.sample measures a system that keeps moving. It was taken twice
    inside one system message — once in build_system_prompt, once in
    build_messages — so one prompt could state two different valences and two
    different focal objects as Aura's state right now."""
    calls = []

    class FakeRenderer:
        @staticmethod
        def render_prompt_block(_now):
            return ""

    class FakeRuntime:
        renderer = FakeRenderer

        def sample(self, _state, objective=""):
            calls.append(objective)

            class Now:
                @staticmethod
                def to_report_packet():
                    return {
                        "attention": {"focal_object": "the pipeline"},
                        "affect": {
                            "valence": 0.1,
                            "arousal": 0.2,
                            "distress": 0.0,
                            "free_energy": 0.3,
                        },
                    }

                @staticmethod
                def compact_prompt_block():
                    return "## AURA NOW\n"

            return Now()

        @staticmethod
        def organismal_workspace_prompt_block(compact=False):
            return ""

    import core.being.runtime as being_runtime

    monkeypatch.setattr(being_runtime, "get_being_runtime", lambda: FakeRuntime())

    state = AuraState.default()
    ContextAssembler.build_messages(state, "debug the retrieval pipeline", max_tokens=4096)

    assert len(calls) == 1, f"sampled {len(calls)} times in one assembly"


def test_cutting_the_users_message_is_disclosed_in_the_same_turn():
    """A marker in the prompt tells the model the middle is gone. Nothing told
    her, so she answered a question she had only the ends of and said nothing
    about it. The reading has to exist before the failure block renders, or the
    one turn that needed to disclose the cut is the one turn that cannot."""
    from core.conversation.failure_context import FailureLedger, bind_failure_ledger

    state = AuraState.default()
    huge = "START-MARKER " + ("filler word " * 4000) + " END-MARKER"

    with bind_failure_ledger(FailureLedger()) as ledger:
        messages = ContextAssembler.build_messages(state, huge, max_tokens=2048)
        recorded = list(ledger.records)

    assert messages[-1]["content"] != huge, "the fixture did not exceed the budget"
    assert "START-MARKER" in messages[-1]["content"]
    assert "END-MARKER" in messages[-1]["content"]

    window = [f for f in recorded if f.capability == "context_window"]
    assert window, f"input was cut with no reading she can narrate: {recorded}"
    facts = window[0].as_facts()
    assert "characters" in facts
    assert window[0].still_possible, "a bounded failure with no remaining option"


def test_an_input_that_fits_records_nothing():
    """The disclosure must not fire on every ordinary turn."""
    from core.conversation.failure_context import FailureLedger, bind_failure_ledger

    state = AuraState.default()

    with bind_failure_ledger(FailureLedger()) as ledger:
        ContextAssembler.build_messages(state, "what time is it?", max_tokens=8192)
        recorded = list(ledger.records)

    assert not [f for f in recorded if f.capability == "context_window"]
