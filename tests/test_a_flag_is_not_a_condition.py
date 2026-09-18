"""The black-box exclusion condition, and the trust level that decides content.

Both were caller-supplied state. `black_box_steering` in response modifiers or
an environment variable turned on the causal-exclusion condition a whole
critique rests on, and nothing recorded whether the state text it excludes was
in fact excluded. `trust_level` in the same dictionary decided which telemetry,
identity, continuity and personhood blocks the prompt exposes.
"""
from __future__ import annotations

import ast
import time
from pathlib import Path

from core.brain.llm.context_assembler import (
    _BLACK_BOX_STATE_MARKERS,
    ContextAssembler,
)
from core.runtime.principal_context import relational_principal_scope
from core.security.trust_engine import TrustLevel
from core.state.aura_state import AuraState

ROOT = Path(__file__).resolve().parents[1]


def test_every_marker_is_a_string_this_module_actually_writes():
    """A first draft guessed "## SOMATIC" and "## PHENOMENAL". Neither is
    written anywhere, so both were checks that could never fail — which is the
    failure the receipt exists to prevent."""
    source = (ROOT / "core" / "brain" / "llm" / "context_assembler.py").read_text("utf-8")
    tree = ast.parse(source)

    emitted: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            emitted.add(node.value)
        elif isinstance(node, ast.JoinedStr):
            for part in node.values:
                if isinstance(part, ast.Constant) and isinstance(part.value, str):
                    emitted.add(part.value)

    for marker in _BLACK_BOX_STATE_MARKERS:
        assert any(
            marker in text and text not in _BLACK_BOX_STATE_MARKERS
            for text in emitted
        ), f"{marker!r} is never written; it is a check that cannot fail"


def test_a_receipt_says_the_condition_was_not_requested():
    state = AuraState.default()

    receipt = ContextAssembler.black_box_receipt(state, "## CURRENT VIBE\nanything")

    assert receipt["requested"] is False
    assert receipt["held"] is False
    assert receipt["source"] == "not_requested"


def test_a_clean_prompt_under_the_condition_holds():
    state = AuraState.default()
    state.response_modifiers["black_box_steering"] = True

    receipt = ContextAssembler.black_box_receipt(state, "identity lock only")

    assert receipt["requested"] is True
    assert receipt["held"] is True
    assert receipt["leaked_markers"] == []
    assert receipt["source"] == "response_modifier"


def test_a_leak_under_the_condition_is_named_not_assumed_away():
    """The flag said the condition held. The prompt says otherwise, and the
    prompt is the artifact the result rests on."""
    state = AuraState.default()
    state.response_modifiers["black_box_steering"] = True
    leaky = "identity\n## COGNITIVE TELEMETRY\n- Valence: +0.30\n"

    receipt = ContextAssembler.black_box_receipt(state, leaky)

    assert receipt["requested"] is True
    assert receipt["held"] is False
    assert receipt["leaked_markers"] == ["## COGNITIVE TELEMETRY"]


def test_the_receipt_binds_to_the_prompt_it_checked():
    state = AuraState.default()

    a = ContextAssembler.black_box_receipt(state, "one")
    b = ContextAssembler.black_box_receipt(state, "two")

    assert a["prompt_sha256"] != b["prompt_sha256"]


def _elevated_state(principal: str, *, age_s: float = 0.0) -> AuraState:
    state = AuraState.default()
    # The gate writes both of these into cognition.modifiers, which is where
    # build_system_prompt reads them.
    state.cognition.modifiers["trust_level"] = TrustLevel.SOVEREIGN
    state.cognition.modifiers["trust_level_binding"] = {
        "session_id": "s1",
        "origin": "gui",
        "recognized_at": time.time() - age_s,
        "level": "SOVEREIGN",
        "principal": principal,
        "principal_scope_bound": True,
    }
    return state


def _refusals_during(build) -> list[str]:
    """The refusal is the observable: it names why elevation was declined."""
    from core.runtime.errors import recent_degradations

    before = len(recent_degradations(limit=500, subsystem_prefixes=("context_assembler.trust",)))
    build()
    after = recent_degradations(limit=500, subsystem_prefixes=("context_assembler.trust",))
    return [str(r.get("message", "") or r.get("error", "")) for r in after[before:]]


def test_a_forged_binding_without_a_principal_scope_does_not_elevate():
    """State construction can write the modifier and a fresh timestamp. It
    cannot arrange to be running inside the right principal scope."""
    state = _elevated_state("bryan")

    refusals = _refusals_during(lambda: ContextAssembler.build_system_prompt(state))

    assert any("elevated trust level refused" in r for r in refusals), refusals


def test_a_binding_for_someone_else_does_not_elevate():
    state = _elevated_state("someone-else")

    def _build():
        with relational_principal_scope("bryan"):
            ContextAssembler.build_system_prompt(state)

    refusals = _refusals_during(_build)

    assert any("principal does not match" in r for r in refusals), refusals


def test_a_matching_principal_still_elevates():
    """The guard has to refuse forgeries without refusing the real case."""
    state = _elevated_state("bryan")

    def _build():
        with relational_principal_scope("bryan"):
            ContextAssembler.build_system_prompt(state)

    assert _refusals_during(_build) == []


def test_the_binding_carries_the_principal_the_gate_recognised():
    gate = (ROOT / "core" / "brain" / "inference_gate.py").read_text("utf-8")
    marker = gate.index('"trust_level_binding"')
    block = gate[marker : marker + 1400]

    assert '"principal": current_relational_principal()' in block
    assert '"principal_scope_bound": relational_principal_scope_is_bound()' in block


def test_a_process_global_agent_id_cannot_key_stored_memory():
    """other_agent_model.active_agent_id holds whoever the estimator last saw.
    It was the last link in the fallback chain that keyed relational memory, so
    one interlocutor's stored history could be assembled into another's
    prompt."""
    # Asked of whichever function decides the identity, not of the text
    # between two anchors. This read from the ``active_agent_id`` getattr
    # forward to the relational-memory block, and went red when the
    # method-size sweep moved the decision into ``_build_system_prompt_agent_id``
    # — which now sits BEFORE the getattr, so the window could not contain it.
    # The binding was untouched the whole time.
    from source_contract import function_containing, module_source

    from core.brain.llm import context_assembler

    source = module_source(context_assembler)
    _name, decides = function_containing(
        context_assembler, "agent_id = bound_agent"
    )

    # The hint is enough to model who she is talking to. It is not enough to
    # hand over what somebody else told her, so it may be read and reported
    # and never assigned.
    assert "hinted_agent" in decides
    assert "agent_id = hinted_agent" not in source
    assert "agent_id = bound_agent" in decides, (
        "the identity that keys relational memory is not the bound one"
    )
    # And the process-global is still only ever a hint.
    assert 'getattr(estimator, "active_agent_id"' in source
    assert "agent_id = estimator" not in source


def test_withholding_relational_memory_is_recorded():
    source = (ROOT / "core" / "brain" / "llm" / "context_assembler.py").read_text("utf-8")

    assert '"context_assembler.relational_scope"' in source
    assert "relational memory withheld" in source
