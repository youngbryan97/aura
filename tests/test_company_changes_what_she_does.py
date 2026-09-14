"""The witness stance reaches the three places it has to change what she does.

A reading nothing consumes is a mechanism that cannot fire. The conversation
phase writes the stance to cognition, routing and the reply both ask
`is_witnessing` of it, and the affect phase keeps matched distress while
company is being kept instead of releasing it back toward neutral.
"""

from __future__ import annotations

from types import SimpleNamespace

from core.expression.register import read
from core.phases.affect_update import AffectUpdatePhase
from core.phases.conversational_dynamics_phase import ConversationalDynamicsPhase
from core.social.witness import is_witnessing
from core.state.aura_state import AuraState
from core.subject.clamp import CLAMPED_FIELDS
from core.subject.state import _SCHEMAS

TESTIMONY = (
    "I was born by the river in a little tent. It's been too hard living but "
    "I'm afraid to die. There's been times that I thought I couldn't last for "
    "long, but now I think I'm able to carry on."
)
REQUEST = "Can you help me figure out why my build keeps failing? What should I check first?"


def _read_into(state: AuraState, text: str, valence: float) -> None:
    ConversationalDynamicsPhase._read_register(state, text)
    state.response_modifiers["user_sentiment"] = {"valence": valence}
    ConversationalDynamicsPhase._read_witness(state)


def test_the_conversation_phase_writes_the_stance_where_routing_and_the_reply_look() -> None:
    state = AuraState.default()
    _read_into(state, TESTIMONY, -0.5)
    assert state.cognition.witness["witnessing"] is True
    assert state.cognition.witness["company"] == 0.5
    assert is_witnessing(state.cognition)


def test_a_request_leaves_her_helping() -> None:
    state = AuraState.default()
    _read_into(state, REQUEST, -0.5)
    assert state.cognition.witness["witnessing"] is False
    assert not is_witnessing(state.cognition)


def test_a_message_too_short_to_read_does_not_inherit_the_last_stance() -> None:
    state = AuraState.default()
    _read_into(state, TESTIMONY, -0.5)
    assert is_witnessing(state.cognition)
    ConversationalDynamicsPhase._read_register(state, "")
    ConversationalDynamicsPhase._read_witness(state)
    assert not is_witnessing(state.cognition)
    assert "asks_to_be_witnessed" not in state.response_modifiers


def test_no_stance_is_not_witnessing() -> None:
    assert not is_witnessing(AuraState.default().cognition)
    assert not is_witnessing(SimpleNamespace(witness="yes"))
    assert not is_witnessing(SimpleNamespace())


def _low_state() -> tuple[AffectUpdatePhase, AuraState]:
    phase = AffectUpdatePhase(SimpleNamespace(organs={}))
    state = AuraState.default()
    state.cognition.current_origin = "user"
    state.cognition.conversation_energy = 0.7
    state.affect.valence = -1.0
    state.affect.emotions["fear"] = 1.0
    state.affect.emotions["sadness"] = 0.8
    phase._ensure_affect_schema(state.affect)
    return phase, state


def test_distress_is_kept_while_she_keeps_somebody_company() -> None:
    phase, state = _low_state()
    state.cognition.witness = {"witnessing": True, "company": 0.6}
    phase._regulate_stale_negative_affect(state.affect, state, TESTIMONY, [])
    assert state.affect.emotions["fear"] == 1.0
    assert state.affect.emotions["sadness"] == 0.8
    assert "distress_kept_for_company" in state.affect.markers
    assert "stale_negative_affect_regulated" not in state.affect.markers


def test_without_company_the_same_distress_is_released() -> None:
    phase, state = _low_state()
    phase._regulate_stale_negative_affect(state.affect, state, TESTIMONY, [])
    assert state.affect.emotions["fear"] < 1.0
    assert "stale_negative_affect_regulated" in state.affect.markers


def test_the_stance_is_a_deliberation_column_and_the_clamp_holds_it() -> None:
    sources = set(_SCHEMAS["D"].sources)
    assert "cognition.witness.witnessing" in sources
    assert "cognition.witness.company" in sources
    assert "cognition.witness" in CLAMPED_FIELDS["D"]


def test_the_partner_register_columns_are_held_as_well() -> None:
    assert "world.partner_register" in CLAMPED_FIELDS["W"]
    assert _SCHEMAS["W"].sources.count("world.partner_register.asking") == 1


def test_register_reading_still_says_testimony_for_this_text() -> None:
    """The fixture above only means something if the register reads it this way."""
    assert read(TESTIMONY).asks_to_be_witnessed()
